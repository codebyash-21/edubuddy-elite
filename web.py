"""
EduBuddy web server — FastAPI + ui.html.

Endpoints
    GET  /              the UI
    GET  /health        which components have finished warming up
    POST /transcribe    audio blob      -> {text, lang}
    POST /chat          {message, lang} -> {reply, citations, lang}
    POST /speak         {text, lang}    -> audio/wav
    POST /clear         reset conversation history
    GET  /books         indexed textbooks

Models load in background threads so the page appears immediately; the UI
polls /health and enables the mic once everything is ready.

Run:  python web.py     then open http://127.0.0.1:7860
"""
from __future__ import annotations

import tempfile
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse, Response

import config
import llm
import rag
import stt
import tts

BASE = Path(__file__).resolve().parent
STATE = {
    "stt": False,
    "llm": False,
    "tts": False,
    "rag": False,
    "books_indexed": False,
    "error": None,
}


# ------------------------------------------------------------------ warm-up
#
# Importing `transformers` from two threads at once can hand the second thread a
# partially initialised module, which fails as:
#     cannot import name 'is_torch_npu_available' from 'transformers'
# Both the TTS and RAG warm-ups pull it in, so one thread performs the import
# and the others wait for it. Model *loading* still happens in parallel.
_imports_ready = threading.Event()


def _preload_imports():
    try:
        import transformers          # noqa: F401
        import sentence_transformers  # noqa: F401
    except Exception as exc:
        STATE["error"] = f"imports: {exc}"
        print(f"[warm] preload FAILED: {exc}")
    finally:
        _imports_ready.set()


def _warm(name, fn):
    def runner():
        try:
            fn()
            STATE[name] = True
            print(f"[warm] {name} ready")
        except Exception as exc:
            import traceback
            STATE["error"] = f"{name}: {exc}"
            print(f"[warm] {name} FAILED: {exc}")
            traceback.print_exc()
    t = threading.Thread(target=runner, daemon=True, name=f"warm-{name}")
    t.start()
    return t


def _warm_tts():
    _imports_ready.wait(timeout=300)
    for lang in config.LANGUAGES:
        tts.load(lang)


def _warm_books():
    _imports_ready.wait(timeout=300)
    rag.library.load_from_disk()
    rag.load_embedder()
    STATE["rag"] = True
    rag.library.index_books_folder()
    STATE["books_indexed"] = True


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("=" * 52)
    print(" EduBuddy — loading models in the background")
    print(f" UI: http://{config.HOST}:{config.PORT}")
    print("=" * 52)
    threading.Thread(target=_preload_imports, daemon=True, name="preload").start()
    _warm("stt", stt.load)
    _warm("llm", llm.load)
    _warm("tts", _warm_tts)
    _warm("rag", _warm_books)
    yield


app = FastAPI(title="EduBuddy", lifespan=lifespan)


# ------------------------------------------------------------------ routes
@app.get("/")
def index():
    return FileResponse(BASE / "ui.html")


@app.get("/health")
def health():
    ready = STATE["stt"] and STATE["llm"] and STATE["tts"]
    return {
        **STATE,
        "ready": ready,
        "books": rag.library.summary() if STATE["rag"] else [],
    }


@app.get("/books")
def books():
    return {"books": rag.library.summary()}


@app.get("/library")
def library_list(lang: str = "all", book: str = "all"):
    """Curated question and answer pairs. No model, no retrieval — a file read."""
    try:
        import library
        all_qs = library.entries()
        return {
            "questions": library.entries(lang, book),
            "languages": sorted({q["lang"] for q in all_qs if q.get("lang")}),
            "books": sorted({q["book"] for q in all_qs if q.get("book")}),
            "total": len(all_qs),
        }
    except Exception as exc:
        print(f"[library] unavailable: {exc}")
        return {"questions": [], "languages": [], "books": [], "total": 0}


@app.post("/library/check")
async def library_check(payload: dict):
    """Mark a student's answer against one entry's keywords.

    Deterministic and instant: no model is consulted, so a wrong verdict can
    always be explained by pointing at which keywords were and were not found.
    """
    try:
        import library
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    q = library.by_id((payload.get("id") or "").strip())
    if not q:
        return JSONResponse({"error": "unknown question"}, status_code=404)

    result = library.mark(payload.get("answer") or "", q)
    print(f"[study] {q['id']} · {result['ratio']} · "
          f"{'correct' if result['correct'] else 'wrong'}")
    return result


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), lang: str = Form("auto")):
    if not STATE["stt"]:
        return JSONResponse({"error": "Speech recogniser is still loading."},
                            status_code=503)

    raw = await audio.read()
    if len(raw) < 2000:
        return {"text": "", "lang": "en"}

    suffix = Path(audio.filename or "clip.webm").suffix or ".webm"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(raw)
        tmp.close()
        forced = None if lang in ("auto", "", None) else config.normalise_lang(lang)
        text, detected = stt.transcribe_file(tmp.name, language=forced)
        return {"text": text, "lang": detected}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except Exception:
            pass


# --------------------------------------------------------------- retrieval
# Retrieval finishes in well under a second; generation takes tens of seconds on
# CPU. The UI therefore asks for the citations first and shows them while the
# model is still writing, which turns a blank wait into visible progress.
#
# The result is cached so /chat reuses what /retrieve just computed instead of
# searching the same question twice. Keyed on the exact question text, with a
# short TTL — this is a within-one-request aid, not a store.
_RETRIEVAL_CACHE: dict[str, tuple[float, str, list]] = {}
_RETRIEVAL_TTL = 120.0
_RETRIEVAL_MAX = 32


def _pack_hit(message: str, lang=None):
    """Instant answer from the question pack, or None.

    Returns the textbook passage verbatim — nothing is generated here. Kept in a
    helper so /retrieve and /chat cannot disagree about whether a question is a
    pack hit, which would show the student one source and cite another.
    """
    if not getattr(config, "QA_ENABLED", True):
        return None
    try:
        import qa_match
        return qa_match.match(message, lang=lang)
    except Exception as exc:
        print(f"[qa] match failed, falling through: {exc}")
        return None


def _retrieve(message: str):
    """Return (context, citations) for a question, reusing a recent result."""
    now = time.time()
    hit = _RETRIEVAL_CACHE.get(message)
    if hit and now - hit[0] < _RETRIEVAL_TTL:
        return hit[1], hit[2]

    context, citations = ("", [])
    if STATE["rag"]:
        try:
            context, citations = rag.library.context_for(message)
        except Exception as exc:
            print(f"[retrieve] failed: {exc}")

    if len(_RETRIEVAL_CACHE) >= _RETRIEVAL_MAX:
        _RETRIEVAL_CACHE.clear()
    _RETRIEVAL_CACHE[message] = (now, context, citations)
    return context, citations


@app.post("/retrieve")
async def retrieve(payload: dict):
    """Search the textbooks without loading or running the language model."""
    message = (payload.get("message") or "").strip()
    if not message:
        return {"citations": [], "ready": bool(STATE["rag"])}
    if not STATE["rag"]:
        return {"citations": [], "ready": False}

    t0 = time.time()
    hit = _pack_hit(message, payload.get("lang"))
    if hit:
        return {"citations": [hit["citation"]], "ready": True, "pack": True,
                "ms": int((time.time() - t0) * 1000)}

    _, citations = _retrieve(message)
    ms = int((time.time() - t0) * 1000)
    print(f"[retrieve] {ms} ms · {len(citations)} citations")
    return {"citations": citations, "ready": True, "pack": False, "ms": ms}


@app.post("/chat")
async def chat(payload: dict):
    if not STATE["llm"]:
        return JSONResponse({"error": "Tutor model is still loading."},
                            status_code=503)

    message = (payload.get("message") or "").strip()
    lang = config.normalise_lang(payload.get("lang") or "en")
    if not message:
        return {"reply": config.NO_SPEECH_MESSAGE[lang], "citations": [], "lang": lang}

    # Question pack first: a hit returns the textbook passage word for word in a
    # few milliseconds and the model is never loaded. `source` tells the UI to
    # present it as a quotation rather than as EduBuddy's own sentence — passing
    # a prewritten passage off as a live answer would be the dishonest version
    # of this optimisation.
    hit = _pack_hit(message, lang)
    if hit:
        print(f"[chat] {lang} · pack hit {hit['score']} · {hit['citation']}")
        return {"reply": hit["answer"], "citations": [hit["citation"]],
                "lang": lang, "grounded": True, "source": "textbook",
                "matched_question": hit["question"], "score": hit["score"]}

    context, citations = _retrieve(message)

    t0 = time.time()
    reply = llm.tutor.ask(message, lang=lang, context=context)

    # The model reports that the retrieved excerpts do not answer the question.
    # Drop the citations: showing page numbers next to an ungrounded answer is
    # worse than showing none, because it makes a guess look sourced.
    grounded = True
    if config.NOT_IN_BOOK_MARKER in reply:
        reply = reply.replace(config.NOT_IN_BOOK_MARKER, "").strip()
        note = config.NOT_IN_BOOK_MESSAGE[lang]
        reply = f"{note} {reply}".strip() if reply else note
        citations = []
        grounded = False

    print(f"[chat] {lang} · {time.time() - t0:.1f}s · "
          f"{len(citations)} citations · {'grounded' if grounded else 'NOT in book'}")

    return {"reply": reply, "citations": citations, "lang": lang,
            "grounded": grounded, "source": "model"}


def _script_of(text: str) -> str:
    """Which of en/hi/ta this text is written in, by counting codepoints."""
    dev = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    tam = sum(1 for c in text if "஀" <= c <= "௿")
    if tam > dev and tam > 0:
        return "ta"
    if dev > 0:
        return "hi"
    return "en"


@app.post("/speak")
async def speak(payload: dict):
    text = (payload.get("text") or "").strip()
    lang = config.normalise_lang(payload.get("lang") or "en")
    if not text:
        return Response(status_code=204)

    # Trust the text over the label. MMS-TTS normalises against a fixed
    # per-language vocabulary and silently drops everything outside it, so the
    # English voice given Tamil text produces near-silence or noise — which is
    # exactly what happened when the UI sent lang="en" for an auto-detected
    # Tamil reply. The script of the text cannot be wrong about itself.
    actual = _script_of(text)
    if actual != lang:
        print(f"[speak] asked for {lang}, text is {actual} — using {actual}")
        lang = actual

    try:
        wav_bytes = tts.speak_bytes(text, lang)
        if not wav_bytes:
            # The unspeakable-text guard rejected it. 204 lets the interface say
            # so rather than play silence and look broken.
            return Response(status_code=204)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as exc:
        print(f"[speak] failed: {exc}")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/clear")
async def clear():
    llm.tutor.clear()
    _RETRIEVAL_CACHE.clear()
    return {"ok": True}


# ------------------------------------------------------------------ launch
def _open_browser_later():
    def runner():
        time.sleep(2.0)
        url = f"http://{config.HOST}:{config.PORT}"
        try:
            webbrowser.open(url)
        except Exception:
            pass
        print(f"\n>>> Open {url} in your browser (Ctrl+C here to stop)\n")
    threading.Thread(target=runner, daemon=True).start()


if __name__ == "__main__":
    if config.AUTO_OPEN_BROWSER:
        _open_browser_later()
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")
