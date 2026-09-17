"""
Post-install self-check.

    python verify.py           full check (loads every model, ~60-90 s)
    python verify.py --quick   imports and files only, no model loading

Exits non-zero if anything is wrong, so setup.sh can gate on it.
"""
from __future__ import annotations

import sys
import time

QUICK = "--quick" in sys.argv
FAILS = []
WARNS = []


def ok(name, extra=""):
    print(f"  \033[32mPASS\033[0m  {name}" + (f"  {extra}" if extra else ""))


def bad(name, why):
    print(f"  \033[31mFAIL\033[0m  {name}\n        {why}")
    FAILS.append(name)


def warn(name, why):
    print(f"  \033[33mWARN\033[0m  {name}\n        {why}")
    WARNS.append(name)


def section(t):
    print(f"\n\033[1m{t}\033[0m")


# ---------------------------------------------------------------- imports
section("Python packages")
for mod, label in [
    ("numpy", "numpy"), ("faster_whisper", "faster-whisper"),
    ("llama_cpp", "llama-cpp-python"), ("torch", "torch"),
    ("transformers", "transformers"), ("sentence_transformers", "sentence-transformers"),
    ("fitz", "pymupdf"), ("fastapi", "fastapi"), ("uvicorn", "uvicorn"),
    ("multipart", "python-multipart"),
]:
    try:
        m = __import__(mod)
        ok(label, getattr(m, "__version__", ""))
    except Exception as e:
        bad(label, f"import failed: {e}")

try:
    import sounddevice  # noqa: F401
    ok("sounddevice", "(terminal mode only)")
except Exception:
    warn("sounddevice", "unavailable — expected on WSL, web.py does not need it")

if FAILS:
    print(f"\n\033[31m{len(FAILS)} package(s) missing. Re-run ./setup.sh\033[0m")
    sys.exit(1)

# ---------------------------------------------------------------- config
section("Project layout")
import config  # noqa: E402

for d, label in [(config.MODELS_DIR, "models/"), (config.BOOKS_DIR, "books/"),
                 (config.LIBRARY_DIR, "library/")]:
    ok(label, "exists") if d.exists() else bad(label, "missing")

_seen = set()
for _lang, (_repo, _file, _nt) in config.LLM_MODELS.items():
    if _file in _seen:
        continue
    _seen.add(_file)
    gguf = config.LLM_DIR / _file
    langs = ",".join(l for l, s in config.LLM_MODELS.items() if s[1] == _file)
    if not gguf.exists():
        bad(f"LLM weights [{langs}]", f"{_file} not found — re-run ./setup.sh")
        continue
    size_gb = gguf.stat().st_size / 1e9
    if size_gb < 1.0:
        bad(f"LLM weights [{langs}]",
            f"{_file} is only {size_gb:.2f} GB — download truncated. "
            f"Delete it and re-run ./setup.sh")
    else:
        ok(f"LLM weights [{langs}]", f"{size_gb:.2f} GB")

ok("resident-model cap", f"LLM_MAX_RESIDENT={getattr(config,'LLM_MAX_RESIDENT',1)}")

if config.WHISPER_DIR.exists() and any(config.WHISPER_DIR.rglob("*.bin")):
    ok("Whisper weights", config.WHISPER_MODEL)
else:
    warn("Whisper weights", "not cached yet — will download on first run")

if QUICK:
    print("\n\033[1mQuick check done (models not loaded).\033[0m")
    sys.exit(1 if FAILS else 0)

# ---------------------------------------------------------------- models
section("Loading models (this is the slow part)")

t0 = time.time()
try:
    import stt
    stt.load(verbose=False)
    ok("faster-whisper loads", f"{time.time()-t0:.1f}s")
except Exception as e:
    bad("faster-whisper loads", e)

t0 = time.time()
try:
    import llm
    llm.load(verbose=False)
    ok("llama.cpp loads", f"{time.time()-t0:.1f}s")
except Exception as e:
    bad("llama.cpp loads", e)

t0 = time.time()
try:
    import tts
    for lang in config.LANGUAGES:
        tts.load(lang, verbose=False)
    ok("MMS-TTS voices load", f"3 voices, {time.time()-t0:.1f}s")
except Exception as e:
    bad("MMS-TTS voices load", e)

t0 = time.time()
try:
    import rag
    rag.load_embedder(verbose=False)
    ok("embedder loads", f"{time.time()-t0:.1f}s")
except Exception as e:
    bad("embedder loads", e)

# ---------------------------------------------------------------- end to end
section("End-to-end smoke test")

if not FAILS:
    try:
        t0 = time.time()
        reply = llm.tutor.ask("What is water made of?", lang="en")
        dt = time.time() - t0
        if len(reply) < 10:
            bad("LLM generates", f"suspiciously short: {reply!r}")
        else:
            ok("LLM generates", f"{dt:.1f}s — {reply[:60]}...")
            if dt > 25:
                warn("LLM speed", f"{dt:.0f}s per answer. See Tuning in the guide.")
    except Exception as e:
        bad("LLM generates", e)

    for lang, phrase in [("en", "Water is made of hydrogen and oxygen."),
                         ("hi", "पानी हाइड्रोजन और ऑक्सीजन से बना है।"),
                         ("ta", "நீர் ஹைட்ரஜன் மற்றும் ஆக்ஸிஜனால் ஆனது.")]:
        try:
            t0 = time.time()
            wav, sr = tts.synth(phrase, lang)
            if wav.size < sr * 0.2:
                bad(f"TTS {lang}", f"only {wav.size/sr:.2f}s of audio")
            else:
                ok(f"TTS {lang}", f"{wav.size/sr:.1f}s audio, {time.time()-t0:.1f}s")
        except Exception as e:
            bad(f"TTS {lang}", e)

    try:
        import numpy as np
        v = rag.load_embedder(verbose=False).encode(
            ["photosynthesis", "ஒளிச்சேர்க்கை"], convert_to_numpy=True,
            normalize_embeddings=True)
        sim = float(v[0] @ v[1])
        ok("multilingual embeddings", f"en/ta similarity {sim:.2f}")
        if sim < 0.3:
            warn("cross-language retrieval", f"similarity {sim:.2f} is low")
    except Exception as e:
        bad("multilingual embeddings", e)

    books = rag.library.summary() if rag.library.books else []
    if books:
        ok("textbooks indexed", f"{len(books)} book(s)")
    else:
        warn("textbooks", "none indexed — drop PDFs in books/ (optional)")

# ---------------------------------------------------------------- verdict
print("\n" + "=" * 52)
if FAILS:
    print(f"\033[31m{len(FAILS)} check(s) FAILED:\033[0m")
    for f in FAILS:
        print(f"  - {f}")
    print("\nSee Part 6 (Troubleshooting) in the build guide.")
    sys.exit(1)

print("\033[32mAll checks passed. EduBuddy is ready.\033[0m")
if WARNS:
    print(f"({len(WARNS)} warning(s), none blocking: {', '.join(WARNS)})")
print("\nStart it with:  python web.py")
sys.exit(0)
