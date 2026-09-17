"""Build a question pack from the indexed textbooks. Run once, offline.

    python qa_pack.py --build              every indexed book
    python qa_pack.py --build --book hindi one book
    python qa_pack.py --list               what the pack contains
    python qa_pack.py --test "a question"  what would match, and by how much

The split of work is the whole idea:

    "what questions does this passage answer?"  <- the model, at build time
    "what is the answer?"                       <- the passage itself, verbatim

Writing a question about a passage that is in front of you is a much easier job
than answering one correctly, so the model is used only for the easy half and
never stands between a student and an answer. At query time no model runs at
all: `qa_match` finds the nearest question and returns the passage it was
written from, with its page citation, in milliseconds.

Building is slow — one generation per chunk — but it happens once, on a laptop.
The resulting `library/qa_pack.json` is a few hundred KB and can be copied to a
Raspberry Pi, which then answers packed questions instantly without ever loading
the language model for them.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime

import config

_PROMPTS = {
    "en": ("Read the textbook passage below. Write {n} short questions that a "
           "school student might ask, which this passage clearly answers. "
           "Write one question per line, nothing else. No numbering, no "
           "explanation.\n\nPassage:\n{text}"),
    "hi": ("नीचे दिए गए पाठ्यपुस्तक अंश को पढ़िए। {n} छोटे प्रश्न लिखिए जो कोई "
           "विद्यार्थी पूछ सकता है और जिनका उत्तर इसी अंश में स्पष्ट रूप से है। "
           "प्रत्येक पंक्ति में एक प्रश्न लिखिए, और कुछ नहीं। कोई क्रमांक नहीं, "
           "कोई व्याख्या नहीं।\n\nअंश:\n{text}"),
    "ta": ("கீழே உள்ள பாடநூல் பகுதியைப் படியுங்கள். ஒரு மாணவர் கேட்கக்கூடிய, "
           "இந்தப் பகுதியில் தெளிவாக விடை உள்ள {n} சிறிய கேள்விகளை எழுதுங்கள். "
           "ஒவ்வொரு வரியிலும் ஒரு கேள்வி மட்டும். எண்ணிடல் வேண்டாம், விளக்கம் "
           "வேண்டாம்.\n\nபகுதி:\n{text}"),
}

_LEAD_NUM = re.compile(r"^\s*(?:\d+[.)]\s*|[-*•]\s*)")


def detect_lang(text: str) -> str:
    dev = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    tam = sum(1 for c in text if "஀" <= c <= "௿")
    if tam > dev and tam > 0:
        return "ta"
    if dev > 0:
        return "hi"
    return "en"


def _generate(model, prompt: str, key: str, max_tokens: int) -> str:
    """One raw completion, with the Gemma system-role fallback applied.

    The pack builder deliberately does not go through `llm.Tutor`: the tutor
    carries the grounding prompt and a rolling history, neither of which belongs
    in a build-time question-writing job.
    """
    import llm

    def call(messages):
        return model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.4,       # a little variety; identical questions are wasted entries
            top_p=0.9,
        )

    try:
        if llm._NO_SYSTEM_ROLE.get(key):
            raise RuntimeError("system role unsupported")
        res = call([{"role": "system", "content": "You write exam-style questions."},
                    {"role": "user", "content": prompt}])
    except Exception as exc:
        if "system" not in str(exc).lower():
            # Not the template complaint — let a real failure surface.
            if not llm._NO_SYSTEM_ROLE.get(key):
                raise
        llm._NO_SYSTEM_ROLE[key] = True
        res = call([{"role": "user", "content": prompt}])

    return res["choices"][0]["message"]["content"].strip()


def _questions_from(reply: str, want: int) -> list:
    out = []
    for line in reply.splitlines():
        line = _LEAD_NUM.sub("", line).strip().strip('"').strip()
        # A usable question is a single short line. Anything long is the model
        # explaining itself rather than following the instruction.
        if 8 <= len(line) <= 160 and "\n" not in line:
            out.append(line)
        if len(out) >= want:
            break
    return out


def build(book_filter: str = None, per_chunk: int = None, limit: int = None,
          verbose: bool = True) -> dict:
    import llm
    import rag

    per_chunk = per_chunk or getattr(config, "QA_PER_CHUNK", 2)

    rag.library.load_from_disk(verbose=False)
    if not rag.library.books:
        raise SystemExit("No textbooks indexed. Run: python ingest.py --all")

    targets = []
    for book_id, data in sorted(rag.library.books.items()):
        if book_filter and book_filter not in book_id:
            continue
        for chunk in data["chunks"]:
            targets.append((book_id, chunk))
    if limit:
        targets = targets[:limit]
    if not targets:
        raise SystemExit(f"No chunks matched book filter {book_filter!r}")

    if verbose:
        print(f"[qa] building from {len(targets)} chunks "
              f"({per_chunk} questions each) — this takes a while")

    entries, started, failures = [], time.time(), 0
    model_name = None

    for i, (book_id, chunk) in enumerate(targets, 1):
        text = (chunk.get("text") or "").strip()
        if len(text) < 120:            # too short to answer anything on its own
            continue
        lang = detect_lang(text)
        prompt = _PROMPTS[lang].format(n=per_chunk, text=text[:1200])

        try:
            model = llm.load(lang, verbose=False)
            key = llm.spec_for(lang)[1]
            model_name = model_name or key
            reply = _generate(model, prompt, key,
                              max_tokens=48 * per_chunk)
            qs = _questions_from(reply, per_chunk)
        except Exception as exc:
            failures += 1
            if verbose and failures <= 3:
                print(f"[qa]   chunk {i} failed: {exc}")
            continue

        for q in qs:
            entries.append({"q": q, "lang": lang, "book": book_id,
                            "page": chunk.get("page"), "answer": text})

        if verbose and (i % 10 == 0 or i == len(targets)):
            done = time.time() - started
            rate = i / done if done else 0
            eta = (len(targets) - i) / rate if rate else 0
            print(f"[qa]   {i}/{len(targets)} chunks · {len(entries)} questions · "
                  f"{done/60:.1f} min elapsed · ~{eta/60:.1f} min left")

    pack = {
        "version": 1,
        "built": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "per_chunk": per_chunk,
        "chunks_used": len(targets),
        "failures": failures,
        "entries": entries,
    }

    config.LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    path = config.LIBRARY_DIR / "qa_pack.json"
    path.write_text(json.dumps(pack, ensure_ascii=False, indent=1), encoding="utf-8")

    if verbose:
        by = {}
        for e in entries:
            by[e["lang"]] = by.get(e["lang"], 0) + 1
        print(f"\n[qa] wrote {path.name} — {len(entries)} questions {by}")
        print(f"[qa] {failures} chunks failed, {time.time()-started:.0f}s total")
        thin = [l for l, n in by.items()
                if n < getattr(config, 'QA_MIN_LANG_ENTRIES', 12)]
        if thin:
            print(f"[qa] NOTE: {thin} have too few entries to be trusted and "
                  f"will not answer from the pack (QA_MIN_LANG_ENTRIES).")
    return pack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--book", default=None, help="only chunks from this book id")
    ap.add_argument("--per-chunk", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N chunks — use this for a quick trial run")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--test", default=None, help="show what this question matches")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    if args.build:
        build(args.book, args.per_chunk, args.limit)
        return 0

    import qa_match
    qa_match.reload()

    if args.list:
        s = qa_match.summary()
        print(json.dumps(s, ensure_ascii=False, indent=2))
        return 0

    if args.test:
        lang = qa_match.detect_lang(args.test)
        print(f"question : {args.test}")
        print(f"language : {lang}")
        thr = args.threshold if args.threshold is not None else \
            getattr(config, "QA_MATCH_MIN", 0.55)
        print(f"threshold: {thr}\n")
        print("nearest pack questions:")
        for r in qa_match.rank(args.test, lang=lang, top=5):
            mark = "MATCH " if r["score"] >= thr else "      "
            print(f"  {mark}{r['score']:.3f}  {r['q'][:70]}   [{r['citation']}]")
        hit = qa_match.match(args.test, lang=lang, threshold=thr)
        print("\nserved:", "pack answer — " + hit["citation"] if hit
              else "nothing; falls through to retrieval + model")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
