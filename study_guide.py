"""Study Guide: the textbook's own questions, answered once and cached.

    python study_guide.py --scan        find the questions, write them for review
    python study_guide.py --review      print what was found
    python study_guide.py --build       answer the reviewed questions (slow)
    python study_guide.py --list        what the finished guide contains

Two stages on purpose, with a human in the middle.

`--scan` writes `library/study_questions.json`. Open it, delete anything that is
not a real question, set `"include": false` on anything you want skipped, then
run `--build`. Detection is a heuristic over extracted PDF text; reviewing 60
lines takes two minutes and is much cheaper than discovering after an hour of
generation that a third of the entries are page headings.

The distinction from `qa_pack.py` matters:

    qa_pack      questions the MODEL invented, answered by the book verbatim
    study_guide  questions the BOOK asks, answered by the model

So a pack answer is a quotation and a study-guide answer is generated text.
They are labelled differently in the interface for exactly that reason, and a
study-guide answer always carries its page citations.

If the model cannot answer a question from the retrieved pages it emits the
not-in-book marker. Those are stored with `answered: false` and are not shown as
answers — a study guide full of confident guesses would be worse than a short
one.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime

import config

QUESTIONS_FILE = "study_questions.json"
GUIDE_FILE = "study_guide.json"

# Sentence-ending question marks, including the full-width form some PDFs use.
_Q_END = re.compile(r"[?？]\s*$")
_SPLIT = re.compile(r"(?<=[.!?।。?？])\s+|\n+")

# Numbered or bulleted exercise items: "1. ...", "(iv) ...", "अ) ...".
_NUMBERED = re.compile(r"^\s*(?:\(?\d{1,2}[.)]|\(?[ivxIVX]{1,4}[.)]|[-*•])\s+")

# Headings that mark an exercise block in the three languages.
_EXERCISE_HEAD = re.compile(
    r"(exercise|questions?|answer the following|practice"
    r"|प्रश्न|अभ्यास|प्रश्नोत्तर"
    r"|வினா|பயிற்சி|கேள்வி)", re.IGNORECASE)

_MIN_LEN, _MAX_LEN = 12, 220


def detect_lang(text: str) -> str:
    dev = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    tam = sum(1 for c in text if "஀" <= c <= "௿")
    if tam > dev and tam > 0:
        return "ta"
    if dev > 0:
        return "hi"
    return "en"


def _clean(line: str) -> str:
    line = _NUMBERED.sub("", line).strip()
    line = re.sub(r"\s+", " ", line)
    return line.strip(" -–—*•\t")


def _candidates(text: str, in_exercise: bool):
    """Yield question-like strings from one chunk.

    Lines first, sentences second — and in that order deliberately. Splitting
    the whole chunk on sentence punctuation breaks "3. Write the uses of a
    telescope." into "3." and the rest, which destroys the numbering that marks
    it as an exercise item in the first place. So a numbered line is taken whole,
    and only un-numbered lines are split into sentences.
    """
    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        if _NUMBERED.match(raw_line):
            # A numbered item inside an exercise block counts even without a
            # question mark: "Write the uses of a convex lens." is a question in
            # every way except punctuation. Outside such a block it needs one,
            # or every bulleted list in the book would be swept up.
            line = _clean(raw_line)
            if _MIN_LEN <= len(line) <= _MAX_LEN and (in_exercise or _Q_END.search(line)):
                yield line
            continue

        for piece in _SPLIT.split(raw_line):
            line = _clean(piece)
            if _MIN_LEN <= len(line) <= _MAX_LEN and _Q_END.search(line):
                yield line


def scan(verbose: bool = True) -> dict:
    """Find candidate questions in every indexed book."""
    import rag

    rag.library.load_from_disk(verbose=False)
    if not rag.library.books:
        raise SystemExit("No textbooks indexed. Run: python ingest.py --all")

    found, seen = [], set()
    for book_id, data in sorted(rag.library.books.items()):
        n_before = len(found)
        for chunk in data["chunks"]:
            text = chunk.get("text") or ""
            in_exercise = bool(_EXERCISE_HEAD.search(text))
            for line in _candidates(text, in_exercise):
                key = re.sub(r"\W+", "", line.lower())[:80]
                if key in seen:
                    continue
                seen.add(key)
                found.append({
                    "id": f"{book_id}-{len(found)+1:03d}",
                    "book": book_id,
                    "page": chunk.get("page"),
                    "lang": detect_lang(line),
                    "question": line,
                    "include": True,
                })
        if verbose:
            print(f"  {book_id:<20} {len(found) - n_before} questions")

    payload = {"version": 1,
               "scanned": datetime.now().isoformat(timespec="seconds"),
               "questions": found}
    config.LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    (config.LIBRARY_DIR / QUESTIONS_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    if verbose:
        by = {}
        for q in found:
            by[q["lang"]] = by.get(q["lang"], 0) + 1
        print(f"\nfound {len(found)} candidate questions {by}")
        print(f"wrote library/{QUESTIONS_FILE}")
        print("\nOPEN THAT FILE AND CHECK IT before building. Detection is a")
        print("heuristic — delete anything that is not a real question, or set")
        print('"include": false on it. Then run: python study_guide.py --build')
    return payload


def load_questions():
    path = config.LIBRARY_DIR / QUESTIONS_FILE
    if not path.exists():
        raise SystemExit("No scan yet. Run: python study_guide.py --scan")
    return json.loads(path.read_text(encoding="utf-8"))


def build(limit: int = None, lang: str = None, verbose: bool = True) -> dict:
    """Answer each included question once, grounded, with citations."""
    import llm
    import rag

    rag.library.load_from_disk(verbose=False)
    rag.load_embedder(verbose=False)

    questions = [q for q in load_questions()["questions"] if q.get("include", True)]
    if lang:
        questions = [q for q in questions if q["lang"] == lang]
    if limit:
        questions = questions[:limit]
    if not questions:
        raise SystemExit("Nothing to build — every question is excluded.")

    if verbose:
        print(f"[guide] answering {len(questions)} questions "
              f"— roughly {len(questions) * 30 // 60} minutes\n")

    entries, started, refused = [], time.time(), 0
    for i, q in enumerate(questions, 1):
        try:
            context, citations = rag.library.context_for(q["question"])
            reply = llm.tutor.ask(q["question"], lang=q["lang"], context=context)
        except Exception as exc:
            if verbose:
                print(f"  {q['id']} failed: {exc}")
            continue

        # The model reports the pages do not answer this. Record that honestly
        # rather than storing whatever it said next — an unanswered question in
        # a study guide is a gap; a confidently wrong one is a lie with a page
        # number attached.
        answered = True
        if config.NOT_IN_BOOK_MARKER in reply:
            reply = reply.replace(config.NOT_IN_BOOK_MARKER, "").strip()
            answered = False
            citations = []
            refused += 1

        entries.append({**{k: q[k] for k in ("id", "book", "page", "lang", "question")},
                        "answer": reply, "citations": citations,
                        "answered": answered})

        if verbose and (i % 5 == 0 or i == len(questions)):
            done = time.time() - started
            eta = (len(questions) - i) * (done / i)
            print(f"  {i}/{len(questions)} · {done/60:.1f} min elapsed · "
                  f"~{eta/60:.1f} min left")

    guide = {"version": 1,
             "built": datetime.now().isoformat(timespec="seconds"),
             "model": config.LLM_FILE,
             "answered": sum(1 for e in entries if e["answered"]),
             "unanswered": refused,
             "entries": entries}
    (config.LIBRARY_DIR / GUIDE_FILE).write_text(
        json.dumps(guide, ensure_ascii=False, indent=1), encoding="utf-8")

    if verbose:
        print(f"\n[guide] wrote library/{GUIDE_FILE}")
        print(f"        {guide['answered']} answered, {refused} not covered by "
              f"the books")
        if refused:
            print("        Unanswered entries are kept and shown as gaps, not "
                  "filled in.")
    return guide


def load_guide():
    """Read the built guide. Returns an empty guide when there is none."""
    path = config.LIBRARY_DIR / GUIDE_FILE
    if not path.exists():
        return {"entries": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[guide] could not read the guide: {exc}")
        return {"entries": []}


def summary():
    g = load_guide()
    by_lang, by_book = {}, {}
    for e in g.get("entries", []):
        by_lang[e["lang"]] = by_lang.get(e["lang"], 0) + 1
        by_book[e["book"]] = by_book.get(e["book"], 0) + 1
    return {"total": len(g.get("entries", [])),
            "answered": g.get("answered"), "unanswered": g.get("unanswered"),
            "built": g.get("built"), "by_lang": by_lang, "by_book": by_book}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.scan:
        scan()
        return 0

    if args.review:
        data = load_questions()
        for q in data["questions"]:
            mark = " " if q.get("include", True) else "x"
            print(f"[{mark}] {q['id']:<18} {q['lang']}  p.{q['page']:<4} "
                  f"{q['question'][:70]}")
        print(f"\n{len(data['questions'])} candidates. "
              f"Edit library/{QUESTIONS_FILE} to exclude any.")
        return 0

    if args.build:
        build(limit=args.limit, lang=args.lang)
        return 0

    if args.list:
        print(json.dumps(summary(), ensure_ascii=False, indent=2))
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
