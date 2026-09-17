"""Curated question and answer pairs, plus the marking for Study mode.

    python library.py --list                     what is loaded
    python library.py --check <id> "an answer"   mark one answer
    python library.py --suggest                  propose keywords for entries missing them
    python library.py --validate                 check the file before a demo

Content lives in `library_questions.json` **in the project root**, not in
`library/` — that directory is generated and git-ignored, and this file is
hand-written and must ship with the repository so a teammate cloning it gets the
questions too.

Everything here is deterministic. Nothing is generated, nothing is retrieved, no
model is loaded: the questions and answers are what you wrote, and marking is a
keyword comparison. That is the point. An earlier attempt auto-detected
questions from the PDFs and had the model write answers, and it was not good
enough to keep — the input was noisy and the output was unverifiable. Writing
the pairs by hand is more work and the result is something you can stand behind
in front of a judge.

Marking
-------
Each entry carries `keywords`: the terms an answer must contain to count. A
keyword may be a list, in which case any one of its alternatives satisfies it —
which is how "2", "two" and "இரண்டு" are treated as the same answer.

A student passes at LIBRARY_PASS_RATIO of the keywords (default 0.6). The
threshold is deliberately not 1.0: requiring every term punishes a correct
answer phrased differently, and a study tool that calls a right answer wrong
teaches the wrong lesson.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata

import config

CONTENT_FILE = "library_questions.json"

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")

_cache = None


# --------------------------------------------------------------------- loading
def content_path():
    return config.BASE_DIR / CONTENT_FILE


def load(force: bool = False) -> dict:
    global _cache
    if _cache is not None and not force:
        return _cache
    path = content_path()
    if not path.exists():
        _cache = {"questions": []}
        return _cache
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[library] could not read {CONTENT_FILE}: {exc}")
        _cache = {"questions": []}
        return _cache
    _cache = data
    return _cache


def reload():
    global _cache
    _cache = None


def entries(lang: str = "all", book: str = "all"):
    qs = load().get("questions") or []
    if lang != "all":
        qs = [q for q in qs if q.get("lang") == lang]
    if book != "all":
        qs = [q for q in qs if q.get("book") == book]
    return qs


def by_id(qid: str):
    for q in load().get("questions") or []:
        if q.get("id") == qid:
            return q
    return None


def summary():
    qs = load().get("questions") or []
    by_lang, by_book = {}, {}
    for q in qs:
        by_lang[q.get("lang", "?")] = by_lang.get(q.get("lang", "?"), 0) + 1
        by_book[q.get("book", "?")] = by_book.get(q.get("book", "?"), 0) + 1
    return {"total": len(qs), "by_lang": by_lang, "by_book": by_book,
            "missing_keywords": [q["id"] for q in qs if not q.get("keywords")]}


# --------------------------------------------------------------------- marking
def _normalise(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFC", text or "").lower()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _contains(haystack_spaced: str, haystack_tight: str, needle: str) -> bool:
    """Is this term present, allowing for spacing differences?

    Checked twice: once with single spaces, once with all whitespace removed.
    Speech transcription and Indic text both produce inconsistent word breaks,
    and a student should not be marked wrong because Whisper split a word.
    """
    n = _normalise(needle)
    if not n:
        return False
    if n in haystack_spaced:
        return True
    return _WS.sub("", n) in haystack_tight


def mark(answer: str, question: dict) -> dict:
    """Score a student's answer against one entry's keywords."""
    keywords = question.get("keywords") or []
    spaced = _normalise(answer)
    tight = _WS.sub("", spaced)

    hit, missed = [], []
    for kw in keywords:
        # A list of alternatives counts as satisfied if any one appears.
        alts = kw if isinstance(kw, (list, tuple)) else [kw]
        label = alts[0]
        if any(_contains(spaced, tight, a) for a in alts):
            hit.append(label)
        else:
            missed.append(label)

    ratio = (len(hit) / len(keywords)) if keywords else 0.0
    threshold = getattr(config, "LIBRARY_PASS_RATIO", 0.6)

    # An empty answer is never correct, however lenient the threshold.
    correct = bool(spaced) and bool(keywords) and ratio >= threshold

    return {
        "correct": correct,
        "ratio": round(ratio, 2),
        "matched": hit,
        "missed": missed,
        "answer": question.get("answer", ""),
        "citation": (f"{question['book']} p.{question['page']}"
                     if question.get("book") and question.get("page") else ""),
        "lang": question.get("lang", "en"),
        "threshold": threshold,
    }


# ------------------------------------------------------------------- authoring
_STOP = {
    "en": {"the","a","an","is","are","was","were","of","in","on","at","to","for",
           "and","or","but","it","its","this","that","these","those","with","by",
           "as","be","been","from","which","when","what","who","how","why","we",
           "you","they","he","she","them","their","there","then","than","can",
           "will","would","has","have","had","not","no","if","so","such","about"},
}


def suggest_keywords(answer: str, lang: str = "en", n: int = 3):
    """Propose keywords from an answer — a starting point, not a decision.

    Longest content words first: in a factual answer the informative terms are
    usually the long ones. Always review these; the whole value of this file is
    that a person chose what counts as correct.
    """
    words = _normalise(answer).split()
    stop = _STOP.get(lang, set())
    seen, out = set(), []
    for w in sorted(words, key=len, reverse=True):
        if w in stop or len(w) < 3 or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= n:
            break
    return out


def validate():
    """Report anything that would misbehave at runtime."""
    qs = load(force=True).get("questions") or []
    problems = []
    seen_ids = set()
    for i, q in enumerate(qs):
        where = q.get("id") or f"entry #{i+1}"
        for field in ("id", "lang", "question", "answer"):
            if not q.get(field):
                problems.append(f"{where}: missing '{field}'")
        if q.get("id") in seen_ids:
            problems.append(f"{where}: duplicate id")
        seen_ids.add(q.get("id"))
        if q.get("lang") not in config.LANGUAGES:
            problems.append(f"{where}: lang {q.get('lang')!r} is not one of "
                            f"{config.LANGUAGES}")
        if not q.get("keywords"):
            problems.append(f"{where}: no keywords — Study mode cannot mark it")
        else:
            # A keyword absent from your own answer will mark a correct student
            # wrong, which is the worst failure this file can have.
            res = mark(q.get("answer", ""), q)
            if res["missed"]:
                problems.append(
                    f"{where}: keyword(s) {res['missed']} do not appear in your "
                    f"own answer — a correct student would be marked wrong")
    return qs, problems


# ------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--suggest", action="store_true",
                    help="propose keywords for entries that have none")
    ap.add_argument("--check", nargs=2, metavar=("ID", "ANSWER"))
    args = ap.parse_args()

    if args.check:
        qid, answer = args.check
        q = by_id(qid)
        if not q:
            print(f"No entry with id {qid!r}")
            return 1
        r = mark(answer, q)
        print(f"question : {q['question']}")
        print(f"answer   : {answer}")
        print(f"matched  : {r['matched']}")
        print(f"missed   : {r['missed']}")
        print(f"ratio    : {r['ratio']} (pass at {r['threshold']})")
        print(f"verdict  : {'CORRECT' if r['correct'] else 'WRONG'}")
        return 0

    if args.validate:
        qs, problems = validate()
        print(f"{len(qs)} entries in {CONTENT_FILE}")
        if not problems:
            print("no problems found")
            return 0
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
        return 1

    if args.suggest:
        data = load(force=True)
        changed = 0
        for q in data.get("questions") or []:
            if q.get("keywords"):
                continue
            q["keywords"] = suggest_keywords(q.get("answer", ""), q.get("lang", "en"))
            changed += 1
            print(f"  {q['id']}: {q['keywords']}")
        if changed:
            content_path().write_text(
                json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"\nadded keywords to {changed} entries — REVIEW THEM. "
                  f"These are guesses from word length, not judgements.")
        else:
            print("every entry already has keywords")
        return 0

    if args.list:
        print(json.dumps(summary(), ensure_ascii=False, indent=2))
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
