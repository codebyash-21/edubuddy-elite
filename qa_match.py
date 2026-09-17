"""Match a student's question against a prebuilt question pack. No model runs.

The pack maps questions to the textbook passage they were written from. A hit
returns that passage **verbatim**, with its page citation, in a few
milliseconds. Nothing is generated at query time, so a pack answer is the book's
own words rather than a cached model reply — which is why it is honest to serve
one, and why the interface labels it as a direct quote.

The whole risk here is a false positive: an off-topic question matching an entry
on shared function words and coming back with a citation attached. Four things
guard against that, in order of importance:

  1. IDF weighting. Every question in a pack shares "what", "is the", "which".
     Unweighted, that scaffolding dominates the score and an unrelated question
     scores as well as a genuine paraphrase — measured, it left no threshold
     that both fired and was safe. Weighting by inverse document frequency
     pushes the shared parts towards zero so the topic words decide.
  2. Jaccard rather than containment. Containment rewards a short query that
     happens to sit inside a long question; Jaccard divides by the union, so a
     query sharing a few substrings with a much longer question scores low.
  3. A language gate. An English question is only ever matched against English
     entries, so shared Latin function words cannot pull in a Hindi passage.
  4. A measured floor (QA_MATCH_MIN). Below it the question goes to normal
     retrieval and the model, which can still answer or refuse.

The score distributions of real and off-topic questions overlap, so the floor
cannot be perfect. It is set low deliberately: a miss costs latency, a false
positive costs a wrong answer wearing a page number.

Use `python qa_pack.py --test "some question"` to probe the margin before
trusting a threshold — especially on questions the books do not cover.
"""
from __future__ import annotations

import json
import re
import threading
import unicodedata

import config

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

_pack = None
_lock = threading.Lock()


# --------------------------------------------------------------------- pack io
def pack_path():
    return config.LIBRARY_DIR / "qa_pack.json"


def load(verbose: bool = False):
    """Read the pack from disk once. Returns {} when there is no pack."""
    global _pack
    if _pack is not None:
        return _pack
    with _lock:
        if _pack is not None:
            return _pack
        path = pack_path()
        if not path.exists():
            _pack = {"entries": []}
            if verbose:
                print("[qa] no question pack — run: python qa_pack.py --build")
            return _pack
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[qa] could not read the pack: {exc}")
            _pack = {"entries": []}
            return _pack

        entries = data.get("entries", [])
        for e in entries:
            e["_grams"] = _grams(e["q"])
        data["_idf"] = _build_idf(entries)
        _pack = data
        if verbose:
            print(f"[qa] pack loaded — {len(entries)} entries")
    return _pack


def _build_idf(entries) -> dict:
    """Inverse document frequency for every n-gram in the pack.

    Without this the score is dominated by the substrings every question shares
    — "what", "is the", "which" — and an off-topic question scores about as well
    as a genuine paraphrase. Measured on a 14-question English pack: unweighted,
    real paraphrases scored 0.23-0.49 while an unrelated question about a night
    market scored 0.37, leaving no threshold that both fires and is safe.
    Weighting by IDF pushes the shared scaffolding towards zero and lets the
    content-bearing grams decide.
    """
    import math

    df: dict[str, int] = {}
    for e in entries:
        for g in e["_grams"]:
            df[g] = df.get(g, 0) + 1
    n = max(1, len(entries))
    return {g: math.log(1.0 + n / c) for g, c in df.items()}


def _idf(pack, gram: str) -> float:
    import math
    table = pack.get("_idf") or {}
    # A gram the pack has never seen is maximally informative, but it also cannot
    # appear in any entry, so it only ever lands in the union — which correctly
    # penalises a query full of unfamiliar wording.
    default = math.log(1.0 + max(1, len(pack.get("entries") or [])))
    return table.get(gram, default)


def reload():
    """Drop the cached pack so the next load() re-reads it."""
    global _pack
    with _lock:
        _pack = None


def is_ready() -> bool:
    return bool((_pack or {}).get("entries"))


# ------------------------------------------------------------------- matching
def _grams(text: str, n: int = None) -> set:
    """Character n-grams of a normalised question.

    Case folded, punctuation dropped and whitespace removed. Whitespace goes
    because the extracted textbook text has stray spaces inside words, which
    destroys word tokens but leaves character n-grams nearly intact.
    """
    n = n or getattr(config, "QA_NGRAM", 3)
    s = unicodedata.normalize("NFC", text or "").lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub("", s)
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: set, b: set, pack=None) -> float:
    """IDF-weighted Jaccard: shared weight over total weight.

    Jaccard rather than containment, because containment rewards a short query
    for sitting inside a long question. Weighted, because unweighted treats
    "what is the" as worth the same as the one substring that carries the topic.
    """
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    if pack is None:
        return len(inter) / len(a | b)
    w_inter = sum(_idf(pack, g) for g in inter)
    w_union = sum(_idf(pack, g) for g in (a | b))
    return (w_inter / w_union) if w_union else 0.0


def detect_lang(text: str) -> str:
    """Which of en/hi/ta a question is written in, by counting codepoints."""
    dev = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    tam = sum(1 for c in text if "஀" <= c <= "௿")
    if tam > dev and tam > 0:
        return "ta"
    if dev > 0:
        return "hi"
    return "en"


def match(question: str, lang: str = None, threshold: float = None):
    """Best pack entry for this question, or None.

    Returns {'answer', 'citation', 'book', 'page', 'question', 'score', 'lang'}.
    """
    if not getattr(config, "QA_ENABLED", True):
        return None
    question = (question or "").strip()
    if not question:
        return None

    pack = load()
    entries = pack.get("entries") or []
    if not entries:
        return None

    lang = config.normalise_lang(lang) if lang else detect_lang(question)
    threshold = getattr(config, "QA_MATCH_MIN", 0.55) if threshold is None else threshold

    # A language with too few entries is not trusted to answer at all: a thin
    # pack means the nearest entry is nearest by accident rather than by topic.
    same_lang = [e for e in entries if e.get("lang") == lang]
    if len(same_lang) < getattr(config, "QA_MIN_LANG_ENTRIES", 12):
        return None

    q_grams = _grams(question)
    best, best_score = None, 0.0
    for e in same_lang:
        score = _jaccard(q_grams, e.get("_grams") or _grams(e["q"]), pack)
        if score > best_score:
            best, best_score = e, score

    if best is None or best_score < threshold:
        return None

    return {
        "answer": best["answer"],
        "citation": f"{best['book']} p.{best['page']}",
        "book": best["book"],
        "page": best["page"],
        "question": best["q"],
        "score": round(best_score, 3),
        "lang": lang,
    }


def rank(question: str, lang: str = None, top: int = 5):
    """Top-N candidates with scores — for tuning the threshold, not for serving."""
    pack = load()
    entries = pack.get("entries") or []
    lang = config.normalise_lang(lang) if lang else detect_lang(question)
    q_grams = _grams(question)
    scored = [
        (_jaccard(q_grams, e.get("_grams") or _grams(e["q"]), pack), e)
        for e in entries if e.get("lang") == lang
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": round(s, 3), "q": e["q"],
             "citation": f"{e['book']} p.{e['page']}"} for s, e in scored[:top]]


def summary():
    pack = load()
    counts = {}
    for e in pack.get("entries") or []:
        counts[e.get("lang", "?")] = counts.get(e.get("lang", "?"), 0) + 1
    return {"total": len(pack.get("entries") or []),
            "by_lang": counts,
            "built": pack.get("built"),
            "model": pack.get("model")}
