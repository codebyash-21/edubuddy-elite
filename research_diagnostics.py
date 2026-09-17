"""
Retrieval ablation harness for EduBuddy.

    python research_diagnostics.py                  all ablations, no LLM
    python research_diagnostics.py --quick          skip the slow chunk-size study
    python research_diagnostics.py --lang hi        one language only

No language model is loaded, so a full run takes a couple of minutes rather than
the half hour evaluate.py needs. That is the point: retrieval quality bounds
everything downstream, so it is worth measuring on its own, repeatedly, and
against alternatives.

Every section is an ABLATION -- the system is measured with one design decision
switched off, so the contribution of that decision can be stated as a number
rather than asserted. Results are written to results/ as JSON (machine-readable,
for later comparison) and Markdown (readable, for the report).

Metrics
-------
coverage@k   share of a question's expected answer terms present in the context
             the model is given at RAG_TOP_K = k. This is a ceiling: no model,
             however good, can answer beyond what retrieval delivered.
first-rank   position of the first chunk containing any expected term, searched
             to depth 50. Lower is better; None means never found.
MRR          mean of 1/first-rank. The standard single-number summary of a
             ranking; rewards putting the right chunk near the top, not merely
             somewhere in the list.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
import time
from datetime import datetime

import config

DEEP = 50           # search depth used for rank metrics
SEP = "=" * 74

# evaluate.py owns this path; config.py does not define it, so mirror it here
# rather than importing evaluate (which pulls in the model stack).
RESULTS_DIR = config.BASE_DIR / "results"


# --------------------------------------------------------------------- helpers
def _cases(path, lang):
    spec = json.loads((config.BASE_DIR / path).read_text(encoding="utf-8"))
    cases = [c for c in spec["cases"] if c.get("kind") != "negative" and c.get("any")]
    if lang != "all":
        cases = [c for c in cases if c["lang"] == lang]
    return cases


def _coverage(rag, case):
    ctx, _ = rag.library.context_for(case["question"])
    want = case["any"]
    return sum(1 for t in want if t in ctx) / len(want)


def _first_rank(rag, case):
    """1-based rank of the first retrieved chunk containing an expected term."""
    hits = rag.library.search(case["question"], top_k=DEEP, min_score=0.0)
    for i, h in enumerate(hits, 1):
        if any(t in h["text"] for t in case["any"]):
            return i
    return None


def _mrr(ranks):
    return statistics.fmean([1.0 / r if r else 0.0 for r in ranks])


def _agg(rag, cases):
    """Coverage / first-rank / MRR under the current config."""
    cov, ranks = [], []
    for c in cases:
        cov.append(_coverage(rag, c))
        ranks.append(_first_rank(rag, c))
    found = [r for r in ranks if r]
    return {
        "coverage": statistics.fmean(cov),
        "mrr": _mrr(ranks),
        "found": len(found),
        "n": len(cases),
        "median_rank": statistics.median(found) if found else None,
        "per_case": {c["id"]: {"coverage": round(v, 3), "rank": r}
                     for c, v, r in zip(cases, cov, ranks)},
    }


class Restore:
    """Set config attributes, then put them back. Ablations must not leak."""

    def __init__(self, **kw):
        self.kw = kw
        self.old = {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.old[k] = getattr(config, k, None)
            setattr(config, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            setattr(config, k, v)
        return False


# ------------------------------------------------------------------ ablations
def ablation_retrieval_mode(rag, cases, out):
    """Hybrid RRF vs semantic-only. The headline retrieval decision."""
    print("\n[1/6] retrieval mode: semantic-only vs hybrid RRF")
    res = {}
    for label, hybrid in (("semantic_only", False), ("hybrid_rrf", True)):
        with Restore(RAG_HYBRID=hybrid):
            res[label] = _agg(rag, cases)
        r = res[label]
        print(f"      {label:<15} coverage {r['coverage']*100:5.1f}%   "
              f"MRR {r['mrr']:.3f}   found {r['found']}/{r['n']}")
    out["retrieval_mode"] = res


def ablation_topk(rag, cases, out, values=(2, 4, 6, 8, 12)):
    """How much context is enough? Coverage rises with k; latency does too."""
    print("\n[2/6] RAG_TOP_K sweep")
    res = {}
    for k in values:
        with Restore(RAG_TOP_K=k):
            res[str(k)] = _agg(rag, cases)
        print(f"      k={k:<3} coverage {res[str(k)]['coverage']*100:5.1f}%")
    out["topk"] = res


def ablation_opening_chunk(rag, cases, out):
    """Prepending the chapter opening: does it help, or only cost a slot?"""
    print("\n[3/6] opening-chunk injection")
    res = {}
    for label, flag in (("off", False), ("on", True)):
        with Restore(RAG_INCLUDE_OPENING=flag):
            res[label] = _agg(rag, cases)
        print(f"      opening={label:<4} coverage {res[label]['coverage']*100:5.1f}%")
    out["opening_chunk"] = res


def ablation_ngram(rag, cases, out, values=(2, 3, 4, 5)):
    """Character n-gram size for the lexical half of the hybrid."""
    print("\n[4/6] lexical n-gram size")
    res = {}
    for n in values:
        with Restore(RAG_LEX_NGRAM=n, RAG_HYBRID=True):
            res[str(n)] = _agg(rag, cases)
        print(f"      n={n}   coverage {res[str(n)]['coverage']*100:5.1f}%   "
              f"MRR {res[str(n)]['mrr']:.3f}")
    out["ngram"] = res


def ablation_min_score(rag, cases, out, values=(0.0, 0.30, 0.45, 0.60)):
    """The cosine floor. Too high and correct chunks get vetoed."""
    print("\n[5/6] RAG_MIN_SCORE floor")
    res = {}
    for s in values:
        with Restore(RAG_MIN_SCORE=s):
            res[f"{s:.2f}"] = _agg(rag, cases)
        print(f"      min_score={s:.2f}  coverage {res[f'{s:.2f}']['coverage']*100:5.1f}%")
    out["min_score"] = res


def study_indic_repair(out):
    """Measure the PDF glyph repair directly on the source files.

    This needs no index and no embedder: extract each page twice, once raw and
    once repaired, and count the corruption. Cheap, and it isolates a fault that
    would otherwise be blamed on the retriever or the model.
    """
    print("\n[6/6] Indic glyph repair (measured on source PDFs)")
    try:
        import fitz  # noqa: F401
    except ImportError:
        print("      PyMuPDF unavailable, skipping")
        return

    import rag as ragmod

    res = {}
    pdfs = sorted(config.BOOKS_DIR.glob("*.pdf"))
    if not pdfs:
        print("      books/ is empty, skipping")
        return

    dbl_pulli = re.compile("்{2,}")
    dbl_matra = re.compile("([ा-ौॢॣ])\\1+")

    for pdf in pdfs:
        import fitz
        raw = []
        with fitz.open(pdf) as doc:
            for page in doc:
                raw.append(page.get_text("text") or "")
        raw_text = "\n".join(raw)
        fixed_text = ragmod._repair_text(raw_text)

        entry = {
            "pages": len(raw),
            "chars": len(raw_text),
            "doubled_pulli_before": len(dbl_pulli.findall(raw_text)),
            "doubled_pulli_after": len(dbl_pulli.findall(fixed_text)),
            "doubled_matra_before": len(dbl_matra.findall(raw_text)),
            "doubled_matra_after": len(dbl_matra.findall(fixed_text)),
            "replacement_chars_before": raw_text.count("�"),
            "replacement_chars_after": fixed_text.count("�"),
        }
        res[pdf.stem] = entry
        touched = (entry["doubled_pulli_before"] + entry["doubled_matra_before"]
                   + entry["replacement_chars_before"])
        print(f"      {pdf.stem:<20} corrupt glyph clusters {touched:>5} -> "
              f"{entry['doubled_pulli_after'] + entry['doubled_matra_after'] + entry['replacement_chars_after']}")
    out["indic_repair"] = res


def study_term_recovery(out, cases):
    """Which expected answer terms exist in the raw text vs the repaired text?

    A term absent from BOTH is a broken test case, not a retrieval failure --
    worth separating before drawing any conclusion from a red row.
    """
    try:
        import fitz  # noqa: F401
    except ImportError:
        return
    import fitz
    import rag as ragmod

    raw_all, fixed_all = [], []
    for pdf in sorted(config.BOOKS_DIR.glob("*.pdf")):
        with fitz.open(pdf) as doc:
            t = "\n".join(page.get_text("text") or "" for page in doc)
        raw_all.append(t)
        fixed_all.append(ragmod._repair_text(t))
    raw_text, fixed_text = "\n".join(raw_all), "\n".join(fixed_all)

    rows = []
    for c in cases:
        for term in c["any"]:
            rows.append({"case": c["id"], "lang": c["lang"], "term": term,
                         "in_raw": term in raw_text, "in_repaired": term in fixed_text})
    recovered = [r for r in rows if r["in_repaired"] and not r["in_raw"]]
    absent = [r for r in rows if not r["in_repaired"]]
    print(f"\n      answer terms checked against source : {len(rows)}")
    print(f"      recovered by the repair             : {len(recovered)}")
    print(f"      absent from the book entirely       : {len(absent)}"
          f"{'  <- these test cases are unanswerable' if absent else ''}")
    for r in absent:
        print(f"          {r['case']}  {r['term']!r}")
    out["term_recovery"] = {"rows": rows,
                            "recovered": len(recovered),
                            "absent": len(absent),
                            "total": len(rows)}


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="all", help="all | en | hi | ta")
    ap.add_argument("--cases", default="testcases.json")
    ap.add_argument("--quick", action="store_true",
                    help="fewer sweep points; finishes in well under a minute")
    args = ap.parse_args()

    started = time.time()

    import rag
    rag.library.load_from_disk(verbose=False)
    if not rag.library.books:
        print("No textbooks indexed. Run: python ingest.py --all")
        return 1
    rag.load_embedder(verbose=False)

    cases = _cases(args.cases, args.lang)
    if not cases:
        print("No matching cases.")
        return 1

    print(SEP)
    print("EduBuddy retrieval diagnostics")
    print(SEP)
    print(f"books    : {[b['book'] for b in rag.library.summary()]}")
    print(f"chunks   : {sum(b['chunks'] for b in rag.library.summary())}")
    print(f"cases    : {len(cases)} ({args.lang})")
    print(f"profile  : {config.PROFILE}   embedder: {config.EMBED_MODEL}")
    print(f"baseline : k={config.RAG_TOP_K} hybrid={config.RAG_HYBRID} "
          f"ngram={config.RAG_LEX_NGRAM} min_score={config.RAG_MIN_SCORE} "
          f"chunk={config.CHUNK_CHARS}/{config.CHUNK_OVERLAP}")

    out = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
        "profile": config.PROFILE,
        "n_cases": len(cases),
        "lang": args.lang,
        "books": rag.library.summary(),
        "baseline_config": {k: getattr(config, k) for k in (
            "EMBED_MODEL", "CHUNK_CHARS", "CHUNK_OVERLAP", "RAG_TOP_K",
            "RAG_MIN_SCORE", "RAG_HYBRID", "RAG_LEX_NGRAM", "RAG_RRF_K",
            "RAG_CANDIDATES", "RAG_LEX_MIN", "RAG_INCLUDE_OPENING")},
        "ablations": {},
    }
    abl = out["ablations"]

    ablation_retrieval_mode(rag, cases, abl)
    ablation_topk(rag, cases, abl, (2, 4, 6, 12) if args.quick else (2, 4, 6, 8, 12))
    ablation_opening_chunk(rag, cases, abl)
    ablation_ngram(rag, cases, abl, (3, 4) if args.quick else (2, 3, 4, 5))
    ablation_min_score(rag, cases, abl, (0.0, 0.30, 0.60) if args.quick else (0.0, 0.30, 0.45, 0.60))
    study_indic_repair(abl)
    study_term_recovery(abl, cases)

    out["runtime_s"] = round(time.time() - started, 1)

    # --------------------------------------------------------------- write out
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    jpath = RESULTS_DIR / f"diagnostics-{stamp}.json"
    mpath = RESULTS_DIR / f"diagnostics-{stamp}.md"
    jpath.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# EduBuddy retrieval diagnostics", "",
             f"Generated {out['generated']} - {len(cases)} cases, "
             f"{sum(b['chunks'] for b in out['books'])} indexed chunks, "
             f"runtime {out['runtime_s']}s", "",
             "## Baseline configuration", "",
             "| parameter | value |", "|---|---|"]
    for k, v in out["baseline_config"].items():
        lines.append(f"| `{k}` | {v} |")

    def table(title, res, keycol, note=""):
        lines.extend(["", f"## {title}", ""])
        if note:
            lines.extend([note, ""])
        lines.append(f"| {keycol} | coverage | MRR | median rank | found |")
        lines.append("|---|---|---|---|---|")
        for k, r in res.items():
            lines.append(f"| {k} | {r['coverage']*100:.1f}% | {r['mrr']:.3f} | "
                         f"{r['median_rank']} | {r['found']}/{r['n']} |")

    table("Retrieval mode", abl["retrieval_mode"], "mode",
          "Semantic-only is the ablation: the lexical half of the hybrid is "
          "switched off and only embedding cosine ranks the chunks.")
    table("Context size (RAG_TOP_K)", abl["topk"], "k",
          "Coverage is a ceiling on answer quality; each extra chunk also adds "
          "prompt tokens and therefore latency.")
    table("Opening-chunk injection", abl["opening_chunk"], "state")
    table("Lexical n-gram size", abl["ngram"], "n")
    table("Cosine floor", abl["min_score"], "RAG_MIN_SCORE")

    if "indic_repair" in abl:
        lines.extend(["", "## Indic glyph repair", "",
                      "| book | pages | corrupt clusters before | after |",
                      "|---|---|---|---|"])
        for book, e in abl["indic_repair"].items():
            before = (e["doubled_pulli_before"] + e["doubled_matra_before"]
                      + e["replacement_chars_before"])
            after = (e["doubled_pulli_after"] + e["doubled_matra_after"]
                     + e["replacement_chars_after"])
            lines.append(f"| {book} | {e['pages']} | {before} | {after} |")

    if "term_recovery" in abl:
        tr = abl["term_recovery"]
        lines.extend(["", "## Answer-term recovery", "",
                      f"- terms checked against source text: **{tr['total']}**",
                      f"- present only after repair: **{tr['recovered']}**",
                      f"- absent from the book entirely: **{tr['absent']}**"])

    mpath.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + SEP)
    print(f"runtime {out['runtime_s']}s")
    print(f"wrote {jpath.name}")
    print(f"wrote {mpath.name}")
    print(SEP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
