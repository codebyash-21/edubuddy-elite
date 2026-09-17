"""
EduBuddy validation harness.

Runs a fixed set of questions through the real pipeline and records, per question:
retrieval correctness, answer correctness, grounding behaviour and a latency
breakdown. Writes a timestamped CSV plus a Markdown summary, and snapshots every
configuration parameter so a run is reproducible and citable.

    python evaluate.py                 # all languages
    python evaluate.py --lang ta       # one language
    python evaluate.py --no-tts        # skip speech synthesis (much faster)
    python evaluate.py --cases my.json # a different question set

Deliberately bypasses the web server and drives rag.py and llm.py directly, so a
failure is attributable to a component rather than to the HTTP layer.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import config

RESULTS_DIR = config.BASE_DIR / "results"


# ----------------------------------------------------------------- environment
def snapshot():
    """Every parameter that could change a result."""
    models = {}
    for lang, (repo, fname, no_think) in config.LLM_MODELS.items():
        models[lang] = fname
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": __import__("os").cpu_count(),
        "llm_per_language": models,
        "llm_max_tokens": config.LLM_MAX_TOKENS,
        "llm_temperature": config.LLM_TEMPERATURE,
        "llm_ctx": config.LLM_CTX,
        "llm_threads": config.LLM_THREADS,
        "llm_batch": getattr(config, "LLM_BATCH", None),
        "llm_history_turns": config.LLM_HISTORY_TURNS,
        "whisper_model": config.WHISPER_MODEL,
        "embed_model": config.EMBED_MODEL,
        "chunk_chars": config.CHUNK_CHARS,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "rag_top_k": config.RAG_TOP_K,
        "rag_min_score": config.RAG_MIN_SCORE,
        "rag_hybrid": getattr(config, "RAG_HYBRID", False),
        "rag_lex_ngram": getattr(config, "RAG_LEX_NGRAM", None),
        "rag_lex_min": getattr(config, "RAG_LEX_MIN", None),
        "rag_rrf_k": getattr(config, "RAG_RRF_K", None),
    }


# ----------------------------------------------------------------- scoring
def score_case(case, reply, citations, grounded):
    """Return (verdict, coverage, note)."""
    kind = case.get("kind", "short")
    wanted = case.get("any") or []

    if kind == "negative":
        # Success is refusing: the system must report the book does not cover it.
        note = config.NOT_IN_BOOK_MESSAGE[config.normalise_lang(case["lang"])]
        refused = (not grounded) or note[:12] in reply
        if refused and not citations:
            return "PASS", 1.0, "correctly refused"
        if refused:
            return "PARTIAL", 0.5, "refused but still showed citations"
        return "FAIL", 0.0, "answered a question the book does not cover"

    hits = [w for w in wanted if w in reply]
    coverage = len(hits) / len(wanted) if wanted else 0.0

    if kind == "long":
        if coverage >= 0.75:
            return "PASS", coverage, f"{len(hits)}/{len(wanted)} points"
        if coverage > 0:
            return "PARTIAL", coverage, f"{len(hits)}/{len(wanted)} points"
        return "FAIL", 0.0, "no expected points"

    if hits:
        return "PASS", 1.0, f"found {hits[0]!r}"
    return "FAIL", 0.0, "expected term absent"


def retrieval_ok(case, citations):
    """Did retrieval surface the expected book?"""
    book = case.get("book")
    if not book:
        return ""
    return "yes" if any(c.startswith(book) for c in citations) else "NO"


# ----------------------------------------------------------------- run
def run(cases, use_tts=True):
    import llm
    import rag

    rag.library.load_from_disk(verbose=False)
    rag.load_embedder(verbose=False)
    if not rag.library.books:
        print("No textbooks indexed. Run: python ingest.py --all")
        sys.exit(1)

    if use_tts:
        import tts

    rows = []
    for i, case in enumerate(cases, 1):
        lang = config.normalise_lang(case["lang"])
        q = case["question"]
        print(f"\n[{i}/{len(cases)}] {case['id']}  ({lang})  {q[:60]}")

        llm.tutor.clear()          # no cross-question contamination

        t0 = time.time()
        context, citations = rag.library.context_for(q)
        t_ret = time.time() - t0
        # Retrieval must be scored on what the search actually returned. The
        # grounding step below clears citations when the model declares the book
        # does not cover the question, so scoring them afterwards would record a
        # retrieval miss whenever the *model* wrongly refused — conflating two
        # independent failures and understating retrieval.
        retrieved = list(citations)

        t0 = time.time()
        reply = llm.tutor.ask(q, lang=lang, context=context)
        t_gen = time.time() - t0

        grounded = True
        if config.NOT_IN_BOOK_MARKER in reply:
            reply = reply.replace(config.NOT_IN_BOOK_MARKER, "").strip()
            reply = f"{config.NOT_IN_BOOK_MESSAGE[lang]} {reply}".strip()
            citations, grounded = [], False

        t_tts = 0.0
        audio_s = 0.0
        if use_tts:
            t0 = time.time()
            try:
                wav, sr = tts.synth(reply, lang)
                audio_s = round(len(wav) / sr, 1) if sr else 0.0
            except Exception as exc:
                print(f"    TTS failed: {exc}")
            t_tts = time.time() - t0

        verdict, coverage, note = score_case(case, reply, citations, grounded)
        print(f"    {verdict:<7} {note}   [ret {t_ret:.1f}s | gen {t_gen:.1f}s | "
              f"tts {t_tts:.1f}s]")
        print(f"    reply: {reply[:110]}")

        rows.append({
            "id": case["id"], "lang": lang, "kind": case.get("kind", "short"),
            "question": q, "verdict": verdict,
            "coverage": round(coverage, 2), "note": note,
            "retrieval_ok": retrieval_ok(case, retrieved),
            "grounded": grounded,
            "citations_retrieved": " | ".join(retrieved),
            "citations_shown": " | ".join(citations),
            "retrieval_s": round(t_ret, 2), "generation_s": round(t_gen, 2),
            "tts_s": round(t_tts, 2),
            "total_s": round(t_ret + t_gen + t_tts, 2),
            "audio_s": audio_s,
            "reply_chars": len(reply), "reply": reply,
        })
    return rows


# ----------------------------------------------------------------- reporting
def summarise(rows):
    langs = sorted({r["lang"] for r in rows})
    out = []
    for lang in langs + ["ALL"]:
        sub = [r for r in rows if lang == "ALL" or r["lang"] == lang]
        if not sub:
            continue
        p = sum(r["verdict"] == "PASS" for r in sub)
        pa = sum(r["verdict"] == "PARTIAL" for r in sub)
        f = sum(r["verdict"] == "FAIL" for r in sub)
        scored = [r for r in sub if r["retrieval_ok"] in ("yes", "NO")]
        ret = sum(r["retrieval_ok"] == "yes" for r in scored)
        gen = [r["generation_s"] for r in sub]
        out.append({
            "scope": lang, "n": len(sub), "pass": p, "partial": pa, "fail": f,
            "pass_rate": f"{100*p/len(sub):.0f}%",
            "retrieval_rate": f"{100*ret/len(scored):.0f}%" if scored else "-",
            "median_gen_s": f"{sorted(gen)[len(gen)//2]:.1f}",
            "max_gen_s": f"{max(gen):.1f}",
        })
    return out


def write_reports(rows, env):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = RESULTS_DIR / f"eval_{stamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = summarise(rows)
    md_path = RESULTS_DIR / f"eval_{stamp}.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# EduBuddy validation run\n\n**{env['timestamp']}**\n\n")
        fh.write("## Result summary\n\n")
        fh.write("| Scope | Cases | Pass | Partial | Fail | Pass rate | "
                 "Retrieval hit | Median gen | Max gen |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|\n")
        for s in summary:
            fh.write(f"| {s['scope']} | {s['n']} | {s['pass']} | {s['partial']} | "
                     f"{s['fail']} | {s['pass_rate']} | {s['retrieval_rate']} | "
                     f"{s['median_gen_s']} s | {s['max_gen_s']} s |\n")
        fh.write("\n## Per-question results\n\n")
        fh.write("| ID | Lang | Verdict | Retrieval | Grounded | Gen (s) | Note |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['id']} | {r['lang']} | {r['verdict']} | "
                     f"{r['retrieval_ok'] or '-'} | {r['grounded']} | "
                     f"{r['generation_s']} | {r['note']} |\n")
        fh.write("\n## Configuration under test\n\n| Parameter | Value |\n|---|---|\n")
        for k, v in env.items():
            fh.write(f"| {k} | {v} |\n")

    json_path = RESULTS_DIR / f"eval_{stamp}.json"
    json_path.write_text(json.dumps({"environment": env, "summary": summary,
                                     "results": rows}, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    return csv_path, md_path, json_path, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="all", help="all | en | hi | ta")
    ap.add_argument("--cases", default="testcases.json")
    ap.add_argument("--no-tts", action="store_true",
                    help="skip speech synthesis (much faster)")
    args = ap.parse_args()

    spec = json.loads(Path(config.BASE_DIR / args.cases).read_text(encoding="utf-8"))
    cases = spec["cases"]
    if args.lang != "all":
        cases = [c for c in cases if c["lang"] == args.lang]
    if not cases:
        print(f"No cases for lang={args.lang}")
        return 1

    env = snapshot()
    print("=" * 66)
    print(" EduBuddy validation run")
    print("=" * 66)
    for k in ("llm_per_language", "llm_max_tokens", "llm_temperature",
              "chunk_chars", "rag_top_k", "rag_hybrid"):
        print(f"  {k:<20} {env[k]}")
    print(f"  cases                {len(cases)}   tts={'off' if args.no_tts else 'on'}")

    t0 = time.time()
    rows = run(cases, use_tts=not args.no_tts)
    csv_p, md_p, json_p, summary = write_reports(rows, env)

    print("\n" + "=" * 66)
    print(f"{'scope':<6}{'n':>4}{'pass':>6}{'part':>6}{'fail':>6}"
          f"{'rate':>7}{'retr':>7}{'medgen':>9}")
    for s in summary:
        print(f"{s['scope']:<6}{s['n']:>4}{s['pass']:>6}{s['partial']:>6}"
              f"{s['fail']:>6}{s['pass_rate']:>7}{s['retrieval_rate']:>7}"
              f"{s['median_gen_s']+'s':>9}")
    print("=" * 66)
    print(f"total wall clock: {time.time()-t0:.0f}s")
    print(f"\nwritten:\n  {csv_p}\n  {md_p}\n  {json_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
