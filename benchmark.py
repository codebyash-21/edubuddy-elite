"""Time each stage of an answer separately, so tuning is measured not guessed.

    python benchmark.py                      one question per language
    python benchmark.py --lang en -n 3       three English questions
    python benchmark.py --sweep 96,128,160   compare LLM_MAX_TOKENS values
    python benchmark.py --no-tts             skip speech synthesis

An end-to-end number tells you the total but not where it went, and the stages
have completely different fixes: retrieval is bounded by the index, prefill by
how much context you feed, decode by tokens generated, TTS by answer length. A
change that halves decode does nothing for prefill, so a single figure hides
whether a tuning change worked.

The sweep is the useful mode. `LLM_MAX_TOKENS` is close to a linear latency dial
and its right value depends on how long your answers actually need to be — which
is a question about your books, not about the model.

Nothing here writes to results/; this is a stopwatch, not a validation run.
Use evaluate.py when you care about whether answers are *correct*.
"""
from __future__ import annotations

import argparse
import statistics
import time

import config

# evaluate.py owns this path; config.py does not define it.
RESULTS_DIR = config.BASE_DIR / "results"

QUESTIONS = {
    "en": [
        "Which is the brightest star in the night sky?",
        "What is light pollution?",
        "How are the inner planets different from the outer planets?",
    ],
    "hi": [
        "यह कहानी किस नगर के बाहर घटित होती है?",
        "टीले से निकला पत्थर असल में क्या था?",
        "लड़कों का खेल क्या था और वह कैसे खेला जाता था?",
    ],
    "ta": [
        "கை கழுவ எவ்வளவு நேரம் எடுக்க வேண்டும்?",
        "நாம் எந்த உறுப்பு மூலம் சுவைக்கிறோம்?",
        "விசையின் விளைவுகள் யாவை?",
    ],
}


def _fmt(seconds: float) -> str:
    return f"{seconds:6.2f}s"


def time_one(question: str, lang: str, do_tts: bool, rag, llm, tts):
    """Return per-stage timings for a single question."""
    out = {"lang": lang, "question": question}

    t = time.time()
    context, citations = rag.library.context_for(question)
    out["retrieval"] = time.time() - t
    out["citations"] = len(citations)
    out["context_chars"] = len(context)

    t = time.time()
    reply = llm.tutor.ask(question, lang=lang, context=context)
    out["generate"] = time.time() - t
    out["reply_chars"] = len(reply)
    # Tokens are not exposed by the wrapper, so approximate from characters.
    # Only ever compared against itself across runs, which is what matters here.
    out["approx_tokens"] = max(1, len(reply) // 4)
    out["tok_per_s"] = out["approx_tokens"] / out["generate"] if out["generate"] else 0

    out["tts"] = 0.0
    if do_tts:
        t = time.time()
        try:
            tts.synth(reply, lang=lang)
        except Exception as exc:
            print(f"    (tts failed: {exc})")
        out["tts"] = time.time() - t

    out["total"] = out["retrieval"] + out["generate"] + out["tts"]
    out["reply"] = reply
    return out


def run(langs, n, do_tts, show_replies, rag, llm, tts):
    rows = []
    for lang in langs:
        qs = QUESTIONS.get(lang, [])[:n]
        if not qs:
            continue
        print(f"\n{config.LANG_NAMES.get(lang, lang)}")
        print(f"  {'retrieval':>10} {'generate':>10} {'tts':>8} {'total':>9} "
              f"{'tok/s':>7}  question")
        for q in qs:
            r = time_one(q, lang, do_tts, rag, llm, tts)
            rows.append(r)
            print(f"  {_fmt(r['retrieval']):>10} {_fmt(r['generate']):>10} "
                  f"{_fmt(r['tts']):>8} {_fmt(r['total']):>9} "
                  f"{r['tok_per_s']:7.1f}  {q[:42]}")
            if show_replies:
                print(f"             -> {r['reply'][:140]}")
    return rows


def summarise(rows, label=""):
    if not rows:
        return None
    med = lambda k: statistics.median([r[k] for r in rows])
    s = {
        "n": len(rows),
        "retrieval": med("retrieval"),
        "generate": med("generate"),
        "tts": med("tts"),
        "total": med("total"),
        "tok_per_s": med("tok_per_s"),
        "reply_chars": med("reply_chars"),
    }
    print(f"\n  median{(' ' + label) if label else ''} — "
          f"retrieval {s['retrieval']:.2f}s · generate {s['generate']:.1f}s · "
          f"tts {s['tts']:.1f}s · total {s['total']:.1f}s · "
          f"{s['tok_per_s']:.1f} tok/s · {s['reply_chars']:.0f} chars")
    return s


# --------------------------------------------------- deploy vs dynamic (packs)
def _correct(answer: str, case: dict) -> float:
    """Share of a case's expected terms present in the answer.

    Speed without this is meaningless. A question pack that answers instantly
    with the wrong passage is worse than the slow path, not better, and a table
    of latencies alone would report it as an improvement.
    """
    want = case.get("any") or []
    if not want:
        return float("nan")          # negative controls are scored elsewhere
    return sum(1 for t in want if t in answer) / len(want)


def compare_pack(cases, rag, llm, qa_match):
    """Same questions, packs off then on. One variable, both metrics."""
    import json
    from datetime import datetime

    pack_size = len((qa_match.load() or {}).get("entries") or [])
    if pack_size == 0:
        print("\nNo question pack found, so there is nothing to compare —")
        print("'dynamic' and 'deploy' are byte-identical without one.")
        print("Build it first:  python qa_pack.py --build")
        return 1

    print(f"\npack: {pack_size} entries · threshold {config.QA_MATCH_MIN}")
    print(f"cases: {len(cases)}\n")
    print(f"  {'case':<8} {'hit':>4} {'deploy':>9} {'dynamic':>9} {'speedup':>8} "
          f"{'dep ok':>7} {'dyn ok':>7}")
    print("  " + "-" * 62)

    rows = []
    original = config.QA_ENABLED
    for c in cases:
        q, lang = c["question"], c["lang"]

        # The model path is measured ONCE per question, never twice.
        #
        # An earlier version ran dynamic and then deploy on the same question.
        # That is invalid: llama.cpp caches the prompt prefix, so whichever path
        # ran second skipped most of its prefill and looked several times faster
        # than it is. It made deploy appear to beat dynamic on pack misses —
        # where the two run identical code and must be identical by definition.
        #
        # So: measure the model once, and on a miss report that time for both,
        # because on a miss dynamic *is* deploy. Only a pack hit makes them
        # differ, which is exactly the thing being measured.
        llm.tutor.clear()                    # no history bleeding between cases

        t0 = time.time()
        ctx, _ = rag.library.context_for(q)
        model_answer = llm.tutor.ask(q, lang=lang, context=ctx)
        model_t = time.time() - t0

        config.QA_ENABLED = True
        t0 = time.time()
        hit = qa_match.match(q, lang=lang)
        pack_t = time.time() - t0

        if hit:
            dyn_answer, dyn_t, dyn_src = hit["answer"], pack_t, "pack"
        else:
            dyn_answer, dyn_t, dyn_src = model_answer, model_t, "model"

        dep_answer, dep_t = model_answer, model_t
        dep_ok, dyn_ok = _correct(dep_answer, c), _correct(dyn_answer, c)
        speed = dep_t / dyn_t if dyn_t > 0 else float("inf")
        rows.append({"id": c["id"], "lang": lang, "kind": c.get("kind"),
                     "source": dyn_src, "deploy_s": dep_t, "dynamic_s": dyn_t,
                     "speedup": speed, "deploy_correct": dep_ok,
                     "dynamic_correct": dyn_ok})

        fmt = lambda v: "  -  " if v != v else f"{v*100:4.0f}%"
        print(f"  {c['id']:<8} {'pack' if dyn_src=='pack' else '  - ':>4} "
              f"{dep_t:>8.1f}s {dyn_t:>8.1f}s "
              f"{speed:>7.1f}x {fmt(dep_ok):>7} {fmt(dyn_ok):>7}")
    config.QA_ENABLED = original

    # ------------------------------------------------------------- summary
    hits = [r for r in rows if r["source"] == "pack"]
    miss = [r for r in rows if r["source"] != "pack"]
    scored = [r for r in rows if r["deploy_correct"] == r["deploy_correct"]]

    print("\n" + "=" * 72)
    print(f"pack coverage      : {len(hits)}/{len(rows)} questions "
          f"({len(hits)/len(rows)*100:.0f}%)")
    if hits:
        print(f"  answered by pack : median {statistics.median([r['dynamic_s'] for r in hits]):.2f}s "
              f"vs {statistics.median([r['deploy_s'] for r in hits]):.1f}s on deploy")
    if miss:
        print(f"  fell through     : median {statistics.median([r['dynamic_s'] for r in miss]):.1f}s "
              f"(identical to deploy by construction — same code path)")
    dep_all = [r["deploy_s"] for r in rows]
    dyn_all = [r["dynamic_s"] for r in rows]
    dep_mean, dyn_mean = statistics.fmean(dep_all), statistics.fmean(dyn_all)
    print(f"\nmean overall       : deploy {dep_mean:.1f}s  ->  dynamic {dyn_mean:.1f}s"
          f"   ({(1 - dyn_mean/dep_mean)*100:+.0f}%)")
    print(f"median overall     : deploy {statistics.median(dep_all):.1f}s"
          f"  ->  dynamic {statistics.median(dyn_all):.1f}s")
    if len(hits) < len(rows) / 2:
        # With coverage under half the median lands on a miss and reports no
        # change, however large the win on the covered questions. The
        # distribution is bimodal — near-zero or a full generation — so the mean
        # is the honest headline here and the median is the misleading one.
        print("  NOTE: coverage is under 50%, so the median sits on a pack miss"
              "\n  and hides the effect entirely. Quote the mean, with coverage"
              "\n  beside it — neither number means much alone.")

    if scored:
        dep_c = statistics.fmean([r["deploy_correct"] for r in scored])
        dyn_c = statistics.fmean([r["dynamic_correct"] for r in scored])
        print(f"answer correctness : deploy {dep_c*100:.0f}%  ->  dynamic {dyn_c*100:.0f}%")
        regressed = [r for r in scored
                     if r["source"] == "pack" and r["dynamic_correct"] < r["deploy_correct"]]
        if regressed:
            print(f"\n  ** {len(regressed)} pack answer(s) SCORED WORSE than the model **")
            for r in regressed:
                print(f"     {r['id']}  {r['deploy_correct']*100:.0f}% -> "
                      f"{r['dynamic_correct']*100:.0f}%")
            print("  A faster wrong answer is not an improvement. Raise "
                  "QA_MATCH_MIN\n  or remove those entries before quoting the "
                  "speed figure.")
        else:
            print("  no pack answer scored worse than the model path")

    _write_records(rows, pack_size, hits, scored)
    return 0


def _write_records(rows, pack_size, hits, scored):
    """Leave a readable artefact, not just machine output.

    CSV for a spreadsheet, Markdown for a report, JSON for a later comparison —
    the same three formats evaluate.py produces, so every experiment in this
    project is recorded the same way. The caveats are written into the Markdown
    on purpose: a table that travels without them invites the numbers being
    quoted alone.
    """
    import csv
    import json
    import platform
    from datetime import datetime

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    now = datetime.now().isoformat(timespec="seconds")

    snapshot = {
        "generated": now,
        "platform": f"{platform.system()} {platform.machine()}",
        "profile": config.PROFILE,
        "model": config.LLM_FILE,
        "llm_threads": config.LLM_THREADS,
        "llm_max_tokens": config.LLM_MAX_TOKENS,
        "llm_ctx": config.LLM_CTX,
        "rag_top_k": config.RAG_TOP_K,
        "pack_entries": pack_size,
        "qa_match_min": config.QA_MATCH_MIN,
        "qa_ngram": getattr(config, "QA_NGRAM", 3),
        "qa_min_lang_entries": getattr(config, "QA_MIN_LANG_ENTRIES", 12),
    }

    # ---------------------------------------------------------------- json
    (RESULTS_DIR / f"compare-pack-{stamp}.json").write_text(
        json.dumps({**snapshot, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    # ----------------------------------------------------------------- csv
    with open(RESULTS_DIR / f"compare-pack-{stamp}.csv", "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "lang", "kind", "answered_by",
                    "deploy_seconds", "dynamic_seconds", "speedup",
                    "deploy_correct", "dynamic_correct"])
        for r in rows:
            fine = lambda v: "" if v != v else f"{v:.3f}"
            w.writerow([r["id"], r["lang"], r["kind"], r["source"],
                        f"{r['deploy_s']:.2f}", f"{r['dynamic_s']:.3f}",
                        f"{r['speedup']:.1f}", fine(r["deploy_correct"]),
                        fine(r["dynamic_correct"])])

    # ------------------------------------------------------------ markdown
    dep_all = [r["deploy_s"] for r in rows]
    dyn_all = [r["dynamic_s"] for r in rows]
    dep_mean, dyn_mean = statistics.fmean(dep_all), statistics.fmean(dyn_all)
    langs = sorted({r["lang"] for r in rows})

    L = [f"# EduBuddy — question packs: deploy vs dynamic", "",
         f"Generated {now}. Same questions, same machine, same model; the only",
         "variable is whether the question pack is consulted before the model.",
         "",
         "## Configuration", "", "| parameter | value |", "|---|---|"]
    L += [f"| `{k}` | {v} |" for k, v in snapshot.items()]

    L += ["", "## Per-question results", "",
          "| case | lang | answered by | deploy | dynamic | speedup | deploy correct | dynamic correct |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        pc = lambda v: "&mdash;" if v != v else f"{v*100:.0f}%"
        L.append(f"| {r['id']} | {r['lang']} | "
                 f"{'**pack**' if r['source']=='pack' else 'model'} | "
                 f"{r['deploy_s']:.1f}s | {r['dynamic_s']:.2f}s | "
                 f"{r['speedup']:.1f}x | {pc(r['deploy_correct'])} | "
                 f"{pc(r['dynamic_correct'])} |")

    L += ["", "## Summary", "", "| | deploy | dynamic |", "|---|---|---|",
          f"| mean answer time | {dep_mean:.1f}s | **{dyn_mean:.1f}s** "
          f"({(1-dyn_mean/dep_mean)*100:+.0f}%) |",
          f"| median answer time | {statistics.median(dep_all):.1f}s | "
          f"{statistics.median(dyn_all):.1f}s |"]
    if scored:
        L.append(f"| answer correctness | "
                 f"{statistics.fmean([r['deploy_correct'] for r in scored])*100:.0f}% | "
                 f"{statistics.fmean([r['dynamic_correct'] for r in scored])*100:.0f}% |")
    L.append(f"| pack coverage | &mdash; | **{len(hits)}/{len(rows)} "
             f"({len(hits)/len(rows)*100:.0f}%)** |")

    L += ["", "## How to read this", "",
          "- **Coverage is the limiting number, not latency.** A pack hit costs",
          "  almost nothing; the benefit is entirely how often one occurs. Quote",
          "  coverage and mean together — neither means much alone.",
          "- **Misses are identical by construction.** On a miss, dynamic runs the",
          "  same code as deploy, so those rows read exactly `1.0x`. The model is",
          "  timed once per question; timing it twice would hit llama.cpp's prompt",
          "  cache and make the second run look artificially fast.",
          "- **Median understates the effect below 50% coverage**, because it lands",
          "  on a miss. The distribution is bimodal — near-zero or a full",
          "  generation — so the mean is the honest headline here.",
          "- **Correctness is reported beside speed on purpose.** A pack that",
          "  answers instantly with the wrong passage would look like a large",
          "  speedup in a latency-only table. If dynamic correctness is below",
          "  deploy, the speed figure should not be quoted at all.",
          "",
          f"Languages covered by this run: **{', '.join(langs)}**"
          + ("" if len(langs) > 1 else
             " &mdash; single-language run, so this does not"
             " support any claim about the others."),
          f"Cases: **{len(rows)}**."
          + (" Small sample; treat as indicative rather than validated."
             if len(rows) < 15 else ""),
          ]

    (RESULTS_DIR / f"compare-pack-{stamp}.md").write_text(
        "\n".join(L) + "\n", encoding="utf-8")

    print(f"\nwrote results/compare-pack-{stamp}.csv")
    print(f"      results/compare-pack-{stamp}.md    <- the one to show people")
    print(f"      results/compare-pack-{stamp}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="all", help="all | en | hi | ta")
    ap.add_argument("-n", type=int, default=1, help="questions per language")
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--show", action="store_true", help="print the replies too")
    ap.add_argument("--sweep", default=None,
                    help="comma-separated LLM_MAX_TOKENS values to compare")
    ap.add_argument("--compare-pack", action="store_true",
                    help="deploy (packs off) vs dynamic (packs on), same questions")
    ap.add_argument("--cases", default="testcases.json")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N cases — the full set takes a while")
    args = ap.parse_args()

    langs = config.LANGUAGES if args.lang == "all" else [args.lang]
    do_tts = not args.no_tts

    import rag
    import llm
    import tts as tts_mod

    print("=" * 72)
    print("EduBuddy benchmark")
    print("=" * 72)
    print(f"profile   : {config.PROFILE}")
    print(f"model     : {config.LLM_FILE}")
    print(f"threads   : {config.LLM_THREADS}   ctx {config.LLM_CTX}   "
          f"batch {config.LLM_BATCH}")
    print(f"max_tokens: {config.LLM_MAX_TOKENS}   top_k {config.RAG_TOP_K}")

    rag.library.load_from_disk(verbose=False)
    if not rag.library.books:
        print("\nNo textbooks indexed. Run: python ingest.py --all")
        return 1

    # Cold start is a one-off that would swamp every other number. Pay it here,
    # deliberately, and report it — then measure everything else warm.
    print("\nwarming up (loading models — this is the cold-start cost) ...")
    t = time.time()
    rag.load_embedder(verbose=False)
    llm.tutor.ask("hello", lang="en", context="")
    warm = time.time() - t
    print(f"  cold start: {warm:.1f}s  (paid once per server start)")
    print("  NOTE: start the server and ask one question before a demo.")

    if args.compare_pack:
        import json as _json
        import qa_match
        spec = _json.loads((config.BASE_DIR / args.cases).read_text(encoding="utf-8"))
        cases = [c for c in spec["cases"] if c.get("kind") != "negative"]
        if args.lang != "all":
            cases = [c for c in cases if c["lang"] == args.lang]
        if args.limit:
            cases = cases[:args.limit]
        if not cases:
            print("No matching cases.")
            return 1
        print("\n" + "=" * 72)
        print("deploy (packs off)  vs  dynamic (packs on)")
        print("=" * 72)
        print("Same questions, same machine, same model — only the pack changes.")
        print("The model runs once per case (measuring it twice would hit")
        print("llama.cpp's prompt cache and make the second run look faster).")
        print(f"Budget roughly {max(1, len(cases)*30//60)} minutes for {len(cases)} cases.")
        print("Use --limit 6 for a quick look first.")
        return compare_pack(cases, rag, llm, qa_match)

    if args.sweep:
        values = [int(v) for v in args.sweep.split(",")]
        original = config.LLM_MAX_TOKENS
        results = {}
        for v in values:
            config.LLM_MAX_TOKENS = v
            print("\n" + "-" * 72)
            print(f"LLM_MAX_TOKENS = {v}")
            print("-" * 72)
            rows = run(langs, args.n, do_tts, args.show, rag, llm, tts_mod)
            results[v] = summarise(rows, f"@{v}")
        config.LLM_MAX_TOKENS = original

        print("\n" + "=" * 72)
        print(f"{'max_tokens':>11} {'total':>9} {'generate':>10} {'reply len':>11}")
        print("-" * 72)
        base = results[values[0]]
        for v in values:
            s = results[v]
            if not s:
                continue
            delta = ((s["total"] - base["total"]) / base["total"] * 100
                     if base and base["total"] else 0)
            print(f"{v:>11} {s['total']:>8.1f}s {s['generate']:>9.1f}s "
                  f"{s['reply_chars']:>10.0f}c   {delta:+.0f}%")
        print("\nShorter answers are faster almost linearly. Pick the smallest "
              "value\nthat still answers your long-answer questions completely "
              "— check the\nreply length column, not just the clock.")
        return 0

    rows = run(langs, args.n, do_tts, args.show, rag, llm, tts_mod)
    print("\n" + "=" * 72)
    s = summarise(rows)
    if s:
        share = lambda k: s[k] / s["total"] * 100 if s["total"] else 0
        print(f"\n  where the time goes: retrieval {share('retrieval'):.0f}% · "
              f"generate {share('generate'):.0f}% · tts {share('tts'):.0f}%")
        print("\n  Generation dominating is expected on CPU. The levers, in order:"
              "\n    1. a question-pack hit skips the model entirely  (qa_pack.py)"
              "\n    2. lower LLM_MAX_TOKENS   — run --sweep to size it"
              "\n    3. lower RAG_TOP_K        — less prompt to read first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
