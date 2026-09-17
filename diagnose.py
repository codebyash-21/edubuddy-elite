"""
Show what retrieval actually hands the model, for every case in testcases.json.

    python diagnose.py                 all languages
    python diagnose.py --lang hi       one language
    python diagnose.py --long          long-answer cases only
    python diagnose.py --show          also print the retrieved text

No language model is loaded, so this runs in seconds. Use it to separate the two
failure modes that look identical from the outside:

    expected terms MISSING from context  -> retrieval problem
    expected terms PRESENT but answer wrong -> generation problem

Diagnosing the wrong layer is the most expensive mistake available here; this is
the cheapest way to avoid it.
"""
from __future__ import annotations

import argparse
import json

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="all", help="all | en | hi | ta")
    ap.add_argument("--long", action="store_true", help="long-answer cases only")
    ap.add_argument("--show", action="store_true", help="print the retrieved text")
    ap.add_argument("--cases", default="testcases.json")
    ap.add_argument("--topk", type=int, default=None,
                    help="override RAG_TOP_K for this run")
    ap.add_argument("--sweep", default=None,
                    help="comma-separated RAG_TOP_K values to compare, e.g. 4,6,8,12")
    args = ap.parse_args()

    if args.topk:
        config.RAG_TOP_K = args.topk

    import rag
    rag.library.load_from_disk(verbose=False)
    if not rag.library.books:
        print("No textbooks indexed. Run: python ingest.py --all")
        return 1
    rag.load_embedder(verbose=False)

    spec = json.loads((config.BASE_DIR / args.cases).read_text(encoding="utf-8"))
    cases = spec["cases"]
    if args.lang != "all":
        cases = [c for c in cases if c["lang"] == args.lang]
    if args.long:
        cases = [c for c in cases if c.get("kind") == "long"]
    if not cases:
        print("No matching cases.")
        return 1


    # --- sweep mode: how much of each answer does retrieval actually deliver? ---
    if args.sweep:
        values = [int(v) for v in args.sweep.split(",")]
        print(f"\nsweeping RAG_TOP_K over {values}\n")
        header = f"{'case':<8}" + "".join(f"k={v:<7}" for v in values)
        print(header); print("-" * len(header))
        totals = {v: 0.0 for v in values}
        n = 0
        for c in cases:
            if c.get("kind") == "negative":
                continue
            want = c.get("any") or []
            if not want:
                continue
            n += 1
            row = f"{c['id']:<8}"
            for v in values:
                config.RAG_TOP_K = v
                ctx, _ = rag.library.context_for(c["question"])
                cov = sum(1 for t in want if t in ctx) / len(want)
                totals[v] += cov
                row += f"{cov*100:>3.0f}%     "
            print(row)
        print("-" * len(header))
        avg = f"{'AVG':<8}" + "".join(f"{totals[v]/n*100:>3.0f}%     " for v in values)
        print(avg)
        print("\nCoverage is the share of expected points present in the retrieved"
              "\ncontext. It caps what any model could possibly answer.")
        return 0

    print(f"indexed books: {[b['book'] for b in rag.library.summary()]}")
    print(f"RAG_TOP_K={config.RAG_TOP_K}  RAG_MIN_SCORE={config.RAG_MIN_SCORE}  "
          f"hybrid={getattr(config,'RAG_HYBRID',False)}  "
          f"opening_chunk={getattr(config,'RAG_INCLUDE_OPENING',False)}")
    print("=" * 72)

    full, partial, none_ = 0, 0, 0
    for c in cases:
        if c.get("kind") == "negative":
            ctx, cites = rag.library.context_for(c["question"])
            print(f"\n{c['id']}  [negative]  {c['question'][:52]}")
            print(f"   retrieved : {cites}")
            print(f"   (expected: the model should refuse this one)")
            continue

        ctx, cites = rag.library.context_for(c["question"])
        want = c.get("any") or []
        present = [t for t in want if t in ctx]
        missing = [t for t in want if t not in ctx]

        if not missing:
            verdict, full = "ALL PRESENT  -> any failure is generation", full + 1
        elif present:
            verdict, partial = "PARTIAL      -> retrieval incomplete", partial + 1
        else:
            verdict, none_ = "NONE FOUND   -> retrieval failed", none_ + 1

        print(f"\n{c['id']}  [{c.get('kind','short')}]  {c['question'][:52]}")
        print(f"   retrieved : {cites}")
        print(f"   present   : {present}")
        print(f"   missing   : {missing}")
        print(f"   {verdict}")
        if args.show:
            print("   --- context ---")
            for line in ctx.splitlines():
                print("   " + line[:110])

    print("\n" + "=" * 72)
    print(f"all terms retrieved : {full}")
    print(f"partially retrieved : {partial}")
    print(f"nothing retrieved   : {none_}")
    print("\nCases with everything present that still answer wrongly are a model "
          "limitation.\nCases with missing terms are a retrieval limitation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
