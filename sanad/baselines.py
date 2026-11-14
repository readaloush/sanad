"""Step 1: how well do off-the-shelf retrievers find the right Saudi legal text?

    python -m sanad.baselines --data data --out results

Scores BM25, three multilingual embedding models, a hybrid of BM25 and the best of them, and a
cross-encoder reranker on top, on three question sets: dev and test (LLM-generated questions from
ArabLegalEval) and najiz (79 expert-verified questions from real FAQs).
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from .data import build
from .retrieval import BM25, Dense, Reranker, metrics, rrf

DENSE = ["intfloat/multilingual-e5-base", "intfloat/multilingual-e5-large", "BAAI/bge-m3"]
SETS = ("dev", "test", "najiz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--dense", nargs="*", default=DENSE)
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--no-alarb", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    d = build(a.data, with_alarb=not a.no_alarb)
    pids = list(d.passages)
    texts = [d.passages[p].text for p in pids]
    text = dict(zip(pids, texts))
    qs = {s: [q.text for q in d.questions[s]] for s in SETS}
    gold = {s: [q.gold for q in d.questions[s]] for s in SETS}

    runs, results, timing = {}, {}, {}

    def record(name, fn):
        t0 = time.time()
        runs[name] = {s: fn(qs[s]) for s in SETS}
        n = sum(len(qs[s]) for s in SETS)
        timing[name] = round((time.time() - t0) / n * 1000, 2)
        results[name] = {s: metrics(runs[name][s], gold[s]) for s in SETS}
        print(f"{name:28s} " + "  ".join(f"{s} R@1 {results[name][s]['R@1']:.3f} R@10 {results[name][s]['R@10']:.3f}" for s in SETS)
              + f"  {timing[name]} ms/q", flush=True)

    bm = BM25(pids, texts)
    record("BM25", bm.search)
    del bm

    best, best_dev = None, -1.0
    for m in a.dense:
        t0 = time.time()
        try:
            r = Dense(m, pids, texts)
        except Exception as e:
            print(f"{m}: could not load ({type(e).__name__}: {str(e)[:200]})")
            continue
        print(f"{m}: corpus encoded in {time.time() - t0:.0f}s", flush=True)
        record(r.name, r.search)
        if results[r.name]["dev"]["MRR@10"] > best_dev:
            best, best_dev = r.name, results[r.name]["dev"]["MRR@10"]
        del r
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    if best:
        hybrid = f"BM25 + {best} (RRF)"
        runs[hybrid] = {s: rrf(runs["BM25"][s], runs[best][s]) for s in SETS}
        results[hybrid] = {s: metrics(runs[hybrid][s], gold[s]) for s in SETS}
        timing[hybrid] = round(timing["BM25"] + timing[best], 2)
        print(f"{hybrid:28s} " + "  ".join(f"{s} R@1 {results[hybrid][s]['R@1']:.3f}" for s in SETS), flush=True)
        try:
            rr = Reranker(a.reranker)
            name = f"{hybrid} + {rr.name}"
            t0 = time.time()
            runs[name] = {s: rr.rerank(qs[s], runs[hybrid][s], text) for s in SETS}
            n = sum(len(qs[s]) for s in SETS)
            timing[name] = round(timing[hybrid] + (time.time() - t0) / n * 1000, 2)
            results[name] = {s: metrics(runs[name][s], gold[s]) for s in SETS}
            print(f"{name:28s} " + "  ".join(f"{s} R@1 {results[name][s]['R@1']:.3f}" for s in SETS), flush=True)
        except Exception as e:
            print(f"reranker failed: {type(e).__name__}: {str(e)[:200]}")

    report = {"data": d.stats, "results": results, "ms_per_query_gpu": timing, "best_dense_on_dev": best}
    (out / "baselines.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))

    print("\nRecall@1 / @5 / @10 / MRR@10")
    for name, r in results.items():
        print(f"{name:52s} " + "   ".join(f"{s} {r[s]['R@1']*100:4.1f}/{r[s]['R@5']*100:4.1f}/{r[s]['R@10']*100:4.1f}/{r[s]['MRR@10']:.3f}" for s in SETS))
    print("questions:", {s: len(qs[s]) for s in SETS}, "| passages:", len(pids))

    # a few Najiz questions the best system still misses, to read by eye
    last = list(results)[-1]
    miss = [(q, g) for q, g, r in zip(d.questions["najiz"], gold["najiz"], runs[last]["najiz"]) if r[:1] != [g]][:5]
    for q, g in miss:
        top = runs[last]["najiz"][d.questions["najiz"].index(q)][0]
        print(f"\nQ: {q.text}\n gold: {text[g][:160]}\n top1: {text[top][:160]}")


if __name__ == "__main__":
    main()
