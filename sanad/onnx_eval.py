"""Does the served retriever (ONNX, int8 weights, 256 tokens) retrieve as well as the PyTorch model it came from?

    python -m sanad.onnx_eval --artifacts artifacts --data data

Runs on CPU. Every encoder embeds the whole 5,929-passage corpus and the test and Najiz questions.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .data import build
from .retrieval import metrics
from .serve import PASSAGE_PREFIX, QUERY_PREFIX, TextEncoder

SETS = ("test", "najiz")


def rank(q: np.ndarray, p: np.ndarray, pids: list[str], k: int = 20) -> list[list[str]]:
    s = q @ p.T
    top = np.argsort(-s, axis=1)[:, :k]
    return [[pids[j] for j in row] for row in top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True, help="folder with text.int8.onnx and tokenizer.json (and text.onnx if kept)")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/onnx_eval.json")
    ap.add_argument("--lengths", default="512,256", help="token limits for the PyTorch reference runs")
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    d = build(a.data)
    pids = list(d.passages)
    ptexts = [PASSAGE_PREFIX + d.passages[i].text for i in pids]
    qs = {s: [QUERY_PREFIX + q.text for q in d.questions[s]] for s in SETS}
    gold = {s: [q.gold for q in d.questions[s]] for s in SETS}

    st = SentenceTransformer(a.model, device="cpu")
    encoders = {}
    for n in [int(x) for x in a.lengths.split(",") if x]:
        def f(texts, n=n):
            st.max_seq_length = n
            return st.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
        encoders[f"PyTorch float32, {n} tokens"] = f
    art = Path(a.artifacts)
    for name, file in (("ONNX float32, 256 tokens", "text.onnx"), ("ONNX int8, 256 tokens", "text.int8.onnx")):
        if (art / file).exists():
            encoders[name] = TextEncoder(art / file, art / "tokenizer.json")

    results, ref = {}, None
    for name, enc in encoders.items():
        t = time.time()
        p = enc(ptexts)
        ms = (time.time() - t) / len(ptexts) * 1000
        r = {s: metrics(rank(enc(qs[s]), p, pids), gold[s]) for s in SETS}
        r["ms_per_passage_cpu"] = round(ms, 1)
        if ref is None:
            ref = p
        else:
            cos = (ref * p).sum(1)
            r["cosine_to_first"] = {"min": float(cos.min()), "mean": float(cos.mean())}
        results[name] = r
        print(name, json.dumps(r), flush=True)

    print("\nRecall@1 / @5 / @10 / MRR@10")
    for name, r in results.items():
        cells = "   ".join(f"{s} {r[s]['R@1']*100:.1f}/{r[s]['R@5']*100:.1f}/{r[s]['R@10']*100:.1f}/{r[s]['MRR@10']:.3f}" for s in SETS)
        print(f"{name:28s} {cells}   {r['ms_per_passage_cpu']} ms/passage on CPU")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
