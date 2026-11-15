"""Saying "not found" as a threshold instead of a training ratio.

    python -m sanad.abstain_curve --adapters before= lora=results/lora lora2=results/lora2 --data data

For every grounded question (answerable: the gold passage is in the top 3; unanswerable: it was removed),
score how likely the model is to start its answer with the "not found" sentence: the mean log-probability
of that sentence's tokens, teacher-forced after the prompt. Then:

  AUC        the chance that a random unanswerable question scores higher than a random answerable one
  threshold  chosen on 150 development questions (LLM-written) for a target false "not found" rate,
             then applied unchanged to test and to Najiz

A model whose scores separate the two cases well can be set to any point on the trade-off after training.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch

from .answer_eval import clip
from .data import build
from .llm import ABSTAIN, load, rag_messages, render
from .retrieval import Dense

TARGETS = (0.02, 0.05, 0.10, 0.20)


@torch.no_grad()
def abstain_score(model, tok, prompts: list[str], batch: int = 8, max_len: int = 3072) -> np.ndarray:
    tail = tok(ABSTAIN, add_special_tokens=False)["input_ids"]
    m = len(tail)
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=max_len,
                  add_special_tokens=False)
        n = enc["input_ids"].shape[0]
        t = torch.tensor([tail] * n)
        ids = torch.cat([enc["input_ids"], t], 1).to(model.device)
        att = torch.cat([enc["attention_mask"], torch.ones_like(t)], 1).to(model.device)
        logits = model(input_ids=ids, attention_mask=att).logits[:, -m - 1:-1, :].float()
        lp = torch.log_softmax(logits, -1).gather(2, ids[:, -m:].unsqueeze(-1)).squeeze(-1)
        out += lp.mean(1).tolist()
    return np.array(out)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of an unanswerable question > score of an answerable one), ties count half."""
    if not len(pos) or not len(neg):
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def prompts_for(qs, ranked, text, k=3):
    ans, un = [], []
    for q in qs:
        r = ranked[q.qid]
        if q.gold in r[:k]:
            ans.append(render_cache(q.text, [clip(text[p]) for p in r[:k]]))
        un.append(render_cache(q.text, [clip(text[p]) for p in [p for p in r if p != q.gold][:k]]))
    return ans, un


_TOK = None


def render_cache(question, passages):
    return render(_TOK, rag_messages(question, passages))


def main():
    global _TOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ALLaM-AI/ALLaM-7B-Instruct-preview")
    ap.add_argument("--adapters", nargs="+", required=True, help="name=path; an empty path means the model as released")
    ap.add_argument("--retriever", default="intfloat/multilingual-e5-base")
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default="results/abstain_curve.json")
    a = ap.parse_args()

    d = build(a.data)
    pids = list(d.passages)
    text = {p: d.passages[p].text for p in pids}
    rng = random.Random(0)                       # the same samples as answer_eval
    dev = rng.sample(d.questions["dev"], min(400, len(d.questions["dev"])))[:a.n]
    test = rng.sample(d.questions["test"], min(400, len(d.questions["test"])))[:a.n]
    sets = {"dev": dev, "test": test, "najiz": d.questions["najiz"]}

    dense = Dense(a.retriever, pids, [text[p] for p in pids])
    allq = [q for qs in sets.values() for q in qs]
    ranked = dict(zip([q.qid for q in allq], dense.search([q.text for q in allq])))
    del dense
    gc.collect()
    torch.cuda.empty_cache()

    report = {}
    for spec in a.adapters:
        name, path = spec.split("=", 1)
        model, tok = load(a.model, path or None)
        _TOK = tok
        s = {}
        for setname, qs in sets.items():
            ans, un = prompts_for(qs, ranked, text)
            s[setname] = (abstain_score(model, tok, ans), abstain_score(model, tok, un))
        r = {"auc": {k: auc(v[1], v[0]) for k, v in s.items()}, "at": []}
        for target in TARGETS:
            thr = float(np.quantile(s["dev"][0], 1 - target))      # dev answerable scored above thr -> "not found"
            row = {"target_false_not_found_on_dev": target, "threshold": thr}
            for setname in ("test", "najiz"):
                pa, pu = s[setname]
                row[setname] = {"false_not_found": float((pa > thr).mean()), "not_found_when_missing": float((pu > thr).mean())}
            r["at"].append(row)
        report[name] = r
        print(name, json.dumps(r), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\nAUC (unanswerable scored above answerable): dev / test / najiz")
    for name, r in report.items():
        print(f"{name:10s} " + " / ".join(f"{r['auc'][k]:.3f}" for k in ("dev", "test", "najiz")))
    print("\nThreshold set on dev for a target false 'not found' rate; then on test and Najiz:"
          " false 'not found' / 'not found' when the law is missing")
    for name, r in report.items():
        for row in r["at"]:
            print(f"{name:10s} dev target {row['target_false_not_found_on_dev']*100:4.0f}%   "
                  f"test {row['test']['false_not_found']*100:5.1f}% / {row['test']['not_found_when_missing']*100:5.1f}%   "
                  f"najiz {row['najiz']['false_not_found']*100:5.1f}% / {row['najiz']['not_found_when_missing']*100:5.1f}%")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
