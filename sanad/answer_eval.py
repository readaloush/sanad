"""Step 3a: how well does a language model answer, with and without the retrieved law?

    python -m sanad.answer_eval --models ALLaM-AI/ALLaM-7B-Instruct-preview Qwen/Qwen2.5-7B-Instruct --data data --out results

Multiple choice (ArabLegalEval), scored by the probability of each option letter, in three settings:
  closed book   the question and options only
  retrieved     plus the top 3 passages from the retriever
  gold passage  plus only the passage the question was written from (an upper bound for retrieval)

Grounded answers (Najiz and part of test), generated in Arabic with a citation such as [2]:
  answerable    the top 3 retrieved passages; scored when the gold passage is among them
  unanswerable  the top 3 retrieved passages with the gold passage removed; the right behaviour is to say so
"""
from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np

from .data import build
from .llm import CITE, abstained, choose, citations, generate, load, mcq_messages, rag_messages, render
from .retrieval import BM25, Dense, rrf
from .text import tokens

MAX_CHARS = 1200


def f1(pred: str, gold: str) -> float:
    p, g = tokens(CITE.sub(" ", pred)), tokens(gold)
    if not p or not g:
        return 0.0
    common = sum(min(p.count(t), g.count(t)) for t in set(p))
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def clip(t: str) -> str:
    return t if len(t) <= MAX_CHARS else t[:MAX_CHARS] + "…"


def mcq_eval(model, tok, qs, ranked, text, k=3) -> dict:
    res = {}
    for setting in ("closed book", "retrieved", "gold passage"):
        prompts = []
        for q, r in zip(qs, ranked):
            ctx = None if setting == "closed book" else [clip(text[p]) for p in r[:k]] if setting == "retrieved" else [clip(text[q.gold])]
            prompts.append(render(tok, mcq_messages(q.text, q.options, ctx)))
        pred = choose(model, tok, prompts)
        res[setting] = float(np.mean([p == q.answer_index for p, q in zip(pred, qs)]))
    res["n"] = len(qs)
    res["gold_in_top3"] = float(np.mean([q.gold in r[:k] for q, r in zip(qs, ranked)]))
    return res


def grounded_eval(model, tok, qs, ranked, text, k=3) -> tuple[dict, list]:
    ans_prompts, ans_meta, un_prompts = [], [], []
    for q, r in zip(qs, ranked):
        top = r[:k]
        if q.gold in top:
            ans_prompts.append(render(tok, rag_messages(q.text, [clip(text[p]) for p in top])))
            ans_meta.append((q, top.index(q.gold) + 1, top))
        no_gold = [p for p in r if p != q.gold][:k]
        un_prompts.append(render(tok, rag_messages(q.text, [clip(text[p]) for p in no_gold])))
    ans = generate(model, tok, ans_prompts)
    un = generate(model, tok, un_prompts)
    rows, f1s, cite_ok, false_abstain = [], [], [], []
    for a, (q, gidx, top) in zip(ans, ans_meta):
        c = citations(a)
        ab = abstained(a)
        false_abstain.append(ab)
        if not ab:
            f1s.append(f1(a, q.answer))
        cite_ok.append(gidx in c)
        rows.append({"qid": q.qid, "question": q.text, "gold_answer": q.answer, "answer": a, "gold_index": gidx, "cited": c})
    res = {
        "answerable": len(ans_meta), "unanswerable": len(un_prompts),
        "answer_f1": float(np.mean(f1s)) if f1s else 0.0,
        "cites_gold": float(np.mean(cite_ok)) if cite_ok else 0.0,
        "false_abstain": float(np.mean(false_abstain)) if false_abstain else 0.0,
        "abstains_when_unanswerable": float(np.mean([abstained(a) for a in un])) if un else 0.0,
    }
    for a, q in list(zip(un, qs))[:20]:
        rows.append({"qid": q.qid, "question": q.text, "answer": a, "unanswerable": True})
    return res, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ALLaM-AI/ALLaM-7B-Instruct-preview", "Qwen/Qwen2.5-7B-Instruct"])
    ap.add_argument("--adapter", default=None, help="LoRA adapter for the first model")
    ap.add_argument("--retriever", default="intfloat/multilingual-e5-base")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-mcq", type=int, default=400)
    ap.add_argument("--n-gen", type=int, default=150)
    ap.add_argument("--skip-mcq", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    d = build(a.data)
    pids = list(d.passages)
    texts = [d.passages[p].text for p in pids]
    text = dict(zip(pids, texts))
    rng = random.Random(0)
    sets = {
        "dev": rng.sample(d.questions["dev"], min(a.n_mcq, len(d.questions["dev"]))),
        "test": rng.sample(d.questions["test"], min(a.n_mcq, len(d.questions["test"]))),
    }
    gen_sets = {"najiz": d.questions["najiz"], "test": sets["test"][:a.n_gen]}

    bm = BM25(pids, texts)
    dense = Dense(a.retriever, pids, texts)
    need = {q.qid: q for qs in list(sets.values()) + list(gen_sets.values()) for q in qs}
    order = list(need)
    ranked_all = rrf(bm.search([need[i].text for i in order]), dense.search([need[i].text for i in order]))
    ranked = dict(zip(order, ranked_all))
    del dense
    gc.collect()

    import torch
    report = {"retriever": f"BM25 + {a.retriever} (RRF)", "adapter": a.adapter, "models": {}}
    samples = {}
    for mi, m in enumerate(a.models):
        model, tok = load(m, a.adapter if mi == 0 else None)
        name = m.split("/")[-1] + (" + LoRA" if a.adapter and mi == 0 else "")
        r = {}
        if not a.skip_mcq:
            for s, qs in sets.items():
                r[f"mcq_{s}"] = mcq_eval(model, tok, qs, [ranked[q.qid] for q in qs], text)
                print(f"{name} mcq {s}: " + json.dumps(r[f"mcq_{s}"]), flush=True)
        for s, qs in gen_sets.items():
            g, rows = grounded_eval(model, tok, qs, [ranked[q.qid] for q in qs], text)
            r[f"grounded_{s}"] = g
            samples[f"{name} {s}"] = rows
            print(f"{name} grounded {s}: " + json.dumps(g), flush=True)
        report["models"][name] = r
        (out / f"answers{a.tag}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
        (out / f"answer_samples{a.tag}.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False))
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\nMultiple choice accuracy (%): closed book / retrieved top-3 / gold passage")
    for name, r in report["models"].items():
        for s in ("dev", "test"):
            if f"mcq_{s}" in r:
                x = r[f"mcq_{s}"]
                print(f"{name:36s} {s:5s} {x['closed book']*100:5.1f} / {x['retrieved']*100:5.1f} / {x['gold passage']*100:5.1f}   (gold in top 3: {x['gold_in_top3']*100:.1f}%)")
    print("\nGrounded answers: answer F1 / cites the gold passage / false 'not found' / says 'not found' when the law is missing")
    for name, r in report["models"].items():
        for s in ("najiz", "test"):
            x = r[f"grounded_{s}"]
            print(f"{name:36s} {s:5s} {x['answer_f1']:.3f} / {x['cites_gold']*100:5.1f}% / {x['false_abstain']*100:5.1f}% / {x['abstains_when_unanswerable']*100:5.1f}%   (n={x['answerable']}/{x['unanswerable']})")


if __name__ == "__main__":
    main()
