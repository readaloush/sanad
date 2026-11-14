"""Step 2: fine-tune the embedding retriever on Saudi legal text.

    python -m sanad.finetune_retriever --base BAAI/bge-m3 --data data --out results

Two training sets are compared, both starting from the same model:

  questions         ArabLegalEval training questions, each paired with the passage it was written from
  questions+cases   the same, plus ALARB court cases: the facts of a case paired with each article it cites

Each pair gets one hard negative: the highest-ranked wrong passage from the starting model.
Loss: in-batch negatives (cached multiple-negatives ranking), so every other passage in a batch of
128 is a negative too. Training pairs never touch a passage that answers a dev, test or Najiz question.
The checkpoint kept is the one with the best dev MRR@10; test and Najiz are scored once at the end.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from .data import build  # noqa: E402
from .retrieval import BM25, PREFIX, Dense, metrics, rrf  # noqa: E402

SETS = ("dev", "test", "najiz")


def mine(base: Dense, queries: list[str], gold: list[set], k: int = 20, seed: int = 0) -> list[str]:
    """One hard negative per query: one of the five best-ranked passages that are not gold."""
    rng = random.Random(seed)
    ranked = base.search(queries, k=k)
    out = []
    for r, g in zip(ranked, gold):
        wrong = [p for p in r if p not in g]
        out.append(rng.choice(wrong[:5]) if wrong else rng.choice(base.pids))
    return out


def evaluate(model, model_name, d, pids, texts, bm_runs) -> dict:
    r = Dense(model_name, pids, texts, model=model)
    res = {}
    for s in SETS:
        qs = [q.text for q in d.questions[s]]
        gold = [q.gold for q in d.questions[s]]
        dense = r.search(qs)
        res[s] = {"dense": metrics(dense, gold), "hybrid": metrics(rrf(bm_runs[s], dense), gold)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="BAAI/bge-m3")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--mini", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--recipes", default="questions,questions+cases")
    ap.add_argument("--no-alarb", action="store_true", help="for a quick local check only")
    ap.add_argument("--limit", type=int, default=0, help="cap on training pairs, for a quick local check only")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    d = build(a.data, with_alarb=not a.no_alarb)
    pids = list(d.passages)
    texts = [d.passages[p].text for p in pids]
    text = dict(zip(pids, texts))
    qp, pp = PREFIX.get(a.base, ("", ""))

    held = {q.gold for s in SETS for q in d.questions[s]}
    bm = BM25(pids, texts)
    bm_runs = {s: bm.search([q.text for q in d.questions[s]]) for s in SETS}

    base = Dense(a.base, pids, texts, max_len=a.max_len)
    report = {"base": a.base, "data": d.stats, "runs": {}}
    report["runs"]["starting model"] = evaluate(base.model, a.base, d, pids, texts, bm_runs)

    train_q = d.questions["train"]
    q_neg = mine(base, [q.text for q in train_q], [{q.gold} for q in train_q])
    pairs_q = [(qp + q.text, pp + text[q.gold], pp + text[n]) for q, n in zip(train_q, q_neg)]

    cases = [(f, [p for p in ps if p not in held]) for f, ps in d.alarb_pairs]
    cases = [(f, ps) for f, ps in cases if ps and f.strip()]
    dropped = sum(1 for f, ps in d.alarb_pairs if any(p in held for p in ps))
    case_q = [f[:1500] for f, _ in cases]
    case_neg = mine(base, case_q, [set(ps) for _, ps in cases], seed=1)
    rng = random.Random(0)
    pairs_c = [(qp + f, pp + text[rng.choice(ps)], pp + text[n]) for f, (_, ps), n in zip(case_q, cases, case_neg)]
    report["training_pairs"] = {"questions": len(pairs_q), "cases": len(pairs_c), "cases_dropped_for_touching_held_out_passages": dropped}
    print(json.dumps(report["training_pairs"]), flush=True)
    del base
    gc.collect()
    torch.cuda.empty_cache()

    best_name, best_dev = "starting model", report["runs"]["starting model"]["dev"]["dense"]["MRR@10"]
    for recipe in a.recipes.split(","):
        pairs = pairs_q + (pairs_c if recipe == "questions+cases" else [])
        random.Random(1).shuffle(pairs)
        if a.limit:
            pairs = pairs[:a.limit]
        ds = Dataset.from_dict({"anchor": [p[0] for p in pairs], "positive": [p[1] for p in pairs], "negative": [p[2] for p in pairs]})
        cuda = torch.cuda.is_available()
        model = SentenceTransformer(a.base, device="cuda" if cuda else "cpu")
        model.max_seq_length = a.max_len
        loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=a.mini)
        run_dir = Path("runs") / recipe.replace("+", "_")
        args = SentenceTransformerTrainingArguments(
            output_dir=str(run_dir), num_train_epochs=a.epochs, per_device_train_batch_size=a.batch,
            learning_rate=a.lr, warmup_ratio=0.1, fp16=cuda, batch_sampler=BatchSamplers.NO_DUPLICATES,
            save_strategy="no", logging_steps=20, report_to="none", seed=0)
        t0 = time.time()
        SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
        minutes = round((time.time() - t0) / 60, 1)
        model.eval()
        res = evaluate(model, a.base, d, pids, texts, bm_runs)
        report["runs"][recipe] = {**res, "train_minutes": minutes, "pairs": len(pairs)}
        print(f"[{recipe}] {minutes} min  " + "  ".join(f"{s} R@1 {res[s]['dense']['R@1']:.3f} MRR {res[s]['dense']['MRR@10']:.3f}" for s in SETS), flush=True)
        if res["dev"]["dense"]["MRR@10"] > best_dev:
            best_name, best_dev = recipe, res["dev"]["dense"]["MRR@10"]
            model.save(str(out / "retriever"))
        (out / "retriever.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
        del model, loss
        gc.collect()
        torch.cuda.empty_cache()

    report["selected_on_dev"] = best_name
    (out / "retriever.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("\nRecall@1 / @5 / @10 / MRR@10, dense alone and hybrid with BM25")
    for name, r in report["runs"].items():
        for kind in ("dense", "hybrid"):
            print(f"{name:18s} {kind:7s} " + "   ".join(
                f"{s} {r[s][kind]['R@1']*100:4.1f}/{r[s][kind]['R@5']*100:4.1f}/{r[s][kind]['R@10']*100:4.1f}/{r[s][kind]['MRR@10']:.3f}" for s in SETS))
    print("selected on dev:", best_name)


if __name__ == "__main__":
    main()
