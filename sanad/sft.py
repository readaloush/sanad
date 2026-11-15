"""Step 3b: teach the language model to answer from the retrieved law, cite it, and say when it is not there.

    python -m sanad.sft --model ALLaM-AI/ALLaM-7B-Instruct-preview --data data --out results/lora

QLoRA (4-bit base model, rank-16 adapters on every linear layer of the transformer blocks).
Each training example is an ArabLegalEval training question with the top 3 retrieved passages:

  1 - p  the gold passage is in the context (put there if the retriever missed it);
       target: the short correct answer followed by the number of that passage, e.g. "خلال خمسة أيام [2]"
  p      the gold passage is removed from the context (p = --p-unanswerable, 0.25 by default);
       target: the fixed "not found" sentence

No dev, test or Najiz question or passage is used.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch

from .data import build
from .llm import ABSTAIN, rag_messages, render
from .retrieval import BM25, Dense, rrf
from .text import tokens

TRAIN_CHARS = 700


def clip(t: str) -> str:
    return t if len(t) <= TRAIN_CHARS else t[:TRAIN_CHARS] + "…"


SENT = re.compile(r"(?<=[.؛:\n])\s*")


def support(passage: str, question: str, answer: str, max_chars: int = 260) -> str:
    """The sentence of the passage that shares the most words with the question and answer: the text the answer rests on."""
    ans, qs = set(tokens(answer)), set(tokens(question))
    best, score = "", -1
    for sent in SENT.split(passage):
        sent = sent.strip(" -")
        if len(sent) < 15:
            continue
        t = set(tokens(sent))
        sc = 3 * len(ans & t) + len(qs & t)
        if sc > score:
            best, score = sent, sc
    return best if len(best) <= max_chars else best[:max_chars].rsplit(" ", 1)[0] + "…"


def examples(d, ranked, n, p_unanswerable=0.25, k=3, seed=0, rewrites=None, full=False):
    rng = random.Random(seed)
    held = {q.gold for s in ("dev", "test", "najiz") for q in d.questions[s]}
    text = {p: d.passages[p].text for p in d.passages}
    qs = [q for q in d.questions["train"] if q.answer.strip() and (not rewrites or q.qid in rewrites)]
    rng.shuffle(qs)
    out = []
    for q in qs[:n]:
        r = [p for p in ranked[q.qid] if p not in held or p == q.gold]
        if rng.random() < p_unanswerable:
            ctx = [p for p in r if p != q.gold][:k]
            target = ABSTAIN
        else:
            ctx = [p for p in r[:k]]
            if q.gold not in ctx:
                ctx[rng.randrange(k)] = q.gold
            idx = ctx.index(q.gold) + 1
            target = f"{q.answer.strip()} [{idx}]"
            if full:
                sup = support(text[q.gold], q.text, q.answer)
                if sup:
                    target = f"{q.answer.strip()}، إذ ينص النص [{idx}] على: «{sup}»"
        question = rewrites[q.qid] if rewrites else q.text
        out.append((rag_messages(question, [clip(text[p]) for p in ctx]), target))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ALLaM-AI/ALLaM-7B-Instruct-preview")
    ap.add_argument("--retriever", default="intfloat/multilingual-e5-base")
    ap.add_argument("--fusion", choices=["dense", "hybrid"], default="dense")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/lora")
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1280)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--p-unanswerable", type=float, default=0.25)
    ap.add_argument("--full-answers", action="store_true",
                    help="target = the short answer + the sentence of the cited passage it rests on, instead of the short answer alone")
    ap.add_argument("--rewrites", default="", help="train on the plain-language rewrites from sanad.rewrite instead of the original questions")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

    d = build(a.data)
    pids = list(d.passages)
    texts = [d.passages[p].text for p in pids]
    rw = json.loads(Path(a.rewrites).read_text()) if a.rewrites else None
    train_q = [q for q in d.questions["train"] if not rw or q.qid in rw]
    asked = [rw[q.qid] if rw else q.text for q in train_q]   # retrieve with the wording the model will be trained on
    dense = Dense(a.retriever, pids, texts)
    runs = dense.search(asked)
    if a.fusion == "hybrid":
        runs = rrf(BM25(pids, texts).search(asked), runs)
    ranked = dict(zip([q.qid for q in train_q], runs))
    del dense
    torch.cuda.empty_cache()

    ex = examples(d, ranked, a.n, p_unanswerable=a.p_unanswerable, rewrites=rw, full=a.full_answers)
    for _, target in ex[:4]:
        print("example target:", target, flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    data = []
    for msgs, target in ex:
        prompt = render(tok, msgs)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tok(target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        p_ids = p_ids[-(a.max_len - len(t_ids)):]
        data.append((p_ids + t_ids, [-100] * len(p_ids) + t_ids))
    lengths = [len(x[0]) for x in data]
    print(f"{len(data)} examples, {sum(t == ABSTAIN for _, t in ex)} of them 'not found'; tokens per example: "
          f"median {sorted(lengths)[len(lengths) // 2]}, max {max(lengths)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        a.model, device_map={"": 0}, torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    names = {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)} - {"lm_head"}
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                     target_modules=sorted(names))
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(data) * a.epochs / a.accum)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler()
    model.train()
    rng = random.Random(1)
    t0, step, running = time.time(), 0, []
    for ep in range(a.epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for i, j in enumerate(order):
            ids, labels = data[j]
            x = torch.tensor([ids], device="cuda")
            y = torch.tensor([labels], device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=x, labels=y).loss / a.accum
            scaler.scale(loss).backward()
            running.append(loss.item() * a.accum)
            if (i + 1) % a.accum == 0 or i == len(order) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 10 == 0:
                    print(f"step {step}/{steps}  loss {sum(running[-80:]) / len(running[-80:]):.3f}  {(time.time() - t0) / 60:.1f} min", flush=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    (out / "train.json").write_text(json.dumps({"base": a.model, "examples": len(data), "epochs": a.epochs, "lr": a.lr,
                                                "p_unanswerable": a.p_unanswerable, "rewrites": bool(rw),
                                                "rank": a.rank, "steps": steps, "minutes": round((time.time() - t0) / 60, 1),
                                                "final_loss": sum(running[-80:]) / max(1, len(running[-80:]))}, indent=1))
    print("saved adapter to", out)


if __name__ == "__main__":
    main()
