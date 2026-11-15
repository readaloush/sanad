"""Step 4a: rewrite training questions the way a non-lawyer would ask them.

    python -m sanad.rewrite --model Qwen/Qwen2.5-7B-Instruct --n 2400 --data data --out results/rewrites.json

The ArabLegalEval questions were written by an LLM that had the legal passage in front of it, so they
reuse its wording. People ask differently, and steps 2 and 3 showed that training on the original
questions helps on LLM-written test questions but hurts on the 79 human-written Najiz questions.

The rewriting model sees only the question, never the passage, so it cannot copy more of the
passage's wording than the question already holds. Only training questions are rewritten.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .data import build
from .llm import generate, load, render

INSTRUCTION = (
    "أعد صياغة السؤال التالي كما يسأله مواطن عادي ليس محاميًا ولا يعرف المصطلحات النظامية: "
    "بلغة عربية فصحى بسيطة، وبكلمات مختلفة عن كلمات السؤال قدر الإمكان، دون أن يتغير المعنى أو المطلوب. "
    "اكتب السؤال الجديد فقط في سطر واحد."
)


def clean(text: str) -> str:
    line = text.strip().split("\n")[0].strip()
    for prefix in ("السؤال الجديد:", "السؤال:", "الصياغة:"):
        if line.startswith(prefix):
            line = line[len(prefix):].strip()
    return line.strip('"«»').strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/rewrites.json")
    a = ap.parse_args()

    d = build(a.data, with_alarb=False)
    qs = list(d.questions["train"])
    random.Random(0).shuffle(qs)
    qs = qs[:a.n]
    model, tok = load(a.model)
    prompts = [render(tok, [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": q.text}]) for q in qs]
    raw = generate(model, tok, prompts, batch=16, max_new=60)
    out = {}
    same = 0
    for q, r in zip(qs, raw):
        c = clean(r)
        if len(c) < 8 or c == q.text:
            same += 1
            continue
        out[q.qid] = c
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=0))
    print(f"rewrote {len(out)} of {len(qs)} training questions ({same} unusable)")
    for q in qs[:8]:
        if q.qid in out:
            print(f"  {q.text}\n  -> {out[q.qid]}")


if __name__ == "__main__":
    main()
