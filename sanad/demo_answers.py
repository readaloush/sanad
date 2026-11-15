"""Answers for the demo page: example questions answered by ALLaM before and after fine-tuning.

    python -m sanad.demo_answers --adapters lora2=results/lora2 lora3=results/lora3 --data data --out results/demo.json

The demo corpus is the 717 statute articles cited in ALARB court cases (Apache-2.0), so the page can
show the articles themselves. The questions are written in plain language and two of them ask about
law that is not in the corpus (labour and traffic law), where the right answer is "not found".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import build
from .llm import generate, load, render
from .prompts import abstained, citations, rag_messages
from .retrieval import Dense

QUESTIONS = [
    ("إذا لم أتفق مع المحامي على أتعابه، فمن يحدد المبلغ؟", True),
    ("توفي المحامي قبل أن تنتهي قضيتي، فكيف تُحسب أتعابه لورثته؟", True),
    ("كنت أعمل محاميًا لدى شركة ثم تركتها، فمتى يحق لي أن أترافع ضدها؟", True),
    ("ما أنواع الشركات التي يمكن تأسيسها في السعودية؟", True),
    ("هل يحق لي بصفتي مساهمًا أن أطلع على حسابات الشركة؟", True),
    ("من يستطيع أن يطلب من المحكمة تصفية شركة عاجزة عن سداد ديونها؟", True),
    ("كم مدة إجازة الأمومة للموظفة في القطاع الخاص؟", False),
    ("ما غرامة قطع الإشارة الحمراء؟", False),
]
MAX_CHARS = 1200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ALLaM-AI/ALLaM-7B-Instruct-preview")
    ap.add_argument("--adapters", nargs="+", required=True, help="name=path; each is compared with the model before fine-tuning")
    ap.add_argument("--retriever", default="intfloat/multilingual-e5-base")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/demo.json")
    a = ap.parse_args()

    import gc

    import torch

    d = build(a.data)
    ps = [p for p in d.passages.values() if "alarb" in p.sources]
    dense = Dense(a.retriever, [p.pid for p in ps], [p.text for p in ps])
    ranked = dense.search([q for q, _ in QUESTIONS], k=3)
    by_id = {p.pid: p for p in ps}
    del dense
    gc.collect()
    torch.cuda.empty_cache()

    rows = []
    for (q, answerable), r in zip(QUESTIONS, ranked):
        src = [by_id[pid] for pid in r]
        rows.append({"question": q, "in_corpus": answerable,
                     "sources": [{"law": p.law, "article": p.article, "text": p.text} for p in src],
                     "messages": rag_messages(q, [p.text if len(p.text) <= MAX_CHARS else p.text[:MAX_CHARS] + "…" for p in src])})

    runs = [("before", None)] + [tuple(x.split("=", 1)) for x in a.adapters]
    for tag, adapter in runs:
        model, tok = load(a.model, adapter)
        answers = generate(model, tok, [render(tok, row["messages"]) for row in rows], batch=4, max_new=160)
        for row, ans in zip(rows, answers):
            row[tag] = {"answer": ans, "cited": citations(ans), "not_found": abstained(ans)}
            print(f"[{tag}] {row['question']}\n   {ans}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    for row in rows:
        row.pop("messages")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
