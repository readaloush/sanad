"""Prompts and answer parsing, shared by training, evaluation and the service. No heavy imports."""
from __future__ import annotations

import re

ABSTAIN = "لا أجد الإجابة في النصوص المتاحة."

SYSTEM_RAG = (
    "أنت مساعد قانوني يجيب عن الأسئلة المتعلقة بالأنظمة السعودية اعتمادًا على النصوص النظامية المرقّمة المعطاة فقط. "
    "أجب بإيجاز في جملة أو جملتين، واذكر رقم النص الذي استندت إليه بين معقوفين مثل [2]. "
    f"إذا لم تكن الإجابة موجودة في النصوص فقل حرفيًا: {ABSTAIN}"
)
SYSTEM_MCQ = "أنت خبير في الأنظمة السعودية. اختر الإجابة الصحيحة وأجب بحرف واحد فقط: A أو B أو C أو D."
LETTERS = "ABCD"


def context_block(passages: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))


def rag_messages(question: str, passages: list[str]) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_RAG},
            {"role": "user", "content": f"النصوص:\n{context_block(passages)}\n\nالسؤال: {question}"}]


def mcq_messages(question: str, options: list[str], passages: list[str] | None) -> list[dict]:
    opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))
    ctx = f"النصوص:\n{context_block(passages)}\n\n" if passages else ""
    return [{"role": "system", "content": SYSTEM_MCQ},
            {"role": "user", "content": f"{ctx}السؤال: {question}\n{opts}\n\nالإجابة (حرف واحد):"}]


CITE = re.compile(r"\[(\d+)\]")


def citations(answer: str) -> list[int]:
    return [int(x) for x in CITE.findall(answer)]


def abstained(answer: str) -> bool:
    return "لا أجد" in answer or "لا توجد الإجابة" in answer or "لم أجد" in answer
