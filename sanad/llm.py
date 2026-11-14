"""The answering model: prompts, loading, multiple-choice scoring and grounded generation."""
from __future__ import annotations

import re

import torch

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


def load(model_name: str, adapter: str | None = None, four_bit: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"device_map": {"": 0}, "torch_dtype": torch.float16}
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def render(tok, messages: list[dict], add_generation_prompt: bool = True) -> str:
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    except Exception:          # templates without a system role: fold it into the first user turn
        merged = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}] + messages[2:]
        return tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=add_generation_prompt)


def letter_ids(tok) -> list[list[int]]:
    """Token ids that can start the answer letter: the bare letter and, where the tokenizer has one, the letter with a leading space."""
    out = []
    for c in LETTERS:
        ids = {tok.encode(c, add_special_tokens=False)[0]}
        spaced = tok.encode(" " + c, add_special_tokens=False)
        if spaced and tok.decode([spaced[-1]]).strip() == c:
            ids.add(spaced[-1])
        out.append(sorted(ids))
    return out


@torch.no_grad()
def choose(model, tok, prompts: list[str], batch: int = 8, max_len: int = 3072) -> list[int]:
    """Index of the option letter with the highest next-token probability, for each prompt."""
    ids = letter_ids(tok)
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=max_len,
                  add_special_tokens=False).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        scores = torch.stack([logits[:, l].max(dim=1).values for l in ids], dim=1)
        out += scores.argmax(dim=1).tolist()
    return out


@torch.no_grad()
def generate(model, tok, prompts: list[str], batch: int = 8, max_new: int = 120, max_len: int = 3072) -> list[str]:
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=max_len,
                  add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id,
                             repetition_penalty=1.05)
        out += [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for g in gen]
    return out


CITE = re.compile(r"\[(\d+)\]")


def citations(answer: str) -> list[int]:
    return [int(x) for x in CITE.findall(answer)]


def abstained(answer: str) -> bool:
    return "لا أجد" in answer or "لا توجد الإجابة" in answer or "لم أجد" in answer
