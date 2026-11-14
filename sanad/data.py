"""Corpus and question sets.

Three public sources, downloaded at run time and never stored in this repository:

- ArabLegalEval (THIQAH and KAUST, github.com/Thiqah/ArabLegalEval). Multiple-choice questions
  generated from Saudi legal texts; each question keeps the text it was written from ("context").
  No licence is stated in that repository, so it is used for evaluation only and not redistributed.
- Najiz QA, in the same repository: 79 expert-verified questions from public FAQs, with the legal text
  that answers each one.
- ALARB (THIQAH, huggingface.co/datasets/HarethahMo/ALARB, Apache-2.0): Saudi commercial court cases
  with the statute articles each case cites, as law name, article number and article text.

The retrieval corpus is the union of all three sources' legal texts, de-duplicated after normalisation.
Questions are split by the passage they were written from, so no passage is the answer to a training
question and a test question at the same time.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .text import key

ALE_REPO = "https://github.com/Thiqah/ArabLegalEval.git"
ALARB = "HarethahMo/ALARB"


@dataclass
class Passage:
    pid: str
    text: str
    law: str = ""
    article: str = ""
    sources: set = field(default_factory=set)


@dataclass
class Question:
    qid: str
    text: str
    gold: str              # pid of the passage the question was written from
    answer: str
    options: list = field(default_factory=list)
    answer_index: int = -1
    source: str = ""


def fetch_arablegaleval(root: Path) -> Path:
    repo = root / "ArabLegalEval"
    if not repo.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", ALE_REPO, str(repo)], check=True)
    return repo


def _article_fields(entry) -> tuple[str, str, str]:
    """(law, article, text) from one ALARB applicable_laws entry, whatever its exact key names."""
    if isinstance(entry, str):
        # ALARB stores each article as "law name:article number: article text"
        parts = entry.split(":", 2)
        if len(parts) == 3 and len(parts[0]) < 120 and len(parts[1].strip()) < 20:
            return parts[0].strip(), parts[1].strip(), parts[2].strip()
        return "", "", entry
    if not isinstance(entry, dict):
        return "", "", str(entry)
    low = {k.lower(): v for k, v in entry.items()}
    text = next((str(low[k]) for k in ("text", "article_text", "content", "body", "clause") if low.get(k)), "")
    if not text:
        text = max((str(v) for v in entry.values()), key=len, default="")
    law = next((str(low[k]) for k in ("law", "law_name", "name", "statute", "regulation", "title") if low.get(k)), "")
    art = next((str(low[k]) for k in ("article", "article_number", "number", "article_no", "num") if low.get(k)), "")
    return law, art, text


def load_alarb():
    from datasets import load_dataset
    return load_dataset(ALARB)


@dataclass
class Data:
    passages: dict[str, Passage]
    questions: dict[str, list[Question]]       # split name -> questions
    alarb_pairs: list[tuple[str, list[str]]]   # (case facts, [pid, ...]) from ALARB train, for fine-tuning
    stats: dict


def _split_of(pid: str, held_out: set) -> str:
    if pid in held_out:            # passages behind the Najiz questions are never trained on
        return "test"
    h = int(hashlib.sha1(pid.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "dev" if h < 85 else "test"


def build(root: str | Path = "data", with_alarb: bool = True, log=print) -> Data:
    import pandas as pd

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    repo = fetch_arablegaleval(root)
    by_key: dict[str, Passage] = {}

    def add(text: str, source: str, law: str = "", article: str = "") -> str:
        text = (text or "").strip()
        k = key(text)
        if not k:
            return ""
        p = by_key.get(k)
        if p is None:
            p = Passage(pid="p" + hashlib.sha1(k.encode()).hexdigest()[:10], text=text)
            by_key[k] = p
        p.sources.add(source)
        if law and not p.law:
            p.law, p.article = law, article
        return p.pid

    stats = {}
    alarb_pairs, alarb_splits = [], {}
    if with_alarb:
        ds = load_alarb()
        sample = next((x for x in ds["train"]["applicable_laws"] if x), None)
        log(f"ALARB applicable_laws entry looks like: {json.dumps(sample[0], ensure_ascii=False)[:300] if sample else None}")
        for split in ds:
            n_cases = 0
            for case in ds[split]:
                pids = []
                for entry in case.get("applicable_laws") or []:
                    law, art, text = _article_fields(entry)
                    pid = add(text, "alarb", law, art)
                    if pid:
                        pids.append(pid)
                facts = case.get("case_facts") or []
                facts = "\n".join(facts) if isinstance(facts, list) else str(facts)
                if pids and split == "train":
                    alarb_pairs.append((facts, sorted(set(pids))))
                n_cases += 1
            alarb_splits[split] = n_cases
        stats["alarb_cases"] = alarb_splits
        stats["alarb_articles"] = sum("alarb" in p.sources for p in by_key.values())

    questions = {"train": [], "dev": [], "test": [], "najiz": []}
    najiz = json.loads((repo / "benchmarkQA" / "Najiz_QA_with_context_v2.benchmark.json").read_text())
    for i, (q, a, c) in enumerate(zip(najiz["questions"], najiz["ground_truths"], najiz["context"])):
        pid = add(str(c), "najiz")
        if pid:
            questions["najiz"].append(Question(f"najiz{i}", str(q), pid, str(a), source="najiz"))
    held_out = {q.gold for q in questions["najiz"]}
    mcq = pd.read_csv(repo / "benchmarkMCQs" / "All_Generated_MCQs_Evaluated_Shuffled.csv")
    for i, r in mcq.iterrows():
        pid = add(str(r["Context"]), "arablegaleval")
        if not pid:
            continue
        opts = [str(r[f"Option {k}"]) for k in range(1, 5)]
        questions[_split_of(pid, held_out)].append(Question(f"mcq{i}", str(r["Question"]), pid, str(r["Correct Answer"]),
                                                 opts, int(r["Answer Key"]), "arablegaleval"))
    ale = [p for p in by_key.values() if "arablegaleval" in p.sources]
    overlap = sum("alarb" in p.sources for p in ale)
    stats["arablegaleval_passages"] = len(ale)
    stats["arablegaleval_passages_also_in_alarb"] = overlap


    passages = {p.pid: p for p in by_key.values()}
    stats["najiz_passages_shared_with_arablegaleval"] = sum("arablegaleval" in passages[q.gold].sources for q in questions["najiz"])
    stats["passages"] = len(passages)
    stats["questions"] = {k: len(v) for k, v in questions.items()}
    log("data: " + json.dumps(stats, ensure_ascii=False))
    return Data(passages, questions, alarb_pairs, stats)
