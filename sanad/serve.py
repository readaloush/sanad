"""The serving path: retrieve articles, ask the language model, check its citations.

Nothing here imports torch. The retriever is multilingual-e5-base exported to ONNX (see
sanad.build_index); the language model is reached over an OpenAI-compatible chat API, which vLLM,
llama.cpp's server and Ollama all provide, so ALLaM with the Sanad LoRA adapter can run wherever
there is a GPU while this service runs on a small CPU machine.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .prompts import abstained, citations, rag_messages

QUERY_PREFIX = "query: "      # multilingual-e5 expects these
PASSAGE_PREFIX = "passage: "


class TextEncoder:
    """Text -> L2-normalised e5 embedding, with ONNX Runtime and the tokenizers library only."""

    def __init__(self, model_path: str | Path, tokenizer_path: str | Path, max_length: int = 256):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(tokenizer_path))
        self.tok.enable_truncation(max_length=max_length)
        pad = next((self.tok.token_to_id(t) for t in ("<pad>", "[PAD]") if self.tok.token_to_id(t) is not None), 0)
        self.tok.enable_padding(pad_id=pad, pad_token=self.tok.id_to_token(pad) or "<pad>")
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def __call__(self, texts: list[str], batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok.encode_batch(texts[i:i + batch])
            feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                    "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
            out.append(self.session.run(None, feed)[0])
        v = np.concatenate(out).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-12)


class Corpus:
    """Legal passages with their law name and article number, and their embeddings."""

    def __init__(self, embeddings: np.ndarray, passages: list[dict]):
        if len(embeddings) != len(passages):
            raise ValueError(f"{len(embeddings)} embeddings for {len(passages)} passages")
        self.emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.passages = passages

    @classmethod
    def load(cls, folder: str | Path) -> "Corpus":
        folder = Path(folder)
        return cls(np.load(folder / "corpus.npy"), json.loads((folder / "corpus.json").read_text()))

    def __len__(self) -> int:
        return len(self.passages)

    def search(self, query_vec: np.ndarray, k: int = 3) -> list[dict]:
        k = max(1, min(k, len(self)))
        sim = self.emb @ query_vec.reshape(-1)
        top = np.argpartition(-sim, k - 1)[:k]
        top = top[np.argsort(-sim[top])]
        return [{**self.passages[i], "score": round(float(sim[i]), 4)} for i in top]


@dataclass
class ChatModel:
    """Any OpenAI-compatible /v1/chat/completions endpoint."""
    url: str
    model: str
    api_key: str = ""
    timeout: float = 120.0

    def __call__(self, messages: list[dict], max_tokens: int = 200) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(self.url.rstrip("/") + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()


MAX_CHARS = 1200


def ask(question: str, encoder: TextEncoder, corpus: Corpus, llm, k: int = 3) -> dict:
    """Retrieve k passages, ask the model, and keep only citations that point at a passage it was shown."""
    t0 = time.perf_counter()
    hits = corpus.search(encoder([QUERY_PREFIX + question])[0], k)
    t1 = time.perf_counter()
    passages = [h["text"] if len(h["text"]) <= MAX_CHARS else h["text"][:MAX_CHARS] + "…" for h in hits]
    answer = llm(rag_messages(question, passages))
    t2 = time.perf_counter()
    cited = citations(answer)
    valid = sorted({c for c in cited if 1 <= c <= len(hits)})
    return {
        "question": question,
        "answer": answer,
        "not_found": abstained(answer),
        "sources": [{"n": i + 1, "law": h.get("law", ""), "article": h.get("article", ""), "text": h["text"],
                     "score": h["score"], "cited": (i + 1) in valid} for i, h in enumerate(hits)],
        "invalid_citations": sorted({c for c in cited if c not in valid}),
        "retrieve_ms": round((t1 - t0) * 1000, 1),
        "answer_ms": round((t2 - t1) * 1000, 1),
    }
