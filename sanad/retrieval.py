"""Retrievers and retrieval metrics.

Every retriever maps a list of questions to, for each question, a ranked list of passage ids.
"""
from __future__ import annotations

import numpy as np

from .text import tokens


class BM25:
    name = "BM25"

    def __init__(self, pids: list[str], texts: list[str], k1: float = 1.2, b: float = 0.75):
        from rank_bm25 import BM25Okapi
        self.pids = pids
        self.bm = BM25Okapi([tokens(t) or ["_"] for t in texts], k1=k1, b=b)

    def search(self, queries: list[str], k: int = 100) -> list[list[str]]:
        out = []
        for q in queries:
            s = self.bm.get_scores(tokens(q) or ["_"])
            top = np.argpartition(-s, min(k, len(s) - 1))[:k]
            out.append([self.pids[i] for i in top[np.argsort(-s[top])]])
        return out


# How each model expects queries and passages to be marked, from its model card.
PREFIX = {
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large-instruct": ("Instruct: Given a question about Saudi law, retrieve the legal text that answers it\nQuery: ", ""),
}


class Dense:
    def __init__(self, model_name: str, pids: list[str], texts: list[str], batch: int = 64, max_len: int = 512, model=None):
        import torch
        from sentence_transformers import SentenceTransformer

        self.name = model_name.split("/")[-1]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model or SentenceTransformer(model_name, device=dev)
        self.model.max_seq_length = max_len
        if dev == "cuda":
            self.model.half()
        self.qp, pp = PREFIX.get(model_name, ("", ""))
        self.pids = pids
        self.emb = self.model.encode([pp + t for t in texts], batch_size=batch, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self.model.encode([self.qp + q for q in queries], batch_size=128, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def search(self, queries: list[str], k: int = 100) -> list[list[str]]:
        q = self.encode_queries(queries)
        out = []
        for i in range(0, len(q), 512):
            s = q[i:i + 512] @ self.emb.T
            top = np.argpartition(-s, k, axis=1)[:, :k]
            for row, t in zip(s, top):
                out.append([self.pids[j] for j in t[np.argsort(-row[t])]])
        return out


def rrf(*rankings: list[list[str]], k: int = 60, depth: int = 100) -> list[list[str]]:
    """Reciprocal rank fusion of several retrievers' rankings for the same queries."""
    out = []
    for lists in zip(*rankings):
        score: dict[str, float] = {}
        for ranked in lists:
            for r, pid in enumerate(ranked[:depth]):
                score[pid] = score.get(pid, 0.0) + 1.0 / (k + r + 1)
        out.append(sorted(score, key=score.get, reverse=True)[:depth])
    return out


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", max_len: int = 512):
        import torch
        from sentence_transformers import CrossEncoder

        self.name = model_name.split("/")[-1]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, max_length=max_len, device=dev)
        if dev == "cuda":
            self.model.model.half()

    def rerank(self, queries: list[str], ranked: list[list[str]], text: dict[str, str], top: int = 50, batch: int = 64) -> list[list[str]]:
        pairs, owner = [], []
        for qi, (q, r) in enumerate(zip(queries, ranked)):
            for pid in r[:top]:
                pairs.append((q, text[pid]))
                owner.append((qi, pid))
        scores = self.model.predict(pairs, batch_size=batch, show_progress_bar=False)
        per: list[list[tuple[float, str]]] = [[] for _ in queries]
        for (qi, pid), s in zip(owner, scores):
            per[qi].append((float(s), pid))
        return [[pid for _, pid in sorted(p, reverse=True)] + r[top:] for p, r in zip(per, ranked)]


def metrics(ranked: list[list[str]], gold: list[str], ks=(1, 5, 10, 20)) -> dict:
    """Recall@k (the gold passage is in the top k) and MRR@10, over questions with exactly one gold passage."""
    ranks = []
    for r, g in zip(ranked, gold):
        ranks.append(r.index(g) if g in r else 10 ** 6)
    ranks = np.array(ranks)
    out = {f"R@{k}": float((ranks < k).mean()) for k in ks}
    out["MRR@10"] = float(np.where(ranks < 10, 1.0 / (ranks + 1), 0.0).mean())
    out["n"] = int(len(ranks))
    return out
