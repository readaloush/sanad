"""Sanad search and answer service.

    SANAD_ARTIFACTS=artifacts SANAD_LLM_URL=http://gpu-host:8000 SANAD_LLM_MODEL=sanad uvicorn service.app:app

GET  /health          corpus size and whether a language model is configured
GET  /search?q=&k=    the k most relevant articles, no language model needed
POST /ask {"question": "...", "k": 3}
                      an answer from the language model, the articles it was given,
                      which of them it cited, and whether it said the law does not answer the question
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from sanad.serve import QUERY_PREFIX, ChatModel, Corpus, TextEncoder, ask

ARTIFACTS = Path(os.environ.get("SANAD_ARTIFACTS", "artifacts"))
MODEL_FILE = os.environ.get("SANAD_ENCODER", "text.onnx")

app = FastAPI(title="Sanad", description="Question answering over Saudi law, in Arabic, with the article it rests on. Not legal advice.")


@lru_cache(maxsize=1)
def state() -> tuple[TextEncoder, Corpus]:
    missing = [f for f in (MODEL_FILE, "tokenizer.json", "corpus.npy", "corpus.json") if not (ARTIFACTS / f).exists()]
    if missing:
        raise HTTPException(503, f"missing in {ARTIFACTS}: {', '.join(missing)}. Run python -m sanad.build_index to build them.")
    return TextEncoder(ARTIFACTS / MODEL_FILE, ARTIFACTS / "tokenizer.json"), Corpus.load(ARTIFACTS)


def llm() -> ChatModel | None:
    url = os.environ.get("SANAD_LLM_URL")
    return ChatModel(url, os.environ.get("SANAD_LLM_MODEL", "sanad"), os.environ.get("SANAD_LLM_KEY", "")) if url else None


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(3, ge=1, le=5)


@app.get("/health")
def health():
    _, corpus = state()
    return {"status": "ok", "articles": len(corpus), "language_model": bool(llm())}


@app.get("/search")
def search(q: str = Query(min_length=3, max_length=500), k: int = Query(5, ge=1, le=20)):
    enc, corpus = state()
    return {"query": q, "articles": corpus.search(enc([QUERY_PREFIX + q.strip()])[0], k)}


@app.post("/ask")
def answer(body: Question):
    model = llm()
    if model is None:
        raise HTTPException(503, "No language model is configured. Set SANAD_LLM_URL to an OpenAI-compatible endpoint, or use /search.")
    enc, corpus = state()
    try:
        return ask(body.question.strip(), enc, corpus, model, body.k)
    except OSError as e:
        raise HTTPException(502, f"The language model did not answer: {e}")
