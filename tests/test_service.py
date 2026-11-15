"""Serving path with a stand-in encoder and language model: tokenizer -> ONNX -> search -> prompt -> citations -> HTTP."""
import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from sanad.prompts import ABSTAIN
from sanad.serve import Corpus, TextEncoder, ask

WORDS = ["<pad>", "<unk>", "query", "passage", ":", "عامل", "إجازة", "شركة", "رأس", "مال", "محكمة", "دعوى"]
TOPIC = {"عامل": 0, "إجازة": 0, "شركة": 1, "رأس": 1, "مال": 1, "محكمة": 2, "دعوى": 2}


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        w = torch.zeros(len(WORDS), 3)
        for word, axis in TOPIC.items():
            w[WORDS.index(word), axis] = 1.0
        self.emb = torch.nn.Embedding.from_pretrained(w)

    def forward(self, input_ids, attention_mask):
        m = attention_mask.unsqueeze(-1).float()
        return (self.emb(input_ids) * m).sum(1) / m.sum(1).clamp(min=1)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    d = tmp_path_factory.mktemp("artifacts")
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(d / "tokenizer.json"))
    ids = torch.tensor([[5, 6, 0]])
    torch.onnx.export(Tiny(), (ids, (ids > 0).long()), str(d / "text.int8.onnx"), dynamo=False,
                      input_names=["input_ids", "attention_mask"], output_names=["embedding"],
                      dynamic_axes={"input_ids": {0: "b", 1: "t"}, "attention_mask": {0: "b", 1: "t"}, "embedding": {0: "b"}})
    np.save(d / "corpus.npy", np.eye(3, dtype=np.float16))
    (d / "corpus.json").write_text(json.dumps([
        {"id": "a", "law": "نظام العمل", "article": "109", "text": "للعامل إجازة سنوية"},
        {"id": "b", "law": "نظام الشركات", "article": "10", "text": "رأس مال الشركة"},
        {"id": "c", "law": "نظام المحاكم التجارية", "article": "22", "text": "تحيل المحكمة الدعوى"}], ensure_ascii=False))
    return d


def test_search_finds_the_right_article(artifacts):
    enc = TextEncoder(artifacts / "text.int8.onnx", artifacts / "tokenizer.json")
    corpus = Corpus.load(artifacts)
    assert corpus.search(enc(["query : إجازة عامل"])[0], 1)[0]["article"] == "109"
    assert corpus.search(enc(["query : دعوى"])[0], 1)[0]["id"] == "c"


def test_ask_keeps_only_citations_to_passages_it_was_shown(artifacts):
    enc = TextEncoder(artifacts / "text.int8.onnx", artifacts / "tokenizer.json")
    corpus = Corpus.load(artifacts)
    seen = {}

    def fake_llm(messages):
        seen["prompt"] = messages[1]["content"]
        return "للعامل إجازة سنوية [1] و[7]"

    r = ask("إجازة عامل", enc, corpus, fake_llm, k=2)
    assert "[1] للعامل إجازة سنوية" in seen["prompt"]
    assert r["sources"][0]["cited"] and r["sources"][0]["law"] == "نظام العمل"
    assert r["invalid_citations"] == [7] and not r["not_found"]


def test_ask_reports_not_found(artifacts):
    enc = TextEncoder(artifacts / "text.int8.onnx", artifacts / "tokenizer.json")
    r = ask("شركة", enc, Corpus.load(artifacts), lambda m: ABSTAIN)
    assert r["not_found"] and not any(s["cited"] for s in r["sources"])


def test_http(artifacts, monkeypatch):
    from fastapi.testclient import TestClient

    import service.app as svc
    monkeypatch.setattr(svc, "ARTIFACTS", artifacts)
    monkeypatch.delenv("SANAD_LLM_URL", raising=False)
    svc.state.cache_clear()
    c = TestClient(svc.app)
    assert c.get("/health").json() == {"status": "ok", "articles": 3, "language_model": False}
    assert c.get("/search", params={"q": "رأس مال شركة", "k": 1}).json()["articles"][0]["id"] == "b"
    assert c.post("/ask", json={"question": "إجازة عامل"}).status_code == 503
    monkeypatch.setattr(svc, "llm", lambda: (lambda messages: "للعامل إجازة [1]"))
    r = c.post("/ask", json={"question": "إجازة عامل"}).json()
    assert r["sources"][0]["cited"] and r["answer"].endswith("[1]")
    assert c.post("/ask", json={"question": "x"}).status_code == 422
    svc.state.cache_clear()
