"""Build what the service needs: the retriever as ONNX, and the embedded corpus.

    python -m sanad.build_index --out artifacts            # ALARB statute articles only (Apache-2.0)
    python -m sanad.build_index --out artifacts --all      # every passage, for private use only

The service uses text.onnx (float32). text.int8.onnx is written too, but measure it with sanad.onnx_eval first:
the first int8 export lost 13 points of Recall@1 on human questions.

By default the served corpus is the statute articles cited in ALARB court cases, which carry the
law's name and the article number and are under Apache-2.0, so the index can be shared. The
ArabLegalEval passages have no stated licence and are left out unless --all is given.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import build
from .serve import PASSAGE_PREFIX


class Wrapper(torch.nn.Module):
    def __init__(self, st):
        super().__init__()
        self.st = st

    def forward(self, input_ids, attention_mask):
        out = self.st({"input_ids": input_ids, "attention_mask": attention_mask})["sentence_embedding"]
        return torch.nn.functional.normalize(out, dim=-1)


def export(st, out: Path) -> dict:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    st = st.cpu().float().eval()
    st.tokenizer.save_pretrained(str(out / "tokenizer"))
    (out / "tokenizer.json").write_bytes((out / "tokenizer" / "tokenizer.json").read_bytes())
    feats = st.tokenize(["query: ما هي مدة الإشعار؟", "passage: نص"])
    kw = dict(input_names=["input_ids", "attention_mask"], output_names=["embedding"], opset_version=17,
              dynamic_axes={"input_ids": {0: "batch", 1: "tokens"}, "attention_mask": {0: "batch", 1: "tokens"},
                            "embedding": {0: "batch"}})
    with torch.no_grad():
        try:
            torch.onnx.export(Wrapper(st), (feats["input_ids"], feats["attention_mask"]), str(out / "text.onnx"), dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(Wrapper(st), (feats["input_ids"], feats["attention_mask"]), str(out / "text.onnx"), **kw)
    # int8 weights for the matrix multiplications only, one scale per output channel. Quantising every op with one
    # scale per tensor (quantize_dynamic's default) cost 13 points of Recall@1 on the Najiz questions (notebook 7).
    quantize_dynamic(str(out / "text.onnx"), str(out / "text.int8.onnx"), weight_type=QuantType.QInt8,
                     per_channel=True, op_types_to_quantize=["MatMul"])
    return {f: round((out / f).stat().st_size / 1e6, 1) for f in ("text.onnx", "text.int8.onnx")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer

    from .serve import TextEncoder

    d = build(a.data)
    ps = [p for p in d.passages.values() if a.all or "alarb" in p.sources]
    st = SentenceTransformer(a.model, device="cuda" if torch.cuda.is_available() else "cpu")
    st.max_seq_length = 256
    sizes = export(st, out)
    enc = TextEncoder(out / "text.onnx", out / "tokenizer.json")
    emb = enc([PASSAGE_PREFIX + p.text for p in ps])
    np.save(out / "corpus.npy", emb.astype(np.float16))
    (out / "corpus.json").write_text(json.dumps([{"id": p.pid, "law": p.law, "article": p.article, "text": p.text} for p in ps],
                                                ensure_ascii=False))
    ref = st.encode([PASSAGE_PREFIX + p.text for p in ps[:64]], normalize_embeddings=True)
    print(f"{len(ps)} passages ({sum(bool(p.law) for p in ps)} with law name and article number); onnx sizes MB {sizes}; "
          f"ONNX vs PyTorch cosine on 64 passages: min {float((ref * emb[:64]).sum(1).min()):.4f}")


if __name__ == "__main__":
    main()
