"""Write the 717 ALARB statute articles (Apache-2.0) to docs/articles.json for the demo page's search box.

    python -m sanad.export_articles --out docs/articles.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .data import build


def _num(article: str) -> int:
    m = re.search(r"\d+", article or "")
    return int(m.group()) if m else 10 ** 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="docs/articles.json")
    a = ap.parse_args()
    d = build(a.data)
    ps = [p for p in d.passages.values() if "alarb" in p.sources]
    ps.sort(key=lambda p: (p.law or "", _num(p.article)))
    rows = [{"law": p.law, "article": p.article, "text": p.text} for p in ps]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")))
    print(f"{len(rows)} articles -> {a.out}")


if __name__ == "__main__":
    main()
