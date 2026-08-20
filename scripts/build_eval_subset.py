"""Carve a fixed eval subset so all four chunking strategies are compared on
identical source passages.

Without a frozen subset the comparison is meaningless: passage_native is
already indexed over all 27,958 passages while the new strategies would cover
something else, and recall differences would just reflect corpus size.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, settings
from src.ingestion.loader import read_jsonl

EVAL_LANGS = ["hi", "bn", "ta", "or"]  # spans Day 1 quality range: .78 .64 .65 .31
PASSAGES_PER_LANG = 500
OUT = DATA_DIR / "eval_subset.jsonl"


def main() -> int:
    kept: list[dict] = []
    per_lang: Counter[str] = Counter()
    gold: dict[str, set[str]] = defaultdict(set)

    for rec in read_jsonl(settings.passages_path):
        lang = rec["language"]
        if lang not in EVAL_LANGS or per_lang[lang] >= PASSAGES_PER_LANG:
            continue
        kept.append(rec)
        per_lang[lang] += 1
        if rec.get("is_selected"):
            gold[f"{lang}::{rec['query_id']}"].add(rec["passage_id"])

    with open(OUT, "w", encoding="utf-8") as fh:
        for rec in kept:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    (DATA_DIR / "eval_gold.json").write_text(
        json.dumps({k: sorted(v) for k, v in gold.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"{'lang':<6}{'passages':>10}{'queries w/ gold':>18}")
    for lang in EVAL_LANGS:
        n_gold = sum(1 for k in gold if k.startswith(f"{lang}::"))
        print(f"{lang:<6}{per_lang[lang]:>10,}{n_gold:>18}")
    print(f"\n{len(kept):,} passages -> {OUT}")
    print(f"{len(gold):,} queries with >=1 gold passage -> {DATA_DIR / 'eval_gold.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
