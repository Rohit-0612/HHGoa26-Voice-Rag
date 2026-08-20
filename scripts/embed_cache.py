"""Embed the corpus to a local cache, independent of Qdrant.

Embedding is the slowest stage and does not depend on credentials, so it can
run while Qdrant is still being provisioned. build_index.py reuses this cache
when present, which turns indexing into a pure upload.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, settings
from src.ingestion.chunking import get_strategy
from src.ingestion.loader import LANGS, read_jsonl
from src.retrieval.embedder import FastEmbedProvider


def cache_path(strategy: str) -> Path:
    return DATA_DIR / f"embeddings_{strategy}.pkl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="passage_native", choices=["passage_native", "semantic"])
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit-per-lang", type=int, default=None)
    args = ap.parse_args()

    langs = set(args.langs) if args.langs else set(LANGS)
    records = [r for r in read_jsonl(settings.passages_path) if r["language"] in langs]

    if args.limit_per_lang:
        from collections import Counter
        seen: Counter[str] = Counter()
        kept = []
        for r in records:
            if seen[r["language"]] < args.limit_per_lang:
                kept.append(r)
                seen[r["language"]] += 1
        records = kept

    print(f"{len(records):,} passages, {len({r['language'] for r in records})} languages")
    provider = FastEmbedProvider()
    print(f"dense={provider.dense_model_name} ({provider.dense_dim}d) sparse={provider.sparse_model_name}")

    if args.strategy == "semantic":
        strategy = get_strategy("semantic", embed_fn=provider.embed_dense_only,
                                percentile=settings.semantic_breakpoint_percentile,
                                min_chars=settings.semantic_min_chars)
    else:
        strategy = get_strategy("passage_native")

    chunks = strategy.chunk_many(records)
    print(f"{len(records):,} passages -> {len(chunks):,} chunks (strategy={strategy.name})")

    dense_all, sparse_all = [], []
    total, t_start = len(chunks), time.perf_counter()

    for i in range(0, total, args.batch_size):
        batch = chunks[i : i + args.batch_size]
        d, s = provider.embed_documents([c.text for c in batch])
        dense_all.extend(d)
        sparse_all.extend(s)
        done = len(dense_all)
        el = time.perf_counter() - t_start
        rate = done / el if el else 0
        print(f"  {done:>6,}/{total:,}  {rate:5.1f}/s  ETA {(total - done) / rate / 60 if rate else 0:5.1f}m",
              flush=True)

    out = cache_path(strategy.name)
    with open(out, "wb") as fh:
        pickle.dump({"chunks": chunks, "dense": dense_all, "sparse": sparse_all,
                     "dense_model": provider.dense_model_name, "dense_dim": provider.dense_dim}, fh)
    print(f"\nCached {total:,} embeddings -> {out} "
          f"({(time.perf_counter() - t_start) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
