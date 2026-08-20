"""Step 4: chunk -> embed (dense+sparse) -> upsert into Qdrant Cloud.

Idempotent: point ids are UUID5 of chunk_id, so re-running overwrites rather
than duplicating. Safe to interrupt and restart.

  python scripts/build_index.py                      # passage_native, all langs
  python scripts/build_index.py --strategy semantic --langs hi ta bn
  python scripts/build_index.py --recreate
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.ingestion.chunking import get_strategy
from src.ingestion.loader import LANGS, read_jsonl
from src.retrieval.embedder import FastEmbedProvider
from src.retrieval.indexer import batched, build_points, ensure_collection, get_client, upsert_batch


def index_from_cache(cache, args) -> int:
    import pickle

    print(f"Using cached embeddings: {cache}")
    with open(cache, "rb") as fh:
        blob = pickle.load(fh)

    chunks, dense, sparse = blob["chunks"], blob["dense"], blob["sparse"]
    if blob.get("dense_dim") != settings.dense_dim:
        print(f"ERROR: cache was built with {blob.get('dense_dim')}d "
              f"({blob.get('dense_model')}) but settings say {settings.dense_dim}d. "
              f"Re-run scripts/embed_cache.py or pass --no-cache.")
        return 1

    langs = set(args.langs) if args.langs else None
    if langs:
        keep = [i for i, c in enumerate(chunks) if c.language in langs]
        chunks = [chunks[i] for i in keep]
        dense = [dense[i] for i in keep]
        sparse = [sparse[i] for i in keep]

    print(f"{len(chunks):,} chunks across {len({c.language for c in chunks})} languages")

    client = get_client()
    ensure_collection(client, recreate=args.recreate)

    total, done, t0 = len(chunks), 0, time.perf_counter()
    for start in range(0, total, args.batch_size):
        sl = slice(start, start + args.batch_size)
        upsert_batch(client, build_points(chunks[sl], dense[sl], sparse[sl]))
        done += len(chunks[sl])
        el = time.perf_counter() - t0
        rate = done / el if el else 0
        print(f"  upserted {done:>6,}/{total:,}  {rate:5.0f}/s  "
              f"ETA {(total - done) / rate / 60 if rate else 0:4.1f}m", flush=True)

    by_lang = Counter(c.language for c in chunks)
    print(f"\nUpserted {total:,} points in {(time.perf_counter() - t0) / 60:.1f} min")
    for lang, n in sorted(by_lang.items()):
        print(f"{lang:<6}{n:>9,}")
    info = client.get_collection(settings.qdrant_collection)
    print(f"\nCollection '{settings.qdrant_collection}' holds {info.points_count:,} points")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="passage_native", choices=["passage_native", "semantic"])
    ap.add_argument("--langs", nargs="*", default=None, help="default: all 14")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--limit-per-lang", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore data/embeddings_*.pkl and re-embed from scratch")
    args = ap.parse_args()

    langs = set(args.langs) if args.langs else set(LANGS)

    # Fast path: embeddings were pre-computed by scripts/embed_cache.py while
    # Qdrant was still being provisioned. Indexing then costs only the upload.
    from scripts.embed_cache import cache_path
    cache = cache_path(args.strategy)
    if cache.exists() and not args.no_cache:
        return index_from_cache(cache, args)

    if not settings.passages_path.exists():
        print(f"ERROR: {settings.passages_path} not found. Run scripts/prepare_data.py first.")
        return 1

    records = [r for r in read_jsonl(settings.passages_path) if r["language"] in langs]
    if args.limit_per_lang:
        seen: Counter[str] = Counter()
        kept = []
        for r in records:
            if seen[r["language"]] < args.limit_per_lang:
                kept.append(r)
                seen[r["language"]] += 1
        records = kept

    print(f"Loaded {len(records):,} passages across {len({r['language'] for r in records})} languages")

    print("Loading embedding models (first run downloads ~2GB) ...")
    provider = FastEmbedProvider()

    if args.strategy == "semantic":
        strategy = get_strategy(
            "semantic",
            embed_fn=provider.embed_dense_only,
            percentile=settings.semantic_breakpoint_percentile,
            min_chars=settings.semantic_min_chars,
        )
    else:
        strategy = get_strategy("passage_native")

    print(f"Chunking with strategy={strategy.name} ...")
    t0 = time.perf_counter()
    chunks = strategy.chunk_many(records)
    print(f"  {len(records):,} passages -> {len(chunks):,} chunks ({time.perf_counter() - t0:.1f}s)")

    client = get_client()
    ensure_collection(client, recreate=args.recreate)

    by_lang: Counter[str] = Counter(c.language for c in chunks)
    total = len(chunks)
    done = 0
    t_start = time.perf_counter()

    for batch in batched(chunks, args.batch_size):
        dense, sparse = provider.embed_documents([c.text for c in batch])
        upsert_batch(client, build_points(batch, dense, sparse))
        done += len(batch)
        elapsed = time.perf_counter() - t_start
        rate = done / elapsed if elapsed else 0
        eta = (total - done) / rate if rate else 0
        print(f"  {done:>6,}/{total:,}  {rate:5.1f} chunks/s  ETA {eta / 60:4.1f}m", flush=True)

    print(f"\nIndexed {total:,} chunks in {(time.perf_counter() - t_start) / 60:.1f} min")
    print(f"{'lang':<6}{'chunks':>9}")
    for lang, n in sorted(by_lang.items()):
        print(f"{lang:<6}{n:>9,}")

    info = client.get_collection(settings.qdrant_collection)
    print(f"\nCollection '{settings.qdrant_collection}' now holds {info.points_count:,} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
