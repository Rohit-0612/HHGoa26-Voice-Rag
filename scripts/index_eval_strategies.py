"""Index ALL FOUR chunking strategies over the SAME 2,000 passages, into a
dedicated eval collection.

Why a separate collection: the production collection holds passage_native over
all 27,958 passages. Comparing that against new strategies covering only 2,000
would measure distractor count, not chunking quality. Identical corpora is the
whole point of the experiment.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import models

from src.config import DATA_DIR, settings
from src.ingestion.chunking import get_strategy
from src.ingestion.loader import read_jsonl
from src.retrieval.embedder import FastEmbedProvider
from src.retrieval.indexer import DENSE, SPARSE, batched, build_points, get_client, upsert_batch

EVAL_COLLECTION = "msmarco_xi_eval"
SUBSET = DATA_DIR / "eval_subset.jsonl"


def ensure_eval_collection(client, recreate: bool):
    if recreate and client.collection_exists(EVAL_COLLECTION):
        client.delete_collection(EVAL_COLLECTION)
    if not client.collection_exists(EVAL_COLLECTION):
        client.create_collection(
            collection_name=EVAL_COLLECTION,
            vectors_config={DENSE: models.VectorParams(
                size=settings.dense_dim, distance=models.Distance.COSINE, on_disk=True)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(
                modifier=models.Modifier.IDF)},
        )
    for field in ("language", "strategy"):
        try:
            client.create_payload_index(
                collection_name=EVAL_COLLECTION, field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD)
        except Exception:
            pass


def main() -> int:
    if not SUBSET.exists():
        print("Run scripts/build_eval_subset.py first."); return 1

    records = list(read_jsonl(SUBSET))
    print(f"{len(records):,} passages, {len({r['language'] for r in records})} languages")

    provider = FastEmbedProvider()
    client = get_client()
    ensure_eval_collection(client, recreate="--recreate" in sys.argv)

    strategies = [
        ("passage_native", {}),
        ("fixed_size", {"chunk_chars": settings.fixed_chunk_chars,
                        "overlap_chars": settings.fixed_overlap_chars}),
        ("sentence_window", {"window_size": settings.window_size}),
        ("semantic", {"embed_fn": provider.embed_dense_only,
                      "percentile": settings.semantic_breakpoint_percentile,
                      "min_chars": settings.semantic_min_chars}),
    ]

    grand_total = time.perf_counter()
    summary: dict[str, int] = {}

    for name, kwargs in strategies:
        strategy = get_strategy(name, **kwargs)
        t0 = time.perf_counter()
        chunks = strategy.chunk_many(records)
        summary[name] = len(chunks)
        print(f"\n[{name}] {len(records):,} passages -> {len(chunks):,} chunks "
              f"({time.perf_counter() - t0:.1f}s chunking)", flush=True)

        done, t_emb = 0, time.perf_counter()
        for batch in batched(chunks, 32):
            dense, sparse = provider.embed_documents([c.text for c in batch])
            pts = build_points(batch, dense, sparse)
            for sub in batched(pts, 128):
                upsert_batch_eval(client, sub)
            done += len(batch)
            el = time.perf_counter() - t_emb
            rate = done / el if el else 0
            print(f"  {done:>6,}/{len(chunks):,}  {rate:4.1f}/s  "
                  f"ETA {(len(chunks)-done)/rate/60 if rate else 0:4.1f}m", flush=True)

    print(f"\n{'strategy':<18}{'chunks':>10}")
    for k, v in summary.items():
        print(f"{k:<18}{v:>10,}")
    info = client.get_collection(EVAL_COLLECTION)
    print(f"\n'{EVAL_COLLECTION}' holds {info.points_count:,} points "
          f"({(time.perf_counter() - grand_total)/60:.1f} min total)")
    return 0


def upsert_batch_eval(client, points, retries: int = 3):
    last = None
    for attempt in range(retries):
        try:
            client.upsert(collection_name=EVAL_COLLECTION, points=points, wait=False)
            return
        except Exception as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"eval upsert failed: {last}")


if __name__ == "__main__":
    raise SystemExit(main())
