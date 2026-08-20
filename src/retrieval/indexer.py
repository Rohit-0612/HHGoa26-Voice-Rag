"""Qdrant Cloud collection management + upserts.

Single collection for ALL languages and ALL chunking strategies, with payload
indexes on `language` and `strategy`. Day 2's two extra strategies land in the
same collection and stay filterable with no migration (decision.md D-07).
"""
from __future__ import annotations

import uuid
from typing import Iterable, Sequence

from qdrant_client import QdrantClient, models

from src.config import settings
from src.ingestion.chunking import Chunk

DENSE = "dense"
SPARSE = "sparse"
NAMESPACE = uuid.UUID("6f1c9a1e-3f9f-4a3e-9c2e-9a0b1f2d3c40")


def point_id(chunk_id: str) -> str:
    """Deterministic id so re-running the indexer overwrites instead of
    duplicating -- the build script must be safely resumable."""
    return str(uuid.uuid5(NAMESPACE, chunk_id))


def get_client(timeout: int = 120) -> QdrantClient:
    if not settings.qdrant_url or not settings.qdrant_api_key:
        raise RuntimeError(
            "QDRANT_URL / QDRANT_API_KEY missing. Copy .env.example to .env and fill them in."
        )
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        # Bulk upserts need a long timeout; queries must NOT inherit it. With a
        # 3-attempt retry wrapper, a 120s per-attempt timeout means a hung
        # cluster stalls a request for 6 minutes. Query paths pass ~20s.
        timeout=timeout,
        prefer_grpc=False,
    )


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    name = settings.qdrant_collection
    exists = client.collection_exists(name)

    if exists and recreate:
        client.delete_collection(name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE: models.VectorParams(
                    size=settings.dense_dim,
                    distance=models.Distance.COSINE,
                    # 1GB RAM tier: keep raw vectors on disk, HNSW graph in RAM.
                    on_disk=True,
                )
            },
            sparse_vectors_config={
                # IDF modifier is REQUIRED for BM25 -- Qdrant computes the
                # inverse-document-frequency term server side. Without it the
                # sparse branch scores on raw term frequency and ranks badly.
                SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    for field in ("language", "strategy"):
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # already indexed


def build_points(
    chunks: Sequence[Chunk],
    dense: Sequence[Sequence[float]],
    sparse: Sequence[models.SparseVector],
    extra: Sequence[dict] | None = None,
) -> list[models.PointStruct]:
    points = []
    for i, chunk in enumerate(chunks):
        payload = {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "language": chunk.language,
            "strategy": chunk.strategy,
            "passage_id": chunk.source_passage_id,
            **(extra[i] if extra else {}),
        }
        points.append(
            models.PointStruct(
                id=point_id(chunk.chunk_id),
                vector={DENSE: list(dense[i]), SPARSE: sparse[i]},
                payload=payload,
            )
        )
    return points


def upsert_batch(client: QdrantClient, points: list[models.PointStruct], retries: int = 3) -> None:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            client.upsert(collection_name=settings.qdrant_collection, points=points, wait=False)
            return
        except Exception as exc:
            last = exc
            import time as _t
            _t.sleep(2 ** attempt)
    raise RuntimeError(f"upsert failed after {retries} attempts: {last}")


def batched(items: Iterable, size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def language_counts(client: QdrantClient) -> dict[str, int]:
    from src.ingestion.loader import LANGS

    out = {}
    for lang in LANGS:
        res = client.count(
            collection_name=settings.qdrant_collection,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="language", match=models.MatchValue(value=lang))]
            ),
            exact=True,
        )
        if res.count:
            out[lang] = res.count
    return out
