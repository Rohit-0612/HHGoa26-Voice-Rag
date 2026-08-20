"""Hybrid retrieval: dense + sparse, server-side RRF fusion, then rerank.

One round trip to Qdrant does both branches and the fusion (decision.md D-08);
reranking then happens locally on the fused candidates.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from qdrant_client import models

from src.config import settings
from src.retrieval.indexer import DENSE, SPARSE


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    language: str
    strategy: str
    fusion_score: float
    rerank_score: float | None = None


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    timing: dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    def __init__(self, client, provider, reranker=None):
        self.client = client
        self.provider = provider
        self.reranker = reranker

    # ---------------- filters ----------------
    @staticmethod
    def _filter(language: str | None, strategy: str | None) -> models.Filter | None:
        must = []
        if language:
            must.append(
                models.FieldCondition(key="language", match=models.MatchValue(value=language))
            )
        if strategy:
            must.append(
                models.FieldCondition(key="strategy", match=models.MatchValue(value=strategy))
            )
        return models.Filter(must=must) if must else None

    # ---------------- sync core (CPU bound) ----------------
    def _search(self, text, language, strategy, candidates) -> tuple[list[RetrievedChunk], dict]:
        timing = {}

        t0 = time.perf_counter()
        dense_vec, sparse_vec = self.provider.embed_query(text)
        timing["embed_ms"] = (time.perf_counter() - t0) * 1000

        flt = self._filter(language, strategy)

        t0 = time.perf_counter()
        resp = self.client.query_points(
            collection_name=settings.qdrant_collection,
            prefetch=[
                models.Prefetch(query=dense_vec, using=DENSE, limit=candidates, filter=flt),
                models.Prefetch(query=sparse_vec, using=SPARSE, limit=candidates, filter=flt),
            ],
            # Reciprocal Rank Fusion: rank-based, so it needs no score
            # normalisation between cosine and BM25 -- which live on totally
            # different scales.
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=candidates,
            with_payload=True,
        )
        timing["search_ms"] = (time.perf_counter() - t0) * 1000

        chunks = [
            RetrievedChunk(
                chunk_id=p.payload.get("chunk_id", str(p.id)),
                text=p.payload.get("text", ""),
                language=p.payload.get("language", ""),
                strategy=p.payload.get("strategy", ""),
                fusion_score=float(p.score),
            )
            for p in resp.points
        ]
        return chunks, timing

    def _rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int):
        if not chunks:
            return chunks, 0.0
        t0 = time.perf_counter()
        if self.reranker is not None:
            scores = list(self.reranker.rerank(query, [c.text for c in chunks]))
            for chunk, score in zip(chunks, scores):
                chunk.rerank_score = float(score)
            chunks.sort(key=lambda c: c.rerank_score, reverse=True)
        return chunks[:top_k], (time.perf_counter() - t0) * 1000

    def retrieve_sync(
        self, text: str, language=None, strategy=None, top_k=None, candidates=None
    ) -> RetrievalResult:
        top_k = top_k or settings.top_k
        candidates = candidates or settings.retrieve_candidates

        chunks, timing = self._search(text, language, strategy, candidates)
        top, rerank_ms = self._rerank(text, chunks, top_k)
        timing["rerank_ms"] = rerank_ms
        timing["retrieval_ms"] = sum(
            timing.get(k, 0.0) for k in ("embed_ms", "search_ms", "rerank_ms")
        )
        return RetrievalResult(chunks=top, timing=timing)

    # ---------------- async wrapper ----------------
    async def retrieve(self, text, language=None, strategy=None, top_k=None, candidates=None):
        """FastEmbed and the cross-encoder are sync and CPU-bound; keep them
        off the event loop so concurrent requests are not serialised."""
        return await asyncio.to_thread(
            self.retrieve_sync, text, language, strategy, top_k, candidates
        )


class FastEmbedReranker:
    """bge-reranker-v2-m3 is not available in FastEmbed; this is the
    multilingual cross-encoder that is (decision.md D-06)."""

    def __init__(self, model_name: str | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name or settings.rerank_model
        self.model = TextCrossEncoder(self.model_name)

    def rerank(self, query: str, documents: list[str]):
        return self.model.rerank(query, documents)
