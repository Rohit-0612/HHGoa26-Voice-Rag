"""Embedding provider.

The retriever and indexer talk to `EmbeddingProvider`, never to a model name.
That indirection is deliberate: the brief specified BAAI/bge-m3, but FastEmbed
does not ship bge-m3 for dense, sparse, OR reranking (verified against
`list_supported_models()` in fastembed 0.8.0). Swapping to a torch/FlagEmbedding
bge-m3 backend on Day 2 means adding one class here and changing .env --
nothing downstream moves. See decision.md D-06.
"""
from __future__ import annotations

from typing import Protocol, Sequence

from qdrant_client import models

from src.config import settings


class EmbeddingProvider(Protocol):
    dense_dim: int

    def embed_documents(self, texts: Sequence[str]) -> tuple[list[list[float]], list[models.SparseVector]]:
        ...

    def embed_query(self, text: str) -> tuple[list[float], models.SparseVector]:
        ...

    def embed_dense_only(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class FastEmbedProvider:
    """Dense (multilingual-e5-large) + sparse (BM25) via FastEmbed / ONNX.

    Models are loaded once and reused; instantiation downloads ~2GB on first
    run, so callers must construct this at process/app startup, never inside a
    request handler.
    """

    def __init__(
        self,
        dense_model: str | None = None,
        sparse_model: str | None = None,
    ) -> None:
        from fastembed import SparseTextEmbedding, TextEmbedding

        self.dense_model_name = dense_model or settings.dense_model
        self.sparse_model_name = sparse_model or settings.sparse_model

        self.dense = TextEmbedding(self.dense_model_name)
        self.sparse = SparseTextEmbedding(self.sparse_model_name)
        self.dense_dim = settings.dense_dim

    # -- e5 models are trained with these prefixes; omitting them measurably
    # -- degrades retrieval, and the asymmetry (query: vs passage:) matters.
    def _is_e5(self) -> bool:
        return "e5" in self.dense_model_name.lower()

    def _doc(self, text: str) -> str:
        return f"passage: {text}" if self._is_e5() else text

    def _query(self, text: str) -> str:
        return f"query: {text}" if self._is_e5() else text

    @staticmethod
    def _to_sparse(vec) -> models.SparseVector:
        return models.SparseVector(indices=vec.indices.tolist(), values=vec.values.tolist())

    def embed_documents(self, texts: Sequence[str]):
        prepped = [self._doc(t) for t in texts]
        dense = [v.tolist() for v in self.dense.embed(prepped)]
        # BM25 term statistics come from the raw text, not the e5-prefixed form.
        sparse = [self._to_sparse(v) for v in self.sparse.embed(list(texts))]
        return dense, sparse

    def embed_query(self, text: str):
        dense = next(iter(self.dense.query_embed(self._query(text)))).tolist()
        sparse = self._to_sparse(next(iter(self.sparse.query_embed(text))))
        return dense, sparse

    def embed_dense_only(self, texts: Sequence[str]) -> list[list[float]]:
        """Used by SemanticChunking for sentence-level breakpoint detection."""
        return [v.tolist() for v in self.dense.embed([self._doc(t) for t in texts])]
