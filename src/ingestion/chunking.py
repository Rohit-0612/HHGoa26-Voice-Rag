"""Pluggable chunking strategies.

Day 1 ships two strategies. Day 2 adds two more by subclassing
`ChunkingStrategy` and applying `@register` -- no change to this interface,
to the indexer, or to the retriever (decision.md D-05).
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    language: str
    source_passage_id: str
    strategy: str
    meta: dict = field(default_factory=dict)


REGISTRY: dict[str, type["ChunkingStrategy"]] = {}


def register(cls: type["ChunkingStrategy"]) -> type["ChunkingStrategy"]:
    REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str, **kwargs) -> "ChunkingStrategy":
    if name not in REGISTRY:
        raise KeyError(f"Unknown chunking strategy {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


class ChunkingStrategy(ABC):
    """Turn one source passage into >=1 chunks.

    Implementations must be deterministic: `chunk_id` is used as the Qdrant
    point id (via UUID5), so re-running the indexer must not create duplicates.
    """

    name: str = "base"

    @abstractmethod
    def chunk(self, text: str, *, language: str, passage_id: str) -> list[Chunk]:
        ...

    def chunk_many(self, records: Iterable[dict]) -> list[Chunk]:
        out: list[Chunk] = []
        for rec in records:
            out.extend(
                self.chunk(
                    rec["text"],
                    language=rec["language"],
                    passage_id=rec["passage_id"],
                )
            )
        return out

    def _mk(self, text: str, language: str, passage_id: str, idx: int, **meta) -> Chunk:
        return Chunk(
            chunk_id=f"{passage_id}::{self.name}::{idx}",
            text=text,
            language=language,
            source_passage_id=passage_id,
            strategy=self.name,
            meta=meta,
        )


# --------------------------------------------------------------------------
# Strategy (a): passage-native
# --------------------------------------------------------------------------
@register
class PassageNativeChunking(ChunkingStrategy):
    """Trust MSMARCO's own passage boundaries -- one chunk per passage.

    This is the honest baseline: MSMARCO passages are already human-curated
    retrieval units of roughly the right size, so any fancier strategy has to
    beat this to justify itself.
    """

    name = "passage_native"

    def chunk(self, text: str, *, language: str, passage_id: str) -> list[Chunk]:
        text = (text or "").strip()
        if not text:
            return []
        return [self._mk(text, language, passage_id, 0, n_chars=len(text))]


# --------------------------------------------------------------------------
# Sentence splitting -- script aware
# --------------------------------------------------------------------------
# Indic scripts terminate sentences with the danda (U+0964) and double danda
# (U+0965); Urdu uses the Arabic full stop (U+06D4). A naive `.split(".")`
# yields exactly ONE chunk for most of these languages, silently turning
# semantic chunking into a no-op. This is the single most important detail in
# this module.
_SENT_END = re.compile(r"(?<=[।॥۔?!.])\s+")


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_END.split(text or "") if s.strip()]
    return parts or ([text.strip()] if text and text.strip() else [])


# --------------------------------------------------------------------------
# Strategy (b): semantic
# --------------------------------------------------------------------------
@register
class SemanticChunking(ChunkingStrategy):
    """Break where consecutive sentences drift apart in embedding space.

    Sentences are embedded, cosine distance is computed between each adjacent
    pair, and a breakpoint is placed wherever that distance exceeds the Nth
    percentile of distances within the passage. Using a *relative* percentile
    rather than an absolute threshold keeps the strategy language-agnostic --
    absolute cosine distances differ systematically across scripts, so a fixed
    cutoff would over-split some languages and under-split others.
    """

    name = "semantic"

    def __init__(
        self,
        embed_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
        percentile: float = 90.0,
        min_chars: int = 120,
    ) -> None:
        self.embed_fn = embed_fn
        self.percentile = percentile
        self.min_chars = min_chars

    def chunk(self, text: str, *, language: str, passage_id: str) -> list[Chunk]:
        text = (text or "").strip()
        if not text:
            return []

        sentences = split_sentences(text)
        if len(sentences) < 2 or self.embed_fn is None:
            return [self._mk(text, language, passage_id, 0, n_sentences=len(sentences),
                             fallback=True)]

        groups = self._group(sentences)
        return [
            self._mk(" ".join(g), language, passage_id, i, n_sentences=len(g))
            for i, g in enumerate(groups)
        ]

    def _group(self, sentences: list[str]) -> list[list[str]]:
        import numpy as np

        vecs = np.asarray(list(self.embed_fn(sentences)), dtype=np.float32)
        if vecs.ndim != 2 or len(vecs) != len(sentences):
            return [sentences]

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        unit = vecs / norms
        sims = np.sum(unit[:-1] * unit[1:], axis=1)
        distances = 1.0 - sims

        threshold = float(np.percentile(distances, self.percentile))

        groups: list[list[str]] = []
        current: list[str] = [sentences[0]]
        for i, dist in enumerate(distances):
            # Only honour a breakpoint once the current chunk is big enough to
            # stand on its own -- otherwise we emit single-clause fragments
            # that retrieve poorly.
            if dist > threshold and len(" ".join(current)) >= self.min_chars:
                groups.append(current)
                current = []
            current.append(sentences[i + 1])
        if current:
            groups.append(current)
        return groups
