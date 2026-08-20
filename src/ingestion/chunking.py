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


# --------------------------------------------------------------------------
# Strategy (c): fixed size with overlap  -- Day 2
# --------------------------------------------------------------------------
@register
class FixedSizeChunking(ChunkingStrategy):
    """Fixed character window with overlap. The deliberately dumb baseline.

    Included so the other three strategies have something to beat. If a
    semantic or structural strategy cannot outperform blind character windows,
    its extra complexity is not earning its place.

    Windows are snapped forward to the nearest sentence boundary when one falls
    inside a tolerance band, which avoids slicing mid-word in scripts without
    spaces between all tokens (Malayalam, Tamil) while keeping sizes roughly
    fixed. Overlap exists so an answer spanning a boundary survives in at least
    one chunk.
    """

    name = "fixed_size"

    def __init__(self, chunk_chars: int = 400, overlap_chars: int = 80, snap_window: int = 60):
        if overlap_chars >= chunk_chars:
            raise ValueError("overlap_chars must be smaller than chunk_chars")
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.snap_window = snap_window

    def _snap(self, text: str, end: int) -> int:
        """Nudge a cut point to a sentence terminator within tolerance."""
        if end >= len(text):
            return len(text)
        window = text[end : end + self.snap_window]
        for i, ch in enumerate(window):
            if ch in "।॥۔.!?":
                return end + i + 1
        return end

    def chunk(self, text: str, *, language: str, passage_id: str) -> list[Chunk]:
        text = (text or "").strip()
        if not text:
            return []

        chunks: list[Chunk] = []
        start, idx = 0, 0
        step = self.chunk_chars - self.overlap_chars

        while start < len(text):
            end = self._snap(text, min(start + self.chunk_chars, len(text)))
            piece = text[start:end].strip()
            if piece:
                chunks.append(
                    self._mk(piece, language, passage_id, idx,
                             n_chars=len(piece), start=start, end=end)
                )
                idx += 1
            if end >= len(text):
                break
            start += step

        return chunks or [self._mk(text, language, passage_id, 0, n_chars=len(text))]


# --------------------------------------------------------------------------
# Strategy (d): sentence-window / parent-child  -- Day 2
# --------------------------------------------------------------------------
@register
class SentenceWindowChunking(ChunkingStrategy):
    """Embed one sentence, but carry its neighbours as context.

    The retrieval unit and the generation unit are different things, and this
    strategy separates them. A single sentence embeds to a tight, unambiguous
    vector -- good for matching a specific question -- but a sentence alone is
    usually too thin to answer from. So the *child* (one sentence) is what gets
    embedded and matched, while `meta["parent_text"]` carries a +/-N sentence
    window that is what the LLM should actually read.

    Cost: this is the most expensive strategy to index, roughly one chunk per
    sentence (~4x passage-native on this corpus). That expense is the thing the
    recall@5 eval has to justify.
    """

    name = "sentence_window"

    def __init__(self, window_size: int = 1):
        self.window_size = window_size

    def chunk(self, text: str, *, language: str, passage_id: str) -> list[Chunk]:
        text = (text or "").strip()
        if not text:
            return []

        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return [self._mk(text, language, passage_id, 0,
                             parent_text=text, n_sentences=len(sentences))]

        chunks = []
        for i, sentence in enumerate(sentences):
            lo = max(0, i - self.window_size)
            hi = min(len(sentences), i + self.window_size + 1)
            chunks.append(
                self._mk(
                    sentence, language, passage_id, i,
                    # The LLM reads this; the embedding is of `sentence` alone.
                    parent_text=" ".join(sentences[lo:hi]),
                    window=[lo, hi],
                    n_sentences=len(sentences),
                )
            )
        return chunks
