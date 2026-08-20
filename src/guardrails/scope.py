"""Off-topic detection, in two stages.

MEASURED FINDING (see decision.md D-24): the pre-retrieval centroid check
specified for Day 2 does NOT reliably separate on-topic from off-topic queries
on this corpus. With 14 languages and k=16, k-means finds *language* clusters,
not *topic* clusters, so the score mostly answers "is this one of our scripts?"
A Tamil weather question scored 0.708 -- above the median real query (0.497) --
while a real Tamil query scored 0.340.

So this module ships two gates with different jobs:

1. `ScopeGuard`  (pre-retrieval, cheap, WEAK). Tuned permissively to catch only
   egregious garbage -- Latin-script code questions, gibberish -- with as few
   false rejections as possible. It saves the full pipeline cost when it fires,
   but it is not trusted to be the only defence.

2. `RelevanceGate` (post-retrieval, accurate). Uses the cross-encoder rerank
   score, which is trained on query-document relevance and separates cleanly:
   in-corpus min +0.452 vs off-topic max -0.309. It fires before generation,
   which is 71% of end-to-end latency, so it still avoids most of the cost.

Cheap-and-wrong first, accurate-and-slightly-later second.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import settings

OUT_OF_SCOPE_ANSWER = (
    "This question appears to be outside the scope of the indexed corpus, "
    "so no grounded answer can be given."
)


@dataclass
class ScopeVerdict:
    in_scope: bool
    score: float
    threshold: float
    stage: str          # "centroid" | "rerank" | "disabled"
    reason: str = ""


class ScopeGuard:
    """Pre-retrieval centroid check. Weak by measurement -- see module docstring."""

    def __init__(self, path=None, threshold: float | None = None):
        path = path or settings.centroids_path
        self.enabled = False
        self.centroids: np.ndarray | None = None
        self.threshold = 0.0

        try:
            z = np.load(path, allow_pickle=True)
        except Exception:
            return  # no centroids built -> guard disabled, pipeline still works

        self.centroids = np.asarray(z["centroids"], dtype=np.float32)
        if threshold is not None:
            self.threshold = float(threshold)
        elif settings.scope_threshold is not None:
            self.threshold = float(settings.scope_threshold)
        else:
            self.threshold = float(z["threshold"])
        self.enabled = True

    def check(self, query_vec) -> ScopeVerdict:
        if not self.enabled or self.centroids is None:
            return ScopeVerdict(True, 1.0, self.threshold, "disabled")

        v = np.asarray(query_vec, dtype=np.float32)
        n = float(np.linalg.norm(v))
        v = v / n if n else v
        score = float(np.max(v @ self.centroids.T))
        ok = score >= self.threshold
        return ScopeVerdict(
            ok, score, self.threshold, "centroid",
            "" if ok else "query is far from every corpus centroid",
        )


class RelevanceGate:
    """Post-retrieval gate on the top cross-encoder score.

    This is the gate that actually works. Threshold defaults to 0.0, which sat
    cleanly between in-corpus (min +0.452) and off-topic (max -0.309) on the
    measured sample.
    """

    def __init__(self, threshold: float | None = None):
        self.threshold = (
            threshold if threshold is not None else settings.relevance_threshold
        )

    def check(self, chunks) -> ScopeVerdict:
        scores = [c.rerank_score for c in chunks if getattr(c, "rerank_score", None) is not None]
        if not scores:
            # No reranker (or no results) -- do not block on a signal we lack.
            return ScopeVerdict(bool(chunks), 0.0, self.threshold, "disabled")

        top = max(scores)
        ok = top >= self.threshold
        return ScopeVerdict(
            ok, float(top), self.threshold, "rerank",
            "" if ok else "no retrieved passage is relevant to the question",
        )
