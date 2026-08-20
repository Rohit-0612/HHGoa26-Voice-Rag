"""Post-generation groundedness, by embedding overlap. No extra LLM call.

Each answer sentence is embedded and compared against the retrieved chunks.
A sentence with no chunk above threshold is unsupported by the context and is
removed. If what remains is too thin, the whole answer is replaced.

Sentence splitting reuses `split_sentences` from the chunking module, which is
danda/`۔`-aware. That matters more here than anywhere: a naive `.split(".")`
returns ONE sentence for most Indic answers, which would make this check
vacuous -- it would score the whole answer as a single unit and never strip
anything, while appearing to work.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from src.config import settings
from src.ingestion.chunking import split_sentences

INSUFFICIENT = "Insufficient grounded information to answer from the retrieved context."


@dataclass
class GroundednessReport:
    score: float                       # mean over kept sentences (0..1)
    min_sentence_score: float
    kept: int
    dropped: int
    replaced: bool
    per_sentence: list[dict] = field(default_factory=list)
    duration_ms: float = 0.0


def _unit(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(n, 1e-9)


class GroundednessChecker:
    def __init__(self, provider, sentence_threshold=None, answer_threshold=None):
        self.provider = provider
        self.sentence_threshold = (
            sentence_threshold if sentence_threshold is not None
            else settings.groundedness_sentence_threshold
        )
        self.answer_threshold = (
            answer_threshold if answer_threshold is not None
            else settings.groundedness_answer_threshold
        )

    def check(self, answer: str, chunks) -> tuple[str, GroundednessReport]:
        t0 = time.perf_counter()
        answer = (answer or "").strip()

        if not answer or not chunks:
            return answer, GroundednessReport(0.0, 0.0, 0, 0, False,
                                              duration_ms=(time.perf_counter() - t0) * 1000)

        sentences = split_sentences(answer)
        if not sentences:
            return answer, GroundednessReport(0.0, 0.0, 0, 0, False,
                                              duration_ms=(time.perf_counter() - t0) * 1000)

        # Compare against what the model was actually shown: for sentence_window
        # that is the parent window, not the single embedded sentence.
        contexts = [getattr(c, "context_text", None) or c.text for c in chunks]

        vecs = _unit(np.asarray(self.provider.embed_dense_only(sentences + contexts),
                                dtype=np.float32))
        sent_v, ctx_v = vecs[: len(sentences)], vecs[len(sentences):]
        best = (sent_v @ ctx_v.T).max(axis=1)   # best-matching chunk per sentence

        per_sentence, kept_text, kept_scores = [], [], []
        for sentence, score in zip(sentences, best):
            score = float(score)
            supported = score >= self.sentence_threshold
            per_sentence.append({"sentence": sentence[:160], "score": round(score, 4),
                                 "kept": supported})
            if supported:
                kept_text.append(sentence)
                kept_scores.append(score)

        mean_score = float(np.mean(kept_scores)) if kept_scores else 0.0
        replaced = (not kept_text) or (mean_score < self.answer_threshold)

        final = INSUFFICIENT if replaced else " ".join(kept_text)

        return final, GroundednessReport(
            score=round(mean_score, 4),
            min_sentence_score=round(float(best.min()), 4),
            kept=len(kept_text),
            dropped=len(sentences) - len(kept_text),
            replaced=replaced,
            per_sentence=per_sentence,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
