"""Day 2: STT language mapping, circuit breaker, provider fallback, guardrails."""
from __future__ import annotations

import numpy as np
import pytest

from src.generation.breaker import CLOSED, HALF_OPEN, OPEN, CircuitBreaker
from src.generation.generator import ChainedGenerator
from src.guardrails.scope import RelevanceGate, ScopeGuard
from src.stt.sarvam import to_corpus_language


class _Chunk:
    def __init__(self, cid="hi::1::0", text="ctx", rerank=None):
        self.chunk_id, self.text, self.context_text = cid, text, text
        self.rerank_score = rerank
        self.language, self.strategy = "hi", "passage_native"


# ---------------------------------------------------------------- STT mapping
@pytest.mark.parametrize("raw,expected", [
    ("hi-IN", "hi"), ("bn-IN", "bn"), ("ta-IN", "ta"), ("sa", "sa"), ("UR-in", "ur"),
])
def test_known_languages_map(raw, expected):
    assert to_corpus_language(raw) == expected


@pytest.mark.parametrize("raw", ["od-IN", "or-IN", "ori"])
def test_odia_spellings_both_map_to_corpus_code(raw):
    """Sarvam writes Odia as `od`; the corpus indexes it as `or`. Getting this
    wrong sends every Odia query to an empty language filter."""
    assert to_corpus_language(raw) == "or"


@pytest.mark.parametrize("raw", [None, "", "unknown", "en-IN", "zz-IN", "fr-FR"])
def test_unmappable_languages_return_none_not_a_guess(raw):
    """None means 'search all languages'. Guessing a wrong filter would return
    confident answers from the wrong language."""
    assert to_corpus_language(raw) is None


# ------------------------------------------------------------ circuit breaker
def test_breaker_opens_after_threshold_and_blocks():
    b = CircuitBreaker("x", failure_threshold=2, reset_after_s=99)
    assert b.state == CLOSED and b.allows()
    b.record_failure()
    assert b.state == CLOSED
    b.record_failure()
    assert b.state == OPEN and not b.allows()


def test_breaker_half_opens_after_cooldown_then_closes_on_success():
    b = CircuitBreaker("x", failure_threshold=1, reset_after_s=0.05)
    b.record_failure()
    assert b.state == OPEN
    import time as _t; _t.sleep(0.08)
    assert b.state == HALF_OPEN and b.allows()
    b.record_success()
    assert b.state == CLOSED and b.consecutive_failures == 0


def test_success_resets_consecutive_failures():
    b = CircuitBreaker("x", failure_threshold=3)
    b.record_failure(); b.record_failure()
    b.record_success()
    b.record_failure()
    assert b.state == CLOSED  # counter restarted, not at 3


# ---------------------------------------------------------- provider fallback
class _Provider:
    def __init__(self, name, reply=None, fail=False):
        self.name, self.model, self._reply, self._fail = name, f"{name}-model", reply, fail
        self.calls = 0

    async def complete(self, messages, force_json=True):
        self.calls += 1
        if self._fail:
            raise RuntimeError(f"{self.name} is down")
        return self._reply


GOOD = '{"answer":"ok","citations":["hi::1::0"],"confidence":0.9}'


async def test_falls_back_to_second_provider_when_first_fails():
    primary = _Provider("groq", fail=True)
    fallback = _Provider("nim", reply=GOOD)
    gen = ChainedGenerator([primary, fallback])

    out = await gen.generate_detailed("q", [_Chunk()])
    assert out["provider"] == "nim"
    assert out["answer"].answer == "ok"
    assert out["answer"].citations == ["hi::1::0"]
    assert any("groq" in e for e in out["errors"])


async def test_primary_is_used_when_healthy_and_fallback_untouched():
    primary = _Provider("groq", reply=GOOD)
    fallback = _Provider("nim", reply=GOOD)
    out = await ChainedGenerator([primary, fallback]).generate_detailed("q", [_Chunk()])
    assert out["provider"] == "groq"
    assert fallback.calls == 0


async def test_open_circuit_skips_provider_entirely():
    primary = _Provider("groq", fail=True)
    fallback = _Provider("nim", reply=GOOD)
    breakers = {"groq": CircuitBreaker("groq", failure_threshold=1, reset_after_s=99),
                "nim": CircuitBreaker("nim")}
    gen = ChainedGenerator([primary, fallback], breakers)

    await gen.generate_detailed("q", [_Chunk()])   # opens groq's circuit
    calls_after_first = primary.calls
    out = await gen.generate_detailed("q", [_Chunk()])

    assert primary.calls == calls_after_first, "open circuit must not be called again"
    assert "groq" in out["skipped"]
    assert out["provider"] == "nim"


async def test_all_providers_down_degrades_instead_of_raising():
    gen = ChainedGenerator([_Provider("groq", fail=True), _Provider("nim", fail=True)])
    out = await gen.generate_detailed("q", [_Chunk()])
    assert out["provider"] == "none"
    assert out["answer"].confidence == 0.0
    assert len(out["errors"]) == 2


# ------------------------------------------------------------------ guardrails
def test_relevance_gate_blocks_low_rerank_scores():
    gate = RelevanceGate(threshold=0.0)
    v = gate.check([_Chunk(rerank=-2.1), _Chunk(rerank=-1.5)])
    assert not v.in_scope and v.stage == "rerank"


def test_relevance_gate_passes_high_rerank_scores():
    assert RelevanceGate(threshold=0.0).check([_Chunk(rerank=0.6)]).in_scope


def test_relevance_gate_does_not_block_when_no_scores_available():
    """Without a reranker there is no signal; blocking would be guessing."""
    v = RelevanceGate(threshold=0.0).check([_Chunk(rerank=None)])
    assert v.in_scope and v.stage == "disabled"


def test_scope_guard_disabled_when_centroids_missing():
    g = ScopeGuard(path="/nonexistent/centroids.npz")
    assert not g.enabled
    assert g.check(np.zeros(384)).in_scope, "a missing artifact must not block traffic"


def test_scope_guard_rejects_orthogonal_vector():
    import tempfile, os
    c = np.eye(2, 4, dtype=np.float32)  # centroids along axes 0 and 1
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as fh:
        np.savez(fh.name, centroids=c, threshold=np.float32(0.5))
        path = fh.name
    try:
        g = ScopeGuard(path=path)
        assert g.check(np.array([1, 0, 0, 0], dtype=np.float32)).in_scope
        assert not g.check(np.array([0, 0, 0, 1], dtype=np.float32)).in_scope
    finally:
        os.unlink(path)
