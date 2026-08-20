from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.generation.generator import _extract_json, build_context
from src.generation.schemas import RAGAnswer


class _C:
    def __init__(self, cid, text="t", lang="hi"):
        self.chunk_id, self.text, self.language = cid, text, lang


# ---------- confidence coercion: models are inconsistent about this ----------
@pytest.mark.parametrize(
    "raw,expected",
    [(0.85, 0.85), ("0.85", 0.85), ("85%", 0.85), (85, 0.85), (1, 1.0), (0, 0.0)],
)
def test_confidence_coercion(raw, expected):
    assert RAGAnswer(answer="a", citations=[], confidence=raw).confidence == pytest.approx(expected)


def test_confidence_is_clamped():
    assert RAGAnswer(answer="a", citations=[], confidence=250).confidence == 1.0


def test_confidence_rejects_garbage():
    with pytest.raises(ValidationError):
        RAGAnswer(answer="a", citations=[], confidence="very high")


# ---------- citations ----------
def test_citations_accepts_bare_string():
    assert RAGAnswer(answer="a", citations="c1", confidence=0.5).citations == ["c1"]


def test_citations_none_becomes_empty():
    assert RAGAnswer(answer="a", citations=None, confidence=0.5).citations == []


def test_hallucinated_citations_are_dropped():
    ans = RAGAnswer(answer="a", citations=["real", "invented"], confidence=0.9)
    out = ans.drop_hallucinated_citations({"real"})
    assert out.citations == ["real"]
    assert out.confidence == 0.9  # at least one real cite -> confidence untouched


def test_confidence_capped_when_every_citation_is_fabricated():
    ans = RAGAnswer(answer="a", citations=["nope1", "nope2"], confidence=0.95)
    out = ans.drop_hallucinated_citations({"real"})
    assert out.citations == []
    assert out.confidence == 0.3


def test_no_citations_offered_leaves_confidence_alone():
    ans = RAGAnswer(answer="a", citations=[], confidence=0.8)
    assert ans.drop_hallucinated_citations({"real"}).confidence == 0.8


# ---------- JSON extraction, so a fence doesn't burn a retry ----------
def test_extract_plain_json():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_from_markdown_fence():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_from_surrounding_prose():
    assert _extract_json('Sure! {"a": 1} hope that helps') == {"a": 1}


def test_extract_raises_on_unrecoverable():
    with pytest.raises(Exception):
        _extract_json("no json here at all")


# ---------- context assembly ----------
def test_build_context_exposes_chunk_ids():
    ctx = build_context([_C("hi::1::0"), _C("hi::2::0")])
    assert "chunk_id: hi::1::0" in ctx and "chunk_id: hi::2::0" in ctx
    assert "[1]" in ctx and "[2]" in ctx


# ---- regression: models echo the context label instead of the bare id ----
@pytest.mark.parametrize(
    "emitted", ["chunk_id: hi::1::0", "  chunk_id:hi::1::0 ", "[hi::1::0]", "(hi::1::0)", "hi::1::0"]
)
def test_citation_label_prefix_is_normalised(emitted):
    """gpt-oss-20b emits `chunk_id: hi::1::0` verbatim from the context block.
    Without normalisation every citation looks fabricated and gets dropped."""
    ans = RAGAnswer(answer="a", citations=[emitted], confidence=0.9)
    out = ans.drop_hallucinated_citations({"hi::1::0"})
    assert out.citations == ["hi::1::0"]
    assert out.confidence == 0.9


def test_normalisation_does_not_rescue_a_genuinely_fake_id():
    ans = RAGAnswer(answer="a", citations=["chunk_id: hi::999::0"], confidence=0.9)
    out = ans.drop_hallucinated_citations({"hi::1::0"})
    assert out.citations == []
    assert out.confidence == 0.3


# ---- regression: provider errors must not escape as 500s ----
async def test_provider_error_falls_back_instead_of_raising():
    """groq.BadRequestError(json_validate_failed) is exactly what the retry
    exists for; before this fix it propagated and 500'd /query."""
    from src.generation.generator import GroqGenerator

    gen = GroqGenerator.__new__(GroqGenerator)
    gen.model = "test"

    calls = []

    async def boom(messages, force_json=True):
        calls.append(force_json)
        raise RuntimeError("Error code: 400 - json_validate_failed")

    gen._call = boom
    ans, ms, retry = await gen.generate("q", [_C("hi::1::0")])

    assert retry is True
    assert ans.confidence == 0.0          # degraded, not raised
    assert ans.citations == []
    assert calls == [True, False]         # retry drops the JSON constraint


async def test_retry_recovers_and_keeps_valid_citation():
    from src.generation.generator import GroqGenerator

    gen = GroqGenerator.__new__(GroqGenerator)
    gen.model = "test"
    state = {"n": 0}

    async def flaky(messages, force_json=True):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("400 json_validate_failed")
        return '{"answer":"ok","citations":["chunk_id: hi::1::0"],"confidence":0.9}'

    gen._call = flaky
    ans, ms, retry = await gen.generate("q", [_C("hi::1::0")])
    assert retry is True
    assert ans.answer == "ok"
    assert ans.citations == ["hi::1::0"]
    assert ans.confidence == 0.9


async def test_rate_limit_is_retried_with_backoff_not_a_sterner_prompt():
    """A 429 is a transport problem. Re-prompting cannot fix it; waiting can."""
    import groq
    import httpx

    from src.generation.generator import GroqGenerator

    gen = GroqGenerator.__new__(GroqGenerator)
    gen.model = "test"
    state = {"n": 0}

    def _rate_limit_error():
        req = httpx.Request("POST", "https://api.groq.com/v1/chat/completions")
        resp = httpx.Response(429, headers={"retry-after": "0"}, request=req)
        return groq.RateLimitError("rate limited", response=resp, body=None)

    async def flaky(messages, force_json=True):
        state["n"] += 1
        if state["n"] <= 2:
            raise _rate_limit_error()
        return '{"answer":"ok","citations":["hi::1::0"],"confidence":0.8}'

    gen._call = flaky
    ans, ms, retry = await gen.generate("q", [_C("hi::1::0")])

    assert state["n"] == 3            # two 429s waited out, third succeeded
    assert retry is False             # backoff is not the parse-failure retry
    assert ans.answer == "ok"
    assert ans.confidence == 0.8
