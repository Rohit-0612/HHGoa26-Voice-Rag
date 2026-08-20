from __future__ import annotations

import math

import pytest

from src.ingestion.chunking import (
    REGISTRY,
    PassageNativeChunking,
    SemanticChunking,
    get_strategy,
    split_sentences,
)

HINDI = "भारत एक देश है। इसकी राजधानी नई दिल्ली है। क्रिकेट यहाँ लोकप्रिय है।"
URDU = "پاکستان ایک ملک ہے۔ اس کا دارالحکومت اسلام آباد ہے۔"


def test_registry_has_both_day1_strategies():
    assert {"passage_native", "semantic"} <= set(REGISTRY)


def test_get_strategy_roundtrip_and_unknown():
    assert isinstance(get_strategy("passage_native"), PassageNativeChunking)
    with pytest.raises(KeyError):
        get_strategy("does_not_exist")


def test_passage_native_is_identity():
    chunks = PassageNativeChunking().chunk(HINDI, language="hi", passage_id="hi::1::0")
    assert len(chunks) == 1
    assert chunks[0].text == HINDI
    assert chunks[0].strategy == "passage_native"
    assert chunks[0].source_passage_id == "hi::1::0"


def test_passage_native_drops_empty():
    assert PassageNativeChunking().chunk("   ", language="hi", passage_id="x") == []


def test_chunk_ids_are_deterministic():
    a = PassageNativeChunking().chunk(HINDI, language="hi", passage_id="hi::1::0")
    b = PassageNativeChunking().chunk(HINDI, language="hi", passage_id="hi::1::0")
    assert a[0].chunk_id == b[0].chunk_id


# --- the detail that makes semantic chunking work at all on Indic scripts ---
def test_danda_splits_hindi():
    assert len(split_sentences(HINDI)) == 3


def test_arabic_full_stop_splits_urdu():
    assert len(split_sentences(URDU)) == 2


def test_latin_punctuation_still_works():
    assert len(split_sentences("One. Two! Three?")) == 3


def test_split_never_returns_empty_for_unpunctuated_text():
    assert split_sentences("no terminator here") == ["no terminator here"]


def test_semantic_falls_back_to_single_chunk_without_embedder():
    chunks = SemanticChunking(embed_fn=None).chunk(HINDI, language="hi", passage_id="p")
    assert len(chunks) == 1
    assert chunks[0].meta["fallback"] is True


def test_semantic_splits_on_topic_shift():
    """Two tight clusters of sentences should break into two chunks."""
    text = ("A" * 60 + "। ") + ("A" * 60 + "। ") + ("B" * 60 + "। ") + ("B" * 60 + "।")

    def embed_fn(sents):
        return [[1.0, 0.0] if s.startswith("A") else [0.0, 1.0] for s in sents]

    chunks = SemanticChunking(embed_fn=embed_fn, percentile=50.0, min_chars=50).chunk(
        text, language="hi", passage_id="p"
    )
    assert len(chunks) == 2
    assert all(c.strategy == "semantic" for c in chunks)
    assert {c.chunk_id for c in chunks} == {"p::semantic::0", "p::semantic::1"}


def test_semantic_respects_min_chars():
    """A breakpoint is ignored while the current chunk is still too short."""
    text = "a। b। c। " + "D" * 200 + "।"

    def embed_fn(sents):
        return [[float(i % 2), float((i + 1) % 2)] for i, _ in enumerate(sents)]

    chunks = SemanticChunking(embed_fn=embed_fn, percentile=10.0, min_chars=150).chunk(
        text, language="hi", passage_id="p"
    )
    assert len(chunks) == 1
