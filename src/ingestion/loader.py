"""Load ai4bharat/MSMARCO-XI across ALL 14 Indic languages.

Why this file exists in the shape it does (see decision.md D-01..D-03):

* The dataset exposes a single `default` config -- there are NO per-language
  configs, so `load_dataset("ai4bharat/MSMARCO-XI", "hi")` does not work.
* The repo ships `ms_marco_translations.py`, a loading script that points at
  `train/{lang}train.jsonl`. Those .jsonl files no longer exist (the repo is
  parquet now), so the script is dead on arrival -- and `datasets>=4` refuses
  to execute repo scripts at all.
* `train/` is missing `teltrain.parquet` -- only 13 of 14 languages. The
  `validation/` directory has all 14 and is ~8x smaller.

So we read the validation parquet files directly over the `hf://` filesystem,
streaming, and never materialise the 55.6GB repo.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterator

# 2-letter language code -> parquet file stem. There is no API that exposes
# this mapping; it is derived from the actual file listing in the HF repo.
LANGS: dict[str, str] = {
    "as": "asm",  # Assamese
    "bn": "ben",  # Bengali
    "gu": "guj",  # Gujarati
    "hi": "hin",  # Hindi
    "kn": "kan",  # Kannada
    "ml": "mal",  # Malayalam
    "mr": "mar",  # Marathi
    "ne": "nep",  # Nepali
    "or": "ori",  # Odia
    "pa": "pan",  # Punjabi
    "sa": "san",  # Sanskrit
    "ta": "tam",  # Tamil
    "te": "tel",  # Telugu
    "ur": "urd",  # Urdu
}

LANG_NAMES: dict[str, str] = {
    "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali",
    "or": "Odia", "pa": "Punjabi", "sa": "Sanskrit", "ta": "Tamil",
    "te": "Telugu", "ur": "Urdu",
}

REPO = "ai4bharat/MSMARCO-XI"


def parquet_uri(lang: str) -> str:
    """hf:// URI for a language's validation parquet."""
    return f"hf://datasets/{REPO}/validation/{LANGS[lang]}val.parquet"


@dataclass(frozen=True)
class PassageRecord:
    passage_id: str
    text: str
    language: str
    query_id: str
    query: str
    answer: str
    is_selected: int


def _coerce_passages(passages: object) -> list[tuple[str, int]]:
    """MSMARCO-XI's `passages` arrives either as a dict-of-lists or a
    list-of-dicts depending on the arrow->python conversion. Handle both, and
    fall back to the English passage when a translation is missing."""
    out: list[tuple[str, int]] = []

    if isinstance(passages, dict):
        texts = passages.get("Translated_passages") or passages.get("English_passages") or []
        selected = passages.get("is_selected") or []
        for i, text in enumerate(texts):
            sel = selected[i] if i < len(selected) else 0
            out.append((text, sel))
    elif isinstance(passages, list):
        for p in passages:
            if not isinstance(p, dict):
                continue
            text = p.get("Translated_passages") or p.get("English_passages") or ""
            out.append((text, p.get("is_selected", 0)))

    return [(t.strip(), int(s or 0)) for t, s in out if isinstance(t, str) and t.strip()]


def stream_language(lang: str, limit: int) -> Iterator[PassageRecord]:
    """Yield one PassageRecord per passage for the first `limit` query rows."""
    from datasets import load_dataset

    ds = load_dataset(
        "parquet",
        data_files=parquet_uri(lang),
        split="train",       # a raw parquet load is always called "train"
        streaming=True,      # never pull the whole file
    )

    for row_i, row in enumerate(ds):
        if row_i >= limit:
            break
        query_id = str(row.get("query_id", f"{lang}{row_i}"))
        query = (row.get("query") or "").strip()
        answer = (row.get("Answer") or "").strip()

        for p_i, (text, is_selected) in enumerate(_coerce_passages(row.get("passages"))):
            yield PassageRecord(
                passage_id=f"{lang}::{query_id}::{p_i}",
                text=text,
                language=lang,
                query_id=query_id,
                query=query,
                answer=answer,
                is_selected=is_selected,
            )


def write_jsonl(records: Iterator[PassageRecord], path) -> int:
    n = 0
    with open(path, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
