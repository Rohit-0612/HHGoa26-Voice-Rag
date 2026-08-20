"""Sarvam AI speech-to-text.

Sarvam returns BCP-47 language codes (`hi-IN`), while the corpus is indexed
under bare ISO-639-1 (`hi`). The mapping is not a simple truncation -- Sarvam
uses `od` for Odia where the corpus uses `or` -- and Sarvam does not cover every
language in the corpus. Both facts are handled in `to_corpus_language`, and the
failure mode is deliberately "don't filter" rather than "filter wrongly".
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import httpx

from src.config import settings

# Corpus languages, from src/ingestion/loader.LANGS.
CORPUS_LANGS = {"as", "bn", "gu", "hi", "kn", "ml", "mr",
                "ne", "or", "pa", "sa", "ta", "te", "ur"}

# Sarvam-specific spellings that differ from the corpus codes.
_ALIASES = {
    "od": "or",   # Sarvam writes Odia as `od-IN`; the corpus uses `or`
    "ori": "or",
    "asm": "as",
    "nep": "ne",
    "san": "sa",
    "urd": "ur",
}


def to_corpus_language(raw: str | None) -> str | None:
    """BCP-47 from Sarvam -> corpus language code, or None.

    None means "do not scope retrieval". That is the correct fallback for an
    unrecognised or unsupported language: searching all 14 languages returns
    something useful, whereas filtering to the wrong language returns confident
    nonsense. English (`en-IN`) also maps to None -- the corpus has no English.
    """
    if not raw:
        return None
    code = raw.strip().lower().replace("_", "-").split("-")[0]
    if not code or code in {"unknown", "en"}:
        return None
    code = _ALIASES.get(code, code)
    return code if code in CORPUS_LANGS else None


@dataclass
class STTResult:
    transcript: str
    language: str | None          # mapped corpus code, or None => unscoped
    raw_language: str             # exactly what Sarvam returned
    provider: str
    duration_ms: float
    request_id: str = ""


class STTError(RuntimeError):
    pass


class AudioTooLongError(STTError):
    """Sarvam rejected the clip for duration. Distinguished from a generic STT
    failure so the API can answer 413 with a clear message instead of leaking
    the provider's nested error JSON as a 502."""


class SarvamSTT:
    name = "sarvam"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.sarvam_api_key
        if not self.api_key:
            raise RuntimeError("SARVAM_API_KEY missing. Add it to .env.")
        self.model = model or settings.sarvam_model
        self.url = settings.sarvam_stt_url

    async def transcribe(
        self,
        audio: bytes,
        filename: str = "audio.wav",
        language_hint: str | None = None,
        content_type: str = "audio/wav",
    ) -> STTResult:
        if not audio:
            raise STTError("empty audio payload")

        size_mb = len(audio) / (1024 * 1024)
        if size_mb > settings.max_audio_mb:
            raise STTError(
                f"audio is {size_mb:.1f}MB, limit is {settings.max_audio_mb}MB "
                f"(Sarvam's REST endpoint also caps at ~30s of audio)"
            )

        # "unknown" asks Sarvam to auto-detect rather than assume.
        data = {"model": self.model, "language_code": language_hint or "unknown"}
        t0 = time.perf_counter()
        last: Exception | None = None

        for attempt in range(settings.stt_retries):
            try:
                async with httpx.AsyncClient(timeout=60) as http:
                    resp = await http.post(
                        self.url,
                        headers={"api-subscription-key": self.api_key},
                        files={"file": (filename, audio, content_type)},
                        data=data,
                    )
                if resp.status_code == 200:
                    body = resp.json()
                    raw_lang = body.get("language_code") or ""
                    return STTResult(
                        transcript=(body.get("transcript") or "").strip(),
                        language=to_corpus_language(raw_lang),
                        raw_language=raw_lang,
                        provider=self.name,
                        duration_ms=(time.perf_counter() - t0) * 1000,
                        request_id=body.get("request_id", ""),
                    )

                # 4xx other than 429 will not improve on retry.
                if resp.status_code != 429 and resp.status_code < 500:
                    body_l = resp.text.lower()
                    if "duration" in body_l and ("exceed" in body_l or "long" in body_l):
                        raise AudioTooLongError(
                            f"Audio is longer than the {settings.max_audio_seconds:.0f}s "
                            f"limit. Please record a shorter clip."
                        )
                    raise STTError(f"Sarvam rejected the audio ({resp.status_code}).")

                last = STTError(f"Sarvam {resp.status_code}: {resp.text[:200]}")
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
            except STTError:
                raise

            # Bounded backoff -- Day 1 D-20: never sleep on an uncapped hint.
            await asyncio.sleep(min(2 ** attempt, settings.max_backoff_s))

        raise STTError(f"Sarvam failed after {settings.stt_retries} attempts: {last}")
