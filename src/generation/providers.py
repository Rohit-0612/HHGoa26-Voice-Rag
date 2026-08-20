"""LLM providers behind one interface, so the generator does not know or care
which vendor answered.

Groq is primary (fastest). NVIDIA NIM is the fallback and is OpenAI-compatible,
so it reuses the `openai` SDK with a different `base_url` rather than needing a
second client implementation.
"""
from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from src.config import settings


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    async def complete(self, messages: list[dict], force_json: bool = True) -> str:
        ...


class GroqProvider:
    name = "groq"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        from groq import AsyncGroq

        key = api_key or settings.groq_api_key
        if not key:
            raise RuntimeError("GROQ_API_KEY missing")
        self.client = AsyncGroq(api_key=key)
        self.model = model or settings.groq_model

    async def complete(self, messages, force_json: bool = True) -> str:
        return await _openai_style_call(self.client, self.model, messages, force_json)

    async def complete_with_backoff(self, messages, force_json: bool = True) -> str:
        """Rate limiting is a transport problem: it needs waiting, not a
        different prompt. Backoff is bounded (Day 1 D-20) -- Groq can suggest a
        `retry-after` of minutes when a daily quota is gone, and honouring that
        literally stalls the request and everything queued behind it."""
        from groq import RateLimitError

        last: Exception | None = None
        for attempt in range(4):
            try:
                return await self.complete(messages, force_json=force_json)
            except RateLimitError as exc:
                last = exc
                hdrs = getattr(getattr(exc, "response", None), "headers", {}) or {}
                try:
                    suggested = float(hdrs.get("retry-after", 0) or 0)
                except (TypeError, ValueError):
                    suggested = 0.0
                await asyncio.sleep(
                    min(suggested or min(2 ** attempt, 8), settings.max_backoff_s)
                )
        raise last


class NIMProvider:
    """NVIDIA NIM (Nemotron). OpenAI-compatible endpoint."""

    name = "nim"

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None):
        from openai import AsyncOpenAI

        key = api_key or settings.nim_api_key
        if not key:
            raise RuntimeError("NIM_API_KEY missing")
        self.client = AsyncOpenAI(api_key=key, base_url=base_url or settings.nim_base_url)
        self.model = model or settings.nim_model

    async def complete(self, messages, force_json: bool = True) -> str:
        return await _openai_style_call(self.client, self.model, messages, force_json)


async def _openai_style_call(client, model: str, messages, force_json: bool) -> str:
    kwargs = {}
    if force_json:
        kwargs["response_format"] = {"type": "json_object"}
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=settings.max_tokens,
        **kwargs,
    )
    return resp.choices[0].message.content or ""
