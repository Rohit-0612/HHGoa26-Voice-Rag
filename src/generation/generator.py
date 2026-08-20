"""Groq generation with enforced structured JSON output.

Note on the model: the brief specified `llama-3.1-8b-instant`, which Groq shut
down on 2026-08-16. `qwen/qwen3.6-27b` is the replacement chosen for its Indic
coverage -- see decision.md D-09.
"""
from __future__ import annotations

import json
import time

from pydantic import ValidationError

from src.config import settings
from src.generation.schemas import RAGAnswer

SYSTEM = """You are a multilingual question-answering assistant for Indic languages.

RULES:
1. Answer ONLY from the numbered CONTEXT passages. Never use outside knowledge.
2. Answer in the SAME LANGUAGE and SAME SCRIPT as the user's question.
3. Cite the chunk_id of every passage you actually used.
4. If the context does not contain the answer, say so in the user's language and return confidence below 0.3.
5. Reply with a single JSON object and nothing else.

SCHEMA:
{"answer": "<string>", "citations": ["<chunk_id>", ...], "confidence": <float 0.0-1.0>}"""

RETRY_SYSTEM = """Your previous reply could not be parsed as valid JSON.

Return ONE JSON object. No markdown fences, no commentary, no trailing text.
Exactly these three keys:
  "answer"     : string, in the same language as the question
  "citations"  : array of chunk_id strings copied verbatim from the context
  "confidence" : number between 0.0 and 1.0

Parse error was: {error}"""


def build_context(chunks) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[{i}] chunk_id: {c.chunk_id}\nlanguage: {c.language}\n{c.text}")
    return "\n\n".join(blocks)


def _extract_json(raw: str) -> dict:
    """Models wrap JSON in prose or ``` fences often enough to be worth handling
    before spending a whole retry round trip on it."""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1] if "```" in raw[3:] else raw.strip("`")
        raw = raw.removeprefix("json").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


class GroqGenerator:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        from groq import AsyncGroq

        key = api_key or settings.groq_api_key
        if not key:
            raise RuntimeError("GROQ_API_KEY missing. Fill it into .env.")
        self.client = AsyncGroq(api_key=key)
        self.model = model or settings.groq_model

    async def _call_with_backoff(self, messages, force_json: bool = True, retries: int = 4) -> str:
        """Rate limiting is a *transport* problem, not a formatting problem.

        Retrying it with a sterner prompt (the parse-failure path) is useless --
        it needs waiting. Groq reports the wait in `retry_after`; honour it when
        present and fall back to exponential backoff when not.
        """
        import asyncio as _a

        from groq import RateLimitError

        last: Exception | None = None
        for attempt in range(retries):
            try:
                return await self._call(messages, force_json=force_json)
            except RateLimitError as exc:
                last = exc
                wait = getattr(getattr(exc, "response", None), "headers", {}) or {}
                delay = float(wait.get("retry-after", 0) or 0) or min(2 ** attempt, 8)
                await _a.sleep(delay)
        raise last

    async def _call(self, messages, force_json: bool = True) -> str:
        """`force_json` toggles Groq's server-side JSON validation.

        That validator can itself 400 with `json_validate_failed` and an empty
        `failed_generation` -- typically when a reasoning model spends its whole
        budget thinking and emits nothing parseable. Retrying with the same
        constraint just reproduces it, so the retry drops the constraint and
        leans on the prompt plus `_extract_json` instead.
        """
        kwargs = {}
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=settings.max_tokens,
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    async def generate(self, question: str, chunks) -> tuple[RAGAnswer, float, bool]:
        """Returns (answer, generation_ms, retry_used)."""
        t0 = time.perf_counter()
        valid_ids = {c.chunk_id for c in chunks}
        context = build_context(chunks)
        user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ]

        retry_used = False
        try:
            answer = RAGAnswer(**_extract_json(await self._call_with_backoff(messages)))
        except Exception as exc:
            # Deliberately broad: as well as parse failures this catches
            # groq.BadRequestError(json_validate_failed), which is the exact
            # condition the retry exists for. Letting it escape 500s the API.
            # One retry, with the actual parse error fed back to the model.
            retry_used = True
            try:
                answer = RAGAnswer(
                    **_extract_json(
                        await self._call_with_backoff(
                            [
                                {"role": "system", "content": SYSTEM},
                                {"role": "user", "content": user},
                                {
                                    "role": "system",
                                    "content": RETRY_SYSTEM.format(error=str(exc)[:200]),
                                },
                            ],
                            force_json=False,
                        )
                    )
                )
            except Exception as exc2:
                # Never 500 the caller over a formatting failure.
                answer = RAGAnswer(
                    answer=f"Could not produce a validated answer ({type(exc2).__name__}).",
                    citations=[],
                    confidence=0.0,
                )

        answer = answer.drop_hallucinated_citations(valid_ids)
        return answer, (time.perf_counter() - t0) * 1000, retry_used
