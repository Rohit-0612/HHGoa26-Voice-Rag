"""FastAPI app.

  POST /query        {text}          -> grounded answer + timing
  POST /query-audio  multipart audio -> STT -> the same pipeline

Both endpoints share `_run_query`, so the voice path cannot drift from the text
path. Order inside a request:

  scope guard (cheap, weak)  ->  retrieval  ->  relevance gate (accurate)
  ->  generation (provider chain)  ->  groundedness  ->  response

The two scope checks bracket retrieval deliberately: the pre-check is nearly
free but unreliable, the post-check is accurate and still fires before
generation, which is ~71% of end-to-end latency.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

import asyncio

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.generation.schemas import (
    Citation,
    Groundedness,
    QueryRequest,
    QueryResponse,
    ScopeInfo,
    Timing,
)
from src.guardrails.scope import OUT_OF_SCOPE_ANSWER

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load every model ONCE. First construction downloads ONNX weights; doing
    it lazily would put a 30s+ stall inside whichever request arrived first."""
    from src.generation.generator import build_default_chain
    from src.guardrails.groundedness import GroundednessChecker
    from src.guardrails.scope import RelevanceGate, ScopeGuard
    from src.retrieval.embedder import FastEmbedProvider
    from src.retrieval.indexer import get_client
    from src.retrieval.retriever import FastEmbedReranker, HybridRetriever

    print("Loading embedding models ...", flush=True)
    provider = FastEmbedProvider()
    print("Loading reranker ...", flush=True)
    reranker = FastEmbedReranker()
    print("Connecting to Qdrant ...", flush=True)
    client = get_client(timeout=20)  # query path fails fast; retries handle blips

    STATE["provider"] = provider
    STATE["client"] = client
    STATE["retriever"] = HybridRetriever(client, provider, reranker)
    STATE["scope_guard"] = ScopeGuard()
    STATE["relevance_gate"] = RelevanceGate()
    STATE["groundedness"] = GroundednessChecker(provider)

    try:
        STATE["generator"] = build_default_chain()
        print(f"Providers: {[p.name for p in STATE['generator'].providers]}", flush=True)
    except Exception as exc:
        print(f"WARNING: generation disabled -- {exc}", flush=True)
        STATE["generator"] = None

    try:
        STATE["stt"] = __import__("src.stt.sarvam", fromlist=["SarvamSTT"]).SarvamSTT()
        print("STT: sarvam ready", flush=True)
    except Exception as exc:
        print(f"WARNING: STT disabled -- {exc}", flush=True)
        STATE["stt"] = None

    print("Ready.", flush=True)
    yield
    STATE.clear()


app = FastAPI(title="HH Goa 2026 - Multilingual Voice RAG", version="1.0.0", lifespan=lifespan)

# Without this the deployed frontend cannot call the API at all -- the single
# most common last-mile deployment failure.
app.add_middleware(
    CORSMiddleware,
    allow_origins=(["*"] if settings.cors_origins.strip() == "*"
                   else [o.strip() for o in settings.cors_origins.split(",") if o.strip()]),
    allow_credentials=False,   # cannot be True alongside allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# The heavy stages (ONNX embed + cross-encoder) are CPU-bound and run in a
# threadpool. Unbounded concurrency on a 2 vCPU box means every request gets
# slower rather than some completing quickly, so admission is bounded and
# excess load is shed with 429 instead of hanging.
_SEMAPHORE = asyncio.Semaphore(settings.max_concurrent_requests)


class _Busy(Exception):
    pass


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    """Judges poking at the API must never see a stack trace."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal error ({type(exc).__name__}). "
                           f"The service is up; this request failed."},
    )


# --------------------------------------------------------------------------
# shared pipeline
# --------------------------------------------------------------------------
async def _run_query(
    text: str,
    language: str | None = None,
    strategy: str | None = None,
    top_k: int | None = None,
    stt_ms: float = 0.0,
    transcript: str | None = None,
    detected_language: str | None = None,
    raw_detected_language: str | None = None,
) -> QueryResponse:
    if "retriever" not in STATE:
        raise HTTPException(503, "Retriever not ready")
    if not (text or "").strip():
        raise HTTPException(400, "Empty query text")

    t_start = time.perf_counter()
    timing = Timing(stt_ms=stt_ms)

    def _respond(answer, *, citations, confidence, contexts, scope, ground,
                 provider="", model="", retry_used=False):
        timing.total_ms = (time.perf_counter() - t_start) * 1000
        return QueryResponse(
            answer=answer, citations=citations, confidence=confidence,
            timing=timing, contexts=contexts, model=model, retry_used=retry_used,
            provider=provider, scope=scope, groundedness=ground,
            transcript=transcript, detected_language=detected_language,
            raw_detected_language=raw_detected_language,
        )

    # ---- 1. pre-retrieval scope guard -------------------------------------
    t0 = time.perf_counter()
    query_vec = STATE["provider"].embed_dense_only([text])[0]
    verdict = STATE["scope_guard"].check(query_vec)
    timing.scope_ms = (time.perf_counter() - t0) * 1000

    if not verdict.in_scope:
        return _respond(
            OUT_OF_SCOPE_ANSWER, citations=[], confidence=0.0, contexts=[],
            scope=ScopeInfo(**verdict.__dict__), ground=Groundedness(),
        )

    # ---- 2. retrieval ------------------------------------------------------
    result = await STATE["retriever"].retrieve(
        text, language=language, strategy=strategy, top_k=top_k
    )
    for k, v in result.timing.items():
        if k in Timing.model_fields:
            setattr(timing, k, v)

    contexts = [
        Citation(chunk_id=c.chunk_id, language=c.language, strategy=c.strategy,
                 text=(c.context_text or c.text)[:400], rerank_score=c.rerank_score)
        for c in result.chunks
    ]

    # ---- 3. post-retrieval relevance gate (the one that actually works) ----
    gate = STATE["relevance_gate"].check(result.chunks)
    scope_info = ScopeInfo(**(gate.__dict__ if not gate.in_scope else verdict.__dict__))

    if not result.chunks or not gate.in_scope:
        return _respond(
            OUT_OF_SCOPE_ANSWER if result.chunks else "No relevant context found.",
            citations=[], confidence=0.0, contexts=contexts,
            scope=scope_info, ground=Groundedness(),
        )

    # ---- 4. generation -----------------------------------------------------
    generator = STATE.get("generator")
    if generator is None:
        raise HTTPException(503, "Generation disabled -- no provider configured")

    gen = await generator.generate_detailed(text, result.chunks)
    answer = gen["answer"]
    timing.generation_ms = gen["generation_ms"]

    # ---- 5. groundedness ---------------------------------------------------
    final_text, report = STATE["groundedness"].check(answer.answer, result.chunks)
    timing.groundedness_ms = report.duration_ms
    confidence = 0.0 if report.replaced else answer.confidence

    return _respond(
        final_text,
        citations=[] if report.replaced else answer.citations,
        confidence=confidence,
        contexts=contexts,
        scope=scope_info,
        ground=Groundedness(score=report.score, min_sentence_score=report.min_sentence_score,
                            kept=report.kept, dropped=report.dropped, replaced=report.replaced),
        provider=gen["provider"], model=gen["model"], retry_used=gen["retry_used"],
    )


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------
@app.get("/health")
async def health():
    ok, detail = True, {}
    try:
        info = STATE["client"].get_collection(settings.qdrant_collection)
        detail["points"] = info.points_count
    except Exception as exc:
        ok = False
        detail["qdrant_error"] = str(exc)[:200]

    gen = STATE.get("generator")
    detail.update(
        models_loaded="retriever" in STATE,
        generation_enabled=gen is not None,
        providers=[p.name for p in gen.providers] if gen else [],
        breakers=gen.breaker_status() if gen else [],
        stt_enabled=STATE.get("stt") is not None,
        scope_guard_enabled=getattr(STATE.get("scope_guard"), "enabled", False),
    )
    return {"status": "ok" if ok else "degraded", **detail}


@app.get("/languages")
async def languages():
    from src.ingestion.loader import LANG_NAMES
    from src.retrieval.indexer import language_counts

    counts = language_counts(STATE["client"])
    return {
        "count": len(counts),
        "languages": [{"code": k, "name": LANG_NAMES.get(k, k), "chunks": v}
                      for k, v in sorted(counts.items(), key=lambda kv: -kv[1])],
    }


async def _admit(coro):
    """Bounded admission: wait for a slot, but shed load rather than hang."""
    try:
        await asyncio.wait_for(_SEMAPHORE.acquire(), timeout=settings.queue_timeout_s)
    except asyncio.TimeoutError:
        raise HTTPException(
            429, "Server is busy handling other requests. Please retry in a moment."
        )
    try:
        return await coro
    finally:
        _SEMAPHORE.release()


ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".oga", ".opus",
                     ".flac", ".webm", ".aac", ".aiff", ".amr", ".wma", ".pcm"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    return await _admit(_run_query(req.text, req.language, req.strategy, req.top_k))


@app.post("/query-audio", response_model=QueryResponse)
async def query_audio(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    strategy: str | None = Form(None),
    top_k: int | None = Form(None),
    scope_to_detected: bool = Form(True),
) -> QueryResponse:
    """Transcribe, then run the identical text pipeline."""
    stt = STATE.get("stt")
    if stt is None:
        raise HTTPException(503, "STT disabled -- SARVAM_API_KEY not configured")

    filename = file.filename or "audio.wav"
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext and ext not in ALLOWED_AUDIO_EXT:
        raise HTTPException(
            415,
            f"Unsupported file type '{ext}'. Supported: "
            f"{', '.join(sorted(ALLOWED_AUDIO_EXT))}",
        )

    audio = await file.read()
    if not audio:
        raise HTTPException(400, "Empty audio upload")

    # Reject oversize BEFORE spending a Sarvam call on it.
    size_mb = len(audio) / (1024 * 1024)
    if size_mb > settings.max_audio_mb:
        raise HTTPException(
            413,
            f"Audio is {size_mb:.1f}MB; the limit is {settings.max_audio_mb:.0f}MB. "
            f"Sarvam's endpoint also caps at ~{settings.max_audio_seconds:.0f}s of audio, "
            f"so please send a shorter clip.",
        )

    from src.stt.sarvam import AudioTooLongError, STTError

    try:
        res = await stt.transcribe(
            audio,
            filename=filename,
            language_hint=language,
            content_type=file.content_type or "audio/wav",
        )
    except AudioTooLongError as exc:
        raise HTTPException(413, str(exc)) from exc
    except STTError as exc:
        raise HTTPException(502, f"Transcription failed: {exc}") from exc

    if not res.transcript:
        raise HTTPException(422, "Transcription produced no text")

    # Scope retrieval to the detected language only when it maps to an indexed
    # one. `res.language is None` means Sarvam returned something we do not
    # index -- searching all languages beats filtering to the wrong one.
    scoped = language or (res.language if scope_to_detected else None)

    return await _admit(_run_query(
        res.transcript,
        language=scoped,
        strategy=strategy,
        top_k=top_k,
        stt_ms=res.duration_ms,
        transcript=res.transcript,
        detected_language=res.language,
        raw_detected_language=res.raw_language,
    ))
