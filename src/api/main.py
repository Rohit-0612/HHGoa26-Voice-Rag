"""FastAPI app. POST /query -> {answer, citations, confidence, timing}.

Timing is instrumented at phase granularity from Day 1 on purpose: Day 2's
latency analytics needs the embed/search/rerank breakdown, and retrofitting
instrumentation after the fact means touching every layer again.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.config import settings
from src.generation.schemas import Citation, QueryRequest, QueryResponse, Timing

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models ONCE at startup. First construction downloads ~2GB of ONNX
    weights; doing that lazily would put a 30s+ stall inside a user request."""
    from src.generation.generator import GroqGenerator
    from src.retrieval.embedder import FastEmbedProvider
    from src.retrieval.indexer import get_client
    from src.retrieval.retriever import FastEmbedReranker, HybridRetriever

    print("Loading embedding models ...")
    provider = FastEmbedProvider()
    print("Loading reranker ...")
    reranker = FastEmbedReranker()
    print("Connecting to Qdrant ...")
    client = get_client()

    STATE["retriever"] = HybridRetriever(client, provider, reranker)
    STATE["client"] = client
    try:
        STATE["generator"] = GroqGenerator()
    except RuntimeError as exc:
        print(f"WARNING: generation disabled -- {exc}")
        STATE["generator"] = None
    print("Ready.")
    yield
    STATE.clear()


app = FastAPI(title="HH Goa 2026 - Multilingual Voice RAG", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    ok = True
    detail = {}
    try:
        info = STATE["client"].get_collection(settings.qdrant_collection)
        detail["points"] = info.points_count
    except Exception as exc:
        ok = False
        detail["qdrant_error"] = str(exc)[:200]
    detail["models_loaded"] = "retriever" in STATE
    detail["generation_enabled"] = STATE.get("generator") is not None
    detail["llm"] = settings.groq_model
    return {"status": "ok" if ok else "degraded", **detail}


@app.get("/languages")
async def languages():
    from src.ingestion.loader import LANG_NAMES
    from src.retrieval.indexer import language_counts

    counts = language_counts(STATE["client"])
    return {
        "count": len(counts),
        "languages": [
            {"code": k, "name": LANG_NAMES.get(k, k), "chunks": v}
            for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
        ],
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    if "retriever" not in STATE:
        raise HTTPException(503, "Retriever not ready")

    t_start = time.perf_counter()

    result = await STATE["retriever"].retrieve(
        req.text, language=req.language, strategy=req.strategy, top_k=req.top_k
    )

    timing = Timing(**{k: v for k, v in result.timing.items() if k in Timing.model_fields})

    generator = STATE.get("generator")
    if generator is None:
        raise HTTPException(503, "Generation disabled -- GROQ_API_KEY not configured")

    if not result.chunks:
        answer, gen_ms, retry_used = (
            type("A", (), {"answer": "No relevant context found.", "citations": [], "confidence": 0.0})(),
            0.0,
            False,
        )
    else:
        answer, gen_ms, retry_used = await generator.generate(req.text, result.chunks)

    timing.generation_ms = gen_ms
    timing.total_ms = (time.perf_counter() - t_start) * 1000

    return QueryResponse(
        answer=answer.answer,
        citations=answer.citations,
        confidence=answer.confidence,
        timing=timing,
        contexts=[
            Citation(
                chunk_id=c.chunk_id,
                language=c.language,
                strategy=c.strategy,
                text=c.text[:400],
                rerank_score=c.rerank_score,
            )
            for c in result.chunks
        ],
        model=settings.groq_model,
        retry_used=retry_used,
    )
