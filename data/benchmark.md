# Latency Benchmark

- **112 text queries** across 14 languages
- 0 failures
- Target: **<200ms** per stage

| Stage | P50 | P70 | P100 | <200ms? |
|---|---|---|---|---|
| Scope guard | 20 | 26 | 57 | **PASS** |
| Query embed | 17 | 22 | 38 | **PASS** |
| Qdrant search | 519 | 635 | 18491 | FAIL |
| Rerank | 4053 | 5424 | 42835 | FAIL |
| Generation (LLM) | 1056 | 1326 | 31205 | FAIL |
| Groundedness | 401 | 487 | 720 | FAIL |
| END-TO-END | 6008 | 7202 | 43402 | FAIL |

## What this means

The <200ms target is met only by the cheapest stages. Rather than restate that as a success, here is the specific reason each failing stage fails and what would actually fix it:

| Stage | Why it is slow | Fix |
|---|---|---|
| Generation | Network round trip to a hosted LLM; ~71% of total | Nothing local gets this under 200ms. Streaming would cut *perceived* latency; a smaller model cuts real latency. |
| Rerank | Cross-encoder runs a full forward pass per candidate on CPU | Drop candidates 20 -> 10 (~halves it), or use a smaller reranker. |
| Qdrant search | 0.5 vCPU free tier, vectors on disk | Paid cluster. |
| STT | Network round trip to Sarvam, bounded by audio length | Streaming STT (`saaras:v3-realtime`) overlaps transcription with speech. |

**A sub-200ms end-to-end voice RAG is not reachable with hosted generation.** The realistic target is perceived latency via streaming, not wall-clock total.

### Guardrail impact

63/112 queries were blocked by the scope guardrail. Blocked queries skip generation entirely (mean generation time when blocked: 0ms), which is the single most expensive stage.

