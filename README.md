# HH Goa 2026 — Task 2: Multilingual Voice-RAG

Voice-enabled RAG over [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI),
covering **all 14 Indic languages** (as, bn, gu, hi, kn, ml, mr, ne, or, pa, sa, ta, te, ur).

- **Day 1** — text query -> grounded answer (this milestone)
- **Day 2** — voice in/out + latency analytics + 2 more chunking strategies
- **Day 3** — evaluation & polish

## Stack

| Layer | Choice |
|---|---|
| Vector DB | Qdrant Cloud (free tier), hybrid dense+sparse |
| Dense | `paraphrase-multilingual-MiniLM-L12-v2` (384d, FastEmbed/ONNX) — see D-16 |
| Sparse | `Qdrant/bm25` (FastEmbed) |
| Fusion | Server-side RRF |
| Rerank | `jinaai/jina-reranker-v2-base-multilingual` |
| LLM | Groq `qwen/qwen3.6-27b`, JSON-mode |
| API | FastAPI (async) |

> Every one of these choices — and the alternatives rejected — is documented in [decision.md](decision.md).

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"
cp .env.example .env      # then paste your Qdrant + Groq credentials
uv run python scripts/prepare_data.py     # download + downsample corpus
uv run python scripts/build_index.py      # embed + upsert to Qdrant
uv run uvicorn src.api.main:app --reload
uv run python scripts/smoke_test.py       # 5 questions per language
```

## Query

```bash
curl -X POST localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"text": "भारत की राजधानी क्या है?"}'
```

```json
{
  "answer": "...",
  "citations": ["hi::12345::0"],
  "confidence": 0.87,
  "timing": {"retrieval_ms": 412.3, "generation_ms": 980.1, "total_ms": 1392.4}
}
```

## Layout

```
src/ingestion/   loader.py (HF parquet -> passages.jsonl), chunking.py (pluggable strategies)
src/retrieval/   embedder.py (EmbeddingProvider), indexer.py (Qdrant), retriever.py (hybrid+rerank)
src/generation/  schemas.py (Pydantic contract), generator.py (Groq + retry)
src/api/         main.py (POST /query, /health, /languages)
scripts/         prepare_data.py, build_index.py, smoke_test.py
```
