# Multilingual Voice-RAG backend.
#
# NOTE ON MEMORY: this image needs ~2.3GB RAM at steady state (measured).
# The jina reranker alone accounts for ~1.3GB. It is NOT optional -- its score
# is the working off-topic guardrail (see decision.md D-24). Do not deploy this
# to a 512MB tier; it will OOM on the first request.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep the ONNX cache inside the image so it survives to runtime.
    FASTEMBED_CACHE_PATH=/opt/models \
    HF_HOME=/opt/hf

WORKDIR /app

# curl is used by the container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src

# Bake the ONNX weights into the image. Without this the FIRST request pays a
# ~2 minute download, which is exactly when a judge is watching.
RUN python -c "\
from fastembed import TextEmbedding, SparseTextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); \
SparseTextEmbedding('Qdrant/bm25'); \
TextCrossEncoder('jinaai/jina-reranker-v2-base-multilingual'); \
print('models cached')"

# centroids.npz is required by the pre-retrieval guardrail (25KB).
COPY data/centroids.npz ./data/centroids.npz

# 7860 suits HF Spaces; anything else can override $PORT.
ENV PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=60s --timeout=10s --start-period=180s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
