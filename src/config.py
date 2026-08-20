"""Central settings. Everything reads from here; no scattered os.getenv calls."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Qdrant Cloud
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "msmarco_xi"

    # Groq
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    # qwen3.6 emits reasoning tokens before the JSON; 1024 truncates it to a
    # bare "{" and burns a retry. 4096 leaves room for both.
    max_tokens: int = 1024
    # Hard ceiling on any single rate-limit backoff sleep, in seconds.
    max_backoff_s: float = 10.0

    # FastEmbed models
    dense_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "jinaai/jina-reranker-v2-base-multilingual"
    dense_dim: int = 384

    # Corpus / retrieval sizing
    rows_per_lang: int = 200
    retrieve_candidates: int = 20
    top_k: int = 5

    # Semantic chunking
    semantic_breakpoint_percentile: float = 90.0
    semantic_min_chars: int = 120

    # Fixed-size chunking (Day 2 baseline)
    fixed_chunk_chars: int = 400
    fixed_overlap_chars: int = 80

    # Sentence-window / parent-child chunking (Day 2)
    window_size: int = 1  # +/- N sentences carried as parent context

    # ---- Sarvam STT (Day 2) ----
    sarvam_api_key: str = ""
    sarvam_stt_url: str = "https://api.sarvam.ai/speech-to-text"
    sarvam_model: str = "saaras:v3"
    # Sarvam's REST endpoint caps at 30s of audio per request.
    max_audio_mb: float = 20.0
    stt_retries: int = 3

    # ---- NVIDIA NIM fallback (Day 2) ----
    nim_api_key: str = ""
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    nim_model: str = "nvidia/nvidia-nemotron-nano-9b-v2"

    # ---- Circuit breaker ----
    breaker_failure_threshold: int = 3
    breaker_reset_s: float = 60.0

    # ---- Guardrail: pre-retrieval scope ----
    centroid_k: int = 16
    scope_percentile: float = 5.0
    # Pre-retrieval centroid guard is measured-weak (D-24); default it permissive
    # so it only catches egregious garbage, and rely on relevance_threshold.
    scope_threshold: float | None = 0.15
    # Post-retrieval cross-encoder gate. Calibrated on 56 real queries across
    # all 14 languages (NOT the 6-query sample that first suggested 0.0 -- that
    # blocked 66% of real traffic). At -1.5: 1.8% of real queries blocked, 5/6
    # off-topic caught. See decision.md D-29.
    # -2.0, not -1.5: a real Kannada corpus query scored -1.51 and was wrongly
    # blocked. False positives are unrecoverable (user gets nothing) while false
    # negatives are caught downstream -- the LLM refuses, then groundedness
    # replaces. Both verified. Bias permissive. See D-31.
    relevance_threshold: float = -2.0

    # ---- Deployment (Day 3) ----
    # Comma-separated origins, or "*". Default is permissive so judges can hit
    # the API from anywhere; tighten for a real deployment.
    cors_origins: str = "*"
    # Measured: 4 concurrent requests on a 2-vCPU box took 96-130s each because
    # the ONNX reranker is CPU-bound and they thrash. Serialising to 2 keeps
    # individual latency near the single-request baseline (~3-6s).
    max_concurrent_requests: int = 2
    # Shed fast. A judge would rather get "busy, retry" in 12s than a 130s wait.
    queue_timeout_s: float = 12.0
    max_audio_seconds: float = 30.0

    # ---- Guardrail: post-generation groundedness ----
    groundedness_sentence_threshold: float = 0.35
    groundedness_answer_threshold: float = 0.30

    @property
    def passages_path(self) -> Path:
        return DATA_DIR / "passages.jsonl"

    @property
    def smoke_results_path(self) -> Path:
        return DATA_DIR / "smoke_results.json"

    @property
    def centroids_path(self) -> Path:
        return DATA_DIR / "centroids.npz"


settings = Settings()
