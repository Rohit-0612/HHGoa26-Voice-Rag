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

    @property
    def passages_path(self) -> Path:
        return DATA_DIR / "passages.jsonl"

    @property
    def smoke_results_path(self) -> Path:
        return DATA_DIR / "smoke_results.json"


settings = Settings()
