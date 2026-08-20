"""The structured-output contract. Pydantic is the enforcement point --
the LLM is asked for JSON, but only what validates here is trusted.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class RAGAnswer(BaseModel):
    answer: str = Field(..., description="Answer in the same language as the question")
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, v):
        """Models emit '0.85', '85%', or 85 depending on their mood."""
        if isinstance(v, str):
            v = v.strip().rstrip("%")
            v = float(v)
            if v > 1.0:
                v /= 100.0
        elif isinstance(v, (int, float)) and v > 1.0:
            v = float(v) / 100.0
        return max(0.0, min(1.0, float(v)))

    @field_validator("citations", mode="before")
    @classmethod
    def _coerce_citations(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [str(x) for x in v]

    @staticmethod
    def _normalise_citation(c: str) -> str:
        """Models echo the context label verbatim ("chunk_id: hi::1::0") or
        wrap it in brackets. Strip that before matching, or every citation
        looks fabricated and gets dropped."""
        c = c.strip().strip("[]() ")
        low = c.lower()
        if low.startswith("chunk_id:"):
            c = c[len("chunk_id:"):].strip()
        return c

    def drop_hallucinated_citations(self, valid_ids: set[str]) -> "RAGAnswer":
        """A citation to a chunk we never retrieved is a fabricated source.
        Drop it rather than pass it to the caller, and damp confidence if the
        model cited nothing real at all."""
        kept = [n for c in self.citations if (n := self._normalise_citation(c)) in valid_ids]
        confidence = self.confidence
        if self.citations and not kept:
            confidence = min(confidence, 0.3)
        return self.model_copy(update={"citations": kept, "confidence": confidence})


class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str | None = Field(None, description="Restrict retrieval to one language code")
    strategy: str | None = Field(None, description="Restrict to one chunking strategy")
    top_k: int | None = Field(None, ge=1, le=20)


class Timing(BaseModel):
    stt_ms: float = 0.0
    scope_ms: float = 0.0
    embed_ms: float = 0.0
    search_ms: float = 0.0
    rerank_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    groundedness_ms: float = 0.0
    total_ms: float = 0.0


class Citation(BaseModel):
    chunk_id: str
    language: str
    strategy: str
    text: str
    rerank_score: float | None = None


class Groundedness(BaseModel):
    score: float = 0.0
    min_sentence_score: float = 0.0
    kept: int = 0
    dropped: int = 0
    replaced: bool = False


class ScopeInfo(BaseModel):
    in_scope: bool = True
    score: float = 0.0
    threshold: float = 0.0
    stage: str = "disabled"
    reason: str = ""


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    confidence: float
    timing: Timing
    contexts: list[Citation] = Field(default_factory=list)
    model: str = ""
    retry_used: bool = False
    # ---- Day 2 ----
    provider: str = ""
    scope: ScopeInfo = Field(default_factory=ScopeInfo)
    groundedness: Groundedness = Field(default_factory=Groundedness)
    transcript: str | None = None
    detected_language: str | None = None
    raw_detected_language: str | None = None
