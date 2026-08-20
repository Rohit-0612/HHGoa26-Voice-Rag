"""Per-provider circuit breaker.

Without this, a provider that is down costs every request its full retry and
backoff budget before the chain moves on. Day 1 measured that concretely: an
exhausted Groq quota turned an 81s stall into a failure on every single call.
After N consecutive failures the circuit opens and the provider is skipped
outright until a cooldown elapses, so the cost of an outage is paid once rather
than once per request.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.config import settings

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = field(default_factory=lambda: settings.breaker_failure_threshold)
    reset_after_s: float = field(default_factory=lambda: settings.breaker_reset_s)

    consecutive_failures: int = 0
    opened_at: float | None = None
    total_failures: int = 0
    total_successes: int = 0

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return CLOSED
        if (time.monotonic() - self.opened_at) >= self.reset_after_s:
            # Cooldown elapsed: allow exactly one probe through.
            return HALF_OPEN
        return OPEN

    def allows(self) -> bool:
        """False only while the circuit is genuinely open."""
        return self.state != OPEN

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.opened_at = None
        self.total_successes += 1

    def record_failure(self) -> None:
        self.total_failures += 1
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.failure_threshold:
            # Re-stamp on each failure so a failing half-open probe restarts
            # the cooldown instead of letting probes through every call.
            self.opened_at = time.monotonic()

    def snapshot(self) -> dict:
        return {
            "provider": self.name,
            "state": self.state,
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
        }
