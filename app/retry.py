"""Bounded retries with exponential backoff + full jitter."""
from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

import anthropic

T = TypeVar("T")


class RetryFailed(Exception):
    """Raised when a call fails for good. `exhausted` is True when we ran out of attempts."""

    def __init__(self, cause: BaseException, attempts: int, exhausted: bool):
        super().__init__(f"{type(cause).__name__} after {attempts} attempt(s)")
        self.cause = cause
        self.attempts = attempts
        self.exhausted = exhausted


def is_retryable(exc: BaseException) -> bool:
    """Retry only failures likely to be temporary.

    Retry: connection errors, timeouts, 429, any 5xx (incl. 529 overloaded).
    Never retry: 400, 401, 403, 404, other 4xx, or anything unexpected.
    """
    if isinstance(exc, (anthropic.APIConnectionError, TimeoutError, ConnectionError)):
        return True  # APITimeoutError is a subclass of APIConnectionError
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code == 429 or code >= 500
    return False


def backoff_delay(attempt: int, base: float, cap: float,
                  rng: Callable[[], float] = random.random) -> float:
    """Full jitter: uniform(0, min(cap, base * 2**(attempt-1))). `attempt` starts at 1."""
    ceiling = min(cap, base * (2 ** (attempt - 1)))
    return rng() * ceiling


def retry_after_seconds(exc: BaseException) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        value = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int,
    base_delay: float,
    max_delay: float,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> tuple[T, int]:
    """Run fn(). Returns (value, attempts_used). Never loops more than max_attempts times."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(), attempt
        except Exception as exc:  # noqa: BLE001 - classified below, always re-raised
            if not is_retryable(exc):
                raise RetryFailed(exc, attempt, exhausted=False) from exc
            if attempt == max_attempts:
                raise RetryFailed(exc, attempt, exhausted=True) from exc
            delay = backoff_delay(attempt, base_delay, max_delay, rng)
            hinted = retry_after_seconds(exc)
            if hinted is not None:
                delay = max(delay, min(hinted, max_delay))  # honour Retry-After, still capped
            if on_retry:
                on_retry(attempt, exc, delay)
            sleep(delay)
    raise AssertionError("unreachable: max_attempts must be >= 1")  # pragma: no cover
