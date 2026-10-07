"""Thin wrapper around the Anthropic Messages API.

Responsibilities: bounded retries, concurrency cap, stop_reason check,
and exactly one JSONL log record per model call.
"""
from __future__ import annotations

import random
import threading
import time
from typing import Any, Callable

import anthropic

from app.config import Settings
from app.logging_config import JsonlLogger, console
from app.models import (ConfigError, IncompleteResponseError, LLMResponse,
                        LLMUnavailableError, Prompt)
from app.retry import RetryFailed, call_with_retry

# stop_reason values that mean "the model finished". Everything else is untrusted.
_COMPLETE = {"end_turn", "stop_sequence"}
_INCOMPLETE_OUTCOME = {"max_tokens": "truncated", "refusal": "refusal"}


class LLMClient:
    def __init__(self, settings: Settings, event_log: JsonlLogger, client: Any = None,
                 sleep: Callable[[float], None] = time.sleep,
                 rng: Callable[[], float] = random.random):
        if client is None:
            if settings.anthropic_api_key is None:
                raise ConfigError("ANTHROPIC_API_KEY is not set")
            # max_retries=0: our retry layer is the only one, so attempts are bounded
            # and visible in logs (the SDK default would silently multiply them).
            client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key.get_secret_value(),
                max_retries=0,
                timeout=settings.request_timeout_s,
            )
        self._client = client
        self._settings = settings
        self._log = event_log
        self._sleep = sleep
        self._rng = rng
        self._slots = threading.BoundedSemaphore(settings.max_concurrent_calls)

    def _cost(self, in_tok: int | None, out_tok: int | None) -> float | None:
        s = self._settings
        if None in (s.price_input_per_mtok, s.price_output_per_mtok) or in_tok is None or out_tok is None:
            return None
        return round((in_tok * s.price_input_per_mtok + out_tok * s.price_output_per_mtok) / 1_000_000, 6)

    def _record(self, *, request_id: str, purpose: str, prompt: Prompt, started: float, outcome: str,
                attempts: int, stop_reason: str | None = None, in_tok: int | None = None,
                out_tok: int | None = None, model: str | None = None, error_type: str | None = None,
                status_code: int | None = None, user_message: str | None = None,
                output_text: str | None = None) -> float:
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        record: dict[str, Any] = {
            "event": "model_call", "request_id": request_id, "purpose": purpose,
            "prompt_version": prompt.name, "prompt_sha256": prompt.sha256,
            "model": model or self._settings.anthropic_model,
            "input_tokens": in_tok, "output_tokens": out_tok, "latency_ms": latency_ms,
            "stop_reason": stop_reason, "outcome": outcome, "attempts": attempts,
            "cost_usd": self._cost(in_tok, out_tok),
            "error_type": error_type, "status_code": status_code,
        }
        if self._settings.log_raw_content:  # off by default: see README privacy policy
            record["raw_user_message"] = user_message
            record["raw_output"] = output_text
        self._log.write(record)
        return latency_ms

    def complete(self, *, prompt: Prompt, user_message: str, request_id: str, purpose: str) -> LLMResponse:
        s = self._settings
        kwargs: dict[str, Any] = {
            "model": s.anthropic_model, "max_tokens": s.max_output_tokens,
            "system": prompt.text,
            "messages": [{"role": "user", "content": user_message}],
        }
        if s.temperature is not None:
            kwargs["extra_body"] = {"temperature": s.temperature}
        started = time.perf_counter()

        def on_retry(attempt: int, exc: BaseException, delay: float) -> None:
            console.warning("request_id=%s retry %d after %s, sleeping %.2fs",
                            request_id, attempt, type(exc).__name__, delay)

        with self._slots:  # cap in-flight calls; held during backoff on purpose
            try:
                msg, attempts = call_with_retry(
                    lambda: self._client.messages.create(**kwargs),
                    max_attempts=s.retry_max_attempts, base_delay=s.retry_base_delay_s,
                    max_delay=s.retry_max_delay_s, sleep=self._sleep, rng=self._rng,
                    on_retry=on_retry,
                )
            except RetryFailed as failure:
                cause = failure.cause
                code = getattr(cause, "status_code", None)
                # Full detail goes to the console only (never to the client or the JSONL log).
                console.error("request_id=%s model call failed: %s", request_id,
                              type(cause).__name__, exc_info=cause)
                self._record(request_id=request_id, purpose=purpose, prompt=prompt, started=started,
                             outcome="retries_exhausted" if failure.exhausted else "non_retryable_error",
                             attempts=failure.attempts, error_type=type(cause).__name__,
                             status_code=code, user_message=user_message)
                raise LLMUnavailableError(
                    f"{type(cause).__name__} after {failure.attempts} attempt(s)",
                    cause_type=type(cause).__name__, status_code=code,
                    attempts=failure.attempts, exhausted=failure.exhausted,
                ) from cause

        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        in_tok, out_tok = msg.usage.input_tokens, msg.usage.output_tokens
        stop = msg.stop_reason or "unknown"
        complete = stop in _COMPLETE
        outcome = "ok" if complete else _INCOMPLETE_OUTCOME.get(stop, "unexpected_stop")
        latency_ms = self._record(
            request_id=request_id, purpose=purpose, prompt=prompt, started=started, outcome=outcome,
            attempts=attempts, stop_reason=stop, in_tok=in_tok, out_tok=out_tok,
            model=getattr(msg, "model", None), user_message=user_message, output_text=text)
        if not complete:
            raise IncompleteResponseError(stop)  # receiving text is NOT proof of success
        return LLMResponse(text=text, input_tokens=in_tok, output_tokens=out_tok, stop_reason=stop,
                           model=getattr(msg, "model", s.anthropic_model), latency_ms=latency_ms,
                           attempts=attempts, cost_usd=self._cost(in_tok, out_tok))
