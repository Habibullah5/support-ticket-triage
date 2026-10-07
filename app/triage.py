"""Triage pipeline: call -> validate -> (one repair) -> result or clear failure."""
from __future__ import annotations

import hashlib
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.llm import LLMClient
from app.logging_config import JsonlLogger
from app.models import (IncompleteResponseError, LLMResponse, LLMUnavailableError, Prompt,
                        TriageResult)
from app.validation import OutputValidationError, parse_triage_output

# failure_reason values
MODEL_UNAVAILABLE = "model_unavailable"
INCOMPLETE_RESPONSE = "incomplete_response"
INVALID_OUTPUT = "invalid_output"


@dataclass
class TriageOutcome:
    request_id: str
    success: bool
    result: TriageResult | None
    repair_attempted: bool
    first_attempt_valid: bool
    failure_reason: str | None
    model_calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float | None


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def load_prompt(prompts_dir: Path, name: str) -> Prompt:
    path = Path(prompts_dir) / f"{name}.txt"
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    return Prompt(name=name, text=text, sha256=hashlib.sha256(text.encode("utf-8")).hexdigest()[:12])


def _clean(text: str) -> str:
    # Remove our own delimiter tags from untrusted text. A mitigation, not a guarantee.
    return re.sub(r"</?\s*(ticket|invalid_output|validation_error)\s*>", "", text, flags=re.I)


def build_triage_message(ticket: str) -> str:
    return f"<ticket>\n{_clean(ticket)}\n</ticket>"


def build_repair_message(ticket: str, bad_output: str, err: OutputValidationError) -> str:
    return (f"<ticket>\n{_clean(ticket)}\n</ticket>\n\n"
            f"<invalid_output>\n{_clean(bad_output)[:2000]}\n</invalid_output>\n\n"
            f"<validation_error>\n{err.kind}: {err.detail}\n</validation_error>")


class TriageService:
    def __init__(self, settings: Settings, llm: LLMClient, event_log: JsonlLogger):
        self._settings = settings
        self._llm = llm
        self._log = event_log
        # Fail fast at startup if a prompt file is missing or empty.
        self._triage_prompt = load_prompt(settings.prompts_dir, settings.triage_prompt)
        self._repair_prompt = load_prompt(settings.prompts_dir, settings.repair_prompt)

    def triage(self, ticket_text: str, request_id: str | None = None) -> TriageOutcome:
        rid = request_id or new_request_id()
        started = time.perf_counter()
        calls: list[LLMResponse] = []

        def finish(success: bool, result: TriageResult | None, repair: bool, first_ok: bool,
                   reason: str | None, kind: str | None = None) -> TriageOutcome:
            costs = [c.cost_usd for c in calls]
            outcome = TriageOutcome(
                request_id=rid, success=success, result=result, repair_attempted=repair,
                first_attempt_valid=first_ok, failure_reason=reason, model_calls=len(calls),
                input_tokens=sum(c.input_tokens for c in calls),
                output_tokens=sum(c.output_tokens for c in calls),
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                cost_usd=None if (not costs or None in costs) else round(sum(costs), 6),
            )
            record = {
                "event": "triage_outcome", "request_id": rid,
                "outcome": ("repaired" if repair else "ok") if success else "failed",
                "failure_reason": reason, "validation_error_kind": kind,
                "first_attempt_valid": first_ok, "repair_attempted": repair,
                "model_calls": outcome.model_calls, "input_tokens": outcome.input_tokens,
                "output_tokens": outcome.output_tokens, "latency_ms": outcome.latency_ms,
                "cost_usd": outcome.cost_usd,
                "triage_prompt_version": self._triage_prompt.name,
                "repair_prompt_version": self._repair_prompt.name,
                # Correlate without storing content: length + short hash.
                "ticket_chars": len(ticket_text),
                "ticket_sha256": hashlib.sha256(ticket_text.encode("utf-8")).hexdigest()[:12],
            }
            if self._settings.log_raw_content:
                record["raw_ticket"] = ticket_text
            self._log.write(record)
            return outcome

        def failure_for(exc: Exception) -> str:
            return MODEL_UNAVAILABLE if isinstance(exc, LLMUnavailableError) else INCOMPLETE_RESPONSE

        # 1) First attempt
        try:
            first = self._llm.complete(prompt=self._triage_prompt, request_id=rid, purpose="triage",
                                       user_message=build_triage_message(ticket_text))
        except (LLMUnavailableError, IncompleteResponseError) as exc:
            return finish(False, None, False, False, failure_for(exc))
        calls.append(first)
        try:
            return finish(True, parse_triage_output(first.text), False, True, None)
        except OutputValidationError as first_err:
            err = first_err

        # 2) Exactly one repair attempt, using the separate repair prompt
        try:
            second = self._llm.complete(prompt=self._repair_prompt, request_id=rid, purpose="repair",
                                        user_message=build_repair_message(ticket_text, first.text, err))
        except (LLMUnavailableError, IncompleteResponseError) as exc:
            return finish(False, None, True, False, failure_for(exc), err.kind)
        calls.append(second)
        try:
            result = parse_triage_output(second.text)
        except OutputValidationError as err2:
            return finish(False, None, True, False, INVALID_OUTPUT, err2.kind)  # clear failure
        return finish(True, result, True, False, None)
