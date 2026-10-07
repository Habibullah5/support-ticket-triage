"""Schemas, small data containers and exceptions shared across the app."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal["billing", "technical", "account", "delivery", "refund", "other"]
Priority = Literal["low", "medium", "high"]
CATEGORIES = ("billing", "technical", "account", "delivery", "refund", "other")
PRIORITIES = ("low", "medium", "high")
MAX_TICKET_CHARS = 5000


class TriageResult(BaseModel):
    """The only shape of model output we are willing to trust."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    category: Category
    priority: Priority
    summary: str = Field(min_length=1, max_length=300)
    reason: str = Field(min_length=1, max_length=400)


class TriageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ticket: str = Field(min_length=1, max_length=MAX_TICKET_CHARS)


class TriageResponse(BaseModel):
    request_id: str
    result: TriageResult
    repaired: bool


class ErrorResponse(BaseModel):
    request_id: str
    error: str


@dataclass(frozen=True)
class Prompt:
    name: str  # e.g. "triage_v1" -> this is the prompt_version in logs
    text: str
    sha256: str  # first 12 hex chars of the file content hash


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    model: str
    latency_ms: float
    attempts: int
    cost_usd: float | None


class ConfigError(Exception):
    """Required configuration (e.g. the API key) is missing."""


class LLMUnavailableError(Exception):
    """The model call failed for good: permanent error or retries exhausted."""

    def __init__(self, message: str, *, cause_type: str, status_code: int | None,
                 attempts: int, exhausted: bool):
        super().__init__(message)
        self.cause_type = cause_type
        self.status_code = status_code
        self.attempts = attempts
        self.exhausted = exhausted


class IncompleteResponseError(Exception):
    """The model did not finish normally (truncated, refused, ...). Never trusted."""

    def __init__(self, stop_reason: str):
        super().__init__(f"model stopped with stop_reason={stop_reason!r}")
        self.stop_reason = stop_reason
