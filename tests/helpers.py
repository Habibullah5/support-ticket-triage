"""Test doubles: fake Anthropic client, fake LLM, and real SDK exception builders."""
from __future__ import annotations

import json
from types import SimpleNamespace

import anthropic
import httpx

from app.models import LLMResponse

VALID = {"category": "billing", "priority": "medium",
         "summary": "Customer was charged twice.", "reason": "Duplicate charge; not urgent."}
VALID_JSON = json.dumps(VALID)


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(cls, code: int, headers: dict | None = None):
    resp = httpx.Response(code, request=_request(), headers=headers or {}, json={"error": {"message": "x"}})
    return cls("error", response=resp, body=None)


def rate_limit(headers=None): return status_error(anthropic.RateLimitError, 429, headers)
def server_error(code=500): return status_error(anthropic.InternalServerError, code)
def bad_request(): return status_error(anthropic.BadRequestError, 400)
def auth_error(): return status_error(anthropic.AuthenticationError, 401)
def forbidden(): return status_error(anthropic.PermissionDeniedError, 403)
def timeout_error(): return anthropic.APITimeoutError(request=_request())
def connection_error(): return anthropic.APIConnectionError(request=_request())


def fake_message(text: str, stop_reason: str = "end_turn", in_tok: int = 100, out_tok: int = 30):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
                           stop_reason=stop_reason, model="claude-test")


class FakeAnthropic:
    """Scripted stand-in for anthropic.Anthropic(). Items are messages or exceptions."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []
        self.messages = self

    ALLOWED = {"model", "max_tokens", "system", "messages", "extra_body"}

    def create(self, **kwargs):
        unknown = set(kwargs) - self.ALLOWED  # mimic the real SDK: unknown kwargs are a TypeError
        if unknown:
            raise TypeError(f"Messages.create() got an unexpected keyword argument {sorted(unknown)[0]!r}")
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeLLM:
    """Stands in for LLMClient inside TriageService. Items: text, or an exception to raise."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return LLMResponse(text=item, input_tokens=100, output_tokens=30, stop_reason="end_turn",
                           model="claude-test", latency_ms=5.0, attempts=1, cost_usd=None)
