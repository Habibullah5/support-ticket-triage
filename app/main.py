
from __future__ import annotations

import uuid
from functools import lru_cache

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.llm import LLMClient
from app.logging_config import JsonlLogger, configure_console_logging, console
from app.models import ConfigError, TriageRequest, TriageResponse
from app.triage import (INCOMPLETE_RESPONSE, INVALID_OUTPUT, MODEL_UNAVAILABLE, TriageService)

configure_console_logging()
app = FastAPI(title="Support Ticket Triage Agent", version="1.0.0")

# Fixed, safe messages. No stack traces, prompts, keys or provider error text.
_SAFE = {
    MODEL_UNAVAILABLE: (503, "The triage model is temporarily unavailable. Please retry later."),
    INCOMPLETE_RESPONSE: (502, "The model did not produce a complete answer. Please retry."),
    INVALID_OUTPUT: (502, "The model produced an invalid answer after one repair attempt."),
}


@lru_cache
def get_service() -> TriageService:
    settings = get_settings()
    event_log = JsonlLogger(settings.log_path)
    return TriageService(settings, LLMClient(settings, event_log), event_log)


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", None) or uuid.uuid4().hex[:16]


def _error(request: Request, status: int, message: str, **extra) -> JSONResponse:
    rid = _rid(request)
    return JSONResponse({"request_id": rid, "error": message, **extra}, status_code=status,
                        headers={"X-Request-ID": rid})


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex[:16]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(RequestValidationError)
async def _bad_request(request: Request, exc: RequestValidationError):
    details = [{"field": ".".join(str(p) for p in e["loc"][1:]) or "body", "problem": e["msg"]}
               for e in exc.errors()]  # never echo the submitted input
    return _error(request, 422, "Invalid request.", details=details)


@app.exception_handler(ConfigError)
async def _not_configured(request: Request, exc: ConfigError):
    console.error("request_id=%s configuration error: %s", _rid(request), exc)
    return _error(request, 503, "Service is not configured.")


@app.exception_handler(Exception)
async def _unexpected(request: Request, exc: Exception):
    console.exception("request_id=%s unhandled error", _rid(request))
    return _error(request, 500, "Internal error.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/triage", response_model=TriageResponse)
def triage(body: TriageRequest, request: Request, service: TriageService = Depends(get_service)):
    outcome = service.triage(body.ticket, request_id=_rid(request))
    if outcome.success and outcome.result is not None:
        return TriageResponse(request_id=outcome.request_id, result=outcome.result,
                              repaired=outcome.repair_attempted)
    status, message = _SAFE.get(outcome.failure_reason or "", (500, "Internal error."))
    return _error(request, status, message)
