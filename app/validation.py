"""Strict parsing of model output. Nothing is trusted until it passes here."""
from __future__ import annotations

import json

from pydantic import ValidationError

from app.models import TriageResult


class OutputValidationError(Exception):
    """kind is one of: empty_output, invalid_json, not_an_object, schema_violation."""

    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


def _summarise(err: ValidationError) -> str:
    # include_input=False: never echo model/user content into error text.
    parts = []
    for e in err.errors(include_input=False):
        loc = ".".join(str(p) for p in e["loc"]) or "<root>"
        parts.append(f"{loc}: {e['msg']}")
    return "; ".join(parts)


def parse_triage_output(text: str) -> TriageResult:
    """Parse model text into a TriageResult or raise OutputValidationError.

    Deliberately strict: no code-fence stripping or regex extraction. A response
    that is not exactly one JSON object goes to the (single) repair step.
    """
    if not isinstance(text, str) or not text.strip():
        raise OutputValidationError("empty_output", "model returned no text")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OutputValidationError(
            "invalid_json", f"{exc.msg} at line {exc.lineno} column {exc.colno}"
        ) from exc
    if not isinstance(data, dict):
        raise OutputValidationError("not_an_object", f"expected a JSON object, got {type(data).__name__}")
    try:
        return TriageResult.model_validate(data)
    except ValidationError as exc:
        raise OutputValidationError("schema_violation", _summarise(exc)) from exc