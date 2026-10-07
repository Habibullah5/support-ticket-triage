import json

import pytest

from app.validation import OutputValidationError, parse_triage_output
from tests.helpers import VALID, VALID_JSON


def test_valid_output_parses():
    r = parse_triage_output(VALID_JSON)
    assert r.category == "billing" and r.priority == "medium"


@pytest.mark.parametrize("text", ["", "   ", "not json", "{'category': 'billing'}", '{"category": ',
                                  "```json\n" + VALID_JSON + "\n```"])
def test_invalid_json_fails_safely(text):
    with pytest.raises(OutputValidationError) as e:
        parse_triage_output(text)
    assert e.value.kind in {"empty_output", "invalid_json"}


@pytest.mark.parametrize("text", ["[]", '"billing"', "42", "null"])
def test_non_object_json_rejected(text):
    with pytest.raises(OutputValidationError) as e:
        parse_triage_output(text)
    assert e.value.kind == "not_an_object"


@pytest.mark.parametrize("category", ["shipping", "Billing", "", None, 3])
def test_invalid_category_rejected(category):
    with pytest.raises(OutputValidationError) as e:
        parse_triage_output(json.dumps({**VALID, "category": category}))
    assert e.value.kind == "schema_violation" and "category" in e.value.detail


@pytest.mark.parametrize("priority", ["urgent", "HIGH", "", None, 1])
def test_invalid_priority_rejected(priority):
    with pytest.raises(OutputValidationError) as e:
        parse_triage_output(json.dumps({**VALID, "priority": priority}))
    assert e.value.kind == "schema_violation" and "priority" in e.value.detail


def test_extra_fields_rejected():
    with pytest.raises(OutputValidationError) as e:
        parse_triage_output(json.dumps({**VALID, "confidence": 0.9}))
    assert e.value.kind == "schema_violation" and "confidence" in e.value.detail


@pytest.mark.parametrize("missing", ["category", "priority", "summary", "reason"])
def test_missing_field_rejected(missing):
    data = {k: v for k, v in VALID.items() if k != missing}
    with pytest.raises(OutputValidationError):
        parse_triage_output(json.dumps(data))


def test_empty_or_oversized_text_rejected():
    for bad in ({"summary": ""}, {"reason": "x" * 401}, {"summary": "x" * 301}):
        with pytest.raises(OutputValidationError):
            parse_triage_output(json.dumps({**VALID, **bad}))


def test_error_detail_does_not_echo_input():
    secret = "SECRET-CUSTOMER-TEXT"
    with pytest.raises(OutputValidationError) as e:
        parse_triage_output(json.dumps({**VALID, "category": secret}))
    assert secret not in e.value.detail
