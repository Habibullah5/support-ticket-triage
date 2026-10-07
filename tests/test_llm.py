import json

import pytest

from app.config import Settings
from app.logging_config import JsonlLogger
from app.llm import LLMClient
from app.models import ConfigError, IncompleteResponseError, LLMUnavailableError, Prompt
from tests.helpers import VALID_JSON, FakeAnthropic, auth_error, fake_message, rate_limit

PROMPT = Prompt(name="triage_v1", text="system", sha256="abc123abc123")


def make(tmp_path, script, **overrides):
    settings = Settings(_env_file=None, anthropic_api_key="test-key", log_path=tmp_path / "log.jsonl",
                        retry_max_attempts=3, **overrides)
    log = JsonlLogger(settings.log_path)
    fake = FakeAnthropic(script)
    return LLMClient(settings, log, client=fake, sleep=lambda s: None, rng=lambda: 0.0), fake, settings


def records(settings):
    return [json.loads(line) for line in settings.log_path.read_text(encoding="utf-8").splitlines()]


def call(llm):
    return llm.complete(prompt=PROMPT, user_message="hi", request_id="r1", purpose="triage")


def test_success_logs_one_complete_record(tmp_path):
    llm, fake, settings = make(tmp_path, [fake_message(VALID_JSON, in_tok=120, out_tok=40)],
                               price_input_per_mtok=1.0, price_output_per_mtok=5.0)
    resp = call(llm)
    assert resp.text == VALID_JSON and resp.attempts == 1
    (rec,) = records(settings)
    for key in ["timestamp", "request_id", "prompt_version", "model", "input_tokens", "output_tokens",
                "latency_ms", "stop_reason", "outcome", "attempts", "cost_usd"]:
        assert key in rec
    assert rec["outcome"] == "ok" and rec["stop_reason"] == "end_turn" and rec["prompt_version"] == "triage_v1"
    assert rec["cost_usd"] == pytest.approx((120 * 1.0 + 40 * 5.0) / 1e6)
    assert fake.calls[0]["max_tokens"] == settings.max_output_tokens


def test_cost_is_null_when_prices_not_configured(tmp_path):
    llm, _, settings = make(tmp_path, [fake_message(VALID_JSON)])
    call(llm)
    assert records(settings)[0]["cost_usd"] is None


def test_truncated_response_is_not_success(tmp_path):
    llm, _, settings = make(tmp_path, [fake_message('{"category": "bil', stop_reason="max_tokens")])
    with pytest.raises(IncompleteResponseError) as e:
        call(llm)
    assert e.value.stop_reason == "max_tokens"
    assert records(settings)[0]["outcome"] == "truncated"


def test_refusal_is_not_success(tmp_path):
    llm, _, settings = make(tmp_path, [fake_message("", stop_reason="refusal")])
    with pytest.raises(IncompleteResponseError):
        call(llm)
    assert records(settings)[0]["outcome"] == "refusal"


def test_transient_error_retried_and_attempts_logged(tmp_path):
    llm, fake, settings = make(tmp_path, [rate_limit(), fake_message(VALID_JSON)])
    resp = call(llm)
    assert resp.attempts == 2 and len(fake.calls) == 2
    assert records(settings)[0]["attempts"] == 2


def test_permanent_error_not_retried(tmp_path):
    llm, fake, settings = make(tmp_path, [auth_error(), fake_message(VALID_JSON)])
    with pytest.raises(LLMUnavailableError) as e:
        call(llm)
    assert len(fake.calls) == 1 and e.value.status_code == 401 and e.value.exhausted is False
    assert records(settings)[0]["outcome"] == "non_retryable_error"


def test_exhausted_retries_logged(tmp_path):
    llm, fake, settings = make(tmp_path, [rate_limit(), rate_limit(), rate_limit(), fake_message("x")])
    with pytest.raises(LLMUnavailableError) as e:
        call(llm)
    assert len(fake.calls) == 3 and e.value.exhausted is True
    assert records(settings)[0]["outcome"] == "retries_exhausted"


def test_raw_content_not_logged_by_default_but_can_be_enabled(tmp_path):
    llm, _, settings = make(tmp_path, [fake_message(VALID_JSON)])
    call(llm)
    rec = records(settings)[0]
    assert "raw_user_message" not in rec and "raw_output" not in rec
    llm, _, settings = make(tmp_path / "on", [fake_message(VALID_JSON)], log_raw_content=True)
    call(llm)
    assert records(settings)[0]["raw_user_message"] == "hi"


def test_missing_api_key_raises_config_error(tmp_path):
    settings = Settings(_env_file=None, anthropic_api_key=None, log_path=tmp_path / "l.jsonl")
    with pytest.raises(ConfigError):
        LLMClient(settings, JsonlLogger(settings.log_path))


def test_only_sdk_supported_arguments_are_sent(tmp_path):
    llm, fake, _ = make(tmp_path, [fake_message(VALID_JSON)])
    call(llm)
    assert set(fake.calls[0]) <= FakeAnthropic.ALLOWED and "temperature" not in fake.calls[0]


def test_temperature_is_opt_in_via_extra_body(tmp_path):
    llm, fake, _ = make(tmp_path, [fake_message(VALID_JSON)], temperature=0.0)
    call(llm)
    assert fake.calls[0]["extra_body"] == {"temperature": 0.0}


def test_real_sdk_accepts_our_arguments(tmp_path):
    """Guard against SDK drift: validate kwargs against the installed SDK's real signature."""
    import inspect
    import anthropic
    params = set(inspect.signature(anthropic.Anthropic(api_key="x").messages.create).parameters)
    assert {"model", "max_tokens", "system", "messages", "extra_body"} <= params
