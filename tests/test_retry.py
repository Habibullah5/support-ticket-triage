import pytest

from app.retry import RetryFailed, backoff_delay, call_with_retry, is_retryable
from tests.helpers import (auth_error, bad_request, connection_error, forbidden, rate_limit,
                           server_error, timeout_error)


def make_fn(script):
    state = {"calls": 0}

    def fn():
        state["calls"] += 1
        item = script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item
    return fn, state


def run(fn, sleeps, max_attempts=4, **kw):
    return call_with_retry(fn, max_attempts=max_attempts, base_delay=1.0, max_delay=20.0,
                           sleep=sleeps.append, rng=lambda: 0.5, **kw)


@pytest.mark.parametrize("make", [rate_limit, server_error, lambda: server_error(503),
                                  lambda: server_error(529), timeout_error, connection_error])
def test_retryable_errors_are_retried_then_succeed(make):
    fn, state = make_fn([make(), "ok"])
    sleeps = []
    value, attempts = run(fn, sleeps)
    assert value == "ok" and attempts == 2 and state["calls"] == 2 and len(sleeps) == 1


@pytest.mark.parametrize("make", [bad_request, auth_error, forbidden, lambda: ValueError("bug")])
def test_non_retryable_errors_fail_immediately(make):
    fn, state = make_fn([make(), "never reached"])
    sleeps = []
    with pytest.raises(RetryFailed) as e:
        run(fn, sleeps)
    assert state["calls"] == 1 and sleeps == [] and e.value.exhausted is False and e.value.attempts == 1


def test_retries_are_bounded():
    fn, state = make_fn([rate_limit() for _ in range(10)])
    sleeps = []
    with pytest.raises(RetryFailed) as e:
        run(fn, sleeps, max_attempts=3)
    assert state["calls"] == 3 and len(sleeps) == 2 and e.value.exhausted is True


def test_single_attempt_means_no_retry():
    fn, state = make_fn([rate_limit(), "ok"])
    with pytest.raises(RetryFailed):
        run(fn, [], max_attempts=1)
    assert state["calls"] == 1


def test_backoff_is_exponential_capped_and_jittered():
    assert backoff_delay(1, 1.0, 20.0, rng=lambda: 1.0) == 1.0
    assert backoff_delay(3, 1.0, 20.0, rng=lambda: 1.0) == 4.0
    assert backoff_delay(10, 1.0, 20.0, rng=lambda: 1.0) == 20.0  # ceiling
    assert backoff_delay(3, 1.0, 20.0, rng=lambda: 0.0) == 0.0     # jitter spans [0, ceiling]
    assert backoff_delay(3, 1.0, 20.0, rng=lambda: 0.25) != backoff_delay(3, 1.0, 20.0, rng=lambda: 0.75)


def test_retry_after_header_is_honoured_but_capped():
    fn, _ = make_fn([rate_limit({"retry-after": "7"}), "ok"])
    sleeps = []
    call_with_retry(fn, max_attempts=3, base_delay=1.0, max_delay=20.0, sleep=sleeps.append, rng=lambda: 0.0)
    assert sleeps == [7.0]
    fn, _ = make_fn([rate_limit({"retry-after": "999"}), "ok"])
    sleeps = []
    call_with_retry(fn, max_attempts=3, base_delay=1.0, max_delay=5.0, sleep=sleeps.append, rng=lambda: 0.0)
    assert sleeps == [5.0]


def test_is_retryable_classification():
    assert is_retryable(rate_limit()) and is_retryable(server_error()) and is_retryable(timeout_error())
    assert not is_retryable(bad_request()) and not is_retryable(auth_error())
