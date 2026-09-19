"""Tests for ``jevctx.jev``: retries, rate limiting, concurrency, and pre-send budgeting.

Every test drives ``HttpJevClient`` through ``httpx.MockTransport``. Nothing here
touches the network or needs an API key -- a test that required either would itself
be a failed test.
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import pytest

from jevctx.jev import HttpJevClient, RateLimiter, RetryPolicy, check_request_budget
from jevctx.testing import FakeJevClient
from jevctx.types import (
    Choice,
    ChoiceAnswer,
    JevAuthError,
    JevBudgetError,
    JevUnavailableError,
    JevValidationError,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    State,
)

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

Handler = Callable[[httpx.Request], httpx.Response]


def _noul(instructions: str = "q", true: str = "yes", false: str = "no") -> Noul:
    return Noul(instructions=instructions, true=true, false=false)


def _noul_ans(value: float) -> dict[str, object]:
    return {"type": "noul", "noul": value}


def _ok(
    answers: dict[str, dict[str, object]], usage: dict[str, object] | None = None
) -> httpx.Response:
    body: dict[str, object] = {"answers": answers}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def _counting(handler: Handler) -> tuple[Handler, list[httpx.Request]]:
    calls: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    return wrapped, calls


def _echo_handler(request: httpx.Request) -> httpx.Response:
    """A generic success handler: answers whatever question types were asked."""
    body = json.loads(request.content)
    answers: dict[str, dict[str, object]] = {}
    for key, q in body["questions"].items():
        if q["type"] == "noul":
            answers[key] = {"type": "noul", "noul": 1.0}
        elif q["type"] == "choice":
            first = next(iter(q["criteria"]))
            answers[key] = {
                "type": "choice",
                "choice": first,
                "probabilities": {first: 1.0},
                "confidence": 1.0,
            }
        elif q["type"] == "score":
            answers[key] = {
                "type": "score",
                "score": 1.0,
                "legend": {},
                "probabilities": {},
                "confidence": 1.0,
            }
        else:  # pragma: no cover - defensive; every case below is one of the above
            raise AssertionError(f"unexpected question type {q['type']!r}")
    return httpx.Response(200, json={"answers": answers, "usage": {"input_tokens": 1}})


# --------------------------------------------------------------------------- #
# Request shape, auth
# --------------------------------------------------------------------------- #


def test_request_shape_matches_spec():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return _ok({"q": _noul_ans(1.0)})

    client = HttpJevClient(
        api_key="secret",
        model="jev-test",
        base_url="https://example.invalid/v1",
        transport=httpx.MockTransport(handler),
    )
    client.ask({"task": "t"}, {"q": _noul()})

    assert captured["url"] == "https://example.invalid/v1/systemone"
    assert captured["auth"] == "Bearer secret"
    body = captured["body"]
    assert body["model"] == "jev-test"
    assert body["state"] == {"task": "t"}
    assert body["questions"]["q"] == {
        "type": "noul",
        "instructions": "q",
        "criteria": {"true": "yes", "false": "no"},
    }


def test_missing_api_key_raises_at_construction(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JevAuthError):
        HttpJevClient()


def test_api_key_from_env_var(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return _ok({"q": _noul_ans(1.0)})

    client = HttpJevClient(transport=httpx.MockTransport(handler))
    client.ask("s", {"q": _noul()})
    assert captured["auth"] == "Bearer env-key"


def test_401_raises_jev_auth_error_no_retry():
    handler, calls = _counting(lambda r: httpx.Response(401, json={"error": "bad key"}))
    client = HttpJevClient(api_key="k", transport=httpx.MockTransport(handler))
    with pytest.raises(JevAuthError):
        client.ask("s", {"q": _noul()})
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# Retries
# --------------------------------------------------------------------------- #


def test_429_then_200_succeeds_request_sent_twice():
    responses = [httpx.Response(429), _ok({"q": _noul_ans(0.9)})]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses[len(calls) - 1]

    handler, calls = _counting(handler)
    sleeps: list[float] = []
    client = HttpJevClient(api_key="k", transport=httpx.MockTransport(handler), sleep=sleeps.append)

    result = client.ask("s", {"q": _noul()})

    assert len(calls) == 2
    assert isinstance(result["q"], NoulAnswer)
    assert result["q"].noul == 0.9
    assert len(sleeps) == 1


def test_retry_after_seconds_is_honoured():
    def handler(request: httpx.Request) -> httpx.Response:
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return _ok({"q": _noul_ans(1.0)})

    handler, calls = _counting(handler)
    sleeps: list[float] = []
    client = HttpJevClient(api_key="k", transport=httpx.MockTransport(handler), sleep=sleeps.append)

    client.ask("s", {"q": _noul()})

    assert sleeps == [2.0]
    assert len(calls) == 2


def test_retry_after_is_capped_at_the_timeout_budget():
    def handler(request: httpx.Request) -> httpx.Response:
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "999"})
        return _ok({"q": _noul_ans(1.0)})

    handler, calls = _counting(handler)
    sleeps: list[float] = []
    client = HttpJevClient(
        api_key="k", transport=httpx.MockTransport(handler), timeout=5.0, sleep=sleeps.append
    )

    client.ask("s", {"q": _noul()})

    assert sleeps == [5.0]


def test_422_is_never_retried_raises_validation_error():
    handler, calls = _counting(lambda r: httpx.Response(422, json={"error": "bad request"}))
    client = HttpJevClient(api_key="k", transport=httpx.MockTransport(handler))
    with pytest.raises(JevValidationError):
        client.ask("s", {"q": _noul()})
    assert len(calls) == 1


def test_other_4xx_raises_validation_error_no_retry():
    handler, calls = _counting(lambda r: httpx.Response(404, json={"error": "no such route"}))
    client = HttpJevClient(api_key="k", transport=httpx.MockTransport(handler))
    with pytest.raises(JevValidationError):
        client.ask("s", {"q": _noul()})
    assert len(calls) == 1


def test_retries_exhausted_on_5xx_raises_unavailable():
    handler, calls = _counting(lambda r: httpx.Response(503))
    client = HttpJevClient(
        api_key="k", transport=httpx.MockTransport(handler), max_retries=2, sleep=lambda s: None
    )
    with pytest.raises(JevUnavailableError):
        client.ask("s", {"q": _noul()})
    assert len(calls) == 3  # the initial attempt plus 2 retries


def test_retries_exhausted_on_transport_error_chains_cause():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = HttpJevClient(
        api_key="k", transport=httpx.MockTransport(handler), max_retries=1, sleep=lambda s: None
    )
    with pytest.raises(JevUnavailableError) as excinfo:
        client.ask("s", {"q": _noul()})
    assert isinstance(excinfo.value.__cause__, httpx.ConnectError)


def test_529_is_retried_like_429():
    responses = [httpx.Response(529), _ok({"q": _noul_ans(1.0)})]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses[len(calls) - 1]

    handler, calls = _counting(handler)
    client = HttpJevClient(api_key="k", transport=httpx.MockTransport(handler), sleep=lambda s: None)

    client.ask("s", {"q": _noul()})
    assert len(calls) == 2


# --------------------------------------------------------------------------- #
# Answer parsing
# --------------------------------------------------------------------------- #


def test_all_three_answer_types_parse():
    questions: dict[str, Question] = {
        "n": _noul(),
        "c": Choice(instructions="pick", criteria={"a": "A", "b": "B"}),
        "s": Score(instructions="rate", criteria=["low", "mid", "high"]),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok(
            {
                "n": _noul_ans(0.75),
                "c": {
                    "type": "choice",
                    "choice": "b",
                    "probabilities": {"a": 0.1, "b": 0.9},
                    "confidence": 0.9,
                },
                "s": {
                    "type": "score",
                    "score": 3.0,
                    "legend": {"0": "low", "1": "mid", "2": "high"},
                    "probabilities": {},
                    "confidence": 1.0,
                },
            }
        )

    client = HttpJevClient(api_key="k", transport=httpx.MockTransport(handler))
    result = client.ask("state", questions)

    assert isinstance(result["n"], NoulAnswer)
    assert result["n"].value == 0.75
    assert isinstance(result["c"], ChoiceAnswer)
    assert result["c"].value == "b"
    assert isinstance(result["s"], ScoreAnswer)
    assert result["s"].value == 3.0
    assert set(result) == {"n", "c", "s"}


def test_missing_answer_key_raises_validation_error():
    client = HttpJevClient(api_key="k", transport=httpx.MockTransport(lambda r: _ok({})))
    with pytest.raises(JevValidationError):
        client.ask("state", {"q": _noul()})


# --------------------------------------------------------------------------- #
# Pre-send budget checks: HttpJevClient must agree with FakeJevClient exactly.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BudgetCase:
    name: str
    state: State
    questions: dict[str, Question]
    valid: bool


def _budget_cases() -> list[BudgetCase]:
    small_state = "hello"
    return [
        BudgetCase("baseline", small_state, {"q0": _noul()}, True),
        BudgetCase("empty_questions", small_state, {}, False),
        BudgetCase(
            "too_many_questions", small_state, {f"q{i}": _noul() for i in range(33)}, False
        ),
        BudgetCase(
            "max_questions_boundary", small_state, {f"q{i}": _noul() for i in range(32)}, True
        ),
        BudgetCase(
            "choice_too_many_options",
            small_state,
            {"c": Choice("pick", {f"o{i}": None for i in range(256)})},
            False,
        ),
        BudgetCase(
            "choice_max_options_boundary",
            small_state,
            {"c": Choice("pick", {f"o{i}": None for i in range(255)})},
            True,
        ),
        BudgetCase(
            "score_too_few_levels", small_state, {"s": Score("rate", ["only"])}, False
        ),
        BudgetCase(
            "score_too_many_levels",
            small_state,
            {"s": Score("rate", [f"L{i}" for i in range(11)])},
            False,
        ),
        BudgetCase(
            "score_min_levels_boundary", small_state, {"s": Score("rate", ["lo", "hi"])}, True
        ),
        BudgetCase(
            "score_max_levels_boundary",
            small_state,
            {"s": Score("rate", [f"L{i}" for i in range(10)])},
            True,
        ),
        BudgetCase("total_tokens_exceeded", "z" * 230_000, {"q": _noul()}, False),
        BudgetCase(
            "longest_question_exceeded",
            "s" * 8_000,
            {"big": Noul("x" * 110_000, "t", "f"), "small": _noul()},
            False,
        ),
        BudgetCase(
            "large_but_within_budget",
            "s" * 8_000,
            {"big": Noul("x" * 90_000, "t", "f"), "small": _noul()},
            True,
        ),
    ]


@pytest.mark.parametrize("case", _budget_cases(), ids=lambda c: c.name)
def test_check_request_budget_agrees_with_fake_client(case: BudgetCase):
    fake = FakeJevClient()
    if case.valid:
        check_request_budget(case.state, case.questions)  # must not raise
        fake._check_limits(case.state, case.questions)  # must not raise
    else:
        with pytest.raises(JevBudgetError):
            check_request_budget(case.state, case.questions)
        with pytest.raises(JevBudgetError):
            fake._check_limits(case.state, case.questions)


@pytest.mark.parametrize("case", _budget_cases(), ids=lambda c: c.name)
def test_http_client_sends_no_request_for_invalid_budget_cases(case: BudgetCase):
    handler, calls = _counting(_echo_handler)
    client = HttpJevClient(api_key="k", transport=httpx.MockTransport(handler))

    if case.valid:
        result = client.ask(case.state, case.questions)
        assert set(result) == set(case.questions)
        assert len(calls) == 1
    else:
        with pytest.raises(JevBudgetError):
            client.ask(case.state, case.questions)
        assert len(calls) == 0


# --------------------------------------------------------------------------- #
# RateLimiter
# --------------------------------------------------------------------------- #


class _FakeClock:
    """A manually-advanced clock. `RateLimiter`'s injected `sleep` advances it,
    so a blocked `acquire()` converges without any real wall-clock wait."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_rate_limiter_admits_at_most_capacity_before_waiting():
    clock = _FakeClock()
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.advance(seconds)

    limiter = RateLimiter(5, per=60.0, clock=clock, sleep=fake_sleep)

    for _ in range(5):
        limiter.acquire()
    assert sleeps == []  # a full bucket admits a burst of `rate` with no wait

    limiter.acquire()  # the 6th call in the same instant must wait for a refill
    assert sleeps == pytest.approx([12.0])  # 1 token at 5/60 tokens/sec = 12s


def test_rate_limiter_refills_over_time():
    clock = _FakeClock()
    sleeps: list[float] = []
    limiter = RateLimiter(5, per=60.0, clock=clock, sleep=sleeps.append)

    for _ in range(5):
        limiter.acquire()
    clock.advance(60.0)  # a full window later, the bucket should be full again
    for _ in range(5):
        limiter.acquire()

    assert sleeps == []  # both bursts fit within their own window


def test_rate_limiter_rejects_nonpositive_config():
    with pytest.raises(ValueError):
        RateLimiter(0)
    with pytest.raises(ValueError):
        RateLimiter(5, per=0)


# --------------------------------------------------------------------------- #
# RetryPolicy
# --------------------------------------------------------------------------- #


def test_retry_policy_backoff_is_within_bounds():
    policy = RetryPolicy(base_delay=0.5, max_delay=8.0, rng=random.Random(0))
    for attempt in range(6):
        cap = min(8.0, 0.5 * 2**attempt)
        delay = policy.delay(attempt)
        assert 0.0 <= delay <= cap


def test_retry_policy_retry_after_overrides_jitter():
    policy = RetryPolicy(rng=random.Random(0))
    assert policy.delay(5, retry_after=3.5) == 3.5
    assert policy.delay(0, retry_after=0.0) == 0.0


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


def test_concurrency_cap_never_exceeded():
    max_concurrency = 4
    lock = threading.Lock()
    in_flight = 0
    max_seen = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, max_seen
        with lock:
            in_flight += 1
            max_seen = max(max_seen, in_flight)
        time.sleep(0.02)
        with lock:
            in_flight -= 1
        return _ok({"q": _noul_ans(1.0)})

    client = HttpJevClient(
        api_key="k", transport=httpx.MockTransport(handler), max_concurrency=max_concurrency
    )

    def worker() -> None:
        client.ask("state", {"q": _noul()})

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert 2 <= max_seen <= max_concurrency  # lower bound: proves overlap really happened


# --------------------------------------------------------------------------- #
# Usage accounting
# --------------------------------------------------------------------------- #


def test_usage_accumulates_across_requests():
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok({"q": _noul_ans(0.5)}, usage={"input_tokens": 100, "output_tokens": 7})

    client = HttpJevClient(api_key="k", transport=httpx.MockTransport(handler))
    client.ask("s1", {"q": _noul()})
    client.ask("s2", {"q": _noul()})

    assert client.usage.input_tokens == 200
    assert client.usage.output_tokens == 14
    assert client.usage.requests == 2


def test_usage_defaults_to_zero_when_absent():
    client = HttpJevClient(
        api_key="k", transport=httpx.MockTransport(lambda r: _ok({"q": _noul_ans(1.0)}))
    )
    client.ask("s", {"q": _noul()})
    assert client.usage.input_tokens == 0
    assert client.usage.output_tokens == 0
    assert client.usage.requests == 1


# --------------------------------------------------------------------------- #
# Context manager
# --------------------------------------------------------------------------- #


def test_context_manager_closes_underlying_client():
    handler = httpx.MockTransport(lambda r: _ok({"q": _noul_ans(1.0)}))
    with HttpJevClient(api_key="k", transport=handler) as client:
        client.ask("s", {"q": _noul()})
        assert client._client.is_closed is False
    assert client._client.is_closed is True
