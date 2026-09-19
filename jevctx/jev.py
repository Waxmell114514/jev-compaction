"""HTTP transport for the Jev API.

``types.py``, ``tokens.py`` and ``testing.py`` are the contract every module in this
package is written against; this module is the one piece that actually issues HTTP
requests. It is kept separate from ``scorer.py`` / ``budget.py`` because its concerns
are transport concerns, not scoring ones: retrying a flaky response, keeping the
outgoing rate under the API's limit, bounding how many requests are in flight at
once, and rejecting a request that would break a hard Jev limit before it ever
reaches the network.

``HttpJevClient`` and ``jevctx.testing.FakeJevClient`` both implement the ``JevClient``
protocol and must reject exactly the same requests, so
``check_request_budget`` reimplements ``FakeJevClient._check_limits`` rather than
importing it: the fake is a test double free to change shape, this is production
code that must not depend on ``testing.py`` internals.

Retries use "full jitter" (``sleep = uniform(0, min(cap, base * 2**attempt))``): it
spreads retries out instead of having every caller wake up in lockstep, which matters
here because ``scorer.score_items`` fans a single admit()/retrieve() call out across
many worker threads sharing one ``HttpJevClient``. ``422`` is never retried because
a 422 means the request itself is malformed, so retrying it
just repeats the same bug instead of fixing it.
"""

from __future__ import annotations

import os
import random
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from jevctx.tokens import estimate_tokens
from jevctx.types import (
    MAX_CHOICE_OPTIONS,
    MAX_QUESTIONS_PER_REQUEST,
    MAX_SCORE_LEVELS,
    MIN_SCORE_LEVELS,
    RATE_LIMIT_RPM,
    STATE_PLUS_ALL_QUESTIONS_TOKENS,
    STATE_PLUS_LONGEST_QUESTION_TOKENS,
    Answer,
    Choice,
    JevAuthError,
    JevBudgetError,
    JevUnavailableError,
    JevValidationError,
    Question,
    Score,
    State,
    parse_answer,
)

__all__ = ["HttpJevClient", "RateLimiter", "RetryPolicy", "Usage", "check_request_budget"]


# --------------------------------------------------------------------------- #
# Usage accounting
# --------------------------------------------------------------------------- #


@dataclass
class Usage:
    """Running Jev token/request totals for one client.

    Written from every worker thread that calls ``ask()`` (the scorer fans a single
    client out across a thread pool), so mutation goes through ``record()``, which
    holds its own lock rather than trusting callers to synchronise. Plain-field
    reads need no lock: a snapshot that is a request or two stale is harmless for
    cost reporting, and CPython attribute reads cannot tear.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False,
                                   compare=False)

    def record(self, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.requests += 1


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #


#: Tolerance on the "do we have a whole token" check, so that a wait computed from
#: one floating-point division and then repaid via a multiply by its reciprocal
#: can't leave `_tokens` a hair under 1.0 and force a second, near-zero wait.
_TOKEN_EPSILON = 1e-9


class RateLimiter:
    """Token bucket: at most ``rate`` admissions per ``per`` seconds, steady state.

    The bucket starts full, so an initial burst of up to ``rate`` calls is admitted
    with no wait at all -- that matches a request-rate *budget* like Jev's, not a
    fixed schedule. ``clock`` and ``sleep`` are injectable so tests can drive the
    bucket deterministically instead of waiting on a wall clock.
    """

    def __init__(
        self,
        rate: int,
        per: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        if per <= 0:
            raise ValueError("per must be positive")
        self._capacity = float(rate)
        self._refill_per_second = rate / per
        self._clock = clock
        self._sleep = sleep
        self._tokens = float(rate)
        self._last_refill = clock()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block (via the injected ``sleep``) until one admission is available."""
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= 1.0 - _TOKEN_EPSILON:
                    self._tokens = max(0.0, self._tokens - 1.0)
                    return
                wait = (1.0 - self._tokens) / self._refill_per_second
            self._sleep(wait)

    def _refill_locked(self) -> None:
        now = self._clock()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_second)
            self._last_refill = now


# --------------------------------------------------------------------------- #
# Retries
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter, capped, honouring a server ``Retry-After``.

    ``rng=None`` draws from the shared ``random`` module; pass a private
    ``random.Random`` for a reproducible sequence in a test.
    """

    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    rng: random.Random | None = None

    def delay(self, attempt: int, *, retry_after: float | None = None) -> float:
        """Seconds to sleep before the retry following failed ``attempt`` (0-indexed).

        A server-supplied ``retry_after`` wins outright: it is a more informed
        instruction than our own guess, so no jitter is applied on top of it.
        """
        if retry_after is not None:
            return max(retry_after, 0.0)
        cap = min(self.max_delay, self.base_delay * (2**attempt))
        uniform = self.rng.uniform if self.rng is not None else random.uniform
        return uniform(0.0, cap)


def _parse_retry_after(value: str) -> float | None:
    """Parse a ``Retry-After`` header: either delta-seconds or an HTTP-date."""
    text = value.strip()
    try:
        return max(float(text), 0.0)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max((when - datetime.now(UTC)).total_seconds(), 0.0)


# --------------------------------------------------------------------------- #
# Pre-send budget check
# --------------------------------------------------------------------------- #


def check_request_budget(state: State, questions: Mapping[str, Question]) -> None:
    """Raise ``JevBudgetError`` if ``(state, questions)`` would violate a hard Jev limit.

    Mirrors ``jevctx.testing.FakeJevClient._check_limits`` -- see the module
    docstring for why this is a reimplementation rather than a shared function.
    """
    if not questions:
        raise JevBudgetError("a request must carry at least one question")
    if len(questions) > MAX_QUESTIONS_PER_REQUEST:
        raise JevBudgetError(
            f"{len(questions)} questions exceeds the per-request limit of "
            f"{MAX_QUESTIONS_PER_REQUEST}"
        )

    question_tokens: dict[str, int] = {}
    for key, question in questions.items():
        if isinstance(question, Choice) and len(question.criteria) > MAX_CHOICE_OPTIONS:
            raise JevBudgetError(
                f"question {key!r} has {len(question.criteria)} choice options, "
                f"the limit is {MAX_CHOICE_OPTIONS}"
            )
        if isinstance(question, Score):
            levels = len(question.criteria)
            if not MIN_SCORE_LEVELS <= levels <= MAX_SCORE_LEVELS:
                raise JevBudgetError(
                    f"question {key!r} has {levels} score levels, needs "
                    f"{MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS}"
                )
        question_tokens[key] = estimate_tokens(question.to_payload())

    state_tokens = estimate_tokens(state)
    total_tokens = state_tokens + sum(question_tokens.values())
    if total_tokens > STATE_PLUS_ALL_QUESTIONS_TOKENS:
        raise JevBudgetError(
            f"state + all questions estimated at {total_tokens} tokens, over the "
            f"{STATE_PLUS_ALL_QUESTIONS_TOKENS} limit"
        )
    longest_tokens = state_tokens + max(question_tokens.values())
    if longest_tokens > STATE_PLUS_LONGEST_QUESTION_TOKENS:
        raise JevBudgetError(
            f"state + longest question estimated at {longest_tokens} tokens, over "
            f"the {STATE_PLUS_LONGEST_QUESTION_TOKENS} limit"
        )


def _to_jsonable(value: State) -> Any:
    """Coerce a ``State`` into plain ``dict``/``list`` so it always serialises as JSON.

    ``json.dumps`` only special-cases actual ``dict``/``list``/``tuple`` instances,
    so a caller-supplied ``Mapping``/``Sequence`` that satisfies the ``State``
    contract (the ``State`` alias in types.py) without being one of those -- a
    ``MappingProxyType``, a custom view -- would otherwise fail to encode.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence):
        return [_to_jsonable(item) for item in value]
    return value


def _error_detail(response: httpx.Response) -> str:
    """Best-effort human-readable detail from an error response, for exception text."""
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:300] if text else response.reason_phrase
    if isinstance(payload, Mapping):
        detail = payload.get("error") or payload.get("message")
        if detail:
            return str(detail)
    return str(payload)


# --------------------------------------------------------------------------- #
# The client
# --------------------------------------------------------------------------- #


class HttpJevClient:
    """``JevClient`` over HTTP, against ``POST {base_url}/systemone``.

    Three independent safety mechanisms wrap the one HTTP call: ``check_request_budget``
    rejects an oversized request before it touches the network; a
    ``threading.Semaphore`` bounds in-flight requests at ``max_concurrency``; and a
    ``RateLimiter`` keeps the outgoing rate under ``RATE_LIMIT_RPM``. All three matter
    because ``ask()`` is meant to be called concurrently, from every worker thread
    ``scorer.score_items`` spawns against one shared client.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "jev-latest",
        base_url: str = "https://api.typesafe.ai/v1",
        timeout: float = 15.0,
        max_retries: int = 3,
        max_concurrency: int = 16,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise JevAuthError(
                "no Jev API key: pass api_key=, or set the TYPESAFE_API_KEY env var"
            )
        self._model = model
        self._timeout = timeout
        self._sleep = sleep
        self._retry_policy = RetryPolicy(max_retries=max_retries, rng=rng)
        self._concurrency = threading.Semaphore(max_concurrency)
        self._rate_limiter = RateLimiter(RATE_LIMIT_RPM, 60.0, sleep=sleep)
        self.usage = Usage()
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {key}"},
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpJevClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.close()

    @property
    def endpoint(self) -> str:
        """The full URL this client posts to. Handy when a check has to report it."""
        return f"{str(self._client.base_url).rstrip('/')}/systemone"

    @property
    def model(self) -> str:
        return self._model

    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        check_request_budget(state, questions)
        body: dict[str, Any] = {
            "model": self._model,
            "state": _to_jsonable(state),
            "questions": {key: question.to_payload() for key, question in questions.items()},
        }

        last_error: Exception | None = None
        with self._concurrency:
            for attempt in range(self._retry_policy.max_retries + 1):
                is_last_attempt = attempt == self._retry_policy.max_retries
                self._rate_limiter.acquire()
                try:
                    response = self._client.post("/systemone", json=body)
                except httpx.TransportError as exc:
                    last_error = exc
                    if is_last_attempt:
                        break
                    self._sleep(self._retry_policy.delay(attempt))
                    continue

                if response.status_code == 200:
                    return self._parse_response(response, questions)
                if response.status_code == 401:
                    raise JevAuthError(_error_detail(response))
                if response.status_code == 422:
                    raise JevValidationError(_error_detail(response))
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = JevUnavailableError(
                        f"Jev returned {response.status_code}: {_error_detail(response)}"
                    )
                    if is_last_attempt:
                        break
                    retry_after = self._retry_after_seconds(response)
                    self._sleep(self._retry_policy.delay(attempt, retry_after=retry_after))
                    continue
                # Any other 4xx: the request is bad in some way we don't special-case.
                raise JevValidationError(
                    f"Jev returned {response.status_code}: {_error_detail(response)}"
                )

        raise JevUnavailableError(
            f"Jev request failed after {self._retry_policy.max_retries + 1} attempt(s)"
        ) from last_error

    def _retry_after_seconds(self, response: httpx.Response) -> float | None:
        header = response.headers.get("retry-after")
        if header is None:
            return None
        seconds = _parse_retry_after(header)
        if seconds is None:
            return None
        return min(seconds, self._timeout)

    def _parse_response(
        self, response: httpx.Response, questions: Mapping[str, Question]
    ) -> dict[str, Answer]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise JevValidationError(f"Jev response was not valid JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise JevValidationError("Jev response body must be a JSON object")
        raw_answers = payload.get("answers")
        if not isinstance(raw_answers, Mapping):
            raise JevValidationError("Jev response is missing an 'answers' object")

        answers: dict[str, Answer] = {}
        for key in questions:
            if key not in raw_answers:
                raise JevValidationError(f"Jev response is missing an answer for {key!r}")
            answers[key] = parse_answer(raw_answers[key])

        usage = payload.get("usage")
        usage_map = usage if isinstance(usage, Mapping) else {}
        self.usage.record(
            input_tokens=int(usage_map.get("input_tokens", 0) or 0),
            output_tokens=int(usage_map.get("output_tokens", 0) or 0),
        )
        return answers
