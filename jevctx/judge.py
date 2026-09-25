"""Judges: whatever answers the gate's questions.

Every module in this package asks its questions through one protocol,
``JevClient.ask(state, questions)``: yes/no (``Noul``), pick-one (``Choice``) and
ordered-scale (``Score``) questions about a JSON or text ``state``. Jev answers them
natively. Any chat model can answer them too, if it is told the answer shapes and
replies in JSON, which is what ``LLMJudgeClient`` does over the OpenAI-compatible
``POST {base_url}/chat/completions`` that OpenAI, OpenRouter, DeepSeek, Groq, Gemini,
vLLM, llama.cpp, LM Studio and Ollama all serve.

``make_judge()`` builds the one the environment asks for:

- ``JEVCTX_JUDGE``: ``jev`` (default) or ``llm``
- for ``jev``: see :mod:`jevctx.jev` (``TYPESAFE_API_KEY``, or ``JEV_BASE_URL`` + key)
- for ``llm``: ``JUDGE_BASE_URL`` and ``JUDGE_MODEL`` (required), ``JUDGE_API_KEY``
  (omit for a local server that needs none), ``JUDGE_JSON_MODE=0`` for a server that
  rejects ``response_format``, ``JUDGE_RPM`` to cap requests per minute

A chat model's probabilities are its own estimates, not a classifier's, so fit the
gate's thresholds to the judge you use (``python -m jevctx.calibrate``). The SWE-bench
numbers in RESULTS.md were measured with Jev.
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from jevctx.jev import (
    HttpJevClient,
    RateLimiter,
    RetryPolicy,
    Usage,
    _error_detail,
    _parse_retry_after,
    _to_jsonable,
    check_request_budget,
    resolve_endpoint,
)
from jevctx.types import (
    Answer,
    Choice,
    ChoiceAnswer,
    JevAuthError,
    JevRejectedError,
    JevUnavailableError,
    JevValidationError,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    State,
)

__all__ = ["JUDGES", "JudgeSpec", "LLMJudgeClient", "make_judge", "resolve_judge"]

JUDGES = ("jev", "llm")

SYSTEM_PROMPT = """\
You are a careful judge. You are given a STATE and QUESTIONS about it, each under a key. \
Answer every question from the state alone.

Answer shapes, by question type:
- "noul": a yes/no question. Answer {"p": P}, where P is the probability, from 0 to 1, \
that the "true" criterion holds rather than the "false" one.
- "choice": pick one of the named options. Answer {"probabilities": {"<option>": P, ...}} \
over the options, summing to 1.
- "score": place the state on an ordered scale whose levels are numbered from 0. Answer \
{"probabilities": {"<level number>": P, ...}}, summing to 1.

Probabilities should be calibrated: near 0 or 1 only when the state makes it clear, \
near 0.5 when it does not.

Reply with one JSON object and nothing else. Its keys are exactly the question keys, and \
each value is the answer object for that question."""


class _Malformed(Exception):
    """The model replied, but not with a usable answer for every question."""


def _question_payload(question: Question) -> dict[str, Any]:
    payload = question.to_payload()
    if isinstance(question, Score):
        # Number the levels, so "level 2" in the answer is unambiguous.
        payload["criteria"] = {str(i): level for i, level in enumerate(question.criteria)}
    return payload


def _probability(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _Malformed(f"not a probability: {value!r}") from exc
    if not math.isfinite(number):
        raise _Malformed(f"not a probability: {value!r}")
    return min(max(number, 0.0), 1.0)


def _weight(value: Any) -> float:
    """A non-negative weight: models write percentages as often as probabilities."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _Malformed(f"not a weight: {value!r}") from exc
    if not math.isfinite(number) or number < 0:
        raise _Malformed(f"not a weight: {value!r}")
    return number


def _distribution(raw: Any, options: list[str]) -> dict[str, float]:
    """A normalised distribution over ``options`` from whatever the model returned."""
    if isinstance(raw, Mapping):
        probs = raw.get("probabilities")
        if isinstance(probs, Mapping):
            found = {str(k): _weight(v) for k, v in probs.items() if str(k) in options}
            total = sum(found.values())
            if total > 0:
                return {option: found.get(option, 0.0) / total for option in options}
        raw = raw.get("choice", raw.get("level"))
    if raw is not None and str(raw) in options:
        return {option: float(option == str(raw)) for option in options}
    raise _Malformed("no usable distribution over the options")


def _answer(question: Question, raw: Any) -> Answer:
    if isinstance(question, Noul):
        if isinstance(raw, Mapping):
            raw = next((raw[k] for k in ("p", "probability", "noul") if k in raw), None)
        return NoulAnswer(noul=_probability(raw))
    if isinstance(question, Choice):
        probs = _distribution(raw, list(question.criteria))
        best = max(probs, key=probs.__getitem__)
        return ChoiceAnswer(choice=best, probabilities=probs, confidence=probs[best])
    levels = [str(i) for i in range(len(question.criteria))]
    probs = _distribution(raw, levels)
    return ScoreAnswer(
        score=sum(int(level) * p for level, p in probs.items()),
        legend=dict(zip(levels, question.criteria, strict=True)),
        probabilities=probs,
        confidence=max(probs.values()),
    )


def _json_object(text: str) -> Mapping[str, Any]:
    """The JSON object in a reply, tolerating code fences or prose around it."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise _Malformed("the reply holds no JSON object")
    try:
        payload = json.loads(text[start:end + 1])
    except ValueError as exc:
        raise _Malformed(f"the reply is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise _Malformed("the reply is not a JSON object")
    return payload


class LLMJudgeClient:
    """``JevClient`` backed by any OpenAI-compatible chat model.

    Settings left as ``None`` come from ``JUDGE_*`` environment variables (see the
    module docstring). A request carries the state and every question once, and gets
    one JSON object back; a reply that misses a question or is not JSON is retried like
    a 5xx (without ``response_format``, which some providers' JSON mode mangles), then
    raised as ``JevUnavailableError`` so the gate fails open. It enforces
    Jev's request limits too, so callers batch exactly as they do for Jev.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        json_mode: bool | None = None,
        rpm: int | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
        max_retries: int = 3,
        max_concurrency: int = 8,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        env = os.environ if env is None else env
        base_url = base_url or env.get("JUDGE_BASE_URL")
        model = model or env.get("JUDGE_MODEL")
        if not base_url or not model:
            raise JevAuthError("the llm judge needs JUDGE_BASE_URL and JUDGE_MODEL")
        api_key = (api_key or env.get("JUDGE_API_KEY") or "").strip() or None
        if json_mode is None:
            json_mode = env.get("JUDGE_JSON_MODE", "1") != "0"
        if rpm is None and env.get("JUDGE_RPM"):
            rpm = int(env["JUDGE_RPM"])
        self._model = model
        self._json_mode = json_mode
        self._temperature = temperature
        self._timeout = timeout
        self._sleep = sleep
        self._retry_policy = RetryPolicy(max_retries=max_retries, rng=rng)
        self._concurrency = threading.Semaphore(max_concurrency)
        self._rate_limiter = RateLimiter(rpm, 60.0, sleep=sleep) if rpm else None
        self.usage = Usage()
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LLMJudgeClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def endpoint(self) -> str:
        return str(self._client.base_url).rstrip("/") + "/chat/completions"

    @property
    def model(self) -> str:
        return self._model

    def request_body(self, state: State, questions: Mapping[str, Question]) -> dict[str, Any]:
        """The chat request for one ``ask``: public so a caller can preview it."""
        plain = _to_jsonable(state)
        state_text = plain if isinstance(plain, str) else json.dumps(plain, ensure_ascii=False)
        asked = {key: _question_payload(question) for key, question in questions.items()}
        body: dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"STATE:\n{state_text}\n\nQUESTIONS:\n"
                                            f"{json.dumps(asked, ensure_ascii=False)}"},
            ],
        }
        if self._json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        check_request_budget(state, questions)
        body = self.request_body(state, questions)
        last_error: Exception | None = None
        with self._concurrency:
            for attempt in range(self._retry_policy.max_retries + 1):
                is_last_attempt = attempt == self._retry_policy.max_retries
                if self._rate_limiter is not None:
                    self._rate_limiter.acquire()
                retry_after = None
                try:
                    response = self._client.post("/chat/completions", json=body)
                    if response.status_code == 200:
                        return self._parse(response, questions)
                    self._raise_for_status(response)
                    last_error = JevUnavailableError(
                        f"judge returned {response.status_code}: {_error_detail(response)}")
                    header = response.headers.get("retry-after")
                    seconds = _parse_retry_after(header) if header else None
                    retry_after = min(seconds, self._timeout) if seconds is not None else None
                except httpx.TransportError as exc:
                    last_error = exc
                except _Malformed as exc:
                    last_error = exc
                    # Some providers' JSON mode mangles the answer (seen: a reasoning
                    # model that worked it out, then emitted {"": "<key>"}), so the
                    # retries ask without it and rely on the prompt alone.
                    body.pop("response_format", None)
                if is_last_attempt:
                    break
                self._sleep(self._retry_policy.delay(attempt, retry_after=retry_after))
        # A transport error's text can quote request headers, so name only its class.
        reason = last_error if isinstance(last_error, _Malformed | JevUnavailableError) \
            else type(last_error).__name__
        raise JevUnavailableError(
            f"judge request failed after {self._retry_policy.max_retries + 1} attempt(s): {reason}"
        ) from last_error

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        """Raise for a status that retrying cannot fix; return for 429 and 5xx."""
        status = response.status_code
        if status == 429 or status >= 500:
            return
        if status == 401:
            raise JevAuthError(_error_detail(response))
        if status == 403:
            raise JevRejectedError("judge returned 403: request refused")
        raise JevValidationError(f"judge returned {status}: {_error_detail(response)}")

    def _parse(self, response: httpx.Response, questions: Mapping[str, Question]
               ) -> dict[str, Answer]:
        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise _Malformed("not a chat completion") from exc
        usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
        self.usage.record(input_tokens=int(usage.get("prompt_tokens", 0) or 0),
                          output_tokens=int(usage.get("completion_tokens", 0) or 0))
        replies = _json_object(content or "")
        missing = [key for key in questions if key not in replies]
        if missing:
            raise _Malformed(f"no answer for {', '.join(missing[:5])}")
        return {key: _answer(question, replies[key]) for key, question in questions.items()}


# --------------------------------------------------------------------------- #
# Picking a judge
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class JudgeSpec:
    """Which judge the environment asks for, and whether it is ready to build.

    ``missing`` names what to set when it is not.
    """

    kind: str
    url: str
    model: str
    missing: str | None

    @property
    def ready(self) -> bool:
        return self.missing is None


def resolve_judge(env: Mapping[str, str] | None = None) -> JudgeSpec:
    env = os.environ if env is None else env
    kind = (env.get("JEVCTX_JUDGE") or "jev").strip().lower()
    if kind == "jev":
        endpoint = resolve_endpoint(env=env)
        return JudgeSpec("jev", endpoint.url, endpoint.model,
                         None if endpoint.api_key else endpoint.key_source)
    if kind == "llm":
        base_url, model = env.get("JUDGE_BASE_URL", ""), env.get("JUDGE_MODEL", "")
        unset = [name for name, value in (("JUDGE_BASE_URL", base_url),
                                          ("JUDGE_MODEL", model)) if not value]
        return JudgeSpec("llm", base_url.rstrip("/") + "/chat/completions", model,
                         " and ".join(unset) or None)
    raise ValueError(f"JEVCTX_JUDGE must be one of {', '.join(JUDGES)}, not {kind!r}")


def make_judge(env: Mapping[str, str] | None = None) -> HttpJevClient | LLMJudgeClient:
    """The judge ``JEVCTX_JUDGE`` names, configured from the environment."""
    env = os.environ if env is None else env
    if resolve_judge(env).kind == "llm":
        return LLMJudgeClient(env=env)
    endpoint = resolve_endpoint(env=env)
    if not endpoint.api_key:
        raise JevAuthError(f"no Jev API key: set {endpoint.key_source}")
    return HttpJevClient(endpoint.api_key, base_url=endpoint.base_url, path=endpoint.path,
                         model=endpoint.model)
