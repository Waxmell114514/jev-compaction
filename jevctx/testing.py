"""Test doubles.

FROZEN CONTRACT -- do not edit. See SPEC.md section 2 and section 7.

``FakeJevClient`` is the reason no test in this package needs an API key. It
enforces the *same* hard limits as the real transport, so a batching bug shows up
as a test failure rather than a 422 in production.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from jevctx.tokens import estimate_tokens
from jevctx.types import (
    MAX_CHOICE_OPTIONS,
    MAX_QUESTIONS_PER_REQUEST,
    MAX_SCORE_LEVELS,
    MIN_SCORE_LEVELS,
    STATE_PLUS_ALL_QUESTIONS_TOKENS,
    STATE_PLUS_LONGEST_QUESTION_TOKENS,
    Answer,
    Choice,
    ChoiceAnswer,
    JevBudgetError,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    State,
)

__all__ = ["RecordedCall", "FakeJevClient"]


@dataclass(frozen=True)
class RecordedCall:
    """One request as the fake received it."""

    state: State
    questions: Mapping[str, Question]

    def state_text(self) -> str:
        """Everything in the state, flattened, for substring assertions."""
        parts: list[str] = []

        def walk(obj: Any) -> None:
            if isinstance(obj, str):
                parts.append(obj)
            elif isinstance(obj, Mapping):
                for key, value in obj.items():
                    parts.append(str(key))
                    walk(value)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    walk(item)
            elif obj is not None:
                parts.append(str(obj))

        walk(self.state)
        return "\n".join(parts)


AnswerFn = Callable[[State, Mapping[str, Question], str], "float | str | Answer"]


class FakeJevClient:
    """A deterministic, offline ``JevClient``.

    ``answer_fn(state, questions, key)`` may return a float (a noul probability or
    a score), a string (a choice), or a fully built ``Answer``.
    """

    def __init__(
        self,
        answer_fn: AnswerFn | None = None,
        *,
        raises: Exception | None = None,
        enforce_limits: bool = True,
    ) -> None:
        self._answer_fn: AnswerFn = answer_fn or (lambda s, q, k: 1.0)
        self._raises = raises
        self._enforce_limits = enforce_limits
        self._lock = threading.Lock()
        self.calls: list[RecordedCall] = []

    # -- constructors ------------------------------------------------------- #

    @classmethod
    def constant(cls, value: float = 1.0, **kw: Any) -> FakeJevClient:
        return cls(lambda s, q, k: value, **kw)

    @classmethod
    def scripted(cls, answers: Mapping[str, float | str | Answer],
                 default: float = 1.0, **kw: Any) -> FakeJevClient:
        """Answer by question key, falling back to ``default``."""
        return cls(lambda s, q, k: answers.get(k, default), **kw)

    @classmethod
    def by_text(cls, fn: Callable[[str], float], default: float = 1.0,
                **kw: Any) -> FakeJevClient:
        """Answer by scoring the text of the item the question refers to.

        Looks the item up in a state shaped ``{"items": [{"ref": ..., "text": ...}]}``.
        """

        def answer(state: State, questions: Mapping[str, Question], key: str) -> float:
            if isinstance(state, Mapping):
                for item in state.get("items") or []:
                    if isinstance(item, Mapping) and item.get("ref") == key:
                        return fn(str(item.get("text", "")))
            return default

        return cls(answer, **kw)

    @classmethod
    def failing(cls, exc: Exception, **kw: Any) -> FakeJevClient:
        return cls(raises=exc, **kw)

    # -- the protocol ------------------------------------------------------- #

    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        with self._lock:
            self.calls.append(RecordedCall(state=state, questions=dict(questions)))
        if self._raises is not None:
            raise self._raises
        if self._enforce_limits:
            self._check_limits(state, questions)
        return {key: self._build(state, questions, key, q) for key, q in questions.items()}

    # -- internals ---------------------------------------------------------- #

    def _build(self, state: State, questions: Mapping[str, Question],
               key: str, question: Question) -> Answer:
        raw = self._answer_fn(state, questions, key)
        if isinstance(raw, (NoulAnswer, ChoiceAnswer, ScoreAnswer)):
            return raw
        if isinstance(question, Noul):
            return NoulAnswer(noul=float(raw))
        if isinstance(question, Choice):
            options = list(question.criteria)
            chosen = str(raw) if str(raw) in question.criteria else options[0]
            probs = {o: (1.0 if o == chosen else 0.0) for o in options}
            return ChoiceAnswer(choice=chosen, probabilities=probs, confidence=1.0)
        if isinstance(question, Score):
            levels = list(question.criteria)
            value = float(raw)
            legend = {str(i): name for i, name in enumerate(levels)}
            return ScoreAnswer(score=value, legend=legend,
                               probabilities={str(i): 0.0 for i in range(len(levels))},
                               confidence=1.0)
        raise TypeError(f"unsupported question type: {type(question)!r}")

    def _check_limits(self, state: State, questions: Mapping[str, Question]) -> None:
        if not questions:
            raise JevBudgetError("a request must carry at least one question")
        if len(questions) > MAX_QUESTIONS_PER_REQUEST:
            raise JevBudgetError(
                f"{len(questions)} questions exceeds the limit of {MAX_QUESTIONS_PER_REQUEST}"
            )
        for key, question in questions.items():
            if isinstance(question, Choice) and len(question.criteria) > MAX_CHOICE_OPTIONS:
                raise JevBudgetError(f"question {key!r}: >{MAX_CHOICE_OPTIONS} choice options")
            if isinstance(question, Score):
                n = len(question.criteria)
                if not (MIN_SCORE_LEVELS <= n <= MAX_SCORE_LEVELS):
                    raise JevBudgetError(
                        f"question {key!r}: score needs {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} "
                        f"levels, got {n}"
                    )
        state_tokens = estimate_tokens(state)
        q_tokens = {k: estimate_tokens(q.to_payload()) for k, q in questions.items()}
        if state_tokens + sum(q_tokens.values()) > STATE_PLUS_ALL_QUESTIONS_TOKENS:
            raise JevBudgetError(
                f"state+questions = {state_tokens + sum(q_tokens.values())} tokens exceeds "
                f"{STATE_PLUS_ALL_QUESTIONS_TOKENS}"
            )
        if state_tokens + max(q_tokens.values()) > STATE_PLUS_LONGEST_QUESTION_TOKENS:
            raise JevBudgetError(
                f"state+longest question = {state_tokens + max(q_tokens.values())} tokens "
                f"exceeds {STATE_PLUS_LONGEST_QUESTION_TOKENS}"
            )
