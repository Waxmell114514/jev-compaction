"""Tests for the live check.

The check is the one thing in this package that talks to the network, so its
failure paths are the ones a new user actually meets. They are tested here
offline; only the happy path is boring.
"""

from __future__ import annotations

import re

import pytest

from jevctx.check import run_check
from jevctx.testing import FakeJevClient
from jevctx.types import (
    Choice,
    JevAuthError,
    JevUnavailableError,
    JevValidationError,
    Score,
)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def answering(*, choice: str = "build", noul: float = 0.97, score: float = 2.0):
    def answer(state, questions, key):
        question = questions[key]
        if isinstance(question, Choice):
            return choice
        if isinstance(question, Score):
            return score
        if key == "is_question":
            return noul
        if isinstance(state, dict):
            for item in state.get("items") or []:
                if item.get("ref") == key:
                    return 0.95 if "ERR" in item["text"].upper() else 0.05
        return 1.0

    return answer


def run(factory, capsys) -> tuple[int, str]:
    code = run_check(factory)
    return code, _ANSI.sub("", capsys.readouterr().out)


def test_no_key_explains_what_to_do(capsys, monkeypatch) -> None:
    for var in ("TYPESAFE_API_KEY", "JEV_API_KEY", "JEV_BASE_URL", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    code, out = run(None, capsys)
    assert code == 2
    assert "TYPESAFE_API_KEY" in out
    assert "openrouter.ai/api/alpha" in out, "must say how to use Jev on OpenRouter"
    assert "python demo.py" in out, "must point at the offline path that still works"


def test_happy_path(capsys) -> None:
    code, out = run(lambda: FakeJevClient(answering()), capsys)
    assert code == 0
    assert "Everything works" in out
    assert "byte-exact" in out
    assert "shadow_only=True" in out, "must land the user on the safe first step"


@pytest.mark.parametrize(
    ("error", "code", "must_say"),
    [
        (JevAuthError("bad key"), 2, "rejected the key"),
        (JevValidationError("malformed"), 1, "bug in jevctx"),
        (JevUnavailableError("timeout"), 1, "fails open"),
    ],
)
def test_each_failure_is_diagnosed(error, code, must_say, capsys) -> None:
    got, out = run(lambda: FakeJevClient.failing(error), capsys)
    assert got == code
    assert must_say in out


def test_a_choice_outside_the_offered_options_is_caught(capsys) -> None:
    """A model answering off-menu would be a parsing bug worth failing on."""
    client = FakeJevClient(answering())

    def rogue(state, questions, key):
        from jevctx.types import ChoiceAnswer
        if isinstance(questions[key], Choice):
            return ChoiceAnswer(choice="not-an-option", probabilities={}, confidence=1.0)
        return answering()(state, questions, key)

    code, out = run(lambda: FakeJevClient(rogue), capsys)
    assert code == 1
    assert "was not offered" in out
    assert client is not None


def test_a_missing_answer_is_caught(capsys) -> None:
    class Forgetful(FakeJevClient):
        def ask(self, state, questions):
            answers = super().ask(state, questions)
            answers.pop("urgency", None)
            return answers

    code, out = run(lambda: Forgetful(answering()), capsys)
    assert code == 1
    assert "did not answer" in out
    assert "urgency" in out


def test_a_scorer_that_keeps_everything_is_reported(capsys) -> None:
    code, out = run(lambda: FakeJevClient.constant(1.0), capsys)
    assert code == 1
    assert "nothing was relocated" in out


def test_the_tripwire_firing_is_reported_not_celebrated(capsys) -> None:
    code, out = run(lambda: FakeJevClient.constant(0.0), capsys)
    assert code == 1
    assert "tripwire" in out
