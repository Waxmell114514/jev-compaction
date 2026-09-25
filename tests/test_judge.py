"""The judge seam: any chat model answering the gate's questions, and picking one."""

from __future__ import annotations

import json
import random

import httpx
import pytest

from jevctx.check import run_check
from jevctx.jev import HttpJevClient
from jevctx.judge import LLMJudgeClient, make_judge, resolve_judge
from jevctx.pipeline import admit, reconstruct
from jevctx.shadow import ShadowLog
from jevctx.store import InMemoryStore
from jevctx.types import (
    Choice,
    ChoiceAnswer,
    JevAuthError,
    JevUnavailableError,
    JevValidationError,
    Noul,
    NoulAnswer,
    Origin,
    Score,
    ScoreAnswer,
)
from tests.test_pipeline import TASK, noisy_log

BASE = "https://llm.example/v1"
NOUL = Noul(instructions="Is it urgent?", true="urgent", false="not urgent")
CHOICE = Choice(instructions="Topic?", criteria={"build": "CI", "billing": "money"})
SCORE = Score(instructions="How bad?", criteria=["fine", "meh", "bad"])


def completion(content: str, *, prompt_tokens: int = 100) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 20},
    })


def asked(request: httpx.Request) -> tuple[object, dict]:
    """The state and questions a chat request carries."""
    user = json.loads(request.content)["messages"][1]["content"]
    state, questions = user.removeprefix("STATE:\n").split("\n\nQUESTIONS:\n")
    try:
        state = json.loads(state)
    except ValueError:
        pass
    return state, json.loads(questions)


def judge(handler, **kw) -> LLMJudgeClient:
    kw.setdefault("sleep", lambda _s: None)
    return LLMJudgeClient("small-model", base_url=BASE, api_key="k", env={},
                          transport=httpx.MockTransport(handler), rng=random.Random(0), **kw)


def test_one_request_answers_all_three_question_types() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return completion(json.dumps({
            "n": {"p": 0.8},
            "c": {"probabilities": {"build": 3, "billing": 1, "made_up": 5}},
            "s": {"probabilities": {"0": 0.0, "1": 0.5, "2": 0.5}},
        }))

    client = judge(handler)
    answers = client.ask({"message": "CI red"}, {"n": NOUL, "c": CHOICE, "s": SCORE})

    assert answers["n"] == NoulAnswer(noul=0.8)
    choice = answers["c"]
    assert isinstance(choice, ChoiceAnswer) and choice.choice == "build"
    assert choice.probabilities == {"build": 0.75, "billing": 0.25}, "unoffered options dropped"
    score = answers["s"]
    assert isinstance(score, ScoreAnswer) and score.score == pytest.approx(1.5)
    assert score.legend == {"0": "fine", "1": "meh", "2": "bad"}

    (request,) = seen
    assert str(request.url) == f"{BASE}/chat/completions"
    assert request.headers["authorization"] == "Bearer k"
    body = json.loads(request.content)
    assert body["model"] == "small-model" and body["temperature"] == 0.0
    assert body["response_format"] == {"type": "json_object"}
    state, questions = asked(request)
    assert state == {"message": "CI red"}
    assert questions["s"]["criteria"] == {"0": "fine", "1": "meh", "2": "bad"}
    assert client.usage.input_tokens == 100 and client.usage.requests == 1


def test_loose_replies_still_parse() -> None:
    reply = 'Sure!\n```json\n{"n": 0.3, "c": {"choice": "billing"}, "s": {"level": 2}}\n```'
    answers = judge(lambda r: completion(reply)).ask("x", {"n": NOUL, "c": CHOICE, "s": SCORE})
    assert answers["n"].value == 0.3
    assert answers["c"].value == "billing" and answers["c"].confidence == 1.0
    assert answers["s"].value == 2.0


def test_a_bad_reply_is_retried_then_fails_open() -> None:
    replies = iter(["no json here", '{"other": {"p": 1}}', '{"n": {"p": 0.6}}'])
    client = judge(lambda r: completion(next(replies)))
    assert client.ask("x", {"n": NOUL})["n"].value == 0.6

    always_bad = judge(lambda r: completion('{"n": {"p": "high"}}'), max_retries=2)
    with pytest.raises(JevUnavailableError, match="3 attempt"):
        always_bad.ask("x", {"n": NOUL})


def test_status_codes() -> None:
    calls = []

    def flaky(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return completion('{"n": {"p": 1}}')

    assert judge(flaky).ask("x", {"n": NOUL})["n"].value == 1.0 and len(calls) == 2
    with pytest.raises(JevAuthError):
        judge(lambda r: httpx.Response(401, json={"error": "bad key"})).ask("x", {"n": NOUL})
    with pytest.raises(JevValidationError, match="404"):
        judge(lambda r: httpx.Response(404, json={"error": "no such model"})).ask("x", {"n": NOUL})


def test_json_mode_and_key_are_optional() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return completion('{"n": {"p": 0.5}}')

    client = LLMJudgeClient(env={"JUDGE_BASE_URL": "http://localhost:11434/v1",
                                 "JUDGE_MODEL": "qwen3:4b", "JUDGE_JSON_MODE": "0"},
                            transport=httpx.MockTransport(handler))
    client.ask("x", {"n": NOUL})
    assert "authorization" not in seen[0].headers
    assert "response_format" not in json.loads(seen[0].content)
    assert client.endpoint == "http://localhost:11434/v1/chat/completions"


def test_the_gate_runs_on_a_chat_model() -> None:
    """admit() is unchanged: the same batched questions, answered by a chat model."""

    def handler(request: httpx.Request) -> httpx.Response:
        state, questions = asked(request)
        texts = {item["ref"]: item["text"] for item in state["items"]}
        return completion(json.dumps(
            {ref: {"p": 0.95 if "IMPORTANT" in texts[ref] else 0.02} for ref in questions}))

    store, raw = InMemoryStore(), noisy_log()
    result = admit(raw, Origin(source="tool:bash", ref="npm install", turn=3),
                   task_digest=TASK, turn=1, client=judge(handler), store=store,
                   log=ShadowLog(path=None))
    assert result.gated and result.pointers
    assert result.result_tokens < result.original_tokens
    assert reconstruct(result.text, store) == raw


def test_the_live_check_passes_on_a_chat_model(capsys) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        state, questions = asked(request)
        texts = {item["ref"]: item["text"] for item in state.get("items", [])} \
            if isinstance(state, dict) else {}
        replies = {}
        for key, question in questions.items():
            if question["type"] == "noul":
                replies[key] = {"p": 0.05 if "npm http fetch" in texts.get(key, "") else 0.9}
            else:
                replies[key] = {"probabilities": {next(iter(question["criteria"])): 1}}
        return completion(json.dumps(replies))

    assert run_check(lambda: judge(handler)) == 0, capsys.readouterr().out


JUDGE_ENV = {"JUDGE_BASE_URL": BASE, "JUDGE_MODEL": "m"}


def test_the_environment_picks_the_judge() -> None:
    assert resolve_judge({}).kind == "jev" and not resolve_judge({}).ready
    assert isinstance(make_judge({"TYPESAFE_API_KEY": "t"}), HttpJevClient)

    spec = resolve_judge({"JEVCTX_JUDGE": "llm"})
    assert spec.missing == "JUDGE_BASE_URL and JUDGE_MODEL"
    with pytest.raises(JevAuthError, match="JUDGE_BASE_URL"):
        make_judge({"JEVCTX_JUDGE": "llm"})

    spec = resolve_judge({"JEVCTX_JUDGE": "LLM", **JUDGE_ENV})
    assert spec.ready and spec.url == f"{BASE}/chat/completions" and spec.model == "m"
    built = make_judge({"JEVCTX_JUDGE": "llm", **JUDGE_ENV})
    assert isinstance(built, LLMJudgeClient) and built.model == "m"

    with pytest.raises(ValueError, match="jev, llm"):
        resolve_judge({"JEVCTX_JUDGE": "gpt"})
