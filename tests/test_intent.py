"""Goal conditioning: judging an output against what the agent was looking for."""

from __future__ import annotations

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import httpx
import pytest

from jevctx.agent import INTENT_PARAMETER, run_agent
from jevctx.pipeline import (
    ADMIT_INTENT_QUESTION,
    ADMIT_QUESTION,
    GateConfig,
    admit,
    intent_digest,
    reconstruct,
)
from jevctx.serve import SidecarState, make_server
from jevctx.shadow import ShadowLog
from jevctx.store import InMemoryStore
from jevctx.testing import FakeJevClient
from jevctx.types import Origin

TASK = "tests/test_dates.py fails: parse_month('13') should raise ValueError."
INTENT = "Let me find where parse_month validates the month."
ORIGIN = Origin(source="tool:read", ref="call-1", turn=2)


def source_file() -> str:
    """A module where only some functions bear on the intent."""
    blocks = []
    for i in range(4):
        blocks.append("\n".join([f"def format_date_{i}(d):"] +
                                [f"    part_{j} = d.strftime('%Y-%m-%d')  # formatting {i}.{j}"
                                 for j in range(12)]))
        blocks.append("\n".join([f"def parse_month_{i}(text):"] +
                                [f"    value_{j} = int(text)  # parse_month step {i}.{j}"
                                 for j in range(12)]))
    return "\n\n".join(blocks) + "\n"


def judge(state, questions, key):
    """Everything looks worth keeping for the task alone; the intent narrows it."""
    ref = key.partition(":")[0]
    text = next(item["text"] for item in state["items"] if item["ref"] == ref)
    dimension = key.partition(":")[2] or "keep"
    if dimension == "injection":
        return 0.0
    if dimension in {"type", "role", "lifetime"}:
        return {"type": "source_code", "role": "reference", "lifetime": "task"}[dimension]
    if "intent" not in state:
        return 0.9
    return 0.95 if "parse_month" in text else 0.03


def gate(intent: str | None, **config):
    client, store = FakeJevClient(judge), InMemoryStore()
    raw = source_file()
    result = admit(raw, ORIGIN, task_digest=TASK, turn=2, client=client, store=store,
                   log=ShadowLog(), config=GateConfig(**config), intent=intent)
    return result, client, store, raw


def test_without_an_intent_the_request_is_unchanged() -> None:
    result, client, _, _ = gate(None)
    assert not result.pointers
    for call in client.calls:
        assert set(call.state) == {"task", "items"}
        assert all(q.instructions.endswith(ADMIT_QUESTION.instructions)
                   for q in call.questions.values())


@pytest.mark.parametrize("profile", [False, True])
def test_the_intent_narrows_what_is_kept(profile: bool) -> None:
    result, client, store, raw = gate(INTENT, profile=profile)
    assert result.pointers, "code unrelated to the intent should be relocated"
    body = "\n".join(line for line in result.text.splitlines()
                     if not line.startswith("[[elided"))
    assert "def parse_month_0" in body and "def format_date_0" not in body
    assert reconstruct(result.text, store) == raw
    for call in client.calls:
        assert call.state["intent"] == INTENT and list(call.state) == ["task", "intent", "items"]
        keep = [q for k, q in call.questions.items() if k.partition(":")[2] in {"", "keep"}]
        assert keep and all(q.instructions.endswith(ADMIT_INTENT_QUESTION.instructions)
                            for q in keep)


def test_the_intent_is_logged_and_kept_with_the_output() -> None:
    log = ShadowLog()
    store = InMemoryStore()
    admit(source_file(), ORIGIN, task_digest=TASK, turn=2, client=FakeJevClient(judge),
          store=store, log=log, config=GateConfig(profile=True), intent=INTENT)
    assert all(entry["labels"]["intent"] == INTENT for entry in log.entries())
    full = [r for r in store.all_records() if r.kind == "tool_output"]
    assert full and full[0].meta["intent"] == INTENT


def test_intent_digest_keeps_the_words_nearest_the_call() -> None:
    assert intent_digest(None) is None and intent_digest("  \n ") is None
    assert intent_digest("Let me\n  look   at it.") == "Let me look at it."
    long = "Earlier I read the README. " * 50 + "Now let me open parse.py."
    digest = intent_digest(long, max_chars=60)
    assert digest.endswith("Now let me open parse.py.") and len(digest) <= 60
    assert long.endswith(digest) and long[-len(digest) - 1] == " ", "starts at a word"


# --------------------------------------------------------------------------- #
# The agent loop
# --------------------------------------------------------------------------- #

TOOL = {"type": "function", "function": {
    "name": "read", "description": "Read the module",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}}


def completion(content=None, calls=None) -> httpx.Response:
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return httpx.Response(200, json={"choices": [{
        "message": message, "finish_reason": "tool_calls" if calls else "stop"}]})


def agent(intent: str, arguments: dict, *, content=INTENT, tools=(TOOL,)):
    requests, jev = [], FakeJevClient(judge)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return completion(content, [{"id": "call-1", "type": "function", "function": {
                "name": "read", "arguments": json.dumps(arguments)}}])
        return completion("done")

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        result = run_agent(TASK, model="m", base_url="https://example.invalid/v1", http=http,
                           tools=list(tools), handlers={"read": lambda: source_file()},
                           store=InMemoryStore(), log=ShadowLog(), mode="on", jev=jev,
                           intent=intent)
    tool_message = requests[1]["messages"][-1]
    return result, jev, requests, tool_message


def test_reply_mode_takes_the_intent_from_the_models_message() -> None:
    result, jev, requests, tool = agent("reply", {})
    assert result.status == "completed"
    assert all(call.state["intent"] == INTENT for call in jev.calls)
    body = "\n".join(line for line in tool["content"].splitlines()
                     if not line.startswith("[[elided"))
    assert "def format_date_0" not in body and "[[elided" in tool["content"]
    assert "intent" not in requests[0]["tools"][0]["function"]["parameters"]["properties"]


def test_arg_mode_offers_an_intent_argument_and_strips_it() -> None:
    stated = "Where does parse_month check the range?"
    _, jev, requests, tool = agent("arg", {"intent": stated}, content=None)
    offered = requests[0]["tools"][0]["function"]["parameters"]
    assert offered["properties"]["intent"] == INTENT_PARAMETER
    assert offered["additionalProperties"] is False and "intent" not in offered.get("required", [])
    assert TOOL["function"]["parameters"]["properties"] == {}, "the caller's schema is untouched"
    assert not tool["content"].startswith("Tool error"), "the handler never sees intent"
    assert all(call.state["intent"] == stated for call in jev.calls)


def test_arg_mode_falls_back_to_the_message_and_spares_a_tool_with_its_own_intent() -> None:
    _, jev, _, _ = agent("arg", {})
    assert all(call.state["intent"] == INTENT for call in jev.calls)

    own = {"type": "function", "function": {"name": "read", "description": "", "parameters": {
        "type": "object", "properties": {"intent": {"type": "string"}}}}}
    _, _, requests, tool = agent("arg", {"intent": "x"}, tools=(own,))
    assert requests[0]["tools"][0] == own
    assert tool["content"].startswith("Tool error"), "its own argument goes to its handler"


def test_off_mode_sends_no_intent() -> None:
    _, jev, _, tool = agent("off", {})
    assert jev.calls and all("intent" not in call.state for call in jev.calls)
    assert "def format_date_0" in tool["content"]
    with pytest.raises(ValueError, match="intent"):
        agent("always", {})


# --------------------------------------------------------------------------- #
# The sidecar
# --------------------------------------------------------------------------- #


def test_the_sidecar_passes_the_intent_through(tmp_path) -> None:
    clients = []

    def factory():
        clients.append(FakeJevClient(judge))
        return clients[-1]

    server = make_server(SidecarState(tmp_path, GateConfig(), client_factory=factory))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/admit"

    def post(body: dict) -> dict:
        with urlopen(Request(url, data=json.dumps(body).encode())) as response:
            return json.load(response)

    try:
        body = {"session": "i", "text": source_file(), "tool": "read", "call_id": "c1",
                "task": TASK, "turn": 1, "intent": INTENT}
        assert post(body)["pointers"]
        assert all(call.state["intent"] == INTENT for call in clients[0].calls)
        with pytest.raises(HTTPError):
            post({**body, "intent": ["not", "a", "string"]})
    finally:
        server.shutdown()
        server.server_close()
