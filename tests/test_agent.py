"""Exercise wire messages, recovery and accounting without live API calls."""

import json

import httpx
import pytest

from jevctx.agent import run_agent
from jevctx.pipeline import GateConfig, find_pointers
from jevctx.shadow import ShadowLog
from jevctx.store import InMemoryStore
from jevctx.testing import FakeJevClient
from jevctx.types import JevUnavailableError
from jevctx.usage import Prices

TOOL = {"type": "function", "function": {
    "name": "read", "description": "Read fixture", "parameters": {"type": "object"},
}}
RAW = "noise line\n" * 80 + "\n" + "IMPORTANT result\n" * 80


def call(identifier="call-1", name="read", arguments="{}"):
    return {"id": identifier, "type": "function",
            "function": {"name": name, "arguments": arguments}}


def response(calls=None, reason=None, usage=True):
    message = {"role": "assistant", "content": None if calls else "answer"}
    if calls:
        message["tool_calls"] = calls
    payload = {"choices": [{"message": message,
                           "finish_reason": reason or ("tool_calls" if calls else "stop")}]}
    if usage:
        payload["usage"] = {"prompt_tokens": 100, "completion_tokens": 10,
                            "prompt_tokens_details": {"cached_tokens": 60}}
    return httpx.Response(200, json=payload)


def run(handler, **kwargs):
    store = kwargs.pop("store", InMemoryStore())
    log = kwargs.pop("log", ShadowLog())
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        return run_agent(
            "Find the result", model="configured-model", base_url="https://example.invalid/v1/",
            http=http, tools=[TOOL], handlers=kwargs.pop("handlers", {"read": lambda: RAW}),
            store=store, log=log, **kwargs,
        )


@pytest.mark.parametrize("mode", ["off", "shadow", "on"])
def test_modes_preserve_protocol_and_expand_original_without_regating(mode):
    requests = []
    jev = FakeJevClient.by_text(lambda text: 0.9 if "IMPORTANT" in text else 0.01)
    store = InMemoryStore()

    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "configured-model"
        requests.append(body)
        if len(requests) == 1:
            return response([call()])
        tool = body["messages"][-1]
        if len(requests) == 2:
            assert tool["tool_call_id"] == "call-1"
            assert body["messages"][-2]["tool_calls"] == [call()]
            if mode == "on":
                pointers = find_pointers(tool["content"])
                assert pointers and "IMPORTANT" in tool["content"]
                return response([call("call-2", "expand", json.dumps({"id": pointers[0].id}))])
            assert tool["content"] == RAW
        else:
            assert tool["content"] == store.all_records()[0].text
            assert tool["tool_call_id"] == "call-2"
            assert body["messages"][:len(requests[1]["messages"])] == requests[1]["messages"]
        return response()

    result = run(handler, mode=mode, jev=jev, store=store,
                 prices=Prices(3, 15, 0.3, 0))
    assert result.status == "completed"
    assert result.metrics["expands"] == (1 if mode == "on" else 0)
    assert bool(jev.calls) == (mode != "off")
    assert result.metrics["host_usage"]["input_tokens"] == 40 * len(requests)
    assert result.metrics["host_usage"]["cache_read_tokens"] == 60 * len(requests)
    assert result.metrics["host_cost_usd"] == pytest.approx(0.000288 * len(requests))


def test_jev_outage_preserves_tool_text():
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        if count == 1:
            return response([call()])
        assert json.loads(request.content)["messages"][-1]["content"] == RAW
        return response()

    result = run(handler, mode="on", jev=FakeJevClient.failing(JevUnavailableError("down")))
    assert result.status == "completed"


@pytest.mark.parametrize("arguments", ["{", "[]", '{"unexpected":1}'])
def test_bad_arguments_are_tool_errors_and_do_not_execute_handler(arguments):
    executed = []
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        if count == 1:
            return response([call(arguments=arguments)])
        assert "Tool error" in json.loads(request.content)["messages"][-1]["content"]
        return response()

    result = run(handler, mode="off", handlers={"read": lambda: executed.append(1)})
    assert result.status == "completed" and not executed


def test_multiple_calls_get_matching_results_in_one_round():
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        if count == 1:
            return response([call("a"), call("b")])
        messages = json.loads(request.content)["messages"]
        assert [m["tool_call_id"] for m in messages[-2:]] == ["a", "b"]
        return response()

    assert run(handler, mode="off").status == "completed"


@pytest.mark.parametrize("reason,steps,status", [("length", 3, "incomplete"),
                                                 ("tool_calls", 1, "step_limit")])
def test_incomplete_or_last_turn_never_executes_tools(reason, steps, status):
    executed = []
    result = run(lambda request: response([call()], reason=reason), mode="off",
                 max_steps=steps, handlers={"read": lambda: executed.append(1)})
    assert result.status == status and not executed


def test_duplicate_ids_fail_before_any_tool_execution():
    executed = []
    result = run(lambda request: response([call(), call()]), mode="off",
                 handlers={"read": lambda: executed.append(1)})
    assert result.status == "error" and not executed


def test_http_failure_keeps_partial_usage_without_leaking_response():
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return response([call()]) if count == 1 else httpx.Response(401, text="secret")

    result = run(handler, mode="off")
    assert result.status == "error"
    assert result.metrics["host_usage"]["requests"] == 1
    assert result.metrics["error"] == "HTTPStatusError (HTTP 401)"
    assert "secret" not in str(result)


def test_missing_usage_is_not_reported_as_zero_cost():
    result = run(lambda request: response(usage=False), mode="off", prices=Prices(1, 1, 1, 1))
    assert result.status == "completed"
    assert result.metrics["total_cost_usd"] is None
    assert result.metrics["host_usage"]["missing_usage"] == 1


def test_mode_overrides_caller_shadow_flag():
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return response([call()]) if count == 1 else response()

    result = run(handler, mode="shadow", jev=FakeJevClient.constant(0.2),
                 gate_config=GateConfig(shadow_only=False, max_elide_fraction=1))
    assert result.messages[-2]["content"] == RAW


@pytest.mark.parametrize("tool", [call(name="unknown"),
                                  call(name="expand", arguments='{"id":"missing"}')])
def test_unknown_tools_and_missing_pointers_return_recoverable_errors(tool):
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        if count == 1:
            return response([tool])
        assert "Tool error" in json.loads(request.content)["messages"][-1]["content"]
        return response()

    assert run(handler, mode="off").status == "completed"


@pytest.mark.parametrize("payload", [[], {"choices": []},
    {"choices": [{"message": None}]},
    {"choices": [{"message": {"role": "assistant", "content": {}}}]}])
def test_malformed_provider_payload_is_an_error_report(payload):
    result = run(lambda request: httpx.Response(200, json=payload), mode="off")
    assert result.status == "error"


def by_card(answer_stale=0.1, answer_needed=0.9, commit=0.9):
    """Items containing STALE have served their purpose; the agent is at a commit point."""
    def answer(state, questions, key):
        if key == "commit":
            return commit
        items = {i["ref"]: i["text"] for i in state.get("items", [])}
        text = items.get(key.split(":")[0], "")
        return answer_stale if "STALE" in text else answer_needed
    return FakeJevClient(answer)


def test_recall_is_offered_only_with_the_gate_on():
    for mode, offered in (("shadow", False), ("on", True)):
        seen: list[list[str]] = []

        def handler(request, seen=seen):
            seen.append([t["function"]["name"] for t in json.loads(request.content)["tools"]])
            return response()

        run(handler, mode=mode, jev=by_card())
        assert ("recall" in seen[0]) == offered


def test_recall_finds_earlier_output_from_a_description():
    from tests.test_recall import TRACEBACK, recall_client

    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return response([call()])
        if len(requests) == 2:
            return response([call("call-2", "recall", json.dumps({"query": "that traceback we hit"}))])
        assert "ValueError: month must be in 1..12" in body["messages"][-1]["content"]
        return response()

    result = run(handler, mode="on", jev=recall_client(), handlers={"read": lambda: TRACEBACK},
                 gate_config=GateConfig(profile=True, max_elide_fraction=1.0))
    assert result.status == "completed" and result.metrics["recalls"] == 1


def test_an_edit_marks_an_earlier_read_out_of_date_on_expand():
    tools = [{"type": "function", "function": {"name": n, "description": n,
                                               "parameters": {"type": "object"}}}
             for n in ("read", "edit")]
    handlers = {"read": lambda path: RAW, "edit": lambda path: "edited"}
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return response([call("r1", "read", json.dumps({"path": "/w/a.py"}))])
        if len(requests) == 2:
            pointer = find_pointers(body["messages"][-1]["content"])[0]
            requests.append(pointer.id)
            return response([call("e1", "edit", json.dumps({"path": "a.py"}))])
        if len(requests) == 4:
            return response([call("x1", "expand", json.dumps({"id": requests[2]}))])
        assert body["messages"][-1]["content"].startswith("[note: possibly out of date: /w/a.py")
        return response()

    jev = FakeJevClient.by_text(lambda text: 0.9 if "IMPORTANT" in text else 0.01)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        result = run_agent("Fix it", model="m", base_url="https://example.invalid/v1", http=http,
                           tools=tools, handlers=handlers, store=InMemoryStore(), log=ShadowLog(),
                           mode="on", jev=jev, cwd="/w")
    assert result.status == "completed" and result.metrics["relations"] == {"superseded": 0, "stale": 1}


def test_the_work_area_rewrites_requests_not_the_stored_conversation():
    from jevctx.workarea import WorkAreaConfig

    outputs = {"a": "STALE directory listing\n" + "entry\n" * 1200,
               "b": "needed traceback\n" + "frame\n" * 1200}
    tools = [{"type": "function", "function": {"name": "run", "description": "run",
                                               "parameters": {"type": "object"}}}]
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) <= 2:
            which = "ab"[len(requests) - 1]
            return response([call(f"c-{which}", "run", json.dumps({"which": which}))])
        return response()

    config = WorkAreaConfig(price_input=3.0, price_cache_read=0.3, min_work_tokens=1000)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        result = run_agent("Fix it", model="m", base_url="https://example.invalid/v1", http=http,
                           tools=tools, handlers={"run": lambda which: outputs[which]},
                           store=InMemoryStore(), log=ShadowLog(), mode="on", jev=by_card(),
                           gate_config=GateConfig(min_gate_tokens=100_000), workarea=config)
    sent = {m["tool_call_id"]: m["content"] for m in requests[2]["messages"] if m["role"] == "tool"}
    assert sent["c-a"].startswith("[[elided") and "compacted run output" in sent["c-a"]
    assert sent["c-b"] == outputs["b"]
    stored = {m["tool_call_id"]: m["content"] for m in result.messages if m["role"] == "tool"}
    assert stored["c-a"] == outputs["a"]              # the conversation itself is untouched
    assert result.metrics["workarea"]["compactions"] == 1
