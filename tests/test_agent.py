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
