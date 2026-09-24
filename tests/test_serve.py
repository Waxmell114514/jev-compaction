import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from jevctx.pipeline import GateConfig, find_pointers, reconstruct
from jevctx.serve import SidecarState, make_server
from jevctx.testing import FakeJevClient
from jevctx.types import JevUnavailableError
from tests.test_pipeline import keep_noise_client, noisy_log
from tests.test_profile import by_dimension
from tests.test_profile import output as profiled_output

NPM_LOG = noisy_log()
TASK = "Install dependencies and get the test suite passing."


@pytest.fixture
def sidecar(tmp_path):
    state = SidecarState(tmp_path, GateConfig(), client_factory=keep_noise_client)
    server = make_server(state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path: str, body: dict) -> dict:
        request = Request(url + path, data=json.dumps(body).encode(),
                          headers={"content-type": "application/json"})
        with urlopen(request) as response:
            return json.load(response)

    def get(path: str) -> dict:
        with urlopen(url + path) as response:
            return json.load(response)

    yield post, get, state
    server.shutdown()
    server.server_close()


def test_admit_then_expand_restores_the_original(sidecar, tmp_path):
    post, get, state = sidecar
    result = post("/admit", {"session": "s1", "text": NPM_LOG, "tool": "bash",
                             "call_id": "c1", "task": TASK, "turn": 1, "mode": "on"})
    assert result["pointers"] and result["result_tokens"] < result["original_tokens"]
    assert "IMPORTANT" in result["text"]
    for pointer in find_pointers(result["text"]):
        assert post("/expand", {"session": "s1", "id": pointer.id, "turn": 2})["text"]
    assert reconstruct(result["text"], state.session("s1").store) == NPM_LOG
    stats = get("/stats?session=s1")
    assert stats["admits"] == 1 and stats["expands"] == len(result["pointers"])
    assert stats["saved_tokens"] == result["original_tokens"] - result["result_tokens"]
    assert (tmp_path / "s1" / "memory.jsonl").exists()


def test_shadow_mode_scores_but_returns_the_original(sidecar):
    post, _, state = sidecar
    result = post("/admit", {"session": "s2", "text": NPM_LOG, "task": TASK, "mode": "shadow"})
    assert result["text"] == NPM_LOG and not result["pointers"]
    assert state.session("s2").log.entries()


def test_sessions_are_isolated(sidecar):
    post, _, _ = sidecar
    result = post("/admit", {"session": "a", "text": NPM_LOG, "task": TASK})
    with pytest.raises(HTTPError) as error:
        post("/expand", {"session": "b", "id": result["pointers"][0]})
    assert error.value.code == 404


@pytest.mark.parametrize("body", [
    {"session": "../x", "text": "t", "task": TASK},
    {"session": "s", "text": 1, "task": TASK},
    {"session": "s", "text": "t", "task": " "},
    {"session": "s", "text": "t", "task": TASK, "mode": "off"},
])
def test_rejects_malformed_requests(sidecar, body):
    post, _, _ = sidecar
    with pytest.raises(HTTPError) as error:
        post("/admit", body)
    assert error.value.code == 400


def test_jev_failure_is_a_502_naming_only_the_error_class(tmp_path):
    state = SidecarState(tmp_path, GateConfig(),
                         client_factory=lambda: FakeJevClient(raises=JevUnavailableError("secret")))
    server = make_server(state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    request = Request(f"http://127.0.0.1:{server.server_address[1]}/admit",
                      data=json.dumps({"session": "s", "text": NPM_LOG, "task": TASK}).encode())
    try:
        with urlopen(request) as response:
            # score_items fails open: an unreachable scorer keeps everything.
            assert json.load(response)["text"] == NPM_LOG
    except HTTPError as error:
        assert error.code == 502 and b"secret" not in error.read()
    finally:
        server.shutdown()
        server.server_close()


def test_max_elide_fraction_override_disables_the_tripwire(sidecar):
    post, _, _ = sidecar
    everything_low = "\n\n".join(f"npm http fetch GET 200 https://registry/pkg-{i} 41ms" * 3
                                 for i in range(80)) + "\n"
    capped = post("/admit", {"session": "c", "text": everything_low, "task": TASK})
    assert capped["tripwire"] == "max_elide_fraction" and not capped["pointers"]
    uncapped = post("/admit", {"session": "u", "text": everything_low, "task": TASK,
                               "max_elide_fraction": 1})
    assert uncapped["tripwire"] is None and uncapped["pointers"]
    with pytest.raises(HTTPError):
        post("/admit", {"session": "u", "text": "t", "task": TASK, "max_elide_fraction": 2})


def test_profile_can_be_requested_per_admit(tmp_path):
    state = SidecarState(tmp_path, GateConfig(), client_factory=lambda: FakeJevClient(by_dimension))
    server = make_server(state)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    def post(body: dict) -> dict:
        with urlopen(Request(url + "/admit", data=json.dumps(body).encode())) as response:
            return json.load(response)

    try:
        text = profiled_output(injected=True)
        result = post({"session": "p", "text": text, "task": TASK, "profile": True})
        assert result["pointers"] and "IGNORE ALL PREVIOUS" not in result["text"]
        records = state.session("p").store.all_records()
        assert any(r.kind == "tool_output" and "profile" in r.meta for r in records)
        assert state.stats("p")["quarantined"] >= 1
        with pytest.raises(HTTPError):
            post({"session": "p", "text": text, "task": TASK, "profile": "yes"})
    finally:
        server.shutdown()
        server.server_close()


def test_an_edit_makes_an_earlier_read_out_of_date(sidecar, tmp_path):
    post, get, state = sidecar
    read = {"session": "r", "text": NPM_LOG, "tool": "read", "call_id": "c1", "task": TASK,
            "turn": 1, "mode": "on", "args": {"filePath": "/w/app.py"}, "cwd": "/w"}
    pointer = find_pointers(post("/admit", read)["text"])[0]
    assert not post("/expand", {"session": "r", "id": pointer.id, "turn": 2})["text"].startswith("[note")
    made = post("/observe", {"session": "r", "tool": "edit", "call_id": "c2", "turn": 3,
                             "args": {"filePath": "app.py", "oldString": "a", "newString": "b"}})
    assert [(r["older"], r["kind"]) for r in made["relations"]] == [("c1", "stale")]
    expanded = post("/expand", {"session": "r", "id": pointer.id, "turn": 4})["text"]
    assert expanded.startswith("[note: possibly out of date: /w/app.py was changed")
    assert get("/stats?session=r")["relations"] == {"stale": 1}
    # Relations persist with the session's store.
    reloaded = SidecarState(tmp_path, GateConfig(), client_factory=keep_noise_client).session("r")
    assert reloaded.relations.status("c1").kind == "stale"
