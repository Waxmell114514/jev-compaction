import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from jevctx.pipeline import GateConfig, admit, expand
from jevctx.recall import excerpt, recall, render_hits
from jevctx.serve import SidecarState, make_server
from jevctx.shadow import ShadowLog
from jevctx.store import InMemoryStore
from jevctx.testing import FakeJevClient
from jevctx.types import Origin
from tests.test_profile import TASK, by_dimension

TRACEBACK = "".join(
    f"collecting tests/unit/test_mod_{i}.py ... ok\n" for i in range(60)
) + (
    "Traceback (most recent call last):\n"
    '  File "src/dates/parser.py", line 88, in parse_month\n'
    "ValueError: month must be in 1..12\n"
    "FAIL tests/test_dates.py::test_parse_leap\n"
)
SOURCE = "".join(f"def helper_{i}(value):\n    return value + {i}\n\n" for i in range(80)) + (
    "def parse_month(value):\n    return int(value)  # no range check\n"
)
LISTING = "".join(f"src/pkg/module_{i}.py\n" for i in range(200))


def recall_client():
    """Profiles like tests.test_profile; recall scores by what the card says."""
    def answer(state, questions, key):
        if ":" in key:
            return by_dimension(state, questions, key)
        text = next(i["text"] for i in state["items"] if i["ref"] == key)
        query = state["task"].lower()
        if "traceback" in query or "failing" in query:
            return 0.9 if "Traceback" in text else 0.05
        if "month" in query:
            return 0.9 if "def parse_month" in text else 0.1
        return 0.02
    return FakeJevClient(answer)


@pytest.fixture
def memory():
    store, log, client = InMemoryStore(), ShadowLog(), recall_client()
    config = GateConfig(profile=True, max_elide_fraction=1.0)
    for turn, (tool, text) in enumerate([("bash", TRACEBACK), ("read", SOURCE), ("bash", LISTING)], 1):
        admit(text, Origin(source=f"tool:{tool}", ref=f"c{turn}", turn=turn), task_digest=TASK,
              turn=turn, client=client, store=store, log=log, config=config)
    return store, log, client


def test_a_loose_description_finds_the_right_output(memory):
    store, log, client = memory
    hits = recall("that traceback we hit earlier", store=store, client=client, log=log, turn=9)
    assert hits and "ValueError: month must be in 1..12" in hits[0].text
    assert hits[0].record.kind == "tool_output"
    assert all(h.record.kind != "elided_segment" for h in hits)  # whole outputs, not fragments
    assert any(e["kind"] == "retrieve" and e["action"] == "injected" for e in log.entries())


def test_name_filter_pins_the_search(memory):
    store, log, client = memory
    hits = recall("where month is parsed", store=store, client=client, log=log, turn=9,
                  name="parse_month")
    assert hits and hits[0].record.origin.source in ("tool:read", "tool:bash")
    assert all("parse_month" in h.record.text for h in hits)
    relaxed = recall("that traceback", store=store, client=client, log=log, turn=9,
                     name="no_such_name")
    assert relaxed and relaxed[0].relaxed and "Traceback" in relaxed[0].text
    assert render_hits(relaxed).startswith("(No stored output mentions that name")


def test_role_filter_matches_what_a_record_contains_not_what_it_mostly_is(memory):
    store, log, client = memory
    [log_output] = [r for r in store.all_records()
                    if r.kind == "tool_output" and "Traceback" in r.text]
    assert log_output.meta["profile"]["role"] == "noise"           # mostly collection noise
    assert log_output.meta["profile"]["role_probs"]["evidence"] < 0.1
    evidence = recall("that traceback", store=store, client=client, log=log, turn=9,
                      role="evidence", min_prob=0.5)
    assert evidence and evidence[0].record.id == log_output.id
    assert recall("that traceback", store=store, client=client, log=log, turn=9,
                  role="change_site", min_prob=0.5) == []


def test_large_records_come_back_as_excerpts_around_the_query():
    text = "".join(f"line {i}\n" for i in range(2000)) + "the needle is here\n" + "tail\n" * 50
    out, truncated = excerpt(text, "where is the needle", budget_tokens=200)
    assert truncated and "the needle is here" in out and "[line" in out
    assert excerpt("short", "q", 200) == ("short", False)


def test_rendered_hits_say_how_to_get_the_rest(memory):
    store, log, client = memory
    hits = recall("that traceback", store=store, client=client, log=log, turn=9, budget_tokens=200)
    rendered = render_hits(hits)
    assert rendered.startswith("[[record id=o:")
    assert "expand id=" in rendered if hits[0].truncated else True
    assert render_hits([]).startswith("No stored output matched")


def test_expanding_a_recalled_output_is_not_a_gate_false_negative(memory):
    store, log, client = memory
    [hit, *_] = recall("that traceback", store=store, client=client, log=log, turn=9)
    before = len([e for e in log.entries() if e["kind"] == "expand"])
    expand(hit.record.id, store=store, log=log, turn=10)
    expands = [e for e in log.entries() if e["kind"] == "expand"]
    assert len(expands) == before + 1 and expands[-1]["item_id"] == hit.record.id


def test_sidecar_recall_endpoint(tmp_path):
    state = SidecarState(tmp_path, GateConfig(profile=True, max_elide_fraction=1.0),
                         client_factory=recall_client)
    server = make_server(state)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path, body):
        with urlopen(Request(url + path, data=json.dumps(body).encode())) as response:
            return json.load(response)

    try:
        post("/admit", {"session": "r", "text": TRACEBACK, "task": TASK, "tool": "bash"})
        found = post("/recall", {"session": "r", "query": "the failing traceback", "turn": 3})
        assert found["hits"] and "ValueError" in found["text"]
        assert state.stats("r")["recalls"] == 1
        for bad in ({"query": ""}, {"query": "x", "role": "nonsense"}, {"query": "x", "k": "3"}):
            with pytest.raises(HTTPError) as error:
                post("/recall", {"session": "r", **bad})
            assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_an_out_of_date_hit_says_so_to_the_reranker_and_the_agent(memory):
    store, log, client = memory
    reason = "src/dates/parser.py was changed after this output (turn 5)"
    hits = recall("where parse_month is defined", store=store, client=client, log=log, turn=9,
                  outdated=lambda r: reason if r.origin.ref == "c2" else "")
    assert hits[0].outdated == reason
    assert f"out of date: {reason}" in render_hits(hits)
    cards = [i["text"] for c in client.calls if "items" in c.state for i in c.state["items"]
             if "Looking for" in c.state.get("task", "")]
    assert any(f"OUT OF DATE: {reason}" in text for text in cards)
