from jevctx.pipeline import expand, find_pointers
from jevctx.shadow import ShadowLog
from jevctx.store import InMemoryStore
from jevctx.testing import FakeJevClient
from jevctx.types import JevUnavailableError
from jevctx.workarea import TailItem, WorkArea, WorkAreaConfig, compaction_pays

TASK = "Fix the date parser."
PRICEY = WorkAreaConfig(price_input=3.0, price_cache_read=0.3, min_work_tokens=1000)
CHEAP = WorkAreaConfig(price_input=0.15, price_cache_read=0.003, min_work_tokens=1000)


def client(commit: float = 0.9):
    """Commit question answers ``commit``; outputs containing STALE have served their purpose."""
    def answer(state, questions, key):
        if key == "commit":
            return commit
        text = next(i["text"] for i in state["items"] if i["ref"] == key)
        return 0.1 if "STALE" in text else 0.9
    return FakeJevClient(answer)


def tool(n: int, stale: bool, tokens: int = 3000) -> TailItem:
    text = ("STALE listing\n" if stale else "needed traceback\n") + "line\n" * 50
    return TailItem(id=f"part{n}", kind="tool", tokens=tokens, text=text, tool="bash",
                    call_id=f"call{n}")


def other(n: int, tokens: int = 200) -> TailItem:
    return TailItem(id=f"msg{n}", kind="other", tokens=tokens)


def transcript() -> list[TailItem]:
    return [other(0, 1000), tool(1, stale=True), other(2), tool(3, stale=True),
            other(4), tool(5, stale=False), other(6)]


def test_the_arithmetic_depends_on_the_cache_price_ratio():
    # Drop 6k of a 9.6k tail with 20 turns left.
    pays_pricey, benefit, cost = compaction_pays(6000, 9600, 20, PRICEY)
    assert pays_pricey and benefit > cost
    pays_cheap, _, _ = compaction_pays(6000, 9600, 20, CHEAP)
    assert not pays_cheap          # cache reads at 1/50 of input: the rewrite never pays
    assert not compaction_pays(100, 50_000, 20, PRICEY)[0]   # tiny saving, huge tail


def test_stale_outputs_are_compacted_and_stay_compacted():
    area, store, log = WorkArea(PRICEY), InMemoryStore(), ShadowLog()
    decision = area.decide(transcript(), task=TASK, recent="found it", turn=5,
                           client=client(), store=store, log=log)
    assert decision.action == "compact+commit"
    assert set(decision.replacements) == {"part1", "part3"}
    pointer = find_pointers(decision.replacements["part1"])[0]
    assert "STALE listing" in expand(pointer.id, store=store, log=log, turn=6)
    # Later calls hand back the same replacements, so every request is rewritten alike.
    later = area.decide([*transcript(), other(7), tool(8, stale=True)], task=TASK,
                        recent="next step", turn=6, client=client(), store=store, log=log)
    assert later.replacements["part1"] == decision.replacements["part1"]


def test_nothing_before_the_commit_point_is_ever_reconsidered():
    area, store, log = WorkArea(PRICEY), InMemoryStore(), ShadowLog()
    area.decide(transcript(), task=TASK, recent="found it", turn=5, client=client(),
                store=store, log=log)
    fake = client()
    area.decide([*transcript(), other(7), tool(8, stale=True), tool(9, stale=True), other(10)],
                task=TASK, recent="done", turn=7, client=fake, store=store, log=log)
    scored = {i["text"].split("\n")[0] for call in fake.calls if "items" in call.state
              for i in call.state["items"]}
    assert len([c for c in fake.calls if "items" in c.state for _ in c.state["items"]]) == 2
    assert all("tokens]" in s for s in scored)      # only the two new outputs were carded


def test_cheap_cache_declines_the_rewrite_but_still_commits():
    area, store, log = WorkArea(CHEAP), InMemoryStore(), ShadowLog()
    decision = area.decide(transcript(), task=TASK, recent="found it", turn=5,
                           client=client(), store=store, log=log)
    assert decision.action == "commit" and not decision.replacements
    assert decision.cost_usd > decision.benefit_usd and area.stats["declined_compactions"] == 1


def test_no_commit_point_keeps_the_work_area_open_until_overdue():
    area, store, log = WorkArea(PRICEY), InMemoryStore(), ShadowLog()
    first = area.decide(transcript(), task=TASK, recent="still looking", turn=2,
                        client=client(commit=0.1), store=store, log=log)
    assert first.action == "compact" and area.committed_through is None
    overdue = area.decide([*transcript(), tool(8, stale=False)], task=TASK, recent="x",
                          turn=2 + PRICEY.max_uncommitted_turns, client=client(commit=0.1),
                          store=store, log=log)
    assert overdue.action.endswith("commit") and area.committed_through == "part8"


def test_small_or_unchanged_work_areas_cost_no_jev_call():
    area, fake = WorkArea(PRICEY), client()
    tiny = [other(0), TailItem(id="p", kind="tool", tokens=300, text="x", tool="bash")]
    assert area.decide(tiny, task=TASK, recent="", turn=1, client=fake, store=InMemoryStore(),
                       log=ShadowLog()).action == "skip"
    assert fake.calls == []


def test_a_rewritten_history_resets_the_marker():
    area, store, log = WorkArea(PRICEY), InMemoryStore(), ShadowLog()
    area.decide(transcript(), task=TASK, recent="found it", turn=5, client=client(),
                store=store, log=log)
    summarised = [other(99, 2000), tool(100, stale=True)]
    assert area.decide(summarised, task=TASK, recent="", turn=9, client=client(), store=store,
                       log=log).action == "reset"
    assert area.committed_through is None and area.stats["resets"] == 1


def test_a_jev_outage_neither_commits_nor_compacts():
    area = WorkArea(PRICEY)
    decision = area.decide(transcript(), task=TASK, recent="", turn=5,
                           client=FakeJevClient(raises=JevUnavailableError("down")),
                           store=InMemoryStore(), log=ShadowLog())
    assert decision.action == "skip" and not decision.replacements
    assert area.committed_through is None


def test_sidecar_workarea_endpoint(tmp_path):
    import json
    import threading
    from urllib.request import Request, urlopen

    from jevctx.pipeline import GateConfig
    from jevctx.serve import SidecarState, make_server

    state = SidecarState(tmp_path, GateConfig(), client_factory=client, workarea=PRICEY)
    server = make_server(state)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/workarea"
    items = [{"id": i.id, "kind": i.kind, "tokens": i.tokens, "text": i.text, "tool": i.tool,
              "call_id": i.call_id} for i in transcript()]
    try:
        body = json.dumps({"session": "w", "turn": 5, "recent": "found it", "task": TASK,
                           "items": items}).encode()
        with urlopen(Request(url, data=body)) as response:
            result = json.load(response)
        assert set(result["replacements"]) == {"part1", "part3"}
        assert result["decision"]["action"] == "compact+commit"
        assert state.stats("w")["workarea"]["compactions"] == 1
    finally:
        server.shutdown()
        server.server_close()


def test_a_long_session_is_expected_to_run_on_but_not_as_long_again():
    from jevctx.workarea import remaining_turns
    config = WorkAreaConfig(expected_turns=30, min_remaining_turns=12)
    assert remaining_turns(5, config) == 25      # early: the prior
    assert remaining_turns(20, config) == 12     # past it: the floor
    assert remaining_turns(45, config) == 12


def test_code_to_change_waits_for_an_edit():
    from jevctx.types import Origin, Record

    area, store, log = WorkArea(PRICEY), InMemoryStore(), ShadowLog()
    # The admission gate profiled call1's output as the code the agent is going to change.
    store.put(Record(id="o:1", text="x", kind="tool_output", tokens=3000, created_turn=1,
                     origin=Origin(source="tool:read", ref="call1", turn=1),
                     meta={"profile": {"role": "change_site", "type": "source_code"}}))
    decision = area.decide(transcript(), task=TASK, recent="found it", turn=5,
                           client=client(), store=store, log=log)
    assert set(decision.replacements) == {"part3"}
    # Once an edit follows it, it is fair game like any other stale output.
    area = WorkArea(PRICEY)
    edited = [*transcript(), TailItem(id="e", kind="tool", tokens=50, text="ok", tool="edit")]
    decision = area.decide(edited, task=TASK, recent="edited", turn=6, client=client(),
                           store=store, log=log)
    assert {"part1", "part3"} <= set(decision.replacements)


def test_outdated_outputs_are_compacted_without_asking_jev():
    area, store, log, fake = WorkArea(PRICEY), InMemoryStore(), ShadowLog(), client()
    # part5 is "needed" by Jev's lights, but a later read replaced it.
    outdated = {"call5": "superseded at turn 4: /w/a.py was viewed again over the same lines"}
    decision = area.decide(transcript(), task=TASK, recent="found it", turn=5, client=fake,
                           store=store, log=log, outdated=lambda call_id: outdated.get(call_id, ""))
    assert set(decision.replacements) == {"part1", "part3", "part5"}
    assert "out of date (superseded" in decision.replacements["part5"]
    carded = [i["text"] for c in fake.calls if "items" in c.state for i in c.state["items"]]
    assert len(carded) == 2                  # part5 needed no question
    assert area.stats["outdated_compacted"] == 1


def test_an_output_outdated_after_it_was_frozen_can_still_go():
    area, store, log = WorkArea(PRICEY), InMemoryStore(), ShadowLog()
    # part5 is still needed, and the agent is at a commit point: it is frozen as it is.
    frozen = area.decide(transcript(), task=TASK, recent="found it", turn=5,
                         client=client(), store=store, log=log)
    assert "part5" not in frozen.replacements and area.committed_through == "msg6"
    # Much later the file part5 showed is edited: part5 is outdated, though frozen.
    later = [*transcript(), other(7), *[tool(n, stale=False, tokens=200) for n in range(8, 10)]]
    fake = client()
    decision = area.decide(later, task=TASK, recent="edited", turn=20, client=fake, store=store,
                           log=log, outdated=lambda c: "the file was changed" if c == "call5" else "")
    assert "part5" in decision.replacements
    assert fake.calls == []                  # no new output worth a Jev question: none asked
