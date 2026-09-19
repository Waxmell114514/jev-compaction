"""Phase 0 and Phase 1 driven together, the way an agent loop would.

The individual module tests prove each part in isolation. This one proves the
claim the design actually makes: that the gate can chew on the work area turn
after turn while the frozen prefix stays byte-identical, so the KV cache over it
is never invalidated.
"""

from __future__ import annotations

from jevctx import (
    DEFAULT_COMMIT_POLICY,
    CacheLedger,
    ContextBuffer,
    FakeJevClient,
    InMemoryStore,
    JsonlStore,
    Origin,
    ShadowLog,
    admit,
    expand,
    find_pointers,
    make_block,
    reconstruct,
    retrieve,
)
from jevctx.types import Record, TurnSignals

TASK = "Get the failing integration test green."


def tool_output(turn: int) -> str:
    noise = "\n".join(
        f"npm http fetch GET 200 https://registry.npmjs.org/dep-{turn}-{i} {i}ms"
        for i in range(30)
    )
    signal = "\n".join(
        f"IMPORTANT resolved dep-{turn}-{i}@1.{i}.0 required by app" for i in range(30)
    )
    return f"{noise}\n\n{signal}\n"


def gate_client() -> FakeJevClient:
    return FakeJevClient.by_text(lambda t: 0.95 if "IMPORTANT" in t else 0.02)


def test_an_agent_loop_keeps_its_frozen_prefix_byte_identical() -> None:
    store, log = InMemoryStore(), ShadowLog(path=None)
    client = gate_client()
    buffer = ContextBuffer()
    ledger = CacheLedger()

    buffer.append_work(make_block("system", "You are a build assistant.\n"))
    buffer.commit(reason="system prompt is stable")
    frozen_after_system = buffer.render()[: buffer.cache_breakpoint]

    raw_by_turn: dict[int, str] = {}
    for turn in range(1, 7):
        raw = tool_output(turn)
        raw_by_turn[turn] = raw
        result = admit(raw, Origin(source="tool:bash", ref=f"npm-{turn}", turn=turn),
                       task_digest=TASK, turn=turn, client=client, store=store, log=log)
        buffer.append_work(make_block("tool", result.text, turn=turn))

        # The prefix frozen at the very start must still be there, unchanged.
        assert buffer.render()[: len(frozen_after_system)] == frozen_after_system

        signals = TurnSignals(turn=turn, tool_depth=0, last_role="tool")
        decision = DEFAULT_COMMIT_POLICY.should_commit(buffer, signals)
        if decision.commit:
            before = buffer.render()[: buffer.cache_breakpoint]
            buffer.commit(decision.upto, reason=decision.reason)
            assert buffer.render()[: len(before)] == before

        stats = buffer.stats()
        ledger.record_render(turn=turn, frozen_tokens=stats.frozen_tokens,
                             work_tokens=stats.work_tokens, cache_written=decision.commit)

    assert buffer.stats().frozen_blocks > 1, "the policy should have frozen something"
    assert store, "the gate should have relocated the noise"
    assert ledger.estimated_cost().cache_read_tokens > 0

    # Nothing the gate removed was lost.
    for block in list(buffer.frozen) + list(buffer.work):
        for pointer in find_pointers(block.text):
            assert store.get(pointer.id) is not None


def test_relocated_content_survives_a_store_restart(tmp_path) -> None:
    path = tmp_path / "memory.jsonl"
    log = ShadowLog(path=tmp_path / "shadow.jsonl")
    raw = tool_output(1)

    store = JsonlStore(path)
    result = admit(raw, Origin(source="tool:bash", ref="npm", turn=1), task_digest=TASK,
                   turn=1, client=gate_client(), store=store, log=log)
    assert result.pointers

    reopened = JsonlStore(path)
    assert reconstruct(result.text, reopened) == raw


def test_expand_and_retrieve_close_the_loop() -> None:
    store, log = InMemoryStore(), ShadowLog(path=None)
    raw = tool_output(1)
    result = admit(raw, Origin(source="tool:bash", ref="npm", turn=1), task_digest=TASK,
                   turn=1, client=gate_client(), store=store, log=log)

    # The agent goes back for something the gate removed: a recorded false negative.
    recovered = expand(result.pointers[0].id, store=store, log=log, turn=2)
    assert "npm http fetch" in recovered
    assert log.false_negative_rate() > 0.0

    # A lower threshold would not have made that mistake, and replay can show it.
    assert log.replay(0.0).false_negatives == 0
    assert log.replay(0.5).false_negatives > 0

    # Retrieval finds the relocated record again from the store side.
    store.put(Record(id="r:pref", text="The user prefers pnpm over npm.", kind="preference",
                     origin=Origin(source="observation", ref=None, turn=1), tokens=8,
                     created_turn=2, summary="The user prefers pnpm over npm."))
    picky = FakeJevClient.by_text(lambda t: 0.9 if "prefers pnpm" in t else 0.1)
    assert [r.id for r in retrieve(TASK, turn=3, client=picky, store=store, log=log,
                                   k=2)] == ["r:pref"]


def test_a_jev_outage_degrades_to_plain_passthrough() -> None:
    """The whole pipeline must keep working, just without the gate."""
    from jevctx.types import JevUnavailableError

    store, log = InMemoryStore(), ShadowLog(path=None)
    down = FakeJevClient.failing(JevUnavailableError("upstream down"))
    buffer = ContextBuffer()

    raw = tool_output(1)
    result = admit(raw, Origin(source="tool:bash", ref="npm", turn=1), task_digest=TASK,
                   turn=1, client=down, store=store, log=log)
    buffer.append_work(make_block("tool", result.text, turn=1))
    buffer.commit(reason="still works with no Jev at all")

    assert result.text == raw
    assert len(store) == 0
    assert buffer.render()[-1].content == raw
