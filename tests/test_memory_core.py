import pytest

from jevctx.context import advance_prefix
from jevctx.pipeline import render_records, retrieve
from jevctx.shadow import ShadowLog
from jevctx.store import InMemoryStore
from jevctx.testing import FakeJevClient
from jevctx.tokens import estimate_tokens
from jevctx.types import JevUnavailableError, Origin, Record


def record(identifier, text, turn=1, **meta):
    return Record(id=identifier, text=text, kind="tool_output", tokens=estimate_tokens(text),
                  created_turn=turn, origin=Origin(source="tool:read", ref=identifier, turn=turn),
                  meta=meta)


def search(store, **kwargs):
    return retrieve("needle", turn=1000, client=FakeJevClient.constant(0.9),
                    store=store, log=ShadowLog(), **kwargs)


def test_old_lexical_match_survives_recent_candidate_limit():
    store = InMemoryStore()
    store.put(record("old", "needle was found in old output", 1))
    for i in range(100):
        store.put(record(str(i), f"unrelated recent output {i}", i + 2))
    assert search(store, k=1)[0].id == "old"


def test_full_payload_budget_dedup_and_skip_oversized():
    store = InMemoryStore()
    store.put(record("a", "needle " * 1000))
    store.put(record("b", "needle small"))
    store.put(record("c", "needle small"))
    store.put(record("d", "needle another"))
    result = search(store, result_budget_tokens=160)
    assert result
    assert "a" not in {r.id for r in result}
    assert len({r.text for r in result}) == len(result)
    assert estimate_tokens(render_records(result)) <= 160


def test_filters_and_full_output_link():
    store = InMemoryStore()
    label = {"type": "configuration", "entities": {"paths": 0.9, "urls": 0.1}}
    full = record("full", "needle config contents", label=label)
    full.lifecycle = "permanent"
    store.put(full)
    store.put(record("fragment", "needle partial", full_output_id="full"))
    store.put(record("unlabelled", "needle no label"))
    assert [r.id for r in search(store, content_type="configuration", entity="paths",
                                 lifecycle="permanent")] == ["full"]
    assert search(store, entity="urls") == []
    assert len(search(store)) == 2


def test_reranking_budgets_summary_not_full_record():
    store, client = InMemoryStore(), FakeJevClient.constant(0.9)
    store.put(record("huge", "needle " * 100000))
    assert retrieve("needle", turn=1, client=client, store=store, log=ShadowLog()) == []
    assert len(client.calls) == 1


def test_shadow_never_logs_injected_and_scoring_failure_propagates():
    store, log = InMemoryStore(), ShadowLog()
    store.put(record("r", "needle"))
    assert retrieve("needle", turn=1, client=FakeJevClient.constant(0.9), store=store,
                    log=log, shadow_only=True)
    assert log.stats().by_action.get("injected", 0) == 0
    with pytest.raises(JevUnavailableError):
        retrieve("needle", turn=1, client=FakeJevClient(raises=JevUnavailableError("offline")),
                 store=store, log=log)


def test_prefix_commit_retry_mutation_and_shortening():
    state = advance_prefix({}, ["system", "user"])
    assert state["epoch"] == 1
    state = advance_prefix(state, ["system", "user", "tool"])
    assert state["epoch"] == 1 and state["common_prefix"] == 2
    state = advance_prefix(state, state["hashes"])
    assert state["epoch"] == 1 and not state["reset"]
    state = advance_prefix(state, ["changed", "user", "tool"])
    assert state["epoch"] == 2 and state["common_prefix"] == 0
    state = advance_prefix(state, ["changed"])
    assert state["epoch"] == 3 and state["common_prefix"] == 1
    assert advance_prefix(state, ["changed"], reason="session_compact")["epoch"] == 4
