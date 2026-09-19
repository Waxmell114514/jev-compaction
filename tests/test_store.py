"""Tests for external memory.

The same behavioural body runs against both implementations, because the point of
having two is that they are interchangeable: pipeline code must not care which one
it was handed.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import pytest

from jevctx.store import InMemoryStore, JsonlStore, summarise
from jevctx.tokens import estimate_tokens
from jevctx.types import Lifecycle, MemoryStore, Origin, Record

ORIGIN = Origin(source="tool:bash", ref="npm install", turn=1)


def make_record(record_id: str, text: str, *, kind: str = "fact",
                lifecycle: Lifecycle = "session", turn: int = 0,
                task_id: str | None = None) -> Record:
    return Record(id=record_id, text=text, kind=kind, origin=ORIGIN,
                  tokens=estimate_tokens(text), created_turn=turn,
                  lifecycle=lifecycle, task_id=task_id)


@pytest.fixture(params=["memory", "jsonl"])
def store(request: pytest.FixtureRequest, tmp_path) -> MemoryStore:
    if request.param == "memory":
        return InMemoryStore()
    return JsonlStore(tmp_path / "store.jsonl")


@pytest.fixture(params=["memory", "jsonl"])
def store_factory(request: pytest.FixtureRequest, tmp_path) -> Callable[[], MemoryStore]:
    """Builds a store over the *same* backing, so durability can be tested uniformly."""
    if request.param == "memory":
        shared = InMemoryStore()
        return lambda: shared
    path = tmp_path / "store.jsonl"
    return lambda: JsonlStore(path)


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #


def test_both_implementations_satisfy_the_protocol(store: MemoryStore) -> None:
    assert isinstance(store, MemoryStore)


def test_put_get_round_trips_every_field(store: MemoryStore) -> None:
    record = make_record("r:1", "the database password is hunter2", turn=4,
                         lifecycle="permanent", task_id="t-9")
    record.meta["segment_ids"] = ["s:a", "s:b"]
    store.put(record)

    loaded = store.get("r:1")
    assert loaded is not None
    assert loaded.text == record.text
    assert loaded.lifecycle == "permanent"
    assert loaded.task_id == "t-9"
    assert loaded.meta["segment_ids"] == ["s:a", "s:b"]
    assert loaded.origin == ORIGIN


def test_get_of_an_unknown_id_is_none_not_an_error(store: MemoryStore) -> None:
    assert store.get("r:nope") is None


def test_touch_accumulates_counters(store: MemoryStore) -> None:
    store.put(make_record("r:1", "content"))
    store.touch("r:1", expand=True)
    store.touch("r:1", expand=True)
    store.touch("r:1", hit=True)

    record = store.get("r:1")
    assert record.expand_count == 2
    assert record.hit_count == 1


def test_touch_of_an_unknown_id_is_a_no_op(store: MemoryStore) -> None:
    store.touch("r:nope", expand=True)


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #


def test_summary_is_generated_when_absent(store: MemoryStore) -> None:
    store.put(make_record("r:1", "\n\n   first real line   here  \nsecond line\n"))
    assert store.get("r:1").summary == "first real line here"


def test_summary_is_not_overwritten_when_supplied(store: MemoryStore) -> None:
    record = make_record("r:1", "some text")
    record.summary = "a summary someone else chose"
    store.put(record)
    assert store.get("r:1").summary == "a summary someone else chose"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("short line", "short line"),
        ("", ""),
        ("\n\n\n", ""),
        ("  spaced   out   words  ", "spaced out words"),
    ],
)
def test_summarise_cases(text: str, expected: str) -> None:
    assert summarise(text) == expected


def test_summarise_truncates_on_a_word_boundary() -> None:
    text = " ".join(["word"] * 200)
    result = summarise(text, 40)
    assert result.endswith("…")
    assert len(result) <= 41
    assert not result[:-1].endswith(" ")


def test_summarise_truncates_unspaced_scripts_without_collapsing() -> None:
    """CJK has no word boundary to fall back on; a naive rsplit would return nothing."""
    text = "中" * 300
    result = summarise(text, 40)
    assert len(result) == 41
    assert result.startswith("中中")


def test_summarise_handles_a_single_enormous_line() -> None:
    assert summarise("x" * 100_000, 50).endswith("…")


# --------------------------------------------------------------------------- #
# Digest
# --------------------------------------------------------------------------- #


def test_digest_is_most_recent_first(store: MemoryStore) -> None:
    for turn in range(5):
        store.put(make_record(f"r:{turn}", f"note number {turn}", turn=turn))
    assert [e.id for e in store.digest()] == ["r:4", "r:3", "r:2", "r:1", "r:0"]


def test_digest_never_exceeds_its_budget(store: MemoryStore) -> None:
    for i in range(200):
        store.put(make_record(f"r:{i:03d}", f"a reasonably wordy note number {i} " * 4, turn=i))

    entries = store.digest(budget_tokens=300)
    cost = sum(estimate_tokens({"ref": e.id, "text": e.summary}) for e in entries)
    assert cost <= 300
    assert 0 < len(entries) < 200


def test_digest_returns_nothing_rather_than_one_oversized_entry(store: MemoryStore) -> None:
    store.put(make_record("r:big", "enormous " * 5000))
    assert store.digest(budget_tokens=5) == []


def test_digest_filters_by_kind_and_lifecycle(store: MemoryStore) -> None:
    store.put(make_record("r:1", "a fact", kind="fact", lifecycle="permanent"))
    store.put(make_record("r:2", "a draft", kind="draft", lifecycle="turn"))

    assert [e.id for e in store.digest(kinds=["fact"])] == ["r:1"]
    assert [e.id for e in store.digest(lifecycle=["turn"])] == ["r:2"]


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


def test_search_ranks_the_relevant_record_first(store: MemoryStore) -> None:
    store.put(make_record("r:1", "the postgres connection pool was exhausted"))
    store.put(make_record("r:2", "a note about frontend styling and css"))
    store.put(make_record("r:3", "postgres tuning notes for the connection pool"))

    hits = store.search("postgres connection pool")
    assert [r.id for r in hits[:2]] == ["r:1", "r:3"] or [r.id for r in hits[:2]] == ["r:3", "r:1"]
    assert "r:2" not in [r.id for r in hits]


def test_search_is_deterministic(store: MemoryStore) -> None:
    for i in range(30):
        store.put(make_record(f"r:{i:02d}", f"shared term plus token{i}", turn=i % 3))
    runs = [[r.id for r in store.search("shared term")] for _ in range(5)]
    assert all(run == runs[0] for run in runs)


def test_search_with_no_match_returns_nothing(store: MemoryStore) -> None:
    store.put(make_record("r:1", "alpha beta"))
    assert store.search("completely unrelated") == []
    assert store.search("") == []


def test_search_is_case_insensitive(store: MemoryStore) -> None:
    store.put(make_record("r:1", "The Database Is Down"))
    assert [r.id for r in store.search("database")] == ["r:1"]


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


def test_purge_evicts_only_the_right_lifecycles(store: MemoryStore) -> None:
    store.put(make_record("r:turn-old", "x", lifecycle="turn", turn=1))
    store.put(make_record("r:turn-new", "x", lifecycle="turn", turn=9))
    store.put(make_record("r:task", "x", lifecycle="task", turn=1, task_id="t-1"))
    store.put(make_record("r:other-task", "x", lifecycle="task", turn=1, task_id="t-2"))
    store.put(make_record("r:session", "x", lifecycle="session", turn=1))
    store.put(make_record("r:permanent", "x", lifecycle="permanent", turn=1))

    removed = store.purge(turn=5, task_id="t-1")

    assert removed == 2
    assert {r.id for r in store.all_records()} == {
        "r:turn-new", "r:other-task", "r:session", "r:permanent",
    }


def test_purge_without_a_task_id_leaves_task_records_alone(store: MemoryStore) -> None:
    store.put(make_record("r:task", "x", lifecycle="task", turn=1, task_id="t-1"))
    assert store.purge(turn=99) == 0


# --------------------------------------------------------------------------- #
# Durability and concurrency
# --------------------------------------------------------------------------- #


def test_state_survives_a_restart(store_factory: Callable[[], MemoryStore]) -> None:
    first = store_factory()
    first.put(make_record("r:keep", "content that must survive", turn=2))
    first.put(make_record("r:gone", "temporary", lifecycle="turn", turn=1))
    first.touch("r:keep", expand=True)
    first.touch("r:keep", hit=True)
    first.purge(turn=2)

    reopened = store_factory()
    assert {r.id for r in reopened.all_records()} == {"r:keep"}
    record = reopened.get("r:keep")
    assert record.expand_count == 1
    assert record.hit_count == 1
    assert record.text == "content that must survive"


def test_a_torn_final_line_does_not_lose_earlier_records(tmp_path) -> None:
    path = tmp_path / "store.jsonl"
    store = JsonlStore(path)
    store.put(make_record("r:1", "good record"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"op": "put", "record": {"id": "r:2"')  # crash mid-write

    reopened = JsonlStore(path)
    assert [r.id for r in reopened.all_records()] == ["r:1"]


def test_concurrent_writers_leave_a_consistent_store(store: MemoryStore) -> None:
    def work(worker: int) -> None:
        for i in range(25):
            record_id = f"r:{worker}-{i}"
            store.put(make_record(record_id, f"payload {worker} {i}", turn=i))
            store.touch(record_id, hit=True)

    threads = [threading.Thread(target=work, args=(w,)) for w in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(store) == 200
    assert all(r.hit_count == 1 for r in store.all_records())
