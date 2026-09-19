"""Integration tests for the write-side gate and read-side retrieval.

These are the tests that exercise segments + scorer + store + shadow together,
so they are also the package's smoke test for the whole of Phase 1.
"""

from __future__ import annotations

import pytest

from jevctx.pipeline import (
    ADMIT_QUESTION,
    EXPAND_TOOL_SCHEMA,
    GateConfig,
    admit,
    expand,
    find_pointers,
    format_pointer,
    mark_hits,
    parse_pointer,
    reconstruct,
    retrieve,
)
from jevctx.shadow import ShadowLog
from jevctx.store import InMemoryStore
from jevctx.testing import FakeJevClient
from jevctx.tokens import estimate_tokens
from jevctx.types import (
    JevUnavailableError,
    Origin,
    Pointer,
    Record,
)

TASK = "Install dependencies and get the test suite passing."
ORIGIN = Origin(source="tool:bash", ref="npm install", turn=3)


def noisy_log(noise_lines: int = 60, signal_lines: int = 60) -> str:
    """Half progress noise, half content worth keeping, interleaved in blocks."""
    blocks: list[str] = []
    for i in range(6):
        blocks.append(
            "\n".join(f"npm http fetch GET 200 https://registry.npmjs.org/pkg-{i}-{j} 41ms"
                      for j in range(noise_lines // 6))
        )
        blocks.append(
            "\n".join(f"IMPORTANT added dependency pkg-{i}-{j}@2.{j}.0 to package.json"
                      for j in range(signal_lines // 6))
        )
    return "\n\n".join(blocks) + "\n"


def keep_noise_client() -> FakeJevClient:
    return FakeJevClient.by_text(lambda t: 0.95 if "IMPORTANT" in t else 0.02)


def fresh() -> tuple[InMemoryStore, ShadowLog]:
    return InMemoryStore(), ShadowLog(path=None)


# --------------------------------------------------------------------------- #
# Admission
# --------------------------------------------------------------------------- #


def test_small_output_is_not_gated_and_costs_no_jev_call() -> None:
    store, log = fresh()
    client = keep_noise_client()
    raw = "ok\n"
    result = admit(raw, ORIGIN, task_digest=TASK, turn=1, client=client, store=store, log=log)

    assert result.text == raw
    assert result.gated is False
    assert client.calls == []


def test_admit_elides_noise_and_saves_tokens() -> None:
    store, log = fresh()
    raw = noisy_log()
    result = admit(raw, ORIGIN, task_digest=TASK, turn=1, client=keep_noise_client(),
                   store=store, log=log)

    assert result.gated is True
    assert result.tripwire is None
    assert result.pointers, "expected the noise runs to be relocated"
    assert result.result_tokens < result.original_tokens
    assert "IMPORTANT added dependency" in result.text
    # The pointer summary quotes its first elided line by design, so check the
    # body with the pointer lines removed.
    body = "\n".join(ln for ln in result.text.splitlines() if not ln.startswith("[[elided"))
    assert "npm http fetch" not in body
    assert len(store) == len(result.pointers)


def test_admit_expand_round_trip_is_byte_exact() -> None:
    """Nothing the gate removes is lost."""
    store, log = fresh()
    raw = noisy_log()
    result = admit(raw, ORIGIN, task_digest=TASK, turn=1, client=keep_noise_client(),
                   store=store, log=log)

    assert reconstruct(result.text, store) == raw

    # ...and the same holds going through the agent-facing expand() tool.
    rebuilt = result.text
    for pointer in find_pointers(result.text):
        original = expand(pointer.id, store=store, log=log, turn=2)
        rebuilt = rebuilt.replace(format_pointer(pointer) + "\n", original)
    assert rebuilt == raw


def test_gate_fails_open_when_jev_is_down() -> None:
    """An outage must never strip an agent's context."""
    store, log = fresh()
    client = FakeJevClient.failing(JevUnavailableError("503"))
    raw = noisy_log()
    result = admit(raw, ORIGIN, task_digest=TASK, turn=1, client=client, store=store, log=log)

    assert result.text == raw
    assert result.pointers == ()
    assert len(store) == 0
    assert all(s.failed and s.score == 1.0 for s in result.scores)


def test_tripwire_distrusts_a_scorer_that_wants_everything_gone() -> None:
    """A scorer that wants to drop almost everything is reporting a bad question."""
    store, log = fresh()
    client = FakeJevClient.constant(0.0)
    raw = noisy_log()
    result = admit(raw, ORIGIN, task_digest=TASK, turn=1, client=client, store=store, log=log)

    assert result.tripwire == "max_elide_fraction"
    assert result.text == raw
    assert len(store) == 0


def test_shadow_only_changes_nothing_but_logs_everything() -> None:
    """Step 2 of the rollout: score and log everything, change nothing."""
    store, log = fresh()
    raw = noisy_log()
    config = GateConfig(shadow_only=True)
    result = admit(raw, ORIGIN, task_digest=TASK, turn=1, client=keep_noise_client(),
                   store=store, log=log, config=config)

    assert result.text == raw
    assert len(store) == 0
    assert log.stats().total == len(result.scores) > 0


def test_protected_kinds_survive_a_low_score() -> None:
    store, log = fresh()
    trace = (
        "Traceback (most recent call last):\n"
        '  File "/app/main.py", line 42, in <module>\n'
        "    run()\n"
        '  File "/app/run.py", line 11, in run\n'
        "    raise ValueError(\"boom\")\n"
        "ValueError: boom\n"
    )
    raw = trace + noisy_log()
    # 0.2 is below keep_threshold but above protected_floor.
    client = FakeJevClient.by_text(lambda t: 0.2 if "Traceback" in t else 0.95)
    result = admit(raw, ORIGIN, task_digest=TASK, turn=1, client=client, store=store, log=log)

    assert "ValueError: boom" in result.text


def test_admit_state_never_carries_more_than_the_batch_asks_about() -> None:
    """The cost rule -- state is billed, questions are not -- through admit()."""
    store, log = fresh()
    client = keep_noise_client()
    raw = noisy_log()
    admit(raw, ORIGIN, task_digest=TASK, turn=1, client=client, store=store, log=log)

    assert client.calls
    total_state = sum(estimate_tokens(call.state) for call in client.calls)
    # Scoring the corpus must cost about one pass over it, not one pass per batch.
    assert total_state < 2 * estimate_tokens(raw)


# --------------------------------------------------------------------------- #
# Pointers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "pointer",
    [
        Pointer(id="r:7f3a91", lines=(12, 25), tokens=380, summary="npm install progress"),
        Pointer(id="r:0000aa", lines=None, tokens=1, summary=""),
        Pointer(id="r:11bb22", lines=(1, 1), tokens=9, summary='he said "hi" \\ bye'),
    ],
)
def test_pointer_format_round_trips(pointer: Pointer) -> None:
    assert parse_pointer(format_pointer(pointer)) == pointer


def test_parse_pointer_rejects_ordinary_text() -> None:
    assert parse_pointer("just a line of output") is None
    assert parse_pointer("[[elided]]") is None


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


def seed_store(store: InMemoryStore, n: int = 20) -> None:
    for i in range(n):
        relevant = i % 4 == 0
        text = (f"The database connection string is postgres://db-{i}/app"
                if relevant else f"Unrelated scratch note number {i}")
        store.put(Record(
            id=f"r:{i:06d}", text=text, kind="fact", origin=ORIGIN,
            tokens=estimate_tokens(text), created_turn=i, summary=text,
        ))


def test_retrieve_returns_top_k_above_threshold() -> None:
    store, log = fresh()
    seed_store(store)
    client = FakeJevClient.by_text(lambda t: 0.9 if "postgres://" in t else 0.1)

    records = retrieve(TASK, turn=5, client=client, store=store, log=log, k=3)

    assert len(records) == 3
    assert all("postgres://" in r.text for r in records)
    assert log.stats().by_action["injected"] == 3


def test_retrieve_on_an_empty_store_makes_no_call() -> None:
    store, log = fresh()
    client = FakeJevClient.constant(1.0)
    assert retrieve(TASK, turn=1, client=client, store=store, log=log) == []
    assert client.calls == []


def test_expand_is_recorded_as_a_false_negative() -> None:
    store, log = fresh()
    raw = noisy_log()
    result = admit(raw, ORIGIN, task_digest=TASK, turn=1, client=keep_noise_client(),
                   store=store, log=log)

    assert log.false_negative_rate() == 0.0
    expand(result.pointers[0].id, store=store, log=log, turn=2)
    assert log.false_negative_rate() > 0.0
    assert store.get(result.pointers[0].id).expand_count == 1


def test_mark_hits_feeds_the_phase_three_prior() -> None:
    store, log = fresh()
    seed_store(store, n=4)
    mark_hits(["r:000000", "r:000001"], store=store, log=log, turn=7)
    assert store.get("r:000000").hit_count == 1
    assert store.get("r:000002").hit_count == 0


def test_expand_tool_schema_is_registrable() -> None:
    assert EXPAND_TOOL_SCHEMA["name"] == "expand"
    assert EXPAND_TOOL_SCHEMA["input_schema"]["required"] == ["id"]


def test_admit_question_is_written_in_english() -> None:
    # CJK is "supported but less reliable"; question text is ours to control.
    assert all(ord(ch) < 0x2000 for ch in ADMIT_QUESTION.instructions)
