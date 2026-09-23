"""Tests for multi-dimensional labelling.

The state-scoping test has the same teeth as the scorer's: every request's
state must hold exactly the items its questions ask about. The rest check that
five questions' answers land on the right item and the right dimension, and
that a labelling outage degrades to "unlabelled", never "mislabelled".
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import pytest

from jevctx.label import (
    ENTITY_QUESTIONS,
    QUESTIONS_PER_ITEM,
    TYPE_QUESTION,
    ItemLabel,
    apply_label,
    label_items,
    label_records,
)
from jevctx.store import InMemoryStore, JsonlStore
from jevctx.testing import FakeJevClient
from jevctx.tokens import estimate_tokens
from jevctx.types import (
    MAX_QUESTIONS_PER_REQUEST,
    ChoiceAnswer,
    JevUnavailableError,
    Lifecycle,
    Origin,
    Question,
    Record,
    ScoreItem,
    State,
)

TASK = "Get the build green."

LIFECYCLES: tuple[Lifecycle, ...] = ("turn", "task", "session", "permanent")


def items(n: int, *, size: int = 40) -> list[ScoreItem]:
    out: list[ScoreItem] = []
    for i in range(n):
        text = f"item {i} " + ("payload " * size)
        out.append(ScoreItem(id=f"s:{i}", text=text, tokens=estimate_tokens(text)))
    return out


def record(n: int, *, text: str | None = None) -> Record:
    return Record(
        id=f"r:label-{n}", text=text or f"record {n} " + "body " * 20, kind="elided_segment",
        origin=Origin(source="tool:bash", ref=f"cmd-{n}", turn=n), tokens=0, created_turn=n,
    )


_NUMBER = re.compile(r"\d+")


def _item_index(state: State, ref: str) -> int:
    """The corpus index baked into the item's text, stable no matter which batch
    (and therefore which in-batch ref) the item was asked under."""
    if isinstance(state, Mapping):
        for item in state.get("items") or []:
            if isinstance(item, Mapping) and item.get("ref") == ref:
                match = _NUMBER.search(str(item.get("text", "")))
                if match:
                    return int(match.group())
    return 0


def scripted_answers(
    state: State, questions: Mapping[str, Question], key: str
) -> float | str:
    ref, dimension = key.split(":")
    index = _item_index(state, ref)
    if dimension == "type":
        return "error" if index % 2 else "configuration"
    if dimension == "lifetime":
        return LIFECYCLES[index % 4]
    if dimension == "paths":
        return 0.9 if index % 2 else 0.1
    if dimension == "urls":
        return 0.8 if index % 3 == 0 else 0.2
    return 0.7  # identifiers


# --------------------------------------------------------------------------- #
# The cost rule
# --------------------------------------------------------------------------- #


def test_state_holds_exactly_the_items_its_questions_ask_about() -> None:
    """Mechanises the cost rule for the multi-question shape."""
    client = FakeJevClient.constant(0.5)
    corpus = items(40)
    label_items(client, TASK, corpus)

    assert len(client.calls) >= 2, "40 items at 5 questions each cannot fit in one request"
    seen: set[str] = set()
    for call in client.calls:
        state_text = call.state_text()
        present = {item.id for item in corpus if item.text in state_text}
        asked_refs = {key.split(":")[0] for key in call.questions}
        assert len(present) == len(asked_refs), (
            f"state carries {len(present)} items but asks about {len(asked_refs)}"
        )
        assert not (present & seen), "an item appeared in more than one request's state"
        seen |= present
    assert seen == {item.id for item in corpus}


def test_total_state_cost_is_about_one_pass_over_the_corpus() -> None:
    client = FakeJevClient.constant(0.5)
    corpus = items(40)
    label_items(client, TASK, corpus)

    corpus_tokens = sum(i.tokens for i in corpus)
    total_state = sum(estimate_tokens(call.state) for call in client.calls)
    assert total_state < 2 * corpus_tokens


def test_no_request_exceeds_the_question_cap() -> None:
    client = FakeJevClient.constant(1.0)
    label_items(client, TASK, items(200))
    assert all(len(c.questions) <= MAX_QUESTIONS_PER_REQUEST for c in client.calls)
    assert all(
        len(c.questions) == QUESTIONS_PER_ITEM * len(c.state["items"]) for c in client.calls
    )


def test_every_item_is_asked_all_five_dimensions_under_its_ref() -> None:
    client = FakeJevClient.constant(1.0)
    label_items(client, TASK, items(3))
    first_call = client.calls[0]
    assert set(first_call.questions) == {
        f"i{item}:{dimension}"
        for item in range(3)
        for dimension in ("type", "lifetime", *ENTITY_QUESTIONS)
    }
    for key, question in first_call.questions.items():
        ref = key.split(":")[0]
        assert ref in question.instructions


# --------------------------------------------------------------------------- #
# Answers reach the right item and the right dimension
# --------------------------------------------------------------------------- #


def test_labels_carry_the_answers_of_the_item_they_name() -> None:
    corpus = items(40)
    client = FakeJevClient(scripted_answers)

    labels = label_items(client, TASK, corpus)

    assert [label.item_id for label in labels] == [item.id for item in corpus]
    for index, label in enumerate(labels):
        assert label.failed is False
        assert label.type == ("error" if index % 2 else "configuration")
        assert label.lifecycle == LIFECYCLES[index % 4]
        assert label.entities["paths"] == (0.9 if index % 2 else 0.1)
        assert label.entities["urls"] == (0.8 if index % 3 == 0 else 0.2)
        assert label.entities["identifiers"] == 0.7


def test_labels_do_not_depend_on_worker_count() -> None:
    corpus = items(30)
    serial = label_items(FakeJevClient(scripted_answers), TASK, corpus, max_workers=1)
    parallel = label_items(FakeJevClient(scripted_answers), TASK, corpus, max_workers=8)
    assert serial == parallel


def test_empty_input_makes_no_request() -> None:
    client = FakeJevClient.constant(1.0)
    assert label_items(client, TASK, []) == []
    assert client.calls == []


# --------------------------------------------------------------------------- #
# Failure behaviour
# --------------------------------------------------------------------------- #


def test_labelling_fails_open_to_the_conservative_defaults() -> None:
    """An outage must degrade to 'unlabelled', never to 'mislabelled'."""
    client = FakeJevClient.failing(JevUnavailableError("upstream down"))
    corpus = items(20)

    labels = label_items(client, TASK, corpus)

    assert len(labels) == len(corpus)
    assert all(label.failed for label in labels)
    assert all(label.type == "other" for label in labels)
    assert all(label.lifecycle == "session" for label in labels)
    assert all(
        label.entities == {name: 0.0 for name in ENTITY_QUESTIONS} for label in labels
    )
    assert all("JevUnavailableError" in (label.error or "") for label in labels)


def test_on_error_raise_propagates_instead() -> None:
    client = FakeJevClient.failing(JevUnavailableError("upstream down"))
    with pytest.raises(JevUnavailableError):
        label_items(client, TASK, items(5), on_error="raise")


def test_an_oversized_item_is_labelled_failed_without_a_request() -> None:
    huge = "z" * 200_000
    corpus = [ScoreItem(id="s:huge", text=huge, tokens=estimate_tokens(huge))]
    client = FakeJevClient.constant(0.0)

    labels = label_items(client, TASK, corpus)

    assert labels[0].failed is True
    assert labels[0].error == "oversized"
    assert client.calls == []


def test_a_choice_outside_the_offered_options_fails_open() -> None:
    corpus = items(4)
    client = FakeJevClient(
        lambda state, questions, key: ChoiceAnswer(choice="bogus", confidence=1.0)
    )

    labels = label_items(client, TASK, corpus)

    assert all(label.failed for label in labels)
    assert all(label.type == "other" for label in labels)
    assert all("type" in (label.error or "") for label in labels)
    assert "other" in TYPE_QUESTION.criteria


# --------------------------------------------------------------------------- #
# Applying labels to records
# --------------------------------------------------------------------------- #


def test_apply_label_sets_lifecycle_and_meta() -> None:
    rec = record(1)
    label = ItemLabel(
        item_id=rec.id, type="error", lifecycle="task",
        entities={"paths": 0.9, "urls": 0.2, "identifiers": 0.7},
    )

    assert apply_label(rec, label) is True
    assert rec.lifecycle == "task"
    assert rec.meta["label"] == {
        "type": "error", "entities": {"paths": 0.9, "urls": 0.2, "identifiers": 0.7},
    }


def test_a_failed_label_never_overwrites_what_the_record_already_had() -> None:
    rec = record(1)
    rec.lifecycle = "permanent"
    failed = ItemLabel(
        item_id=rec.id, type="other", lifecycle="session",
        entities={"paths": 0.0, "urls": 0.0, "identifiers": 0.0},
        failed=True, error="upstream down",
    )

    assert apply_label(rec, failed) is False
    assert rec.lifecycle == "permanent"
    assert "label" not in rec.meta


def test_label_records_updates_the_store_in_place() -> None:
    store = InMemoryStore()
    recs = [record(1), record(2)]
    for rec in recs:
        store.put(rec)
    client = FakeJevClient(scripted_answers)

    labels = label_records(client, TASK, recs, store=store)

    assert set(labels) == {rec.id for rec in recs}
    stored = [store.get(rec.id) for rec in recs]
    assert all(stored_rec is not None for stored_rec in stored)
    assert stored[0].lifecycle == LIFECYCLES[1]
    assert stored[1].lifecycle == LIFECYCLES[2]
    assert stored[0].meta["label"]["type"] == "error"


def test_labelled_records_survive_a_jsonl_store_replay(tmp_path) -> None:
    path = tmp_path / "store.jsonl"
    store = JsonlStore(path)
    rec = record(3)
    store.put(rec)
    client = FakeJevClient(scripted_answers)

    label_records(client, TASK, [rec], store=store)

    reopened = JsonlStore(path)
    stored = reopened.get(rec.id)
    assert stored is not None
    assert stored.lifecycle == LIFECYCLES[3]
    assert stored.meta["label"]["type"] == "error"


def test_a_failed_label_is_not_re_put_to_the_store(tmp_path) -> None:
    path = tmp_path / "store.jsonl"
    store = JsonlStore(path)
    rec = record(4)
    rec.lifecycle = "permanent"
    store.put(rec)
    ops_before = path.read_text(encoding="utf-8").count("\n")

    label_records(FakeJevClient.failing(JevUnavailableError("down")), TASK, [rec], store=store)

    assert path.read_text(encoding="utf-8").count("\n") == ops_before
    assert store.get(rec.id).lifecycle == "permanent"
