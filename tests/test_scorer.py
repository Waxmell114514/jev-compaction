"""Tests for the shared scoring primitive.

The state-scoping test is the one with teeth. Everything else here checks that
scores reach the right items; that one checks the design's central cost claim.
"""

from __future__ import annotations

import pytest

from jevctx.budget import BudgetPlanner
from jevctx.scorer import build_state, score_items, score_map
from jevctx.testing import FakeJevClient
from jevctx.tokens import estimate_tokens
from jevctx.types import (
    MAX_QUESTIONS_PER_REQUEST,
    JevUnavailableError,
    Noul,
    ScoreItem,
)

QUESTION = Noul(
    instructions="Will this be needed later?",
    true="It will be needed.",
    false="It is noise.",
)
TASK = "Get the build green."


def items(n: int, *, size: int = 40, seed: int = 0) -> list[ScoreItem]:
    out: list[ScoreItem] = []
    for i in range(n):
        text = f"item-{seed}-{i} " + ("payload " * size)
        out.append(ScoreItem(id=f"s:{seed}-{i}", text=text, tokens=estimate_tokens(text)))
    return out


# --------------------------------------------------------------------------- #
# The cost rule
# --------------------------------------------------------------------------- #


def test_state_holds_exactly_the_items_its_questions_ask_about() -> None:
    """SPEC.md section 7.3, mechanising the section 1.1 cost rule."""
    client = FakeJevClient.constant(0.5)
    corpus = items(100)
    score_items(client, TASK, corpus, QUESTION)

    assert len(client.calls) >= 4, "100 items must not fit in one 32-question request"
    seen: set[str] = set()
    for call in client.calls:
        state_text = call.state_text()
        present = {item.id for item in corpus if item.text in state_text}
        asked = set(call.questions)
        assert len(present) == len(asked), (
            f"state carries {len(present)} items but asks {len(asked)} questions"
        )
        assert not (present & seen), "an item appeared in more than one request's state"
        seen |= present
    assert seen == {item.id for item in corpus}


def test_total_state_cost_is_about_one_pass_over_the_corpus() -> None:
    """The 'whole document in every state' bug shows up here as a multiple."""
    client = FakeJevClient.constant(0.5)
    corpus = items(100)
    score_items(client, TASK, corpus, QUESTION)

    corpus_tokens = sum(i.tokens for i in corpus)
    total_state = sum(estimate_tokens(call.state) for call in client.calls)
    assert total_state < 2 * corpus_tokens


def test_no_request_exceeds_the_question_cap() -> None:
    client = FakeJevClient.constant(1.0)
    score_items(client, TASK, items(200), QUESTION)
    assert all(len(c.questions) <= MAX_QUESTIONS_PER_REQUEST for c in client.calls)


# --------------------------------------------------------------------------- #
# Answers reach the right items
# --------------------------------------------------------------------------- #


def test_every_item_receives_its_own_score_across_batches() -> None:
    """The off-by-one that batching bugs actually produce."""
    corpus = items(90)
    expected = {item.id: (hash(item.id) % 100) / 100 for item in corpus}
    lookup = {item.text: expected[item.id] for item in corpus}
    client = FakeJevClient.by_text(lambda t: lookup[t])

    results = score_items(client, TASK, corpus, QUESTION)

    assert [r.item_id for r in results] == [i.id for i in corpus]
    for result in results:
        assert result.score == pytest.approx(expected[result.item_id])
        assert result.failed is False


@pytest.mark.parametrize("workers", [1, 8])
def test_results_do_not_depend_on_worker_count(workers: int) -> None:
    corpus = items(70)
    lookup = {item.text: (i % 7) / 10 for i, item in enumerate(corpus)}
    client = FakeJevClient.by_text(lambda t: lookup[t])

    results = score_items(client, TASK, corpus, QUESTION, max_workers=workers)
    assert [(r.item_id, r.score) for r in results] == [
        (item.id, pytest.approx(lookup[item.text])) for item in corpus
    ]


def test_score_map_keys_by_item_id() -> None:
    corpus = items(5)
    client = FakeJevClient.constant(0.25)
    assert score_map(client, TASK, corpus, QUESTION) == {i.id: 0.25 for i in corpus}


def test_question_instructions_name_the_ref() -> None:
    client = FakeJevClient.constant(1.0)
    score_items(client, TASK, items(3), QUESTION)
    for key, question in client.calls[0].questions.items():
        assert key in question.instructions
        assert QUESTION.instructions in question.instructions
        assert question.true == QUESTION.true


def test_build_state_pairs_refs_with_items_positionally() -> None:
    corpus = items(3)
    state = build_state(TASK, corpus, ["i0", "i1", "i2"])
    assert state["task"] == TASK
    assert [entry["ref"] for entry in state["items"]] == ["i0", "i1", "i2"]
    assert [entry["text"] for entry in state["items"]] == [i.text for i in corpus]


# --------------------------------------------------------------------------- #
# Failure behaviour
# --------------------------------------------------------------------------- #


def test_scoring_fails_open_so_an_outage_cannot_strip_context() -> None:
    """SPEC.md section 7.5."""
    client = FakeJevClient.failing(JevUnavailableError("upstream down"))
    corpus = items(50)

    results = score_items(client, TASK, corpus, QUESTION)

    assert len(results) == len(corpus)
    assert all(r.score == 1.0 and r.failed for r in results)
    assert all("JevUnavailableError" in (r.error or "") for r in results)


def test_on_error_raise_propagates_instead() -> None:
    client = FakeJevClient.failing(JevUnavailableError("upstream down"))
    with pytest.raises(JevUnavailableError):
        score_items(client, TASK, items(5), QUESTION, on_error="raise")


def test_an_oversized_item_is_kept_without_a_request() -> None:
    huge = "z" * 200_000
    corpus = [ScoreItem(id="s:huge", text=huge, tokens=estimate_tokens(huge))]
    client = FakeJevClient.constant(0.0)

    results = score_items(client, TASK, corpus, QUESTION)

    assert results[0].score == 1.0
    assert results[0].failed is True
    assert results[0].error == "oversized"
    assert client.calls == []


def test_empty_input_makes_no_request() -> None:
    client = FakeJevClient.constant(1.0)
    assert score_items(client, TASK, [], QUESTION) == []
    assert client.calls == []


def test_a_custom_planner_is_honoured() -> None:
    client = FakeJevClient.constant(1.0)
    planner = BudgetPlanner(max_questions=4)
    score_items(client, TASK, items(20), QUESTION, planner=planner)
    assert all(len(c.questions) <= 4 for c in client.calls)
    assert len(client.calls) == 5
