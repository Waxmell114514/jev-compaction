"""Tests for jevctx.budget.BudgetPlanner.

No network: these tests only exercise pure packing logic over ``ScoreItem``s built by
hand, never a ``JevClient``.
"""

from __future__ import annotations

import random

from jevctx.budget import Batch, BudgetPlanner
from jevctx.tokens import estimate_tokens
from jevctx.types import ScoreItem

# The same per-item JSON-wrapper cost BudgetPlanner accounts for internally, re-derived
# here from the state shape scorer.build_state produces (`{"ref": ..., "text": ...}`) --
# via the public, frozen `estimate_tokens`, not by importing BudgetPlanner's private
# constant -- so these tests check the real invariant, not an implementation detail.
_ITEM_WRAPPER_TOKENS = estimate_tokens({"ref": "i0", "text": ""})


def _item(item_id: str, tokens: int) -> ScoreItem:
    return ScoreItem(id=item_id, text=f"text for {item_id}", tokens=tokens)


def _item_state_tokens(item: ScoreItem) -> int:
    return item.tokens + _ITEM_WRAPPER_TOKENS


def _batch_state_tokens(batch: Batch, envelope_tokens: int) -> int:
    return envelope_tokens + sum(_item_state_tokens(it) for it in batch.items)


def test_empty_input_returns_no_batches() -> None:
    assert BudgetPlanner().plan([], question_tokens=10) == []


def test_question_cap_binds_when_tokens_are_plentiful() -> None:
    # 40 tiny items: token budgets never bind, so only the 32-question cap should.
    items = [_item(f"i{i}", tokens=5) for i in range(40)]
    batches = BudgetPlanner().plan(items, question_tokens=5, envelope_tokens=10)

    assert [len(b.items) for b in batches] == [32, 8]
    assert all(not b.meta.get("oversized") for b in batches)


def test_order_is_preserved_and_every_item_appears_exactly_once() -> None:
    items = [_item(f"i{i}", tokens=5) for i in range(97)]
    batches = BudgetPlanner().plan(items, question_tokens=5, envelope_tokens=10)

    flat_ids = [it.id for b in batches for it in b.items]
    assert flat_ids == [it.id for it in items]


def test_question_keys_align_positionally_with_items() -> None:
    items = [_item(f"i{i}", tokens=5) for i in range(5)]
    batches = BudgetPlanner().plan(items, question_tokens=5, envelope_tokens=10)

    assert len(batches) == 1
    batch = batches[0]
    assert batch.question_keys == ["i0", "i1", "i2", "i3", "i4"]
    assert len(batch.question_keys) == len(batch.items)


def test_token_budget_splits_batches_before_question_cap_would() -> None:
    # Small planner budgets and large-ish items: the token caps should force a split well
    # before 32 items ever accumulate in one batch.
    planner = BudgetPlanner(state_plus_all_questions=300, state_plus_longest_question=200,
                             headroom=1.0)
    items = [_item(f"i{i}", tokens=50) for i in range(10)]
    batches = planner.plan(items, question_tokens=10, envelope_tokens=0)

    assert len(batches) > 1
    assert all(len(b.items) < 32 for b in batches)


def test_oversized_item_is_isolated_and_not_dropped() -> None:
    items = [
        _item("small-1", tokens=10),
        _item("huge", tokens=10_000_000),
        _item("small-2", tokens=10),
    ]
    batches = BudgetPlanner().plan(items, question_tokens=10, envelope_tokens=10)

    # every item still appears exactly once, in order
    assert [it.id for b in batches for it in b.items] == ["small-1", "huge", "small-2"]

    oversized_batches = [b for b in batches if b.meta.get("oversized")]
    assert len(oversized_batches) == 1
    assert [it.id for it in oversized_batches[0].items] == ["huge"]
    assert oversized_batches[0].question_keys == ["i0"]

    # the huge item was never merged into a neighbour's batch
    for b in batches:
        if not b.meta.get("oversized"):
            assert all(it.id != "huge" for it in b.items)


def test_plan_is_deterministic() -> None:
    items = [_item(f"i{i}", tokens=random.Random(1).randint(1, 500)) for i in range(150)]
    planner = BudgetPlanner()

    first = planner.plan(items, question_tokens=30, envelope_tokens=100)
    second = planner.plan(items, question_tokens=30, envelope_tokens=100)

    assert [[it.id for it in b.items] for b in first] == [[it.id for it in b.items] for b in second]


def test_plan_never_violates_caps_under_randomised_sizes() -> None:
    """Over randomised item-size distributions, plan() must never
    emit a batch that violates the 32-question cap or either token budget; every item must
    appear exactly once, order must be preserved, and an item too large to fit alone must
    come back as an oversized single-item batch."""
    rng = random.Random(20240917)
    planner = BudgetPlanner(headroom=0.9)  # defaults for the token/question caps
    question_tokens = 40
    envelope_tokens = 120
    all_budget = planner.state_plus_all_questions * planner.headroom
    longest_budget = planner.state_plus_longest_question * planner.headroom

    for _trial in range(25):
        n = rng.randint(1, 220)
        sizes = [
            rng.choice(
                [
                    rng.randint(1, 50),  # tiny
                    rng.randint(50, 3000),  # ordinary
                    rng.randint(3000, 40000),  # occasionally oversized alone
                ]
            )
            for _ in range(n)
        ]
        items = [_item(f"it{i}", tokens=size) for i, size in enumerate(sizes)]

        batches = planner.plan(items, question_tokens, envelope_tokens)

        # every item appears exactly once, order preserved
        assert [it.id for b in batches for it in b.items] == [it.id for it in items]

        for batch in batches:
            assert len(batch.items) == len(batch.question_keys)
            count = len(batch.items)
            state_tokens = _batch_state_tokens(batch, envelope_tokens)

            if batch.meta.get("oversized"):
                assert count == 1
                # the flag must be justified: it really doesn't fit alone
                assert (
                    state_tokens + question_tokens > longest_budget
                    or state_tokens + question_tokens > all_budget
                )
                continue

            assert count <= planner.max_questions
            assert state_tokens + question_tokens * count <= all_budget
            assert state_tokens + question_tokens <= longest_budget


def test_custom_limits_are_honoured() -> None:
    planner = BudgetPlanner(max_questions=3)
    items = [_item(f"i{i}", tokens=5) for i in range(7)]
    batches = planner.plan(items, question_tokens=5, envelope_tokens=5)

    assert [len(b.items) for b in batches] == [3, 3, 1]
