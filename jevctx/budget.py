"""BudgetPlanner: pack ScoreItems into batches that respect Jev's per-request limits.

This module exists because of how Jev is priced: it bills for ``state`` and gives
questions away free, so the only way to score N items without paying N times for the same
state is to fan many questions out over one shared, minimal state. That in turn means every
Jev request is bounded by three independent caps:

* at most ``MAX_QUESTIONS_PER_REQUEST`` questions per request,
* ``state + all questions`` under ``STATE_PLUS_ALL_QUESTIONS_TOKENS``,
* ``state + the single longest question`` under ``STATE_PLUS_LONGEST_QUESTION_TOKENS``.

``BudgetPlanner.plan`` is the one place that decides how many items share a batch (and
therefore a ``state``). Getting it wrong either overflows a hard Jev limit (a 422, a wasted
round trip) or -- the bug this whole package exists to prevent -- ships a ``state`` bigger
than the batch it accompanies, paying repeatedly for content that was already billed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from jevctx.tokens import estimate_tokens
from jevctx.types import (
    MAX_QUESTIONS_PER_REQUEST,
    STATE_PLUS_ALL_QUESTIONS_TOKENS,
    STATE_PLUS_LONGEST_QUESTION_TOKENS,
    ScoreItem,
)

__all__ = ["Batch", "BudgetPlanner"]

# Token cost of the per-item wrapper scorer.build_state() places around each item's text
# (the `{"ref": "iN", "text": ...}` entry, minus the text itself). Derived once from
# tokens.py's own estimator -- against the state shape scorer.build_state produces --
# rather than a guessed constant, so this planner's notion of "how big is this batch's
# state" tracks what actually gets sent. The ref itself is 2-3 characters ("i0".."i31");
# that variance is far inside the headroom this planner already applies below.
_ITEM_WRAPPER_TOKENS = estimate_tokens({"ref": "i0", "text": ""})


def _question_keys(count: int) -> list[str]:
    """The ``i0..i{count-1}`` refs scorer.py keys a batch's per-item questions by."""
    return [f"i{i}" for i in range(count)]


@dataclass(frozen=True)
class Batch:
    """One Jev request's worth of items.

    ``question_keys[j]`` is the ref ``items[j]`` will be asked under; the two lists are
    built side by side (rather than a single list of pairs) so scorer.py can hand them
    straight to ``build_state`` and to the questions dict without re-deriving the
    numbering. ``meta`` carries out-of-band flags -- currently only
    ``{"oversized": True}``, for an item that cannot fit into a request even alone.
    """

    items: list[ScoreItem]
    question_keys: list[str]
    meta: Mapping[str, Any] = field(default_factory=dict)


class BudgetPlanner:
    """Packs ``ScoreItem``s into `Batch`es that respect the three caps above.

    Packing strategy: greedy first-fit in input order. Items are appended to the batch
    currently being built until adding the next one would break a cap, at which point that
    batch is closed and a new one is started. This is not bin-packing-optimal -- sorting or
    reordering items could sometimes squeeze one more into a batch -- but two things the
    spec cares about outweigh that here: ``plan()`` must preserve input order, and batch
    membership is logged by shadow.py, so it needs to stay stable and predictable
    -- driven by the corpus order, not by an optimizer's incidental choices.
    """

    def __init__(
        self,
        *,
        max_questions: int = MAX_QUESTIONS_PER_REQUEST,
        state_plus_all_questions: int = STATE_PLUS_ALL_QUESTIONS_TOKENS,
        state_plus_longest_question: int = STATE_PLUS_LONGEST_QUESTION_TOKENS,
        headroom: float = 0.9,
    ) -> None:
        self.max_questions = max_questions
        self.state_plus_all_questions = state_plus_all_questions
        self.state_plus_longest_question = state_plus_longest_question
        self.headroom = headroom

    def plan(
        self,
        items: Sequence[ScoreItem],
        question_tokens: int,
        envelope_tokens: int = 0,
    ) -> list[Batch]:
        """Pack ``items`` into order-preserving batches.

        ``question_tokens`` is the estimated cost of one item's question (every item is
        asked a copy of the same question, so a single estimate stands for all of them).
        ``envelope_tokens`` is the fixed, batch-level cost of the state wrapper (the task
        digest plus the outer JSON scaffolding); it is paid once per batch, not once per
        item. ``headroom`` is applied to both token budgets here, not to the raw limits,
        because ``estimate_tokens`` is an estimate and a 422 costs a round trip.

        Every item appears in exactly one returned batch, in input order. An item that
        cannot fit into a request even by itself is returned as its own single-item batch
        with ``meta["oversized"] = True``; it is never dropped and never merged with a
        neighbour.
        """
        all_budget = self.state_plus_all_questions * self.headroom
        longest_budget = self.state_plus_longest_question * self.headroom

        batches: list[Batch] = []
        current: list[ScoreItem] = []
        current_tokens = envelope_tokens

        def flush(pending: list[ScoreItem]) -> None:
            if pending:
                batches.append(
                    Batch(items=list(pending), question_keys=_question_keys(len(pending)))
                )

        for item in items:
            item_tokens = item.tokens + _ITEM_WRAPPER_TOKENS
            solo_tokens = envelope_tokens + item_tokens

            if not self._fits(solo_tokens, 1, question_tokens, all_budget, longest_budget):
                flush(current)
                batches.append(
                    Batch(items=[item], question_keys=_question_keys(1),
                          meta={"oversized": True})
                )
                current = []
                current_tokens = envelope_tokens
                continue

            candidate_tokens = current_tokens + item_tokens
            candidate_count = len(current) + 1
            fits = self._fits(
                candidate_tokens, candidate_count, question_tokens, all_budget, longest_budget
            )
            if fits:
                current.append(item)
                current_tokens = candidate_tokens
            else:
                flush(current)
                current = [item]
                current_tokens = solo_tokens

        flush(current)
        return batches

    def _fits(
        self,
        state_tokens: int,
        count: int,
        question_tokens: int,
        all_budget: float,
        longest_budget: float,
    ) -> bool:
        """Would a batch with this many items and this much state stay under all three caps?"""
        if count > self.max_questions:
            return False
        if state_tokens + question_tokens * count > all_budget:
            return False
        if state_tokens + question_tokens > longest_budget:
            return False
        return True
