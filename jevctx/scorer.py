"""score_items(): the shared Jev-scoring primitive used at both the admit and retrieve ends.

Fails open. A Jev outage (or a malformed answer) must never silently strip an agent's
context: with the default ``on_error="keep"``, anything that goes wrong scores the affected
items ``1.0`` -- "keep it" -- rather than dropping them or raising. Silently discarding
context because a *scoring service* had an outage would be a far worse failure mode than
occasionally keeping something that could have been elided; the caller can always retry the
score later, but it cannot recover text this function silently dropped.

This module fans many single-item ``Noul`` questions out over one shared ``state`` per
batch -- never one request per item -- because Jev is priced by state, not
by question. ``BudgetPlanner`` (jevctx.budget) decides batch membership; this module only
builds the request each batch implies, sends it, and maps the answers back to the caller's
items by position.

The API is synchronous: parallelism across batches comes from a ``ThreadPoolExecutor``, not
asyncio, so an agent loop that is not itself async can still call this directly. The Jev
client's own concurrency semaphore is the real rate limiter; this
module does not add a second one on top of it.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from jevctx.budget import Batch, BudgetPlanner
from jevctx.tokens import estimate_tokens
from jevctx.types import (
    JevClient,
    JevError,
    JevValidationError,
    Noul,
    NoulAnswer,
    Question,
    ScoreItem,
    ScoreResult,
)

__all__ = ["build_state", "score_items", "score_map"]


def build_state(
    task_digest: str, items: Sequence[ScoreItem], refs: Sequence[str],
    intent: str | None = None,
) -> dict[str, Any]:
    """The exact Jev ``state`` for one batch: the task digest plus these items only.

    With an ``intent`` (what the agent was looking for when it made the call that
    produced these items) the state carries it too, under ``intent``; without one the
    state is exactly what it always was.

    Public, not just an implementation detail, so the state-scoping test -- and any
    caller that wants to preview a request -- can rely on this shape
    directly instead of re-deriving it. ``refs`` and ``items`` must be the same length and
    are paired up positionally, exactly as ``Batch.question_keys`` and ``Batch.items`` are.
    """
    state: dict[str, Any] = {"task": task_digest}
    if intent:
        state["intent"] = intent
    state["items"] = [
        {"ref": ref, "text": item.text} for ref, item in zip(refs, items, strict=True)
    ]
    return state


def _ref_question(question: Noul, ref: str) -> Noul:
    """A copy of ``question`` whose instructions name ``ref``.

    Every item in a batch shares one ``state``, so the question text is the only thing that
    tells the model which of the batch's items a given answer is about.
    """
    return Noul(
        instructions=f"Considering item {ref} only: {question.instructions}",
        true=question.true,
        false=question.false,
    )


def _fail_open(batch: Batch, batch_index: int, error: str) -> list[ScoreResult]:
    return [
        ScoreResult(item_id=item.id, score=1.0, failed=True, error=error, batch_index=batch_index)
        for item in batch.items
    ]


def _score_batch(
    client: JevClient,
    task_digest: str,
    batch: Batch,
    batch_index: int,
    question: Noul,
    on_error: Literal["keep", "raise"],
    intent: str | None = None,
) -> list[ScoreResult]:
    if batch.meta.get("oversized"):
        # Never sent: an oversized item defaults to "keep" unconditionally, regardless of
        # `on_error` -- this is not a Jev failure, it is a request we chose not to make.
        return _fail_open(batch, batch_index, "oversized")

    refs = batch.question_keys
    state = build_state(task_digest, batch.items, refs, intent)
    questions: dict[str, Question] = {ref: _ref_question(question, ref) for ref in refs}

    try:
        answers = client.ask(state, questions)
        results: list[ScoreResult] = []
        for ref, item in zip(refs, batch.items, strict=True):
            answer = answers.get(ref)
            if not isinstance(answer, NoulAnswer):
                # A shape violation is a protocol bug, not a per-item judgement -- treated
                # as a JevError so it falls into the exact same fail-open path below.
                raise JevValidationError(
                    f"expected a NoulAnswer for ref {ref!r}, got {type(answer).__name__}"
                )
            results.append(
                ScoreResult(item_id=item.id, score=answer.value, failed=False, error=None,
                            batch_index=batch_index)
            )
        return results
    except JevError as exc:
        if on_error == "raise":
            raise
        return _fail_open(batch, batch_index, f"{type(exc).__name__}: {exc}")


def score_items(
    client: JevClient,
    task_digest: str,
    items: Sequence[ScoreItem],
    question: Noul,
    *,
    planner: BudgetPlanner | None = None,
    max_workers: int = 8,
    on_error: Literal["keep", "raise"] = "keep",
    intent: str | None = None,
) -> list[ScoreResult]:
    """Score every item in ``items`` against ``question``, batched under Jev's limits.

    Fails open -- see the module docstring. Returns results in the same order as ``items``,
    with ``batch_index`` filled in so a caller or ``ShadowLog`` can tell which request
    produced which score. Batches are planned once up front by ``planner`` (a fresh
    ``BudgetPlanner()`` if omitted) and then run concurrently through a
    ``ThreadPoolExecutor(max_workers)``; ``max_workers=1`` and ``max_workers=8`` must (and
    do) produce identical results, since concurrency only changes execution order, never
    batch membership or the answers themselves.
    """
    if not items:
        return []

    planner = planner or BudgetPlanner()
    question_tokens = estimate_tokens(_ref_question(question, "i0").to_payload())
    envelope_tokens = estimate_tokens(build_state(task_digest, [], [], intent))
    batches = planner.plan(items, question_tokens, envelope_tokens)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_score_batch, client, task_digest, batch, idx, question, on_error, intent)
            for idx, batch in enumerate(batches)
        ]
        batch_results = [future.result() for future in futures]

    by_item_id = {result.item_id: result for results in batch_results for result in results}
    return [by_item_id[item.id] for item in items]


def score_map(
    client: JevClient,
    task_digest: str,
    items: Sequence[ScoreItem],
    question: Noul,
    *,
    planner: BudgetPlanner | None = None,
    max_workers: int = 8,
    on_error: Literal["keep", "raise"] = "keep",
) -> dict[str, float]:
    """``score_items`` for callers that just want ``item_id -> score``."""
    results = score_items(
        client, task_digest, items, question, planner=planner, max_workers=max_workers,
        on_error=on_error,
    )
    return {result.item_id: result.score for result in results}
