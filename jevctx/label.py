"""Multi-dimensional labelling: type, lifetime, entities, in one batched call.

Scoring answers one question per item. But a record in the store needs more than
"keep it" to be useful: retrieval wants to know *what* it is, eviction wants to
know *how long* it matters, and indexing wants to know *what it references*.
Asking those dimensions one Jev call each would pay for the item's state several
times over -- so this module fans all of a batch's dimension questions out over
one shared ``state``, exactly as :mod:`jevctx.scorer` does for the gate.

Jev bills for state and gives questions away free, and the per-request cap is
32 questions. Five questions per item (one type ``Choice``, one lifetime
``Choice``, three entity ``Noul``s) means six labelled items share a request.
The dimensions were chosen to be the three the rest of the package can act on:
``lifecycle`` feeds ``MemoryStore.purge``, ``type`` and ``entities`` feed
``meta`` for retrieval filtering, and nothing here invents text -- every answer
is a probability or a pick from a fixed set.

Fails open, with the same reasoning as the scorer: a labelling outage must never
corrupt or evict memory. A failed label is marked ``failed=True`` and
:func:`apply_label` refuses to write it, so the record keeps whatever lifecycle
and metadata it already had -- the conservative defaults are "session" (never
purged) and "other".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from jevctx.tokens import estimate_tokens
from jevctx.types import (
    MAX_QUESTIONS_PER_REQUEST,
    STATE_PLUS_ALL_QUESTIONS_TOKENS,
    STATE_PLUS_LONGEST_QUESTION_TOKENS,
    Answer,
    Choice,
    ChoiceAnswer,
    JevClient,
    JevError,
    Lifecycle,
    MemoryStore,
    Noul,
    NoulAnswer,
    Question,
    Record,
    ScoreItem,
)

__all__ = [
    "ENTITY_QUESTIONS",
    "ItemLabel",
    "LIFETIME_QUESTION",
    "QUESTIONS_PER_ITEM",
    "TYPE_QUESTION",
    "apply_label",
    "label_items",
    "label_records",
]

# --------------------------------------------------------------------------- #
# Questions
#
# Written in English even when the state is not: TypeSafe documents CJK as
# "supported but less reliable", and the question text is the part we control.
# --------------------------------------------------------------------------- #

TYPE_QUESTION = Choice(
    instructions="What kind of content is this item?",
    criteria={
        "error": "A failure: exceptions, stack traces, non-zero exits, compile or lint errors",
        "result": "Output the work produced: computed values, test results, command results",
        "configuration": "Configuration, dependency pins, versions, environment, manifests",
        "documentation": "Reference material: manuals, help text, specs, examples",
        "other": "None of the above",
    },
)

LIFETIME_QUESTION = Choice(
    instructions="How long will this item stay relevant to the task described in `task`?",
    criteria={
        "turn": "Only the immediately following step; stale after that",
        "task": "Until the current subtask finishes, then dead weight",
        "session": "The whole session; the safe default",
        "permanent": "Durable facts worth keeping across sessions",
    },
)

#: Whether the item mentions each kind of durable identifier worth indexing.
ENTITY_QUESTIONS: Mapping[str, Noul] = {
    "paths": Noul(
        instructions="Does this item mention filesystem paths or file names?",
        true="A concrete path or file name appears in the item.",
        false="No path or file name appears in the item.",
    ),
    "urls": Noul(
        instructions="Does this item mention URLs, endpoints, or host names?",
        true="A URL, endpoint, or host name appears in the item.",
        false="No URL, endpoint, or host name appears in the item.",
    ),
    "identifiers": Noul(
        instructions=(
            "Does this item mention durable identifiers: error codes, package or "
            "version pins, ticket or trace ids, function or class names?"
        ),
        true="At least one such identifier appears in the item.",
        false="No such identifier appears in the item.",
    ),
}

#: The per-item questions, in the order their keys are built.
_DIMENSIONS: tuple[str, ...] = ("type", "lifetime", *ENTITY_QUESTIONS)
QUESTIONS_PER_ITEM = len(_DIMENSIONS)


# --------------------------------------------------------------------------- #
# Per-item questions
# --------------------------------------------------------------------------- #


def _item_questions(ref: str) -> dict[str, Question]:
    """The five questions one item is asked, keyed ``"{ref}:{dimension}"``.

    Every item in a batch shares one ``state``, so the instructions name the ref
    -- the same trick :mod:`jevctx.scorer` uses, extended to several dimensions
    per ref.
    """
    named = f"Considering item {ref} only:"
    questions: dict[str, Question] = {
        f"{ref}:type": Choice(instructions=f"{named} {TYPE_QUESTION.instructions}",
                              criteria=TYPE_QUESTION.criteria),
        f"{ref}:lifetime": Choice(instructions=f"{named} {LIFETIME_QUESTION.instructions}",
                                  criteria=LIFETIME_QUESTION.criteria),
    }
    for name, question in ENTITY_QUESTIONS.items():
        questions[f"{ref}:{name}"] = Noul(
            instructions=f"{named} {question.instructions}",
            true=question.true, false=question.false,
        )
    return questions


# Token cost of each dimension's question payload, estimated once with a
# representative ref ("i0") baked into the instructions. Ref length varies by at
# most a character across a batch; the packing headroom absorbs that.
_QUESTION_TOKENS: Mapping[str, int] = {
    dimension: estimate_tokens(question.to_payload())
    for dimension, question in _item_questions("i0").items()
}

# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #

#: What a failed label degrades to. "session" is never purged, "other" filters nothing.
_FAILED_TYPE = "other"
_FAILED_LIFETIME: Lifecycle = "session"


@dataclass(frozen=True)
class ItemLabel:
    """The three dimensions of one item, or the failure to get them."""

    item_id: str
    type: str
    lifecycle: Lifecycle
    entities: Mapping[str, float]
    failed: bool = False
    error: str | None = None
    batch_index: int = -1


def _failed(item_id: str, error: str, batch_index: int) -> ItemLabel:
    return ItemLabel(
        item_id=item_id, type=_FAILED_TYPE, lifecycle=_FAILED_LIFETIME,
        entities={name: 0.0 for name in ENTITY_QUESTIONS},
        failed=True, error=error, batch_index=batch_index,
    )


def _label_from_answers(
    item_id: str, answers: Mapping[str, Answer | None], batch_index: int
) -> ItemLabel:
    type_answer = answers.get("type")
    lifetime_answer = answers.get("lifetime")
    if not isinstance(type_answer, ChoiceAnswer) \
            or type_answer.choice not in TYPE_QUESTION.criteria:
        return _failed(item_id, "missing or invalid 'type' answer", batch_index)
    if not isinstance(lifetime_answer, ChoiceAnswer) \
            or lifetime_answer.choice not in LIFETIME_QUESTION.criteria:
        return _failed(item_id, "missing or invalid 'lifetime' answer", batch_index)

    entities: dict[str, float] = {}
    for name in ENTITY_QUESTIONS:
        answer = answers.get(name)
        if not isinstance(answer, NoulAnswer):
            return _failed(item_id, f"missing or invalid {name!r} answer", batch_index)
        entities[name] = answer.noul

    return ItemLabel(
        item_id=item_id, type=type_answer.choice,
        lifecycle=cast(Lifecycle, lifetime_answer.choice),
        entities=entities, batch_index=batch_index,
    )


# --------------------------------------------------------------------------- #
# Batching
#
# The same three caps :mod:`jevctx.budget` enforces, with one twist: an item now
# costs ``QUESTIONS_PER_ITEM`` questions, not one, so the question cap bounds
# items per request directly and the token budgets multiply it out.
# --------------------------------------------------------------------------- #

_HEADROOM = 0.9
_WRAPPER_TOKENS = estimate_tokens({"ref": "i0", "text": ""})
_SUM_QUESTION_TOKENS = sum(_QUESTION_TOKENS.values())
_LONGEST_QUESTION_TOKENS = max(_QUESTION_TOKENS.values())


@dataclass(frozen=True)
class _Batch:
    items: list[ScoreItem]
    meta: Mapping[str, Any] = field(default_factory=dict)


def _plan_batches(task_digest: str, items: Sequence[ScoreItem]) -> list[_Batch]:
    """Greedy, order-preserving packing -- the BudgetPlanner strategy, re-derived
    for multi-question items rather than bent through a planner built for one."""
    envelope = estimate_tokens(_state_for(task_digest, [], []))
    all_budget = STATE_PLUS_ALL_QUESTIONS_TOKENS * _HEADROOM
    longest_budget = STATE_PLUS_LONGEST_QUESTION_TOKENS * _HEADROOM
    max_items = max(1, MAX_QUESTIONS_PER_REQUEST // QUESTIONS_PER_ITEM)

    def fits(state_tokens: int, count: int) -> bool:
        if count > max_items:
            return False
        if state_tokens + _SUM_QUESTION_TOKENS * count > all_budget:
            return False
        return state_tokens + _LONGEST_QUESTION_TOKENS <= longest_budget

    batches: list[_Batch] = []
    current: list[ScoreItem] = []
    current_tokens = envelope

    def flush(pending: list[ScoreItem]) -> None:
        if pending:
            batches.append(_Batch(items=list(pending)))

    for item in items:
        item_tokens = item.tokens + _WRAPPER_TOKENS
        if not fits(envelope + item_tokens, 1):
            flush(current)
            batches.append(_Batch(items=[item], meta={"oversized": True}))
            current, current_tokens = [], envelope
            continue
        if fits(current_tokens + item_tokens, len(current) + 1):
            current.append(item)
            current_tokens += item_tokens
        else:
            flush(current)
            current, current_tokens = [item], envelope + item_tokens
    flush(current)
    return batches


def _state_for(
    task_digest: str, items: Sequence[ScoreItem], refs: Sequence[str]
) -> dict[str, Any]:
    return {
        "task": task_digest,
        "items": [
            {"ref": ref, "text": item.text} for ref, item in zip(refs, items, strict=True)
        ],
    }


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #


def _label_batch(
    client: JevClient,
    task_digest: str,
    batch: _Batch,
    batch_index: int,
    on_error: Literal["keep", "raise"],
) -> list[ItemLabel]:
    if batch.meta.get("oversized"):
        # Never sent: an oversized item cannot fit into a request even alone, so
        # it degrades to the conservative defaults without a round trip.
        return [_failed(item.id, "oversized", batch_index) for item in batch.items]

    refs = [f"i{i}" for i in range(len(batch.items))]
    state = _state_for(task_digest, batch.items, refs)
    questions = {
        key: question
        for ref in refs
        for key, question in _item_questions(ref).items()
    }

    try:
        answers = client.ask(state, questions)
        return [
            _label_from_answers(
                item.id,
                {dimension: answers.get(f"{ref}:{dimension}") for dimension in _DIMENSIONS},
                batch_index,
            )
            for ref, item in zip(refs, batch.items, strict=True)
        ]
    except JevError as exc:
        if on_error == "raise":
            raise
        return [_failed(item.id, f"{type(exc).__name__}: {exc}", batch_index)
                for item in batch.items]


def label_items(
    client: JevClient,
    task_digest: str,
    items: Sequence[ScoreItem],
    *,
    max_workers: int = 8,
    on_error: Literal["keep", "raise"] = "keep",
) -> list[ItemLabel]:
    """Label every item with its type, lifetime, and entities, batched under Jev's limits.

    Fails open -- see the module docstring. Returns labels in the same order as
    ``items``. Batches are planned once up front and run concurrently through a
    ``ThreadPoolExecutor(max_workers)``; worker count never changes batch
    membership or the answers, only execution order.
    """
    if not items:
        return []

    batches = _plan_batches(task_digest, items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_label_batch, client, task_digest, batch, index, on_error)
            for index, batch in enumerate(batches)
        ]
        batch_labels = [future.result() for future in futures]

    by_item_id = {label.item_id: label for labels in batch_labels for label in labels}
    return [by_item_id[item.id] for item in items]


# --------------------------------------------------------------------------- #
# Applying labels to records
# --------------------------------------------------------------------------- #


def apply_label(record: Record, label: ItemLabel) -> bool:
    """Write a label onto a record in place. Returns ``False`` for a failed label.

    A failed label never overwrites what the record already had: the whole point
    of failing open is that an outage degrades to "unlabelled", never to
    "mislabelled". ``lifecycle`` is written through because it drives
    ``MemoryStore.purge``; type and entities go to ``meta["label"]``.
    """
    if label.failed:
        return False
    record.lifecycle = label.lifecycle
    record.meta["label"] = {"type": label.type, "entities": dict(label.entities)}
    return True


def label_records(
    client: JevClient,
    task_digest: str,
    records: Sequence[Record],
    *,
    store: MemoryStore | None = None,
    max_workers: int = 8,
    on_error: Literal["keep", "raise"] = "keep",
) -> dict[str, ItemLabel]:
    """Label records and apply the labels in place.

    Records are usually already in a store; pass ``store`` to re-``put`` each
    successfully labelled record, which is what makes the labels survive a
    ``JsonlStore`` restart (last op for an id wins on replay). Returns one
    label per record id, in input order.
    """
    items = [
        ScoreItem(id=record.id, text=record.text,
                  tokens=record.tokens or estimate_tokens(record.text))
        for record in records
    ]
    labels = label_items(client, task_digest, items,
                         max_workers=max_workers, on_error=on_error)

    applied: dict[str, ItemLabel] = {}
    for record, label in zip(records, labels, strict=True):
        if apply_label(record, label) and store is not None:
            store.put(record)
        applied[record.id] = label
    return applied
