"""The write-side gate and the read-side retrieval, wired together.

The observation this module is built on: **admission and retrieval are the same
primitive**. At write time you ask "will this be needed later"; at read time you
ask "is this needed now". Both are ``score_items`` over a task digest, so both go
through :func:`jevctx.scorer.score_items` and differ only in the question.

The gate **never deletes**. Low-scoring content is relocated to the store and
leaves a pointer the agent can ``expand()``. That one decision is what makes the
gate safe to turn on: a false drop becomes a round trip instead of lost
information, and the prefix still only ever grows.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from jevctx.jev import HttpJevClient
from jevctx.label import ENTITY_QUESTIONS, LIFETIME_QUESTION, TYPE_QUESTION
from jevctx.scorer import score_items
from jevctx.segments import segment
from jevctx.shadow import ShadowLog
from jevctx.tokens import estimate_tokens
from jevctx.types import (
    DigestEntry,
    JevBudgetError,
    JevClient,
    MemoryStore,
    Noul,
    Origin,
    Pointer,
    Record,
    ScoreItem,
    ScoreResult,
    Segment,
    content_id,
)

__all__ = [
    "GateConfig", "DEFAULT_GATE_CONFIG", "AdmitResult", "admit", "retrieve", "render_records", "expand", "reconstruct",
    "format_pointer", "parse_pointer", "find_pointers",
    "ADMIT_QUESTION", "RETRIEVE_QUESTION", "EXPAND_TOOL_SCHEMA",
]


# --------------------------------------------------------------------------- #
# Questions
#
# Written in English even when the state is not: TypeSafe documents CJK as
# "supported but less reliable", and the question text is the part we control.
# --------------------------------------------------------------------------- #

ADMIT_QUESTION = Noul(
    instructions=(
        "Will this item still be needed later in the task described in `task`? "
        "Answer true if it contains facts, identifiers, errors, results, or decisions "
        "that a later step may have to refer back to. Answer false only if it is "
        "progress noise, repeated boilerplate, or formatting with no retained content."
    ),
    true="The item carries information a later step may need.",
    false="The item is noise that can be recovered from the store if ever needed.",
)

RETRIEVE_QUESTION = Noul(
    instructions=(
        "Is this stored item relevant to the task described in `task` right now? "
        "Answer true if the next step is likely to need it. Answer false if it belongs "
        "to unrelated work, or is superseded by something more recent."
    ),
    true="The item is relevant to the current step.",
    false="The item is not relevant right now.",
)


# --------------------------------------------------------------------------- #
# Pointers
# --------------------------------------------------------------------------- #

_POINTER_RE = re.compile(
    r"\[\[elided id=(?P<id>[A-Za-z0-9:_.-]+)"
    r"(?: lines=(?P<start>\d+)-(?P<end>\d+))?"
    r" tokens=(?P<tokens>\d+)"
    r' "(?P<summary>(?:[^"\\]|\\.)*)"\]\]'
)


def format_pointer(pointer: Pointer) -> str:
    """Render the one-line stand-in left in context for relocated content."""
    lines = f" lines={pointer.lines[0]}-{pointer.lines[1]}" if pointer.lines else ""
    summary = pointer.summary.replace("\\", "\\\\").replace('"', '\\"')
    return f'[[elided id={pointer.id}{lines} tokens={pointer.tokens} "{summary}"]]'


def parse_pointer(line: str) -> Pointer | None:
    """Inverse of :func:`format_pointer`. Returns None if the line is not a pointer."""
    match = _POINTER_RE.search(line)
    if match is None:
        return None
    start, end = match.group("start"), match.group("end")
    summary = re.sub(r"\\(.)", r"\1", match.group("summary"))
    return Pointer(
        id=match.group("id"),
        lines=(int(start), int(end)) if start and end else None,
        tokens=int(match.group("tokens")),
        summary=summary,
    )


def find_pointers(text: str) -> list[Pointer]:
    """Every pointer in a rendered block, in order of appearance."""
    return [p for p in (parse_pointer(m.group(0)) for m in _POINTER_RE.finditer(text))
            if p is not None]


def reconstruct(text: str, store: MemoryStore) -> str:
    """Substitute every pointer back with the text it stands for.

    Exact inverse of what :func:`admit` produced, byte for byte, provided the
    records are still in the store.
    """

    def substitute(match: re.Match[str]) -> str:
        pointer = parse_pointer(match.group(0))
        if pointer is None:  # pragma: no cover - the regex just matched
            return match.group(0)
        record = store.get(pointer.id)
        return record.text if record is not None else match.group(0)

    # The rendered form is the pointer line plus its newline; consume both so the
    # substituted text lands exactly where the original ran.
    return re.sub(_POINTER_RE.pattern + r"\n?", substitute, text)


# --------------------------------------------------------------------------- #
# Admission
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GateConfig:
    #: Deliberately low. A false drop costs far more than a false keep, so the
    #: threshold is biased hard toward keeping.
    keep_threshold: float = 0.35
    #: Below this, don't gate at all -- the Jev round trip costs more than the tokens.
    min_gate_tokens: int = 400
    #: If the scorer wants to drop more than this, distrust the scorer, not the output.
    max_elide_fraction: float = 0.7
    protected_kinds: frozenset[str] = frozenset({"stacktrace", "diff"})
    protected_floor: float = 0.05
    #: Step 2 of the supported rollout: score and log everything,
    #: elide nothing. Not a debug flag.
    shadow_only: bool = False
    summary_max_chars: int = 120
    max_workers: int = 8


#: Module-level singleton so the default is a stable object, not a per-call one.
DEFAULT_GATE_CONFIG = GateConfig()


@dataclass(frozen=True)
class AdmitResult:
    text: str
    gated: bool
    kept: tuple[Segment, ...] = ()
    pointers: tuple[Pointer, ...] = ()
    scores: tuple[ScoreResult, ...] = ()
    tripwire: str | None = None
    original_tokens: int = 0
    result_tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def saved_tokens(self) -> int:
        return max(0, self.original_tokens - self.result_tokens)


def _summarise(text: str, limit: int) -> str:
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if line:
            if len(line) <= limit:
                return line
            cut = line[:limit].rsplit(" ", 1)[0] or line[:limit]
            return cut + "…"
    return ""


def admit(
    raw: str,
    origin: Origin,
    *,
    task_digest: str,
    turn: int,
    client: JevClient,
    store: MemoryStore,
    log: ShadowLog,
    config: GateConfig = DEFAULT_GATE_CONFIG,
) -> AdmitResult:
    """Gate one piece of raw output on its way into the work area.

    Returns the text to append, with low-scoring runs replaced by pointers. The
    elided content is in ``store``; nothing is destroyed.
    """
    original_tokens = estimate_tokens(raw)
    if original_tokens < config.min_gate_tokens:
        return AdmitResult(text=raw, gated=False, original_tokens=original_tokens,
                           result_tokens=original_tokens)

    segments = segment(raw, origin)
    if not segments:
        return AdmitResult(text=raw, gated=False, original_tokens=original_tokens,
                           result_tokens=original_tokens)

    items = [ScoreItem.from_segment(s) for s in segments]
    results = score_items(client, task_digest, items, ADMIT_QUESTION,
                          max_workers=config.max_workers)
    by_id = {r.item_id: r for r in results}

    # Decide per segment, then check the tripwire before acting on any of it.
    elide_flags: list[bool] = []
    for seg in segments:
        score = by_id[seg.id].score
        floor = config.protected_floor if seg.kind in config.protected_kinds \
            else config.keep_threshold
        elide_flags.append(score < floor)

    elided_tokens = sum(s.tokens for s, drop in zip(segments, elide_flags, strict=True) if drop)
    total_tokens = sum(s.tokens for s in segments) or 1
    tripwire: str | None = None
    if elided_tokens / total_tokens > config.max_elide_fraction:
        tripwire = "max_elide_fraction"
        elide_flags = [False] * len(segments)

    for seg, drop in zip(segments, elide_flags, strict=True):
        log.decision(
            kind="admit", item_id=seg.id, score=by_id[seg.id].score,
            threshold=config.keep_threshold,
            action="elided" if (drop and not config.shadow_only) else "kept",
            tokens=seg.tokens, origin=origin, text=seg.text, turn=turn,
        )

    if config.shadow_only or tripwire is not None or not any(elide_flags):
        return AdmitResult(
            text=raw, gated=True, kept=tuple(segments), scores=tuple(results),
            tripwire=tripwire, original_tokens=original_tokens,
            result_tokens=original_tokens,
            meta={"shadow_only": config.shadow_only},
        )

    parts: list[str] = []
    kept: list[Segment] = []
    pointers: list[Pointer] = []
    run: list[Segment] = []

    def flush_run() -> None:
        if not run:
            return
        text = "".join(s.text for s in run)
        record = Record(
            id=content_id(text, salt=origin.ref or "", prefix="r"),
            text=text, kind="elided_segment", origin=origin,
            tokens=sum(s.tokens for s in run), created_turn=turn,
            summary=_summarise(text, config.summary_max_chars),
            meta={"segment_ids": [s.id for s in run],
                  "kinds": sorted({s.kind for s in run})},
        )
        record_id = store.put(record)
        pointer = Pointer(
            id=record_id,
            lines=(run[0].line_span[0], run[-1].line_span[1])
            if run[0].line_span and run[-1].line_span else None,
            tokens=record.tokens,
            summary=record.summary,
        )
        pointers.append(pointer)
        parts.append(format_pointer(pointer) + "\n")
        run.clear()

    for seg, drop in zip(segments, elide_flags, strict=True):
        if drop:
            run.append(seg)
        else:
            flush_run()
            parts.append(seg.text)
            kept.append(seg)
    flush_run()

    text = "".join(parts)
    return AdmitResult(
        text=text, gated=True, kept=tuple(kept), pointers=tuple(pointers),
        scores=tuple(results), tripwire=None, original_tokens=original_tokens,
        result_tokens=estimate_tokens(text),
    )


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


def render_records(records: Sequence[Record]) -> str:
    """Exact model-facing payload, including metadata, for retrieval budgeting."""
    return json.dumps([
        {"id": r.id, "origin": r.origin.to_dict(), "lifecycle": r.lifecycle,
         "label": r.meta.get("label"), "text": r.text}
        for r in records
    ], ensure_ascii=False)


def retrieve(
    task_digest: str,
    *,
    turn: int,
    client: JevClient | None,
    store: MemoryStore,
    log: ShadowLog,
    k: int = 5,
    budget_tokens: int = 24_000,
    threshold: float = 0.5,
    max_workers: int = 8,
    result_budget_tokens: int = 4000,
    content_type: str | None = None,
    lifecycle: str | None = None,
    entity: str | None = None,
    shadow_only: bool = False,
    usage: dict | None = None,
) -> list[Record]:
    """Merge lexical/recent candidates, filter, rerank and budget whole records."""
    if not isinstance(task_digest, str) or not task_digest.strip():
        raise ValueError("query must be nonempty")
    if type(k) is not int or not 1 <= k <= 10:
        raise ValueError("k must be between 1 and 10")
    if type(result_budget_tokens) is not int or not 128 <= result_budget_tokens <= 16000:
        raise ValueError("result budget must be between 128 and 16000")
    for value, choices in ((content_type, TYPE_QUESTION.criteria),
                           (lifecycle, LIFETIME_QUESTION.criteria),
                           (entity, ENTITY_QUESTIONS)):
        if value is not None and value not in choices:
            raise ValueError("invalid label filter")
    candidates = store.search(task_digest, limit=50)
    candidates += [store.get(e.id) for e in store.digest(budget_tokens=budget_tokens)[:50]]
    by_record = {}
    entries = []
    spent = 0
    for record in candidates:
        if record is None:
            continue
        # An elided fragment links to its full output; legacy fragments still work.
        record = store.get(record.meta.get("full_output_id", "")) or record
        if record.id in by_record:
            continue
        label = record.meta.get("label", {})
        if content_type is not None and label.get("type") != content_type:
            continue
        if lifecycle is not None and record.lifecycle != lifecycle:
            continue
        if entity is not None and label.get("entities", {}).get(entity, 0) < 0.5:
            continue
        entry = DigestEntry(id=record.id, summary=record.summary, kind=record.kind,
                            tokens=record.tokens, created_turn=record.created_turn)
        cost = estimate_tokens({"ref": entry.id, "text": entry.summary})
        if spent + cost > budget_tokens:
            continue
        spent += cost
        entries.append(entry)
        by_record[record.id] = record
    if not entries:
        return []

    items = [ScoreItem(id=e.id, text=e.summary, tokens=estimate_tokens(e.summary)) for e in entries]

    def score(active_client):
        results = score_items(active_client, task_digest, items, RETRIEVE_QUESTION,
                              max_workers=max_workers, on_error="raise")
        if any(result.failed for result in results):
            raise JevBudgetError("retrieval summaries or query exceed scoring budget")
        if usage is not None:
            meter = getattr(active_client, "usage", None)
            usage.update({key: getattr(meter, key, None)
                          for key in ("input_tokens", "output_tokens", "requests")})
        return results

    if client is None:
        with HttpJevClient() as live:
            results = score(live)
    else:
        results = score(client)
    ranked = sorted(results, key=lambda r: r.score, reverse=True)
    selected = []
    seen_text = set()
    for result in ranked:
        record = by_record[result.item_id]
        if result.score < threshold or record.text in seen_text:
            continue
        if estimate_tokens(render_records([*selected, record])) > result_budget_tokens:
            continue
        selected.append(record)
        seen_text.add(record.text)
        if len(selected) == k:
            break
    chosen = {r.id for r in selected} if not shadow_only else set()

    by_id = {e.id: e for e in entries}
    for result in results:
        entry = by_id[result.item_id]
        log.decision(
            kind="retrieve", item_id=result.item_id, score=result.score,
            threshold=threshold,
            action="injected" if result.item_id in chosen else "skipped",
            tokens=entry.tokens, origin=Origin(source="retrieval", ref=None, turn=turn),
            text=entry.summary, turn=turn,
        )

    return selected


def expand(record_id: str, *, store: MemoryStore, log: ShadowLog, turn: int) -> str:
    """Return the original text behind a pointer.

    Every call is a recorded false negative: the gate removed something the agent
    turned out to need. ``ShadowLog.false_negative_rate`` is built on this.
    """
    record = store.get(record_id)
    if record is None:
        raise KeyError(f"no stored record for pointer id {record_id!r}")
    store.touch(record_id, expand=True)
    log.outcome(kind="expand", item_id=record_id, turn=turn)
    # A record merges a run of segments, but the gate logged its decision per
    # segment. Attribute the expand back to each of them, or the false negative
    # never links up with the score that caused it -- which is exactly what
    # ShadowLog.replay needs to tune a threshold.
    for segment_id in record.meta.get("segment_ids", []):
        log.outcome(kind="expand", item_id=segment_id, turn=turn)
    return record.text


def mark_hits(record_ids: Sequence[str], *, store: MemoryStore, log: ShadowLog,
              turn: int) -> None:
    """Record that retrieved items were actually used. Phase 3 calibrates on this."""
    for record_id in record_ids:
        store.touch(record_id, hit=True)
        log.outcome(kind="hit", item_id=record_id, turn=turn)


#: Ready to register with the agent's tool list. The gate is only safe if the
#: agent can actually undo it, so this is not optional.
EXPAND_TOOL_SCHEMA: dict[str, Any] = {
    "name": "expand",
    "description": (
        "Retrieve the full text behind an [[elided id=... ]] pointer in the conversation. "
        "Use it whenever a pointer's summary suggests it holds something you need."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "The id from the pointer, e.g. r:7f3a91."}
        },
        "required": ["id"],
    },
}
