"""External memory: where relocated content lives.

The gate in ``pipeline.py`` never deletes. Everything it takes out of context ends
up here, which makes the store the thing that turns a false drop from lost
information into a round trip. That is the whole reason it exists, and it is why
``get`` must be cheap and total: ``expand()`` is on the agent's critical path.

Two implementations share one behavioural contract. ``JsonlStore`` is append-only
for the same reason the context buffer is: an append can be replayed and tailed,
a rewrite cannot, and a store that is only ever appended to cannot be corrupted by
a crash halfway through an update.

``digest()`` carries the constraint that shapes everything upstream. Its output
becomes a Jev ``state``, and Jev caps state at 32k tokens, so the digest is
budgeted rather than complete. Retrieval reranks a window of memory, not all of it.
"""

from __future__ import annotations

import json
import math
import re
import threading
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jevctx.tokens import estimate_tokens
from jevctx.types import DigestEntry, Lifecycle, Origin, Record

__all__ = ["InMemoryStore", "JsonlStore", "summarise", "record_to_dict", "record_from_dict"]

_WORD = re.compile(r"\w+", re.UNICODE)
_DEFAULT_SUMMARY_CHARS = 120

#: BM25's term-frequency saturation constant. The exact value barely matters here:
#: search is only a candidate prefilter, and Jev does the real ranking.
_K1 = 1.2


def summarise(text: str, limit: int = _DEFAULT_SUMMARY_CHARS) -> str:
    """First meaningful line, collapsed and truncated. Deterministic, no model call.

    Phase 2 replaces this with a Jev labelling pass; until then a store must still
    produce a digest, and a digest needs a one-line view of every record.
    """
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if not line:
            continue
        if len(line) <= limit:
            return line
        cut = line[:limit]
        head = cut.rsplit(" ", 1)[0]
        # CJK and other unspaced scripts have no word boundary to fall back on.
        if len(head) < limit // 2:
            head = cut
        return head + "…"
    return ""


def record_to_dict(record: Record) -> dict[str, Any]:
    return {
        "id": record.id, "text": record.text, "kind": record.kind,
        "origin": record.origin.to_dict(), "tokens": record.tokens,
        "created_turn": record.created_turn, "lifecycle": record.lifecycle,
        "summary": record.summary, "task_id": record.task_id, "meta": record.meta,
        "expand_count": record.expand_count, "hit_count": record.hit_count,
    }


def record_from_dict(data: dict[str, Any]) -> Record:
    return Record(
        id=data["id"], text=data["text"], kind=data["kind"],
        origin=Origin.from_dict(data["origin"]), tokens=int(data["tokens"]),
        created_turn=int(data["created_turn"]), lifecycle=data.get("lifecycle", "session"),
        summary=data.get("summary", ""), task_id=data.get("task_id"),
        meta=dict(data.get("meta") or {}),
        expand_count=int(data.get("expand_count", 0)),
        hit_count=int(data.get("hit_count", 0)),
    )


@dataclass
class _Index:
    """The in-memory state both implementations operate on."""

    records: dict[str, Record] = field(default_factory=dict)


class InMemoryStore:
    """A store that lives and dies with the process. The reference implementation."""

    def __init__(self, *, summary_max_chars: int = _DEFAULT_SUMMARY_CHARS) -> None:
        self._index = _Index()
        self._lock = threading.RLock()
        self._summary_max_chars = summary_max_chars

    # -- writes ------------------------------------------------------------- #

    def put(self, record: Record) -> str:
        with self._lock:
            stored = self._prepare(record)
            self._index.records[stored.id] = stored
            self._on_put(stored)
            return stored.id

    def touch(self, record_id: str, *, expand: bool = False, hit: bool = False) -> None:
        with self._lock:
            record = self._index.records.get(record_id)
            if record is None:
                return
            if expand:
                record.expand_count += 1
            if hit:
                record.hit_count += 1
            self._on_touch(record_id, expand=expand, hit=hit)

    def purge(self, *, turn: int, task_id: str | None = None) -> int:
        """Evict by lifecycle. ``session`` and ``permanent`` are never touched."""
        with self._lock:
            doomed = [
                r.id for r in self._index.records.values()
                if (r.lifecycle == "turn" and r.created_turn < turn)
                or (r.lifecycle == "task" and task_id is not None and r.task_id == task_id)
            ]
            for record_id in doomed:
                del self._index.records[record_id]
            if doomed:
                self._on_purge(doomed)
            return len(doomed)

    # -- reads -------------------------------------------------------------- #

    def get(self, record_id: str) -> Record | None:
        with self._lock:
            return self._index.records.get(record_id)

    def digest(
        self,
        *,
        budget_tokens: int = 24_000,
        kinds: Collection[str] | None = None,
        lifecycle: Collection[Lifecycle] | None = None,
    ) -> list[DigestEntry]:
        """Most-recent-first, truncated at ``budget_tokens``.

        Budgeted because the result becomes a Jev state and Jev caps state at 32k.
        An entry that does not fit is dropped, never truncated: a half summary
        would be scored as if it were whole.
        """
        with self._lock:
            candidates = [
                r for r in self._index.records.values()
                if (kinds is None or r.kind in kinds)
                and (lifecycle is None or r.lifecycle in lifecycle)
            ]
        candidates.sort(key=lambda r: (-r.created_turn, r.id))

        entries: list[DigestEntry] = []
        spent = 0
        for record in candidates:
            entry = DigestEntry(
                id=record.id, summary=record.summary or summarise(record.text),
                kind=record.kind, tokens=record.tokens, created_turn=record.created_turn,
            )
            cost = estimate_tokens({"ref": entry.id, "text": entry.summary})
            if spent + cost > budget_tokens:
                break
            entries.append(entry)
            spent += cost
        return entries

    def search(self, query: str, *, limit: int = 50) -> list[Record]:
        """BM25-lite prefilter. Deterministic; Jev does the real ranking afterwards."""
        terms = _tokenise(query)
        with self._lock:
            records = list(self._index.records.values())
        if not terms or not records:
            return []

        tokenised = {r.id: _tokenise(f"{r.summary}\n{r.text}") for r in records}
        total = len(records)
        scored: list[tuple[float, int, str, Record]] = []
        for record in records:
            tokens = tokenised[record.id]
            score = 0.0
            for term in set(terms):
                frequency = tokens.count(term)
                if not frequency:
                    continue
                document_frequency = sum(1 for t in tokenised.values() if term in t)
                idf = math.log(1 + total / document_frequency)
                score += idf * frequency / (frequency + _K1)
            if score > 0:
                scored.append((-score, record.created_turn, record.id, record))
        scored.sort(key=lambda row: (row[0], row[1], row[2]))
        return [row[3] for row in scored[:limit]]

    def all_records(self) -> list[Record]:
        with self._lock:
            return sorted(self._index.records.values(), key=lambda r: r.id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._index.records)

    # -- hooks for the durable subclass -------------------------------------- #

    def _prepare(self, record: Record) -> Record:
        if not record.summary:
            record.summary = summarise(record.text, self._summary_max_chars)
        if not record.tokens:
            record.tokens = estimate_tokens(record.text)
        return record

    def _on_put(self, record: Record) -> None: ...

    def _on_touch(self, record_id: str, *, expand: bool, hit: bool) -> None: ...

    def _on_purge(self, record_ids: Sequence[str]) -> None: ...


class JsonlStore(InMemoryStore):
    """Append-only JSONL with an in-memory index, rebuilt on construction.

    Every mutation is an appended op rather than a rewrite, so a crash mid-write
    can lose the last line but cannot corrupt what came before, and the file stays
    safe to tail while an agent is running. Last op for an id wins on replay.
    """

    def __init__(self, path: str | Path, *,
                 summary_max_chars: int = _DEFAULT_SUMMARY_CHARS) -> None:
        super().__init__(summary_max_chars=summary_max_chars)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._replay()

    def _replay(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    op = json.loads(line)
                except ValueError:
                    continue  # a torn final write; everything before it stands
                self._apply(op)

    def _apply(self, op: dict[str, Any]) -> None:
        kind = op.get("op")
        if kind == "put":
            record = record_from_dict(op["record"])
            self._index.records[record.id] = record
        elif kind == "touch":
            record = self._index.records.get(op["id"])
            if record is not None:
                record.expand_count += int(bool(op.get("expand")))
                record.hit_count += int(bool(op.get("hit")))
        elif kind == "purge":
            for record_id in op.get("ids", []):
                self._index.records.pop(record_id, None)

    def _append(self, op: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(op, ensure_ascii=False) + "\n")

    def _on_put(self, record: Record) -> None:
        self._append({"op": "put", "record": record_to_dict(record)})

    def _on_touch(self, record_id: str, *, expand: bool, hit: bool) -> None:
        self._append({"op": "touch", "id": record_id, "expand": expand, "hit": hit})

    def _on_purge(self, record_ids: Sequence[str]) -> None:
        self._append({"op": "purge", "ids": list(record_ids)})


def _tokenise(text: str) -> list[str]:
    return _WORD.findall(text.casefold())
