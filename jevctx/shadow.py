"""The decision log that makes threshold calibration possible.

Every gate decision and every later outcome is recorded here. That is what turns
"is 0.35 the right threshold" from an argument into a measurement, and it is why
this module exists in Phase 1 even though nothing reads it until Phase 3: the data
has to be accumulating before you need it.

The number to watch is :meth:`ShadowLog.false_negative_rate`. An ``expand`` on
something the gate elided is ground truth that the gate was too aggressive -- the
agent went looking for exactly what was taken away. Nothing else in this design
gives that signal directly.

Records keep a hash and a short preview, never the full text. The store already
holds the original, and a log that is cheap stays enabled.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from jevctx.types import Origin

__all__ = ["ShadowLog", "ShadowStats", "PREVIEW_CHARS"]

PREVIEW_CHARS = 200

DecisionKind = Literal["admit", "retrieve"]
Action = Literal["kept", "elided", "injected", "skipped"]
OutcomeKind = Literal["expand", "hit"]

_ACTIONS: tuple[Action, ...] = ("kept", "elided", "injected", "skipped")


@dataclass(frozen=True)
class ShadowStats:
    total: int
    by_action: Mapping[str, int]
    elided_tokens: int
    kept_tokens: int
    mean_score_by_action: Mapping[str, float]
    false_negatives: int
    false_negative_rate: float

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return (f"{self.total} decisions, {self.by_action.get('elided', 0)} elided "
                f"({self.elided_tokens} tok), {self.false_negatives} later expanded "
                f"({self.false_negative_rate:.1%})")


@dataclass
class _Entry:
    """One logged row, in the shape it is written to disk."""

    type: str
    turn: int
    item_id: str
    kind: str = ""
    score: float = 0.0
    threshold: float = 0.0
    action: str = ""
    tokens: int = 0
    origin: dict[str, Any] = field(default_factory=dict)
    text_sha256: str = ""
    text_preview: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


class ShadowLog:
    """Append-only decision log. Pass ``path=None`` to keep it in memory."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._entries: list[_Entry] = []

    # -- writing ------------------------------------------------------------ #

    def decision(
        self,
        *,
        kind: DecisionKind,
        item_id: str,
        score: float,
        threshold: float,
        action: Action,
        tokens: int,
        origin: Origin,
        text: str,
        turn: int,
    ) -> None:
        entry = _Entry(
            type="decision", turn=turn, item_id=item_id, kind=kind, score=score,
            threshold=threshold, action=action, tokens=tokens, origin=origin.to_dict(),
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            text_preview=text[:PREVIEW_CHARS],
        )
        self._append(entry)

    def outcome(self, *, kind: OutcomeKind, item_id: str, turn: int) -> None:
        self._append(_Entry(type="outcome", turn=turn, item_id=item_id, kind=kind))

    def _append(self, entry: _Entry) -> None:
        with self._lock:
            self._entries.append(entry)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(entry.to_json() + "\n")

    # -- reading ------------------------------------------------------------ #

    @classmethod
    def load(cls, path: str | Path) -> ShadowLog:
        log = cls(path=path)
        source = Path(path)
        if not source.exists():
            return log
        entries: list[_Entry] = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(_Entry(**json.loads(line)))
                except (ValueError, TypeError):
                    continue  # a torn final write
        log._entries = entries
        return log

    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(e.__dict__) for e in self._entries]

    # -- the numbers that matter --------------------------------------------- #

    def _split(self) -> tuple[list[_Entry], list[_Entry]]:
        with self._lock:
            snapshot = list(self._entries)
        return ([e for e in snapshot if e.type == "decision"],
                [e for e in snapshot if e.type == "outcome"])

    def false_negative_rate(self) -> float:
        """Expands of elided items over total elisions.

        Zero elisions means the gate has not removed anything yet, so there is no
        rate to report and this returns 0.0 rather than dividing by zero.
        """
        decisions, outcomes = self._split()
        elided = {e.item_id for e in decisions if e.action == "elided"}
        if not elided:
            return 0.0
        expanded = {e.item_id for e in outcomes if e.kind == "expand"}
        return len(elided & expanded) / len(elided)

    def stats(self) -> ShadowStats:
        decisions, outcomes = self._split()
        return self._summarise(decisions, outcomes,
                               {e.item_id: e.action for e in decisions})

    def replay(self, threshold: float) -> ShadowStats:
        """What a different threshold would have done, from the scores already logged.

        Lets a threshold be tuned offline against real traffic instead of by
        re-running the agent. The number to look at is ``false_negatives``: how many
        items the agent later went back for would this threshold still have removed.
        """
        decisions, outcomes = self._split()
        actions = {
            e.item_id: _counterfactual(e.kind, e.score, threshold)
            for e in decisions
        }
        return self._summarise(decisions, outcomes, actions)

    def _summarise(self, decisions: list[_Entry], outcomes: list[_Entry],
                   actions: Mapping[str, str]) -> ShadowStats:
        by_action = dict.fromkeys(_ACTIONS, 0)
        totals: dict[str, list[float]] = {a: [] for a in _ACTIONS}
        elided_tokens = kept_tokens = 0

        for entry in decisions:
            action = actions.get(entry.item_id, entry.action)
            by_action[action] = by_action.get(action, 0) + 1
            totals.setdefault(action, []).append(entry.score)
            if action == "elided":
                elided_tokens += entry.tokens
            elif action == "kept":
                kept_tokens += entry.tokens

        elided = {i for i, a in actions.items() if a == "elided"}
        expanded = {e.item_id for e in outcomes if e.kind == "expand"}
        false_negatives = len(elided & expanded)

        return ShadowStats(
            total=len(decisions),
            by_action=by_action,
            elided_tokens=elided_tokens,
            kept_tokens=kept_tokens,
            mean_score_by_action={
                a: (sum(v) / len(v) if v else 0.0) for a, v in totals.items()
            },
            false_negatives=false_negatives,
            false_negative_rate=(false_negatives / len(elided)) if elided else 0.0,
        )


def _counterfactual(kind: str, score: float, threshold: float) -> str:
    """The action a given threshold would have produced for a logged score."""
    if kind == "retrieve":
        return "injected" if score >= threshold else "skipped"
    return "kept" if score >= threshold else "elided"
