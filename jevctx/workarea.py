"""Work-area compaction: rewrite only the tail, only when it pays, then freeze it.

The admission gate (:func:`jevctx.pipeline.admit`) decides once, as output
arrives, with no hindsight. A few turns later much more is known: the agent has
read the file it was looking for, the test it ran has told it what it needed. But
by then that output sits in every request, and every request after re-reads it.

This module is the second, later decision, on the harness's actual transcript::

    [frozen prefix .......... | work area ...........]
     never rewritten, cached    tool outputs here may be replaced by a pointer

- **What may change.** Tool outputs after the last commit point, and nothing else:
  user and assistant messages are never touched. A compacted output becomes one
  ``[[elided id=...]]`` line; the text the model had seen goes to the store, so
  ``expand`` and ``recall`` still reach it. An output the admission profile marks
  as the code to change (``change_site``) waits until an edit follows it. An output
  a later call made obsolete (:mod:`jevctx.supersede`: run again, read again, its
  file edited since) is a candidate without asking Jev.
- **When to look.** Only when new tool output arrived since the last look and the
  work area's tool output is big enough to be worth a Jev call.
- **What Jev is asked**, in one request: is the agent at a *commit point* (it just
  finished a step -- reproduced the bug, found the code, made or verified a
  change)? And, per work-area output, will later steps still need it?
- **Whether to act: the arithmetic.** Rewriting the transcript at the first
  compacted output invalidates the provider's cache from there on, so the
  ``T`` tokens after it are billed once at the full input price instead of the
  cache-read price. Dropping ``S`` tokens saves the cache-read price on them on
  every remaining turn ``R``. Compact only if::

      S * R * p_cache  >  (T - S) * (p_input - p_cache)

  With ``p_input / p_cache`` of 50 (cheap models) this almost never holds; at 10
  (typical of pricier ones) a half-stale work area pays back within a few turns.
  :class:`jevctx.ledger.CacheLedger` assumes the 10x case throughout; this module
  takes the prices as given.
- **Freezing.** At a commit point, or after ``max_uncommitted_turns``, everything
  up to now becomes frozen prefix and is never considered again.

Every compaction is remembered and handed back on every later call, so the harness
rewrites the same outputs the same way on every request: the cache breaks once, at
the compaction, not on every turn after.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from jevctx.pipeline import format_pointer
from jevctx.scorer import score_items
from jevctx.shadow import ShadowLog
from jevctx.tokens import estimate_tokens
from jevctx.types import (
    JevClient,
    JevError,
    MemoryStore,
    Noul,
    NoulAnswer,
    Origin,
    Pointer,
    Record,
    ScoreItem,
    content_id,
)

__all__ = [
    "COMMIT_QUESTION", "STILL_NEEDED_QUESTION", "TailItem", "WorkAreaConfig", "WorkArea",
    "Decision", "compaction_pays", "remaining_turns",
]

COMMIT_QUESTION = Noul(
    instructions=(
        "`task` is what the agent is doing; `recent` is what it said and did most recently. "
        "Has the agent just finished a distinct step of the task -- for example reproduced the "
        "problem, located the code responsible, made a change, or checked that a change works -- "
        "so that the tool output it gathered for that step has served its purpose?"
    ),
    true="The agent has just completed a step and is moving on.",
    false="The agent is still in the middle of a step.",
)

STILL_NEEDED_QUESTION = Noul(
    instructions=(
        "`task` is what the agent is doing and `recent` what it is doing now. This item is "
        "earlier tool output from the same session. Will a later step still need to look at "
        "it? Answer false if it has served its purpose: it was read to find something that has "
        "since been found, it is superseded by newer output, or it concerns work that is done."
    ),
    true="A later step will still need this output.",
    false="This output has served its purpose.",
)


@dataclass(frozen=True)
class TailItem:
    """One transcript element, as the harness sees it, in order.

    ``id`` must be stable across requests (a message or part id). Only ``kind ==
    "tool"`` items with ``text`` are ever compacted; the rest count only as tokens
    that a rewrite would push out of the cache.
    """

    id: str
    kind: str
    tokens: int
    text: str | None = None
    tool: str | None = None
    call_id: str | None = None


@dataclass(frozen=True)
class WorkAreaConfig:
    #: USD per million tokens; only the ratio matters to the decision.
    price_input: float = 3.0
    price_cache_read: float = 0.3
    #: Below this much work-area tool output, don't ask Jev at all.
    min_work_tokens: int = 4000
    #: Outputs Jev thinks still needed at or above this are never compacted.
    still_needed_threshold: float = 0.3
    commit_threshold: float = 0.5
    #: Remaining turns: ``expected_turns - turn`` early on, never below
    #: ``min_remaining_turns``. A session past the prior still runs on, but not for as
    #: long again: on 101 recorded SWE-bench runs (median 24 turns) the mean number
    #: of turns left was 11-16 from turn 15 on, whatever the turn. The earlier
    #: estimate, ``turn`` itself, said 30 at turn 30 and paid for late rewrites that
    #: never paid back.
    expected_turns: int = 30
    min_remaining_turns: int = 12
    #: Freeze the work area after this many turns without a commit point.
    max_uncommitted_turns: int = 12
    #: Outputs smaller than this are not worth a pointer.
    min_item_tokens: int = 200
    #: An output whose admission profile gives it one of these roles (the argmax, or
    #: presence at or above ``protected_presence``) is not compacted until an edit
    #: follows it: until then it is the code the agent is about to change. Live, 3 of
    #: the 4 compacted outputs the agent went on to edit were such outputs.
    protected_roles: tuple[str, ...] = ("change_site",)
    protected_presence: float = 0.5
    edit_tools: tuple[str, ...] = ("edit", "write", "multiedit", "patch", "apply_patch")
    card_tokens: int = 250
    recent_chars: int = 1500


def compaction_pays(saved: int, tail_after: int, remaining_turns: int,
                    config: WorkAreaConfig) -> tuple[bool, float, float]:
    """``(pays, benefit, cost)`` in USD for dropping ``saved`` tokens from a tail of
    ``tail_after`` tokens (counted from the first rewritten item), ``remaining_turns``
    requests before the session ends."""
    per = 1e-6
    benefit = saved * remaining_turns * config.price_cache_read * per
    cost = max(0, tail_after - saved) * (config.price_input - config.price_cache_read) * per
    return benefit > cost, benefit, cost


def remaining_turns(turn: int, config: WorkAreaConfig) -> int:
    return max(config.min_remaining_turns, config.expected_turns - turn)


@dataclass
class Decision:
    action: str                     # "skip", "commit", "compact", "compact+commit", "reset"
    replacements: dict[str, str]
    commit_probability: float | None = None
    candidates: int = 0
    saved_tokens: int = 0
    tail_tokens: int = 0
    remaining_turns: int = 0
    benefit_usd: float = 0.0
    cost_usd: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "replacements"} | {
            "replaced": len(self.replacements)}


@dataclass
class WorkArea:
    """Per-session state: where the frozen prefix ends and what has been compacted."""

    config: WorkAreaConfig = field(default_factory=WorkAreaConfig)
    #: Id of the last frozen transcript item; None means nothing is frozen yet.
    committed_through: str | None = None
    committed_turn: int = 0
    replacements: dict[str, str] = field(default_factory=dict)
    seen_tool_ids: set[str] = field(default_factory=set)
    _outdated_seen: set[str] = field(default_factory=set)
    stats: dict[str, int] = field(default_factory=lambda: {
        "evaluations": 0, "commits": 0, "compactions": 0, "compacted_items": 0,
        "compacted_tokens": 0, "declined_compactions": 0, "resets": 0,
        "outdated_candidates": 0, "outdated_compacted": 0})

    # -- the decision ------------------------------------------------------- #

    def decide(self, items: Sequence[TailItem], *, task: str, recent: str, turn: int,
               client: JevClient, store: MemoryStore, log: ShadowLog,
               outdated: Callable[[str | None], str] | None = None) -> Decision:
        """``outdated(call_id)`` says why a later action made that output obsolete ('' if
        none, :mod:`jevctx.supersede`). Such outputs need no Jev question: they are
        candidates as they stand, and the code-to-change rule does not hold them."""
        ids = [item.id for item in items]
        if self.committed_through is not None and self.committed_through not in ids:
            # The harness rewrote history itself (its own compaction): the cache is gone
            # anyway, so start over rather than guess where the prefix now ends.
            self.committed_through, self.committed_turn = None, turn
            self.stats["resets"] += 1
            return Decision("reset", dict(self.replacements), reason="commit marker vanished")
        start = ids.index(self.committed_through) + 1 if self.committed_through else 0
        work = list(items[start:])
        tools = [i for i in work if i.kind == "tool" and i.text is not None
                 and i.id not in self.replacements]
        new = [i for i in tools if i.id not in self.seen_tool_ids]
        self.seen_tool_ids.update(i.id for i in tools)
        work_tool_tokens = sum(i.tokens for i in tools)
        overdue = turn - self.committed_turn >= self.config.max_uncommitted_turns

        # Outputs a later call made obsolete, anywhere: freezing protects the prefix
        # from second-guessing, and these need no guess. The price check still decides.
        reasons: dict[str, str] = {}
        if outdated is not None:
            for item in items:
                if item.kind == "tool" and item.text is not None and item.id not in self.replacements:
                    why = outdated(item.call_id)
                    if why:
                        reasons[item.id] = why
        newly_outdated = set(reasons) - self._outdated_seen
        self.stats["outdated_candidates"] += len(newly_outdated)
        self._outdated_seen |= newly_outdated
        prefix_obsolete = [i for i in items[:start] if i.id in reasons]
        ask_jev = bool(new) and work_tool_tokens >= self.config.min_work_tokens
        if not ask_jev and not newly_outdated:
            if overdue and work:
                self._commit(work[-1].id, turn)
                return Decision("commit", dict(self.replacements), reason="overdue, nothing to score")
            return Decision("skip", dict(self.replacements),
                            reason="no new tool output" if not new else "work area too small")

        commit_p: float | None = None
        scores = {item_id: 0.0 for item_id in reasons}
        if ask_jev:
            self.stats["evaluations"] += 1
            state_recent = recent[-self.config.recent_chars:]
            try:
                commit_p = self._ask_commit(client, task, state_recent)
            except JevError:
                commit_p = 0.0   # fail safe: no commit, and no guessed compaction below
            digest = f"{task[:2000]}\n\nRecent: {state_recent}"
            cards = [ScoreItem(id=i.id, text=self._card(i), tokens=0) for i in tools
                     if i.id not in reasons]
            cards = [ScoreItem(id=c.id, text=c.text, tokens=estimate_tokens(c.text)) for c in cards]
            scores.update({r.item_id: (r.score if not r.failed else 1.0)
                           for r in (score_items(client, digest, cards, STILL_NEEDED_QUESTION)
                                     if cards else [])})
        stale = [i for i in [*prefix_obsolete, *tools] if i.tokens >= self.config.min_item_tokens
                 and scores.get(i.id, 1.0) < self.config.still_needed_threshold
                 and (i.id in reasons or not self._protected(i, items, store))]
        for item in (tools if ask_jev else []):
            log.decision(kind="compact", item_id=item.id, score=scores.get(item.id, 1.0),
                         threshold=self.config.still_needed_threshold,
                         action="elided" if item in stale else "kept", tokens=item.tokens,
                         origin=Origin(source=f"tool:{item.tool}", ref=item.call_id, turn=turn),
                         text=item.text or "", turn=turn,
                         labels={"commit_probability": commit_p,
                                 "outdated": reasons.get(item.id, "")})

        decision = Decision("skip", {}, commit_probability=commit_p, candidates=len(stale))
        if stale:
            first = min(ids.index(i.id) for i in stale)
            tail = sum(i.tokens for i in items[first:])
            pointer_tokens = 40 * len(stale)
            saved = sum(i.tokens for i in stale) - pointer_tokens
            remaining = remaining_turns(turn, self.config)
            pays, benefit, cost = compaction_pays(saved, tail, remaining, self.config)
            decision.saved_tokens, decision.tail_tokens = saved, tail
            decision.remaining_turns, decision.benefit_usd, decision.cost_usd = remaining, benefit, cost
            if pays and saved > 0:
                for item in stale:
                    self.replacements[item.id] = self._relocate(item, store, turn,
                                                                reasons.get(item.id, ""))
                self.stats["compactions"] += 1
                self.stats["outdated_compacted"] += sum(i.id in reasons for i in stale)
                self.stats["compacted_items"] += len(stale)
                self.stats["compacted_tokens"] += saved
                decision.action = "compact"
            else:
                self.stats["declined_compactions"] += 1
                decision.reason = "rewrite would cost more than it saves"
        committing = commit_p is not None and commit_p >= self.config.commit_threshold
        if (committing or overdue) and work:
            self._commit(work[-1].id, turn)
            decision.action = "compact+commit" if decision.action == "compact" else "commit"
            decision.reason = (decision.reason + "; " if decision.reason else "") + (
                "commit point" if committing else "overdue")
        decision.replacements = dict(self.replacements)
        return decision

    # -- helpers --------------------------------------------------------------- #

    def _commit(self, through: str, turn: int) -> None:
        self.committed_through, self.committed_turn = through, turn
        self.stats["commits"] += 1

    def _ask_commit(self, client: JevClient, task: str, recent: str) -> float:
        answer = client.ask({"task": task[:2000], "recent": recent}, {"commit": COMMIT_QUESTION})
        value = answer.get("commit")
        return value.noul if isinstance(value, NoulAnswer) else 0.0

    def _protected(self, item: TailItem, items: Sequence[TailItem], store: MemoryStore) -> bool:
        whole = _admitted(item, store)
        profile = (whole.meta.get("profile") if whole is not None else None) or {}
        presence = profile.get("role_presence") or {}
        if not (profile.get("role") in self.config.protected_roles
                or any(presence.get(r, 0.0) >= self.config.protected_presence
                       for r in self.config.protected_roles)):
            return False
        after = items[[i.id for i in items].index(item.id) + 1:]
        return not any(i.tool in self.config.edit_tools for i in after)

    def _card(self, item: TailItem) -> str:
        # Unlike recall's cards there is no query to centre on: show the head.
        head = f"[{item.tool} output, {item.tokens} tokens]"
        body, _ = _excerpt_head(item.text or "", self.config.card_tokens)
        return f"{head}\n{body}"

    def _relocate(self, item: TailItem, store: MemoryStore, turn: int, why: str = "") -> str:
        """Store the text the model saw; return the pointer line that replaces it."""
        text = item.text or ""
        whole = _admitted(item, store)
        profile = whole.meta.get("profile") if whole is not None else None
        summary = next((line.strip() for line in text.splitlines() if line.strip()), "")[:100]
        if profile:
            summary = f"{profile['role']}, {profile['type']}: {summary}"
        record = Record(
            id=content_id(text, salt=item.id, prefix="w"), text=text, kind="compacted_output",
            origin=Origin(source=f"tool:{item.tool}", ref=item.call_id, turn=turn),
            tokens=item.tokens, created_turn=turn, summary=summary,
            meta={"part_id": item.id, "full_output_id": whole.id if whole is not None else None,
                  **({"profile": profile, "names": whole.meta.get("names", [])} if profile else {})},
        )
        record_id = store.put(record)
        state = f"out of date ({why}); " if why else ""
        return format_pointer(Pointer(id=record_id, lines=None, tokens=item.tokens,
                                      summary=f"compacted {item.tool} output -- {state}{summary}"))


def _admitted(item: TailItem, store: MemoryStore) -> Record | None:
    """The admission gate's record of this output, profile included, if it has one."""
    if item.call_id is None:
        return None
    return next((r for r in store.all_records()
                 if r.kind == "tool_output" and r.origin.ref == item.call_id), None)


def _excerpt_head(text: str, budget: int) -> tuple[str, bool]:
    if estimate_tokens(text) <= budget:
        return text, False
    lines, out = text.splitlines(), []
    for line in lines:
        if estimate_tokens("\n".join([*out, line])) > budget:
            break
        out.append(line)
    return "\n".join(out) + "\n...", True

