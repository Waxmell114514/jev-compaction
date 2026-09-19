"""The frozen-prefix / work-area context buffer (SPEC.md section 3).

An agent's context is ``[frozen prefix] + [work area]``. The frozen prefix is
append-only, and that is the entire trick: as long as the bytes of every message up
to some index never change between one render and the next, a host LLM's prompt
cache over that prefix stays warm. Everything still being iterated on -- a tool
result the agent hasn't finished with, a draft it may revise, a question it hasn't
resolved -- lives in the work area instead, where it is free to be replaced or
dropped because nothing downstream depends on its bytes staying put.

``ContextBuffer`` enforces the one invariant that makes this useful (SPEC.md section
3.3) at *runtime*, not just by construction: it remembers the messages it rendered
for the frozen prefix last time, and both ``render()`` and ``commit()`` re-check that
memo before returning. A cache-invalidating bug -- a bad slice, code that reaches into
``buffer.frozen`` and edits it directly, a future refactor that gets an index wrong --
fails loudly with ``FrozenPrefixError`` instead of silently doubling the cost of every
turn that follows. That failure mode is exactly what this package exists to prevent,
so the check always runs; it is not a debug-only assertion.

Commit *policies* (``ToolDepthZero``, ``WorkAreaTokens``, ``TurnCount``, and the
``AnyOf`` / ``AllOf`` / ``NeverCommit`` combinators) decide *when* it is safe to fold
work-area blocks into the frozen prefix. Each is a pure function of
``(buffer, signals) -> CommitDecision`` satisfying the ``CommitPolicy`` protocol from
``types.py``. Nothing in this module inspects or imports a concrete policy class --
``ContextBuffer`` itself never even calls one -- so Phase 2's ``JevCommitPolicy``, or a
one-off policy object a test builds on the spot, composes with ``AnyOf`` / ``AllOf``
and drops in with zero changes here.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from jevctx.tokens import estimate_tokens
from jevctx.types import (
    Block,
    BufferStats,
    CommitDecision,
    CommitPolicy,
    CommitResult,
    FrozenPrefixError,
    RenderedMessage,
    Segment,
    TurnSignals,
    content_id,
)

__all__ = [
    "ContextBuffer",
    "make_block",
    "ToolDepthZero",
    "WorkAreaTokens",
    "TurnCount",
    "AnyOf",
    "AllOf",
    "NeverCommit",
    "DEFAULT_COMMIT_POLICY",
]


class ContextBuffer:
    """``[frozen prefix] + [work area]``. See the module docstring for the model.

    ``frozen`` and ``work`` are plain public lists, matching SPEC.md section 3.1
    exactly, so callers can inspect them freely. They are not wrapped in an immutable
    type because Python cannot make a list element truly unwritable without diverging
    from that spec'd shape -- instead, tampering is *detected*: every ``render()`` and
    ``commit()`` recomputes the frozen prefix's rendered messages and compares them
    against the last-observed copy, raising ``FrozenPrefixError`` on any mismatch.
    """

    def __init__(
        self,
        frozen: Sequence[Block] | None = None,
        work: Sequence[Block] | None = None,
    ) -> None:
        self.frozen: list[Block] = list(frozen) if frozen else []
        self.work: list[Block] = list(work) if work else []
        self._frozen_memo: tuple[RenderedMessage, ...] | None = None
        if self.frozen:
            # Seed the memo immediately so a buffer constructed with pre-existing
            # frozen history is protected even before the first render() call.
            self._frozen_memo = self._render_frozen()

    # -- operations table (SPEC.md section 3.2) ------------------------------ #

    def append_work(self, block: Block) -> None:
        """Always allowed: the work area is freely mutable."""
        self.work.append(block)

    def replace_work(self, blocks: Sequence[Block]) -> None:
        """Replace the entire work area. No cache impact -- ``frozen`` is untouched."""
        self.work = list(blocks)

    def drop_work(self, block_ids: Collection[str]) -> None:
        """Remove blocks from the work area only; ids of frozen blocks are ignored."""
        ids = set(block_ids)
        self.work = [b for b in self.work if b.id not in ids]

    def commit(self, upto: int | None = None, *, reason: str) -> CommitResult:
        """Move the first ``upto`` work blocks (all, if ``None``) into ``frozen``.

        Raises ``ValueError`` if ``upto`` is out of range for the current work area,
        and ``FrozenPrefixError`` if -- despite ``frozen`` only ever being appended to
        here -- the recomputed prefix does not match what was last observed (see the
        module docstring). The check runs, and the buffer is left unmodified, before
        either list is mutated.
        """
        k = len(self.work) if upto is None else upto
        if k < 0 or k > len(self.work):
            raise ValueError(
                f"commit(upto={upto!r}) is out of range for a work area of "
                f"{len(self.work)} block(s)"
            )
        moved = self.work[:k]
        candidate_frozen = tuple(self.frozen) + tuple(moved)
        candidate_messages = self._render_frozen(candidate_frozen)
        self._assert_frozen_unchanged(candidate_messages)
        self.frozen = list(candidate_frozen)
        self.work = self.work[k:]
        return CommitResult(
            committed_blocks=k,
            committed_tokens=sum(b.tokens for b in moved),
            reason=reason,
            cache_breakpoint=len(self.frozen),
        )

    def render(self) -> list[RenderedMessage]:
        """Materialise the buffer. ``result[:cache_breakpoint]`` is the frozen prefix."""
        frozen_messages = self._render_frozen()
        self._assert_frozen_unchanged(frozen_messages)
        work_messages = [
            RenderedMessage(role=b.role, content=b.text, cacheable=False) for b in self.work
        ]
        return list(frozen_messages) + work_messages

    @property
    def cache_breakpoint(self) -> int:
        """Index of the first work-area message in ``render()``."""
        return len(self.frozen)

    def stats(self) -> BufferStats:
        return BufferStats(
            frozen_tokens=sum(b.tokens for b in self.frozen),
            work_tokens=sum(b.tokens for b in self.work),
            frozen_blocks=len(self.frozen),
            work_blocks=len(self.work),
            cache_breakpoint=self.cache_breakpoint,
        )

    # -- the invariant (SPEC.md section 3.3) ---------------------------------- #

    def _render_frozen(
        self, blocks: Sequence[Block] | None = None
    ) -> tuple[RenderedMessage, ...]:
        source = self.frozen if blocks is None else blocks
        return tuple(RenderedMessage(role=b.role, content=b.text, cacheable=True) for b in source)

    def _assert_frozen_unchanged(self, frozen_messages: tuple[RenderedMessage, ...]) -> None:
        memo = self._frozen_memo
        if memo is not None and frozen_messages[: len(memo)] != memo:
            raise FrozenPrefixError(
                "the frozen prefix changed between renders -- this would invalidate "
                "the host LLM's KV cache and must never happen (SPEC.md section 3.3)"
            )
        self._frozen_memo = frozen_messages


def make_block(
    role: str,
    text: str,
    *,
    segments: Sequence[Segment] = (),
    meta: Mapping[str, Any] | None = None,
    turn: int | None = None,
) -> Block:
    """Build a ``Block``, computing its id and token count so callers never hand-build one.

    The id is content-addressed (``types.content_id``, salted with ``role``, prefix
    ``"b"``): the same role and text always produce the same id, mirroring how
    ``Segment.id`` is derived elsewhere in this package. ``turn``, if given, is stored
    as ``committed_at_turn`` -- it names the turn the block was *authored* on.
    ``ContextBuffer.commit()`` has no ``turn`` parameter of its own (SPEC.md section
    3.2 gives its exact signature), so it never overwrites this value; a caller that
    wants ``committed_at_turn`` to reflect the turn a block actually froze on should
    set ``turn=`` here right before calling ``commit()``.
    """
    return Block(
        id=content_id(text, salt=role, prefix="b"),
        role=role,
        text=text,
        tokens=estimate_tokens(text),
        segments=tuple(segments),
        committed_at_turn=turn,
        meta=dict(meta) if meta is not None else {},
    )


# --------------------------------------------------------------------------- #
# Commit policies (SPEC.md section 3.4)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ToolDepthZero:
    """Commit once control has fully returned from tool use.

    ``tool_depth == 0`` means nothing is mid-tool-call; ``min_blocks`` keeps a single
    reply right after startup from triggering a commit of almost nothing.
    """

    min_blocks: int = 2

    def should_commit(self, buffer: ContextBuffer, signals: TurnSignals) -> CommitDecision:
        work_blocks = len(buffer.work)
        if signals.tool_depth == 0 and work_blocks >= self.min_blocks:
            return CommitDecision(
                commit=True,
                reason=f"tool_depth is 0 and work area has {work_blocks} "
                f">= min_blocks {self.min_blocks}",
            )
        return CommitDecision(
            commit=False,
            reason=f"tool_depth={signals.tool_depth} or work area ({work_blocks} blocks) "
            f"below min_blocks {self.min_blocks}",
        )


@dataclass(frozen=True)
class WorkAreaTokens:
    """Commit once the work area itself is expensive to keep re-sending uncached."""

    threshold: int = 8000

    def should_commit(self, buffer: ContextBuffer, signals: TurnSignals) -> CommitDecision:
        work_tokens = buffer.stats().work_tokens
        if work_tokens >= self.threshold:
            return CommitDecision(
                commit=True, reason=f"work_tokens {work_tokens} >= threshold {self.threshold}"
            )
        return CommitDecision(
            commit=False, reason=f"work_tokens {work_tokens} < threshold {self.threshold}"
        )


@dataclass(frozen=True)
class TurnCount:
    """Commit every ``n`` turns, so a slow-growing work area is folded in eventually.

    Fires when ``signals.turn`` is a positive multiple of ``n`` (turns n, 2n, 3n, ...).
    """

    n: int = 8

    def should_commit(self, buffer: ContextBuffer, signals: TurnSignals) -> CommitDecision:
        if self.n > 0 and signals.turn > 0 and signals.turn % self.n == 0:
            return CommitDecision(
                commit=True, reason=f"turn {signals.turn} is a multiple of {self.n}"
            )
        return CommitDecision(
            commit=False, reason=f"turn {signals.turn} is not a positive multiple of {self.n}"
        )


class AnyOf:
    """Commit as soon as any one of ``policies`` votes to (first match wins, in order)."""

    def __init__(self, *policies: CommitPolicy) -> None:
        self.policies: tuple[CommitPolicy, ...] = tuple(policies)

    def should_commit(self, buffer: ContextBuffer, signals: TurnSignals) -> CommitDecision:
        for policy in self.policies:
            decision = policy.should_commit(buffer, signals)
            if decision.commit:
                return decision
        return CommitDecision(commit=False, reason="no policy in AnyOf voted to commit")


class AllOf:
    """Commit only once every policy in ``policies`` agrees.

    The combined ``upto`` is the smallest ``upto`` actually offered by a sub-decision,
    or ``None`` (commit everything) if every sub-decision left it unset.
    """

    def __init__(self, *policies: CommitPolicy) -> None:
        self.policies: tuple[CommitPolicy, ...] = tuple(policies)

    def should_commit(self, buffer: ContextBuffer, signals: TurnSignals) -> CommitDecision:
        decisions = [policy.should_commit(buffer, signals) for policy in self.policies]
        if not decisions or not all(d.commit for d in decisions):
            return CommitDecision(commit=False, reason="not all policies in AllOf voted to commit")
        uptos = [d.upto for d in decisions if d.upto is not None]
        return CommitDecision(
            commit=True,
            reason="; ".join(d.reason for d in decisions),
            upto=min(uptos) if uptos else None,
        )


class NeverCommit:
    """A policy that never fires."""

    def should_commit(self, buffer: ContextBuffer, signals: TurnSignals) -> CommitDecision:
        return CommitDecision(commit=False, reason="NeverCommit always declines")


DEFAULT_COMMIT_POLICY: CommitPolicy = AnyOf(WorkAreaTokens(8000), ToolDepthZero(min_blocks=4))
