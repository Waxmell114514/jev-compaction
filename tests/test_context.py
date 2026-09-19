"""Tests for jevctx.context: the frozen-prefix invariant and the commit policies.

The property test (test_frozen_prefix_property) is the most important test in the
package: SPEC.md section 3.3 is the entire point of ContextBuffer, and this is the
only test that actually drives an adversarial, randomised sequence against it.
"""

from __future__ import annotations

import random

import pytest

from jevctx.context import (
    DEFAULT_COMMIT_POLICY,
    AllOf,
    AnyOf,
    ContextBuffer,
    NeverCommit,
    ToolDepthZero,
    TurnCount,
    WorkAreaTokens,
    make_block,
)
from jevctx.tokens import estimate_tokens
from jevctx.types import Block, CommitDecision, CommitPolicy, FrozenPrefixError, TurnSignals

# --------------------------------------------------------------------------- #
# The property test (SPEC.md section 7.1)
# --------------------------------------------------------------------------- #

_ROLES = ("user", "assistant", "tool")
_OPS = (
    "append",
    "append",
    "append",
    "replace",
    "drop",
    "commit_all",
    "commit_partial",
    "commit_zero",
)


def _random_block(rng: random.Random, step: int, tag: str) -> Block:
    role = rng.choice(_ROLES)
    # step + tag + a random nonce keeps generated text unique enough that this test's
    # own logic never has to reason about two distinct blocks sharing a content-id.
    text = f"step={step} tag={tag} nonce={rng.randrange(10**12)}"
    return make_block(role, text, turn=step)


def _run_property_sequence(seed: int, steps: int) -> None:
    rng = random.Random(seed)
    buf = ContextBuffer()
    observed_prefix: list = []

    for step in range(steps):
        op = rng.choice(_OPS)

        if op == "append":
            buf.append_work(_random_block(rng, step, "append"))
        elif op == "replace":
            n = rng.randint(0, 5)
            buf.replace_work([_random_block(rng, step, f"replace{i}") for i in range(n)])
        elif op == "drop":
            if buf.work:
                k = rng.randint(1, len(buf.work))
                ids = {b.id for b in rng.sample(buf.work, k)}
                buf.drop_work(ids)
        elif op == "commit_zero":
            result = buf.commit(upto=0, reason="empty commit")
            assert result.committed_blocks == 0
            assert result.committed_tokens == 0
        elif op == "commit_partial":
            if buf.work:
                k = rng.randint(1, len(buf.work))
                before = len(buf.work)
                result = buf.commit(upto=k, reason=f"partial commit of {k}")
                assert result.committed_blocks == k
                assert len(buf.work) == before - k
        elif op == "commit_all":
            buf.commit(reason="commit everything")
            assert len(buf.work) == 0
        else:  # pragma: no cover - _OPS is exhaustive
            raise AssertionError(f"unhandled op {op!r}")

        # The invariant (SPEC.md 3.3): the previously-observed frozen prefix must be
        # a prefix of the new render, byte-identical, no matter which op just ran.
        rendered = buf.render()
        assert rendered[: len(observed_prefix)] == observed_prefix
        observed_prefix = rendered[: buf.cache_breakpoint]


@pytest.mark.parametrize("seed", [0, 1, 2, 17, 12345])
def test_frozen_prefix_property(seed: int) -> None:
    _run_property_sequence(seed, steps=250)


def test_replace_then_commit_preserves_prior_frozen_prefix() -> None:
    buf = ContextBuffer()
    buf.append_work(make_block("user", "first"))
    buf.commit(reason="freeze first")
    prefix_before = buf.render()[: buf.cache_breakpoint]

    buf.replace_work([make_block("assistant", "second"), make_block("tool", "third")])
    buf.commit(reason="freeze after replace")

    rendered = buf.render()
    assert rendered[: len(prefix_before)] == prefix_before
    assert buf.cache_breakpoint == 3


def test_drop_everything_then_commit_is_a_no_op_on_frozen() -> None:
    buf = ContextBuffer()
    buf.append_work(make_block("user", "keep-me"))
    buf.commit(reason="freeze")
    prefix_before = buf.render()[: buf.cache_breakpoint]

    for i in range(5):
        buf.append_work(make_block("tool", f"scratch-{i}"))
    buf.drop_work({b.id for b in buf.work})
    assert buf.work == []

    result = buf.commit(reason="commit nothing, work area is empty")
    assert result.committed_blocks == 0
    rendered = buf.render()
    assert rendered[: len(prefix_before)] == prefix_before


# --------------------------------------------------------------------------- #
# Operations table (SPEC.md section 3.2)
# --------------------------------------------------------------------------- #


def test_commit_upto_moves_exactly_k_blocks() -> None:
    buf = ContextBuffer()
    for i in range(6):
        buf.append_work(make_block("assistant", f"block-{i}"))

    result = buf.commit(upto=4, reason="partial")

    assert result.committed_blocks == 4
    assert len(buf.frozen) == 4
    assert len(buf.work) == 2
    assert [b.text for b in buf.work] == ["block-4", "block-5"]


def test_commit_upto_out_of_range_raises() -> None:
    buf = ContextBuffer()
    buf.append_work(make_block("user", "only one"))
    with pytest.raises(ValueError):
        buf.commit(upto=2, reason="too many")
    with pytest.raises(ValueError):
        buf.commit(upto=-1, reason="negative")


def test_work_area_mutation_never_changes_rendered_frozen_prefix() -> None:
    buf = ContextBuffer()
    buf.append_work(make_block("user", "hello"))
    buf.append_work(make_block("assistant", "world"))
    buf.commit(reason="freeze both")
    frozen_part = buf.render()[: buf.cache_breakpoint]
    breakpoint_before = buf.cache_breakpoint

    buf.append_work(make_block("tool", "scratch-1"))
    buf.append_work(make_block("tool", "scratch-2"))
    buf.replace_work([make_block("tool", "scratch-3")])
    buf.drop_work({buf.work[0].id})
    buf.append_work(make_block("tool", "scratch-4"))

    rendered = buf.render()
    assert rendered[:breakpoint_before] == frozen_part
    assert buf.cache_breakpoint == breakpoint_before


def test_direct_mutation_of_frozen_list_raises_frozen_prefix_error() -> None:
    buf = ContextBuffer()
    buf.append_work(make_block("user", "hello"))
    buf.commit(reason="freeze")
    buf.render()  # establishes the memo

    # Reach past the public API and tamper with an already-frozen block directly.
    buf.frozen[0] = make_block("user", "TAMPERED")

    with pytest.raises(FrozenPrefixError):
        buf.render()


def test_commit_that_would_alter_frozen_prefix_raises_before_mutating() -> None:
    buf = ContextBuffer()
    buf.append_work(make_block("user", "hello"))
    buf.commit(reason="freeze")
    buf.render()

    buf.frozen[0] = make_block("user", "TAMPERED")
    buf.append_work(make_block("user", "more"))

    with pytest.raises(FrozenPrefixError):
        buf.commit(reason="should not succeed")

    # The failed commit must not have partially applied: "more" was never moved.
    assert [b.text for b in buf.work] == ["more"]


def test_stats_token_counts_agree_with_estimate_tokens_over_rendered_content() -> None:
    buf = ContextBuffer()
    buf.append_work(make_block("user", "a" * 500))
    buf.append_work(make_block("assistant", "b" * 300))
    buf.commit(upto=1, reason="freeze one")
    buf.append_work(make_block("tool", "c" * 700))

    rendered = buf.render()
    stats = buf.stats()
    frozen_part = rendered[: stats.cache_breakpoint]
    work_part = rendered[stats.cache_breakpoint :]

    assert stats.frozen_tokens == sum(estimate_tokens(m.content) for m in frozen_part)
    assert stats.work_tokens == sum(estimate_tokens(m.content) for m in work_part)
    assert stats.frozen_blocks == len(buf.frozen) == 1
    assert stats.work_blocks == len(buf.work) == 2
    assert stats.cache_breakpoint == buf.cache_breakpoint


# --------------------------------------------------------------------------- #
# Commit policies (SPEC.md section 3.4)
# --------------------------------------------------------------------------- #


def test_tool_depth_zero_fires_and_declines() -> None:
    policy = ToolDepthZero(min_blocks=2)
    buf = ContextBuffer()
    buf.append_work(make_block("assistant", "one"))
    buf.append_work(make_block("assistant", "two"))

    assert policy.should_commit(buf, TurnSignals(turn=1, tool_depth=0)).commit
    assert not policy.should_commit(buf, TurnSignals(turn=1, tool_depth=2)).commit

    stricter = ToolDepthZero(min_blocks=5)
    assert not stricter.should_commit(buf, TurnSignals(turn=1, tool_depth=0)).commit


def test_work_area_tokens_fires_and_declines() -> None:
    policy = WorkAreaTokens(threshold=10)
    buf = ContextBuffer()
    buf.append_work(make_block("assistant", "x"))
    assert not policy.should_commit(buf, TurnSignals(turn=1)).commit

    buf.append_work(make_block("assistant", "y" * 200))
    assert policy.should_commit(buf, TurnSignals(turn=1)).commit


def test_turn_count_fires_on_positive_multiples_only() -> None:
    policy = TurnCount(n=8)
    buf = ContextBuffer()

    assert not policy.should_commit(buf, TurnSignals(turn=0)).commit
    assert not policy.should_commit(buf, TurnSignals(turn=7)).commit
    assert policy.should_commit(buf, TurnSignals(turn=8)).commit
    assert policy.should_commit(buf, TurnSignals(turn=16)).commit
    assert not policy.should_commit(buf, TurnSignals(turn=17)).commit


def test_never_commit_always_declines() -> None:
    buf = ContextBuffer()
    assert not NeverCommit().should_commit(buf, TurnSignals(turn=100)).commit


def test_any_of_and_all_of_compose() -> None:
    buf = ContextBuffer()
    buf.append_work(make_block("assistant", "one"))

    never = NeverCommit()
    always = TurnCount(n=1)  # every turn >= 1 is a multiple of 1

    assert AnyOf(never, always).should_commit(buf, TurnSignals(turn=1)).commit
    assert not AllOf(never, always).should_commit(buf, TurnSignals(turn=1)).commit
    assert AllOf(always, TurnCount(n=1)).should_commit(buf, TurnSignals(turn=2)).commit


def test_all_of_combines_upto_as_the_minimum_offered() -> None:
    class _FixedUpto:
        def __init__(self, upto: int) -> None:
            self._upto = upto

        def should_commit(self, buffer: ContextBuffer, signals: TurnSignals) -> CommitDecision:
            return CommitDecision(commit=True, reason=f"fixed upto={self._upto}", upto=self._upto)

    buf = ContextBuffer()
    decision = AllOf(_FixedUpto(5), _FixedUpto(2), _FixedUpto(9)).should_commit(
        buf, TurnSignals(turn=1)
    )
    assert decision.commit
    assert decision.upto == 2


def test_default_commit_policy_matches_spec_default() -> None:
    buf = ContextBuffer()
    for i in range(3):
        buf.append_work(make_block("assistant", f"filler-{i}"))

    # Small work area, mid-tool-call: neither WorkAreaTokens(8000) nor
    # ToolDepthZero(min_blocks=4) fires.
    assert not DEFAULT_COMMIT_POLICY.should_commit(buf, TurnSignals(turn=1, tool_depth=1)).commit

    # A 4th block plus tool_depth back to 0 satisfies ToolDepthZero(min_blocks=4).
    buf.append_work(make_block("assistant", "filler-3"))
    assert DEFAULT_COMMIT_POLICY.should_commit(buf, TurnSignals(turn=1, tool_depth=0)).commit


class _CustomPhase2StylePolicy:
    """A policy jevctx.context has never heard of -- stands in for JevCommitPolicy.

    Deliberately does not inherit from anything in context.py or types.py: the point
    of CommitPolicy being a ``Protocol`` is that structural typing is enough.
    """

    def should_commit(self, buffer: ContextBuffer, signals: TurnSignals) -> CommitDecision:
        return CommitDecision(commit=True, reason="custom Phase-2-style policy", upto=1)


def test_foreign_policy_object_satisfies_protocol_and_composes_unchanged() -> None:
    custom = _CustomPhase2StylePolicy()
    assert isinstance(custom, CommitPolicy)  # structural check, no inheritance needed

    buf = ContextBuffer()
    buf.append_work(make_block("assistant", "one"))
    buf.append_work(make_block("assistant", "two"))

    decision = AnyOf(NeverCommit(), custom).should_commit(buf, TurnSignals(turn=1))
    assert decision.commit
    assert decision.upto == 1

    result = buf.commit(upto=decision.upto, reason=decision.reason)
    assert result.committed_blocks == 1
    assert len(buf.frozen) == 1
    assert len(buf.work) == 1
