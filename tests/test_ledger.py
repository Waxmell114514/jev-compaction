"""Tests for jevctx.ledger: cache-cost accounting and the break-even calculation."""

from __future__ import annotations

import math

import pytest

from jevctx.ledger import CacheLedger, CostBreakdown
from jevctx.types import CACHE_READ_MULT, CACHE_WRITE_MULT, PRICE_PER_INPUT_TOKEN


def test_breakeven_turns_reference_value() -> None:
    """SPEC.md section 7.10."""
    ledger = CacheLedger()
    assert ledger.breakeven_turns(100_000, 20_000) == pytest.approx(3.125)


@pytest.mark.parametrize(
    ("p", "p_prime"),
    [(100_000, 20_000), (50_000, 1_000), (8_000, 7_999), (1_000, 1)],
)
def test_would_compaction_pay_agrees_with_breakeven_on_both_sides(p: int, p_prime: int) -> None:
    ledger = CacheLedger()
    breakeven = ledger.breakeven_turns(p, p_prime)
    just_under = math.floor(breakeven)
    just_over = math.ceil(breakeven) + 1

    assert not ledger.would_compaction_pay(p, p_prime, remaining_turns=just_under)
    assert ledger.would_compaction_pay(p, p_prime, remaining_turns=just_over)


def test_p_prime_gte_p_is_inf_and_never_pays_for_any_horizon() -> None:
    ledger = CacheLedger()
    assert ledger.breakeven_turns(100, 100) == math.inf
    assert ledger.breakeven_turns(100, 150) == math.inf
    assert not ledger.would_compaction_pay(100, 100, remaining_turns=10**9)
    assert not ledger.would_compaction_pay(100, 150, remaining_turns=10**9)


@pytest.mark.parametrize(
    ("p", "p_prime"),
    [(0, 0), (0, 10), (10, 0), (-5, -1), (-5, 3)],
)
def test_degenerate_inputs_never_divide_by_zero(p: int, p_prime: int) -> None:
    ledger = CacheLedger()
    # The point of this test is that these calls must not raise ZeroDivisionError.
    result = ledger.breakeven_turns(p, p_prime)
    assert result == math.inf
    assert not ledger.would_compaction_pay(p, p_prime, remaining_turns=10**6)


def test_breakeven_uses_constructor_multipliers_not_hardcoded_defaults() -> None:
    default_ledger = CacheLedger()
    custom_ledger = CacheLedger(cache_read_mult=0.2, cache_write_mult=2.0)

    default_value = default_ledger.breakeven_turns(100_000, 20_000)
    custom_value = custom_ledger.breakeven_turns(100_000, 20_000)
    expected_custom = (2.0 * 20_000) / (0.2 * (100_000 - 20_000))

    assert custom_value == pytest.approx(expected_custom)
    assert custom_value != pytest.approx(default_value)


def test_estimated_cost_accounting_across_scripted_renders() -> None:
    ledger = CacheLedger()
    ledger.record_render(turn=1, frozen_tokens=1000, work_tokens=200, cache_written=True)
    ledger.record_render(turn=2, frozen_tokens=1000, work_tokens=150, cache_written=False)
    ledger.record_render(turn=3, frozen_tokens=1000, work_tokens=300, cache_written=False)
    # A compaction: the frozen prefix shrinks from 1000 to 400 and gets a fresh write.
    ledger.record_render(turn=4, frozen_tokens=400, work_tokens=50, cache_written=True)

    cost = ledger.estimated_cost()

    assert isinstance(cost, CostBreakdown)
    assert cost.cache_write_tokens == 1000 + 400
    assert cost.cache_read_tokens == 1000 + 1000
    assert cost.uncached_tokens == 200 + 150 + 300 + 50

    expected_usd = (
        cost.cache_write_tokens * CACHE_WRITE_MULT
        + cost.cache_read_tokens * CACHE_READ_MULT
        + cost.uncached_tokens * 1.0
    ) * PRICE_PER_INPUT_TOKEN
    assert cost.usd == pytest.approx(expected_usd)


def test_estimated_cost_with_no_renders_is_all_zero() -> None:
    ledger = CacheLedger()
    cost = ledger.estimated_cost()
    assert cost == CostBreakdown(
        cache_write_tokens=0, cache_read_tokens=0, uncached_tokens=0, usd=0.0
    )


def test_estimated_cost_uses_custom_price_per_input_token() -> None:
    ledger = CacheLedger(price_per_input_token=1.0)
    ledger.record_render(turn=1, frozen_tokens=100, work_tokens=0, cache_written=True)
    cost = ledger.estimated_cost()
    assert cost.usd == pytest.approx(100 * CACHE_WRITE_MULT * 1.0)
