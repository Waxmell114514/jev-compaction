"""Actual usage accounting must not confuse cache reads with ordinary input."""

import pytest

from jevctx.usage import ModelUsage, Prices


def test_mixed_cache_usage_is_billed_once_per_category():
    usage = ModelUsage()
    usage.record(input_tokens=100, output_tokens=20, cache_read_tokens=800,
                 cache_write_tokens=200)
    usage.record(input_tokens=50, output_tokens=10, cache_read_tokens=1000)
    assert usage.requests == 2
    assert usage.cost(Prices(input=3, output=15, cache_read=0.3, cache_write=3.75)) \
        == pytest.approx((150 * 3 + 30 * 15 + 1800 * 0.3 + 200 * 3.75) / 1_000_000)


def test_missing_prices_or_usage_never_looks_free():
    usage = ModelUsage()
    assert usage.cost(None) is None
    usage.missing_usage = 1
    assert usage.cost(Prices(0, 0, 0, 0)) is None


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_reject_invalid_prices(value):
    with pytest.raises(ValueError):
        Prices(value, 0, 0, 0)


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_reject_invalid_counts_without_partial_update(value):
    usage = ModelUsage()
    with pytest.raises(ValueError):
        usage.record(input_tokens=10, output_tokens=value)
    assert usage == ModelUsage()
