"""Measured model usage; prices are explicit USD per million tokens."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Prices:
    input: float
    output: float
    cache_read: float
    cache_write: float

    def __post_init__(self) -> None:
        for value in (self.input, self.output, self.cache_read, self.cache_write):
            if not math.isfinite(value) or value < 0:
                raise ValueError("prices must be finite and non-negative")


@dataclass
class ModelUsage:
    """Disjoint token categories, accumulated from provider responses.

    Missing usage makes the total cost unknown, not zero. Input excludes cache
    reads and writes; adapters must normalize providers that report them inside
    their input total. This is separate from CacheLedger's simulated renders.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    requests: int = 0
    missing_usage: int = 0

    def record(
        self, *, input_tokens: int, output_tokens: int,
        cache_read_tokens: int = 0, cache_write_tokens: int = 0,
    ) -> None:
        values = (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("token counts must be non-negative integers")
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_read_tokens += cache_read_tokens
        self.cache_write_tokens += cache_write_tokens
        self.requests += 1

    def cost(self, prices: Prices | None) -> float | None:
        if prices is None or self.missing_usage:
            return None
        return (
            self.input_tokens * prices.input
            + self.output_tokens * prices.output
            + self.cache_read_tokens * prices.cache_read
            + self.cache_write_tokens * prices.cache_write
        ) / 1_000_000
