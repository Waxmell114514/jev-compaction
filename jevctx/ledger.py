"""Cache-cost accounting for prefix compaction (SPEC.md sections 3.5 and 3.6).

Host LLM prompt caching bills a stable prefix at roughly ``CACHE_READ_MULT`` (0.1x) of
the base input price on every turn it is merely replayed, and roughly
``CACHE_WRITE_MULT`` (1.25x) the one time it is (re)written into the cache. Compacting
a frozen prefix of ``P`` tokens down to ``P'`` tokens (``P' < P``) is therefore a
trade: one expensive rewrite now, for a cheaper read on every turn after.

Derivation of the break-even point, ``k`` turns after the rewrite:

    without compaction, k further turns cost   0.1 * P  * k              (k reads of P)
    with compaction,    the same k turns cost  1.25 * P' + 0.1 * P' * k  (1 write + k reads of P')

Setting them equal and solving for k::

    0.1*P*k = 1.25*P' + 0.1*P'*k
    k * 0.1 * (P - P') = 1.25 * P'
    k = 12.5 * P' / (P - P')                  (12.5 == CACHE_WRITE_MULT / CACHE_READ_MULT)

For P=100_000 -> P'=20_000 that is k ~= 3.125 turns: almost any compaction that
actually shrinks the prefix pays for itself within a handful of turns. That is
precisely why this design does not gate commits on cost: ``CacheLedger`` quantifies
the trade, it does not decide it. The commit policies in ``context.py`` gate on
*safety* -- is the work area actually done, is anything in it still unresolved --
because the economics essentially always say yes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from jevctx.types import CACHE_READ_MULT, CACHE_WRITE_MULT, PRICE_PER_INPUT_TOKEN

__all__ = ["CostBreakdown", "CacheLedger"]


@dataclass(frozen=True)
class CostBreakdown:
    """USD accounting for every render recorded so far, split by how it was billed."""

    cache_write_tokens: int
    cache_read_tokens: int
    uncached_tokens: int
    usd: float


@dataclass(frozen=True)
class _RenderRecord:
    """One ``record_render`` call. Internal -- ``estimated_cost()`` is the public view."""

    turn: int
    frozen_tokens: int
    work_tokens: int
    cache_written: bool


class CacheLedger:
    """Accounts for the cache read/write cost of rendering a ``ContextBuffer`` over time.

    See the module docstring for the break-even derivation. ``cache_read_mult``,
    ``cache_write_mult`` and ``price_per_input_token`` default to the constants in
    ``types.py`` but are always taken as constructor arguments, never hardcoded at a
    call site, so a different host LLM's pricing can be plugged in without editing
    this module.
    """

    def __init__(
        self,
        *,
        cache_read_mult: float = CACHE_READ_MULT,
        cache_write_mult: float = CACHE_WRITE_MULT,
        price_per_input_token: float = PRICE_PER_INPUT_TOKEN,
    ) -> None:
        self._cache_read_mult = cache_read_mult
        self._cache_write_mult = cache_write_mult
        self._price_per_input_token = price_per_input_token
        self._records: list[_RenderRecord] = []

    def record_render(
        self, turn: int, frozen_tokens: int, work_tokens: int, cache_written: bool
    ) -> None:
        """Record one render.

        ``frozen_tokens`` is billed as a cache write or a cache read depending on
        ``cache_written``. ``work_tokens`` is always uncached: the work area is, by
        construction (SPEC.md section 3), free to differ on every render, so a host
        LLM's prompt cache never covers it.
        """
        self._records.append(
            _RenderRecord(
                turn=turn,
                frozen_tokens=frozen_tokens,
                work_tokens=work_tokens,
                cache_written=cache_written,
            )
        )

    def breakeven_turns(self, p: int, p_prime: int) -> float:
        """Turns after a rewrite of ``p`` tokens down to ``p_prime`` at which the
        rewrite's one-time cost is recovered by the cheaper reads that follow.

        Returns ``math.inf`` ("never") for every degenerate or non-improving input:
        ``p <= 0`` or ``p_prime <= 0`` (no valid prefix size to reason about), and
        ``p_prime >= p`` (compaction that does not shrink the prefix -- also the only
        one of these cases that would otherwise divide by zero).
        """
        if p <= 0 or p_prime <= 0 or p_prime >= p:
            return math.inf
        return (self._cache_write_mult * p_prime) / (self._cache_read_mult * (p - p_prime))

    def would_compaction_pay(self, p: int, p_prime: int, remaining_turns: int) -> bool:
        """Whether ``remaining_turns`` more turns is enough to clear ``breakeven_turns``."""
        breakeven = self.breakeven_turns(p, p_prime)
        return math.isfinite(breakeven) and remaining_turns > breakeven

    def estimated_cost(self) -> CostBreakdown:
        cache_write_tokens = sum(r.frozen_tokens for r in self._records if r.cache_written)
        cache_read_tokens = sum(r.frozen_tokens for r in self._records if not r.cache_written)
        uncached_tokens = sum(r.work_tokens for r in self._records)
        usd = (
            cache_write_tokens * self._cache_write_mult
            + cache_read_tokens * self._cache_read_mult
            + uncached_tokens * 1.0
        ) * self._price_per_input_token
        return CostBreakdown(
            cache_write_tokens=cache_write_tokens,
            cache_read_tokens=cache_read_tokens,
            uncached_tokens=uncached_tokens,
            usd=usd,
        )
