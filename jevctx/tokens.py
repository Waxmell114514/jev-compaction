"""Token estimation.

FROZEN CONTRACT -- do not edit. See SPEC.md section 2.

No tokenizer dependency: ``jevctx`` must be usable without pulling in a model's
vocabulary. The estimate is deliberately *conservative* (it over-counts), because
every caller uses it to stay under a hard Jev budget. Under-counting causes a 422
from the API; over-counting only costs a little packing efficiency.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence

__all__ = ["estimate_tokens", "CJK_TOKENS_PER_CHAR", "LATIN_CHARS_PER_TOKEN"]

# A BPE vocabulary typically spends ~1.5 tokens on a CJK character and ~4 latin
# characters per token. We use 3.5 for latin to over-count by ~15%.
CJK_TOKENS_PER_CHAR = 1.5
LATIN_CHARS_PER_TOKEN = 3.5

# JSON structural overhead charged per container element (braces, quotes, commas).
_CONTAINER_OVERHEAD = 2


def _is_wide(ch: str) -> bool:
    """True for CJK/Kana/Hangul and other characters that tokenize expensively."""
    if ch < "\u0080":
        return False
    return unicodedata.east_asian_width(ch) in ("W", "F")


def _estimate_str(text: str) -> int:
    wide = 0
    for ch in text:
        if _is_wide(ch):
            wide += 1
    narrow = len(text) - wide
    return int(wide * CJK_TOKENS_PER_CHAR + narrow / LATIN_CHARS_PER_TOKEN) + 1


def estimate_tokens(obj: object) -> int:
    """Conservatively estimate the token cost of ``obj`` as Jev would receive it.

    Accepts the same shapes the API accepts for ``state`` (str / mapping / sequence)
    as well as question payloads, so callers can budget state and questions with one
    function.
    """
    if obj is None:
        return 1
    if isinstance(obj, str):
        return _estimate_str(obj)
    if isinstance(obj, bool):
        return 1
    if isinstance(obj, (int, float)):
        return _estimate_str(repr(obj))
    if isinstance(obj, Mapping):
        total = _CONTAINER_OVERHEAD
        for key, value in obj.items():
            total += estimate_tokens(key) + estimate_tokens(value) + _CONTAINER_OVERHEAD
        return total
    if isinstance(obj, Sequence):
        total = _CONTAINER_OVERHEAD
        for item in obj:
            total += estimate_tokens(item) + _CONTAINER_OVERHEAD
        return total
    return _estimate_str(str(obj))
