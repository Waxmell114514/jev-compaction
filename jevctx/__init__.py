"""jevctx -- cache-preserving agent context compaction and memory, gated by Jev.

See SPEC.md. The short version: context is ``[frozen prefix] + [work area]``, the
prefix is append-only so the KV cache over it is never invalidated, and content the
gate removes is relocated to a store behind an expandable pointer rather than
deleted.
"""

from jevctx.budget import Batch, BudgetPlanner
from jevctx.check import run_check
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
from jevctx.jev import HttpJevClient, RateLimiter, RetryPolicy
from jevctx.ledger import CacheLedger, CostBreakdown
from jevctx.pipeline import (
    ADMIT_QUESTION,
    DEFAULT_GATE_CONFIG,
    EXPAND_TOOL_SCHEMA,
    RETRIEVE_QUESTION,
    AdmitResult,
    GateConfig,
    admit,
    expand,
    find_pointers,
    format_pointer,
    mark_hits,
    parse_pointer,
    reconstruct,
    retrieve,
)
from jevctx.scorer import build_state, score_items, score_map
from jevctx.segments import detect_kind, segment
from jevctx.shadow import ShadowLog, ShadowStats
from jevctx.store import InMemoryStore, JsonlStore
from jevctx.testing import FakeJevClient
from jevctx.tokens import estimate_tokens
from jevctx.types import (
    Block,
    Choice,
    CommitPolicy,
    DigestEntry,
    JevClient,
    JevError,
    MemoryStore,
    Noul,
    Origin,
    Pointer,
    Record,
    Score,
    ScoreItem,
    ScoreResult,
    Segment,
    TurnSignals,
)

__version__ = "0.1.0"

__all__ = [
    "ADMIT_QUESTION", "DEFAULT_COMMIT_POLICY", "DEFAULT_GATE_CONFIG",
    "EXPAND_TOOL_SCHEMA", "RETRIEVE_QUESTION",
    "AdmitResult", "AllOf", "AnyOf", "Batch", "Block", "BudgetPlanner", "CacheLedger",
    "Choice", "CommitPolicy", "ContextBuffer", "CostBreakdown", "DigestEntry",
    "FakeJevClient", "GateConfig", "HttpJevClient", "InMemoryStore", "JevClient",
    "JevError", "JsonlStore", "MemoryStore", "NeverCommit", "Noul", "Origin",
    "Pointer", "RateLimiter", "Record", "RetryPolicy", "Score", "ScoreItem",
    "ScoreResult", "Segment", "ShadowLog", "ShadowStats", "ToolDepthZero",
    "TurnCount", "TurnSignals", "WorkAreaTokens",
    "admit", "build_state", "run_check", "detect_kind", "estimate_tokens", "expand",
    "find_pointers", "format_pointer", "make_block", "mark_hits", "parse_pointer",
    "reconstruct", "retrieve", "score_items", "score_map", "segment",
]
