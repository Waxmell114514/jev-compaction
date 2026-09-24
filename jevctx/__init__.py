"""jevctx -- cache-preserving agent context compaction and memory, gated by Jev.

The short version: context is ``[frozen prefix] + [work area]``, the
prefix is append-only so the KV cache over it is never invalidated, and content the
gate removes is relocated to a store behind an expandable pointer rather than
deleted.
"""

from jevctx.budget import Batch, BudgetPlanner
from jevctx.calibrate import fit_thresholds
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
from jevctx.jev import HttpJevClient, JevEndpoint, RateLimiter, RetryPolicy, resolve_endpoint
from jevctx.label import (
    ENTITY_QUESTIONS,
    LIFETIME_QUESTION,
    TYPE_QUESTION,
    ItemLabel,
    apply_label,
    label_items,
    label_records,
)
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
from jevctx.profile import INJECTION_QUESTION, ROLE_QUESTION, Profile, profile_items
from jevctx.recall import RecallHit, recall, render_hits
from jevctx.scorer import build_state, score_items, score_map
from jevctx.segments import detect_kind, segment
from jevctx.shadow import ShadowLog, ShadowStats
from jevctx.store import InMemoryStore, JsonlStore
from jevctx.supersede import Relation, SupersessionIndex, footprints, note_for
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
from jevctx.workarea import TailItem, WorkArea, WorkAreaConfig, compaction_pays

__version__ = "0.2.0"

__all__ = [
    "ADMIT_QUESTION", "DEFAULT_COMMIT_POLICY", "DEFAULT_GATE_CONFIG", "INJECTION_QUESTION",
    "ROLE_QUESTION", "Profile", "RecallHit", "Relation", "SupersessionIndex", "TailItem",
    "WorkArea", "WorkAreaConfig", "compaction_pays", "fit_thresholds", "footprints",
    "note_for", "profile_items", "recall", "render_hits",
    "ENTITY_QUESTIONS", "EXPAND_TOOL_SCHEMA", "LIFETIME_QUESTION",
    "RETRIEVE_QUESTION", "TYPE_QUESTION",
    "AdmitResult", "AllOf", "AnyOf", "Batch", "Block", "BudgetPlanner", "CacheLedger",
    "Choice", "CommitPolicy", "ContextBuffer", "CostBreakdown", "DigestEntry",
    "FakeJevClient", "GateConfig", "HttpJevClient", "InMemoryStore", "ItemLabel", "JevEndpoint",
    "JevClient", "JevError", "JsonlStore", "MemoryStore", "NeverCommit", "Noul",
    "Origin", "Pointer", "RateLimiter", "Record", "RetryPolicy", "Score", "ScoreItem",
    "ScoreResult", "Segment", "ShadowLog", "ShadowStats", "ToolDepthZero",
    "TurnCount", "TurnSignals", "WorkAreaTokens",
    "admit", "apply_label", "build_state", "run_check", "detect_kind", "estimate_tokens",
    "expand", "find_pointers", "format_pointer", "label_items", "label_records",
    "make_block", "mark_hits", "parse_pointer", "reconstruct", "resolve_endpoint", "retrieve",
    "score_items", "score_map", "segment",
]
