"""Shared contracts: constants, data model, protocols, exceptions.

One of the shared contracts the rest of the package is written against.

If something here is wrong or missing, report it rather than editing it: every
module in the package is written against this file in parallel.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    # limits
    "MAX_QUESTIONS_PER_REQUEST", "MAX_CHOICE_OPTIONS", "MIN_SCORE_LEVELS",
    "MAX_SCORE_LEVELS", "STATE_PLUS_ALL_QUESTIONS_TOKENS",
    "STATE_PLUS_LONGEST_QUESTION_TOKENS", "RATE_LIMIT_RPM",
    "PRICE_PER_INPUT_TOKEN", "CACHE_READ_MULT", "CACHE_WRITE_MULT",
    # questions / answers
    "Noul", "Choice", "Score", "Question",
    "NoulAnswer", "ChoiceAnswer", "ScoreAnswer", "Answer", "parse_answer",
    # data model
    "Origin", "Segment", "SegmentKind", "Lifecycle", "ScoreItem", "ScoreResult",
    "Record", "DigestEntry", "Pointer", "Block", "RenderedMessage", "BufferStats",
    "CommitDecision", "CommitResult", "TurnSignals",
    # protocols
    "JevClient", "MemoryStore", "CommitPolicy",
    # errors
    "JevError", "JevAuthError", "JevValidationError", "JevRejectedError", "JevBudgetError",
    "JevUnavailableError", "FrozenPrefixError",
    # helpers
    "content_id",
]

# --------------------------------------------------------------------------- #
# Hard limits of the Jev API -- these are not tunables.
# --------------------------------------------------------------------------- #

MAX_QUESTIONS_PER_REQUEST = 32
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
STATE_PLUS_ALL_QUESTIONS_TOKENS = 64_000
STATE_PLUS_LONGEST_QUESTION_TOKENS = 32_000
RATE_LIMIT_RPM = 1200

#: USD per input token. Jev output tokens are free.
PRICE_PER_INPUT_TOKEN = 0.042 / 1_000_000

#: Prompt-cache multipliers of the *host LLM* (not Jev), used by CacheLedger.
CACHE_READ_MULT = 0.1
CACHE_WRITE_MULT = 1.25

SegmentKind = Literal["text", "json", "log", "stacktrace", "table", "code", "diff"]
Lifecycle = Literal["turn", "task", "session", "permanent"]

#: A Jev ``state``: plain text, or structured data.
State = str | Mapping[str, Any] | Sequence[Any]


def content_id(text: str, salt: str = "", prefix: str = "s") -> str:
    """Stable short id for a piece of content. Same input -> same id, always."""
    digest = hashlib.sha256((salt + "\x00" + text).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest[:8]}"


# --------------------------------------------------------------------------- #
# Questions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Noul:
    """A yes/no question. The answer is a probability in [0, 1]."""

    instructions: str
    true: str
    false: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "noul",
            "instructions": self.instructions,
            "criteria": {"true": self.true, "false": self.false},
        }


@dataclass(frozen=True)
class Choice:
    """Pick one of up to 255 options. Probabilities are normalised and sum to 1.

    Because the distribution is normalised, Choice answers a *ranking* question
    ("which one"), not an independent gating question ("should each be kept").
    Use Noul for the latter.
    """

    instructions: str
    criteria: Mapping[str, str | None]

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }


@dataclass(frozen=True)
class Score:
    """Position on an ordered rubric of 2..10 named levels."""

    instructions: str
    criteria: Sequence[str]

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": list(self.criteria),
        }


Question = Noul | Choice | Score


# --------------------------------------------------------------------------- #
# Answers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NoulAnswer:
    noul: float
    type: Literal["noul"] = "noul"

    @property
    def value(self) -> float:
        return self.noul


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: Mapping[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    type: Literal["choice"] = "choice"

    @property
    def value(self) -> str:
        return self.choice


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    legend: Mapping[str, str] = field(default_factory=dict)
    probabilities: Mapping[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    type: Literal["score"] = "score"

    @property
    def value(self) -> float:
        return self.score


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


def parse_answer(payload: Mapping[str, Any]) -> Answer:
    """Build an Answer from a raw API answer object."""
    kind = payload.get("type")
    if kind == "noul":
        return NoulAnswer(noul=float(payload["noul"]))
    if kind == "choice":
        return ChoiceAnswer(
            choice=str(payload["choice"]),
            probabilities=dict(payload.get("probabilities") or {}),
            confidence=float(payload.get("confidence", 0.0)),
        )
    if kind == "score":
        return ScoreAnswer(
            score=float(payload["score"]),
            legend=dict(payload.get("legend") or {}),
            probabilities=dict(payload.get("probabilities") or {}),
            confidence=float(payload.get("confidence", 0.0)),
        )
    raise JevValidationError(f"unknown answer type: {kind!r}")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Origin:
    """Where a piece of content came from."""

    source: str                 # "tool:bash", "tool:read_file", "retrieval", "observation"
    ref: str | None = None      # command, path, url -- whatever identifies the instance
    turn: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "ref": self.ref, "turn": self.turn}

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> Origin:
        return Origin(source=d["source"], ref=d.get("ref"), turn=int(d.get("turn", 0)))


@dataclass(frozen=True)
class Segment:
    """A verbatim slice of some raw output, the unit of scoring and relocation."""

    id: str
    text: str
    kind: SegmentKind
    tokens: int
    origin: Origin
    line_span: tuple[int, int] | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreItem:
    """Anything that can be scored: a Segment, a DigestEntry, a Record."""

    id: str
    text: str
    tokens: int
    meta: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_segment(seg: Segment) -> ScoreItem:
        return ScoreItem(id=seg.id, text=seg.text, tokens=seg.tokens,
                         meta={"kind": seg.kind})


@dataclass(frozen=True)
class ScoreResult:
    item_id: str
    score: float
    failed: bool = False
    error: str | None = None
    batch_index: int = -1


@dataclass
class Record:
    """A unit of external memory."""

    id: str
    text: str
    kind: str
    origin: Origin
    tokens: int
    created_turn: int
    lifecycle: Lifecycle = "session"
    summary: str = ""
    task_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    expand_count: int = 0
    hit_count: int = 0


@dataclass(frozen=True)
class DigestEntry:
    """The cheap view of a Record, small enough that hundreds fit in a Jev state."""

    id: str
    summary: str
    kind: str
    tokens: int
    created_turn: int

    def to_score_item(self) -> ScoreItem:
        return ScoreItem(id=self.id, text=self.summary, tokens=self.tokens,
                         meta={"kind": self.kind, "created_turn": self.created_turn})


@dataclass(frozen=True)
class Pointer:
    """The one-line stand-in left in context for relocated content."""

    id: str
    lines: tuple[int, int] | None
    tokens: int
    summary: str


# --------------------------------------------------------------------------- #
# Context buffer
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Block:
    """The unit of freezing."""

    id: str
    role: str                               # "system" | "user" | "assistant" | "tool"
    text: str
    tokens: int
    segments: tuple[Segment, ...] = ()
    committed_at_turn: int | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderedMessage:
    role: str
    content: str
    cacheable: bool = False


@dataclass(frozen=True)
class BufferStats:
    frozen_tokens: int
    work_tokens: int
    frozen_blocks: int
    work_blocks: int
    cache_breakpoint: int


@dataclass(frozen=True)
class TurnSignals:
    """What the agent loop knows at the moment a commit decision is made."""

    turn: int
    tool_depth: int = 0
    last_role: str = "assistant"
    open_questions: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommitDecision:
    commit: bool
    reason: str
    upto: int | None = None


@dataclass(frozen=True)
class CommitResult:
    committed_blocks: int
    committed_tokens: int
    reason: str
    cache_breakpoint: int


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #


@runtime_checkable
class JevClient(Protocol):
    """Anything that can answer typed questions about a state."""

    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]: ...


@runtime_checkable
class MemoryStore(Protocol):
    def put(self, record: Record) -> str: ...

    def get(self, record_id: str) -> Record | None: ...

    def digest(
        self,
        *,
        budget_tokens: int = 24_000,
        kinds: Collection[str] | None = None,
        lifecycle: Collection[Lifecycle] | None = None,
    ) -> list[DigestEntry]: ...

    def search(self, query: str, *, limit: int = 50) -> list[Record]: ...

    def touch(self, record_id: str, *, expand: bool = False, hit: bool = False) -> None: ...

    def purge(self, *, turn: int, task_id: str | None = None) -> int: ...


@runtime_checkable
class CommitPolicy(Protocol):
    def should_commit(self, buffer: Any, signals: TurnSignals) -> CommitDecision: ...


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class JevError(Exception):
    """Base for every Jev transport or protocol failure."""


class JevAuthError(JevError):
    """401 -- bad or missing API key."""


class JevValidationError(JevError):
    """422 -- the request is malformed. Never retried; it is a bug."""


class JevRejectedError(JevValidationError):
    """403 -- the service refused to read the request's content.

    Seen from TypeSafe's edge firewall on content that looks like an attack
    (``cat /etc/passwd``, SQL injection strings). Retrying the same content is
    pointless; which item tripped it can only be found by asking about fewer.
    """


class JevBudgetError(JevError):
    """A request would violate a hard Jev limit. Raised before sending; split and retry."""


class JevUnavailableError(JevError):
    """429/529/timeout, retries exhausted. Callers should fail open."""


class FrozenPrefixError(RuntimeError):
    """An operation would have mutated the frozen prefix. The cache invariant is absolute."""
