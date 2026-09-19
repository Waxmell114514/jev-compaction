# jevctx — Specification (Phase 0 & Phase 1)

Cache-preserving context management and agent memory, with [TypeSafe Jev](https://docs.typesafe.ai)
as the semantic judgement layer.

Status: **Phase 0 + Phase 1 are in scope for this spec.** Phase 2 and 3 are sketched at the end
only to the extent needed to keep the Phase 0/1 interfaces extensible.

---

## 0. The one-paragraph version

An agent's context is `[frozen prefix] + [work area]`. The frozen prefix is append-only, so the
KV cache over it is never invalidated. Everything mutable happens in the work area. Raw tool output
enters the work area in full; oversized output is **relocated** (never deleted) to an external
store, leaving a one-line pointer that the agent can `expand()`. Jev decides *what* to relocate at
write time and *what* to pull back at read time — the same scoring primitive applied at both ends.

---

## 1. Hard constraints from the Jev API

These are not preferences. Every design decision below traces back to one of them.

| Constraint | Value | Consequence for this design |
|---|---|---|
| State size | `state + longest single question` ≲ **32k tokens** | Jev can never see a full agent context. Every Jev call operates on a **digest or a chunk**, never on the whole prefix. |
| Request budget | `state + all questions` ≲ **64k tokens** | The budget planner must account for question text, not just state. |
| Questions per request | **32** | Independent per-item judgements batch at 32/request. "Score 500 lines in one call" is only possible with `Choice` (≤255 options), which is a *sum-to-1 ranking*, not independent keep/drop. |
| Choice options | ≤ **255** | Use `Choice` for top-k ranking, `Noul` for independent gating. |
| Score levels | **2–10** | Tiering / lifecycle labels fit natively. |
| Pricing | `$0.042 / MTok` input, **output free** | **Cost is a function of state only.** Questions 2..32 are free. |
| Latency | 70–500 ms | At most 2 serial Jev rounds on the hot path. |
| Rate limit | 1200 req/min, 250k tok/s | Bounded concurrency is mandatory, not optional. |
| Language | English best; CJK "supported but less reliable" | `instructions` / `criteria` are **always written in English**, even when `state` is not. |

### 1.1 The central cost rule

> **Billed by state, free by question.** Fan out many questions over one shared state.
> Never re-send the same state once per item.

Concretely, to score 150 segments:

- **Correct:** 5 requests. Request *i* has `state = {segments 32i..32i+31}` and 32 `Noul`s.
  Total input ≈ the document once, plus question boilerplate.
- **Wrong:** 5 requests, each with `state = the whole document` and 32 `Noul`s.
  Total input ≈ 5× the document.

Any implementation that puts more into `state` than the questions of that same request actually
refer to is a bug. There is a test for this (§7.3).

---

## 2. Module map and ownership

```
jevctx/
  types.py      # shared dataclasses, protocols, constants        [FROZEN — do not edit]
  tokens.py     # estimate_tokens()                               [FROZEN — do not edit]
  testing.py    # FakeJevClient, fixtures                         [FROZEN — do not edit]
  jev.py        # HttpJevClient, retry, BudgetPlanner             (A)
  segments.py   # deterministic segmentation of raw output        (B)
  context.py    # ContextBuffer: frozen prefix + work area        (C)
  ledger.py     # CacheLedger: cache-cost accounting              (C)
  store.py      # MemoryStore: InMemoryStore, JsonlStore          (D)
  shadow.py     # ShadowLog: decision + outcome logging           (D)
  scorer.py     # score_items(): chunking + parallel Jev          (E)
  pipeline.py   # admit() / retrieve() / expand()                 (E)
```

`types.py`, `tokens.py` and `testing.py` are written up front and are the contract. Implementers
**must not modify them**; if something in them is wrong, report it rather than editing.

---

## 3. Phase 0 — the staging area (no Jev required)

Phase 0 must be independently useful. An agent that adopts only Phase 0, with zero Jev calls,
should already see a higher cache hit rate.

### 3.1 Model

```python
ContextBuffer:
    frozen: list[Block]   # append-only. Index i, once written, is immutable forever.
    work:   list[Block]   # freely mutable, reorderable, discardable
```

A `Block` is the unit of freezing: `{id, role, segments, tokens, committed_at_turn}`.

### 3.2 Operations

| Method | Contract |
|---|---|
| `append_work(block) -> None` | Always allowed. |
| `replace_work(blocks) -> None` | Replaces the entire work area. No cache impact. |
| `drop_work(block_ids) -> None` | Removes blocks from the work area only. |
| `commit(upto: int \| None = None, *, reason: str) -> CommitResult` | Moves the first `upto` work blocks (all, if `None`) into `frozen`. Raises `FrozenPrefixError` if anything already frozen would change. |
| `render() -> list[RenderedMessage]` | Materialises the buffer. The element at index `i < cache_breakpoint` **must be byte-identical to the previous render**. |
| `cache_breakpoint -> int` | Index of the first work-area message in `render()`. |
| `stats() -> BufferStats` | `frozen_tokens`, `work_tokens`, `frozen_blocks`, `work_blocks`, `cache_breakpoint`. |

### 3.3 The invariant

> For any two renders `r1` (earlier) and `r2` (later) of the same buffer,
> `r1[:r1.cache_breakpoint] == r2[:r1.cache_breakpoint]`.

This is *the* property of the whole design. It must be enforced by a runtime assertion in
`commit()` and covered by a property-style test that drives a randomised sequence of operations
(§7.1).

### 3.4 Commit policies

```python
class CommitPolicy(Protocol):
    def should_commit(self, buffer: ContextBuffer, signals: TurnSignals) -> CommitDecision: ...
```

`TurnSignals` carries what the agent loop knows: `turn`, `tool_depth`, `last_role`,
`open_questions: int`, plus a free-form `extra: dict`.

Phase 0 ships these, all pure code:

- `ToolDepthZero()` — commit when `tool_depth == 0` and the work area holds ≥ `min_blocks` (default 2).
- `WorkAreaTokens(threshold=8000)`
- `TurnCount(n=8)`
- `AnyOf(*policies)` / `AllOf(*policies)` / `NeverCommit()`
- Default: `AnyOf(WorkAreaTokens(8000), ToolDepthZero(min_blocks=4))`

Phase 2 will add `JevCommitPolicy`. It must be droppable in **without editing `context.py`** —
that is the point of the protocol.

### 3.5 What the commit question is actually about

Cost almost never argues against compaction. With prompt caching at ~0.1× for reads and ~1.25×
for writes, rewriting a prefix of `P` tokens down to `P'` pays back after

```
k = 12.5 * P' / (P - P')   turns
```

(derivation in `ledger.py` docstring: `0.1·P·k = 1.25·P' + 0.1·P'·k`). For `P=100k → P'=20k`
that is **~3 turns**. So the commit decision is a *safety* question, not an economic one:
is the subtask actually finished, is anything in the work area still unresolved. `CacheLedger`
exists to make this quantitative, not to gate on it.

### 3.6 `ledger.py`

```python
class CacheLedger:
    def record_render(self, turn: int, frozen_tokens: int, work_tokens: int,
                      cache_written: bool) -> None
    def breakeven_turns(self, p: int, p_prime: int) -> float
    def estimated_cost(self) -> CostBreakdown   # cache_write_tokens, cache_read_tokens,
                                                # uncached_tokens, usd (base price configurable)
    def would_compaction_pay(self, p: int, p_prime: int, remaining_turns: int) -> bool
```

Multipliers (`CACHE_READ_MULT = 0.1`, `CACHE_WRITE_MULT = 1.25`) and the base input price are
constructor arguments with those defaults, never hardcoded at the call site.

---

## 4. Phase 1 — one scorer, used at both ends

### 4.1 `segments.py` — deterministic segmentation (no model)

```python
def segment(text: str, origin: Origin, *, max_segment_tokens: int = 2000) -> list[Segment]
```

Line-splitting is wrong for most tool output. Detect and handle:

| Kind | Detection | Segmentation rule |
|---|---|---|
| `json` | parses as JSON | One segment per top-level array element / object key. Never split mid-value. |
| `stacktrace` | language-specific frame patterns (Python `File "...", line N`, JS `at fn (...)`, Java `\tat ...`) | The **whole trace is one segment**. Splitting a trace destroys it. |
| `diff` | `diff --git` / `@@ -n,m +n,m @@` | One segment per hunk, header retained. |
| `table` | ≥3 lines sharing a consistent separator (`|`, aligned columns, TSV) | Header + rows; header duplicated into every segment. |
| `log` | ≥60% of lines start with a timestamp or level token | One segment per *run* of same-level lines, capped at `max_segment_tokens`. |
| `code` | fenced block, or file-extension hint in `origin.ref` | Split at top-level `def`/`class`/`function`/blank-line-delimited blocks. |
| `text` | fallback | Paragraph runs (blank-line delimited), then hard-capped. |

Rules that hold for every kind:

1. **Lossless.** `"".join(s.text for s in segment(t, o)) == t`. Non-negotiable; there is a test.
2. Each `Segment.id` is `"s:" + sha256(text + origin.ref)[:8]` — stable across runs.
3. `line_span` is a 1-indexed inclusive `(start, end)` into the original.
4. A segment exceeding `max_segment_tokens` is split at the nearest line boundary and both halves
   are marked `meta["split"] = True`.

### 4.2 `jev.py` — transport and budget

```python
class HttpJevClient:
    def __init__(self, api_key: str | None = None, *, model: str = "jev-latest",
                 base_url: str = "https://api.typesafe.ai/v1",
                 timeout: float = 15.0, max_retries: int = 3,
                 max_concurrency: int = 16) -> None
    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]
```

- `POST {base_url}/systemone`, body `{"model":..., "state":..., "questions":...}`.
- Retry on `429` and `529` with exponential backoff + full jitter (base 0.5s, cap 8s).
  Honour `Retry-After` when present. **Never retry `422`** — that is a bug in the request.
- `401 → JevAuthError`, `422 → JevValidationError`, exhausted retries → `JevUnavailableError`.
  All inherit `JevError`.
- Raise `JevBudgetError` *before* sending if `len(questions) > 32`, or a `Score` has <2 or >10
  levels, or a `Choice` has >255 options, or either token budget is exceeded. Callers split.
- A bounded thread pool / semaphore caps in-flight requests at `max_concurrency`, and a token
  bucket keeps the request rate under `RATE_LIMIT_RPM`.

```python
@dataclass(frozen=True)
class Batch:
    items: list[ScoreItem]       # the items whose text goes into this batch's state
    question_keys: list[str]

class BudgetPlanner:
    def __init__(self, *, max_questions: int = 32,
                 state_plus_all_questions: int = 64_000,
                 state_plus_longest_question: int = 32_000,
                 headroom: float = 0.9) -> None
    def plan(self, items: Sequence[ScoreItem], question_tokens: int,
             envelope_tokens: int = 0) -> list[Batch]
```

`plan()` packs items into batches that satisfy *both* token budgets and the 32-question cap,
applying `headroom` (default 10%) because `estimate_tokens` is an estimate. An item that cannot
fit alone is emitted as a **single-item batch with `meta["oversized"] = True`**; the scorer must
then default that item to *keep*, never drop it.

### 4.3 `scorer.py` — the shared primitive

```python
def score_items(
    client: JevClient,
    task_digest: str,
    items: Sequence[ScoreItem],
    question: Noul,
    *,
    planner: BudgetPlanner | None = None,
    max_workers: int = 8,
    on_error: Literal["keep", "raise"] = "keep",
) -> list[ScoreResult]
```

State per batch:

```json
{"task": "<task_digest>",
 "items": [{"ref": "i0", "text": "..."}, {"ref": "i1", "text": "..."}]}
```

Questions: one `Noul` per item, keyed `i0..i31`, with `instructions` that name the ref:
`"Considering item i0 only: <question.instructions>"`. Answers map back by ref.

Rules:

- **`state` contains only the items of that batch.** Enforced by test §7.3.
- Batches run in parallel through a `ThreadPoolExecutor(max_workers)`; the client's own semaphore
  is the real rate limiter.
- **Fail-open.** On `JevError` with `on_error="keep"`, every item in the failed batch gets
  `score=1.0`, `failed=True`, `error="<class>: <msg>"`. A Jev outage must never silently strip an
  agent's context. `on_error="raise"` exists for tests and offline batch jobs.
- Oversized items get `score=1.0, failed=True, error="oversized"` without a request.
- `ScoreResult` is `{item_id, score, failed, error, batch_index}`; order matches `items`.

The API is synchronous. Parallelism is threads, not asyncio — an agent loop should not have to be
async to use this.

### 4.4 `store.py` — relocation target

```python
@dataclass
class Record:
    id: str; text: str; kind: str; origin: Origin; tokens: int; created_turn: int
    lifecycle: Lifecycle = "session"   # Phase 2 fills this via Jev
    summary: str = ""                  # <= summary_max_chars, for the digest
    meta: dict = field(default_factory=dict)
    expand_count: int = 0              # false-negative signal
    hit_count: int = 0                 # Phase 3 Bayesian prior input

class MemoryStore(Protocol):
    def put(self, record: Record) -> str
    def get(self, record_id: str) -> Record | None
    def digest(self, *, budget_tokens: int = 24_000, kinds: Collection[str] | None = None,
               lifecycle: Collection[Lifecycle] | None = None) -> list[DigestEntry]
    def search(self, query: str, *, limit: int = 50) -> list[Record]
    def touch(self, record_id: str, *, expand: bool = False, hit: bool = False) -> None
    def purge(self, *, turn: int, task_id: str | None = None) -> int
```

- `InMemoryStore` and `JsonlStore` (append-only JSONL + in-memory index, survives restart).
- `digest()` must respect `budget_tokens` because it becomes a Jev `state` (32k cap). Ordering:
  most recent first; truncate at the budget. `DigestEntry` = `{id, summary, kind, tokens, created_turn}`.
- `summary` is generated deterministically when absent: first non-empty line, collapsed
  whitespace, truncated to `summary_max_chars` (default 120) with `…`. **No model call.**
- `search()` in Phase 1 is a cheap prefilter only: case-folded token overlap scoring
  (BM25-lite is fine, an embedding index is not in scope).
- `purge()` implements lifecycle eviction: `turn` → end of turn, `task` → end of task,
  `session` → never automatically, `permanent` → never.

### 4.5 `shadow.py` — the calibration substrate

Every decision and every outcome is logged as JSONL. This is what makes Phase 3 possible, and it
costs nothing to add now.

```python
class ShadowLog:
    def decision(self, *, kind: Literal["admit","retrieve"], item_id: str, score: float,
                 threshold: float, action: Literal["kept","elided","injected","skipped"],
                 tokens: int, origin: Origin, text: str, turn: int) -> None
    def outcome(self, *, kind: Literal["expand","hit"], item_id: str, turn: int) -> None
    def false_negative_rate(self) -> float          # expands / elisions
    def stats(self) -> ShadowStats
    def replay(self, threshold: float) -> ShadowStats  # what a different threshold would have done
```

- Records store `text_sha256` and a `text_preview` (≤200 chars), **never the full text** — the
  store already has it, and the log should be cheap to keep forever.
- `false_negative_rate()` is the single number that says whether the threshold is wrong. An
  `expand` on something that was elided is ground truth that the gate was too aggressive.
- `replay()` recomputes what a counterfactual threshold would have kept/elided from logged scores,
  so thresholds can be tuned offline without re-running the agent.

### 4.6 `pipeline.py` — wiring

```python
@dataclass
class GateConfig:
    keep_threshold: float = 0.35        # deliberately low: false-drop costs >> false-keep
    min_gate_tokens: int = 400          # below this, don't gate at all — skip the latency
    max_elide_fraction: float = 0.7     # if the scorer wants to drop more, don't trust it
    protected_kinds: frozenset[str] = frozenset({"stacktrace", "diff"})
    protected_floor: float = 0.05       # protected kinds elide only below this
    shadow_only: bool = False           # log decisions, elide nothing — the rollout mode
    summary_max_chars: int = 120

def admit(raw: str, origin: Origin, *, task_digest: str, turn: int,
          client: JevClient, store: MemoryStore, log: ShadowLog,
          config: GateConfig = GateConfig()) -> AdmitResult

def retrieve(task_digest: str, *, turn: int, client: JevClient, store: MemoryStore,
             log: ShadowLog, k: int = 5, budget_tokens: int = 24_000,
             threshold: float = 0.5) -> list[Record]

def expand(record_id: str, *, store: MemoryStore, log: ShadowLog, turn: int) -> str
```

`admit()`:

1. If `estimate_tokens(raw) < min_gate_tokens` → return unchanged, `gated=False`. No Jev call.
2. `segment(raw, origin)`.
3. `score_items(client, task_digest, segments, ADMIT_QUESTION)`.
4. Partition at `keep_threshold`, with the `protected_kinds` / `protected_floor` override.
5. If the elided fraction would exceed `max_elide_fraction`, **keep everything** and log
   `action="kept"` with `meta["tripwire"]="max_elide_fraction"`. A scorer that wants to drop 90%
   of a tool output is reporting a bad question, not a worthless output.
6. Merge contiguous elided segments into one record; `store.put()`; replace with a pointer.
7. If `config.shadow_only`, do steps 3–5 and log, but return the original text untouched.

Pointer format, exactly:

```
[[elided id=7f3a91 lines=12-25 tokens=380 "npm install progress output"]]
```

Parseable by `parse_pointer(line) -> Pointer | None`, which round-trips with `format_pointer`.

`ADMIT_QUESTION` (English, per §1):

```python
Noul(
  instructions=("Will this item still be needed later in the task described in `task`? "
                "Answer true if it contains facts, identifiers, errors, results, or decisions "
                "that a later step may have to refer back to. Answer false only if it is "
                "progress noise, repeated boilerplate, or formatting with no retained content."),
  true="The item carries information a later step may need.",
  false="The item is noise that can be recovered from the store if ever needed.",
)
```

`retrieve()` scores `store.digest(budget_tokens=...)` against `task_digest` with
`RETRIEVE_QUESTION`, takes those above `threshold`, returns the top `k` full records, and logs
`action="injected"` / `"skipped"`. Callers put the results in the **work area**, never the frozen
prefix.

`expand()` returns the original text and calls `store.touch(id, expand=True)` plus
`log.outcome(kind="expand", ...)`. Also export `EXPAND_TOOL_SCHEMA`, a ready-to-register tool
definition, because the gate is only safe if the agent can actually undo it.

---

## 5. Rollout order (this is a product requirement, not advice)

1. Phase 0 alone. No Jev, no store. Measure cache hit rate.
2. Phase 1 with `shadow_only=True`. Everything is scored and logged, nothing is elided.
3. Read `ShadowLog.replay(t)` across candidate thresholds; pick one; set `shadow_only=False`.
4. Watch `false_negative_rate()`. If it rises above ~2%, the threshold is too high.

`shadow_only` is not a debug flag. It is step 2 of the supported rollout and must work end to end.

---

## 6. Non-goals for Phase 0/1

- No Jev-driven commit points (`JevCommitPolicy` is Phase 2).
- No multi-dimensional metadata labelling (Phase 2). `Record.lifecycle` exists and defaults to
  `"session"`; nothing sets it from a model yet.
- No supersession/version chains, no tier promotion, no Bayesian threshold calibration (Phase 3).
  `hit_count` / `expand_count` are recorded now so Phase 3 has inputs.
- No embedding index, no vector store.
- No async API.
- **Jev is not a security boundary.** Typed output means Jev itself cannot be turned into an
  instruction emitter, but its *judgement* can be influenced by the text it is judging. Nothing
  here replaces tool allowlists or approval gates, and no docstring should imply otherwise.

---

## 7. Required tests

Every module ships tests. `FakeJevClient` (in `jevctx/testing.py`) means **no test hits the
network** — a test that requires an API key is a failed test.

1. **Frozen-prefix property test.** Drive a randomised sequence of
   `append_work / replace_work / drop_work / commit` and assert after every step that
   `render()[:previous_cache_breakpoint]` is unchanged. ≥200 randomised steps, seeded.
2. **Segmentation losslessness.** For every fixture in `tests/fixtures/`,
   `"".join(s.text for s in segment(t, o)) == t`, and stack traces come back as exactly one segment.
3. **State-scoping test.** Run `score_items` over 100 items through `FakeJevClient`; assert that
   for every recorded request, the set of item texts present in `state` equals exactly the set of
   items that request's questions refer to. This is the §1.1 rule, mechanised.
4. **Budget test.** `BudgetPlanner.plan` never emits a batch violating 32 questions / 64k / 32k;
   an item too large to fit alone comes back as an oversized single-item batch.
5. **Fail-open test.** A `FakeJevClient` that raises `JevUnavailableError` yields all scores `1.0`
   with `failed=True`, and `admit()` returns the input text byte-identical.
6. **Tripwire test.** A scorer that scores everything 0.0 must not elide more than
   `max_elide_fraction`; `admit()` returns everything with the tripwire logged.
7. **Round-trip test.** `admit()` then `expand()` on each pointer reconstructs the original `raw`
   byte-for-byte.
8. **shadow_only test.** With `shadow_only=True`, output is byte-identical to input and the log
   contains one decision per segment.
9. **Retry test.** 429 then 200 succeeds; 422 is not retried; `Retry-After` is honoured.
10. **Ledger test.** `breakeven_turns(100_000, 20_000) == pytest.approx(3.125)`.

---

## 8. Phase 2/3 hooks (build nothing, just don't block it)

- `CommitPolicy` protocol → `JevCommitPolicy`.
- `Record.lifecycle` + `Record.meta` → the multi-dimensional labelling pass (one call,
  ~15 questions, one record — the 32-questions-per-request cap is the natural budget).
- `Record.hit_count` / `expand_count` + `ShadowLog` → Bayesian threshold calibration.
- `MemoryStore.search()` → the candidate prefilter that keeps supersession checks linear
  instead of quadratic.
