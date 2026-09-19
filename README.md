# awesome-jev-compaction

Cache-preserving agent context compaction and memory, with
[TypeSafe Jev](https://docs.typesafe.ai) as the semantic judgement layer.

> An agent's context is `[frozen prefix] + [work area]`. The frozen prefix is append-only, so the
> KV cache over it is never invalidated. Raw tool output is **relocated**, never deleted: what the
> gate removes goes to an external store and leaves a one-line pointer the agent can `expand()`.
> Jev decides what to relocate at write time and what to pull back at read time — the same scoring
> primitive at both ends.

See **[SPEC.md](SPEC.md)** for the full design and the Jev constraints every decision is derived
from. Phase 0 and Phase 1 are implemented; 312 tests, no network required.

## The three constraints that shape everything

| Constraint | Consequence |
|---|---|
| `state` + longest question ≲ **32k tokens** | Jev never sees a full agent context. Every call works on a digest or a chunk. |
| **32 questions** per request | Independent per-item judgements batch at 32. `Choice`'s 255 options are a *ranking*, not independent gating. |
| `$0.042`/MTok in, **output free** | Cost is a function of `state` alone. Fan out many questions over one shared state; never re-send the same state per item. |

That last one is the design's central claim, so it is mechanised as a test: for every request,
the set of item texts in `state` must equal exactly the set its questions ask about.

## Usage

```python
from jevctx import (
    ContextBuffer, InMemoryStore, ShadowLog, HttpJevClient, Origin,
    admit, expand, make_block, DEFAULT_COMMIT_POLICY,
)
from jevctx.types import TurnSignals

store, log = InMemoryStore(), ShadowLog(path="shadow.jsonl")
buffer, client = ContextBuffer(), HttpJevClient()   # reads TYPESAFE_API_KEY

buffer.append_work(make_block("system", "You are a build assistant."))
buffer.commit(reason="system prompt is stable")

# Gate a tool result on its way into the work area.
result = admit(raw_output, Origin(source="tool:bash", ref="npm install", turn=1),
               task_digest="Fix the dependency resolution failure.",
               turn=1, client=client, store=store, log=log)

buffer.append_work(make_block("tool", result.text, turn=1))
decision = DEFAULT_COMMIT_POLICY.should_commit(buffer, TurnSignals(turn=1, tool_depth=0))
if decision.commit:
    buffer.commit(decision.upto, reason=decision.reason)

# Nothing was destroyed; the agent can go back for it.
original = expand(result.pointers[0].id, store=store, log=log, turn=2)
```

Running that against a fake scorer over a 60-line npm log:

```
875 -> 405 tokens, 1 pointer(s)
[[elided id=r:243d3a59 lines=1-31 tokens=501 "npm http fetch GET 200 https://registry.npmjs.org/p0 0ms"]]
expanded 1751 chars back; false-negative rate now 100%
replay at 0.0 would have caused 0 false negatives
```

Register `EXPAND_TOOL_SCHEMA` with the agent's tools. The gate is only safe if the agent can
actually undo it.

## Rolling it out

1. **Phase 0 alone.** `ContextBuffer` + a commit policy. No Jev, no store. Measure cache hit rate.
2. **Phase 1 with `GateConfig(shadow_only=True)`.** Everything is scored and logged; nothing is
   elided. This is a supported mode, not a debug flag.
3. `ShadowLog.replay(t)` across candidate thresholds, then set `shadow_only=False`.
4. Watch `false_negative_rate()`. Above ~2%, the threshold is too high.

## Safety rails

The gate is wrong sometimes, so three things bound the damage:

- **Relocation, not deletion.** A false drop costs a round trip, not information.
- **A tripwire.** If the scorer wants to elide more than 70% of an output, the output is kept and
  the tripwire is logged — a scorer that wants to drop almost everything is reporting a bad
  question, not a worthless output.
- **Fail-open.** A Jev outage scores everything 1.0 and passes text through untouched.

**Jev is not a security boundary.** Typed output means Jev itself cannot be turned into an
instruction emitter, but its *judgement* can be influenced by the text it is judging. Tool
allowlists and approval gates still apply.

## Development

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest      # 312 tests, no network, no API key
ruff check .
```
