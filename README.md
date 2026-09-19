# awesome-jev-compaction

**Using [Jev](https://docs.typesafe.ai) to manage an agent's memory and keep its context small.**

```bash
git clone https://github.com/Waxmell114514/awesome-jev-compaction
cd awesome-jev-compaction
uv venv .venv && uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python demo.py      # runs offline, no API key needed
```

Or open [`docs/showcase.html`](docs/showcase.html) — the same tour as a web page, with a
slider you can drag.

---

## The problem

An agent's context fills up. The usual fix is **compaction**: when it gets too long, ask a
model to summarise the old parts and replace them.

Two things go wrong. First, rewriting the middle of the context **breaks the prompt cache**,
so every later turn pays full price again. Second, a summary is *generated* text — the model
can quietly invent a detail, and now that invented detail is your agent's memory.

---

## Our idea

Three parts. Each one is simple on its own.

三点，每一点单独看都很简单。

### 1. Filter at the door

Split the context into two pieces: a **frozen part** at the front, and a **work area** at the
end. The frozen part only ever grows — nothing in it is ever rewritten. So the cache over it
never breaks.

New tool output goes into the work area. Before it goes in, ask Jev whether it's worth
keeping. Low-scoring parts never enter.

### 2. Move it out and don't delete it

The agent gets an `expand` tool. If it needs what's behind that line, it asks for it and gets
the **original text, byte for byte**. So a wrong decision costs one extra round trip — not
lost information.

Because Jev can't write text, what you keep is always the original lines. Nothing in memory
was ever made up.

### 3. Record every decision, so you can tune it later

Every decision is logged with its score. When the agent later calls `expand` on something you
removed, that's **proof the filter was too aggressive** — it went looking for exactly what you
took away.

Later, replay the log at a different threshold. No re-running the agent, no extra Jev calls:

```
  threshold    kept   relocated   tokens saved   still missed
  0.10           12           3          1,092              0
  0.35            9           6          1,260              3
```

`0.10` saves 1,092 tokens and takes away nothing the agent came back for. `0.35` saves 168
more tokens and costs three. Now you can choose.

---

## Using the repo

Start with the demo. It runs offline against a scripted stand-in, so you don't need a key:

```bash
.venv/bin/python demo.py                  # six short acts / 六小节
TYPESAFE_API_KEY=sk-... .venv/bin/python demo.py   # same code, real Jev / 同样的代码，真实的 Jev
```

### Connecting a real key

Set `TYPESAFE_API_KEY` and everything switches to real Jev — `demo.py` picks it up, and
`HttpJevClient()` reads it by default. To check the key actually works before wiring it into
an agent:

```bash
export TYPESAFE_API_KEY=...
.venv/bin/python -m jevctx.check
```

That makes real requests and checks three things: all three question types come back and
parse, the gate actually relocates something from a real tool output, and the pointer expands
back byte for byte. It prints latency, token usage and cost, and on failure it tells you
which of the three broke — a bad key, a malformed request, or an unreachable server.

Then read these, in this order

| File | What's in it |
|---|---|
| [`demo.py`](demo.py) | The tour. Start here. |
| `jevctx/pipeline.py` | `admit()` / `retrieve()` / `expand()` — the ~200 lines that matter |
| `jevctx/scorer.py` | How 32 questions get packed into one request |
| `jevctx/context.py` | The frozen-prefix buffer |
| `jevctx/segments.py` | Splitting output without breaking it |

To use it in your own agent

```python
from jevctx import admit, expand, InMemoryStore, ShadowLog, HttpJevClient, Origin, GateConfig

store, log, client = InMemoryStore(), ShadowLog("shadow.jsonl"), HttpJevClient()

result = admit(tool_output, Origin(source="tool:bash", ref="npm install", turn=1),
               task_digest="What the agent is currently doing.",
               turn=1, client=client, store=store, log=log,
               config=GateConfig(shadow_only=True))   # start here

# result.text goes into your context. Register `expand` as a tool.
```

**Start with `shadow_only=True`.** It scores and logs everything but changes nothing. Run it
for a day, replay the log to pick a threshold, then turn it on. Turning a context filter
straight on is how you lose a week to "the agent got worse and nobody knows when."

---

## What's not built · 没做的部分

- Jev-driven commit points — asking "is this subtask finished?"
- Multi-dimensional labelling — type, lifetime, entities, in one call of ~15 questions
- Supersession chains — does this new record replace that old one?
- Bayesian threshold calibration — combine Jev's score with observed hit counts
