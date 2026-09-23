# awesome-jev-compaction

**Using [Jev](https://docs.typesafe.ai) to manage an agent's memory and keep its context small.**

[![Tool output arrives, Jev scores every segment, the low-value parts move out to a store](docs/gate.gif)](docs/index.html)

That's [`docs/index.html`](docs/index.html) — open it to drive it yourself.
[`docs/showcase.html`](docs/showcase.html) has the numbers behind it, with a threshold
slider you can drag.

```bash
git clone https://github.com/Waxmell114514/awesome-jev-compaction
cd awesome-jev-compaction
uv venv .venv && uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python demo.py      # runs offline, no API key needed
```

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

### 1. Filter at the door

Split the context into two pieces: a **frozen part** at the front, and a **work area** at the
end. The frozen part only ever grows — nothing in it is ever rewritten. So the cache over it
never breaks.

New tool output goes into the work area. Before it goes in, ask Jev whether it's worth
keeping. Low-scoring parts never enter.

### 2. Move it out and don't delete it

What the gate removes doesn't disappear. It goes into a store, and leaves one line behind:

```
[[elided id=r:8506e122 lines=1-21 tokens=364 "npm http fetch GET 200 https://registry..."]]
```

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
.venv/bin/python demo.py                            # six short acts
TYPESAFE_API_KEY=sk-... .venv/bin/python demo.py    # same code, real Jev
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
| `jevctx/label.py` | Type, lifetime and entity labels for stored records, batched like the scorer |
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

## Run a real OpenAI-compatible model

`run_agent.py` connects the gate to a Chat Completions tool loop, including
OpenCode endpoints that implement that protocol. It uses the existing `httpx`
dependency. Configure your endpoint and model; neither is hardcoded:

```powershell
$env:OPENAI_BASE_URL = "https://YOUR-ENDPOINT/v1"
$env:OPENAI_MODEL = "YOUR-MODEL"
$env:OPENAI_API_KEY = "YOUR-KEY"
$env:TYPESAFE_API_KEY = "YOUR-JEV-KEY"  # only shadow/on require this

python -X utf8 run_agent.py "Find the dependency conflict in the log" --file tests/fixtures/npm_install.log --mode off --output runs/baseline
python -X utf8 run_agent.py "Find the dependency conflict in the log" --file tests/fixtures/npm_install.log --mode shadow --output runs/shadow
python -X utf8 run_agent.py "Find the dependency conflict in the log" --file tests/fixtures/npm_install.log --mode on --output runs/gated
```

The endpoint must accept `/chat/completions`, function tools and
`max_completion_tokens`. Environment variables are read from the process; `.env`
files are not loaded automatically. Only files explicitly passed with `--file`
are available to the model (UTF-8, up to 1 MB each). File contents can be sent to
the configured model and, in shadow/on modes, to Jev. There is no shell or write
tool. For application-defined tools, call `jevctx.agent.run_agent` with matching
function schemas and Python handlers; handlers own validation and permissions.

- `off`: original tool results, no Jev calls.
- `shadow` (default): score and log, but send original tool results.
- `on`: filter tool results and let the model recover originals with `expand`.

Each run requires a **new** output directory. `run.json` contains the transcript,
completion status, elapsed time, tool/expand counts and provider-reported usage.
`shadow.jsonl` stores decisions; `memory.jsonl` stores relocated original text
when needed. These files contain task data; `runs/` is gitignored. Runs do not
resume existing sessions or rebuild an overflowing context window yet.

Costs are unknown (`null`) unless prices are supplied. Use
`--prices INPUT OUTPUT CACHE_READ CACHE_WRITE` and `--jev-input-price PRICE`,
all in USD per million tokens. Token usage comes from API responses, while
`estimated_tool_tokens_saved` is only the local text-size estimate. Standard
Chat Completions reports cached reads inside `prompt_tokens`; the runner
subtracts them before pricing ordinary input. The standard protocol has no
separate cache-write count, so those tokens stay in ordinary input. Providers
with extra cache-write charges need a provider-specific usage adapter before
the cost estimate is authoritative. The prices should reflect your endpoint's
actual rates; totals are estimates, not invoices. Missing host usage also makes
cost unknown. Usage reported before a failed run is retained, but an unsuccessful
request may have incurred additional unreported charges.

The runner keeps complete assistant/tool messages and their call IDs under
`ContextBuffer`'s frozen-prefix invariant. Prefix stability does not guarantee a
provider cache hit; inspect reported cached tokens. `completed` means the model
stopped normally, **not** that the task answer is correct. Compare answers or
external checks as well as costs; repeated runs need not take identical paths.
The offline test suite checks the protocol and recovery with mock transports;
live endpoint compatibility and task quality still require your configuration.

`CacheLedger` remains a render-cost simulator. Its USD estimate now defaults to
`None` until a host `price_per_input_token` is supplied; it no longer borrows
Jev's price. Actual runs use separate host and Jev accounting.

## What's not built

- Jev-driven commit points — asking "is this subtask finished?"
- Supersession chains — does this new record replace that old one?
- Bayesian threshold calibration — combine Jev's score with observed hit counts