# Long sessions with Claude Haiku 5.5: metered cache cost of four context policies

Every number here is metered. Token counts are the Anthropic API's own `usage` fields
(`cache_creation_input_tokens`, `cache_read_input_tokens`) for every request. Dollars
are those counts at Claude Haiku 5.5's price sheet: $0.10 input, $0.01 cache read,
$0.125 cache write, $0.50 output per million tokens, and 5x all four past 100K prompt
tokens. That the cache rates also scale 5x past 100K is our assumption.

## Setup

| | |
|---|---|
| Agent | `bench/haiku/agent.py`: native Messages API, tools `bash`/`read`/`edit`/`write` (+ `expand`, `recall` where offered), cache breakpoints on the system prompt and the last block, thinking off, effort medium |
| Model | `claude-haiku-5-5`, also the judge (`LLMJudgeClient`, OpenAI-compatible endpoint) |
| Sessions | 8 chains of 10 django SWE-bench Verified issues (Django 4.0–4.2, all difficulties), worked in one conversation each; after each issue the patch is graded and the checkout reset to the next issue's base commit |
| Grading | SWE-bench's FAIL_TO_PASS and PASS_TO_PASS tests, run locally (no Docker); gold patches checked on 3.2, 4.0 and 4.1 instances |
| Runs | one seed; 480 issue runs in all |
| Statistics | ratios of summed cost, 95% bootstrap over the 8 chains (a session is the independent unit) |
| Judge thresholds | one global keep threshold of 0.1, from a shadow run on 4 chains of 5 Django 3.2 issues (59 scored segments, 9 later edited: thin) |

Arms:

| arm | what the request sees |
|---|---|
| `off` | the transcript as stored |
| `mask` | tool results older than 10 rounds replaced by a placeholder (observation masking, "The Complexity Trap") |
| `stale` | outputs the agent's own later calls made obsolete (file edited, same lines re-read, same command rerun, checkout reset) become `expand`-able pointers, when the break-even rule says the cache rewrite pays; no judge |
| `staleh` | `stale`, with turns left estimated from the issues still queued, and compaction regardless of price past ~85K tokens |
| `jev` | profiled admission gate, `expand`, `recall`, supersession and the work area, judged by Haiku |
| `jevh` | `jev` with the `staleh` horizon and price-tier rule |

## Results

| arm | resolved /80 | cost | cost vs `off` (host only) | prompt tokens vs `off` | cache hit | cache writes | requests past 100K |
|---|---:|---:|---|---|---:|---:|---:|
| `off` | 55 | $0.465 | — | — | 96% | 0.80M | 10 |
| `mask` | 53 | $1.496 | **3.22 [2.27, 4.72]**, dearer on 8/8 chains | **0.63 [0.48, 0.84]** | 19% | 10.97M | 0 |
| `stale` | 58 | $0.443 | 0.95 [0.59, 1.64], cheaper on 7/8 | 1.00 [0.69, 1.56] | 97% | 0.67M | 9 |
| `staleh` | 55 | $0.400 | 0.86 [0.67, 1.12], cheaper on 4/8 | 0.96 [0.78, 1.18] | 96% | 0.78M | 0 |
| `jev` | 52 | $0.635 + judge $0.658 (miscounted, see 2) | 1.37 [0.69, 2.25] | 1.07 [0.83, 1.26] | 97% | 0.61M | 40 |
| `jevh` | 52 | $0.364 + judge $0.710 (miscounted, see 2) | 0.78 [0.60, 1.04], cheaper on 6/8 | 0.87 [0.67, 1.12] | 97% | 0.65M | 0 |

Resolved, against `off`: `mask` −2 [−5, 0], `stale` +3 [0, 6], `staleh` 0 [−6, 5],
`jev` −3 [−6, −1], `jevh` −3 [−7, 1]. One seed: a difference of a few issues is
within run-to-run noise (2–6 points on single SWE-bench runs).

### 1. Masking shrinks the context and triples the bill

`mask` sent 37% fewer prompt tokens and cost 3.2x as much. It masks one more output
every round, so every request rewrites the cache from that output on: 10.97M tokens
written against 0.80M, a cache hit rate of 19% against 96%. On every one of the 8
chains it was dearer than doing nothing. A context size measured in tokens says
nothing about the bill under prompt caching; the edit pattern does.

### 2. A judge has to read at about the cache price to pay, and a cache-aware layout gets a Haiku judge most of the way

**Correction.** In the first `jev`/`jevh` runs, sixteen concurrent sessions shared one
judge object and each issue's judge cost was read off its running totals, so requests
were counted more than once; the judge's output tokens were not priced at all. Those
judge costs ($0.658, $0.710) and "41x the tool output" are not reliable, and are left
in the table above only for the record. The host costs, cache figures and resolved
counts of those runs are per session and stand. From here on every session has its own
judge, priced with its output.

Counted correctly, `jevh` with the judge as `LLMJudgeClient` lays it out (`jevh-flat`)
spent $0.178 on the judge and saved $0.088 of host cost against `off`. Output was most
of a judge's bill: every choice question is answered with a distribution over all its
options, and Haiku's output costs 5x its input.

`bench/haiku/judges.py` asks the same questions laid out for the prompt cache
(`cached`): the answer rules and a catalog of the distinct question templates in a
cached system block, the task in a cached user block, then only the items and one short
line per question ("i0:role: T3, about item i0"). Replayed on the 159 admission requests
of the eight `off` sessions (`judge_replay.py`, every request sent to every layout):

| judge layout | input tokens | of which cache reads | output tokens | cost | gate decision agrees with `flat` |
|---|---:|---:|---:|---:|---:|
| `flat` | 918K | 0 | 135K | $0.159 | — |
| `flat`, asked again | 922K | 0 | 135K | $0.160 | 97.8% |
| `cached` | 660K | 295K | 140K | $0.111 | 96.9% |
| `compact` (cached, short reply) | 645K | 291K | 61K | $0.070 | 90.1% |

`cached` changes the gate's decisions no more than asking `flat` twice does. `compact`
asks for one line of JSON listing only options at 0.05 or more; it costs least, but
reports P(change_site) as exactly 0 for a quarter of segments, elides a quarter more of
them, and no threshold restores agreement. It changes the judge's answers, not just
their length; it was not run live.

Live, on the same eight chains:

| arm | host | judge | total | total vs `off` | resolved /80 |
|---|---:|---:|---:|---|---:|
| `off` | $0.465 | — | $0.465 | — | 55 |
| `staleh` (no judge) | $0.400 | — | $0.400 | 0.86 [0.68, 1.13] | 55 |
| `jevh-flat` | $0.377 | $0.178 | $0.555 | 1.19 [0.98, 1.47] | 54 |
| `jevh-cached` | $0.369 | $0.122 | $0.491 | 1.06 [0.80, 1.42] | 52 |

The cached layout cut the judge's cost by 32% [15%, 41%], cheaper on 7 of 8 sessions,
as the replay said. It was still not enough: the judge cost $0.122 and saved $0.096 of
host cost, and the judge-free `staleh` was cheaper than `jevh-cached` on 7 of 8
sessions (ratio 0.81 [0.71, 0.93]). The host reads its context at the cache price, so
a judge that reads each output once must cost about that much per useful decision to
pay. Jev bills the state only (`jevctx/budget.py`); estimated that way, not metered, a
Jev judge on `jevh` would read about 1.5M tokens, $0.064 at $0.042/M.

### 3. The long-context price tier decides the bill of the sessions that reach it

Most sessions stayed under 100K prompt tokens. The few requests past it cost 5x, and
whether a session crosses is a matter of its trajectory: `off` crossed on two chains
(5 requests each), `stale` on another (9), `jev` on one (40 requests, most of its
cost). The `h` arms, which compact past ~85K whatever the cache price, never crossed.
Their lower totals come mostly from those few chains: against their fixed-prior arms
they were cheaper on only 3/8 (`staleh`) and 4/8 (`jevh`) chains. The horizon estimate
did not show an effect of its own; the price tier did, where it was reached.

### 4. Removing what the agent made obsolete is cheap and mostly safe

`stale` needs no judge. It was cheaper than `off` on 7 of 8 chains (the eighth is the
one where it crossed 100K), resolved 58 against 55, and made 5 compactions in 80
issues; the break-even rule declined 29 more. The saving is small: obsolete outputs
in a Haiku session are few and small, because Haiku reads files narrowly.

### 5. Nothing was asked back

In 480 issue runs the agent never called `expand` or `recall`, and edited text that
was hidden from its request once (`jevh`). Recoverability was never exercised, so
these runs say nothing for or against it. This matches other work: agents rarely
recall on their own.

## What this does not show

- One seed, 8 sessions, one model. The resolved differences are noise-sized, and the
  cost intervals of every arm but `mask` include 1.
- Haiku as the judge, not Jev. Jev's cost is estimated from the repository's account of
  its billing, not metered. The `compact` judge's extra elisions were not checked against
  what the agent later used.
- The cache-rate scaling past 100K tokens is assumed.
- Django only, and short issues: about 7 requests per issue.

## Reproduce

```bash
export ANTHROPIC_API_KEY=... JEVCTX_JUDGE=llm JUDGE_BASE_URL=https://api.anthropic.com/v1 \
       JUDGE_MODEL=claude-haiku-5-5 JUDGE_API_KEY=$ANTHROPIC_API_KEY JUDGE_JSON_MODE=0 \
       JUDGE_TEMPERATURE=none
python bench/haiku/run.py --scratch $SCRATCH --out runs/haiku-main --arms off,mask,stale,staleh \
    --versions 4.0,4.1,4.2 --difficulty "<15 min fix,15 min - 1 hour,1-4 hours" \
    --chain 10 --chains 8 --seed 0 --workers 16
python bench/haiku/run.py --scratch $SCRATCH --out runs/haiku-main --arms jev,jevh \
    --chains-file runs/haiku-main/chains.json --thresholds bench/haiku/thresholds.json
python bench/haiku/analyze.py runs/haiku-main
```

`$SCRATCH` needs `data/verified.parquet` (SWE-bench Verified), a bare django clone at
`repos/django.git`, and virtualenvs `envs/dj3` (Python 3.8) and `envs/dj4`
(Python 3.10) with `asgiref sqlparse pytz tzdata jinja2 Pillow PyYAML numpy docutils
tblib argon2-cffi bcrypt`. The whole sweep cost about $5 in API calls; the judge replay
and the two `jevh-*` arms about $2 more.

The judge comparison: `python bench/haiku/judge_replay.py runs/haiku-main --out runs/judge-replay`,
`python bench/haiku/judge_agreement.py runs/judge-replay`, and `run.py ... --arms jevh
--judge flat --label flat` (and `--judge cached --label cached`).
