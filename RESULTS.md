# Results on SWE-bench Verified

What each mechanism did, measured on real agent runs. The headline: **Jev keeps a
coding agent's context clean without costing it solves.** It moves the bill only a
little, because the bill is set by how many turns a task takes and by prompt-cache
prices, not by how much tool output sits in the context.

## Setup

| | |
|---|---|
| Harness | [OpenCode](https://opencode.ai) 1.18 with [the plugin](integrations/opencode); an earlier run used [Pi](https://github.com/earendil-works/pi) with the first gate |
| Model | `mimo-v2.6-flash-free` via OpenCode Zen (free); `deepseek-v4-flash` on Pi |
| Tasks | a fixed sample of SWE-bench Verified: 21–24 instances per comparison on OpenCode, 35 on Pi |
| Isolation, grading | each run in the instance's own container on an internal network, egress to the model provider only; SWE-bench's own grader, offline |
| Control | a `shadow` arm: the plugin scores everything and changes nothing (A/A) |
| Statistics | geometric mean of paired per-instance ratios, 95% bootstrap interval; one run per arm |

The model was free, so dollar figures price the recorded token counts at a
hypothetical $3 input, $0.30 cache read and $15 output per million tokens, plus Jev
at $0.042 per million. Two things to know when reading them. Run-to-run noise is
large: the A/A control sits at 1.25× plain OpenCode on prompt tokens. And headless
OpenCode ends a session when it auto-rejects a call outside the work tree, which
cut short 1–4 runs in every arm; comparisons are also given without those.

## Admission: the gate

| | first gate (keep score) | profiled gate (`role: change_site`) |
|---|---:|---:|
| tool output in context vs control | **0.83 [0.68, 1.02]** | 0.89 [0.71, 1.13] |
| prompt tokens vs control | 1.05 [0.78, 1.39] | 1.12 [0.81, 1.56] |
| resolved vs control | 20/24 vs 21/24 | 20/23 vs 20/23 |
| *on the 21 instances with every arm:* | | |
| tool output removed at admission | 27% | 26% |
| net of re-reads and `expand` | 80% of that | 74% of that |
| elided records holding code the agent later edited | 21 | **11** |
| `expand` calls | 9 | 29 |

On Pi the first gate gave the same picture: tool output 0.81–0.83 of the control's,
prompt tokens and cost within noise, resolve rate unchanged.

**The profile.** One Jev request asks each segment five questions: keep it, what
type, what role, how long relevant, injection? Gating on *what a segment is for*
spares the code a fix touches. It is visible offline too, replaying 36 recorded
runs:

| gate signal | tokens elided | edited segments lost |
|---|---:|---:|
| keep, fixed threshold 0.35 | 33% | 18% |
| keep, thresholds fitted per tool | 26% | 14% |
| **role: change_site, fitted per tool** | 24% | **9%** |

**Injection.** Of 2,499 segments of real repository output, none was flagged. Of
18 planted instructions, 17 were quarantined, and the pointer does not quote them.

## Getting it back: `expand` and `recall`

`expand` returns the text behind a pointer byte for byte. `recall` finds earlier
output from a description:
1. filter on the profile (type, role, names mentioned);
2. shortlist lexically;
3. Jev reranks the shortlist in one request.

Offline, with about 200 stored outputs:

| query | recall, top 3 | BM25 alone |
|---|---:|---:|
| the agent's own words | **80%** | 49% |
| with every identifier removed | **69%** | 18% |

Live, SWE-bench tasks are short, about 25 turns, and `recall` was used about once
per 20 runs. Long sessions are where it should matter, and this benchmark does
not have them.

## The work area: compacting later

Before each request, outputs in the transcript's tail that have served their
purpose become pointers. A rewrite breaks the provider's prompt cache from that
point on, so it happens only when

    dropped tokens × turns left × cache price  >  rest of the tail × (input price − cache price)

The OpenCode plugin applies it to the request only; the stored session is never
rewritten.

| | `work` vs control | `work` vs the gate alone |
|---|---:|---:|
| tokens per request | **0.83 [0.68, 1.01]** | **0.84 [0.73, 0.95]** |
| cost at the price sheet, comparable runs | 0.92 [0.72, 1.18] | 0.90 [0.74, 1.08] |
| total cost, comparable runs | $8.72 vs $9.74 (−10%) | $10.34 vs $11.50 (−10%) |
| provider cache-hit share | 93% vs 95% | |

It compacted 95 outputs and never expanded one back. In the 5 cases where the agent
later edited code a compacted output had shown, another output still in context
showed the same lines.

## Supersession: what a later call made obsolete

No Jev call is involved. The relations come from each tool call's arguments:
- **superseded**: the same command run again, or the same lines viewed again;
- **stale**: the file was written since.

On 28 recorded runs:
- **21% of all tool-output tokens went stale** through the agent's own edits.
- The agent rarely edited from an out-of-date copy: 2 of 92 stale outputs.
- Exact reruns were rare, with 9 outputs superseded.

`expand` and `recall` now say when an output is out of date. The work area compacts
such outputs without asking Jev, including in the frozen prefix.

| work-area replay, 28 runs, $3 / $0.30 | net saving |
|---|---:|
| without supersession | 4.0% |
| with supersession | 5.1% |

Each figure is the mean of two replays. With supersession, 10 runs were cheaper,
6 dearer and 12 the same.

A live rerun of the work arm with supersession and a more cautious turns-left
estimate (`work2`) resolved 21/24, the same as the gate alone. It recorded 115
relations. The cautious estimate declined twice as many rewrites, so requests were
only 6% smaller than the control's, within noise. **How long a session will still
run is what decides whether a rewrite pays**, and it is estimated from a fixed
prior, not from the task.

## Goal conditioning: judging against the call's intent (offline)

`admit(..., intent=...)` also tells the judge what the agent was looking for when
it made the call. Scoring was replayed with Jev on 35 recorded Pi runs
(`deepseek-v4-flash`, shadow arm). This covers 2,742 segments, 152 of them code the
agent later edited. Each segment was scored four times: without an intent; with the
call itself (`bash: grep -rn …`); with the model's words before the call (its text or
thinking, as `run_agent(intent="reply")` sends); and with the words followed by the
call.

| intent | AUC, later-edited code | edited segments lost at 25% / 40% of tokens elided |
|---|---:|---:|
| none | 0.793 | 15.8% / 22.4% |
| the call | 0.797 | 17.1% / 25.7% |
| the model's words | 0.780 | 11.2% / 23.0% |
| words, then the call | 0.807 | 13.2% / 22.4% |
| *profiled gate, `role: change_site`:* | | |
| none | 0.799 | 6.6% / 13.2% |
| words, then the call | 0.793 | 5.9% / 12.5% |

**No condition beat no intent at 95%.** Words followed by the call did best, at
ΔAUC +0.014 [−0.009, +0.042], with a paired bootstrap over runs. The profiled gate
is still much the better signal whether or not an intent is given. Two caveats:
- This label asks whether the task needed a segment later, not whether it answered
  the call. Whatever intent saves in turns, by reading less beside the point, needs
  a live run to show.
- Intent is only as available as the model's narration. Pi's `deepseek-v4-flash`
  wrote or thought something before 71% of calls. OpenCode's recorded events show
  text before 4% of calls, and those recordings carry no reasoning.

## Other judges

`jevctx.check` passes end to end with each of the following:

| judge | question-types request | gate request |
|---|---:|---:|
| Jev on TypeSafe | 0.6 s | 0.16 s |
| Jev on OpenRouter | passes | passes |
| `z-ai/glm-5.2:free` on OpenRouter | 14 s | 18 s |
| `qwen/qwen3.8-27b:free` on OpenRouter, without JSON mode | 14 s | 65 s |

On the check's sample, all four kept the error and relocated the progress lines.
Free models are rate-limited to a shared pool, and on a recorded SWE-bench run
`glm-5.2:free` failed 55 of 86 segments. A quality comparison with Jev therefore
needs a paid model; one has not been run.

## What is not established

- **Resolve-rate effects of a point or two.** With one run per arm and harness
  endings in every arm, differences of 1–4 instances are noise.
- **The value of `recall` and the out-of-date notes in long sessions.** SWE-bench
  runs are too short to need them.
- **Cost on a paid model.** Every dollar figure here is a price sheet applied to a
  free model's token counts.
- **What goal conditioning does live.** Whether judging against the call's intent
  cuts turns, as SWE-Pruner's goal-conditioned pruning did, has not been run.
- **How other judges compare with Jev.** They run end to end, but only Jev has been
  measured on SWE-bench.
