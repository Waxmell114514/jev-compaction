# awesome-jev-compaction

**Using [Jev](https://docs.typesafe.ai) to manage an agent's memory and keep its context small.**

**用 [Jev](https://docs.typesafe.ai) 管理 agent 的记忆，控制 context 的大小。**

This is a demo, not a library. You already know what Jev is. This repo is about *what to
do with it* — one specific set of ideas, implemented and runnable.

这是一个 demo，不是库。你已经知道 Jev 是什么了，这个仓库讲的是**拿它来做什么**——一套具体的想法，写成了能跑的代码。

```bash
git clone https://github.com/Waxmell114514/awesome-jev-compaction
cd awesome-jev-compaction
uv venv .venv && uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python demo.py      # runs offline, no API key needed
```

Or open [`docs/showcase.html`](docs/showcase.html) — the same tour as a web page, with a
slider you can drag. / 或者打开 [`docs/showcase.html`](docs/showcase.html)，同样的内容做成了网页，带一个可以拖的滑块。

---

## The problem · 问题

An agent's context fills up. The usual fix is **compaction**: when it gets too long, ask a
model to summarise the old parts and replace them.

Two things go wrong. First, rewriting the middle of the context **breaks the prompt cache**,
so every later turn pays full price again. Second, a summary is *generated* text — the model
can quietly invent a detail, and now that invented detail is your agent's memory.

agent 的 context 会被填满。常见做法是**压缩**：太长了就让模型总结一下旧内容，替换掉。

这里有两个问题。第一，改写 context 中间的部分会**让 prompt cache 失效**，后面每一轮都要重新按全价付一次。第二，总结是**生成**出来的文本——模型可能悄悄编造一个细节，而这个细节就成了 agent 的记忆。

---

## Our idea · 我们的想法

Three parts. Each one is simple on its own.

三点，每一点单独看都很简单。

### 1. Filter at the door, not afterwards · 在入口过滤，而不是事后压缩

Split the context into two pieces: a **frozen part** at the front, and a **work area** at the
end. The frozen part only ever grows — nothing in it is ever rewritten. So the cache over it
never breaks.

New tool output goes into the work area. Before it goes in, ask Jev whether it's worth
keeping. Low-scoring parts never enter.

把 context 分成两块：前面是**冻结区**，后面是**工作区**。冻结区只增不改，所以它上面的缓存永远不会失效。

新的工具输出先进工作区。进去之前，先问 Jev 这段值不值得留。分低的根本不进来。

### 2. Move it out, don't delete it · 搬走，而不是删掉

This is the part that makes the rest safe.

What the gate removes doesn't disappear. It goes into a store, and leaves one line behind:

这一点是让其余部分变安全的关键。

被过滤掉的内容不会消失。它会进入一个外部存储，并在原地留下一行：

```
[[elided id=r:8506e122 lines=1-21 tokens=364 "npm http fetch GET 200 https://..."]]
```

The agent gets an `expand` tool. If it needs what's behind that line, it asks for it and gets
the **original text, byte for byte**. So a wrong decision costs one extra round trip — not
lost information.

Because Jev can't write text, what you keep is always the original lines. Nothing in memory
was ever made up.

agent 手上有一个 `expand` 工具。如果它需要那一行背后的东西，要一下就能拿到**一字不差的原文**。所以判断错了，代价是多一次往返——而不是信息丢失。

因为 Jev 不会生成文本，留下来的永远是原始的行。记忆里不会出现任何编造的内容。

### 3. Record every decision, so you can tune it later · 把每次判断记下来，之后再调

Every decision is logged with its score. When the agent later calls `expand` on something you
removed, that's **proof the filter was too aggressive** — it went looking for exactly what you
took away.

Later, replay the log at a different threshold. No re-running the agent, no extra Jev calls:

每次判断都会连同分数记录下来。如果 agent 后来对某个被移走的内容调用了 `expand`，那就是**过滤太狠的铁证**——它正好在找你拿掉的那个东西。

之后拿这份日志换个阈值重放一遍就行。不用重跑 agent，也不用额外调 Jev：

```
  threshold    kept   relocated   tokens saved   still missed
  0.10           12           3          1,092              0
  0.35            9           6          1,260              3
```

`0.10` saves 1,092 tokens and takes away nothing the agent came back for. `0.35` saves 168
more tokens and costs three. Now you can choose.

`0.10` 省下 1,092 个 token，而且没拿走任何 agent 回头要过的东西。`0.35` 多省 168 个 token，代价是三次。现在你可以做选择了。

---

## Why Jev makes this work · 为什么 Jev 能做到

| Jev | What it lets you do · 能做什么 |
|---|---|
| 70–500 ms | Fast enough to run on **every tool call**, not just at compaction time.<br>快到可以在**每次工具调用**时跑，而不只在压缩时跑。 |
| Can't write text<br>不会生成文本 | Compaction becomes extraction. Memory can't contain a hallucination.<br>压缩变成抽取，记忆里不可能出现幻觉。 |
| Typed output<br>结构化输出 | Decisions are data. Your code acts on them; no parsing a model's prose.<br>判断结果就是数据，代码直接用，不用解析模型的话。 |
| Input-only billing<br>只按输入计费 | Asking 32 questions costs the same as asking 1.<br>问 32 个问题和问 1 个问题一样贵。 |

That last row leads to the **one rule** worth remembering:

最后一行引出了唯一一条值得记住的规则：

> **Many questions, one `state`.** Never send the same `state` once per item.
>
> **多个问题共享一份 `state`。** 绝不要为每个条目重复发送同一份 `state`。

Scoring 120 log segments, done right vs. done wrong:

给 120 段日志打分，正确做法 vs. 错误做法：

```
  chunk in state (correct)          10,252 tok   $0.000431
  whole corpus in every state       35,040 tok   $0.001472
  → 3.4x cheaper, exact same answers
```

Both give identical answers. One costs 3.4× more. This is easy to get wrong and hard to
notice, so the repo has a test that checks every request carries exactly the items its
questions ask about.

两种做法答案完全一样，一种贵 3.4 倍。这很容易写错而且不容易发现，所以仓库里有一个测试专门检查：每个请求携带的内容，必须恰好是它的问题要问的那些。

---

## Using the repo · 怎么用这个仓库

Start with the demo. It runs offline against a scripted stand-in, so you don't need a key:

先跑 demo。它用一个脚本替身离线运行，不需要 key：

```bash
.venv/bin/python demo.py                  # six short acts / 六小节
TYPESAFE_API_KEY=sk-... .venv/bin/python demo.py   # same code, real Jev / 同样的代码，真实的 Jev
```

### Connecting a real key · 接入真实的 key

Set `TYPESAFE_API_KEY` and everything switches to real Jev — `demo.py` picks it up, and
`HttpJevClient()` reads it by default. To check the key actually works before wiring it into
an agent:

设置 `TYPESAFE_API_KEY`，所有东西就会切到真实的 Jev——`demo.py` 会自动识别，`HttpJevClient()` 默认读它。在接进 agent 之前，先确认 key 真的能用：

```bash
export TYPESAFE_API_KEY=...
.venv/bin/python -m jevctx.check
```

That makes real requests and checks three things: all three question types come back and
parse, the gate actually relocates something from a real tool output, and the pointer expands
back byte for byte. It prints latency, token usage and cost, and on failure it tells you
which of the three broke — a bad key, a malformed request, or an unreachable server.

它会发真实请求，检查三件事：三种问题类型都能返回并解析、gate 确实从真实工具输出里搬走了东西、指针能一字不差地展开回来。它会打印延迟、token 用量和花费；失败时会告诉你是哪一环出了问题——key 不对、请求格式错、还是连不上。

Then read these, in this order / 然后按这个顺序读：

| File | What's in it · 里面是什么 |
|---|---|
| [`demo.py`](demo.py) | The tour. Start here. · 导览，从这里开始 |
| `jevctx/pipeline.py` | `admit()` / `retrieve()` / `expand()` — the ~200 lines that matter<br>真正干活的那 200 行 |
| `jevctx/scorer.py` | How 32 questions get packed into one request · 怎么把 32 个问题塞进一个请求 |
| `jevctx/context.py` | The frozen-prefix buffer · 冻结区 + 工作区 |
| `jevctx/segments.py` | Splitting output without breaking it · 切分输出而不破坏它 |
| [`SPEC.md`](SPEC.md) | The full reasoning · 完整的推理过程 |

To use it in your own agent / 用在自己的 agent 里：

```python
from jevctx import admit, expand, InMemoryStore, ShadowLog, HttpJevClient, Origin, GateConfig

store, log, client = InMemoryStore(), ShadowLog("shadow.jsonl"), HttpJevClient()

result = admit(tool_output, Origin(source="tool:bash", ref="npm install", turn=1),
               task_digest="What the agent is currently doing.",
               turn=1, client=client, store=store, log=log,
               config=GateConfig(shadow_only=True))   # start here / 从这里开始

# result.text goes into your context. Register `expand` as a tool.
# result.text 放进 context。把 `expand` 注册成一个工具。
```

**Start with `shadow_only=True`.** It scores and logs everything but changes nothing. Run it
for a day, replay the log to pick a threshold, then turn it on. Turning a context filter
straight on is how you lose a week to "the agent got worse and nobody knows when."

**一开始请用 `shadow_only=True`。** 它会打分、记录，但什么都不改。跑一天，用日志重放选一个阈值，再打开。直接开过滤，就等着花一周排查"agent 变笨了但不知道什么时候开始的"。

---

## When things break · 出问题的时候

Every failure mode costs tokens, never information.

每一种失败模式的代价都是 token，而不是信息。

- **Jev is down** → everything scores 1.0, text passes through untouched.<br>**Jev 挂了** → 所有内容打 1.0 分，文本原样通过。
- **The scorer wants to drop 90% of an output** → keep all of it. A scorer that wants to drop
  almost everything is telling you the *question* is bad, not the output.<br>**打分器想删掉 90%** → 全部保留。想删掉几乎所有东西的打分器，说明是**问题**写错了，不是输出没用。
- **A stack trace scores low** → kept anyway. Half a stack trace isn't a stack trace.<br>**堆栈分数很低** → 照样保留。半个堆栈不是堆栈。

---

## What's not built · 没做的部分

Four more ideas follow from the same primitive. None are implemented — each needs data that
doesn't exist on day one, so the hooks are in place and the collection is already running.

还有四个想法从同一个原语延伸出来，都**没有实现**——每一个都需要第一天还不存在的数据，所以钩子留好了，数据也已经在采集了。

- Jev-driven commit points — asking "is this subtask finished?"<br>用 Jev 判断提交点——问"这个子任务结束了吗"
- Multi-dimensional labelling — type, lifetime, entities, in one call of ~15 questions<br>多维标注——类型、生命周期、涉及实体，一次调用问 15 个问题
- Supersession chains — does this new record replace that old one?<br>取代关系链——这条新记录是否取代了那条旧的？
- Bayesian threshold calibration — combine Jev's score with observed hit counts<br>贝叶斯阈值校准——把 Jev 的分数和实际命中次数结合起来

---

## Two honest notes · 两点说明

**Jev is not a security boundary.** Typed output means Jev itself can't be tricked into
emitting instructions — genuinely useful. But its *judgement* can still be swayed by the text
it's judging. Keep your tool allowlists and approval gates.

**Jev 不是安全边界。** 结构化输出意味着 Jev 本身不会被诱导去发出指令——这确实有用。但它的**判断**仍然可能被它正在判断的文本影响。工具白名单和人工确认闸门不能撤。

**The demo's scores aren't Jev's.** Without an API key, scores come from a hand-written lookup
table, labelled as such in the output. The code path is identical. Everything else —
token counts, segment boundaries, buffer state, the replay arithmetic — is real output from
the implementation.

**demo 里的分数不是 Jev 给的。** 没有 API key 时，分数来自一个手写的查找表，输出里有明确标注。代码路径完全一样。其余所有东西——token 数、分段边界、缓冲区状态、重放计算——都是实现的真实输出。
