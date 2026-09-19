"""Regenerate the numbers embedded in docs/showcase.html.

    python tools/build_showcase_data.py > data.json

Every figure on the showcase page comes from here, so none of them are
hand-copied and all of them can be checked.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import demo  # noqa: E402
from jevctx import (  # noqa: E402
    DEFAULT_COMMIT_POLICY,
    CacheLedger,
    ContextBuffer,
    FakeJevClient,
    InMemoryStore,
    Origin,
    ShadowLog,
    admit,
    estimate_tokens,
    expand,
    make_block,
    score_items,
    segment,
)
from jevctx.types import (  # noqa: E402
    PRICE_PER_INPUT_TOKEN,
    Noul,
    ScoreItem,
    TurnSignals,
)

out = {}

# --- Act 1: the cost rule -------------------------------------------------- #
corpus = [ScoreItem(id=f"s:{i}", text=f"log line {i}: " + "payload " * 30,
                    tokens=estimate_tokens(f"log line {i}: " + "payload " * 30))
          for i in range(120)]
probe = FakeJevClient.constant(0.5)
score_items(probe, "t", corpus, Noul("Needed?", "yes", "no"))
corpus_tokens = sum(i.tokens for i in corpus)
actual = sum(estimate_tokens(c.state) for c in probe.calls)
out["cost"] = {
    "items": len(corpus), "corpusTokens": corpus_tokens,
    "requests": len(probe.calls),
    "chunked": actual, "naive": len(probe.calls) * corpus_tokens,
    "pricePerToken": PRICE_PER_INPUT_TOKEN,
}

# --- Act 2: buffer over turns ---------------------------------------------- #
client = FakeJevClient.by_text(demo.stand_in_judgement)
store, log = InMemoryStore(), ShadowLog(path=None)
buffer, ledger = ContextBuffer(), CacheLedger()
buffer.append_work(make_block("system", "You are a build assistant.\n"))
buffer.commit(reason="system prompt is stable")
anchor = buffer.render()[: buffer.cache_breakpoint]
turns = []
for turn in range(1, 9):
    res = admit(demo.build_log(turn), Origin("tool:bash", f"npm-{turn}", turn),
                task_digest=demo.TASK, turn=turn, client=client, store=store, log=log)
    buffer.append_work(make_block("tool", res.text, turn=turn))
    d = DEFAULT_COMMIT_POLICY.should_commit(buffer, TurnSignals(turn=turn, tool_depth=0))
    if d.commit:
        buffer.commit(d.upto, reason=d.reason)
    st = buffer.stats()
    turns.append({"turn": turn, "frozen": st.frozen_tokens, "work": st.work_tokens,
                  "committed": bool(d.commit),
                  "intact": buffer.render()[:len(anchor)] == anchor})
out["turns"] = turns
out["breakeven"] = [{"p": 100000, "pp": pp, "k": round(ledger.breakeven_turns(100000, pp), 2)}
                    for pp in (10000, 20000, 33000, 50000, 70000)]

# --- Act 3 + 4: real segments, scores, and the gate's output --------------- #
store2, log2 = InMemoryStore(), ShadowLog(path=None)
raw = demo.build_log(1)
res = admit(raw, Origin("tool:bash", "npm install", 1), task_digest=demo.TASK,
            turn=1, client=client, store=store2, log=log2)
segs = segment(raw, Origin("tool:bash", "npm install", 1))
scores = {s.id: demo.stand_in_judgement(s.text) for s in segs}
out["gate"] = {
    "originalTokens": res.original_tokens, "resultTokens": res.result_tokens,
    "output": res.text,
    "segments": [
        {"id": s.id, "kind": s.kind, "tokens": s.tokens, "score": scores[s.id],
         "lines": list(s.line_span),
         "preview": s.text.strip().splitlines()[0][:78],
         "lineCount": len(s.text.rstrip("\n").split("\n"))}
        for s in segs
    ],
}

# --- Act 4: the shadow log across 3 turns, with real expands --------------- #
store3, log3 = InMemoryStore(), ShadowLog(path=None)
for turn in range(1, 4):
    admit(demo.build_log(turn), Origin("tool:bash", f"npm-{turn}", turn),
          task_digest=demo.TASK, turn=turn, client=client, store=store3, log=log3)
for rec in store3.all_records():
    if "notice" in rec.text:
        expand(rec.id, store=store3, log=log3, turn=9)

decisions = [e for e in log3.entries() if e["type"] == "decision"]
expanded = {e["item_id"] for e in log3.entries()
            if e["type"] == "outcome" and e["kind"] == "expand"}
out["log"] = {
    "liveThreshold": 0.35,
    "liveFalseNegativeRate": log3.false_negative_rate(),
    "decisions": [
        {"id": d["item_id"], "score": d["score"], "tokens": d["tokens"],
         "preview": d["text_preview"].strip().splitlines()[0][:70],
         "expanded": d["item_id"] in expanded}
        for d in decisions
    ],
}

print(json.dumps(out, separators=(",", ":")))
