#!/usr/bin/env python3
"""A runnable tour of using Jev for agent memory and context compaction.

    python demo.py

Runs offline against a scripted stand-in for Jev, so it works with no API key.
Set TYPESAFE_API_KEY to run the exact same code against the real model.

Each act shows one idea. The interesting ones are Act 1 (why the batching shape
matters more than anything else) and Act 4 (why you can afford to measure whether
your threshold is right).
"""

from __future__ import annotations

import os
import sys

from jevctx import (
    DEFAULT_COMMIT_POLICY,
    EXPAND_TOOL_SCHEMA,
    CacheLedger,
    ContextBuffer,
    FakeJevClient,
    GateConfig,
    HttpJevClient,
    InMemoryStore,
    Origin,
    ShadowLog,
    admit,
    estimate_tokens,
    expand,
    make_block,
    reconstruct,
    score_items,
)
from jevctx.types import (
    PRICE_PER_INPUT_TOKEN,
    JevUnavailableError,
    Noul,
    ScoreItem,
    TurnSignals,
)

TASK = "The build fails on a dependency conflict. Find which package is pinned wrong."

WIDTH = 74


def rule(title: str = "") -> None:
    if title:
        print(f"\n\033[1m{title}\033[0m")
        print("─" * WIDTH)
    else:
        print("─" * WIDTH)


def note(text: str) -> None:
    print(f"\033[2m{text}\033[0m")


def usd(tokens: int) -> str:
    return f"${tokens * PRICE_PER_INPUT_TOKEN:.6f}"


def build_log(turn: int) -> str:
    """A tool result shaped like real npm output: stanzas of varying usefulness.

    Deliberately interleaved rather than sorted by usefulness. Adjacent low-scoring
    stanzas merge into a single stored record, and then an expand() cannot say which
    part of it the agent actually wanted -- a real limitation of merging runs.
    """
    stanzas = [
        "\n".join(f"npm http fetch GET 200 https://registry.npmjs.org/dep-{turn}-{i} "
                  f"{20 + i}ms" for i in range(20)),
        "\n".join(f"npm ERROR peer react@^18.0.0 required by ui-kit-{turn}-{i}, "
                  f"found 17.0.2" for i in range(12)),
        "\n".join(f"npm notice created a lockfile entry for dep-{turn}-{i}"
                  for i in range(4)),
        "\n".join(f"npm WARN deprecated glob@7.2.{i}: no longer supported"
                  for i in range(6)),
        f"added 412 packages, audited 1204 packages in {8 + turn}s",
    ]
    return "\n\n".join(stanzas) + "\n"


def stand_in_judgement(text: str) -> float:
    """What a scripted stand-in guesses, so the demo has a realistic score spread.

    Real Jev returns a calibrated probability per item. This is a lookup table
    pretending to be one -- enough to show the shape of the decisions, and clearly
    labelled so nobody mistakes it for the model's actual judgement.
    """
    if "ERROR" in text:
        return 0.96
    if "added" in text and "packages" in text:
        return 0.72
    if "WARN deprecated" in text:
        return 0.38
    if "notice" in text:
        return 0.16
    return 0.02


def make_client():
    """The real model when a key is present, a scripted stand-in otherwise."""
    if os.environ.get("TYPESAFE_API_KEY"):
        print("\033[32m✓ TYPESAFE_API_KEY found — running against real Jev\033[0m")
        return HttpJevClient(), True
    print("\033[33m! No TYPESAFE_API_KEY — running against a scripted stand-in.\033[0m")
    note("  The judgements below are a hand-written heuristic, not Jev's. The code")
    note("  path is identical; only the client differs. Set the key to see the real")
    note("  thing, including real latency.")
    return FakeJevClient.by_text(stand_in_judgement), False


# --------------------------------------------------------------------------- #


def act1_the_cost_rule(client) -> None:
    rule("ACT 1 — Jev bills for state, not for questions")
    print("Jev caps a request at 32 questions and ~32k of state, and charges only")
    print("for input. So the shape that matters is: many questions over ONE state.")
    print()

    corpus = [
        ScoreItem(id=f"s:{i}", text=f"log line {i}: " + "payload " * 30,
                  tokens=estimate_tokens(f"log line {i}: " + "payload " * 30))
        for i in range(120)
    ]
    question = Noul(instructions="Will this be needed later?",
                    true="Needed later.", false="Noise.")

    probe = FakeJevClient.constant(0.5)
    score_items(probe, TASK, corpus, question)

    corpus_tokens = sum(i.tokens for i in corpus)
    actual = sum(estimate_tokens(call.state) for call in probe.calls)
    naive = len(probe.calls) * corpus_tokens

    print(f"  120 items, {corpus_tokens:,} tokens of text")
    print(f"  {len(probe.calls)} requests × ≤32 questions")
    print()
    print(f"  {'this shape (chunk in state)':<34} {actual:>7,} tok   {usd(actual)}")
    print(f"  {'naive (whole corpus each time)':<34} {naive:>7,} tok   {usd(naive)}")
    print(f"  \033[1m{naive / actual:.1f}x cheaper, same answers\033[0m")
    print()
    note("  This is the one mistake that costs real money, and it is invisible")
    note("  until you look at what you put in `state`. jevctx has a test that")
    note("  asserts every request's state holds exactly the items its questions")
    note("  ask about — no more, no fewer.")


def act2_the_staging_area(client) -> None:
    rule("ACT 2 — Compact the tail, never the prefix")
    print("Context is [frozen prefix] + [work area]. The prefix only ever grows, so")
    print("the KV cache over it is never invalidated. All churn happens in the tail.")
    print()

    store, log = InMemoryStore(), ShadowLog(path=None)
    buffer, ledger = ContextBuffer(), CacheLedger()

    buffer.append_work(make_block("system", "You are a build assistant.\n"))
    buffer.commit(reason="system prompt is stable")
    anchor = buffer.render()[: buffer.cache_breakpoint]

    print(f"  {'turn':<6}{'frozen':>9}{'work':>9}{'cached prefix':>16}{'committed':>12}")
    for turn in range(1, 7):
        result = admit(build_log(turn), Origin(source="tool:bash", ref=f"npm-{turn}",
                                               turn=turn),
                       task_digest=TASK, turn=turn, client=client, store=store, log=log)
        buffer.append_work(make_block("tool", result.text, turn=turn))

        decision = DEFAULT_COMMIT_POLICY.should_commit(
            buffer, TurnSignals(turn=turn, tool_depth=0, last_role="tool"))
        if decision.commit:
            buffer.commit(decision.upto, reason=decision.reason)

        stats = buffer.stats()
        ledger.record_render(turn=turn, frozen_tokens=stats.frozen_tokens,
                             work_tokens=stats.work_tokens, cache_written=decision.commit)
        intact = buffer.render()[: len(anchor)] == anchor
        mark = "\033[32mintact\033[0m" if intact else "\033[31mBROKEN\033[0m"
        print(f"  {turn:<6}{stats.frozen_tokens:>9,}{stats.work_tokens:>9,}"
              f"{mark:>25}{('yes' if decision.commit else '-'):>12}")

    print()
    print("  The prefix frozen on turn 0 is byte-identical six turns later.")
    note("  jevctx enforces this at runtime rather than trusting it: the buffer")
    note("  memoises what it last rendered and raises if a commit would change it.")
    print()
    print("  When is compacting worth it? At 0.1x cache reads and 1.25x writes,")
    print(f"  100k -> 20k pays back in {ledger.breakeven_turns(100_000, 20_000):.1f} turns.")
    note("  Which is the real lesson: cost almost always says 'compact'. So the")
    note("  question worth asking Jev at a commit point is not 'is it worth it'")
    note("  but 'is this subtask actually finished' — a safety question.")


def act3_relocation(client) -> None:
    rule("ACT 3 — Relocate, don't delete")
    print("Jev can't write text, which is a feature here: compaction becomes")
    print("extractive. Kept lines are verbatim originals, so memory never contains")
    print("a fact that was hallucinated into it.")
    print()

    store, log = InMemoryStore(), ShadowLog(path=None)
    raw = build_log(1)
    result = admit(raw, Origin(source="tool:bash", ref="npm install", turn=1),
                   task_digest=TASK, turn=1, client=client, store=store, log=log)

    print(f"  tool output   {result.original_tokens:>6,} tok")
    print(f"  after gate    {result.result_tokens:>6,} tok    "
          f"\033[1m({result.saved_tokens / result.original_tokens:.0%} smaller)\033[0m")
    print()
    for line in result.text.splitlines()[:4]:
        print(f"  \033[36m{line[:WIDTH - 4]}\033[0m" if line.startswith("[[elided")
              else f"  {line[:WIDTH - 4]}")
    print("  ...")
    print()
    print("  The gate removed nothing. It moved it and left a pointer:")
    recovered = expand(result.pointers[0].id, store=store, log=log, turn=2)
    print(f"  expand({result.pointers[0].id!r}) -> {len(recovered):,} chars")
    print(f"  full output reconstructs byte-exact: "
          f"\033[32m{reconstruct(result.text, store) == raw}\033[0m")
    print()
    note("  Register EXPAND_TOOL_SCHEMA with the agent's tools. A gate the agent")
    note(f"  cannot undo is not safe to turn on. Tool name: {EXPAND_TOOL_SCHEMA['name']!r}")


def act4_measuring_the_threshold(client) -> None:
    rule("ACT 4 — Cheap enough to measure whether you were right")
    print("Every gate decision is logged with its score. When the agent later calls")
    print("expand(), that is ground truth that the gate was too aggressive.")
    print()

    store, log = InMemoryStore(), ShadowLog(path=None)
    for turn in range(1, 4):
        admit(build_log(turn), Origin(source="tool:bash", ref=f"npm-{turn}", turn=turn),
              task_digest=TASK, turn=turn, client=client, store=store, log=log)

    # The agent goes back for the lockfile notices the gate removed.
    for record in store.all_records():
        if "notice" in record.text:
            expand(record.id, store=store, log=log, turn=9)

    print(f"  live threshold 0.35 → false-negative rate "
          f"\033[1m{log.false_negative_rate():.0%}\033[0m")
    print()
    print(f"  {'threshold':<12}{'kept':>8}{'elided':>9}{'tokens saved':>15}"
          f"{'still missed':>15}")
    for threshold in (0.0, 0.1, 0.35, 0.6, 0.9):
        stats = log.replay(threshold)
        print(f"  {threshold:<12.2f}{stats.by_action['kept']:>8}"
              f"{stats.by_action['elided']:>9}{stats.elided_tokens:>15,}"
              f"{stats.false_negatives:>15}")
    print()
    note("  Replayed from logged scores — no agent re-run, no extra Jev calls.")
    note("  The rightmost column is what you actually tune on: how many things the")
    note("  agent had to go back for would this threshold still have taken away.")


def act5_when_jev_is_down(client) -> None:
    rule("ACT 5 — Failing open")
    store, log = InMemoryStore(), ShadowLog(path=None)
    down = FakeJevClient.failing(JevUnavailableError("503 from upstream"))
    raw = build_log(1)

    result = admit(raw, Origin(source="tool:bash", ref="npm", turn=1), task_digest=TASK,
                   turn=1, client=down, store=store, log=log)

    print("  Jev unreachable → every item scores 1.0, text passes through untouched")
    print(f"  output unchanged: \033[32m{result.text == raw}\033[0m   "
          f"records written: {len(store)}")
    print()
    note("  An outage must degrade the agent's context to 'uncompacted', never to")
    note("  'silently missing things'. Same reason the gate relocates instead of")
    note("  deleting: every failure mode should cost tokens, not information.")


def act6_shadow_mode(client) -> None:
    rule("ACT 6 — How you'd actually roll this out")
    store, log = InMemoryStore(), ShadowLog(path=None)
    raw = build_log(1)
    result = admit(raw, Origin(source="tool:bash", ref="npm", turn=1), task_digest=TASK,
                   turn=1, client=client, store=store, log=log,
                   config=GateConfig(shadow_only=True))

    print("  GateConfig(shadow_only=True): score everything, log everything,")
    print("  change nothing.")
    print()
    print(f"  context modified: \033[32m{result.text != raw}\033[0m        "
          f"decisions logged: {log.stats().total}")
    print(f"  would have elided: {log.replay(0.35).elided_tokens:,} tok")
    print()
    note("  Run it here for a day, replay the log to pick a threshold, then turn")
    note("  it on. Shipping a context gate straight to on is how you lose a week")
    note("  to 'the agent got dumber and nobody knows when'.")


def main() -> int:
    print()
    print("\033[1m  jevctx — Jev for agent memory and context compaction\033[0m")
    print(f"  {'─' * (WIDTH - 2)}")
    client, _real = make_client()

    act1_the_cost_rule(client)
    act2_the_staging_area(client)
    act3_relocation(client)
    act4_measuring_the_threshold(client)
    act5_when_jev_is_down(client)
    act6_shadow_mode(client)

    rule()
    print("Read SPEC.md for the reasoning, jevctx/pipeline.py for the 200 lines")
    print("that do the work. Everything here runs offline, tests included.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
