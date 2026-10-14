"""Replay recorded sessions' admission scoring through several judge layouts at once.

Every tool output of the ``off`` arm's sessions goes through :func:`jevctx.pipeline.admit`
(profiled, shadow-only) in the order it arrived, under the task the agent was working on.
Each judge request is sent, unchanged, to every judge: ``flat`` (as ``LLMJudgeClient``
lays it out), a second ``flat`` (to measure how much a judge disagrees with itself),
``cached`` and ``compact`` (:mod:`bench.haiku.judges`). Metered usage is recorded per
judge, and every answer per question, so the layouts can be compared on cost and on
what they would have decided.

    python bench/haiku/judge_replay.py runs/haiku-main --out runs/judge-replay
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.haiku.judges import AnthropicJudge  # noqa: E402
from jevctx.pipeline import GateConfig, admit  # noqa: E402
from jevctx.shadow import ShadowLog  # noqa: E402
from jevctx.store import InMemoryStore  # noqa: E402
from jevctx.types import Answer, JevError, Origin, Question, State  # noqa: E402

JUDGES = ("flat", "flat2", "cached", "compact")


def _plain(answer: Answer) -> object:
    if answer.type == "noul":
        return answer.noul
    return dict(answer.probabilities)


class Tee:
    """Send each request to every judge; answer with the first's."""

    def __init__(self, judges: Mapping[str, AnthropicJudge], sink) -> None:
        self.judges, self.sink = judges, sink
        self.usage = None

    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        found: dict[str, dict[str, Answer] | None] = {}

        def one(name: str) -> None:
            try:
                found[name] = self.judges[name].ask(state, questions)
            except JevError:
                found[name] = None

        threads = [threading.Thread(target=one, args=(n,)) for n in self.judges]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.sink({"keys": list(questions),
                   **{n: ({k: _plain(a) for k, a in r.items()} if r else None)
                      for n, r in found.items()}})
        first = found[next(iter(self.judges))]
        if first is None:
            raise JevError("primary judge failed")
        return first


def replay(session_dir: Path, out: Path) -> dict:
    messages = json.loads((session_dir / "messages.json").read_text())
    judges = {name: AnthropicJudge("flat" if name.startswith("flat") else name) for name in JUDGES}
    lock = threading.Lock()
    sink_file = (out / f"{session_dir.name}.jsonl").open("w")

    def sink(event: dict) -> None:
        with lock:
            sink_file.write(json.dumps(event) + "\n")
            sink_file.flush()

    tee = Tee(judges, sink)
    calls: dict[str, tuple[str, dict]] = {}
    task, turn, outputs = "", 0, 0
    config = GateConfig(profile=True, gate_on="role:change_site", shadow_only=True)
    for message in messages:
        for block in message["content"]:
            if message["role"] == "assistant" and block.get("type") == "tool_use":
                calls[block["id"]] = (block["name"], block.get("input") or {})
            elif message["role"] == "user" and block.get("type") == "text" and "<issue>" in block["text"]:
                task = block["text"]
            elif block.get("type") == "tool_result" and not block.get("is_error"):
                turn += 1
                name = calls.get(block["tool_use_id"], ("?", {}))[0]
                content = block["content"]
                if name in {"expand", "recall"} or not isinstance(content, str):
                    continue
                outputs += 1
                try:
                    admit(content, Origin(source=f"tool:{name}", ref=block["tool_use_id"], turn=turn),
                          task_digest=task, turn=turn, client=tee, store=InMemoryStore(),
                          log=ShadowLog(), config=config)
                except JevError:
                    pass
    sink_file.close()
    return {"session": session_dir.name, "outputs": outputs,
            **{name: j.usage.snapshot() for name, j in judges.items()}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    sessions = sorted(p for p in (args.root / "off").iterdir() if (p / "messages.json").exists())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda s: replay(s, args.out), sessions))
    (args.out / "usage.json").write_text(json.dumps(results, indent=1))
    for name in JUDGES:
        total = {k: sum(r[name][k] for r in results) for k in results[0][name]}
        print(f"{name:8} requests={total['requests']:5} input={total['input_tokens']:>9} "
              f"uncached={total['uncached']:>9} cache_read={total['cache_read']:>9} "
              f"cache_write={total['cache_write']:>8} output={total['output_tokens']:>8} "
              f"failures={total['failures']:3} ${total['cost_usd']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
