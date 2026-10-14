"""Turn shadow-run traces into calibration samples: was each scored segment used later?

A segment counts as used when the agent later edited text that came from it: at least
two of its non-blank lines (or one of 30+ characters) appear in a later edit's
``oldString``, compared without the ``N: `` line-number prefix the read tool adds.
That is the label the SWE-bench calibration used (:mod:`jevctx.calibrate`).

    python bench/haiku/label.py runs/haiku-cal --out samples.jsonl
    python -m jevctx.calibrate samples.jsonl --max-loss 0.1 --out thresholds.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_NUMBERED = re.compile(r"^\s*\d+[:\-]\s?")


def _lines(text: str) -> list[str]:
    found = []
    for raw in text.splitlines():
        line = _NUMBERED.sub("", raw).strip()
        if line:
            found.append(line)
    return found


def edits_of(messages: list[dict]) -> list[tuple[int, str]]:
    """(assistant message index, oldString) for every edit call, in order."""
    found = []
    for position, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        for block in message["content"]:
            if block.get("type") == "tool_use" and block.get("name") == "edit":
                old = (block.get("input") or {}).get("oldString")
                if isinstance(old, str):
                    found.append((position, old))
    return found


def call_positions(messages: list[dict]) -> dict[str, int]:
    found = {}
    for position, message in enumerate(messages):
        if message["role"] == "assistant":
            for block in message["content"]:
                if block.get("type") == "tool_use":
                    found[block["id"]] = position
    return found


def used(segment: str, later_edits: list[str]) -> bool:
    lines = _lines(segment)
    if not lines:
        return False
    for old in later_edits:
        edited = {line for line in _lines(old)}
        hits = [line for line in lines if line in edited]
        if len(hits) >= 2 or any(len(h) >= 30 for h in hits):
            return True
    return False


def samples_for(run_dir: Path) -> list[dict]:
    trace = run_dir / "trace.jsonl"
    messages_path = run_dir / "messages.json"
    if not trace.exists() or not messages_path.exists():
        return []
    messages = json.loads(messages_path.read_text())
    edits = edits_of(messages)
    positions = call_positions(messages)
    found = []
    for line in trace.read_text().splitlines():
        event = json.loads(line)
        sample = event.get("sample")
        if not sample or sample.get("failed") or sample.get("score") is None:
            continue
        at = positions.get(sample["call_id"], -1)
        later = [old for position, old in edits if position > at]
        found.append({"source": sample["source"], "kind": sample["kind"], "score": sample["score"],
                      "tokens": sample["tokens"], "used": used(sample["text"], later),
                      "run": run_dir.name})
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="a run directory written by bench/haiku/run.py")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    samples = []
    for run_dir in sorted(p for p in (args.root / "off").iterdir() if p.is_dir()):
        samples += samples_for(run_dir)
    args.out.write_text("".join(json.dumps(s) + "\n" for s in samples))
    print(f"{len(samples)} samples, {sum(s['used'] for s in samples)} used, "
          f"{sum(s['tokens'] for s in samples)} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
