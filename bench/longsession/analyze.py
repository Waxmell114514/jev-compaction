"""Summarise a long-session sweep: per arm, and per position in the chain.

    python bench/longsession/analyze.py runs/long --sidecar jev=runs/long/sidecar-jev \\
        --sidecar jev0=runs/long/sidecar-jev0 [--json out.json]

For each arm: issues resolved, overall and by position in the chain; OpenCode's own
compactions (LLM summaries) and whether issues after the first one fare worse; context
and cost; and, for plugin arms, what the gate and the work area removed, and how often
the agent reached back with ``expand`` or ``recall`` -- in particular to output from an
*earlier issue*, which only a long session can need.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_chains(root: Path) -> dict[str, dict[str, dict]]:
    """arm -> chain id -> chain.json (finished chains only)."""
    found: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(root.glob("*/*/chain.json")):
        chain = json.loads(path.read_text())
        found[chain["arm"]][chain["chain"]] = chain
    return found


def issue_of_turn(chain: dict, turn: int) -> int | None:
    """Which issue of the chain a sidecar turn number falls in (turns run on across issues)."""
    start = 0
    for issue in chain["issues"]:
        if start < turn <= start + issue["turns"]:
            return issue["index"]
        start += issue["turns"]
    return None


def memory_use(chain: dict, sidecar: Path) -> dict:
    """expand and recall calls, and how many reached output from an earlier issue."""
    folder = sidecar / chain["jev_session"]
    records = {}
    memory = folder / "memory.jsonl"
    if memory.exists():
        for line in memory.open(encoding="utf-8"):
            record = json.loads(line).get("record") or {}
            if "id" in record:
                records[record["id"]] = (record.get("origin") or {}).get("turn", 0)
    counts = {"expand": 0, "expand_earlier_issue": 0, "recall_hits": 0,
              "recall_hits_earlier_issue": 0}
    shadow = folder / "shadow.jsonl"
    if not shadow.exists():
        return counts
    for line in shadow.open(encoding="utf-8"):
        entry = json.loads(line)
        kind = entry.get("kind")
        if entry.get("type") == "outcome" and kind == "expand":
            key = "expand"
        elif entry.get("type") == "decision" and kind == "retrieve" and entry.get("action") == "injected":
            key = "recall_hits"
        else:
            continue
        counts[key] += 1
        now = issue_of_turn(chain, entry.get("turn", 0))
        origin_turn = records.get(entry.get("item_id"))
        then = issue_of_turn(chain, origin_turn) if origin_turn else None
        if now is not None and then is not None and then < now:
            counts[f"{key}_earlier_issue"] += 1
    return counts


def summarise(arm: str, chains: dict[str, dict], sidecar: Path | None) -> dict:
    issues = [i for c in chains.values() for i in c["issues"]]
    length = max((len(c["issues"]) for c in chains.values()), default=0)
    by_position = [[i["resolved"] for c in chains.values() for i in c["issues"] if i["index"] == k]
                   for k in range(length)]
    after, before = [], []
    for chain in chains.values():
        seen = 0
        for issue in chain["issues"]:
            (after if seen else before).append(issue["resolved"])
            seen += issue.get("compactions", 0)
    contexts = [x for i in issues for x in i["contexts"]]
    half = length // 2
    out = {
        "arm": arm, "chains": len(chains), "issues": len(issues),
        "resolved": sum(i["resolved"] for i in issues),
        "resolved_by_position": [sum(p) for p in by_position],
        "resolved_first_half": sum(sum(p) for p in by_position[:half]),
        "resolved_second_half": sum(sum(p) for p in by_position[half:]),
        "compactions": sum(i.get("compactions", 0) for i in issues),
        "chains_compacted": sum(any(i.get("compactions") for i in c["issues"]) for c in chains.values()),
        "resolved_before_first_compaction": [sum(before), len(before)],
        "resolved_after_first_compaction": [sum(after), len(after)],
        "turns": sum(i["turns"] for i in issues),
        "mean_context": round(sum(contexts) / len(contexts)) if contexts else 0,
        "peak_context": max((i["peak_context"] for i in issues), default=0),
        "prompt_tokens": sum(i["prompt_tokens"] for i in issues),
        "uncached_input": sum(i["input"] for i in issues),
        "cache_read": sum(i["cache_read"] for i in issues),
        "output_tokens": sum(i["output"] for i in issues),
        "cost_usd": round(sum(i["cost_usd"] for i in issues), 4),
        "timeouts": sum(i["timed_out"] for i in issues),
        "empty_patches": sum(i["patch_lines"] == 0 for i in issues),
    }
    jev = [c["jev"] for c in chains.values() if c.get("jev")]
    if jev:
        work = [j.get("workarea") or {} for j in jev]
        out["jev"] = {
            "saved_at_admission": sum(j["saved_tokens"] for j in jev),
            "scored_at_admission": sum(j["original_tokens"] for j in jev),
            "workarea_compacted_tokens": sum(w.get("compacted_tokens", 0) for w in work),
            "workarea_compactions": sum(w.get("compactions", 0) for w in work),
            "pressure_compactions": sum(w.get("pressure_compactions", 0) for w in work),
            "declined_compactions": sum(w.get("declined_compactions", 0) for w in work),
            "jev_input_tokens": sum(j["jev_input_tokens"] for j in jev),
            "stale": sum((j.get("relations") or {}).get("stale", 0) for j in jev),
            "superseded": sum((j.get("relations") or {}).get("superseded", 0) for j in jev),
        }
        if sidecar is not None:
            use = defaultdict(int)
            for chain in chains.values():
                for key, value in memory_use(chain, sidecar).items():
                    use[key] += value
            out["jev"].update(use)
    return out


def paired(chains: dict[str, dict[str, dict]], arm: str, base: str) -> dict:
    """Per-issue outcomes against the base arm, on the chains both finished."""
    both = only_arm = only_base = neither = 0
    for chain_id, chain in chains.get(arm, {}).items():
        other = chains.get(base, {}).get(chain_id)
        if other is None:
            continue
        for a, b in zip(chain["issues"], other["issues"], strict=False):
            if a["resolved"] and b["resolved"]:
                both += 1
            elif a["resolved"]:
                only_arm += 1
            elif b["resolved"]:
                only_base += 1
            else:
                neither += 1
    return {"both": both, f"only_{arm}": only_arm, f"only_{base}": only_base, "neither": neither}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--sidecar", action="append", default=[], metavar="ARM=DIR")
    parser.add_argument("--base", default="off")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    sidecars = {k: Path(v) for k, v in (s.split("=", 1) for s in args.sidecar)}
    chains = load_chains(args.root)
    report = {"arms": [summarise(arm, chains[arm], sidecars.get(arm)) for arm in sorted(chains)],
              "paired": {arm: paired(chains, arm, args.base) for arm in chains if arm != args.base}}
    for row in report["arms"]:
        print(json.dumps(row))
    for arm, counts in report["paired"].items():
        print(arm, "vs", args.base, counts)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
