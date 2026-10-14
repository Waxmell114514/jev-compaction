"""Summarise bench/haiku runs: per arm, paired by chain against ``off``.

    python bench/haiku/analyze.py runs/haiku-main [--json out.json]

Everything here is metered: token counts are the provider's own usage fields, and
dollars are those counts at Claude Haiku 5.5's price sheet (the long-context tier
included). Intervals are 95% bootstrap intervals over chains (a chain is one
session, the unit that is independent), for the ratio of summed costs.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ORDER = ["off", "mask", "stale", "stalec", "staleh", "jev", "jevc", "jevh", "jevh-flat", "jevh-cached", "jevh-compact"]


def load(root: Path) -> list[dict]:
    rows = []
    for line in (root / "results.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("index", -1) >= 0:
                rows.append(row)
    return rows


def total_cost(row: dict) -> float:
    return row.get("host_cost_usd", 0.0) + row.get("judge_cost_usd", 0.0)


def boot_ratio(pairs: list[tuple[float, float]], *, n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Ratio of sums (arm / control) with a bootstrap interval over the pairs."""
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    point = sum(a for a, _ in pairs) / max(1e-12, sum(b for _, b in pairs))
    draws = []
    for _ in range(n):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        draws.append(sum(a for a, _ in sample) / max(1e-12, sum(b for _, b in sample)))
    draws.sort()
    return point, draws[int(0.025 * n)], draws[int(0.975 * n)]


def boot_diff(pairs: list[tuple[float, float]], *, n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    point = sum(a - b for a, b in pairs)
    draws = []
    for _ in range(n):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        draws.append(sum(a - b for a, b in sample))
    draws.sort()
    return point, draws[int(0.025 * n)], draws[int(0.975 * n)]


def per_chain(rows: list[dict]) -> dict[tuple[str, int], dict[str, dict]]:
    """(chain, seed) -> arm -> totals over the chain's issues; complete chains only."""
    grouped: dict[tuple[str, int], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(row["chain"], row.get("seed", 0))][row["arm"]].append(row)
    out: dict[tuple[str, int], dict[str, dict]] = {}
    for key, arms in grouped.items():
        out[key] = {}
        for arm, issues in arms.items():
            if len(issues) != issues[0]["chain_length"]:
                continue
            out[key][arm] = {
                "cost": sum(total_cost(i) for i in issues),
                "host_cost": sum(i["host_cost_usd"] for i in issues),
                "judge_cost": sum(i.get("judge_cost_usd", 0.0) for i in issues),
                "resolved": sum(bool(i.get("resolved")) for i in issues),
                "issues": len(issues),
                "cache_write": sum(i["usage"]["cache_write"] for i in issues),
                "cache_read": sum(i["usage"]["cache_read"] for i in issues),
                "input": sum(i["usage"]["input"] for i in issues),
                "output": sum(i["usage"]["output"] for i in issues),
                "requests": sum(i["usage"]["requests"] for i in issues),
                "prompt_tokens": sum(i.get("prompt_tokens", 0) for i in issues),
                "max_prompt": max(i.get("max_prompt_tokens", 0) for i in issues),
                "long_context_requests": sum(i.get("long_context_requests", 0) for i in issues),
                "hidden_edits": sum(i.get("hidden_edits", 0) for i in issues),
                "hidden_rereads": sum(i.get("hidden_rereads", 0) for i in issues),
                "expands": sum(i.get("expands", 0) for i in issues),
                "recalls": sum(i.get("recalls", 0) for i in issues),
                "edits": sum(i.get("edits", 0) for i in issues),
                "hidden_outputs": issues[-1].get("hidden_outputs", 0),
                "hidden_tokens": issues[-1].get("hidden_tokens", 0),
                "tool_output_tokens": sum(i.get("tool_output_tokens", 0) for i in issues),
                "admit_elided_tokens": sum(i.get("admit_elided_tokens", 0) for i in issues),
                "compactions": (issues[-1].get("workarea") or {}).get("compactions", 0),
                "declined": (issues[-1].get("workarea") or {}).get("declined_compactions", 0),
                "pressure": (issues[-1].get("workarea") or {}).get("pressure_compactions", 0),
                "errors": sum(not str(i.get("status", "")).startswith(("completed", "step_limit"))
                              for i in issues),
            }
    return out


def summarise(root: Path) -> dict:
    rows = load(root)
    chains = per_chain(rows)
    arms = sorted({arm for c in chains.values() for arm in c},
                  key=lambda a: ORDER.index(a)
                  if a in ORDER else 99)
    summary: dict = {"chains": len(chains), "arms": {}}
    for arm in arms:
        keys = [k for k, c in chains.items() if arm in c and "off" in c]
        totals = [chains[k][arm] for k in keys]
        if not totals:
            continue
        agg = {f: sum(t[f] for t in totals) for f in totals[0] if f != "max_prompt"}
        agg["max_prompt"] = max(t["max_prompt"] for t in totals)
        agg["chains"] = len(totals)
        agg["cost_per_resolved"] = agg["cost"] / agg["resolved"] if agg["resolved"] else None
        agg["cache_hit_share"] = agg["cache_read"] / max(1, agg["cache_read"] + agg["cache_write"] + agg["input"])
        if arm != "off":
            agg["cost_vs_off"] = boot_ratio([(chains[k][arm]["cost"], chains[k]["off"]["cost"]) for k in keys])
            agg["prompt_vs_off"] = boot_ratio([(chains[k][arm]["prompt_tokens"], chains[k]["off"]["prompt_tokens"])
                                              for k in keys])
            agg["cache_write_vs_off"] = boot_ratio([(chains[k][arm]["cache_write"], chains[k]["off"]["cache_write"])
                                                   for k in keys])
            agg["resolved_minus_off"] = boot_diff([(chains[k][arm]["resolved"], chains[k]["off"]["resolved"])
                                                   for k in keys])
        summary["arms"][arm] = agg
    summary["by_position"] = by_position(rows)
    return summary


def by_position(rows: list[dict]) -> dict:
    """Per arm and position in the chain: mean metered prompt per request and cost."""
    acc: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        acc[row["arm"]][row["index"]].append(row)
    out: dict = {}
    for arm, positions in acc.items():
        out[arm] = {}
        for index, issues in sorted(positions.items()):
            requests = sum(i["usage"]["requests"] for i in issues) or 1
            out[arm][index] = {
                "n": len(issues), "resolved": sum(bool(i.get("resolved")) for i in issues),
                "prompt_per_request": round(sum(i.get("prompt_tokens", 0) for i in issues) / requests),
                "cost": round(sum(total_cost(i) for i in issues), 4),
            }
    return out


def fmt_ci(ci: tuple[float, float, float] | None, digits: int = 2) -> str:
    if not ci:
        return "-"
    return f"{ci[0]:.{digits}f} [{ci[1]:.{digits}f}, {ci[2]:.{digits}f}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    summary = summarise(args.root)
    print(f"{summary['chains']} chains")
    head = (f"{'arm':7} {'n':>2} {'res':>4} {'$total':>8} {'$judge':>7} {'$/res':>7} "
            f"{'hit':>5} {'cw Mtok':>8} {'long':>5} {'maxP':>7} {'hidE':>4} {'hidR':>4} "
            f"{'exp':>4} {'rec':>4} {'cmp':>4} {'dec':>4} {'prs':>4}  cost/off            "
            f"prompt/off          Δresolved")
    print(head)
    for arm, a in summary["arms"].items():
        cpr = f"{a['cost_per_resolved']:.4f}" if a["cost_per_resolved"] else "-"
        print(f"{arm:7} {a['chains']:>2} {a['resolved']:>4} {a['cost']:>8.4f} {a['judge_cost']:>7.4f} "
              f"{cpr:>7} {a['cache_hit_share']:>5.2f} {a['cache_write'] / 1e6:>8.3f} "
              f"{a['long_context_requests']:>5} {a['max_prompt']:>7} {a['hidden_edits']:>4} "
              f"{a['hidden_rereads']:>4} {a['expands']:>4} {a['recalls']:>4} {a['compactions']:>4} "
              f"{a['declined']:>4} {a['pressure']:>4}  {fmt_ci(a.get('cost_vs_off')):19} "
              f"{fmt_ci(a.get('prompt_vs_off')):19} {fmt_ci(a.get('resolved_minus_off'), 0)}")
    print("\nprompt tokens per request, by position in the chain")
    for arm, positions in summary["by_position"].items():
        print(f"  {arm:7} " + "  ".join(f"#{i}:{p['prompt_per_request']:>6}" for i, p in positions.items()))
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
