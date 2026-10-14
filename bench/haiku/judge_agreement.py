"""How often judge layouts agree, per scored segment, in a judge_replay.py output.

    python bench/haiku/judge_agreement.py runs/judge-replay

``flat`` against ``flat2`` is the floor: the same request asked twice. A layout that
agrees with ``flat`` about as often as ``flat2`` does changes nothing but the bill.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

GATE = 0.1          # the jev arms' keep threshold on P(role = change_site)
INJECTION = 0.8     # GateConfig.injection_threshold


def segments(root: Path) -> list[dict[str, dict]]:
    """One entry per scored segment: judge -> {keep, change_site, role, type, injection}."""
    found = []
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            items: dict[str, dict[str, dict]] = defaultdict(dict)
            for judge in ("flat", "flat2", "cached", "compact"):
                answers = event.get(judge)
                if answers is None:
                    continue
                for key, value in answers.items():
                    item, _, field = key.rpartition(":")
                    entry = items[item].setdefault(judge, {})
                    if field in ("keep", "injection"):
                        entry[field] = value
                    elif field in ("role", "type") and isinstance(value, dict):
                        entry[field] = max(value, key=value.get)
                        if field == "role":
                            entry["change_site"] = value.get("change_site", 0.0)
            found += [v for v in items.values() if "flat" in v]
    return found


def compare(segs: list[dict[str, dict]], other: str) -> dict[str, float]:
    pairs = [(s["flat"], s[other]) for s in segs if other in s
             and {"keep", "change_site", "role", "type", "injection"} <= s["flat"].keys() <= s[other].keys()]
    n = len(pairs) or 1
    return {
        "segments": len(pairs),
        "gate decision agrees": sum((a["change_site"] < GATE) == (b["change_site"] < GATE) for a, b in pairs) / n,
        "role agrees": sum(a["role"] == b["role"] for a, b in pairs) / n,
        "type agrees": sum(a["type"] == b["type"] for a, b in pairs) / n,
        "injection flag agrees": sum((a["injection"] >= INJECTION) == (b["injection"] >= INJECTION)
                                     for a, b in pairs) / n,
        "|Δ keep|": sum(abs(a["keep"] - b["keep"]) for a, b in pairs) / n,
        "|Δ P(change_site)|": sum(abs(a["change_site"] - b["change_site"]) for a, b in pairs) / n,
        "elided by flat": sum(a["change_site"] < GATE for a, _ in pairs) / n,
        "elided by other": sum(b["change_site"] < GATE for _, b in pairs) / n,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    segs = segments(args.root)
    rows = {other: compare(segs, other) for other in ("flat2", "cached", "compact")}
    metrics = list(next(iter(rows.values())))
    print(f"{'vs flat':24}" + "".join(f"{o:>10}" for o in rows))
    for m in metrics:
        print(f"{m:24}" + "".join(f"{rows[o][m]:>10.3f}" if isinstance(rows[o][m], float)
                                  else f"{rows[o][m]:>10}" for o in rows))
    (args.root / "agreement.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
