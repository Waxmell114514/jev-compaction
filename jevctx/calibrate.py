"""Fit per-source keep thresholds to what the agent actually used.

Jev's admit score is a prior: "will a later step need this?". Whether a later
step *did* need it is observable afterwards, from what the agent went on to do
-- the code it edited, the identifiers it named. This module turns logged
(score, used) pairs into keep thresholds: for each group of segments (a tool, or
a tool and segment kind), the highest threshold that still elides no more than
``max_loss`` of the segments that turned out to be used.

One threshold for everything is the wrong shape. On SWE-bench traffic the score
separates used from unused ``read`` output well and ``bash`` output poorly, and
source files the agent went on to edit sometimes score far below log noise. A
group gets its own threshold only when it has at least ``min_used`` used
samples; otherwise it inherits its parent's (``tool:read/code`` → ``tool:read``
→ the global default), so a thin group never gets a threshold fitted to noise.

The fitted thresholds go straight into ``GateConfig.thresholds``::

    python -m jevctx.calibrate samples.jsonl --max-loss 0.1 --out thresholds.json

where each line of ``samples.jsonl`` is ``{"source", "kind", "score", "tokens",
"used"}``. How "used" is decided is harness-specific: on SWE-bench, a segment counted
as used when the agent later edited at least two of its lines (or one of 30+
characters).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DEFAULT_GRID", "CurvePoint", "UsageSample", "curve", "fit_thresholds", "main"]

#: Candidate thresholds. 0.0 means "never elide". Dense near zero: role
#: probabilities (``gate_on="role:..."``) pile up there, and on SWE-bench the useful
#: ``change_site`` thresholds were 0.01-0.05.
DEFAULT_GRID: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075,
                                   *(round(0.05 * i, 2) for i in range(2, 17)))


@dataclass(frozen=True)
class UsageSample:
    source: str
    kind: str
    score: float
    tokens: int
    used: bool

    def groups(self) -> tuple[str, str, str]:
        """Most to least specific."""
        return f"{self.source}/{self.kind}", self.source, "*"


@dataclass(frozen=True)
class CurvePoint:
    threshold: float
    elided_tokens: int
    total_tokens: int
    used_elided: int
    used_total: int

    @property
    def elided_share(self) -> float:
        return self.elided_tokens / self.total_tokens if self.total_tokens else 0.0

    @property
    def loss(self) -> float:
        """Share of used segments this threshold would have elided."""
        return self.used_elided / self.used_total if self.used_total else 0.0


def curve(samples: Sequence[UsageSample], grid: Sequence[float] = DEFAULT_GRID) -> list[CurvePoint]:
    """Tokens elided and used segments lost at each threshold."""
    total = sum(s.tokens for s in samples)
    used = [s for s in samples if s.used]
    return [
        CurvePoint(
            threshold=t,
            elided_tokens=sum(s.tokens for s in samples if s.score < t),
            total_tokens=total,
            used_elided=sum(1 for s in used if s.score < t),
            used_total=len(used),
        )
        for t in grid
    ]


def _best(points: Sequence[CurvePoint], max_loss: float) -> float:
    allowed = [p.threshold for p in points if p.loss <= max_loss]
    return max(allowed) if allowed else 0.0


def fit_thresholds(
    samples: Iterable[UsageSample],
    *,
    max_loss: float = 0.1,
    min_used: int = 10,
    grid: Sequence[float] = DEFAULT_GRID,
) -> tuple[float, dict[str, float]]:
    """Return ``(default, overrides)`` for ``GateConfig(keep_threshold=default,
    thresholds=overrides)``.

    An override is written only where it differs from what the group would
    inherit, so the mapping stays short and every entry is a real finding.
    """
    if not 0 <= max_loss <= 1:
        raise ValueError("max_loss must be between 0 and 1")
    by_group: dict[str, list[UsageSample]] = defaultdict(list)
    for sample in samples:
        for group in sample.groups():
            by_group[group].append(sample)
    if not by_group:
        raise ValueError("no samples")

    def fitted(group: str) -> float | None:
        members = by_group.get(group, [])
        if sum(s.used for s in members) < min_used:
            return None
        return _best(curve(members, grid), max_loss)

    default = fitted("*")
    if default is None:
        raise ValueError(f"fewer than {min_used} used samples overall")
    overrides: dict[str, float] = {}
    for source in sorted({g for g in by_group if "/" not in g and g != "*"}):
        own = fitted(source)
        source_threshold = default if own is None else own
        if own is not None and own != default:
            overrides[source] = own
        for group in sorted(g for g in by_group if g.startswith(source + "/")):
            fine = fitted(group)
            if fine is not None and fine != source_threshold:
                overrides[group] = fine
    return default, overrides


def _load(path: Path) -> list[UsageSample]:
    samples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                samples.append(UsageSample(source=row["source"], kind=row.get("kind", ""),
                                           score=float(row["score"]), tokens=int(row["tokens"]),
                                           used=bool(row["used"])))
    return samples


def render(samples: Sequence[UsageSample], groups: Sequence[str],
           grid: Sequence[float] = DEFAULT_GRID) -> str:
    lines = []
    for group in groups:
        members = [s for s in samples if group in s.groups()]
        if not members:
            continue
        lines.append(f"{group}: {len(members)} segments, {sum(s.used for s in members)} used")
        lines.append("  threshold  elided tokens  used segments lost")
        for p in curve(members, grid):
            lines.append(f"  {p.threshold:9.2f}  {p.elided_share:13.0%}  {p.loss:18.0%}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("samples", type=Path)
    parser.add_argument("--max-loss", type=float, default=0.1)
    parser.add_argument("--min-used", type=int, default=10)
    parser.add_argument("--out", type=Path, help="Write {keep_threshold, thresholds} JSON")
    parser.add_argument("--curve", action="store_true", help="Print the per-group curves")
    args = parser.parse_args(argv)
    samples = _load(args.samples)
    default, overrides = fit_thresholds(samples, max_loss=args.max_loss, min_used=args.min_used)
    result: Mapping[str, object] = {"keep_threshold": default, "thresholds": overrides}
    if args.curve:
        print(render(samples, ["*", *sorted({s.source for s in samples})]))
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
