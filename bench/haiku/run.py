"""Chains of django SWE-bench Verified issues in one Claude Haiku session, per policy.

    export ANTHROPIC_API_KEY=...
    export JEVCTX_JUDGE=llm JUDGE_BASE_URL=https://api.anthropic.com/v1 \\
           JUDGE_MODEL=claude-haiku-5-5 JUDGE_API_KEY=$ANTHROPIC_API_KEY \\
           JUDGE_JSON_MODE=0 JUDGE_TEMPERATURE=none
    python bench/haiku/run.py --scratch <dir> --out runs/haiku --arms off,mask,stale,jev \\
        --chain 6 --chains 8 --workers 4 [--shadow] [--thresholds t.json] [--seed 0]

A chain is ``--chain`` issues of one Django version, in the order they were reported,
worked in one session: after each issue the patch is saved and graded, the checkout
is reset to the next issue's base commit, and the next issue is sent into the same
conversation. One result line per (chain, arm, issue) in ``results.jsonl``; a finished
(chain, arm) is skipped on rerun. Transcripts go to ``<out>/<arm>/<chain>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.haiku.agent import Policy, Session  # noqa: E402
from bench.haiku.tasks import Instance, Workspace, load_instances  # noqa: E402
from jevctx.judge import make_judge  # noqa: E402
from jevctx.pipeline import GateConfig  # noqa: E402
from jevctx.shadow import ShadowLog  # noqa: E402
from jevctx.store import JsonlStore  # noqa: E402

TASK = """{preamble}

<issue>
{problem}
</issue>

Make the smallest correct change to the library code (under django/), run the relevant
existing tests, and stop with a short summary when done. Do not add or modify tests."""
FIRST = "Fix the following issue in the Django repository checked out at {root}."
NEXT = ("Your previous change has been saved and the checkout at {root} has been reset to "
        "a different commit for the next issue. Fix this issue now.")


def build_chains(instances: list[Instance], *, length: int, count: int, seed: int,
                 difficulty: tuple[str, ...]) -> list[list[Instance]]:
    """``count`` chains of ``length`` issues, each from one version, in report order."""
    by_version: dict[str, list[Instance]] = defaultdict(list)
    for inst in instances:
        if inst.difficulty in difficulty:
            by_version[inst.version].append(inst)
    rng = random.Random(seed)
    chains: list[list[Instance]] = []
    versions = sorted(by_version)
    while len(chains) < count and versions:
        version = versions[len(chains) % len(versions)]
        pool = by_version[version]
        if len(pool) < length:
            versions.remove(version)
            continue
        picked = rng.sample(pool, length)
        for inst in picked:
            pool.remove(inst)
        picked.sort(key=lambda i: i.instance_id)   # issue numbers rise with report date
        chains.append(picked)
    return chains


def policy_for(arm: str, *, shadow: bool, mask_turns: int, gate: GateConfig,
               cliff_tokens: int) -> Policy:
    """``stalec``/``jevc`` also compact, whatever the cache price, once the transcript nears
    the long-context price tier; ``staleh``/``jevh`` do that and estimate the turns left
    from the issues still queued, instead of a fixed prior."""
    base = arm[:-1] if arm in {"stalec", "jevc", "staleh", "jevh"} else arm
    if base not in {"off", "mask", "stale", "jev"}:
        raise ValueError(f"unknown arm {arm}")
    policy = Policy(arm=base, shadow=shadow and base == "off", mask_turns=mask_turns, gate=gate,
                    horizon=arm.endswith("h"))
    if base != arm:
        policy.workarea = replace(policy.workarea, pressure_tokens=cliff_tokens)
    return policy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default="off")
    parser.add_argument("--chain", type=int, default=6, help="issues per session")
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--chains-file", type=Path, help="JSON list of lists of instance ids")
    parser.add_argument("--versions", default="3.2,4.0,4.1,4.2")
    parser.add_argument("--difficulty", default="15 min - 1 hour,1-4 hours")
    parser.add_argument("--seed", type=int, default=0, help="run label; also the sampling seed")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=60, help="per issue")
    parser.add_argument("--shadow", action="store_true", help="off arm: score with the judge")
    parser.add_argument("--thresholds", type=Path, help="JSON from jevctx.calibrate for the jev arm")
    parser.add_argument("--mask-turns", type=int, default=10)
    parser.add_argument("--cliff-tokens", type=int, default=85_000,
                        help="stalec/jevc: estimated transcript size that forces compaction")
    parser.add_argument("--keep-worktrees", action="store_true")
    args = parser.parse_args(argv)

    instances = load_instances(args.scratch, versions=tuple(args.versions.split(",")))
    by_id = {i.instance_id: i for i in instances}
    if args.chains_file:
        chains = [[by_id[i] for i in ids] for ids in json.loads(args.chains_file.read_text())]
    else:
        chains = build_chains(instances, length=args.chain, count=args.chains, seed=args.seed,
                              difficulty=tuple(args.difficulty.split(",")))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "chains.json").write_text(json.dumps(
        [[i.instance_id for i in chain] for chain in chains], indent=0))
    arms = args.arms.split(",")
    results_path = args.out / "results.jsonl"
    done: set[tuple[str, str, int]] = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("index") == r.get("chain_length", -1) - 1:
                    done.add((r["chain"], r["arm"], r.get("seed", 0)))
    gate = GateConfig(profile=True, gate_on="role:change_site")
    if args.thresholds:
        fitted = json.loads(args.thresholds.read_text())
        gate = GateConfig(profile=True, gate_on="role:change_site",
                          keep_threshold=fitted["keep_threshold"], thresholds=fitted["thresholds"])
    need_judge = any(a.startswith("jev") for a in arms) or args.shadow
    judge = make_judge() if need_judge else None
    workspace = Workspace(args.scratch)
    lock = threading.Lock()

    def emit(record: dict) -> None:
        with lock:
            with results_path.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{record['arm']:6} {record['chain']:14} #{record['index']} "
                  f"{record['instance_id']:26} {record.get('status')} resolved="
                  f"{record.get('resolved')} turns={record.get('turns')} "
                  f"ctx={record.get('context_tokens_last')} ${record.get('host_cost_usd', 0):.4f}",
                  flush=True)

    def one(chain: list[Instance], arm: str) -> None:
        chain_id = f"{chain[0].version}-{chain[0].instance_id.split('-')[-1]}"
        if (chain_id, arm, args.seed) in done:
            return
        run_dir = args.out / arm / f"{chain_id}-s{args.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        root = args.scratch / "work" / arm / f"{chain_id}-s{args.seed}"
        session = None
        try:
            workspace.checkout(chain[0], root)
            store = JsonlStore(run_dir / "memory.jsonl")
            log = ShadowLog(run_dir / "shadow.jsonl")
            trace_file = (run_dir / "trace.jsonl").open("w")

            def trace(event: dict) -> None:
                trace_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                trace_file.flush()

            policy = policy_for(arm, shadow=args.shadow, mask_turns=args.mask_turns, gate=gate,
                                cliff_tokens=args.cliff_tokens)
            session = Session(root=root, python=workspace.python_for(chain[0]), policy=policy,
                              store=store, log=log, judge=judge, trace=trace)
            for index, instance in enumerate(chain):
                record: dict = {"chain": chain_id, "chain_length": len(chain), "index": index,
                                "instance_id": instance.instance_id, "arm": arm, "seed": args.seed,
                                "version": instance.version, "difficulty": instance.difficulty}
                if index:
                    workspace.reset(root, instance.base_commit)
                    session.note_reset()
                preamble = (FIRST if index == 0 else NEXT).format(root=root)
                result = session.run(TASK.format(preamble=preamble, problem=instance.problem_statement),
                                     max_steps=args.max_steps, issues_left=len(chain) - index - 1)
                patch = workspace.model_patch(root)
                issue_dir = run_dir / f"{index:02d}-{instance.instance_id}"
                issue_dir.mkdir(exist_ok=True)
                (issue_dir / "patch.diff").write_text(patch)
                (issue_dir / "per_request.json").write_text(json.dumps(result.pop("per_request")))
                record.update(result)
                record["patch_lines"] = patch.count("\n")
                grade = workspace.grade(instance, root) if patch.strip() else {
                    "resolved": False, "error": "empty patch"}
                (issue_dir / "grade.json").write_text(json.dumps(grade, indent=1))
                record["resolved"] = grade["resolved"]
                record["grade"] = {k: v for k, v in grade.items() if k != "log_tail"}
                emit(record)
            trace_file.close()
            (run_dir / "messages.json").write_text(
                json.dumps(session.messages, ensure_ascii=False, indent=0))
        except Exception as exc:  # noqa: BLE001 - one failed chain must not stop the sweep
            (run_dir / "error.txt").write_text(traceback.format_exc())
            emit({"chain": chain_id, "chain_length": len(chain), "index": -1,
                  "instance_id": "-", "arm": arm, "seed": args.seed, "status": "harness_error",
                  "error": f"{type(exc).__name__}: {exc}"})
            if session is not None:
                (run_dir / "messages.json").write_text(
                    json.dumps(session.messages, ensure_ascii=False, indent=0))
        finally:
            if not args.keep_worktrees:
                workspace.remove(root)

    jobs = [(chain, arm) for chain in chains for arm in arms]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda job: one(*job), jobs))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
