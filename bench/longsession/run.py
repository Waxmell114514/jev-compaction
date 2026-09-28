"""Long sessions: one OpenCode session works through a chain of SWE-bench issues.

SWE-bench tasks are short (about 25 turns), so nothing a context manager does late in
a session ever happens there. Here one agent session takes a *chain* of Verified
issues from the same repository and version, in the order they were reported. After
each issue the harness keeps the patch, resets ``/testbed`` to the next issue's base
commit and sends the next issue into the same session (``opencode run -s``). The
session's context carries everything from earlier issues, and, with a capped context
window, OpenCode has to manage it: its own LLM summary (``off``), plus pruning of old
tool outputs (``prune``), or the jev plugin in front of both (``jev``).

With ``--revisit`` a chain is ``N`` issues and then the same ``N`` again: the agent is
told it worked on the issue earlier in the session and that its change was lost when
the repository was reset (lost work, a reset branch, the same fix needed twice).
Here earlier context is worth something, and the second attempt has a natural
baseline, the first.

Issues from one repository and version share an environment, so a chain runs in the
image of its first issue, and every issue is graded in a fresh container of that
image, checked out at its own base commit (``--validate`` grades the gold patches
this way first, to show the shared environment is sound).

Arms:

- ``off``: plain OpenCode: auto-compaction (an LLM summary) when the window fills
- ``prune``: OpenCode with ``compaction.prune``: old tool outputs cleared as well
- ``jev``: the plugin with the profiled gate, ``recall`` and work-area compaction; the
  sidecar is told of every reset, so outputs from before it are out of date. Its
  sidecar runs with ``--pressure-tokens``, so outputs go to pointers before the window
  forces OpenCode's summary
- ``jevp``: ``jev`` recalling on the agent's behalf (``JEV_PROACTIVE``): each new issue is
  also a ``recall`` query, and what it finds is appended to the issue
- ``jev0``: ``jev`` without window pressure (point it at a second sidecar with
  ``--arm-url jev0=...``): the price check alone decides

Setup, as for bench/swebench (an internal network whose only exit is egress.py)::

    docker network create --internal --subnet 172.30.0.0/24 --gateway 172.30.0.1 jevbench
    python bench/longsession/egress.py --bind 172.30.0.1 --port 3128 --allow opencode.ai &
    python -m jevctx.serve --host 172.30.0.1 --port 8765 --data-dir runs/long/sidecar \\
        --profile --gate-on role:change_site --thresholds bench/longsession/thresholds.json \\
        --max-elide-fraction 1.0 --price-input 0.15 --price-cache-read 0.003 ... &
    python bench/longsession/run.py --dataset verified.jsonl --out runs/long \\
        --groups django/django@3.0,django/django@3.2 --chain 8 --arms off,prune,jev \\
        --model deepseek-v4-flash --context 128000

Results are resumable per (chain, arm): a finished chain is skipped, and an
unfinished one continues from its first issue without a result.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from swebench.harness.constants import APPLY_PATCH_FAIL
from swebench.harness.grading import get_eval_report
from swebench.harness.run_evaluation import GIT_APPLY_CMDS
from swebench.harness.utils import make_test_spec

REPO = Path(__file__).resolve().parents[2]
IMAGE = "ghcr.io/epoch-research/swe-bench.eval.x86_64.{}:latest"
CONTAINER_PATH = ("/opt/node22/bin:/opt/opencode/node_modules/opencode-linux-x64/bin:"
                  "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
                  "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")

FIRST = """You will work through a series of issues in this repository, one after another, \
in this same session. After each one your change is saved and the repository is reset \
for the next issue.

<issue>
{problem}
</issue>

The repository at /testbed is checked out at the commit where this issue was reported, \
with its dependencies installed. Resolve the issue by editing non-test source files. \
You may run code and the existing tests to check your work; there is no internet access. \
Do not modify existing tests. \
When you are done, reply with a one-paragraph summary of the change."""

NEXT = """Your change for the previous issue has been saved. The repository at /testbed has \
been reset to commit {commit}, where the next issue was reported: your earlier edits are \
no longer there, and files may differ from what you saw before.

<issue>
{problem}
</issue>

Resolve this issue the same way: edit non-test source files, check your work if you \
like, do not modify existing tests, and reply with a one-paragraph summary when done."""

REVISIT = """Your change for the previous issue has been saved. The repository at /testbed has \
been reset to commit {commit}. The next issue is one you already worked on earlier in \
this session: the change you made then was lost when the repository was reset, so it has \
to be made again.

<issue>
{problem}
</issue>

Resolve it again: edit non-test source files, check your work if you like, do not modify \
existing tests, and reply with a one-paragraph summary when done."""

ARMS: dict[str, dict] = {
    "off": {"plugin": None, "compaction": {}},
    "prune": {"plugin": None, "compaction": {"prune": True}},
    "jev": {"plugin": {"JEV_MODE": "on", "JEV_PROFILE": "1", "JEV_MAX_ELIDE_FRACTION": "1",
                       "JEV_WORKAREA": "1"}, "compaction": {}},
}
ARMS["jev0"] = ARMS["jev"]
# ``jev`` plus recall on the agent's behalf: each new issue is also a recall query.
ARMS["jevp"] = {"plugin": {**ARMS["jev"]["plugin"], "JEV_PROACTIVE": "1"}, "compaction": {}}


def sh(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, **kwargs)


def docker_exec(name: str, command: str, **kwargs) -> subprocess.CompletedProcess:
    return sh(["docker", "exec", "-w", "/testbed", name, "bash", "-c", command], **kwargs)


def build_chains(rows: list[dict], groups: list[str], length: int, seed: int,
                 revisit: bool = False) -> list[dict]:
    """One chain per ``repo@version``: ``length`` issues, sampled, in reported order;
    with ``revisit``, followed by the same issues again."""
    chains = []
    for group in groups:
        repo, _, version = group.partition("@")
        pool = [r for r in rows if r["repo"] == repo and r["version"] == version]
        if len(pool) < length:
            raise SystemExit(f"{group}: only {len(pool)} issues")
        picked = random.Random(f"{seed}:{group}").sample(pool, length)
        picked.sort(key=lambda r: r["created_at"])
        if revisit:
            picked += [{**r, "revisit": True} for r in picked]
        chains.append({"id": f"{repo.split('/')[1]}-{version}-s{seed}" + ("-rv" if revisit else ""),
                       "image": IMAGE.format(picked[0]["instance_id"]),
                       "issues": picked})
    return chains


def pull(image: str) -> None:
    if sh(["docker", "image", "inspect", image]).returncode == 0:
        return
    for attempt in range(4):
        if sh(["docker", "pull", "-q", image]).returncode == 0:
            return
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"could not pull {image}")


def reset_to(name: str, commit: str) -> None:
    out = docker_exec(name, f"git reset -q --hard && git clean -fdq && git checkout -q -f {commit}")
    if out.returncode != 0:
        raise RuntimeError(f"reset to {commit} failed: {out.stderr[-500:]}")


def usage_from_events(path: Path) -> dict:
    """Per-issue summary of ``opencode run --format json``: usage, turns, tools, stop."""
    totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "cost_usd": 0.0}
    turns, tools, contexts, tool_result_chars = 0, {}, [], 0
    final_stop, session_id, answer, rejected = None, None, "", False
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            session_id = session_id or event.get("sessionID")
            part = event.get("part") or {}
            if event.get("type") == "step_finish":
                tokens = part.get("tokens") or {}
                cache = tokens.get("cache") or {}
                turns += 1
                totals["input"] += tokens.get("input", 0)
                totals["output"] += tokens.get("output", 0) + tokens.get("reasoning", 0)
                totals["cache_read"] += cache.get("read", 0)
                totals["cache_write"] += cache.get("write", 0)
                totals["cost_usd"] += part.get("cost") or 0.0
                contexts.append(tokens.get("input", 0) + cache.get("read", 0) + cache.get("write", 0))
                final_stop = part.get("reason", "?")
            elif event.get("type") == "tool_use":
                state = part.get("state") or {}
                tools[part.get("tool")] = tools.get(part.get("tool"), 0) + 1
                tool_result_chars += len(state.get("output") or "")
                if "rejected permission" in str(state.get("error") or ""):
                    rejected = True
            elif event.get("type") == "text":
                answer = part.get("text", "")
            elif event.get("type") == "error":
                message = ((event.get("error") or {}).get("data") or {}).get("message", "")
                final_stop = "permission_rejected" if "rejected permission" in message else "error"
    if rejected and final_stop == "tool-calls":
        final_stop = "permission_rejected"
    return {**totals, "prompt_tokens": totals["input"] + totals["cache_read"] + totals["cache_write"],
            "turns": turns, "tool_calls": tools, "contexts": contexts,
            "peak_context": max(contexts, default=0), "tool_result_chars": tool_result_chars,
            "final_stop": final_stop, "session_id": session_id, "answer": answer[-1500:]}


def opencode_config(arm: str, args) -> dict:
    model = {"limit": {"context": args.context, "output": args.max_output}} if args.context else {}
    config = {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False,
        "share": "disabled",
        # Headless, any ask is auto-rejected and ends the run: allow what the task needs.
        "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny",
                       "external_directory": "allow", "doom_loop": "allow"},
        "compaction": ARMS[arm]["compaction"],
    }
    if model:
        config["provider"] = {args.provider: {"models": {args.model: model}}}
    return config


def container_env(args, extra: dict[str, str]) -> list[str]:
    gateway = urlparse(args.egress).hostname
    env = {"PATH": CONTAINER_PATH, "NODE_EXTRA_CA_CERTS": "/ca.crt", "HTTPS_PROXY": args.egress,
           "NO_PROXY": gateway, "XDG_CONFIG_HOME": "/bench/config", "XDG_DATA_HOME": "/bench/data",
           "OPENCODE_API_KEY": os.environ.get("OPENCODE_API_KEY", ""), **extra}
    return [part for key, value in env.items() for part in ("-e", f"{key}={value}")]


def post(url: str, body: dict) -> dict:
    request = Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def run_issue(name: str, arm: str, args, chain_dir: Path, index: int, issue: dict,
              session_id: str | None, jev_session: str, turn_offset: int) -> dict:
    task_dir = chain_dir / f"{index:02d}-{issue['instance_id']}"
    task_dir.mkdir(parents=True, exist_ok=True)
    reset_to(name, issue["base_commit"])
    template = FIRST if session_id is None else REVISIT if issue.get("revisit") else NEXT
    prompt = template.format(problem=issue["problem_statement"], commit=issue["base_commit"][:12])
    (chain_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    plugin, jev_url = ARMS[arm]["plugin"], args.arm_urls.get(arm, args.jev_url)
    extra = {}
    if plugin is not None:
        extra = {**plugin, "JEV_URL": jev_url, "JEV_SESSION": jev_session,
                 "JEV_TURN_OFFSET": str(turn_offset)}
        if session_id is not None:
            # The reset rewrote the work tree: every earlier view of a file is out of date.
            post(f"{jev_url}/observe", {
                "session": jev_session, "tool": "bash", "call_id": f"reset-{index}",
                "args": {"command": f"git checkout -f {issue['base_commit']}"},
                "turn": turn_offset, "cwd": "/testbed"})
    model = f"{args.provider}/{args.model}"
    resume = f"-s {session_id} " if session_id else ""
    command = (f"git config --global --add safe.directory /testbed; timeout {args.timeout} "
               f"opencode run --format json -m {model} {resume}\"$(cat /bench/prompt.txt)\"")
    events = task_dir / "events.jsonl"
    started = time.time()
    started_ms = int(started * 1000)
    with events.open("w") as out, (task_dir / "agent.stderr").open("w") as err:
        proc = subprocess.Popen(["docker", "exec", *container_env(args, extra), "-w", "/testbed",
                                 name, "bash", "-c", command], stdout=out, stderr=err)
        size, changed, stalled = 0, started, False
        while proc.poll() is None:
            now = time.time()
            if events.stat().st_size != size:
                size, changed = events.stat().st_size, now
            stalled = now - changed > args.stall_timeout
            if now - started > args.timeout + 120 or stalled:
                proc.kill()
                proc.wait()
                break
            time.sleep(5)
    elapsed = time.time() - started
    usage = usage_from_events(events)
    if stalled or usage["turns"] == 0 or (proc.returncode != 124 and usage["final_stop"] == "error"):
        raise RuntimeError(f"provider error on {issue['instance_id']} after {usage['turns']} turns "
                           f"(exit {proc.returncode}, stalled {stalled})")
    patch = docker_exec(name, "git add -A && git diff --cached --binary HEAD").stdout
    docker_exec(name, "git reset -q")
    (task_dir / "patch.diff").write_text(patch, encoding="utf-8")
    return {"index": index, "instance_id": issue["instance_id"], "arm": arm, "model": args.model,
            "revisit": bool(issue.get("revisit")),
            "exit_code": proc.returncode, "timed_out": proc.returncode == 124,
            "started_ms": started_ms, "ended_ms": int(time.time() * 1000),
            "elapsed_s": round(elapsed, 1), "patch_lines": patch.count("\n"), **usage}


def compactions(chain_dir: Path) -> list[dict]:
    """OpenCode's own compactions in this chain's session store: when, and auto or not."""
    path = chain_dir / "data" / "opencode" / "opencode.db"
    if not path.exists():
        return []
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = db.execute("SELECT time_created, data FROM part").fetchall()
    finally:
        db.close()
    found = []
    for created, data in rows:
        part = json.loads(data)
        if part.get("type") == "compaction":
            found.append({"time_ms": created, "auto": part.get("auto"),
                          "overflow": part.get("overflow")})
    return sorted(found, key=lambda c: c["time_ms"])


def grade(image: str, issue: dict, patch: str, task_dir: Path) -> dict:
    """Apply the patch at the issue's own base commit in a fresh container, and grade it."""
    log_path = task_dir / "eval.log"
    if not patch.strip():
        log_path.write_text("empty patch\n")
        return {"resolved": False, "applied": False}
    spec = make_test_spec(issue)
    (task_dir / "eval.sh").write_text(spec.eval_script)
    name = f"eval-{uuid.uuid4().hex[:10]}"
    sh(["docker", "run", "-d", "--name", name, "--network", "none",
        "-v", f"{task_dir.resolve()}:/bench:ro", image, "sleep", "infinity"], check=True)
    try:
        docker_exec(name, "git config --global --add safe.directory /testbed")
        log, applied = [], False
        for command in GIT_APPLY_CMDS:
            reset_to(name, issue["base_commit"])
            out = docker_exec(name, f"{command} /bench/patch.diff")
            log.append(f"$ {command}\n{out.stdout}{out.stderr}")
            if out.returncode == 0:
                applied = True
                break
        if not applied:
            log.append(APPLY_PATCH_FAIL)
            log_path.write_text("\n".join(log))
            return {"resolved": False, "applied": False}
        out = sh(["docker", "exec", name, "bash", "/bench/eval.sh"], timeout=3600)
        log.append(out.stdout + out.stderr)
        log_path.write_text("\n".join(log))
    finally:
        sh(["docker", "rm", "-f", name])
    report = get_eval_report(spec, {"instance_id": issue["instance_id"], "model_patch": patch},
                             str(log_path), include_tests_status=False)[issue["instance_id"]]
    return {"resolved": report["resolved"], "applied": report["patch_successfully_applied"]}


def run_chain(chain: dict, arm: str, args) -> list[dict]:
    chain_dir = args.out / chain["id"] / arm
    done = chain_dir / "chain.json"
    if done.exists():
        return json.loads(done.read_text())["issues"]
    chain_dir.mkdir(parents=True, exist_ok=True)
    state_path = chain_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "session_id": None, "jev_session": f"{chain['id']}__{arm}__{uuid.uuid4().hex[:6]}",
        "turns": 0, "results": []}
    config = chain_dir / "config" / "opencode"
    (config / "plugins").mkdir(parents=True, exist_ok=True)
    (config / "opencode.json").write_text(json.dumps(opencode_config(arm, args), indent=2))
    link = config.parent / "node_modules"
    if not link.is_symlink():
        link.symlink_to(Path(args.opencode_dir) / "node_modules")
    if ARMS[arm]["plugin"] is not None:
        shutil.copy(REPO / "integrations/opencode/jev.ts", config / "plugins" / "jev.ts")
    (chain_dir / "data").mkdir(exist_ok=True)
    pull(chain["image"])
    name = f"long-{chain['id']}-{arm}-{uuid.uuid4().hex[:6]}".lower().replace(".", "")
    sh(["docker", "run", "-d", "--name", name, "--network", args.network,
        "-v", f"{args.opencode_dir}:{args.opencode_dir}:ro", "-v", f"{args.node_dir}:/opt/node22:ro",
        "-v", f"{args.ca}:/ca.crt:ro", "-v", f"{chain_dir.resolve()}:/bench",
        chain["image"], "sleep", "infinity"], check=True)
    try:
        for index, issue in enumerate(chain["issues"]):
            if index < len(state["results"]):
                continue
            if index > 0 and state["session_id"] is None:
                raise RuntimeError("no session to continue")
            result = run_issue(name, arm, args, chain_dir, index, issue, state["session_id"],
                               state["jev_session"], state["turns"])
            events = compactions(chain_dir)
            result["compactions"] = sum(result["started_ms"] <= c["time_ms"] <= result["ended_ms"]
                                        for c in events)
            result.update(grade(chain["image"], issue,
                                (chain_dir / f"{index:02d}-{issue['instance_id']}" / "patch.diff").read_text(),
                                chain_dir / f"{index:02d}-{issue['instance_id']}"))
            state["session_id"] = state["session_id"] or result["session_id"]
            state["turns"] += result["turns"]
            state["results"].append(result)
            state_path.write_text(json.dumps(state, indent=2))
            print(f"{chain['id']:18s} {arm:6s} #{index} {issue['instance_id']:28s} "
                  f"resolved={result['resolved']!s:5s} turns={result['turns']:3d} "
                  f"peak={result['peak_context']:>7,} compactions={result['compactions']} "
                  f"cost=${result['cost_usd']:.4f} "
                  f"t={result['elapsed_s']:.0f}s", flush=True)
    finally:
        sh(["docker", "rm", "-f", name])
    summary = {"chain": chain["id"], "arm": arm, "model": args.model, "context": args.context,
               "session_id": state["session_id"], "jev_session": state["jev_session"],
               "issues": state["results"]}
    summary["compaction_events"] = compactions(chain_dir)
    if ARMS[arm]["plugin"] is not None:
        with urlopen(f"{args.arm_urls.get(arm, args.jev_url)}/stats?session={state['jev_session']}", timeout=30) as response:
            summary["jev"] = json.load(response)
    done.write_text(json.dumps(summary, indent=2))
    return state["results"]


def validate(chains: list[dict], args) -> None:
    """Grade every issue's gold patch in its chain's image: is the shared environment sound?"""
    for chain in chains:
        pull(chain["image"])
        for issue in {i["instance_id"]: i for i in chain["issues"]}.values():
            task_dir = args.out / "validate" / chain["id"] / issue["instance_id"]
            if (task_dir / "result.json").exists():
                continue
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "patch.diff").write_text(issue["patch"])
            result = grade(chain["image"], issue, issue["patch"], task_dir)
            print(f"{chain['id']:18s} {issue['instance_id']:28s} gold resolved={result['resolved']}",
                  flush=True)
            (task_dir / "result.json").write_text(json.dumps(result))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="SWE-bench Verified as JSONL")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups", required=True, help="repo@version,... one chain each")
    parser.add_argument("--chain", type=int, default=8, help="Issues per chain")
    parser.add_argument("--revisit", action="store_true",
                        help="Follow the chain's issues with the same issues again")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude", nargs="*", default=[], help="Instance ids to leave out")
    parser.add_argument("--arms", default="off,prune,jev")
    parser.add_argument("--provider", default="opencode-go")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--context", type=int, default=128_000,
                        help="Context window OpenCode is told the model has (0: the model's own)")
    parser.add_argument("--max-output", type=int, default=32_000)
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per issue")
    parser.add_argument("--stall-timeout", type=int, default=600)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--network", default="jevbench")
    parser.add_argument("--egress", default="http://172.30.0.1:3128")
    parser.add_argument("--jev-url", default="http://172.30.0.1:8765")
    parser.add_argument("--arm-url", action="append", default=[], metavar="ARM=URL",
                        help="A different sidecar for one arm, e.g. jev0=http://172.30.0.1:8766")
    parser.add_argument("--opencode-dir", default="/opt/opencode")
    parser.add_argument("--node-dir", default="/opt/node22")
    parser.add_argument("--ca", default=os.environ.get("NODE_EXTRA_CA_CERTS", "/etc/ssl/certs/ca-certificates.crt"))
    parser.add_argument("--validate", action="store_true", help="Grade gold patches, run nothing")
    args = parser.parse_args(argv)
    args.arm_urls = dict(item.split("=", 1) for item in args.arm_url)
    arms = args.arms.split(",")
    if set(arms) - set(ARMS):
        parser.error(f"unknown arms: {sorted(set(arms) - set(ARMS))}")
    rows = [json.loads(line) for line in args.dataset.open(encoding="utf-8")]
    rows = [r for r in rows if r["instance_id"] not in set(args.exclude)]
    chains = build_chains(rows, args.groups.split(","), args.chain, args.seed, args.revisit)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "chains.json").write_text(json.dumps(
        [{"id": c["id"], "image": c["image"], "issues": [i["instance_id"] for i in c["issues"]]}
         for c in chains], indent=2))
    if args.validate:
        validate(chains, args)
        return 0
    lock = threading.Lock()
    # Chain-major order keeps a chain's arms in flight together, so their images and
    # the provider's load are the same for every arm.
    jobs = [(chain, arm) for chain in chains for arm in arms]
    with ThreadPoolExecutor(args.workers) as pool, (args.out / "results.jsonl").open("a") as sink:
        futures = {pool.submit(run_chain, chain, arm, args): (chain["id"], arm) for chain, arm in jobs}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results = future.result()
            except Exception as exc:  # keep the sweep going; a rerun resumes the chain
                print(f"FAILED {key}: {type(exc).__name__}: {exc}", flush=True)
                continue
            with lock:
                for result in results:
                    sink.write(json.dumps({"chain": key[0], **result}) + "\n")
                sink.flush()
            solved = sum(r["resolved"] for r in results)
            print(f"DONE {key[0]} {key[1]}: {solved}/{len(results)} resolved", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
