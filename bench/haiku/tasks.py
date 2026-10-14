"""SWE-bench Verified django instances, run and graded locally, without Docker.

Django's test suite needs only SQLite and three pure-Python packages, so an instance
runs from a git worktree at its base commit with ``PYTHONPATH`` pointing at it, in a
per-major-version virtualenv (Python 3.8 for Django 3.x, 3.10 for 4.x and 5.0).

Grading follows SWE-bench's django parser: apply the instance's test patch, run the
FAIL_TO_PASS and PASS_TO_PASS tests with ``tests/runtests.py -v 2`` and read each
test's ``ok``/``FAIL``/``ERROR`` line. Resolved means every listed test passed.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

VERIFIED_PARQUET = "data/verified.parquet"
ENV_FOR_VERSION = {"3.0": "dj3", "3.1": "dj3", "3.2": "dj3", "4.0": "dj4", "4.1": "dj4",
                   "4.2": "dj4", "5.0": "dj4"}
_STATUS = re.compile(r"^(?P<name>\S+) \((?P<cls>[\w.]+)\)(?: \.\.\. | ... )?(?P<rest>.*)$")


@dataclass(frozen=True)
class Instance:
    instance_id: str
    base_commit: str
    version: str
    problem_statement: str
    patch: str
    test_patch: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    difficulty: str

    @property
    def test_files(self) -> list[str]:
        return re.findall(r"^diff --git a/(\S+) b/", self.test_patch, re.M)

    @property
    def test_modules(self) -> list[str]:
        """``tests/migrations/test_operations.py`` -> ``migrations.test_operations``: the
        labels SWE-bench runs (whole modules, then it reads each test's status)."""
        found = []
        for path in self.test_files:
            if path.startswith("tests/") and path.endswith(".py"):
                mod = path[len("tests/"):-3].replace("/", ".")
                mod = mod[:-len(".__init__")] if mod.endswith(".__init__") else mod
                found.append(mod)
        return found


def load_instances(scratch: Path, *, versions: tuple[str, ...] | None = None) -> list[Instance]:
    rows = pq.read_table(scratch / VERIFIED_PARQUET).to_pylist()
    found = []
    for row in rows:
        if row["repo"] != "django/django" or row["version"] not in ENV_FOR_VERSION:
            continue
        if versions and row["version"] not in versions:
            continue
        found.append(Instance(
            row["instance_id"], row["base_commit"], row["version"], row["problem_statement"],
            row["patch"], row["test_patch"], tuple(json.loads(row["FAIL_TO_PASS"])),
            tuple(json.loads(row["PASS_TO_PASS"])), row["difficulty"]))
    return found


def label(test: str) -> str:
    """``test_x (module.Class)`` -> ``module.Class.test_x`` (what runtests.py takes)."""
    match = re.fullmatch(r"(\S+) \(([\w.]+)\)(?: .*)?", test)
    if not match:
        return test
    return f"{match.group(2)}.{match.group(1)}"


class Workspace:
    """One worktree per (instance, arm), from a shared bare mirror."""

    def __init__(self, scratch: Path) -> None:
        self.scratch = scratch
        self.mirror = scratch / "repos" / "django.git"
        self._fetch_lock = threading.Lock()

    def fetch(self, commit: str) -> None:
        """Fetch one commit into the mirror once; concurrent fetches would race on its lock."""
        with self._fetch_lock:
            have = subprocess.run(["git", "-C", str(self.mirror), "cat-file", "-e", f"{commit}^{{commit}}"],
                                  capture_output=True)
            if have.returncode == 0:
                return
            subprocess.run(["git", "-C", str(self.mirror), "fetch", "-q", "--depth", "1", "origin",
                            commit], check=True, capture_output=True, timeout=600)

    def reset(self, worktree: Path, commit: str) -> None:
        """Point an existing worktree at another commit, discarding every change."""
        self.fetch(commit)
        checkout = ["git", "-C", str(worktree), "checkout", "-q", "-f", "--detach", commit]
        if subprocess.run(checkout, capture_output=True).returncode != 0:
            # The agent can leave the checkout in a state git refuses to leave (a stale
            # index.lock, an unfinished merge): clear it and try once more, then rebuild.
            lock = subprocess.run(["git", "-C", str(worktree), "rev-parse", "--git-path", "index.lock"],
                                  capture_output=True, text=True).stdout.strip()
            if lock:
                Path(worktree, lock).unlink(missing_ok=True)
            subprocess.run(["git", "-C", str(worktree), "reset", "-q", "--hard"], capture_output=True)
            if subprocess.run(checkout, capture_output=True).returncode != 0:
                self.remove(worktree)
                subprocess.run(["rm", "-rf", str(worktree)], check=True)
                with self._fetch_lock:
                    subprocess.run(["git", "-C", str(self.mirror), "worktree", "add", "-q", "--detach",
                                    str(worktree), commit], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(worktree), "clean", "-qfdx"], capture_output=True)

    def python_for(self, instance: Instance) -> Path:
        return self.scratch / "envs" / ENV_FOR_VERSION[instance.version] / "bin" / "python"

    def checkout(self, instance: Instance, dest: Path) -> Path:
        if dest.exists():
            subprocess.run(["git", "-C", str(self.mirror), "worktree", "remove", "--force",
                            str(dest)], capture_output=True)
            subprocess.run(["rm", "-rf", str(dest)], check=True)
        self.fetch(instance.base_commit)
        with self._fetch_lock:
            subprocess.run(["git", "-C", str(self.mirror), "worktree", "add", "-q", "--detach",
                            str(dest), instance.base_commit], check=True, capture_output=True)
        # The agent may `git diff`/`git status`; a detached worktree is enough for that.
        return dest

    def remove(self, dest: Path) -> None:
        with self._fetch_lock:
            subprocess.run(["git", "-C", str(self.mirror), "worktree", "remove", "--force",
                            str(dest)], capture_output=True)
            subprocess.run(["git", "-C", str(self.mirror), "worktree", "prune"], capture_output=True)

    def model_patch(self, worktree: Path) -> str:
        subprocess.run(["git", "-C", str(worktree), "add", "-A"], capture_output=True)
        out = subprocess.run(["git", "-C", str(worktree), "diff", "--cached", "--binary"],
                             capture_output=True, text=True, errors="replace")
        subprocess.run(["git", "-C", str(worktree), "reset", "-q"], capture_output=True)
        return out.stdout

    def grade(self, instance: Instance, worktree: Path, *, timeout: int = 1500) -> dict:
        """Apply the test patch and run the listed tests. Leaves the worktree dirty."""
        for path in instance.test_files:
            subprocess.run(["git", "-C", str(worktree), "checkout", "-q", instance.base_commit,
                            "--", path], capture_output=True)
        applied = subprocess.run(["git", "-C", str(worktree), "apply", "-"],
                                 input=instance.test_patch, capture_output=True, text=True)
        if applied.returncode != 0:
            return {"resolved": False, "error": "test patch failed: " + applied.stderr[-500:]}
        labels = instance.test_modules
        if not labels:
            return {"resolved": False, "error": "no test modules in the test patch"}
        cmd = [str(self.python_for(instance)), "tests/runtests.py", "-v", "2", "--parallel", "1",
               "--noinput", *labels]
        try:
            run = subprocess.run(cmd, cwd=worktree, capture_output=True, text=True, errors="replace",
                                 timeout=timeout, env={"PYTHONPATH": str(worktree), "PATH": "/usr/bin:/bin",
                                                       "LANG": "C.UTF-8", "HOME": str(worktree)})
        except subprocess.TimeoutExpired:
            return {"resolved": False, "error": "test timeout"}
        status = parse_statuses(run.stdout + "\n" + run.stderr)
        f2p = {t: status.get(t) for t in instance.fail_to_pass}
        p2p = {t: status.get(t) for t in instance.pass_to_pass}
        ok = lambda s: s == "PASSED"  # noqa: E731
        return {
            "resolved": all(ok(v) for v in f2p.values()) and all(ok(v) for v in p2p.values()),
            "fail_to_pass": sum(ok(v) for v in f2p.values()), "fail_to_pass_total": len(f2p),
            "pass_to_pass": sum(ok(v) for v in p2p.values()), "pass_to_pass_total": len(p2p),
            "missing": sum(v is None for v in [*f2p.values(), *p2p.values()]),
            "returncode": run.returncode, "log_tail": (run.stdout + run.stderr)[-3000:],
        }


def parse_statuses(log: str) -> dict[str, str]:
    """SWE-bench's ``parse_log_django``: the key is whatever precedes `` ... ok``, which
    for a test with a docstring is the docstring's first line, as in the dataset."""
    found: dict[str, str] = {}
    previous = ""
    for raw in log.splitlines():
        line = raw.strip()
        for suffix, status in ((" ... ok", "PASSED"), (" ... OK", "PASSED"),
                               (" ... FAIL", "FAILED"), (" ... ERROR", "ERROR")):
            if line.endswith(suffix):
                found[line[:-len(suffix)].strip()] = status
                # A test with a docstring prints its id on one line and the docstring
                # with the status on the next; the dataset keys by either, so record both.
                if re.fullmatch(r"\S+ \([\w.]+\)", previous):
                    found[previous] = status
                break
        else:
            if " ... skipped" in line:
                found[line.split(" ... skipped")[0].strip()] = "SKIPPED"
            elif line.startswith("FAIL: ") or line.startswith("ERROR: "):
                status, _, rest = line.partition(": ")
                name = rest.split(" ... ")[0].strip()
                found[name] = "FAILED" if status == "FAIL" else "ERROR"
        previous = line
    return found
