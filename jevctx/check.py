"""Verify a real Jev API key end to end.

    python -m jevctx.check

Everything else in this package is tested against a mock transport, which proves
the code is self-consistent but never proves it can talk to Jev. This module is the
one thing that makes a real request, so the first time you plug in a key you find
out in one command rather than halfway through an agent run.

It checks three things, in order, and stops at the first failure:

1. All three question types come back from one request and parse.
2. The gate runs on a real tool output and relocates something.
3. The pointer expands back to the original, byte for byte.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable

from jevctx.pipeline import admit, expand, reconstruct
from jevctx.shadow import ShadowLog
from jevctx.store import InMemoryStore
from jevctx.types import (
    PRICE_PER_INPUT_TOKEN,
    Choice,
    JevAuthError,
    JevClient,
    JevError,
    JevUnavailableError,
    JevValidationError,
    Noul,
    Origin,
    Score,
)

__all__ = ["run_check", "main", "SAMPLE_OUTPUT", "SAMPLE_TASK"]

ENV_VAR = "TYPESAFE_API_KEY"

SAMPLE_TASK = "The build fails on a dependency conflict. Find which package is pinned wrong."

SAMPLE_OUTPUT = (
    "\n".join(f"npm http fetch GET 200 https://registry.npmjs.org/dep-{i} {20 + i}ms"
              for i in range(20))
    + "\n\n"
    + "\n".join(f"npm ERROR peer react@^18.0.0 required by ui-kit-{i}, found 17.0.2"
                for i in range(12))
    + "\n\n"
    + "\n".join(f"npm notice created a lockfile entry for dep-{i}" for i in range(4))
    + "\n\nadded 412 packages, audited 1204 packages in 9s\n"
)

_GREEN, _RED, _DIM, _BOLD, _OFF = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def _ok(text: str) -> None:
    print(f"  {_GREEN}✓{_OFF} {text}")


def _fail(text: str) -> None:
    print(f"  {_RED}✗{_OFF} {text}")


def _dim(text: str) -> None:
    print(f"    {_DIM}{text}{_OFF}")


def run_check(client_factory: Callable[[], JevClient] | None = None) -> int:
    """Run the live check. Returns a process exit code."""
    print(f"\n{_BOLD}jevctx — live Jev check{_OFF}\n")

    if client_factory is None:
        key = os.environ.get(ENV_VAR)
        if not key:
            _fail(f"{ENV_VAR} is not set")
            _dim(f"export {ENV_VAR}=... and run this again.")
            _dim("Without a key everything still runs offline: python demo.py")
            return 2
        print(f"  key       {_DIM}{ENV_VAR} = {key[:6]}…{key[-4:]}{_OFF}")

        from jevctx.jev import HttpJevClient
        client_factory = HttpJevClient

    try:
        client = client_factory()
    except JevAuthError as exc:
        _fail(f"could not build a client: {exc}")
        return 2

    endpoint = getattr(client, "endpoint", "(custom client)")
    print(f"  endpoint  {_DIM}{endpoint}{_OFF}\n")

    try:
        if not _check_question_types(client):
            return 1
        if not _check_the_gate(client):
            return 1
    except JevAuthError as exc:
        _fail("Jev rejected the key")
        _dim(str(exc))
        _dim(f"Check {ENV_VAR}, or generate a new key at https://typesafe.ai")
        return 2
    except JevValidationError as exc:
        _fail("Jev rejected the request as malformed — this is a bug in jevctx")
        _dim(str(exc))
        _dim("Please open an issue with this message.")
        return 1
    except JevUnavailableError as exc:
        _fail("could not reach Jev")
        _dim(str(exc))
        _dim("Network, proxy, or an outage. The gate fails open, so an agent using")
        _dim("jevctx would keep working here — just without compaction.")
        return 1
    except JevError as exc:
        _fail(f"{type(exc).__name__}: {exc}")
        return 1

    usage = getattr(client, "usage", None)
    if usage is not None and getattr(usage, "input_tokens", 0):
        cost = usage.input_tokens * PRICE_PER_INPUT_TOKEN
        print(f"\n  usage     {_DIM}{usage.requests} requests · "
              f"{usage.input_tokens:,} input tokens · ${cost:.6f}{_OFF}")

    print(f"\n{_GREEN}{_BOLD}Everything works.{_OFF} Drop HttpJevClient() into your agent:\n")
    print(f"{_DIM}    from jevctx import HttpJevClient, admit, InMemoryStore, ShadowLog")
    print("    result = admit(tool_output, origin, task_digest=..., turn=n,")
    print("                   client=HttpJevClient(), store=store, log=log,")
    print(f"                   config=GateConfig(shadow_only=True))   # start here{_OFF}\n")
    return 0


def _check_question_types(client: JevClient) -> bool:
    """One request, all three answer shapes. Proves parsing against a real body."""
    print(f"{_BOLD}1. three question types in one request{_OFF}")
    state = {
        "message": "Hi there! Quick question about the build — is the CI red again?",
        "context": "A message posted in an engineering chat channel.",
    }
    questions = {
        "is_question": Noul(
            instructions="Is the author asking for something?",
            true="The author is asking a question or making a request.",
            false="The author is not asking for anything.",
        ),
        "topic": Choice(
            instructions="What is this message about?",
            criteria={"build": "CI, builds or deploys", "billing": "Money or invoices",
                      "social": "Small talk with no request"},
        ),
        "urgency": Score(
            instructions="How urgent is this message?",
            criteria=["Not urgent", "Worth a look today", "Needs attention now"],
        ),
    }

    started = time.monotonic()
    answers = client.ask(state, questions)
    elapsed_ms = (time.monotonic() - started) * 1000

    missing = set(questions) - set(answers)
    if missing:
        _fail(f"Jev did not answer: {', '.join(sorted(missing))}")
        return False

    noul, choice, score = answers["is_question"], answers["topic"], answers["urgency"]
    _ok(f"one request, three answers, {elapsed_ms:.0f} ms")
    _dim(f"noul   is_question = {noul.noul:.2f}")
    _dim(f"choice topic       = {choice.choice!r} (confidence {choice.confidence:.2f})")
    _dim(f"score  urgency     = {score.score:.2f} (confidence {score.confidence:.2f})")

    if not 0.0 <= noul.noul <= 1.0:
        _fail(f"noul outside [0,1]: {noul.noul}")
        return False
    if choice.choice not in questions["topic"].criteria:
        _fail(f"choice returned an option that was not offered: {choice.choice!r}")
        return False
    _ok("all three answer types parsed and are in range")
    return True


def _check_the_gate(client: JevClient) -> bool:
    """The whole pipeline, against real judgements."""
    print(f"\n{_BOLD}2. the gate on a real tool output{_OFF}")
    store, log = InMemoryStore(), ShadowLog(path=None)
    origin = Origin(source="tool:bash", ref="npm install", turn=1)

    started = time.monotonic()
    result = admit(SAMPLE_OUTPUT, origin, task_digest=SAMPLE_TASK, turn=1,
                   client=client, store=store, log=log)
    elapsed_ms = (time.monotonic() - started) * 1000

    if any(s.failed for s in result.scores):
        errors = {s.error for s in result.scores if s.error}
        _fail(f"scoring failed: {', '.join(sorted(errors))}")
        return False

    _ok(f"{len(result.scores)} segments scored, {elapsed_ms:.0f} ms")
    for score, seg_line in zip(result.scores,
                               [s.split("\n")[0][:52] for s in _segment_previews()],
                               strict=False):
        _dim(f"{score.score:.2f}  {seg_line}")

    if result.tripwire:
        _fail(f"the max-elide tripwire fired ({result.tripwire})")
        _dim("Jev wanted to remove most of this output. Not fatal, but the gate")
        _dim("kept everything, so there is nothing further to check here.")
        return False
    if not result.pointers:
        _fail("nothing was relocated — Jev scored every segment above the threshold")
        _dim("Not necessarily wrong, but this sample is mostly progress noise, so it")
        _dim("is worth a look at the scores above before trusting the gate.")
        return False

    saved = result.original_tokens - result.result_tokens
    _ok(f"{result.original_tokens:,} → {result.result_tokens:,} tokens "
        f"({saved / result.original_tokens:.0%} smaller), {len(result.pointers)} pointer(s)")

    print(f"\n{_BOLD}3. the pointer expands back{_OFF}")
    recovered = expand(result.pointers[0].id, store=store, log=log, turn=2)
    if not recovered:
        _fail("expand() returned nothing")
        return False
    if reconstruct(result.text, store) != SAMPLE_OUTPUT:
        _fail("reconstruction did not match the original byte for byte")
        return False
    _ok(f"expand() returned {len(recovered):,} chars; full output reconstructs byte-exact")
    return True


def _segment_previews() -> list[str]:
    from jevctx.segments import segment
    return [s.text.strip() for s in
            segment(SAMPLE_OUTPUT, Origin(source="tool:bash", ref="npm install", turn=1))]


def main() -> int:
    return run_check()


if __name__ == "__main__":
    sys.exit(main())
