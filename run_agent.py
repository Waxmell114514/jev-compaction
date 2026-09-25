"""Run a read-only task against explicitly selected files with a compatible API."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import httpx

from jevctx.agent import run_agent
from jevctx.judge import make_judge, resolve_judge
from jevctx.pipeline import GateConfig
from jevctx.shadow import ShadowLog
from jevctx.store import JsonlStore
from jevctx.usage import Prices
from jevctx.workarea import WorkAreaConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--file", action="append", required=True,
                        help="Allow reading this UTF-8 file; repeat for multiple files")
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--mode", choices=("off", "shadow", "on"), default="shadow")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--prices", type=float, nargs=4,
                        metavar=("INPUT", "OUTPUT", "CACHE_READ", "CACHE_WRITE"),
                        help="USD per million tokens; omitted means unknown cost")
    parser.add_argument("--jev-input-price", type=float,
                        help="Judge USD per million input tokens (Jev: 0.042); omitted means unknown cost")
    parser.add_argument("--profile", action="store_true",
                        help="Label every segment's type, role, lifetime and injection risk")
    parser.add_argument("--gate-on", default="keep",
                        help='"keep" or "role:<name>", e.g. role:change_site (needs --profile)')
    parser.add_argument("--thresholds", type=Path,
                        help="JSON from `python -m jevctx.calibrate --out`")
    parser.add_argument("--workarea", action="store_true",
                        help="Compact the transcript's tail before each request when it pays "
                             "(priced with --prices, else $3 input / $0.30 cache read)")
    parser.add_argument("--intent", choices=("off", "reply", "arg"), default="off",
                        help="Judge each output against what the model was looking for: "
                             "its message before the call (reply), or also an optional "
                             "`intent` argument on every tool (arg)")
    args = parser.parse_args(argv)
    key = os.environ.get("OPENAI_API_KEY")
    if not key or not args.base_url or not args.model:
        parser.error("set OPENAI_API_KEY, OPENAI_BASE_URL and OPENAI_MODEL (or URL/model flags)")
    if args.mode != "off":
        try:
            judge = resolve_judge()
        except ValueError as exc:
            parser.error(str(exc))
        if not judge.ready:
            parser.error(f"shadow/on needs a judge: set {judge.missing}")
    if args.max_steps < 1 or args.max_completion_tokens < 1:
        parser.error("step/token limits must be positive")
    if args.gate_on != "keep" and not (args.profile and args.gate_on.startswith("role:")):
        parser.error('--gate-on must be "keep", or "role:<name>" with --profile')
    try:
        prices = Prices(*args.prices) if args.prices is not None else None
        if args.jev_input_price is not None:
            Prices(args.jev_input_price, 0, 0, 0)
        # Freeze explicitly allowed files before model execution. A model-supplied
        # path never reaches the filesystem, including traversal/symlink targets.
        files = {}
        for name in args.file:
            with Path(name).open("rb") as handle:
                data = handle.read(1_000_001)
            if len(data) > 1_000_000:
                parser.error("each selected file must be at most 1 MB")
            files[name] = data.decode("utf-8")
        args.output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError) as exc:
        parser.error(f"invalid prices, unreadable UTF-8 input, or existing output: {type(exc).__name__}")

    def read_file(path: str) -> str:
        if not isinstance(path, str) or path not in files:
            raise ValueError("path is not in the allowed file list")
        return files[path]

    schema = {"type": "function", "function": {
        "name": "read_file", "description": "Read an explicitly selected input file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "enum": list(files)},
        }, "required": ["path"], "additionalProperties": False},
    }}
    try:
        fitted = json.loads(args.thresholds.read_text(encoding="utf-8")) \
            if args.thresholds is not None else {}
        gate = GateConfig(profile=args.profile, gate_on=args.gate_on,
                          keep_threshold=fitted.get("keep_threshold", GateConfig.keep_threshold),
                          thresholds=fitted.get("thresholds", {}))
    except (OSError, ValueError) as exc:
        parser.error(f"invalid --gate-on/--profile or thresholds file: {type(exc).__name__}")
    workarea = None
    if args.workarea:
        workarea = WorkAreaConfig(price_input=prices.input, price_cache_read=prices.cache_read) \
            if prices is not None else WorkAreaConfig()
    with ExitStack() as stack:
        http = stack.enter_context(httpx.Client(
            headers={"Authorization": f"Bearer {key}"}, timeout=120,
        ))
        jev = stack.enter_context(make_judge()) if args.mode != "off" else None
        result = run_agent(
            args.task, model=args.model, base_url=args.base_url, http=http,
            tools=[schema], handlers={"read_file": read_file},
            store=JsonlStore(args.output / "memory.jsonl"),
            log=ShadowLog(args.output / "shadow.jsonl"),
            mode=args.mode, jev=jev, max_steps=args.max_steps,
            max_completion_tokens=args.max_completion_tokens,
            prices=prices, jev_input_price=args.jev_input_price,
            gate_config=gate, workarea=workarea, intent=args.intent,
        )
    with (args.output / "run.json").open("x", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, ensure_ascii=False, indent=2)
    print(json.dumps({"status": result.status, **result.metrics}, ensure_ascii=False, indent=2))
    print(result.answer)
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
