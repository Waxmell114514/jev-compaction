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
from jevctx.jev import HttpJevClient
from jevctx.shadow import ShadowLog
from jevctx.store import JsonlStore
from jevctx.usage import Prices


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
                        help="Jev USD per million input tokens; omitted means unknown cost")
    args = parser.parse_args(argv)
    key = os.environ.get("OPENAI_API_KEY")
    if not key or not args.base_url or not args.model:
        parser.error("set OPENAI_API_KEY, OPENAI_BASE_URL and OPENAI_MODEL (or URL/model flags)")
    if args.mode != "off" and not os.environ.get("TYPESAFE_API_KEY"):
        parser.error("shadow/on requires TYPESAFE_API_KEY")
    if args.max_steps < 1 or args.max_completion_tokens < 1:
        parser.error("step/token limits must be positive")
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
    with ExitStack() as stack:
        http = stack.enter_context(httpx.Client(
            headers={"Authorization": f"Bearer {key}"}, timeout=120,
        ))
        jev = stack.enter_context(HttpJevClient()) if args.mode != "off" else None
        result = run_agent(
            args.task, model=args.model, base_url=args.base_url, http=http,
            tools=[schema], handlers={"read_file": read_file},
            store=JsonlStore(args.output / "memory.jsonl"),
            log=ShadowLog(args.output / "shadow.jsonl"),
            mode=args.mode, jev=jev, max_steps=args.max_steps,
            max_completion_tokens=args.max_completion_tokens,
            prices=prices, jev_input_price=args.jev_input_price,
        )
    with (args.output / "run.json").open("x", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, ensure_ascii=False, indent=2)
    print(json.dumps({"status": result.status, **result.metrics}, ensure_ascii=False, indent=2))
    print(result.answer)
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
