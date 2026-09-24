"""A local HTTP sidecar that exposes ``admit`` and ``expand`` to non-Python harnesses.

The OpenCode plugin in ``integrations/opencode`` is TypeScript; the gate is Python.
Rather than port the gate, the plugin posts every tool result here and gets back
the text to put in context. One process serves many agent sessions: each request names
its session, and each session gets its own store, shadow log and Jev client so
usage can be attributed per run.

    python -m jevctx.serve --port 8765 --data-dir runs/opencode

Endpoints (JSON in, JSON out):

- ``POST /admit``  ``{session, text, tool, call_id, task, turn, mode, max_elide_fraction?,
  profile?, args?, cwd?}`` →
  ``{text, gated, original_tokens, result_tokens, pointers, tripwire}``; with the
  call's ``args`` the session also learns what it viewed, ran or searched
- ``POST /observe`` ``{session, tool, call_id, args, turn, cwd?}`` → ``{relations}``: a
  tool call the gate does not see (``edit``, ``write``...), so later reads of an
  edited file make earlier ones stale (:mod:`jevctx.supersede`)
- ``POST /expand`` ``{session, id, turn}`` → ``{text}``, headed by a note if a later
  action made that output out of date
- ``POST /recall`` ``{session, query, turn, k?, type?, role?, name?, source?}`` →
  ``{text, hits: [{id, score, truncated}]}``
- ``POST /workarea`` ``{session, turn, recent, task?, items: [{id, kind, tokens, text?,
  tool?, call_id?}]}`` → ``{replacements: {id: text}, decision}``: the transcript's
  tail is offered for compaction (:mod:`jevctx.workarea`); apply every replacement
  to every request
- ``GET  /stats?session=...`` → per-session counters and Jev usage

The server binds to loopback by default and has no authentication: anything that can
reach the port can read what the agent read.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from collections import Counter
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from jevctx.jev import HttpJevClient
from jevctx.pipeline import GateConfig, admit, expand
from jevctx.profile import ROLE_QUESTION, TYPE_QUESTION
from jevctx.recall import recall, render_hits
from jevctx.shadow import ShadowLog
from jevctx.store import JsonlStore
from jevctx.supersede import Relation, SupersessionIndex, note_for
from jevctx.types import JevClient, JevError, MemoryStore, Origin
from jevctx.workarea import TailItem, WorkArea, WorkAreaConfig

__all__ = ["Session", "SidecarState", "make_server", "main"]

_SESSION_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
#: Jev's state limit is far above this, but every batch repeats the digest, so a
#: long problem statement multiplies the Jev bill. The head carries the task.
TASK_DIGEST_CHARS = 4000


@dataclass
class Session:
    store: MemoryStore
    log: ShadowLog
    client: JevClient | None
    lock: threading.Lock = field(default_factory=threading.Lock)
    admits: int = 0
    gated: int = 0
    tripwires: int = 0
    quarantined: int = 0
    recalls: int = 0
    task: str = ""
    workarea: WorkArea | None = None
    relations: SupersessionIndex = field(default_factory=SupersessionIndex)
    relations_path: Path | None = None
    expands: int = 0
    errors: int = 0
    original_tokens: int = 0
    result_tokens: int = 0


class SidecarState:
    """Sessions by name, created on first use."""

    def __init__(self, data_dir: Path | None, config: GateConfig,
                 client_factory=HttpJevClient,
                 workarea: WorkAreaConfig | None = None) -> None:
        self.data_dir = data_dir
        self.config = config
        self.workarea_config = workarea or WorkAreaConfig()
        self.client_factory = client_factory
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def session(self, name: Any, *, need_client: bool = True) -> Session:
        if not isinstance(name, str) or not _SESSION_RE.match(name):
            raise ValueError("session must match [A-Za-z0-9_.-]{1,128}")
        with self._lock:
            found = self._sessions.get(name)
            if found is None:
                if self.data_dir is not None:
                    directory = self.data_dir / name
                    directory.mkdir(parents=True, exist_ok=True)
                    store: MemoryStore = JsonlStore(directory / "memory.jsonl")
                    log = ShadowLog(directory / "shadow.jsonl")
                else:
                    from jevctx.store import InMemoryStore
                    store, log = InMemoryStore(), ShadowLog()
                found = Session(store=store, log=log, client=None)
                if self.data_dir is not None:
                    found.relations_path = self.data_dir / name / "relations.jsonl"
                    if found.relations_path.exists():
                        with found.relations_path.open(encoding="utf-8") as handle:
                            found.relations.load(json.loads(line) for line in handle if line.strip())
                self._sessions[name] = found
            if need_client and found.client is None:
                found.client = self.client_factory()
            return found

    def _observe(self, session: Session, body: dict[str, Any]) -> list[Relation]:
        args, call_id, turn = body.get("args"), body.get("call_id"), body.get("turn", 0)
        if not isinstance(args, dict) or not isinstance(call_id, str) or not call_id:
            return []
        cwd = body.get("cwd")
        with session.lock:
            if isinstance(cwd, str) and cwd and session.relations.cwd is None:
                session.relations.cwd = cwd
            made = session.relations.observe(call_id, str(body.get("tool") or ""), args,
                                             turn if type(turn) is int else 0)
            if made and session.relations_path is not None:
                with session.relations_path.open("a", encoding="utf-8") as handle:
                    for relation in made:
                        handle.write(json.dumps(relation.to_dict()) + "\n")
        return made

    def observe(self, body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body.get("args"), dict) or not isinstance(body.get("call_id"), str):
            raise ValueError("args must be an object and call_id a string")
        session = self.session(body.get("session"), need_client=False)
        return {"relations": [r.to_dict() for r in self._observe(session, body)]}

    @staticmethod
    def _outdated(session: Session, call_id: str | None) -> str:
        relation = session.relations.status(call_id) if call_id else None
        if relation is None:
            return ""
        if relation.kind == "superseded":
            return f"superseded at turn {relation.turn}: {relation.reason}"
        return f"{relation.reason} (turn {relation.turn})"

    def admit(self, body: dict[str, Any]) -> dict[str, Any]:
        text, task = body.get("text"), body.get("task")
        mode = body.get("mode", "on")
        if not isinstance(text, str) or not isinstance(task, str) or not task.strip():
            raise ValueError("text and task are required strings")
        if mode not in {"shadow", "on"}:
            raise ValueError("mode must be shadow or on")
        turn = body.get("turn", 0)
        if type(turn) is not int or turn < 0:
            raise ValueError("turn must be a non-negative integer")
        cap = body.get("max_elide_fraction", self.config.max_elide_fraction)
        if isinstance(cap, bool) or not isinstance(cap, int | float) or not 0 <= cap <= 1:
            raise ValueError("max_elide_fraction must be between 0 and 1")
        profile = body.get("profile", self.config.profile)
        if not isinstance(profile, bool):
            raise ValueError("profile must be a boolean")
        tool = str(body.get("tool") or "tool")
        call_id = str(body.get("call_id") or "")
        session = self.session(body.get("session"))
        session.task = task
        self._observe(session, body)
        # 1.0 turns the tripwire off: no result can elide more than all of itself.
        config = replace(self.config, shadow_only=mode == "shadow", max_elide_fraction=float(cap),
                         profile=profile)
        try:
            result = admit(
                text, Origin(source=f"tool:{tool}", ref=call_id or None, turn=turn),
                task_digest=task[:TASK_DIGEST_CHARS], turn=turn, client=session.client,
                store=session.store, log=session.log, config=config,
            )
        except JevError:
            # Fail open: an unreachable scorer must never cost the agent its output.
            with session.lock:
                session.admits += 1
                session.errors += 1
            raise
        with session.lock:
            session.admits += 1
            session.gated += bool(result.pointers)
            session.tripwires += result.tripwire is not None
            session.quarantined += result.meta.get("quarantined", 0)
            session.original_tokens += result.original_tokens
            session.result_tokens += result.result_tokens
        return {
            "text": result.text, "gated": result.gated, "tripwire": result.tripwire,
            "original_tokens": result.original_tokens, "result_tokens": result.result_tokens,
            "pointers": [p.id for p in result.pointers],
        }

    def expand(self, body: dict[str, Any]) -> dict[str, Any]:
        record_id, turn = body.get("id"), body.get("turn", 0)
        if not isinstance(record_id, str) or type(turn) is not int:
            raise ValueError("id must be a string and turn an integer")
        session = self.session(body.get("session"), need_client=False)
        text = expand(record_id, store=session.store, log=session.log, turn=turn)
        with session.lock:
            session.expands += 1
        record = session.store.get(record_id)
        call_id = record.origin.ref if record is not None else None
        note = note_for(session.relations.status(call_id)) if call_id else ""
        return {"text": f"{note}\n{text}" if note else text}

    def recall(self, body: dict[str, Any]) -> dict[str, Any]:
        query, turn, k = body.get("query"), body.get("turn", 0), body.get("k", 3)
        if not isinstance(query, str) or not query.strip() or type(turn) is not int \
                or type(k) is not int:
            raise ValueError("query must be a nonempty string, turn and k integers")
        filters: dict[str, str] = {}
        for field_name, key, allowed in (("type", "type_", TYPE_QUESTION.criteria),
                                         ("role", "role", ROLE_QUESTION.criteria),
                                         ("name", "name", None), ("source", "source", None)):
            value = body.get(field_name)
            if value in (None, ""):
                continue
            if not isinstance(value, str) or (allowed is not None and value not in allowed):
                raise ValueError(f"invalid {field_name}")
            filters[key] = value
        session = self.session(body.get("session"))
        hits = recall(query, store=session.store, client=session.client, log=session.log,
                      turn=turn, task=session.task, k=k,
                      outdated=lambda r: self._outdated(session, r.origin.ref), **filters)
        with session.lock:
            session.recalls += 1
        return {"text": render_hits(hits),
                "hits": [{"id": h.record.id, "score": h.score, "truncated": h.truncated}
                         for h in hits]}

    def workarea(self, body: dict[str, Any]) -> dict[str, Any]:
        turn, recent, raw = body.get("turn", 0), body.get("recent", ""), body.get("items")
        if type(turn) is not int or not isinstance(recent, str) or not isinstance(raw, list):
            raise ValueError("turn must be an integer, recent a string and items a list")
        items = []
        for entry in raw:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) \
                    or entry.get("kind") not in ("tool", "other") \
                    or type(entry.get("tokens")) is not int:
                raise ValueError("each item needs a string id, kind tool|other, integer tokens")
            text = entry.get("text")
            items.append(TailItem(id=entry["id"], kind=entry["kind"], tokens=entry["tokens"],
                                  text=text if isinstance(text, str) else None,
                                  tool=str(entry.get("tool") or "tool"),
                                  call_id=str(entry.get("call_id") or "") or None))
        session = self.session(body.get("session"))
        task = body.get("task") if isinstance(body.get("task"), str) else session.task
        with session.lock:
            if session.workarea is None:
                session.workarea = WorkArea(self.workarea_config)
            decision = session.workarea.decide(
                items, task=task or "", recent=recent, turn=turn, client=session.client,
                store=session.store, log=session.log,
                outdated=lambda call_id: self._outdated(session, call_id))
        return {"replacements": decision.replacements, "decision": decision.to_dict()}

    def stats(self, name: str) -> dict[str, Any]:
        session = self.session(name, need_client=False)
        usage = getattr(session.client, "usage", None)
        return {
            "admits": session.admits, "gated": session.gated,
            "tripwires": session.tripwires, "quarantined": session.quarantined,
            "expands": session.expands, "recalls": session.recalls,
            "errors": session.errors, "original_tokens": session.original_tokens,
            "result_tokens": session.result_tokens,
            "saved_tokens": session.original_tokens - session.result_tokens,
            "jev_input_tokens": getattr(usage, "input_tokens", 0),
            "jev_output_tokens": getattr(usage, "output_tokens", 0),
            "jev_requests": getattr(usage, "requests", 0),
            "workarea": dict(session.workarea.stats) if session.workarea else None,
            "relations": dict(Counter(
                found.kind for found in map(session.relations.status, session.relations.relations)
                if found is not None)),
        }


def make_server(state: SidecarState, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 - http.server naming
            url = urlparse(self.path)
            try:
                if url.path == "/health":
                    self._reply(200, {"ok": True})
                elif url.path == "/stats":
                    self._reply(200, state.stats(parse_qs(url.query).get("session", [""])[0]))
                else:
                    self._reply(404, {"error": "not found"})
            except ValueError as exc:
                self._reply(400, {"error": str(exc)})

        def do_POST(self) -> None:  # noqa: N802
            routes = {"/admit": state.admit, "/expand": state.expand, "/recall": state.recall,
                      "/workarea": state.workarea, "/observe": state.observe}
            route = routes.get(urlparse(self.path).path)
            if route is None:
                self._reply(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("expected a JSON object")
                self._reply(200, route(body))
            except KeyError as exc:
                self._reply(404, {"error": str(exc)})
            except (ValueError, json.JSONDecodeError) as exc:
                self._reply(400, {"error": str(exc)})
            except JevError as exc:
                # Name the class only: messages can echo request content.
                self._reply(502, {"error": type(exc).__name__})

        def log_message(self, *_args: Any) -> None:
            pass

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address; anything that can reach it can read the store")
    parser.add_argument("--port", type=int, default=8765, help="0 picks a free port")
    parser.add_argument("--data-dir", type=Path, help="Persist stores/logs here (default: memory)")
    parser.add_argument("--keep-threshold", type=float, default=GateConfig.keep_threshold)
    parser.add_argument("--min-gate-tokens", type=int, default=GateConfig.min_gate_tokens)
    parser.add_argument("--max-elide-fraction", type=float, default=GateConfig.max_elide_fraction)
    parser.add_argument("--profile", action="store_true",
                        help="Ask the jevctx.profile dimensions with every admit")
    parser.add_argument("--gate-on", default="keep",
                        help='"keep" or "role:<name>", e.g. role:change_site (needs --profile)')
    parser.add_argument("--protected-roles", nargs="*", default=[],
                        help="Profiled roles never to elide, e.g. change_site evidence")
    parser.add_argument("--price-input", type=float, default=WorkAreaConfig.price_input,
                        help="Host model USD per 1M input tokens, for /workarea's arithmetic")
    parser.add_argument("--price-cache-read", type=float, default=WorkAreaConfig.price_cache_read,
                        help="Host model USD per 1M cached input tokens")
    parser.add_argument("--expected-turns", type=int, default=WorkAreaConfig.expected_turns)
    parser.add_argument("--thresholds", type=Path,
                        help="JSON from `python -m jevctx.calibrate --out`; overrides --keep-threshold")
    args = parser.parse_args(argv)
    keep_threshold, thresholds = args.keep_threshold, {}
    if args.thresholds is not None:
        fitted = json.loads(args.thresholds.read_text(encoding="utf-8"))
        keep_threshold, thresholds = fitted["keep_threshold"], fitted.get("thresholds", {})
    config = GateConfig(keep_threshold=keep_threshold, thresholds=thresholds,
                        profile=args.profile, protected_roles=frozenset(args.protected_roles),
                        gate_on=args.gate_on,
                        min_gate_tokens=args.min_gate_tokens,
                        max_elide_fraction=args.max_elide_fraction)
    workarea = WorkAreaConfig(price_input=args.price_input, price_cache_read=args.price_cache_read,
                              expected_turns=args.expected_turns)
    server = make_server(SidecarState(args.data_dir, config, workarea=workarea),
                         host=args.host, port=args.port)
    # The extension reads this line to find a port picked with --port 0.
    print(json.dumps({"listening": server.server_address[1]}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
