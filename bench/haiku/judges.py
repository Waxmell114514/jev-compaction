"""Claude as the judge, on the native Messages API, in two request layouts.

``flat`` is :class:`jevctx.judge.LLMJudgeClient`'s request: the system prompt, then the
state and every question written out in full. Each item's questions repeat the same
instructions and options, and nothing is cached.

``cached`` asks the same questions laid out for the prompt cache (``compact`` also asks
for a short reply: output costs five times input, and a full distribution over every
option of every choice question is most of a reply):

    system   the answer rules + a catalog of the distinct question templates   [cached]
    user     TASK: the task digest                                             [cached]
             ITEMS: the rest of the state; QUESTIONS: one short line per key,
             naming its template and the item it is about

Requests of one kind (all profile batches, say) share the system block byte for byte,
and every request within one task shares the task block too, so both are read at the
cache price after the first request. Only the items, and a few tokens per question,
are new input.

Both record metered usage (uncached input, cache reads and writes, output) and price
it at Claude Haiku 5.5's sheet.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import anthropic

from jevctx.jev import _to_jsonable, check_request_budget
from jevctx.judge import SYSTEM_PROMPT, _answer, _json_object, _Malformed, _question_payload
from jevctx.types import Answer, JevUnavailableError, Question, State

PRICES = {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125}
_ABOUT = re.compile(r"^Considering item (\S+) only: ")

CATALOG_RULES = """

Every question below names a TEMPLATE from this catalog, and usually the item it is \
about: answer the template's question for that item. A template's criteria are its \
options (choice), its true/false criteria (noul), or its levels numbered from 0 (score)."""

COMPACT_RULES = """

Keep the reply short: one line of JSON with no spaces, probabilities with at most two \
decimals, and in every "probabilities" object only the options with probability 0.05 or \
more (the rest count as 0)."""


@dataclass
class JudgeUsage:
    input_tokens: int = 0          # every input token, cached or not (what Jev would bill)
    uncached: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output_tokens: int = 0
    requests: int = 0
    failures: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, usage: Any) -> None:
        with self._lock:
            read = usage.cache_read_input_tokens or 0
            write = usage.cache_creation_input_tokens or 0
            self.uncached += usage.input_tokens
            self.cache_read += read
            self.cache_write += write
            self.input_tokens += usage.input_tokens + read + write
            self.output_tokens += usage.output_tokens
            self.requests += 1

    def cost_usd(self) -> float:
        return (self.uncached * PRICES["input"] + self.cache_read * PRICES["cache_read"]
                + self.cache_write * PRICES["cache_write"]
                + self.output_tokens * PRICES["output"]) / 1e6

    def snapshot(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("input_tokens", "uncached", "cache_read",
                                               "cache_write", "output_tokens", "requests",
                                               "failures")} | {"cost_usd": round(self.cost_usd(), 6)}


def _template(question: Question) -> tuple[str | None, dict[str, Any]]:
    """(the item it is about, the question without that reference)."""
    payload = _question_payload(question)
    match = _ABOUT.match(payload["instructions"])
    about = match.group(1) if match else None
    if match:
        payload = {**payload, "instructions": payload["instructions"][match.end():]}
    return about, payload


class AnthropicJudge:
    """A ``JevClient`` that asks Claude, laid out ``flat`` or ``cached``."""

    def __init__(self, layout: str = "cached", *, model: str = "claude-haiku-5-5",
                 effort: str = "low", client: anthropic.Anthropic | None = None,
                 max_parse_retries: int = 2) -> None:
        if layout not in {"flat", "cached", "compact"}:
            raise ValueError("layout is flat, cached or compact")
        self.layout, self.model, self.effort = layout, model, effort
        self.client = client or anthropic.Anthropic(max_retries=6, timeout=120.0)
        self.max_parse_retries = max_parse_retries
        self.usage = JudgeUsage()

    # -- the request ------------------------------------------------------------- #

    def request(self, state: State, questions: Mapping[str, Question]) -> dict[str, Any]:
        plain = _to_jsonable(state)
        if self.layout == "flat":
            text = plain if isinstance(plain, str) else json.dumps(plain, ensure_ascii=False)
            asked = {key: _question_payload(q) for key, q in questions.items()}
            return {"system": SYSTEM_PROMPT, "messages": [{"role": "user", "content": [
                {"type": "text", "text": f"STATE:\n{text}\n\nQUESTIONS:\n"
                                         f"{json.dumps(asked, ensure_ascii=False)}"}]}]}
        templates: dict[str, str] = {}
        lines = []
        for key, question in questions.items():
            about, payload = _template(question)
            body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            templates.setdefault(body, "")
            lines.append((key, body, about))
        # Number templates by their text, so one kind of request always gets one catalog.
        names = {body: f"T{i + 1}" for i, body in enumerate(sorted(templates))}
        catalog = "\n".join(f"{names[body]}: {body}" for body in sorted(templates))
        task, rest = None, plain
        if isinstance(plain, dict) and isinstance(plain.get("task"), str):
            task = plain["task"]
            rest = {k: v for k, v in plain.items() if k != "task"}
        rest_text = rest if isinstance(rest, str) else json.dumps(rest, ensure_ascii=False)
        asked = "\n".join(f"{key}: {names[body]}" + (f", about item {about}" if about else "")
                          for key, body, about in lines)
        content: list[dict[str, Any]] = []
        if task is not None:
            content.append({"type": "text", "text": f"TASK:\n{task}",
                            "cache_control": {"type": "ephemeral"}})
        content.append({"type": "text", "text": f"STATE:\n{rest_text}\n\nQUESTIONS:\n{asked}"})
        rules = CATALOG_RULES + (COMPACT_RULES if self.layout == "compact" else "")
        return {"system": [{"type": "text", "text": SYSTEM_PROMPT + rules
                            + "\n\nTEMPLATES:\n" + catalog,
                            "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": content}]}

    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        check_request_budget(state, questions)
        body = self.request(state, questions)
        last: Exception | None = None
        for _ in range(self.max_parse_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model, max_tokens=4000, thinking={"type": "disabled"},
                    output_config={"effort": self.effort}, **body)
            except anthropic.APIError as exc:
                last = exc
                break
            self.usage.record(response.usage)
            text = "".join(b.text for b in response.content if b.type == "text")
            try:
                replies = _json_object(text)
                missing = [key for key in questions if key not in replies]
                if missing:
                    raise _Malformed(f"no answer for {', '.join(missing[:5])}")
                return {key: _answer(q, replies[key]) for key, q in questions.items()}
            except _Malformed as exc:
                last = exc
        self.usage.failures += 1
        raise JevUnavailableError(f"judge failed: {type(last).__name__}") from last
