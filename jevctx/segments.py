"""Deterministic segmentation of raw tool output.

Splitting on lines is wrong for most of what an agent reads. A stack trace cut in
half is no longer a stack trace; a JSON record cut mid-value is not parseable; a
table row without its header is unreadable. This module exists so that the scoring
layer is handed units that *mean* something on their own.

No model call happens here. Everything is pure, deterministic code.

**Losslessness is the invariant.** ``"".join(s.text for s in segment(t, o)) == t``
for every input, always. It is guaranteed structurally rather than by care: each
splitter returns only a set of *cut points* into ``text.splitlines(keepends=True)``,
and the segments are derived from those cuts, so the pieces tile the input exactly
by construction. A splitter cannot lose a byte even if its heuristics are wrong.

One consequence worth stating: a table's header is *not* copied into each row
segment, because that would duplicate bytes and break the invariant. It goes in
``meta["header"]`` instead, so the scorer can prepend it when building state without
corrupting the text that gets stored and later expanded.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from jevctx.tokens import estimate_tokens
from jevctx.types import Origin, Segment, SegmentKind, content_id

__all__ = ["segment", "detect_kind", "find_trace_regions"]

# Grouping targets. Segments much smaller than these cost more in question
# boilerplate than the text they gate, so rows and log lines are grouped.
_ROWS_PER_SEGMENT = 20
_LOG_LINES_PER_SEGMENT = 25

_PY_TRACE_HEADER = re.compile(r"^\s*Traceback \(most recent call last\):\s*$")
_PY_FRAME = re.compile(r'^\s+File "[^"]*", line \d+')
_JS_FRAME = re.compile(r"^\s+at\s+\S")
_JAVA_FRAME = re.compile(r"^\s+at\s+[\w$./]+\(")
_JAVA_CAUSE = re.compile(r"^\s*(Caused by|\.\.\. \d+ more|Suppressed)")
_EXC_LINE = re.compile(r"^[\w.]*(Error|Exception|Throwable)\b.*")
_DIFF_FILE = re.compile(r"^diff --git ")
_DIFF_HUNK = re.compile(r"^@@ -\d+(,\d+)? \+\d+(,\d+)? @@")
_LOG_PREFIX = re.compile(
    r"^\s*(\[?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"          # ISO timestamp
    r"|\[?\d{2}:\d{2}:\d{2}"                              # bare clock time
    r"|(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|TRACE)\b"      # level first
    r"|npm\s|pip\s|yarn\s)"
)
_LOG_LEVEL = re.compile(r"\b(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|TRACE)\b")
_CODE_TOP_LEVEL = re.compile(
    r"^(def |class |async def |function |export |public |private |protected |"
    r"@[\w.]+\s*$|const \w+ = \(|var \w+ = function)"
)
_CODE_EXTENSIONS = (
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".h",
    ".cpp", ".rb", ".php", ".cs", ".swift", ".kt", ".scala",
)


# --------------------------------------------------------------------------- #
# Cut points -> tiling
# --------------------------------------------------------------------------- #


def _ranges_from_cuts(cuts: Sequence[int], n_lines: int) -> list[tuple[int, int]]:
    """Turn a set of segment-start line indices into an exact tiling of [0, n_lines)."""
    if n_lines <= 0:
        return []
    starts = sorted({c for c in cuts if 0 < c < n_lines} | {0})
    bounds = [*starts, n_lines]
    return [(bounds[i], bounds[i + 1] - 1) for i in range(len(bounds) - 1)]


# --------------------------------------------------------------------------- #
# Stack traces -- found first, because they are the one thing never split
# --------------------------------------------------------------------------- #


def _is_frame(line: str) -> bool:
    return bool(_PY_FRAME.match(line) or _JS_FRAME.match(line) or _JAVA_FRAME.match(line))


def find_trace_regions(lines: Sequence[str]) -> list[tuple[int, int]]:
    """Inclusive line-index ranges holding a stack trace.

    Found before any other segmentation so that a trace embedded in a log is still
    kept whole. Splitting a trace destroys it, so these regions are never divided,
    not even by ``max_segment_tokens``.
    """
    regions: list[tuple[int, int]] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip("\r\n")
        start: int | None = None
        if _PY_TRACE_HEADER.match(line):
            start = i
        elif _EXC_LINE.match(line) and i + 1 < n and _is_frame(lines[i + 1].rstrip("\r\n")):
            start = i
        elif _is_frame(line):
            start = i
        if start is None:
            i += 1
            continue

        j = start
        last_frame = start
        while j + 1 < n:
            nxt = lines[j + 1].rstrip("\r\n")
            if _is_frame(nxt) or _JAVA_CAUSE.match(nxt):
                j += 1
                last_frame = j
            elif nxt.strip() == "" and j + 2 < n and _is_frame(lines[j + 2].rstrip("\r\n")):
                j += 2
                last_frame = j
            elif nxt.startswith((" ", "\t")) and nxt.strip():
                j += 1  # continuation of the previous frame (source echo, ``^^^^``)
            elif _EXC_LINE.match(nxt) and last_frame > start:
                j += 1
                last_frame = j
                break
            else:
                break
        if last_frame > start or _PY_TRACE_HEADER.match(line):
            regions.append((start, last_frame))
            i = last_frame + 1
        else:
            i = start + 1
    return regions


# --------------------------------------------------------------------------- #
# Kind detection
# --------------------------------------------------------------------------- #


def detect_kind(text: str, origin: Origin) -> SegmentKind:
    """Classify a region. A wrong kind is acceptable; a lossy split is not."""
    stripped = text.strip()
    if not stripped:
        return "text"

    if stripped[0] in "[{" and stripped[-1] in "]}":
        try:
            json.loads(stripped)
            return "json"
        except ValueError:
            pass

    lines = [ln.rstrip("\r\n") for ln in text.splitlines()]
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return "text"

    if any(_DIFF_FILE.match(ln) or _DIFF_HUNK.match(ln) for ln in lines):
        return "diff"
    if find_trace_regions(text.splitlines(keepends=True)):
        return "stacktrace"
    if len(non_empty) >= 3 and _looks_tabular(non_empty):
        return "table"
    if sum(1 for ln in non_empty if _LOG_PREFIX.match(ln)) / len(non_empty) >= 0.6:
        return "log"
    if (origin.ref or "").endswith(_CODE_EXTENSIONS):
        return "code"
    if sum(1 for ln in non_empty if _CODE_TOP_LEVEL.match(ln)) >= 2:
        return "code"
    return "text"


def _looks_tabular(non_empty: Sequence[str]) -> bool:
    for sep in ("|", "\t"):
        counts = [ln.count(sep) for ln in non_empty]
        if min(counts) >= 1 and len(set(counts)) <= 2:
            return True
    return False


# --------------------------------------------------------------------------- #
# Per-kind cut points
# --------------------------------------------------------------------------- #


def _cuts_json(lines: Sequence[str]) -> list[int]:
    """Cut between top-level elements, tracking depth outside string literals."""
    cuts: list[int] = []
    depth = 0
    in_string = False
    escaped = False
    entered = False
    for index, raw in enumerate(lines):
        line_started_at_top = depth <= 1 and entered
        for ch in raw:
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "[{":
                depth += 1
                entered = True
            elif ch in "]}":
                depth -= 1
        if line_started_at_top and depth >= 1:
            cuts.append(index)
    return _merge_structural(lines, cuts)


def _merge_structural(lines: Sequence[str], cuts: Sequence[int]) -> list[int]:
    """Drop cuts that would isolate a line of pure JSON punctuation.

    A lone ``[`` scored and relocated on its own would cost a whole pointer to save
    one character, so such a range is merged into its neighbour -- backwards where
    there is one, forwards when it is the first.
    """
    ranges = _ranges_from_cuts(cuts, len(lines))

    def structural(span: tuple[int, int]) -> bool:
        return not "".join(lines[span[0]:span[1] + 1]).strip(" \t\r\n[]{},:")

    merged: list[tuple[int, int]] = []
    for span in ranges:
        if merged and structural(span):
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)
    while len(merged) > 1 and structural(merged[0]):
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)
    return [start for start, _ in merged[1:]]


def _cuts_diff(lines: Sequence[str]) -> list[int]:
    return [i for i, ln in enumerate(lines)
            if _DIFF_FILE.match(ln) or _DIFF_HUNK.match(ln)]


def _table_header_size(lines: Sequence[str]) -> int:
    size = 1
    if len(lines) > 1 and re.match(r"^[\s|:+-]+$", lines[1].rstrip("\r\n")):
        size = 2
    return size


def _cuts_table(lines: Sequence[str]) -> list[int]:
    header = _table_header_size(lines)
    return list(range(header, len(lines), _ROWS_PER_SEGMENT))


def _cuts_log(lines: Sequence[str]) -> list[int]:
    """Cut at level changes, at blank-line boundaries, and every N lines otherwise.

    Blank lines matter as much as levels here: a log that groups its output into
    stanzas is telling you where one thing ended and the next began, and a segment
    that straddles that boundary mixes content the gate should judge separately.
    """
    cuts: list[int] = []
    current: str | None = None
    since_cut = 0
    previous_blank = False
    for index, raw in enumerate(lines):
        match = _LOG_LEVEL.search(raw)
        level = match.group(1) if match else None
        blank = raw.strip() == ""
        boundary = (
            (level is not None and current is not None and level != current)
            or (previous_blank and not blank)
            or since_cut >= _LOG_LINES_PER_SEGMENT
        )
        if index and boundary:
            cuts.append(index)
            since_cut = 0
        if level is not None:
            current = level
        previous_blank = blank
        since_cut += 1
    return cuts


def _cuts_code(lines: Sequence[str]) -> list[int]:
    cuts = [i for i, ln in enumerate(lines) if i and _CODE_TOP_LEVEL.match(ln.rstrip("\r\n"))]
    return cuts or _cuts_text(lines)


def _cuts_text(lines: Sequence[str]) -> list[int]:
    """Paragraph starts: the first non-blank line after a blank run."""
    cuts: list[int] = []
    previous_blank = False
    for index, raw in enumerate(lines):
        blank = raw.strip() == ""
        if index and previous_blank and not blank:
            cuts.append(index)
        previous_blank = blank
    return cuts


_CUTTERS = {
    "json": _cuts_json,
    "diff": _cuts_diff,
    "table": _cuts_table,
    "log": _cuts_log,
    "code": _cuts_code,
    "text": _cuts_text,
}


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def _split_oversized(lines: Sequence[str], span: tuple[int, int],
                     max_tokens: int) -> list[tuple[int, int]]:
    start, end = span
    if estimate_tokens("".join(lines[start:end + 1])) <= max_tokens:
        return [span]
    pieces: list[tuple[int, int]] = []
    current = start
    running = 0
    for index in range(start, end + 1):
        cost = estimate_tokens(lines[index])
        if running and running + cost > max_tokens:
            pieces.append((current, index - 1))
            current = index
            running = 0
        running += cost
    pieces.append((current, end))
    return pieces


def segment(text: str, origin: Origin, *, max_segment_tokens: int = 2000) -> list[Segment]:
    """Split ``text`` into scoreable units without losing a byte.

    Stack traces are located first and kept whole, so a trace embedded in a log
    survives intact. An oversized trace is marked ``meta["oversized"]`` rather than
    split, because a partial trace is worse than a large one.
    """
    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    traces = find_trace_regions(lines)
    regions: list[tuple[int, int, bool]] = []
    cursor = 0
    for start, end in traces:
        if start > cursor:
            regions.append((cursor, start - 1, False))
        regions.append((start, end, True))
        cursor = end + 1
    if cursor < len(lines):
        regions.append((cursor, len(lines) - 1, False))

    spans: list[tuple[int, int, SegmentKind, bool]] = []
    for start, end, is_trace in regions:
        if is_trace:
            spans.append((start, end, "stacktrace", False))
            continue
        chunk_lines = lines[start:end + 1]
        kind = detect_kind("".join(chunk_lines), origin)
        if kind == "stacktrace":  # detected inside a region that was not matched whole
            kind = "text"
        cuts = _CUTTERS[kind](chunk_lines)
        for local_start, local_end in _ranges_from_cuts(cuts, len(chunk_lines)):
            spans.append((start + local_start, start + local_end, kind, True))

    header_text = ""
    if spans and spans[0][2] == "table":
        header = _table_header_size(lines[spans[0][0]:spans[0][1] + 1])
        header_text = "".join(lines[spans[0][0]:spans[0][0] + header])

    segments: list[Segment] = []
    for start, end, kind, splittable in spans:
        pieces = _split_oversized(lines, (start, end), max_segment_tokens) \
            if splittable else [(start, end)]
        was_split = len(pieces) > 1
        for piece_start, piece_end in pieces:
            body = "".join(lines[piece_start:piece_end + 1])
            tokens = estimate_tokens(body)
            meta: dict[str, object] = {}
            if was_split:
                meta["split"] = True
            if not splittable and tokens > max_segment_tokens:
                meta["oversized"] = True
            if kind == "table" and header_text and not body.startswith(header_text):
                meta["header"] = header_text
            segments.append(Segment(
                id=content_id(body, salt=origin.ref or ""),
                text=body, kind=kind, tokens=tokens, origin=origin,
                line_span=(piece_start + 1, piece_end + 1), meta=meta,
            ))
    return segments
