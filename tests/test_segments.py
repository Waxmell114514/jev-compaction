"""Tests for deterministic segmentation.

The losslessness test is the one that matters: every other property of this module
can be wrong without losing information, and that one cannot.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from jevctx.segments import detect_kind, find_trace_regions, segment
from jevctx.tokens import estimate_tokens
from jevctx.types import Origin

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
FIXTURE_FILES = sorted(p.name for p in FIXTURES.iterdir())

ADVERSARIAL = [
    "",
    "\n",
    "   ",
    "\n\n\n",
    "no trailing newline",
    "crlf line one\r\ncrlf line two\r\n",
    "x" * 20_000,
    "a\n" * 500,
    "﻿{\"k\": 1}\n",
    "中文输出的一行\n另一行中文\n",
]


def origin_for(name: str) -> Origin:
    return Origin(source="tool:bash", ref=name, turn=1)


def read(name: str) -> str:
    return (FIXTURES / name).read_text()


# --------------------------------------------------------------------------- #
# The invariant
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_segmentation_is_lossless_over_fixtures(name: str) -> None:
    text = read(name)
    assert "".join(s.text for s in segment(text, origin_for(name))) == text


@pytest.mark.parametrize("text", ADVERSARIAL)
def test_segmentation_is_lossless_over_adversarial_input(text: str) -> None:
    assert "".join(s.text for s in segment(text, origin_for("adv"))) == text


@pytest.mark.parametrize("name", FIXTURE_FILES)
@pytest.mark.parametrize("max_tokens", [10, 50, 2000])
def test_segmentation_is_lossless_at_every_size_cap(name: str, max_tokens: int) -> None:
    text = read(name)
    segments = segment(text, origin_for(name), max_segment_tokens=max_tokens)
    assert "".join(s.text for s in segments) == text


def test_empty_input_yields_no_segments() -> None:
    assert segment("", origin_for("empty")) == []


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("python_traceback.txt", "stacktrace"),
        ("js_stack.txt", "stacktrace"),
        ("java_stack.txt", "stacktrace"),
        ("changes.diff", "diff"),
        ("records.json", "json"),
        ("config.json", "json"),
        ("results.md", "table"),
        ("deps.tsv", "table"),
        ("app.log", "log"),
        ("npm_install.log", "log"),
        ("sample.py", "code"),
        ("prose.txt", "text"),
    ],
)
def test_detect_kind(name: str, expected: str) -> None:
    assert detect_kind(read(name), origin_for(name)) == expected


# --------------------------------------------------------------------------- #
# Per-kind behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["python_traceback.txt", "js_stack.txt", "java_stack.txt"])
@pytest.mark.parametrize("max_tokens", [5, 20, 2000])
def test_a_stack_trace_is_always_exactly_one_segment(name: str, max_tokens: int) -> None:
    """A partial trace is worse than a large one, so the size cap never splits it."""
    segments = segment(read(name), origin_for(name), max_segment_tokens=max_tokens)
    assert len(segments) == 1
    assert segments[0].kind == "stacktrace"


def test_an_oversized_trace_is_marked_rather_than_split() -> None:
    name = "python_traceback.txt"
    segments = segment(read(name), origin_for(name), max_segment_tokens=5)
    assert segments[0].meta.get("oversized") is True
    assert segments[0].meta.get("split") is None


def test_a_trace_embedded_in_a_log_survives_whole() -> None:
    name = "mixed_log_trace.txt"
    segments = segment(read(name), origin_for(name))
    traces = [s for s in segments if s.kind == "stacktrace"]
    assert len(traces) == 1
    assert traces[0].text.startswith("Traceback (most recent call last):")
    assert traces[0].text.rstrip().endswith("ValueError: bad manifest")
    assert any(s.kind == "log" for s in segments)


def test_each_diff_hunk_keeps_its_header() -> None:
    name = "changes.diff"
    segments = segment(read(name), origin_for(name))
    hunks = [s for s in segments if s.text.lstrip().startswith("@@")]
    assert len(hunks) == 2
    assert all(s.text.lstrip().startswith("@@ -") for s in hunks)


def test_json_array_splits_per_element_and_each_piece_is_parseable() -> None:
    name = "records.json"
    segments = segment(read(name), origin_for(name))
    assert len(segments) == 3
    for seg in segments:
        body = seg.text.strip().strip("[],")
        assert json.loads(body)["id"] in (1, 2, 3)


def test_table_header_travels_in_meta_not_in_the_text() -> None:
    """Duplicating the header into every row would break losslessness."""
    name = "results.md"
    segments = segment(read(name), origin_for(name))
    rows = [s for s in segments if "react" in s.text]
    assert rows, "expected a segment holding the data rows"
    assert rows[0].meta["header"].startswith("| package")
    assert rows[0].text.count("| package") == 0


# --------------------------------------------------------------------------- #
# Segment metadata
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_line_spans_are_contiguous_and_reconstruct_the_original(name: str) -> None:
    text = read(name)
    lines = text.splitlines(keepends=True)
    segments = segment(text, origin_for(name))

    assert segments[0].line_span[0] == 1
    assert segments[-1].line_span[1] == len(lines)
    for previous, nxt in zip(segments, segments[1:], strict=False):
        assert nxt.line_span[0] == previous.line_span[1] + 1
    for seg in segments:
        start, end = seg.line_span
        assert "".join(lines[start - 1:end]) == seg.text


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_ids_are_stable_across_runs_and_vary_with_origin(name: str) -> None:
    text = read(name)
    first = segment(text, origin_for(name))
    second = segment(text, origin_for(name))
    assert [s.id for s in first] == [s.id for s in second]

    other = segment(text, Origin(source="tool:bash", ref="somewhere-else", turn=1))
    assert [s.id for s in first] != [s.id for s in other]


def test_oversized_splittable_segments_are_marked_and_capped() -> None:
    text = "".join(f"line number {i} with some filler content to spend tokens\n"
                   for i in range(400))
    segments = segment(text, origin_for("big.log"), max_segment_tokens=100)

    assert len(segments) > 1
    assert any(s.meta.get("split") for s in segments)
    # One line can always exceed the cap on its own; everything wider must not.
    assert all(s.tokens <= 100 or s.line_span[0] == s.line_span[1] for s in segments)


def test_tokens_agree_with_the_estimator() -> None:
    name = "app.log"
    for seg in segment(read(name), origin_for(name)):
        assert seg.tokens == estimate_tokens(seg.text)


def test_find_trace_regions_returns_disjoint_ordered_ranges() -> None:
    text = read("mixed_log_trace.txt") + read("python_traceback.txt")
    regions = find_trace_regions(text.splitlines(keepends=True))
    assert len(regions) == 2
    assert regions[0][1] < regions[1][0]


def test_a_single_enormous_line_is_one_segment_and_is_not_lost() -> None:
    text = "x" * 50_000 + "\n"
    segments = segment(text, origin_for("huge.txt"), max_segment_tokens=100)
    assert len(segments) == 1
    assert segments[0].text == text
