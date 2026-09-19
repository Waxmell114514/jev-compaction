"""The showcase page must not drift away from the implementation.

Every figure on docs/showcase.html is generated, not hand-copied. If the
implementation changes and the page isn't regenerated, the page starts telling
people something that is no longer true -- so that is a test failure, not a
documentation chore.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "showcase.html"
GENERATOR = ROOT / "tools" / "build_showcase_data.py"


def embedded() -> dict:
    match = re.search(r"const DATA = (\{.*?\});", PAGE.read_text(), re.S)
    assert match, "docs/showcase.html has no embedded DATA block"
    return json.loads(match.group(1))


def regenerated() -> dict:
    result = subprocess.run([sys.executable, str(GENERATOR)], capture_output=True,
                            text=True, cwd=ROOT, check=True)
    return json.loads(result.stdout)


def test_the_page_still_matches_the_implementation() -> None:
    assert embedded() == regenerated(), (
        "docs/showcase.html is stale -- regenerate it with tools/build_showcase_data.py"
    )


def test_the_page_carries_the_numbers_it_claims() -> None:
    data = embedded()
    assert data["cost"]["naive"] > data["cost"]["chunked"]
    assert all(t["intact"] for t in data["turns"]), "a turn broke the frozen prefix"
    assert sum(d["expanded"] for d in data["log"]["decisions"]) > 0, (
        "the threshold explorer needs real false negatives to be worth dragging"
    )
    assert len({d["score"] for d in data["log"]["decisions"]}) >= 4, (
        "a flat score distribution makes the threshold table meaningless"
    )
