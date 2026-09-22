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


# --------------------------------------------------------------------------- #
# The animation
# --------------------------------------------------------------------------- #

ANIMATION = ROOT / "docs" / "index.html"


def test_the_animation_page_is_whole() -> None:
    """It shipped duplicated once already; structure is cheap to check."""
    page = ANIMATION.read_text()
    assert page.count("<h1>") == 1
    assert page.count("<footer>") == 1
    assert page.count("const TURNS") == 1
    assert "<title>The Context Gate</title>" in page


def test_the_animation_scores_are_well_formed() -> None:
    page = ANIMATION.read_text()
    commands = re.findall(r"cmd:'([^']+)'", page)
    assert len(commands) == 4, "four turns in the loop"
    assert len(set(commands)) == 4, "each turn should show a different tool"

    scores = [float(m) for m in re.findall(r"\{s:([0-9.]+),", page)]
    assert len(scores) == 20, "four turns of five segments"
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert len({round(s, 2) for s in scores}) > 8, (
        "a flat score distribution would make the scan look scripted"
    )


def test_the_animation_does_not_overclaim() -> None:
    """Only the first turn's numbers come from the implementation; say so."""
    page = ANIMATION.read_text()
    assert "scripted stand-in" in page
    assert "authored scenarios" in page


def test_the_readme_headline_numbers_come_from_the_data() -> None:
    """The first screen quotes numbers. Regenerating DATA must not leave it lying.

    Every figure in the README's opening paragraph is derived here from the same
    DATA the showcase renders, so a rebuild that moves a number fails loudly
    instead of turning the landing page into a false advertisement. Matching is
    done against the README with its line breaks flattened, because a phrase
    that straddles a wrap is still a phrase the reader sees.
    """
    data = embedded()
    readme = " ".join((ROOT / "README.md").read_text().split())

    gate = data["gate"]
    original, result = gate["originalTokens"], gate["resultTokens"]
    shrink = round((original - result) / original * 100)
    cost = original * data["cost"]["pricePerToken"]

    # At 0.10 nothing the agent later came back for would have been moved out.
    below = [d for d in data["log"]["decisions"] if d["score"] < 0.10]
    saved = sum(d["tokens"] for d in below)
    assert not any(d["expanded"] for d in below), (
        "the README claims nothing below 0.10 was ever expanded"
    )
    # ...which is only interesting because the shipped threshold does worse.
    assert data["log"]["liveFalseNegativeRate"] == 0.5, (
        "the README says half the elided segments got expanded again"
    )

    for claim in (
        f"**{original} to {result} tokens**",
        f"{shrink}% smaller",
        f"**${cost:.6f}**",
        f"**{saved:,} tokens",
        f"the {data['log']['liveThreshold']} the demo ships with",
    ):
        assert claim in readme, f"README no longer says {claim!r}"
