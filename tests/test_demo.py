"""The demo is the point of this repo, so it gets a test like anything else."""

from __future__ import annotations

import re

import demo

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def test_demo_runs_offline(capsys, monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert demo.main() == 0

    out = _ANSI.sub("", capsys.readouterr().out)
    for act in ("ACT 1", "ACT 2", "ACT 3", "ACT 4", "ACT 5", "ACT 6"):
        assert act in out
    assert "No TYPESAFE_API_KEY" in out, "the stand-in must announce itself"
    assert "BROKEN" not in out, "the frozen prefix must stay intact for every turn"
    assert "byte-exact: True" in out


def test_the_stand_in_spreads_scores() -> None:
    """A flat scorer would make the threshold-replay table meaningless."""
    samples = [
        demo.stand_in_judgement("npm ERROR peer react@^18.0.0 required"),
        demo.stand_in_judgement("added 412 packages, audited 1204 packages in 9s"),
        demo.stand_in_judgement("npm WARN deprecated glob@7.2.1: no longer supported"),
        demo.stand_in_judgement("npm notice created a lockfile entry"),
        demo.stand_in_judgement("npm http fetch GET 200 https://registry.npmjs.org/x"),
    ]
    assert samples == sorted(samples, reverse=True)
    assert len(set(samples)) == 5
