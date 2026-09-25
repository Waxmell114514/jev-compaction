"""The demo is how most people will meet this repo, so it gets a test like anything else."""

from __future__ import annotations

import re

import demo

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def test_demo_runs_offline_and_shows_each_mechanism(capsys, monkeypatch) -> None:
    for var in ("TYPESAFE_API_KEY", "JEV_API_KEY", "JEV_BASE_URL", "OPENROUTER_API_KEY",
                "JEVCTX_JUDGE"):
        monkeypatch.delenv(var, raising=False)
    assert demo.main() == 0
    out = _ANSI.sub("", capsys.readouterr().out)
    for act in ("ACT 1", "ACT 2", "ACT 3", "ACT 4", "ACT 5", "ACT 6"):
        assert act in out
    assert "No Jev key" in out, "the stand-in must announce itself"
    assert "planted instruction reached the model: False" in out
    assert "byte-exact: True" in out
    assert "AssertionError: parse_month('13') did not raise ValueError" in out.split("ACT 3")[1]
    assert "r1 stale" in out and "t1 superseded" in out
    work_area = out.split("ACT 5")[1].split("ACT 6")[0]
    assert "(10:1): compact" in work_area and "(50:1): commit" in work_area


def test_the_stand_in_labels_like_a_reader() -> None:
    """Offline, the demo is only as convincing as its stand-in's labels."""
    class Profiled:
        instructions = "a profile question"

    state = {"items": [{"ref": "s1", "text": "def parse_month(value):\n    return int(value)"}]}
    assert demo.judge(state, {"s1:role": Profiled()}, "s1:role") == "change_site"
    assert demo.judge(state, {"s1:type": Profiled()}, "s1:type") == "source_code"
    assert demo.judge(state, {"s1:injection": Profiled()}, "s1:injection") < 0.1
