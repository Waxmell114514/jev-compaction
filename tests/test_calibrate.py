import json

import pytest

from jevctx.calibrate import UsageSample, curve, fit_thresholds, main
from jevctx.pipeline import GateConfig, admit
from jevctx.shadow import ShadowLog
from jevctx.store import InMemoryStore
from jevctx.types import Origin
from tests.test_pipeline import keep_noise_client, noisy_log


def samples(source: str, kind: str, used_scores, unused_scores):
    return ([UsageSample(source, kind, s, 100, True) for s in used_scores]
            + [UsageSample(source, kind, s, 100, False) for s in unused_scores])


def test_curve_counts_elided_tokens_and_lost_used_segments():
    points = {p.threshold: p for p in curve(samples("tool:read", "code", [0.9, 0.2], [0.1, 0.3]))}
    assert points[0.0].elided_share == 0 and points[0.0].loss == 0
    assert points[0.25].elided_share == 0.5 and points[0.25].loss == 0.5
    assert points[0.35].elided_share == 0.75


def test_fit_picks_the_highest_threshold_within_the_loss_budget_per_source():
    data = (samples("tool:read", "code", [0.6] * 20, [0.1] * 50)      # separable: elide freely
            + samples("tool:bash", "log", [0.1] * 10 + [0.6] * 10, [0.1] * 20))
    default, overrides = fit_thresholds(data, max_loss=0.1, min_used=10)
    assert overrides["tool:read"] > default
    assert overrides.get("tool:bash", default) <= 0.1


def test_thin_groups_inherit_instead_of_fitting_noise():
    data = samples("tool:read", "code", [0.6] * 20, [0.1] * 20) + samples("tool:read", "json", [0.1], [])
    default, overrides = fit_thresholds(data, max_loss=0.1, min_used=10)
    assert "tool:read/json" not in overrides


def test_fit_rejects_too_few_used_samples():
    with pytest.raises(ValueError):
        fit_thresholds(samples("tool:read", "code", [0.5], [0.1]), min_used=10)


def test_gate_uses_the_most_specific_threshold():
    config = GateConfig(keep_threshold=0.35, thresholds={"tool:read": 0.2, "tool:read/code": 0.1})
    assert config.threshold_for("tool:read", "code") == 0.1
    assert config.threshold_for("tool:read", "log") == 0.2
    assert config.threshold_for("tool:bash", "log") == 0.35


def test_a_lower_source_threshold_keeps_what_the_default_would_elide():
    raw, client = noisy_log(), keep_noise_client()   # noise scores 0.02, signal 0.95
    origin = Origin(source="tool:bash", ref="x", turn=1)
    common = dict(task_digest="t", turn=1, client=client, config=None)
    elided = admit(raw, origin, store=InMemoryStore(), log=ShadowLog(), **{**common, "config": GateConfig()})
    log = ShadowLog()
    kept = admit(raw, origin, store=InMemoryStore(), log=log,
                 **{**common, "config": GateConfig(thresholds={"tool:bash": 0.01})})
    assert elided.pointers and not kept.pointers
    assert {e["threshold"] for e in log.entries()} == {0.01}
    assert all(e["labels"]["segment_kind"] for e in log.entries())


def test_cli_writes_a_gate_config(tmp_path, capsys):
    path = tmp_path / "samples.jsonl"
    rows = samples("tool:read", "code", [0.6] * 20, [0.1] * 20)
    path.write_text("\n".join(json.dumps(r.__dict__) for r in rows))
    out = tmp_path / "thresholds.json"
    assert main([str(path), "--out", str(out)]) == 0
    fitted = json.loads(out.read_text())
    GateConfig(keep_threshold=fitted["keep_threshold"], thresholds=fitted["thresholds"])
