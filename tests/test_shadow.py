"""Tests for the decision log.

Two properties carry the weight: the log must never hold the full text it is
logging about, and ``replay`` must be able to recover what a different threshold
would have done -- including how many known false negatives it would still cause.
"""

from __future__ import annotations

import threading

import pytest

from jevctx.shadow import PREVIEW_CHARS, ShadowLog
from jevctx.types import Origin

ORIGIN = Origin(source="tool:bash", ref="npm install", turn=1)


def log_with_scores(scores: list[float], threshold: float = 0.35,
                    path=None) -> ShadowLog:
    log = ShadowLog(path=path)
    for index, score in enumerate(scores):
        log.decision(
            kind="admit", item_id=f"s:{index}", score=score, threshold=threshold,
            action="kept" if score >= threshold else "elided",
            tokens=10, origin=ORIGIN, text=f"content {index}", turn=1,
        )
    return log


# --------------------------------------------------------------------------- #
# What gets written
# --------------------------------------------------------------------------- #


def test_the_log_never_stores_the_full_text(tmp_path) -> None:
    path = tmp_path / "shadow.jsonl"
    log = ShadowLog(path=path)
    secret = "SENSITIVE" + "x" * 5000
    log.decision(kind="admit", item_id="s:1", score=0.1, threshold=0.35,
                 action="elided", tokens=99, origin=ORIGIN, text=secret, turn=1)

    written = path.read_text()
    assert secret not in written
    assert secret[:PREVIEW_CHARS] in written
    assert len(written) < 1500


def test_preview_is_capped(tmp_path) -> None:
    log = ShadowLog(path=tmp_path / "shadow.jsonl")
    log.decision(kind="admit", item_id="s:1", score=0.1, threshold=0.35,
                 action="elided", tokens=1, origin=ORIGIN, text="y" * 10_000, turn=1)
    assert len(log.entries()[0]["text_preview"]) == PREVIEW_CHARS


def test_an_in_memory_log_writes_no_file(tmp_path) -> None:
    log = ShadowLog(path=None)
    log.decision(kind="admit", item_id="s:1", score=0.9, threshold=0.35,
                 action="kept", tokens=1, origin=ORIGIN, text="x", turn=1)
    assert log.stats().total == 1
    assert list(tmp_path.iterdir()) == []


def test_load_round_trips(tmp_path) -> None:
    path = tmp_path / "shadow.jsonl"
    original = log_with_scores([0.1, 0.9, 0.2], path=path)
    original.outcome(kind="expand", item_id="s:0", turn=2)

    reloaded = ShadowLog.load(path)
    assert reloaded.stats() == original.stats()
    assert reloaded.false_negative_rate() == original.false_negative_rate()


def test_load_of_a_missing_file_is_empty(tmp_path) -> None:
    assert ShadowLog.load(tmp_path / "nothing.jsonl").stats().total == 0


def test_a_torn_final_line_is_skipped(tmp_path) -> None:
    path = tmp_path / "shadow.jsonl"
    log_with_scores([0.9, 0.1], path=path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "decision", "turn"')

    assert ShadowLog.load(path).stats().total == 2


# --------------------------------------------------------------------------- #
# The number that matters
# --------------------------------------------------------------------------- #


def test_false_negative_rate_is_zero_before_anything_is_elided() -> None:
    assert log_with_scores([0.9, 0.8]).false_negative_rate() == 0.0


def test_false_negative_rate_counts_expands_of_elided_items() -> None:
    log = log_with_scores([0.1, 0.1, 0.1, 0.9])  # three elided
    log.outcome(kind="expand", item_id="s:0", turn=2)
    assert log.false_negative_rate() == pytest.approx(1 / 3)


def test_expanding_a_kept_item_is_not_a_false_negative() -> None:
    log = log_with_scores([0.9, 0.1])
    log.outcome(kind="expand", item_id="s:0", turn=2)
    assert log.false_negative_rate() == 0.0


def test_repeated_expands_of_one_item_count_once() -> None:
    log = log_with_scores([0.1, 0.1])
    for _ in range(5):
        log.outcome(kind="expand", item_id="s:0", turn=2)
    assert log.false_negative_rate() == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Stats and replay
# --------------------------------------------------------------------------- #


def test_stats_reports_actions_and_tokens() -> None:
    stats = log_with_scores([0.9, 0.8, 0.1, 0.05]).stats()
    assert stats.total == 4
    assert stats.by_action["kept"] == 2
    assert stats.by_action["elided"] == 2
    assert stats.kept_tokens == 20
    assert stats.elided_tokens == 20
    assert stats.mean_score_by_action["kept"] == pytest.approx(0.85)


def test_replay_at_a_lower_threshold_keeps_strictly_more() -> None:
    log = log_with_scores([0.1, 0.3, 0.5, 0.7, 0.9])
    assert log.replay(0.0).by_action["kept"] == 5
    assert log.replay(0.4).by_action["kept"] == 3
    assert log.replay(1.01).by_action["kept"] == 0


def test_replay_is_monotonic_in_the_threshold() -> None:
    log = log_with_scores([i / 50 for i in range(50)])
    kept = [log.replay(t / 10).by_action["kept"] for t in range(11)]
    assert kept == sorted(kept, reverse=True)


def test_replay_reports_the_false_negatives_a_threshold_would_still_cause() -> None:
    """The number a human actually tunes on."""
    log = log_with_scores([0.2, 0.4, 0.6])
    log.outcome(kind="expand", item_id="s:0", turn=2)  # score 0.2 was needed after all
    log.outcome(kind="expand", item_id="s:1", turn=2)  # score 0.4 was needed after all

    assert log.replay(0.5).false_negatives == 2
    assert log.replay(0.3).false_negatives == 1
    assert log.replay(0.1).false_negatives == 0


def test_replay_of_retrieve_decisions_uses_inject_and_skip() -> None:
    log = ShadowLog(path=None)
    for index, score in enumerate([0.2, 0.8]):
        log.decision(kind="retrieve", item_id=f"r:{index}", score=score, threshold=0.5,
                     action="injected" if score >= 0.5 else "skipped", tokens=5,
                     origin=ORIGIN, text="summary", turn=1)

    replayed = log.replay(0.1)
    assert replayed.by_action["injected"] == 2
    assert replayed.by_action["skipped"] == 0


def test_stats_of_an_empty_log_is_all_zeroes() -> None:
    stats = ShadowLog(path=None).stats()
    assert stats.total == 0
    assert stats.false_negative_rate == 0.0
    assert stats.by_action["elided"] == 0


def test_concurrent_appends_lose_nothing(tmp_path) -> None:
    log = ShadowLog(path=tmp_path / "shadow.jsonl")

    def work(worker: int) -> None:
        for i in range(50):
            log.decision(kind="admit", item_id=f"s:{worker}-{i}", score=0.5,
                         threshold=0.35, action="kept", tokens=1, origin=ORIGIN,
                         text="x", turn=1)

    threads = [threading.Thread(target=work, args=(w,)) for w in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert log.stats().total == 300
    assert ShadowLog.load(tmp_path / "shadow.jsonl").stats().total == 300
