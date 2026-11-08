"""Supervisor-level tests for the pre-registered attempt-3 collection-failure policy.

Fake venue clients only; no network. Every test drives the real ``PaperCollector`` over
a real journal and control database and reads the durable registry back, so the policy
is checked where it acts: retry inside a slot, skipped slots, the three skip budgets,
held-rule staleness, and the terminal classes.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import subprocess
import sys
import urllib.error
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation.test_paper_supervisor import NOW, Clock, FeedTransport, setup
from trade_rl.evaluation.paper import supervisor
from trade_rl.evaluation.paper.collection_policy import SkipBudgetExceeded
from trade_rl.evaluation.paper.control import CollectionControl
from trade_rl.evaluation.paper.engine import PaperEngine
from trade_rl.integrations.binance.forward import ForwardFreshnessError
from trade_rl.integrations.binance.types import BinanceTransportError


def http_failure(code: int) -> BinanceTransportError:
    """What ``BinancePublicTransport`` raises: a transport error chained to the status."""
    error = BinanceTransportError("Binance request failed for https://example.test")
    error.__cause__ = urllib.error.HTTPError(
        "https://example.test",
        code,
        "venue error",
        {},  # type: ignore[arg-type]
        None,
    )
    return error


class Scripted(FeedTransport):
    """A fake venue whose captures fail on a script (by capture number or kind)."""

    def __init__(self, clock: Clock) -> None:
        super().__init__(clock)
        self.captures = 0
        self.kinds: list[str] = []
        self.script: dict[int, BaseException] = {}
        self.snapshot_error: BaseException | None = None
        self.rules_error: BaseException | None = None
        self.failure_delay = 0.0

    def _request_bytes(self, url: str, *, max_response_bytes: int | None = None) -> Any:
        kind = None
        if url.endswith("/api/v3/time"):
            kind = "snapshot"
        elif "api/v3/exchangeInfo" in url:
            kind = "rules"
        if kind is not None:
            self.captures += 1
            self.kinds.append(kind)
            error = self.script.get(self.captures) or (
                self.snapshot_error if kind == "snapshot" else self.rules_error
            )
            if error is not None:
                self.urls.append(url)
                self.clock.value += timedelta(seconds=self.failure_delay)
                raise error
        return super()._request_bytes(url, max_response_bytes=max_response_bytes)


class Sleeper:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.value += timedelta(seconds=seconds)


def build(tmp_path, monkeypatch, *, close_seconds=40_000):
    root, digest, clock, _ = setup(tmp_path, monkeypatch, close_seconds=close_seconds)
    feed = Scripted(clock)
    sleeper = Sleeper(clock)
    return root, digest, clock, feed, sleeper


def collector(root, digest, clock, feed, sleeper):
    return supervisor.PaperCollector(
        root,
        expected_protocol_sha256=digest,
        clock=clock,
        transport=feed,
        monotonic=lambda: (clock.value - NOW).total_seconds(),
        sleep=sleeper,
    )


def at(slot: int, offset: float = 1.0):
    """Wall time ``offset`` seconds into slot ``slot`` (the screen starts at NOW+1 s)."""
    return NOW + timedelta(seconds=1 + 60 * slot + offset)


def rows(root: Path, table: str) -> list[dict[str, Any]]:
    connection = sqlite3.connect(root / "collection-control.sqlite")
    try:
        connection.row_factory = sqlite3.Row
        found = connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
        return [dict(row) for row in found]
    finally:
        connection.close()


def skipped(root: Path) -> list[int]:
    return [row["slot"] for row in rows(root, "skips")]


@contextlib.contextmanager
def running(root, digest, clock, feed, sleeper):
    """A collector that has completed slot 0 (positions opened, rules held)."""
    with collector(root, digest, clock, feed, sleeper) as active:
        clock.value = at(0)
        active.cycle()
        yield active


def marker(root: Path) -> dict[str, Any]:
    return json.loads((root / "collection-failure.json").read_bytes())


def events(root: Path, digest: str) -> int:
    return PaperEngine(root, expected_protocol_sha256=digest).status()["events"]


def test_transient_failure_is_retried_in_the_same_slot_and_then_commits(
    tmp_path, monkeypatch
):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    feed.script = {2: http_failure(503)}  # capture 1 is the rules, 2 the first snapshot
    with collector(root, digest, clock, feed, sleeper) as active:
        clock.value = at(0)
        outcome = active.cycle()
    assert outcome["status"]["account"]["fill_count"] == 4
    assert sleeper.calls == [2]
    assert feed.kinds == ["rules", "snapshot", "snapshot", "snapshot"]
    attempts = rows(root, "attempts")
    assert len(attempts) == 1
    assert attempts[0]["purpose"] == "snapshot" and attempts[0]["outcome"] == "failed"
    assert attempts[0]["error_type"] == "BinanceTransportError"
    assert attempts[0]["slot"] == 0 and attempts[0]["ordinal"] == 1
    assert skipped(root) == []
    # The failed capture keeps its evidence; the retry used a new directory.
    failures = list((root / "sources").rglob("failure.json"))
    assert len(failures) == 1
    assert attempts[0]["path"] == failures[0].parent.relative_to(root).as_posix()
    assert not (root / "collection-failure.json").exists()


def test_two_retries_are_the_limit_and_exhaustion_skips_the_slot(tmp_path, monkeypatch):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    feed.script = {2: http_failure(503), 3: http_failure(502), 4: http_failure(504)}
    with collector(root, digest, clock, feed, sleeper) as active:
        clock.value = at(0)
        outcome = active.cycle()
        assert outcome["phase"] == "skipped"
        assert sleeper.calls == [2, 2]
        assert feed.kinds == ["rules", "snapshot", "snapshot", "snapshot"]
        assert events(root, digest) == 0  # a skipped slot appends no command
        assert [row["outcome"] for row in rows(root, "attempts")] == ["failed"] * 3
        assert [row["ordinal"] for row in rows(root, "attempts")] == [1, 2, 3]
        skips = rows(root, "skips")
        assert [(row["slot"], row["reason"]) for row in skips] == [
            (0, "snapshot_unavailable")
        ]
        assert not (root / "collection-failure.json").exists()
        # The cycle was acknowledged, so the next slot runs normally.
        clock.value = at(1)
        assert active.cycle()["phase"] == "observed"
    assert events(root, digest) > 0


def test_a_retry_may_only_start_inside_thirty_seconds_of_the_slot_start(
    tmp_path, monkeypatch
):
    root, digest, clock, feed, sleeper = build(tmp_path / "late", monkeypatch)
    feed.script = {2: http_failure(503)}
    with collector(root, digest, clock, feed, sleeper) as active:
        clock.value = at(0, 28.5)  # a retry would start at about 30.5 s
        outcome = active.cycle()
    assert outcome["phase"] == "skipped" and sleeper.calls == []
    assert feed.kinds == ["rules", "snapshot"]

    root, digest, clock, feed, sleeper = build(tmp_path / "inside", monkeypatch)
    feed.script = {2: http_failure(503)}
    with collector(root, digest, clock, feed, sleeper) as active:
        clock.value = at(0, 27.5)  # a retry starts at about 29.5 s
        outcome = active.cycle()
    assert outcome["phase"] == "observed" and sleeper.calls == [2]


def test_the_retry_allowance_is_per_slot_not_per_capture(tmp_path, monkeypatch):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    # Two rules failures spend the whole allowance; the snapshot failure that follows
    # cannot be retried and skips the slot.
    feed.script = {1: http_failure(503), 2: http_failure(503), 4: http_failure(503)}
    with collector(root, digest, clock, feed, sleeper) as active:
        clock.value = at(0)
        outcome = active.cycle()
    assert feed.kinds == ["rules", "rules", "rules", "snapshot"]
    assert outcome["phase"] == "skipped" and sleeper.calls == [2, 2]
    assert [row["purpose"] for row in rows(root, "attempts")] == [
        "rules",
        "rules",
        "snapshot",
    ]


def test_consecutive_skip_budget_breach_ends_the_attempt(tmp_path, monkeypatch):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with running(root, digest, clock, feed, sleeper) as active:
        feed.snapshot_error = http_failure(503)
        clock.value = at(1)
        assert active.cycle()["phase"] == "skipped"
        clock.value = at(2)
        with pytest.raises(SkipBudgetExceeded) as breach:
            active.cycle()
        assert breach.value.budget == "consecutive" and breach.value.slot == 2
    record = marker(root)
    assert record["failure_class"] == "B" and record["budget"] == "consecutive"
    assert record["error_type"] == "SkipBudgetExceeded"
    assert skipped(root) == [1, 2]
    with pytest.raises(RuntimeError, match="halted"):
        with collector(root, digest, clock, feed, sleeper):
            pass


def test_skip_budget_state_survives_a_restart(tmp_path, monkeypatch):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with running(root, digest, clock, feed, sleeper) as active:
        feed.snapshot_error = http_failure(503)
        clock.value = at(1)
        assert active.cycle()["phase"] == "skipped"
    with collector(root, digest, clock, feed, sleeper) as reopened:
        clock.value = at(2)
        with pytest.raises(SkipBudgetExceeded):
            reopened.cycle()


def test_more_than_two_skips_in_sixty_slots_breach_the_hourly_budget(
    tmp_path, monkeypatch
):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with running(root, digest, clock, feed, sleeper) as active:
        for slot in range(1, 6):
            feed.snapshot_error = http_failure(503) if slot % 2 == 0 else None
            clock.value = at(slot)
            expected = "skipped" if slot % 2 == 0 else "observed"
            assert active.cycle()["phase"] == expected
        feed.snapshot_error = http_failure(503)
        clock.value = at(6)
        with pytest.raises(SkipBudgetExceeded) as breach:
            active.cycle()
    assert breach.value.budget == "per_hour" and breach.value.slot == 6
    assert skipped(root) == [2, 4, 6]
    assert marker(root)["failure_class"] == "B"


def test_total_skip_budget_breach_on_the_130th_skip(tmp_path, monkeypatch):
    root, digest, clock, feed, sleeper = build(
        tmp_path, monkeypatch, close_seconds=300_000
    )
    # 129 earlier skips spread so that neither the consecutive nor the hourly budget
    # is touched, written the way a long run would have registered them.
    control = CollectionControl(root, digest)
    control.register_skips(0, [n * 30 for n in range(129)], "snapshot_unavailable")
    control.close()
    with collector(root, digest, clock, feed, sleeper) as active:
        late = 129 * 30 + 100
        clock.value = at(late)
        feed.snapshot_error = http_failure(503)
        with pytest.raises(SkipBudgetExceeded) as breach:
            active.cycle()
    assert breach.value.budget == "total" and breach.value.slot == late


def test_the_129th_skip_is_allowed(tmp_path, monkeypatch):
    root, digest, clock, feed, sleeper = build(
        tmp_path, monkeypatch, close_seconds=300_000
    )
    control = CollectionControl(root, digest)
    control.register_skips(0, [n * 30 for n in range(128)], "snapshot_unavailable")
    control.close()
    with collector(root, digest, clock, feed, sleeper) as active:
        clock.value = at(128 * 30 + 100)
        feed.snapshot_error = http_failure(503)
        assert active.cycle()["phase"] == "skipped"
    assert not (root / "collection-failure.json").exists()
    assert len(skipped(root)) == 129


def test_missed_slots_are_registered_as_skips_and_count(tmp_path, monkeypatch):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with running(root, digest, clock, feed, sleeper) as active:
        clock.value = at(2)  # slot 1 never ran: 120 s between decisions, no gap command
        active.cycle()
        assert skipped(root) == [1]
        assert rows(root, "skips")[0]["reason"] == "missed_slot"
        clock.value = at(5)  # slots 3 and 4 missed: adjacent, the consecutive budget
        with pytest.raises(SkipBudgetExceeded) as breach:
            active.cycle()
    assert breach.value.budget == "consecutive" and breach.value.slot == 4
    assert skipped(root) == [1, 3, 4]


def test_a_recorded_observation_gap_is_not_also_counted_as_missed_slots(
    tmp_path, monkeypatch
):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with running(root, digest, clock, feed, sleeper) as active:
        clock.value = at(10)  # 600 s: the frozen 180 s gap rule records the failure
        active.cycle()
    assert skipped(root) == []
    status = PaperEngine(root, expected_protocol_sha256=digest).status()
    assert "observation_gap" in status["quality_failures"]


@pytest.mark.parametrize("code", [401, 403, 418, 429, 451])
def test_denials_are_terminal_and_never_retried(tmp_path, monkeypatch, code):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    feed.script = {2: http_failure(code)}
    with collector(root, digest, clock, feed, sleeper) as active:
        clock.value = at(0)
        with pytest.raises(BinanceTransportError):
            active.cycle()
    assert feed.captures == 2 and sleeper.calls == []
    assert marker(root)["failure_class"] == "D"
    assert skipped(root) == [] and rows(root, "attempts") == []


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("source unavailable"),
        ValueError("depth book is crossed or locked"),
        http_failure(404),
        http_failure(501),
    ],
)
def test_integrity_and_unknown_errors_are_terminal_and_never_retried(
    tmp_path, monkeypatch, error
):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    feed.script = {2: error}
    with collector(root, digest, clock, feed, sleeper) as active:
        clock.value = at(0)
        with pytest.raises(type(error)):
            active.cycle()
    assert feed.captures == 2 and sleeper.calls == []
    assert marker(root)["failure_class"] == "I"
    assert skipped(root) == [] and rows(root, "attempts") == []


def test_a_freshness_rejection_of_one_capture_is_retried(tmp_path, monkeypatch):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    feed.script = {2: ForwardFreshnessError("source timestamp is stale")}
    with collector(root, digest, clock, feed, sleeper) as active:
        clock.value = at(0)
        outcome = active.cycle()
    assert outcome["status"]["account"]["fill_count"] == 4 and sleeper.calls == [2]


def test_a_clock_reversal_during_a_retry_is_terminal(tmp_path, monkeypatch):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    feed.script = {2: http_failure(503)}

    def backwards(seconds: float) -> None:
        clock.value -= timedelta(seconds=30)

    with collector(root, digest, clock, feed, backwards) as active:
        clock.value = at(0)
        with pytest.raises(ValueError, match="clock"):
            active.cycle()
    assert marker(root)["failure_class"] == "I"


def test_a_transient_refresh_failure_inside_the_hold_proceeds_on_held_rules(
    tmp_path, monkeypatch
):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with running(root, digest, clock, feed, sleeper) as active:
        held = active._rule
        refresh = feed.captures + 1
        feed.script = {refresh: http_failure(503)}
        clock.value = at(56)  # rules are about 3360 s old: refresh due, hold intact
        before = events(root, digest)
        outcome = active.cycle()
        assert outcome["phase"] != "skipped"
        assert active._rule == held  # no new rules were taken
    assert sleeper.calls == []  # held rules need no retry
    assert feed.kinds[refresh - 1 : refresh + 1] == ["rules", "snapshot"]
    assert skipped(root) == []  # it consumes no skip budget
    assert [(r["purpose"], r["outcome"]) for r in rows(root, "attempts")] == [
        ("rules", "failed")
    ]
    assert events(root, digest) > before
    assert not (root / "collection-failure.json").exists()


@pytest.mark.parametrize(
    ("error", "failure_class"),
    [(http_failure(429), "D"), (RuntimeError("malformed rules"), "I")],
)
def test_a_denial_or_integrity_error_on_a_refresh_ends_the_attempt_even_when_held(
    tmp_path, monkeypatch, error, failure_class
):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with running(root, digest, clock, feed, sleeper) as active:
        feed.script = {feed.captures + 1: error}
        clock.value = at(56)
        with pytest.raises(type(error)):
            active.cycle()
    assert marker(root)["failure_class"] == failure_class
    assert sleeper.calls == [] and skipped(root) == []


@pytest.mark.parametrize(
    ("error", "failure_class"),
    [(http_failure(403), "D"), (ValueError("rule precision"), "I")],
)
def test_a_denial_or_integrity_error_on_the_first_rules_ends_the_attempt(
    tmp_path, monkeypatch, error, failure_class
):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    feed.script = {1: error}
    with collector(root, digest, clock, feed, sleeper) as active:
        clock.value = at(0)
        with pytest.raises(type(error)):
            active.cycle()
    assert marker(root)["failure_class"] == failure_class and feed.captures == 1


def journal_kinds(root: Path, digest: str) -> list[str]:
    engine = PaperEngine(root, expected_protocol_sha256=digest)
    return [record["event"]["kind"] for record in engine.journal.iter_events()]


def test_rules_older_than_one_hour_skip_the_slot_and_no_decision_is_made(
    tmp_path, monkeypatch
):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with running(root, digest, clock, feed, sleeper) as active:
        feed.rules_error = http_failure(503)
        clock.value = at(61)  # rules are about 3660 s old
        captures = feed.captures
        outcome = active.cycle()
    assert outcome["phase"] == "skipped"
    assert sleeper.calls == [2, 2]  # a refresh that must succeed uses the retries
    assert feed.kinds[captures:] == ["rules", "rules", "rules"]  # no snapshot taken
    assert [(r["slot"], r["reason"]) for r in rows(root, "skips")] == [
        (61, "rules_unavailable")
    ]
    # Only the frozen observation gap follows the first slot's decision and execution.
    assert journal_kinds(root, digest) == ["decision", "execution", "gap"]


def test_rules_that_cross_one_hour_during_the_capture_skip_without_deciding(
    tmp_path, monkeypatch
):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with running(root, digest, clock, feed, sleeper) as active:
        feed.script = {feed.captures + 1: http_failure(503)}  # the refresh fails
        feed.delay = 0.5  # a ten-request snapshot capture takes about five seconds
        clock.value = at(59, 58.5)  # rules are about 3597 s old when the cycle starts
        outcome = active.cycle()
    assert outcome["phase"] == "skipped"
    assert [(r["slot"], r["reason"]) for r in rows(root, "skips")] == [
        (59, "rules_expired")
    ]
    outcomes = [(r["purpose"], r["outcome"]) for r in rows(root, "attempts")]
    assert outcomes == [("rules", "failed"), ("snapshot", "unused")]
    assert journal_kinds(root, digest) == ["decision", "execution", "gap"]


def test_a_stale_rules_skip_shares_the_budgets_with_other_skips(tmp_path, monkeypatch):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with running(root, digest, clock, feed, sleeper) as active:
        feed.rules_error = http_failure(503)
        clock.value = at(61)
        assert active.cycle()["phase"] == "skipped"
        clock.value = at(62)
        with pytest.raises(SkipBudgetExceeded) as breach:
            active.cycle()
    assert breach.value.budget == "consecutive"


def test_a_failed_execution_capture_is_never_retried_and_skips_the_slot(
    tmp_path, monkeypatch
):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    feed.script = {3: http_failure(503)}  # rules, decision snapshot, execution snapshot
    with collector(root, digest, clock, feed, sleeper) as active:
        clock.value = at(0)
        outcome = active.cycle()
        assert outcome["phase"] == "skipped" and outcome["status"]["pending"]
        assert feed.kinds == ["rules", "snapshot", "snapshot"] and sleeper.calls == []
        assert [r["reason"] for r in rows(root, "skips")] == [
            "execution_capture_failed"
        ]
        # The pending decision is left to expire; the next slot cancels it with the
        # existing permanent gap and never executes it late.
        clock.value = at(1)
        later = active.cycle()
    assert later["phase"] == "cancelled" and not later["status"]["pending"]
    assert "expired_pending_decision" in later["status"]["quality_failures"]
    assert later["status"]["account"]["fill_count"] == 0


def test_no_retry_after_a_consumed_capture_when_the_decision_fails(
    tmp_path, monkeypatch
):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)

    def late(*args, **kwargs):
        raise TimeoutError("decision store timed out")

    monkeypatch.setattr(PaperEngine, "decide", late)
    with collector(root, digest, clock, feed, sleeper) as active:
        clock.value = at(0)
        with pytest.raises(TimeoutError):
            active.cycle()
    assert feed.kinds == ["rules", "snapshot"] and sleeper.calls == []
    assert marker(root)["failure_class"] == "I"
    assert rows(root, "attempts") == [] and skipped(root) == []


def test_every_failed_attempt_is_registered_once_before_the_next_attempt(
    tmp_path, monkeypatch
):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    feed.script = {2: http_failure(503), 3: http_failure(503)}
    seen: list[int] = []

    def registered(seconds: float) -> None:
        seen.append(len(rows(root, "attempts")))
        sleeper(seconds)

    with collector(root, digest, clock, feed, registered) as active:
        clock.value = at(0)
        active.cycle()
    assert seen == [1, 2]
    paths = [row["path"] for row in rows(root, "attempts")]
    assert len(set(paths)) == 2
    on_disk = {
        p.parent.relative_to(root).as_posix()
        for p in (root / "sources").rglob("failure.json")
    }
    assert set(paths) == on_disk


def test_a_death_between_a_failed_attempt_and_its_retry_is_never_resumed(
    tmp_path, monkeypatch
):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    script = """
import os, sys
from tests.evaluation.test_paper_collection_retry import (
    Scripted, at, collector, http_failure,
)
from tests.evaluation.test_paper_supervisor import Clock
from trade_rl.evaluation.paper import supervisor
supervisor.build_candidate_run_provenance = lambda: {'source': 'original'}
clock = Clock()
feed = Scripted(clock)
feed.script = {2: http_failure(503)}
def die(seconds):
    os._exit(73)
with collector(sys.argv[1], sys.argv[2], clock, feed, die) as active:
    clock.value = at(0)
    active.cycle()
"""
    stopped = subprocess.run(
        [sys.executable, "-c", script, str(root), digest],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert stopped.returncode == 73, stopped.stderr
    assert len(rows(root, "attempts")) == 1 and skipped(root) == []
    clock.value = at(1)
    with pytest.raises(RuntimeError, match="interrupted|halted"):
        with collector(root, digest, clock, feed, sleeper):
            pass
    assert feed.captures == 0
    assert (root / "collection-failure.json").is_file()


def test_the_collection_loop_continues_past_a_skipped_slot(tmp_path, monkeypatch):
    from trade_rl.evaluation.paper.operations import collect_until_finished

    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch, close_seconds=170)
    # Captures 1-3 open the position; slot 1 then fails its snapshot three times.
    feed.script = {4: http_failure(503), 5: http_failure(503), 6: http_failure(503)}
    published: list[dict[str, Any]] = []
    clock.value = NOW + timedelta(seconds=1)
    with collector(root, digest, clock, feed, sleeper) as active:
        result = collect_until_finished(
            active, clock=clock, sleep=sleeper, publish=published.append
        )
    phases = [item["phase"] for item in published]
    assert "skipped" in phases and result["phase"] in {"finished", "rejected"}
    assert not (root / "collection-failure.json").exists()


def test_the_unchanged_assessor_still_rejects_a_root_with_a_skipped_slot(
    tmp_path, monkeypatch
):
    """Fail-closed until the screen v3 and its evaluator audit exist."""
    from trade_rl.evaluation.paper import assessment

    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with running(root, digest, clock, feed, sleeper) as active:
        feed.snapshot_error = http_failure(503)
        clock.value = at(1)
        assert active.cycle()["phase"] == "skipped"
        feed.snapshot_error = None
        clock.value = at(2)
        active.cycle()
    records = PaperEngine(root, expected_protocol_sha256=digest).journal.events()
    reasons = assessment.collection_evidence_reasons(
        root, digest, records, start=at(0, 0), end=at(5, 0)
    )
    assert "source_capture_failure" in reasons
    assert "cycle_command_mismatch" in reasons


def test_rules_that_expire_while_a_held_refresh_fails_skip_before_any_capture(
    tmp_path, monkeypatch
):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with running(root, digest, clock, feed, sleeper) as active:
        feed.script = {feed.captures + 1: http_failure(503)}
        feed.failure_delay = 4.0  # the failed refresh itself outlasts the hold
        clock.value = at(59, 58.5)  # rules are about 3597 s old when the cycle starts
        captures = feed.captures
        outcome = active.cycle()
    assert outcome["phase"] == "skipped"
    assert feed.kinds[captures:] == ["rules"]  # no snapshot was taken
    assert [(r["slot"], r["reason"]) for r in rows(root, "skips")] == [
        (59, "rules_expired")
    ]
    assert journal_kinds(root, digest) == ["decision", "execution", "gap"]


def test_slots_that_never_ran_before_the_first_cycle_are_registered(
    tmp_path, monkeypatch
):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with collector(root, digest, clock, feed, sleeper) as active:
        clock.value = at(2)  # 121 s after the screen start: inside the 180 s gap limit
        active.cycle()
    assert skipped(root) == [1]
    assert rows(root, "skips")[0]["reason"] == "missed_slot"
    status = PaperEngine(root, expected_protocol_sha256=digest).status()
    assert "observation_gap" not in status["quality_failures"]


def test_a_first_cycle_in_slot_zero_or_one_registers_no_missed_slot(
    tmp_path, monkeypatch
):
    for slot in (0, 1):
        root, digest, clock, feed, sleeper = build(tmp_path / f"s{slot}", monkeypatch)
        with collector(root, digest, clock, feed, sleeper) as active:
            clock.value = at(slot)
            active.cycle()
        assert skipped(root) == [], slot


def test_two_slots_missing_before_the_first_cycle_breach_the_consecutive_budget(
    tmp_path, monkeypatch
):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with collector(root, digest, clock, feed, sleeper) as active:
        # at(3, 0) is exactly 180 s after the screen start and the frozen gap limit is
        # strictly greater than 180 s, so no gap is recorded at this instant; any later
        # first cycle in slot 3 (offset > 0) would record the gap instead. The
        # registration is exercised directly so the collector stays open for the check.
        clock.value = at(3, 0)
        active._cycle_id = active._control.begin(clock.value)
        active._slot = 3
        with pytest.raises(SkipBudgetExceeded) as breach:
            active._register_missed_slots(False)
    assert breach.value.budget == "consecutive" and breach.value.slot == 2
    assert skipped(root) == [1, 2]


def test_a_recorded_gap_on_the_first_cycle_does_not_also_register_slots(
    tmp_path, monkeypatch
):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with collector(root, digest, clock, feed, sleeper) as active:
        clock.value = at(5)
        active.cycle()
    assert skipped(root) == []


def test_first_cycle_missed_slots_are_not_registered_twice_across_a_restart(
    tmp_path, monkeypatch
):
    root, digest, clock, feed, sleeper = build(tmp_path, monkeypatch)
    with collector(root, digest, clock, feed, sleeper) as active:
        clock.value = at(2)
        active.cycle()
    assert skipped(root) == [1]
    with collector(root, digest, clock, feed, sleeper) as reopened:
        clock.value = at(3)
        reopened.cycle()  # the previous cycle (slot 2) now anchors the check
    assert skipped(root) == [1]
    assert rows(root, "skips")[0]["cycle_id"] == 1
