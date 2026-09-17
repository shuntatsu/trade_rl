import json
from datetime import timedelta

import pytest

from tests.evaluation.test_paper_engine import Feed
from tests.integrations.test_binance_forward import NOW
from tests.integrations.test_binance_forward_rules import RuleFeed
from trade_rl.evaluation.paper import supervisor
from trade_rl.evaluation.paper.engine import PaperEngine, PaperSettings


class Clock:
    def __init__(self):
        self.value = NOW

    def __call__(self):
        self.value += timedelta(milliseconds=1)
        return self.value


class FeedTransport:
    def __init__(self, clock):
        self.clock = clock
        self.urls = []
        self.fail = False
        self.delay = 0

    def _request_bytes(self, url):
        self.urls.append(url)
        self.clock.value += timedelta(seconds=self.delay)
        if self.fail:
            raise RuntimeError("source unavailable")
        feed = RuleFeed() if "exchangeInfo" in url else Feed(self.clock.value)
        return feed._request_bytes(url)


def setup(tmp_path, monkeypatch, *, close_seconds=60):
    monkeypatch.setattr(
        supervisor, "build_candidate_run_provenance", lambda: {"source": "original"}
    )
    clock = Clock()
    settings = PaperSettings(
        start_at=NOW + timedelta(seconds=1),
        close_at=NOW + timedelta(seconds=close_seconds),
    )
    root = tmp_path / "collection"
    digest = supervisor.seal_collection(
        root,
        settings=settings,
        research_plan={"purpose": "software_validation_only"},
        clock=clock,
    )
    feed = FeedTransport(clock)
    return root, digest, clock, feed


def collector(root, digest, clock, feed):
    return supervisor.PaperCollector(
        root,
        expected_protocol_sha256=digest,
        clock=clock,
        transport=feed,
        monotonic=lambda: 0,
    )


def test_collection_waits_for_start_then_opens_and_closes_with_reused_rules(
    tmp_path, monkeypatch
):
    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    with collector(root, digest, clock, feed) as active:
        assert active.cycle()["phase"] == "waiting"
        assert not feed.urls
        clock.value = NOW + timedelta(seconds=2)
        entered = active.cycle()
        assert entered["status"]["account"]["fill_count"] == 4
        clock.value = NOW + timedelta(seconds=61)
        exited = active.cycle()
        assert exited["status"]["terminal_flat"]
        assert exited["status"]["account"]["fill_count"] == 8
        count = len(feed.urls)
        clock.value = NOW + timedelta(seconds=241)
        assert active.cycle()["phase"] == "finished"
        assert len(feed.urls) == count
    assert sum("exchangeInfo" in url for url in feed.urls) == 2
    assert (
        PaperEngine(root, expected_protocol_sha256=digest).status() == exited["status"]
    )


def test_second_collector_is_excluded_and_close_releases_the_lock(
    tmp_path, monkeypatch
):
    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    with collector(root, digest, clock, feed):
        with pytest.raises(RuntimeError, match="collector"):
            with collector(root, digest, clock, feed):
                pass
    with collector(root, digest, clock, feed) as reopened:
        assert reopened.cycle()["phase"] == "waiting"


def test_source_drift_halts_before_network_and_is_durable(tmp_path, monkeypatch):
    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    with collector(root, digest, clock, feed) as active:
        monkeypatch.setattr(
            supervisor, "build_candidate_run_provenance", lambda: {"source": "changed"}
        )
        clock.value = NOW + timedelta(seconds=2)
        with pytest.raises(ValueError, match="provenance"):
            active.cycle()
    assert not feed.urls
    assert (root / "collection-failure.json").is_file()
    monkeypatch.setattr(
        supervisor, "build_candidate_run_provenance", lambda: {"source": "original"}
    )
    with pytest.raises(RuntimeError, match="halted"):
        with collector(root, digest, clock, feed):
            pass


def test_failure_marker_write_error_cannot_allow_another_cycle(tmp_path, monkeypatch):
    from pathlib import Path

    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    original_open = Path.open

    def unavailable_marker(path, *args, **kwargs):
        if path.name == "collection-failure.json":
            raise PermissionError("failure marker unavailable")
        return original_open(path, *args, **kwargs)

    with collector(root, digest, clock, feed) as active:
        clock.value = NOW + timedelta(seconds=2)
        feed.fail = True
        with monkeypatch.context() as patch:
            patch.setattr(Path, "open", unavailable_marker)
            with pytest.raises(PermissionError, match="marker unavailable"):
                active.cycle()
        count = len(feed.urls)
        feed.fail = False
        clock.value = NOW + timedelta(seconds=3)
        with pytest.raises(RuntimeError, match="interrupted|halted"):
            active.cycle()
        assert len(feed.urls) == count
    monkeypatch.setattr(
        supervisor, "build_candidate_run_provenance", lambda: {"source": "original"}
    )
    with pytest.raises(RuntimeError, match="halted"):
        with collector(root, digest, clock, feed):
            pass


def test_failed_source_preserves_positions_and_never_automatically_retries(
    tmp_path, monkeypatch
):
    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    with collector(root, digest, clock, feed) as active:
        clock.value = NOW + timedelta(seconds=2)
        quantities = active.cycle()["status"]["account"]["quantities"]
        feed.fail = True
        clock.value = NOW + timedelta(seconds=20)
        with pytest.raises(RuntimeError, match="source unavailable"):
            active.cycle()
        count = len(feed.urls)
        with pytest.raises(RuntimeError, match="halted"):
            active.cycle()
        assert len(feed.urls) == count
    engine = PaperEngine(root, expected_protocol_sha256=digest)
    assert engine.status()["account"]["quantities"] == quantities
    assert engine.status()["stop_reason"] == "data_gap"
    failure = json.loads((root / "collection-failure.json").read_bytes())
    assert failure["protocol_sha256"] == digest
    assert failure["production_eligible"] is False


def test_wrong_protocol_digest_does_not_create_a_failure_marker(tmp_path, monkeypatch):
    root, _, clock, feed = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="digest"):
        with collector(root, "0" * 64, clock, feed):
            pass
    assert not feed.urls and not (root / "collection-failure.json").exists()


def test_clock_reversal_cannot_turn_an_active_collection_back_into_waiting(
    tmp_path, monkeypatch
):
    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    with collector(root, digest, clock, feed) as active:
        clock.value = NOW + timedelta(seconds=2)
        active.cycle()
        clock.value = NOW - timedelta(seconds=10)
        with pytest.raises(ValueError, match="clock"):
            active.cycle()
    assert (root / "collection-failure.json").is_file()


def test_broken_clock_still_persists_the_original_failure(tmp_path, monkeypatch):
    root, digest, clock, feed = setup(tmp_path, monkeypatch)

    def broken():
        raise RuntimeError("clock disconnected")

    with collector(root, digest, clock, feed) as active:
        active.clock = broken
        with pytest.raises(RuntimeError, match="clock disconnected"):
            active.cycle()
    assert (root / "collection-failure.json").is_file()


@pytest.mark.parametrize("resume_second", [3, 20])
def test_restart_pending_decision_only_executes_inside_its_window(
    tmp_path, monkeypatch, resume_second
):
    from tests.evaluation.test_paper_engine import market, rules

    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    engine = PaperEngine(root, expected_protocol_sha256=digest)
    engine.decide(
        "saved",
        market=market(root, "market", 2),
        rules=rules(root),
        at=NOW + timedelta(seconds=2.1),
    )
    clock.value = NOW + timedelta(seconds=resume_second)
    with collector(root, digest, clock, feed) as active:
        outcome = active.cycle()
    if resume_second == 3:
        assert outcome["status"]["account"]["fill_count"] == 4
    else:
        assert not feed.urls
        assert outcome["phase"] == "cancelled"
        assert outcome["status"]["account"]["fill_count"] == 0
        assert not outcome["status"]["pending"]
        assert "expired_pending_decision" in outcome["status"]["quality_failures"]


def test_capture_crossing_terminal_grace_cannot_execute_extra_exits(
    tmp_path, monkeypatch
):
    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    with collector(root, digest, clock, feed) as active:
        clock.value = NOW + timedelta(seconds=2)
        initial = active.cycle()["status"]
        clock.value = NOW + timedelta(seconds=239.5)
        feed.delay = 0.06
        finished = active.cycle()
    assert finished["phase"] == "finished"
    assert finished["status"]["account"]["fill_count"] == 4
    assert (
        finished["status"]["account"]["quantities"] == initial["account"]["quantities"]
    )
    assert not finished["status"]["terminal_flat"]


def test_rule_evidence_refreshes_before_expiry(tmp_path, monkeypatch):
    root, digest, clock, feed = setup(tmp_path, monkeypatch, close_seconds=7200)
    with collector(root, digest, clock, feed) as active:
        clock.value = NOW + timedelta(seconds=2)
        active.cycle()
        clock.value = NOW + timedelta(seconds=3402)
        outcome = active.cycle()
    assert sum("exchangeInfo" in url for url in feed.urls) == 4
    assert "observation_gap" in outcome["status"]["quality_failures"]


def test_failed_acknowledgement_reconciles_committed_fills_before_gap(
    tmp_path, monkeypatch
):
    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    execute = PaperEngine.execute

    def lost_ack(*args, **kwargs):
        execute(*args, **kwargs)
        raise RuntimeError("lost execution acknowledgement")

    monkeypatch.setattr(PaperEngine, "execute", lost_ack)
    with collector(root, digest, clock, feed) as active:
        clock.value = NOW + timedelta(seconds=2)
        with pytest.raises(RuntimeError, match="acknowledgement"):
            active.cycle()
    status = PaperEngine(root, expected_protocol_sha256=digest).status()
    assert status["events"] == 3
    assert status["account"]["fill_count"] == 4
    assert not status["terminal_flat"] and status["stop_reason"] == "data_gap"


def test_collector_process_death_releases_the_exclusive_lock(tmp_path, monkeypatch):
    import subprocess
    import sys

    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    script = """
import sys
from trade_rl.evaluation.paper import supervisor
supervisor.build_candidate_run_provenance = lambda: {'source': 'original'}
with supervisor.PaperCollector(sys.argv[1], expected_protocol_sha256=sys.argv[2]):
    print('locked', flush=True)
    sys.stdin.readline()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(root), digest],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        assert child.stdout.readline().strip() == "locked"
        with pytest.raises(RuntimeError, match="collector"):
            with collector(root, digest, clock, feed):
                pass
        child.terminate()
        child.wait(timeout=10)
        with collector(root, digest, clock, feed) as active:
            assert active.cycle()["phase"] == "waiting"
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


def test_protocol_drift_halts_before_any_new_acquisition(tmp_path, monkeypatch):
    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    with collector(root, digest, clock, feed) as active:
        clock.value = NOW + timedelta(seconds=2)
        active.cycle()
        count = len(feed.urls)
        path = root / "protocol.json"
        path.write_bytes(path.read_bytes() + b" ")
        clock.value = NOW + timedelta(seconds=20)
        with pytest.raises(ValueError, match="digest"):
            active.cycle()
    assert len(feed.urls) == count


@pytest.mark.parametrize("stage", ["source_failure", "before_command"])
def test_crash_during_collection_cannot_hide_a_failed_or_unused_capture(
    tmp_path, monkeypatch, stage
):
    import subprocess
    import sys

    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    script = """
import os, sys
from datetime import timedelta
from tests.evaluation.test_paper_supervisor import Clock, FeedTransport, NOW
from trade_rl.evaluation.paper import supervisor
from trade_rl.evaluation.paper.engine import PaperEngine
supervisor.build_candidate_run_provenance = lambda: {'source': 'original'}
clock = Clock()
clock.value = NOW + timedelta(seconds=2)
feed = FeedTransport(clock)
if sys.argv[3] == 'source_failure':
    feed.fail = True
    supervisor.PaperCollector._halt = lambda *args: os._exit(73)
else:
    PaperEngine.decide = lambda *args, **kwargs: os._exit(73)
with supervisor.PaperCollector(sys.argv[1], expected_protocol_sha256=sys.argv[2], clock=clock, transport=feed, monotonic=lambda: 0) as collector:
    collector.cycle()
"""
    stopped = subprocess.run(
        [sys.executable, "-c", script, str(root), digest, stage],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert stopped.returncode == 73, stopped.stderr
    assert list((root / "sources").rglob("*.json"))
    assert not (root / "collection-failure.json").exists()
    clock.value = NOW + timedelta(seconds=3)
    with pytest.raises(RuntimeError, match="interrupted|halted"):
        with collector(root, digest, clock, feed):
            pass
    assert not feed.urls
    assert (root / "collection-failure.json").is_file()
