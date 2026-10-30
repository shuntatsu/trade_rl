"""Collector liveness, failure records and supervision, with fakes only.

No network, no real collector and no economics: children are scripted fakes, time is
a fake clock, and the sealed roots hold software-validation plans.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation.test_paper_supervisor import collector, setup
from tests.integrations.test_binance_forward import NOW
from trade_rl.evaluation.paper import lock as lock_module
from trade_rl.evaluation.paper import operations, watchdog
from trade_rl.evaluation.paper.control import CollectionControl
from trade_rl.evaluation.paper.lock import CollectorLockHeld, collector_lock_held
from trade_rl.evaluation.paper.supervisor import write_collection_failure
from trade_rl.evaluation.paper.watchdog import (
    CollectorTerminated,
    ExitGuard,
    HeartbeatWriter,
    PopenChild,
    WatchdogPolicy,
    check_collection,
    has_incomplete_cycle,
    install_termination_handlers,
    popen_launcher,
    read_heartbeat,
    read_supervisor_heartbeat,
    supervise_collector,
    write_heartbeat,
    write_supervisor_heartbeat,
)

START = NOW + timedelta(seconds=1)


class WallClock:
    def __init__(self, value=NOW):
        self.value = value

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += timedelta(seconds=seconds)


@pytest.fixture(autouse=True)
def short_lock_patience(monkeypatch):
    """In-process tests need no real backoff window; the subprocess race keeps it."""
    monkeypatch.setattr(lock_module, "LOCK_PATIENCE_SECONDS", 0.05)


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    root, digest, _clock, _feed = setup(tmp_path, monkeypatch)
    return root, digest


def failure_of(root: Path) -> dict[str, Any]:
    return json.loads((root / "collection-failure.json").read_bytes())


def journal_fingerprint(root: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("protocol.json", "journal.sqlite")
    }


def beat(root, digest, clock, *, state="cycle_completed", last=None, cycles=1):
    write_heartbeat(
        root,
        protocol_sha256=digest,
        state=state,
        pid=4242,
        at=clock(),
        last_command_at=last,
        cycles=cycles,
    )


# ---------------------------------------------------------------- heartbeat


def test_heartbeat_round_trips_atomically_and_leaves_no_temporary(sealed):
    root, digest = sealed
    assert read_heartbeat(root, digest) is None
    beat(root, digest, WallClock(), last=START, cycles=3)
    heartbeat = read_heartbeat(root, digest)
    assert heartbeat is not None
    assert (heartbeat.state, heartbeat.pid, heartbeat.cycles) == (
        "cycle_completed",
        4242,
        3,
    )
    assert heartbeat.last_command_at == START and heartbeat.written_at == NOW
    assert not (root / "collector-heartbeat.json.tmp").exists()
    raw = json.loads((root / "collector-heartbeat.json").read_bytes())
    assert raw["production_eligible"] is False


def test_heartbeat_rejects_foreign_malformed_or_unknown_state(sealed):
    root, digest = sealed
    beat(root, digest, WallClock())
    with pytest.raises(ValueError, match="malformed"):
        read_heartbeat(root, "b" * 64)
    path = root / "collector-heartbeat.json"
    path.write_bytes(b"{not json")
    with pytest.raises(ValueError, match="unreadable"):
        read_heartbeat(root, digest)
    path.write_bytes(json.dumps({"schema": "other"}).encode())
    with pytest.raises(ValueError, match="malformed"):
        read_heartbeat(root, digest)
    with pytest.raises(ValueError, match="state"):
        beat(root, digest, WallClock(), state="bogus")


def test_heartbeat_replace_retries_a_transient_windows_sharing_error(
    sealed, monkeypatch
):
    root, digest = sealed
    real = watchdog.os.replace
    calls = []

    def flaky(source, target):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("file is being read")
        return real(source, target)

    monkeypatch.setattr(watchdog.os, "replace", flaky)
    monkeypatch.setattr(watchdog.time, "sleep", lambda _s: None)
    beat(root, digest, WallClock())
    assert len(calls) == 3 and read_heartbeat(root, digest) is not None
    monkeypatch.setattr(
        watchdog.os, "replace", lambda *_: (_ for _ in ()).throw(PermissionError("x"))
    )
    with pytest.raises(PermissionError):
        write_heartbeat(
            root,
            protocol_sha256=digest,
            state="sleeping",
            pid=1,
            at=NOW,
            last_command_at=None,
            cycles=0,
            attempts=2,
        )


def test_writer_tracks_last_command_and_cycle_count(sealed):
    root, digest = sealed
    clock = WallClock()
    writer = HeartbeatWriter(root, digest, clock=clock, pid=7)
    writer("starting", None)
    assert read_heartbeat(root, digest).last_command_at is None
    writer("cycle_completed", {"last_command_at": START.isoformat()})
    writer("sleeping", None)
    heartbeat = read_heartbeat(root, digest)
    assert heartbeat.last_command_at == START and heartbeat.cycles == 1
    writer("finished", None)
    assert read_heartbeat(root, digest).cycles == 2


def test_cadence_loop_writes_heartbeats_through_a_real_round_trip(
    tmp_path, monkeypatch
):
    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    writer = HeartbeatWriter(root, digest, clock=lambda: NOW)
    states = []

    def record(state, status):
        states.append(state)
        writer(state, status)

    def wait(seconds):
        assert 0 < seconds <= 20
        clock.value += timedelta(seconds=seconds)

    with collector(root, digest, clock, feed) as active:
        operations.collect_until_finished(
            active,
            clock=clock,
            sleep=wait,
            publish=lambda _result: None,
            heartbeat=record,
            heartbeat_interval_seconds=20,
        )
    assert states[0] == "starting" and states[-1] == "finished"
    assert {"cycle_started", "cycle_completed", "sleeping"} <= set(states)
    final = read_heartbeat(root, digest)
    assert final.state == "finished" and final.last_command_at is not None


def test_cadence_loop_rejects_a_nonpositive_heartbeat_interval(sealed):
    root, digest = sealed
    clock = WallClock()
    with collector(root, digest, clock, None) as active:
        with pytest.raises(ValueError, match="interval"):
            operations.collect_until_finished(
                active,
                clock=clock,
                publish=lambda _r: None,
                heartbeat_interval_seconds=0,
            )


def test_failed_heartbeat_write_halts_the_collector_durably(tmp_path, monkeypatch):
    root, digest, clock, feed = setup(tmp_path, monkeypatch)

    def broken(_state, _status):
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        with collector(root, digest, clock, feed) as active:
            operations.collect_until_finished(
                active, clock=clock, publish=lambda _r: None, heartbeat=broken
            )
    assert failure_of(root)["error_type"] == "OSError"


# ------------------------------------------------------------ check (stateless)


def test_check_before_start_never_records_a_failure(sealed):
    root, digest = sealed
    clock = WallClock(NOW)
    report = check_collection(root, protocol_sha256=digest, clock=clock)
    assert report["state"] == "not_started" and report["healthy"]
    beat(root, digest, clock)
    clock.value = NOW + timedelta(seconds=0.5)
    stale = check_collection(
        root, protocol_sha256=digest, clock=clock, stale_seconds=0.1
    )
    assert stale["state"] == "stale_before_start" and not stale["healthy"]
    assert not (root / "collection-failure.json").exists()


def test_check_without_any_heartbeat_fails_only_after_the_gap_limit(sealed):
    root, digest = sealed
    clock = WallClock(START + timedelta(seconds=180))
    assert check_collection(root, protocol_sha256=digest, clock=clock)["healthy"]
    clock.value = START + timedelta(seconds=181)
    report = check_collection(root, protocol_sha256=digest, clock=clock)
    assert report["state"] == "failed_never_started" and report["failure_written"]
    record = failure_of(root)
    assert record["error_type"] == "CollectorNeverStarted"
    assert record["reported_by"] == "watchdog" and not record["production_eligible"]


def test_check_stale_heartbeat_after_start_writes_permanent_failure_once(sealed):
    root, digest = sealed
    clock = WallClock(START + timedelta(seconds=10))
    beat(root, digest, clock)
    fresh = check_collection(root, protocol_sha256=digest, clock=clock)
    assert fresh["state"] == "running" and fresh["heartbeat_age_seconds"] == 0
    before = journal_fingerprint(root)
    clock.value += timedelta(seconds=151)
    stale = check_collection(root, protocol_sha256=digest, clock=clock)
    assert stale["state"] == "failed_stale" and stale["failure_written"]
    first = failure_of(root)
    assert first["error_type"] == "CollectorHeartbeatStale"
    assert first["heartbeat_pid"] == 4242
    clock.value += timedelta(seconds=500)
    again = check_collection(root, protocol_sha256=digest, clock=clock)
    assert again["state"] == "halted" and not again["failure_written"]
    assert failure_of(root) == first
    assert journal_fingerprint(root) == before


def test_check_treats_a_corrupt_heartbeat_as_missing_after_start(sealed):
    root, digest = sealed
    (root / "collector-heartbeat.json").write_bytes(b"\x00torn")
    clock = WallClock(START + timedelta(seconds=300))
    report = check_collection(root, protocol_sha256=digest, clock=clock)
    assert report["state"] == "failed_never_started"
    assert failure_of(root)["heartbeat_invalid"] is True


def test_check_ignores_age_once_the_collector_reached_a_terminal_state(sealed):
    root, digest = sealed
    clock = WallClock(START + timedelta(days=100))
    beat(root, digest, WallClock(START), state="finished")
    report = check_collection(root, protocol_sha256=digest, clock=clock)
    assert report["state"] == "finished" and report["healthy"]
    assert not (root / "collection-failure.json").exists()


def test_check_requires_the_pinned_protocol_and_writes_nothing_otherwise(sealed):
    root, digest = sealed
    clock = WallClock(START + timedelta(days=1))
    with pytest.raises(ValueError, match="digest mismatch"):
        check_collection(root, protocol_sha256="c" * 64, clock=clock)
    with pytest.raises(ValueError, match="positive"):
        check_collection(root, protocol_sha256=digest, clock=clock, stale_seconds=0)
    assert not (root / "collection-failure.json").exists()


def test_watchdog_failure_halts_a_later_collector(tmp_path, monkeypatch):
    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    wall = WallClock(START + timedelta(seconds=400))
    check_collection(root, protocol_sha256=digest, clock=wall)
    with pytest.raises(RuntimeError, match="halted"):
        with collector(root, digest, clock, feed):
            pass
    assert not feed.urls


# ---------------------------------------------------- exit and signal hooks


def test_exit_guard_records_an_unfinished_exit_but_not_after_disarm(sealed):
    root, digest = sealed
    guard = ExitGuard(root, digest, clock=WallClock())
    guard.disarm()
    guard()
    assert not (root / "collection-failure.json").exists()
    armed = ExitGuard(root, digest, clock=WallClock())
    armed()
    record = failure_of(root)
    assert record["error_type"] == "CollectorExitedUnfinished"
    assert record["reported_by"] == "atexit"
    armed()  # idempotent and never overwrites
    assert failure_of(root) == record


def test_exit_guard_does_not_replace_an_earlier_failure(sealed):
    root, digest = sealed
    assert write_collection_failure(
        root, protocol_sha256=digest, error_type="First", error="original"
    )
    ExitGuard(root, digest)()
    assert failure_of(root)["error_type"] == "First"
    assert (
        write_collection_failure(
            root, protocol_sha256=digest, error_type="Second", error="again"
        )
        is None
    )


def test_console_close_event_writes_the_failure_synchronously(sealed):
    root, digest = sealed
    ExitGuard(root, digest).console_event("CTRL_CLOSE_EVENT")
    record = failure_of(root)
    assert record["error_type"] == "CollectorConsoleClosed"
    assert record["error"] == "CTRL_CLOSE_EVENT"


class FakeSignals:
    SIGTERM = 15
    SIGBREAK = 21

    def __init__(self):
        self.handlers = {}

    def signal(self, number, handler):
        self.handlers[number] = handler


def test_termination_signals_raise_into_the_collector_and_are_recorded(
    tmp_path, monkeypatch
):
    fake = FakeSignals()
    installed = install_termination_handlers(
        lambda _name: None, signal_module=fake, platform="linux"
    )
    assert installed == ["SIGTERM", "SIGBREAK"]
    assert set(fake.handlers) == {15, 21}
    with pytest.raises(CollectorTerminated, match="signal 15"):
        fake.handlers[15](15, None)
    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    with pytest.raises(CollectorTerminated):
        with collector(root, digest, clock, feed):
            fake.handlers[21](21, None)
    assert failure_of(root)["error_type"] == "CollectorTerminated"


def test_posix_signal_module_without_sigbreak_installs_only_sigterm():
    class Posix:
        SIGTERM = 15

        def __init__(self):
            self.seen = []

        def signal(self, number, handler):
            self.seen.append(number)

    posix = Posix()
    assert install_termination_handlers(
        lambda _name: None, signal_module=posix, platform="linux"
    ) == ["SIGTERM"]
    assert posix.seen == [15]


class FakeKernel32:
    def __init__(self, result=1):
        self.result, self.callback, self.enabled = result, None, None

    def SetConsoleCtrlHandler(self, callback, enable):  # noqa: N802
        self.callback, self.enabled = callback, enable
        return self.result


def test_windows_console_handler_records_close_events_and_declines_others():
    fake, kernel = FakeSignals(), FakeKernel32()
    seen = []
    installed = install_termination_handlers(
        seen.append,
        signal_module=fake,
        platform="win32",
        kernel32=kernel,
        function_type=lambda function: function,
    )
    assert installed[-1] == "console_ctrl_handler" and kernel.enabled is True
    assert [kernel.callback(event) for event in (0, 1, 2, 5, 6)] == [0] * 5
    assert seen == ["CTRL_CLOSE_EVENT", "CTRL_LOGOFF_EVENT", "CTRL_SHUTDOWN_EVENT"]
    with pytest.raises(OSError, match="console control"):
        install_termination_handlers(
            seen.append,
            signal_module=fake,
            platform="win32",
            kernel32=FakeKernel32(result=0),
            function_type=lambda function: function,
        )


# -------------------------------------------------------------- supervision


class FakeChild:
    """Scripted child: `script(child, polls)` may write files and returns an exit code."""

    def __init__(self, pid, script, *, ignores_terminate=False):
        self.pid, self.script = pid, script
        self.polls, self.exit_code = 0, None
        self.terminated = self.killed = False
        self.ignores_terminate = ignores_terminate

    def poll(self):
        if self.exit_code is None and not self.killed:
            self.polls += 1
            self.exit_code = self.script(self, self.polls)
        return self.exit_code

    def terminate(self):
        self.terminated = True
        if not self.ignores_terminate:
            self.exit_code = -15

    def kill(self):
        self.killed = True
        self.exit_code = -9

    def wait(self, timeout):
        return self.exit_code


class Harness:
    def __init__(self, root, digest, scripts, **child_options):
        self.root, self.digest = root, digest
        self.clock = WallClock(START + timedelta(seconds=5))
        self.scripts = list(scripts)
        self.children: list[FakeChild] = []
        self.events: list[dict[str, Any]] = []
        self.child_options = child_options

    def launch(self):
        script = self.scripts[len(self.children)]
        child = FakeChild(100 + len(self.children), script, **self.child_options)
        self.children.append(child)
        return child

    def run(self, **policy):
        return supervise_collector(
            self.root,
            protocol_sha256=self.digest,
            launch=self.launch,
            policy=WatchdogPolicy(**policy),
            clock=self.clock,
            sleep=self.clock.sleep,
            publish=self.events.append,
        )

    def beat(self, **kwargs):
        beat(self.root, self.digest, self.clock, **kwargs)


def test_child_killed_between_cycles_is_relaunched_without_touching_the_journal(
    sealed,
):
    root, digest = sealed
    before = journal_fingerprint(root)

    def killed(child, polls):
        harness.beat(last=harness.clock.value - timedelta(seconds=20))
        return -9  # TerminateProcess / external kill, nothing written by the child

    def finishes(child, polls):
        harness.beat(state="finished", last=harness.clock.value)
        return 0

    harness = Harness(root, digest, [killed, finishes])
    result = harness.run()
    assert result["outcome"] == "finished" and result["restarts"] == 1
    assert result["exit_code"] == 0 and len(harness.children) == 2
    assert not (root / "collection-failure.json").exists()
    assert journal_fingerprint(root) == before
    assert [e["event"] for e in harness.events].count("launched") == 2


def test_child_death_inside_a_cycle_is_recorded_and_never_resumed(sealed):
    root, digest = sealed
    control = CollectionControl(root, digest)
    control.begin(START)  # begun, never acknowledged: the process died mid-cycle
    control.close()
    assert has_incomplete_cycle(root)

    def dies(child, polls):
        harness.beat(last=START)
        return 1

    harness = Harness(root, digest, [dies, dies])
    result = harness.run()
    assert result["outcome"] == "failed" and result["exit_code"] == 3
    assert len(harness.children) == 1
    record = failure_of(root)
    assert record["error_type"] == "CollectorDiedMidCycle"
    assert record["child_exit_code"] == 1 and record["reported_by"] == "watchdog"


def test_exception_in_child_leaves_its_own_failure_and_no_relaunch(sealed):
    root, digest = sealed

    def raises(child, polls):
        write_collection_failure(
            root, protocol_sha256=digest, error_type="RuntimeError", error="boom"
        )
        return 1

    harness = Harness(root, digest, [raises, raises])
    result = harness.run()
    assert result["outcome"] == "halted" and len(harness.children) == 1
    assert failure_of(root)["error_type"] == "RuntimeError"


def test_hang_is_detected_from_a_stale_heartbeat_and_the_child_stopped(sealed):
    root, digest = sealed

    def hangs(child, polls):
        if polls == 1:
            harness.beat(last=START)
        return None

    harness = Harness(root, digest, [hangs])
    result = harness.run(stale_seconds=60, poll_seconds=10)
    assert result["outcome"] == "failed" and result["exit_code"] == 3
    child = harness.children[0]
    assert child.terminated and not child.killed
    record = failure_of(root)
    assert record["error_type"] == "CollectorHung" and record["pid"] == child.pid
    assert record["limit_seconds"] == 60


def test_a_child_that_ignores_terminate_is_killed(sealed):
    root, digest = sealed
    harness = Harness(root, digest, [lambda c, p: None], ignores_terminate=True)
    harness.run(startup_grace_seconds=20, poll_seconds=10)
    child = harness.children[0]
    assert child.terminated and child.killed
    assert failure_of(root)["error_type"] == "CollectorHung"


def test_silent_child_that_never_beats_hits_the_startup_grace(sealed):
    root, digest = sealed
    beat(root, digest, WallClock(NOW))  # a previous run's old heartbeat is ignored
    harness = Harness(root, digest, [lambda c, p: None])
    harness.run(startup_grace_seconds=30, stale_seconds=10, poll_seconds=5)
    record = failure_of(root)
    assert record["error_type"] == "CollectorHung" and record["limit_seconds"] == 30


def test_restart_that_cannot_resume_inside_the_gap_limit_is_a_failure(sealed):
    root, digest = sealed

    def dies_late(child, polls):
        harness.beat(last=harness.clock.value - timedelta(seconds=150))
        return -9

    harness = Harness(root, digest, [dies_late, dies_late])
    harness.clock.value = START + timedelta(seconds=200)
    result = harness.run(restart_margin_seconds=60)
    assert result["outcome"] == "failed" and len(harness.children) == 1
    record = failure_of(root)
    assert record["error_type"] == "CollectorGapUnrecoverable"
    assert record["projected_gap_seconds"] == 210


def test_restart_loop_is_bounded(sealed):
    root, digest = sealed

    def dies(child, polls):
        harness.beat(last=harness.clock.value)
        return 1

    harness = Harness(root, digest, [dies] * 5)
    result = harness.run(max_restarts=2)
    assert result["restarts"] == 2 and len(harness.children) == 3
    assert failure_of(root)["error_type"] == "CollectorRestartLimit"


def test_deterministic_refusal_and_early_clean_exit_are_not_restarted(
    tmp_path, monkeypatch
):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    root, digest, _clock, _feed = setup(tmp_path / "a", monkeypatch)
    harness = Harness(root, digest, [lambda c, p: 2, lambda c, p: 2])
    harness.run()
    assert failure_of(root)["error_type"] == "CollectorRefused"
    assert len(harness.children) == 1
    root, digest, _clock, _feed = setup(tmp_path / "b", monkeypatch)
    harness = Harness(root, digest, [lambda c, p: 0, lambda c, p: 0])
    harness.run()
    assert failure_of(root)["error_type"] == "CollectorExitedEarly"
    assert len(harness.children) == 1


def test_rejected_screen_finishes_with_a_nonzero_child_code(sealed):
    root, digest = sealed

    def rejected(child, polls):
        harness.beat(state="rejected", last=START)
        return 1

    harness = Harness(root, digest, [rejected])
    result = harness.run()
    assert (result["outcome"], result["exit_code"]) == ("finished", 1)
    assert not (root / "collection-failure.json").exists()


def test_terminal_or_failed_roots_are_never_relaunched(sealed):
    root, digest = sealed
    beat(root, digest, WallClock(), state="finished")
    harness = Harness(root, digest, [])
    assert harness.run()["outcome"] == "already_terminal"
    assert not harness.children
    (root / "collector-heartbeat.json").unlink()
    write_collection_failure(
        root, protocol_sha256=digest, error_type="Earlier", error="x"
    )
    harness = Harness(root, digest, [])
    assert harness.run()["outcome"] == "halted" and not harness.children
    assert failure_of(root)["error_type"] == "Earlier"


def test_supervisor_validates_protocol_policy_and_unreadable_control(sealed):
    root, digest = sealed
    harness = Harness(root, digest, [])
    with pytest.raises(ValueError, match="digest mismatch"):
        supervise_collector(
            root,
            protocol_sha256="d" * 64,
            launch=harness.launch,
            clock=harness.clock,
            sleep=harness.clock.sleep,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        harness.run(stale_seconds=181)
    for bad in ({"poll_seconds": 0}, {"max_restarts": -1}, {"stale_seconds": True}):
        with pytest.raises(ValueError, match="positive|negative"):
            WatchdogPolicy(**bad)
    assert not harness.children and not (root / "collection-failure.json").exists()
    (root / "collection-control.sqlite").write_bytes(b"not sqlite" * 50)
    harness = Harness(root, digest, [lambda c, p: 1])
    harness.run()
    assert failure_of(root)["error_type"] == "CollectorControlUnreadable"


# ------------------------------------------------------- real process wrappers


class FakePopen:
    def __init__(self, command, **options):
        self.command, self.options = command, options
        self.pid = 99
        self.sent = []
        self.terminated = self.killed = False
        self.waits = []

    def send_signal(self, number):
        self.sent.append(number)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def poll(self):
        return None

    def wait(self, timeout):
        self.waits.append(timeout)
        raise subprocess.TimeoutExpired("x", timeout)


def test_launcher_builds_the_collector_command_outside_the_sealed_root(
    sealed, tmp_path
):
    root, digest = sealed
    created = []

    def popen(command, **options):
        created.append(FakePopen(command, **options))
        return created[-1]

    launch = popen_launcher(
        root, digest, tmp_path / "child.log", python="py", popen=popen
    )
    child = launch()
    process = created[0]
    assert process.command[:5] == [
        "py",
        "-m",
        "trade_rl.evaluation.paper.cli",
        "run",
        "--root",
    ]
    assert process.command[-2:] == ["--protocol-sha256", digest]
    assert process.options["stdin"] == subprocess.DEVNULL
    assert child.pid == 99 and child.poll() is None
    if sys.platform == "win32":
        assert process.options["creationflags"] == subprocess.CREATE_NEW_PROCESS_GROUP
    with pytest.raises(ValueError, match="outside the sealed root"):
        popen_launcher(root, digest, root / "child.log")


def test_popen_child_uses_a_graceful_signal_then_reports_wait_timeouts(monkeypatch):
    process = FakePopen(["x"])
    child = PopenChild(process)  # type: ignore[arg-type]
    child.terminate()
    if sys.platform == "win32":
        assert process.sent == [signal.CTRL_BREAK_EVENT] and not process.terminated
    else:
        assert process.terminated and not process.sent
    assert child.wait(0.5) is None and process.waits == [0.5]
    child.kill()
    assert process.killed


def test_popen_child_wraps_a_real_short_lived_process(tmp_path):
    process = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(7)"])
    child = PopenChild(process)
    assert child.wait(30) == 7 and child.poll() == 7
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    live = PopenChild(sleeper)
    assert live.poll() is None and live.wait(0.05) is None
    live.kill()
    assert live.wait(30) is not None


# --------------------------------------------------- lock contention (M1)


def test_lock_probe_is_non_destructive_and_sees_a_real_second_collector(
    tmp_path, monkeypatch
):
    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    assert collector_lock_held(root) is False  # no lock file yet: never created
    assert not (root / "collector.sqlite").exists()
    with collector(root, digest, clock, feed):
        assert collector_lock_held(root) is True
        with pytest.raises(CollectorLockHeld, match="another collector"):
            with collector(root, digest, clock, feed):
                pass
    assert not (root / "collection-failure.json").exists()
    assert collector_lock_held(root) is False
    assert (root / "collector.sqlite").exists()
    with collector(root, digest, clock, feed) as reopened:
        assert reopened.cycle()["phase"] == "waiting"  # the probe left no lock behind


def test_second_collector_process_exits_four_and_writes_no_marker(
    tmp_path, monkeypatch, capsys
):
    from trade_rl.evaluation.paper import cli

    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    with collector(root, digest, clock, feed):
        before = journal_fingerprint(root)
        code = cli.main(["run", "--root", str(root), "--protocol-sha256", digest])
        assert code == 4
    assert json.loads(capsys.readouterr().err)["error_type"] == "CollectorLockHeld"
    assert not (root / "collection-failure.json").exists()
    assert not (root / "collector-heartbeat.json").exists()
    assert journal_fingerprint(root) == before


def test_supervisor_refuses_a_live_collector_without_spawning_or_marking(
    tmp_path, monkeypatch
):
    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    harness = Harness(root, digest, [lambda c, p: pytest.fail("must not spawn")])
    with collector(root, digest, clock, feed):
        beat(root, digest, harness.clock, last=START)
        before = journal_fingerprint(root)
        result = harness.run()
    assert (result["outcome"], result["exit_code"]) == ("lock_contention", 4)
    assert not harness.children
    assert not (root / "collection-failure.json").exists()
    assert journal_fingerprint(root) == before
    assert read_heartbeat(root, digest).state == "cycle_completed"  # untouched


def test_child_losing_a_startup_lock_race_is_contention_not_a_refusal(sealed):
    root, digest = sealed
    harness = Harness(root, digest, [lambda c, p: 4, lambda c, p: 4])
    result = harness.run()
    assert (result["outcome"], result["exit_code"]) == ("lock_contention", 4)
    assert len(harness.children) == 1
    assert not (root / "collection-failure.json").exists()


def test_supervisor_restart_with_a_surviving_child_does_not_invalidate_it(
    tmp_path, monkeypatch
):
    """A dead supervisor leaves its child alive; the next supervisor must back off."""
    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    with collector(root, digest, clock, feed):
        for _ in range(2):
            harness = Harness(root, digest, [lambda c, p: 2])
            assert harness.run()["outcome"] == "lock_contention"
    assert not (root / "collection-failure.json").exists()


# ------------------------------------------- relaunch vs stale check (L1)


def supervisor_beat(root, digest, **overrides):
    values = dict(
        protocol_sha256=digest,
        state="monitoring",
        pid=77,
        at=START,
        child_launched_at=START,
        startup_grace_seconds=300.0,
        poll_seconds=5.0,
    )
    values.update(overrides)
    assert write_supervisor_heartbeat(root, **values)


def test_check_defers_to_a_live_supervisors_relaunch_grace_only(sealed):
    root, digest = sealed
    last_beat = START + timedelta(seconds=10)
    beat(root, digest, WallClock(last_beat), cycles=9)
    launched = last_beat + timedelta(seconds=20)
    now = launched + timedelta(seconds=140)  # heartbeat 160 s old, replay running
    supervisor_beat(
        root, digest, at=now - timedelta(seconds=5), child_launched_at=launched
    )
    report = check_collection(root, protocol_sha256=digest, clock=WallClock(now))
    assert report["state"] == "relaunch_in_progress" and report["healthy"]
    assert report["severity"] == "healthy"
    assert not (root / "collection-failure.json").exists()
    # The deferral ends when the supervisor's own grace is spent ...
    late = launched + timedelta(seconds=301)
    supervisor_beat(root, digest, at=late, child_launched_at=launched)
    assert (
        check_collection(root, protocol_sha256=digest, clock=WallClock(late))["state"]
        == "failed_stale"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        dict(state="stopped"),
        dict(at=START - timedelta(seconds=500)),  # supervisor itself is dead
        dict(child_launched_at=None),
        dict(startup_grace_seconds=901.0),  # beyond the bounded maximum
    ],
)
def test_check_does_not_defer_for_a_dead_stopped_or_unbounded_supervisor(
    sealed, overrides
):
    root, digest = sealed
    beat(root, digest, WallClock(START + timedelta(seconds=10)))
    now = START + timedelta(seconds=200)
    values = dict(
        at=now - timedelta(seconds=5), child_launched_at=now - timedelta(seconds=30)
    )
    values.update(overrides)
    supervisor_beat(root, digest, **values)
    report = check_collection(root, protocol_sha256=digest, clock=WallClock(now))
    assert report["state"] == "failed_stale" and report["severity"] == "failed"


def test_check_does_not_defer_once_the_relaunched_child_has_beaten(sealed):
    root, digest = sealed
    now = START + timedelta(seconds=400)
    beat(root, digest, WallClock(now - timedelta(seconds=200)))  # old, child beat
    supervisor_beat(
        root,
        digest,
        at=now,
        child_launched_at=now - timedelta(seconds=250),
    )
    beat(root, digest, WallClock(now - timedelta(seconds=160)))  # after the launch
    assert (
        check_collection(root, protocol_sha256=digest, clock=WallClock(now))["state"]
        == "failed_stale"
    )


def test_supervisor_publishes_its_own_state_for_the_check(sealed):
    root, digest = sealed
    seen = []

    def child_script(child, polls):
        if polls == 1:
            assert read_supervisor_heartbeat(root, digest).state == "launching"
            harness.beat(last=START)
            return None
        seen.append(read_supervisor_heartbeat(root, digest))
        harness.beat(state="finished", last=START)
        return 0

    harness = Harness(root, digest, [child_script])
    harness.run()
    during = seen[0]
    assert during.state == "monitoring" and during.child_launched_at is not None
    assert during.startup_grace_seconds == 300 and during.poll_seconds == 5
    assert read_supervisor_heartbeat(root, digest).state == "stopped"
    assert read_supervisor_heartbeat(root, "e" * 64) is None
    (root / "supervisor-heartbeat.json").write_bytes(b"junk")
    assert read_supervisor_heartbeat(root, digest) is None
    with pytest.raises(ValueError, match="grace"):
        WatchdogPolicy(startup_grace_seconds=901)


def test_supervisor_heartbeat_io_errors_never_stop_the_supervisor(sealed, monkeypatch):
    root, digest = sealed
    real = watchdog.os.replace

    def locked(source, target):
        if Path(target).name == "supervisor-heartbeat.json":
            raise PermissionError("antivirus holds the status file")
        return real(source, target)

    monkeypatch.setattr(watchdog.os, "replace", locked)

    def finishes(child, polls):
        harness.beat(state="finished", last=START)
        return 0

    harness = Harness(root, digest, [finishes])
    assert harness.run()["outcome"] == "finished"
    assert read_supervisor_heartbeat(root, digest) is None
    assert not write_supervisor_heartbeat(
        root / "missing",
        protocol_sha256=digest,
        state="monitoring",
        pid=1,
        at=START,
        child_launched_at=None,
        startup_grace_seconds=300.0,
        poll_seconds=5.0,
    )


# --------------------------------------------- check bounds and codes (L2)


def test_check_stale_limit_is_bounded_like_the_supervisor(sealed):
    root, digest = sealed
    clock = WallClock(START + timedelta(days=1))
    for bad in (0, -1, 180.5, 1e9):
        with pytest.raises(ValueError, match="within the frozen gap limit"):
            check_collection(
                root, protocol_sha256=digest, clock=clock, stale_seconds=bad
            )
    assert not (root / "collection-failure.json").exists()
    report = check_collection(
        root, protocol_sha256=digest, clock=clock, stale_seconds=180
    )
    assert report["state"] == "failed_never_started"


def test_check_exit_codes_separate_marker_from_degraded_states(
    tmp_path, capsys, monkeypatch
):
    from trade_rl.evaluation.paper import cli

    root, digest, _clock, _feed = setup(tmp_path, monkeypatch)
    wall = WallClock(NOW)
    original = watchdog.check_collection
    monkeypatch.setattr(
        cli, "check_collection", lambda *a, **k: original(*a, clock=wall, **k)
    )
    args = ["check", "--root", str(root), "--protocol-sha256", digest]

    def run():
        code = cli.main(args)
        return code, json.loads(capsys.readouterr().out)

    assert run()[0] == 0  # not started: healthy
    beat(root, digest, wall)
    wall.value = NOW + timedelta(seconds=0.9)
    monkeypatch.setattr(
        cli,
        "check_collection",
        lambda *a, **k: original(*a, clock=wall, **{**k, "stale_seconds": 0.1}),
    )
    code, report = run()
    assert (code, report["state"], report["severity"]) == (
        5,
        "stale_before_start",
        "degraded",
    )
    assert not (root / "collection-failure.json").exists()  # exit 5 wrote no marker
    wall.value = START + timedelta(seconds=300)
    code, report = run()
    assert (code, report["state"], report["failure_written"]) == (
        3,
        "failed_stale",
        True,
    )
    code, report = run()
    assert (code, report["state"], report["failure_written"]) == (3, "halted", False)


def test_unreadable_heartbeat_inside_the_start_grace_is_degraded_not_a_marker(sealed):
    root, digest = sealed
    (root / "collector-heartbeat.json").write_bytes(b"\x00torn")
    report = check_collection(
        root, protocol_sha256=digest, clock=WallClock(START + timedelta(seconds=60))
    )
    assert (report["state"], report["severity"]) == ("heartbeat_invalid", "degraded")
    assert not (root / "collection-failure.json").exists()


# ------------------------------------------- heartbeat write tolerance (L3)


def test_heartbeat_write_errors_are_tolerated_until_they_persist(sealed, monkeypatch):
    root, digest = sealed
    clock = WallClock(START)
    writer = HeartbeatWriter(root, digest, clock=clock, pid=1, give_up_seconds=120)
    real = watchdog.write_heartbeat
    failing = {"on": False}

    def flaky(*args, **kwargs):
        if failing["on"]:
            raise PermissionError("sharing violation")
        return real(*args, **kwargs)

    monkeypatch.setattr(watchdog, "write_heartbeat", flaky)
    writer("starting", None)
    failing["on"] = True
    for _ in range(5):  # 75 s of misses: tolerated, collector keeps running
        clock.value += timedelta(seconds=15)
        writer("sleeping", None)
    assert writer.missed == 5
    failing["on"] = False
    clock.value += timedelta(seconds=15)
    writer("sleeping", None)
    assert writer.missed == 0 and read_heartbeat(root, digest).written_at == clock.value
    failing["on"] = True
    with pytest.raises(PermissionError):
        for _ in range(10):  # persisting past the 120 s window finally halts
            clock.value += timedelta(seconds=15)
            writer("sleeping", None)
    with pytest.raises(ValueError, match="give-up"):
        HeartbeatWriter(root, digest, give_up_seconds=0)


def test_programming_errors_in_the_heartbeat_are_never_swallowed(sealed):
    root, digest = sealed
    with pytest.raises(ValueError, match="state"):
        HeartbeatWriter(root, digest, clock=WallClock())("bogus", None)


def test_control_schema_is_read_through_the_control_module(sealed):
    from trade_rl.evaluation.paper import control

    root, digest = sealed
    assert watchdog.has_incomplete_cycle is control.incomplete_cycle_exists
    assert control.incomplete_cycle_exists(root) is False
    owner = CollectionControl(root, digest)
    cycle = owner.begin(START)
    assert control.incomplete_cycle_exists(root) is True
    owner.finish(cycle)
    owner.close()
    assert control.incomplete_cycle_exists(root) is False


# ------------------------------- status file untouched on contention (L5)


def test_contention_and_terminal_exits_leave_a_live_supervisors_status_untouched(
    tmp_path, monkeypatch
):
    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    status = root / "supervisor-heartbeat.json"
    supervisor_beat(root, digest, at=START, child_launched_at=START)
    original = status.read_bytes()
    harness = Harness(root, digest, [lambda c, p: pytest.fail("must not spawn")])
    with collector(root, digest, clock, feed):
        assert harness.run()["outcome"] == "lock_contention"
    assert status.read_bytes() == original
    # already_terminal
    beat(root, digest, harness.clock, state="finished", last=START)
    assert harness.run()["outcome"] == "already_terminal"
    assert status.read_bytes() == original
    # halted before any launch
    (root / "collector-heartbeat.json").unlink()
    write_collection_failure(
        root, protocol_sha256=digest, error_type="Earlier", error="x"
    )
    assert harness.run()["outcome"] == "halted"
    assert status.read_bytes() == original


def test_child_race_loss_does_not_write_a_stopped_status(sealed):
    root, digest = sealed
    harness = Harness(root, digest, [lambda c, p: 4])
    assert harness.run()["outcome"] == "lock_contention"
    assert read_supervisor_heartbeat(root, digest).state != "stopped"
    assert not (root / "collection-failure.json").exists()


# ------------------------- only a busy database is contention (L6)


class _FakeConnection:
    def __init__(self, error):
        self.error = error
        self.closed = False

    def execute(self, *_):
        raise self.error

    def close(self):
        self.closed = True

    def rollback(self):
        pass


def _operational(message, code=None):
    error = sqlite3.OperationalError(message)
    if code is not None:
        error.sqlite_errorcode = code
    return error


@pytest.mark.parametrize(
    ("message", "code", "held"),
    [
        ("database is locked", 5, True),  # SQLITE_BUSY
        ("database table is locked", 6, True),  # SQLITE_LOCKED
        ("busy", 5 | (1 << 8), True),  # extended BUSY code
        ("database is locked", None, True),  # no code: message fallback
        ("disk I/O error", 10, False),  # SQLITE_IOERR
        ("unable to open database file", 14, False),  # SQLITE_CANTOPEN
        ("database disk image is malformed", 11, False),  # SQLITE_CORRUPT
        ("something else", None, False),
        ("disk I/O error", 10 | (1 << 8), False),
    ],
)
def test_only_busy_or_locked_errors_count_as_contention(
    tmp_path, monkeypatch, message, code, held
):
    (tmp_path / "collector.sqlite").write_bytes(b"")
    fake = _FakeConnection(_operational(message, code))
    monkeypatch.setattr(lock_module.sqlite3, "connect", lambda *a, **k: fake)
    if held:
        assert collector_lock_held(tmp_path) is True
        with pytest.raises(CollectorLockHeld):
            lock_module.acquire_collector_lock(tmp_path)
    else:
        with pytest.raises(sqlite3.OperationalError, match=message):
            collector_lock_held(tmp_path)
        with pytest.raises(sqlite3.OperationalError):
            lock_module.acquire_collector_lock(tmp_path)
    assert fake.closed


def test_a_non_busy_probe_error_is_an_error_everywhere_and_never_exit_four(
    sealed, tmp_path, monkeypatch, capsys
):
    from trade_rl.evaluation.paper import cli

    root, digest = sealed
    (root / "collector.sqlite").write_bytes(b"")
    fake = _FakeConnection(_operational("disk I/O error", 10))
    monkeypatch.setattr(lock_module.sqlite3, "connect", lambda *a, **k: fake)
    harness = Harness(root, digest, [lambda c, p: pytest.fail("must not spawn")])
    with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
        harness.run()
    assert not harness.children
    code = cli.main(
        [
            "supervise",
            "--root",
            str(root),
            "--protocol-sha256",
            digest,
            "--log-file",
            str(tmp_path / "child.log"),
        ]
    )
    assert code == 2  # an error, never the contention code 4
    assert json.loads(capsys.readouterr().err)["error_type"] == "OperationalError"
    assert not (root / "collection-failure.json").exists()
    assert not (root / "supervisor-heartbeat.json").exists()
    code = cli.main(["run", "--root", str(root), "--protocol-sha256", digest])
    assert code == 2
    assert not (root / "collection-failure.json").exists()


# --------------------------- real two-process lock race (L7)


def test_two_real_collector_processes_race_for_the_lock_exactly_one_owns_it(tmp_path):
    import os
    from datetime import UTC, datetime

    root = tmp_path / "race"
    digest = operations.seal_paper_study(
        root,
        start_at=datetime.now(UTC) + timedelta(hours=2),
        attempt_lineage={
            "schema": "carry_paper_attempt_lineage_v2",
            "attempt_number": 1,
            "predecessor": None,
        },
    )
    repo = Path(__file__).resolve().parents[2]
    # A venv python may be a launcher that spawns the real interpreter, so each
    # process reports its own pid first; ownership is judged by that real pid.
    bootstrap = (
        "import os, sys\n"
        "print(os.getpid(), flush=True)\n"
        "from trade_rl.evaluation.paper.cli import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )
    command = [
        sys.executable,
        "-c",
        bootstrap,
        "run",
        "--root",
        str(root),
        "--protocol-sha256",
        digest,
    ]
    options = dict(
        cwd=repo,
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    launchers = [subprocess.Popen(command, **options) for _ in range(2)]
    real_pids: list[int] = []
    try:
        deadline = time.monotonic() + 240
        for launcher in launchers:
            real_pids.append(int(launcher.stdout.readline().decode().strip()))
        loser_index = None
        while loser_index is None and time.monotonic() < deadline:
            done = [i for i, p in enumerate(launchers) if p.poll() is not None]
            if done:
                loser_index = done[0]
            else:
                time.sleep(0.2)
        assert loser_index is not None, "neither process resolved the lock race"
        winner_index = 1 - loser_index
        loser, winner = launchers[loser_index], launchers[winner_index]
        assert loser.returncode == 4
        assert winner.poll() is None  # the owner is still collecting
        error = json.loads(loser.stderr.read().decode().strip().splitlines()[-1])
        assert error["error_type"] == "CollectorLockHeld"
        # The owner writes its heartbeat only after it holds the lock; it must be the
        # winner's, never the loser's, and no marker may exist.
        heartbeat = None
        while heartbeat is None and time.monotonic() < deadline:
            try:
                heartbeat = read_heartbeat(root, digest)
            except ValueError:
                heartbeat = None
            if heartbeat is None:
                time.sleep(0.2)
        assert heartbeat is not None
        assert heartbeat.pid == real_pids[winner_index] != real_pids[loser_index]
        assert not (root / "collection-failure.json").exists()
        assert collector_lock_held(root) is True
        assert winner.poll() is None
    finally:
        for pid in real_pids:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGTERM)  # TerminateProcess on Windows
        for launcher in launchers:
            if launcher.poll() is None:
                launcher.kill()
            launcher.wait(timeout=60)
            for stream in (launcher.stdout, launcher.stderr):
                if stream is not None:
                    stream.close()
    deadline = time.monotonic() + 60
    while collector_lock_held(root) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert collector_lock_held(root) is False  # death released the lock


def test_transient_mutual_busy_is_retried_so_a_simultaneous_start_has_one_owner(
    tmp_path, monkeypatch
):
    (tmp_path / "collector.sqlite").write_bytes(b"")
    attempts = []

    class Flaky:
        def __init__(self):
            self.closed = False

        def execute(self, *_):
            attempts.append(1)
            if len(attempts) < 3:
                raise _operational("database is locked", 5)

        def close(self):
            self.closed = True

    monkeypatch.setattr(lock_module.sqlite3, "connect", lambda *a, **k: Flaky())
    slept = []
    connection = lock_module.acquire_collector_lock(
        tmp_path, patience_seconds=5, sleep=slept.append
    )
    assert len(attempts) == 3 and len(slept) == 2
    assert all(0.02 <= pause < 0.1 for pause in slept)
    assert isinstance(connection, Flaky)


def test_a_genuinely_held_lock_is_reported_after_the_bounded_patience(
    tmp_path, monkeypatch
):
    (tmp_path / "collector.sqlite").write_bytes(b"")
    fake = _FakeConnection(_operational("database is locked", 5))
    monkeypatch.setattr(lock_module.sqlite3, "connect", lambda *a, **k: fake)
    ticks = iter(range(0, 100))
    with pytest.raises(CollectorLockHeld):
        lock_module.acquire_collector_lock(
            tmp_path,
            patience_seconds=3,
            sleep=lambda _s: None,
            monotonic=lambda: float(next(ticks)),
        )
