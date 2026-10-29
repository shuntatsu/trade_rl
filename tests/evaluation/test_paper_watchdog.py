"""Collector liveness, failure records and supervision, with fakes only.

No network, no real collector and no economics: children are scripted fakes, time is
a fake clock, and the sealed roots hold software-validation plans.
"""

from __future__ import annotations

import hashlib
import json
import signal
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation.test_paper_supervisor import collector, setup
from tests.integrations.test_binance_forward import NOW
from trade_rl.evaluation.paper import operations, watchdog
from trade_rl.evaluation.paper.control import CollectionControl
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
    supervise_collector,
    write_heartbeat,
)

START = NOW + timedelta(seconds=1)


class WallClock:
    def __init__(self, value=NOW):
        self.value = value

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += timedelta(seconds=seconds)


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
