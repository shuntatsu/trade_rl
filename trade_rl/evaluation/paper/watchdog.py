"""Collector liveness: heartbeat, stale detection, termination hooks, supervision.

Attempt 1 died silently: no shutdown reason, no failure marker, nothing restarted
it. This module makes every end of the collector leave a durable trace and bounds
the only restart that cannot damage the evidence. It never touches the journal, the
control database or captured sources, and it never retries a denied request.

* The collector process writes an atomic heartbeat beside its journal.
* Python exits and OS termination signals write the permanent failure marker.
* An external, stateless check turns a stale heartbeat into the same marker.
* The supervisor relaunches a child that died between cycles inside the frozen
  observation-gap limit. A death in a cycle, a hang, a gap that cannot be
  recovered, or a restart loop is recorded as a failure instead of resumed.

Heartbeats and failure markers are operational facts, not financial evidence.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.paper.account import timestamp
from trade_rl.evaluation.paper.control import incomplete_cycle_exists
from trade_rl.evaluation.paper.failure import FAILURE_NAME, write_collection_failure
from trade_rl.evaluation.paper.lock import collector_lock_held

HEARTBEAT_NAME = "collector-heartbeat.json"
HEARTBEAT_SCHEMA = "paper_collector_heartbeat_v1"
CHECK_SCHEMA = "paper_collector_check_v1"
TERMINAL_STATES = frozenset({"finished", "rejected"})
HEARTBEAT_STATES = frozenset(
    {"starting", "cycle_started", "cycle_completed", "sleeping"} | TERMINAL_STATES
)
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 15.0
DEFAULT_STALE_SECONDS = 150.0
# A heartbeat miss only matters once it persists; stop tolerating write errors
# before the 150 s stale limit so the collector then halts with the real cause.
HEARTBEAT_WRITE_GIVE_UP_SECONDS = 120.0
SUPERVISOR_HEARTBEAT_NAME = "supervisor-heartbeat.json"
SUPERVISOR_HEARTBEAT_SCHEMA = "paper_supervisor_heartbeat_v1"
MAXIMUM_STARTUP_GRACE_SECONDS = 900.0
EXIT_OK = 0
EXIT_FAILED = 3  # a failure marker exists or was just written
EXIT_LOCK_HELD = 4  # another live collector owns the root; nothing was written
EXIT_DEGRADED = 5  # check only: unhealthy but no marker written yet
_FAILURE = FAILURE_NAME
_CONSOLE_EVENTS = {
    2: "CTRL_CLOSE_EVENT",
    5: "CTRL_LOGOFF_EVENT",
    6: "CTRL_SHUTDOWN_EVENT",
}


def _now() -> datetime:
    return datetime.now(UTC)


class CollectorTerminated(BaseException):
    """Raised in the collector's main thread by a termination signal.

    A BaseException so broad ``except Exception`` blocks cannot swallow it; the
    collector's own context manager records it as the durable failure.
    """


@dataclass(frozen=True)
class ProtocolWindow:
    start_at: datetime
    close_at: datetime
    maximum_gap_seconds: float


def read_protocol_window(root: Path, protocol_sha256: str) -> ProtocolWindow:
    """Read the frozen schedule only after the pinned protocol digest matches."""
    path = root / "protocol.json"
    if root.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("collection protocol must be a regular file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != protocol_sha256:
        raise ValueError("collection protocol digest mismatch")
    settings = json.loads(raw)["protocol"]["settings"]
    return ProtocolWindow(
        start_at=timestamp(settings["start_at"]),
        close_at=timestamp(settings["close_at"]),
        maximum_gap_seconds=float(settings["maximum_gap_seconds"]),
    )


@dataclass(frozen=True)
class Heartbeat:
    state: str
    pid: int
    written_at: datetime
    last_command_at: datetime | None
    cycles: int


def write_heartbeat(
    root: Path,
    *,
    protocol_sha256: str,
    state: str,
    pid: int,
    at: datetime,
    last_command_at: datetime | None,
    cycles: int,
    attempts: int = 8,
) -> None:
    """Atomically replace the heartbeat; a reader never sees a partial file.

    A Windows sharing violation is retried with exponential backoff (about 3.55 s
    in total) before the error is raised to the caller."""
    if state not in HEARTBEAT_STATES:
        raise ValueError("unsupported heartbeat state")
    record = dict(
        schema=HEARTBEAT_SCHEMA,
        protocol_sha256=protocol_sha256,
        state=state,
        pid=pid,
        written_at=at.isoformat(),
        last_command_at=None
        if last_command_at is None
        else last_command_at.isoformat(),
        cycles=cycles,
        production_eligible=False,
    )
    target = root / HEARTBEAT_NAME
    if target.is_symlink():
        raise ValueError("collector heartbeat must be a regular file")
    temporary = root / (HEARTBEAT_NAME + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical_json_bytes(record))
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(attempts):
        try:
            os.replace(temporary, target)
            return
        except PermissionError:
            # Windows refuses to replace a file another process is reading.
            if attempt == attempts - 1:
                raise
            time.sleep(min(1.0, 0.05 * 2**attempt))


def read_heartbeat(root: Path, protocol_sha256: str) -> Heartbeat | None:
    """Return None when absent; raise ValueError for a malformed or foreign file."""
    path = root / HEARTBEAT_NAME
    if path.is_symlink():
        raise ValueError("collector heartbeat must be a regular file")
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise ValueError("collector heartbeat is unreadable") from error
    expected = {
        "schema",
        "protocol_sha256",
        "state",
        "pid",
        "written_at",
        "last_command_at",
        "cycles",
        "production_eligible",
    }
    if (
        not isinstance(record, dict)
        or set(record) != expected
        or record["schema"] != HEARTBEAT_SCHEMA
        or record["protocol_sha256"] != protocol_sha256
        or record["state"] not in HEARTBEAT_STATES
        or isinstance(record["pid"], bool)
        or not isinstance(record["pid"], int)
        or isinstance(record["cycles"], bool)
        or not isinstance(record["cycles"], int)
        or not isinstance(record["written_at"], str)
    ):
        raise ValueError("collector heartbeat is malformed")
    last = record["last_command_at"]
    if last is not None and not isinstance(last, str):
        raise ValueError("collector heartbeat is malformed")
    return Heartbeat(
        state=record["state"],
        pid=record["pid"],
        written_at=timestamp(record["written_at"]),
        last_command_at=None if last is None else timestamp(last),
        cycles=record["cycles"],
    )


class HeartbeatWriter:
    """Callable handed to the cadence loop; remembers the last committed command.

    A write error is tolerated (after the backoff in :func:`write_heartbeat`) until
    nothing has been written successfully for ``give_up_seconds``; only then is it
    raised so the collector halts durably with the real cause. A transient Windows
    sharing violation therefore cannot end a 90-day screen, while a persistent one
    ends it before the external stale check would.
    """

    def __init__(
        self,
        root: Path,
        protocol_sha256: str,
        *,
        clock: Callable[[], datetime] = _now,
        pid: int | None = None,
        give_up_seconds: float = HEARTBEAT_WRITE_GIVE_UP_SECONDS,
    ) -> None:
        if not give_up_seconds > 0:
            raise ValueError("heartbeat give-up window must be positive")
        self.root = Path(root)
        self.digest = protocol_sha256
        self.clock = clock
        self.pid = os.getpid() if pid is None else pid
        self.give_up_seconds = give_up_seconds
        self.cycles = 0
        self.missed = 0
        self._last_command: datetime | None = None
        self._last_success = clock()

    def __call__(self, state: str, status: dict[str, Any] | None = None) -> None:
        if status is not None and status.get("last_command_at") is not None:
            self._last_command = timestamp(status["last_command_at"])
        if state == "cycle_completed" or state in TERMINAL_STATES:
            self.cycles += 1
        at = self.clock()
        try:
            write_heartbeat(
                self.root,
                protocol_sha256=self.digest,
                state=state,
                pid=self.pid,
                at=at,
                last_command_at=self._last_command,
                cycles=self.cycles,
            )
        except OSError:
            self.missed += 1
            if (at - self._last_success).total_seconds() > self.give_up_seconds:
                raise
            return
        self.missed = 0
        self._last_success = at


@dataclass(frozen=True)
class SupervisorHeartbeat:
    state: str
    pid: int
    written_at: datetime
    child_launched_at: datetime | None
    startup_grace_seconds: float
    poll_seconds: float


def write_supervisor_heartbeat(
    root: Path,
    *,
    protocol_sha256: str,
    state: str,
    pid: int,
    at: datetime,
    child_launched_at: datetime | None,
    startup_grace_seconds: float,
    poll_seconds: float,
) -> bool:
    """Best-effort atomic write; returns False instead of raising on I/O errors.

    The supervisor must never die because its own status file is momentarily locked.
    """
    record = dict(
        schema=SUPERVISOR_HEARTBEAT_SCHEMA,
        protocol_sha256=protocol_sha256,
        state=state,
        pid=pid,
        written_at=at.isoformat(),
        child_launched_at=None
        if child_launched_at is None
        else child_launched_at.isoformat(),
        startup_grace_seconds=startup_grace_seconds,
        poll_seconds=poll_seconds,
        production_eligible=False,
    )
    target = root / SUPERVISOR_HEARTBEAT_NAME
    temporary = root / (SUPERVISOR_HEARTBEAT_NAME + ".tmp")
    try:
        if target.is_symlink():
            return False
        with temporary.open("wb") as stream:
            stream.write(canonical_json_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except OSError:
        return False
    return True


def read_supervisor_heartbeat(
    root: Path, protocol_sha256: str
) -> SupervisorHeartbeat | None:
    """Return None when absent or unusable; this file only ever defers a check."""
    path = root / SUPERVISOR_HEARTBEAT_NAME
    try:
        if path.is_symlink() or not path.is_file():
            return None
        record = json.loads(path.read_bytes())
        if (
            record["schema"] != SUPERVISOR_HEARTBEAT_SCHEMA
            or record["protocol_sha256"] != protocol_sha256
            or record["state"] not in {"launching", "monitoring", "stopped"}
        ):
            return None
        launched = record["child_launched_at"]
        return SupervisorHeartbeat(
            state=record["state"],
            pid=int(record["pid"]),
            written_at=timestamp(record["written_at"]),
            child_launched_at=None if launched is None else timestamp(launched),
            startup_grace_seconds=float(record["startup_grace_seconds"]),
            poll_seconds=float(record["poll_seconds"]),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _relaunch_in_progress(
    root: Path, protocol_sha256: str, heartbeat: Heartbeat, now: datetime
) -> bool:
    """True while a live supervisor is inside the startup grace of a fresh child.

    A relaunched child replays the whole journal before its first heartbeat, which
    can legitimately outlast the 150 s stale limit but not the supervisor's own
    grace. Deferring is bounded: the supervisor must itself be alive (fresh status),
    the child must not have beaten since its launch, and the grace must not have
    run out.
    """
    supervisor = read_supervisor_heartbeat(root, protocol_sha256)
    if (
        supervisor is None
        or supervisor.state == "stopped"
        or supervisor.child_launched_at is None
        or supervisor.startup_grace_seconds > MAXIMUM_STARTUP_GRACE_SECONDS
    ):
        return False
    alive_for = max(30.0, 3 * supervisor.poll_seconds)
    return (
        (now - supervisor.written_at).total_seconds() <= alive_for
        and heartbeat.written_at < supervisor.child_launched_at
        and (now - supervisor.child_launched_at).total_seconds()
        <= supervisor.startup_grace_seconds
    )


has_incomplete_cycle = incomplete_cycle_exists


def _failure_present(root: Path) -> bool:
    path = root / _FAILURE
    return path.exists() or path.is_symlink()


def _record_failure(
    root: Path,
    protocol_sha256: str,
    *,
    error_type: str,
    error: str,
    clock: Callable[[], datetime],
    detail: dict[str, Any] | None = None,
    reporter: str = "watchdog",
) -> bool:
    extra: dict[str, Any] = dict(reported_by=reporter)
    if detail:
        extra.update(detail)
    written = write_collection_failure(
        root,
        protocol_sha256=protocol_sha256,
        error_type=error_type,
        error=error,
        extra=extra,
        clock=clock,
    )
    return written is not None


def _finish(
    report: dict[str, Any], state: str, severity: str, **extra: Any
) -> dict[str, Any]:
    return dict(
        report, state=state, severity=severity, healthy=severity == "healthy", **extra
    )


def check_collection(
    root: str | Path,
    *,
    protocol_sha256: str,
    clock: Callable[[], datetime] = _now,
    stale_seconds: float = DEFAULT_STALE_SECONDS,
) -> dict[str, Any]:
    """Stateless liveness check, safe to run from an independent scheduled task.

    After the frozen start a stale or missing heartbeat writes the permanent failure
    marker; before start it only reports, because nothing has been lost yet. While a
    live supervisor is relaunching a child inside its startup grace (journal replay),
    a stale heartbeat is reported as ``relaunch_in_progress`` instead. The journal and
    control database are not modified or even opened for writing.

    ``severity`` is ``healthy``, ``degraded`` (unhealthy, no marker written) or
    ``failed`` (a marker exists or was just written); the CLI maps them to exit
    codes 0, 5 and 3.
    """
    root = Path(root).absolute()
    window = read_protocol_window(root, protocol_sha256)
    if not 0 < stale_seconds <= window.maximum_gap_seconds:
        raise ValueError("stale limit must be positive and within the frozen gap limit")
    now = clock()
    report: dict[str, Any] = dict(
        schema=CHECK_SCHEMA,
        protocol_sha256=protocol_sha256,
        checked_at=now.isoformat(),
        stale_seconds=stale_seconds,
        production_eligible=False,
        failure_written=False,
        heartbeat_age_seconds=None,
        heartbeat_state=None,
        heartbeat_pid=None,
    )
    if _failure_present(root):
        return _finish(report, "halted", "failed")
    try:
        heartbeat = read_heartbeat(root, protocol_sha256)
        invalid = False
    except ValueError:
        heartbeat, invalid = None, True
    started = now >= window.start_at
    if heartbeat is None:
        if started and (now - window.start_at).total_seconds() > (
            window.maximum_gap_seconds
        ):
            written = _record_failure(
                root,
                protocol_sha256,
                error_type="CollectorNeverStarted",
                error="no valid collector heartbeat after the frozen start",
                clock=clock,
                detail=dict(heartbeat_invalid=invalid),
            )
            return _finish(
                report, "failed_never_started", "failed", failure_written=written
            )
        if invalid:
            return _finish(report, "heartbeat_invalid", "degraded")
        return _finish(report, "not_started", "healthy")
    age = (now - heartbeat.written_at).total_seconds()
    report.update(
        heartbeat_age_seconds=age,
        heartbeat_state=heartbeat.state,
        heartbeat_pid=heartbeat.pid,
    )
    if heartbeat.state in TERMINAL_STATES:
        return _finish(report, heartbeat.state, "healthy")
    if age <= stale_seconds:
        return _finish(report, "running", "healthy")
    if not started:
        return _finish(report, "stale_before_start", "degraded")
    if _relaunch_in_progress(root, protocol_sha256, heartbeat, now):
        return _finish(report, "relaunch_in_progress", "healthy")
    written = _record_failure(
        root,
        protocol_sha256,
        error_type="CollectorHeartbeatStale",
        error=f"collector heartbeat is {age:.1f}s old (limit {stale_seconds:.1f}s)",
        clock=clock,
        detail=dict(
            heartbeat_state=heartbeat.state,
            heartbeat_pid=heartbeat.pid,
            heartbeat_age_seconds=age,
        ),
    )
    return _finish(report, "failed_stale", "failed", failure_written=written)


class ExitGuard:
    """Registered with atexit: an exit before a terminal state leaves a failure."""

    def __init__(
        self,
        root: Path,
        protocol_sha256: str,
        *,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self.root = Path(root)
        self.digest = protocol_sha256
        self.clock = clock
        self.armed = True

    def disarm(self) -> None:
        self.armed = False

    def __call__(self) -> None:
        if self.armed:
            self.armed = False
            _record_failure(
                self.root,
                self.digest,
                error_type="CollectorExitedUnfinished",
                error="collector process exited before a terminal state",
                clock=self.clock,
                reporter="atexit",
            )

    def console_event(self, name: str) -> None:
        """Console close / logoff / shutdown: the OS ends the process in seconds."""
        self.armed = False
        _record_failure(
            self.root,
            self.digest,
            error_type="CollectorConsoleClosed",
            error=name,
            clock=self.clock,
            reporter="console_handler",
        )


_KEEP_ALIVE: list[object] = []


def install_termination_handlers(
    on_console_event: Callable[[str], None],
    *,
    signal_module: Any = signal,
    platform: str = sys.platform,
    kernel32: Any = None,
    function_type: Any = None,
) -> list[str]:
    """Route SIGTERM/SIGBREAK to CollectorTerminated and Windows console events to a
    synchronous failure write. Returns the names installed (for evidence/tests)."""

    def raise_terminated(signum: int, frame: Any) -> None:
        raise CollectorTerminated(f"terminated by signal {signum}")

    installed: list[str] = []
    for name in ("SIGTERM", "SIGBREAK"):
        number = getattr(signal_module, name, None)
        if number is not None:
            signal_module.signal(number, raise_terminated)
            installed.append(name)
    if platform == "win32":
        if kernel32 is None:
            kernel32 = getattr(getattr(ctypes, "windll", None), "kernel32", None)
        if function_type is None:
            factory = getattr(ctypes, "WINFUNCTYPE", None)
            function_type = (
                None if factory is None else factory(ctypes.c_int, ctypes.c_uint)
            )
        if kernel32 is not None and function_type is not None:

            def handle(event: int) -> int:
                name = _CONSOLE_EVENTS.get(int(event))
                if name is not None:
                    on_console_event(name)
                return 0  # not handled: default processing continues

            callback = function_type(handle)
            _KEEP_ALIVE.append(callback)
            if not kernel32.SetConsoleCtrlHandler(callback, True):
                raise OSError("could not register the console control handler")
            installed.append("console_ctrl_handler")
    return installed


class ChildProcess(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float) -> int | None: ...


class PopenChild:
    """Adapter giving subprocess.Popen the narrow ChildProcess contract."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self.process = process
        self.pid = process.pid

    def poll(self) -> int | None:
        return self.process.poll()

    def terminate(self) -> None:
        # Popen.terminate() is TerminateProcess on Windows and would skip every
        # in-process hook; CTRL_BREAK reaches the child's SIGBREAK handler instead.
        break_event = getattr(signal, "CTRL_BREAK_EVENT", None)
        if sys.platform == "win32" and break_event is not None:
            self.process.send_signal(break_event)
        else:
            self.process.terminate()

    def kill(self) -> None:
        self.process.kill()

    def wait(self, timeout: float) -> int | None:
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None


def collector_command(root: Path, protocol_sha256: str, python: str) -> list[str]:
    return [
        python,
        "-m",
        "trade_rl.evaluation.paper.cli",
        "run",
        "--root",
        str(root),
        "--protocol-sha256",
        protocol_sha256,
    ]


def popen_launcher(
    root: str | Path,
    protocol_sha256: str,
    log_path: str | Path,
    *,
    python: str = sys.executable,
    popen: Callable[..., Any] = subprocess.Popen,
) -> Callable[[], ChildProcess]:
    """Build a launcher for the real collector child (never called by tests)."""
    root_path = Path(root).absolute()
    log = Path(log_path).absolute()
    if log.is_relative_to(root_path):
        raise ValueError("supervisor log must live outside the sealed root")

    def launch() -> ChildProcess:
        options: dict[str, Any] = {}
        if sys.platform == "win32":
            options["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        with log.open("ab") as stream:
            process = popen(
                collector_command(root_path, protocol_sha256, python),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                **options,
            )
        return PopenChild(process)

    return launch


@dataclass(frozen=True)
class WatchdogPolicy:
    poll_seconds: float = 5.0
    stale_seconds: float = DEFAULT_STALE_SECONDS
    startup_grace_seconds: float = 300.0
    restart_margin_seconds: float = 60.0
    restart_delay_seconds: float = 2.0
    max_restarts: int = 5
    terminate_grace_seconds: float = 10.0

    def __post_init__(self) -> None:
        for name in (
            "poll_seconds",
            "stale_seconds",
            "startup_grace_seconds",
            "restart_margin_seconds",
            "terminate_grace_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not value > 0:
                raise ValueError(f"{name} must be positive")
        if self.restart_delay_seconds < 0 or self.max_restarts < 0:
            raise ValueError("restart settings must not be negative")
        if self.startup_grace_seconds > MAXIMUM_STARTUP_GRACE_SECONDS:
            raise ValueError("startup grace exceeds the bounded maximum")


def supervise_collector(
    root: str | Path,
    *,
    protocol_sha256: str,
    launch: Callable[[], ChildProcess],
    policy: WatchdogPolicy | None = None,
    clock: Callable[[], datetime] = _now,
    sleep: Callable[[float], None] = time.sleep,
    publish: Callable[[dict[str, Any]], None] = lambda _event: None,
) -> dict[str, Any]:
    """Run the collector child; relaunch only when that cannot lose evidence.

    Every non-restartable end writes the permanent failure marker before returning,
    so a stopped supervisor can never leave an unexplained silent screen.
    """
    root = Path(root).absolute()
    policy = policy or WatchdogPolicy()
    window = read_protocol_window(root, protocol_sha256)
    if policy.stale_seconds > window.maximum_gap_seconds:
        raise ValueError("stale limit cannot exceed the frozen observation-gap limit")
    restarts = 0
    launched_at: datetime | None = None

    def status_beat(state: str) -> None:
        write_supervisor_heartbeat(
            root,
            protocol_sha256=protocol_sha256,
            state=state,
            pid=os.getpid(),
            at=clock(),
            child_launched_at=launched_at,
            startup_grace_seconds=policy.startup_grace_seconds,
            poll_seconds=policy.poll_seconds,
        )

    def finish(
        outcome: str, exit_code: int, *, record_stop: bool = True, **extra: Any
    ) -> dict[str, Any]:
        # A supervisor that never launched a child owns no status file, and a
        # contention exit must leave the live owner's supervisor status untouched.
        if record_stop and launched_at is not None:
            status_beat("stopped")
        result = dict(
            schema="paper_collector_supervision_v1",
            outcome=outcome,
            exit_code=exit_code,
            restarts=restarts,
            production_eligible=False,
            **extra,
        )
        publish(dict(event="supervision_finished", **result))
        return result

    def heartbeat_or_none() -> Heartbeat | None:
        try:
            return read_heartbeat(root, protocol_sha256)
        except ValueError:
            return None

    def fail(error_type: str, error: str, detail: dict[str, Any]) -> dict[str, Any]:
        written = _record_failure(
            root,
            protocol_sha256,
            error_type=error_type,
            error=error,
            clock=clock,
            detail=detail,
        )
        return finish(
            "failed",
            EXIT_FAILED,
            error_type=error_type,
            failure_written=written,
            detail=detail,
        )

    while True:
        if _failure_present(root):
            return finish("halted", EXIT_FAILED)
        before = heartbeat_or_none()
        if before is not None and before.state in TERMINAL_STATES:
            return finish("already_terminal", 0, heartbeat_state=before.state)
        if collector_lock_held(root):
            # A live collector (for example the survivor of a supervisor restart)
            # owns this root. Spawning would only collide; recording a failure
            # would invalidate a healthy screen. Refuse without writing a marker.
            publish(dict(event="collector_lock_held"))
            return finish("lock_contention", EXIT_LOCK_HELD)
        started = launched_at = clock()
        status_beat("launching")
        child = launch()
        publish(dict(event="launched", pid=child.pid, at=started.isoformat()))
        hung = False
        code: int | None = None
        while True:
            code = child.poll()
            if code is not None:
                break
            now = clock()
            status_beat("monitoring")
            heartbeat = heartbeat_or_none()
            if heartbeat is None or heartbeat.written_at < started:
                limit, reference = policy.startup_grace_seconds, started
            else:
                limit, reference = policy.stale_seconds, heartbeat.written_at
            if (now - reference).total_seconds() > limit:
                hung = True
                break
            sleep(policy.poll_seconds)
        if hung:
            fail_result = fail(
                "CollectorHung",
                "collector stopped producing heartbeats",
                dict(pid=child.pid, limit_seconds=limit),
            )
            _stop(child, policy)
            return fail_result
        publish(dict(event="child_exited", exit_code=code, pid=child.pid))
        if _failure_present(root):
            return finish("halted", EXIT_FAILED, child_exit_code=code)
        last = heartbeat_or_none()
        if last is not None and last.state in TERMINAL_STATES:
            return finish(
                "finished",
                0 if code == 0 else 1,
                child_exit_code=code,
                heartbeat_state=last.state,
            )
        detail: dict[str, Any] = dict(child_exit_code=code, pid=child.pid)
        if code == 0:
            return fail(
                "CollectorExitedEarly",
                "collector exited 0 without a terminal state",
                detail,
            )
        if code == EXIT_LOCK_HELD:
            # Lost a start-up race for the lock to another live collector. Not a
            # collector refusal: write nothing and leave the owner undisturbed.
            return finish(
                "lock_contention",
                EXIT_LOCK_HELD,
                record_stop=False,
                child_exit_code=code,
            )
        if code == 2:
            return fail(
                "CollectorRefused",
                "collector refused to run; relaunching cannot change that",
                detail,
            )
        try:
            mid_cycle = has_incomplete_cycle(root)
        except (sqlite3.Error, ValueError) as error:
            return fail("CollectorControlUnreadable", str(error), detail)
        if mid_cycle:
            return fail(
                "CollectorDiedMidCycle",
                "collector process ended inside an unacknowledged cycle",
                detail,
            )
        now = clock()
        reference = window.start_at
        if last is not None and last.last_command_at is not None:
            reference = max(window.start_at, last.last_command_at)
        projected = (now - reference).total_seconds() + policy.restart_margin_seconds
        if projected > window.maximum_gap_seconds:
            return fail(
                "CollectorGapUnrecoverable",
                "a relaunch could not resume inside the frozen observation-gap limit",
                dict(detail, projected_gap_seconds=projected),
            )
        if restarts >= policy.max_restarts:
            return fail("CollectorRestartLimit", "restart limit reached", detail)
        restarts += 1
        publish(dict(event="restarting", restarts=restarts))
        sleep(policy.restart_delay_seconds)


def _stop(child: ChildProcess, policy: WatchdogPolicy) -> None:
    """Best-effort termination; the failure marker is already durable."""
    try:
        child.terminate()
        if child.wait(policy.terminate_grace_seconds) is None:
            child.kill()
            child.wait(policy.terminate_grace_seconds)
    except OSError:
        pass
