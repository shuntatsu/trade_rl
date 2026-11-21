"""Linux operating platform for the attempt-3 carry paper screen (systemd user service).

Attempt 3 was designed around a Windows host (Task Scheduler, ``C:\\`` run roots). This
module is the Linux counterpart of that operating platform and nothing else: it builds
the run layout, renders the systemd *user* units and reads the clock state. It changes
no economic or protocol parameter, never touches a run root, a service or the network,
and writes a unit file only when ``render --output-dir`` is given (exclusive create).
Enabling, starting and stopping the units, ``loginctl enable-linger`` and every other
change of host state stay with the owner (``systemctl --user`` needs the user's own
D-Bus session).

Fail-closed equivalence with the Windows tasks:

* ``carry-a3-supervisor.service`` runs ``attempt3 launch`` at boot (with lingering) and
  restarts it on failure at most three times a minute apart, never after a permanent
  failure marker (exit 3), a live collector owning the root (exit 4) or a rejected
  terminal state (exit 1). The default kill mode stays ``control-group``: stopping the
  unit, a logout without lingering or a shutdown sends SIGTERM to the collector, which
  writes the permanent marker (SIGHUP is handled the same way), so the screen ends
  INVALID instead of silently losing observations.
* ``carry-a3-check.timer`` runs the stateless ``cli check`` every minute; exit 3 or 5
  leaves the check service in the failed state for monitoring.
* The clock is read, never set, from chrony or systemd-timesyncd.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

ATTEMPT = "attempt3"
RUNS_DIRECTORY = "trade_rl_runs"
FROZEN_DIRECTORY = "trade_rl_frozen"
PREDECESSOR_ROOT_NAME = "root-20261107"
SUPERVISOR_UNIT = "carry-a3-supervisor.service"
CHECK_UNIT = "carry-a3-check.service"
CHECK_TIMER = "carry-a3-check.timer"
# The Windows task restarts three times, one minute apart; systemd counts the first
# start as well, so four starts inside the interval equal one start plus three restarts.
RESTART_SECONDS = 60
START_LIMIT_INTERVAL_SECONDS = 600
START_LIMIT_BURST = 4
# 1 rejected terminal state, 3 permanent failure marker, 4 a live collector owns the root.
NO_RESTART_EXIT_CODES = (1, 3, 4)
STOP_TIMEOUT_SECONDS = 30
# An operating pre-flight bound, not a protocol parameter: a quarter of the 1000 ms
# future-skew tolerance, so the local clock stays well inside it.
MAX_CLOCK_OFFSET_SECONDS = 0.25
CLOCK_COMMAND_TIMEOUT_SECONDS = 10.0
CLOCK_SCHEMA = "paper_linux_clock_report_v1"
LAYOUT_SCHEMA = "paper_linux_layout_v1"
UNITS_SCHEMA = "paper_linux_units_v1"
# The only commands the clock report may run: all of them read state.
READ_ONLY_CLOCK_COMMANDS = (
    ("timedatectl", "show", "--property=NTPSynchronized"),
    ("chronyc", "tracking"),
    ("timedatectl", "timesync-status"),
)

_HOME = re.compile(r"/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*")
_DATE = re.compile(r"[0-9]{8}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_BARE = re.compile(r"[A-Za-z0-9_./:@+=,-]+")
_QUOTED = re.compile(r"[A-Za-z0-9_./:@+=,#-]+")


class LinuxServiceRefusal(ValueError):
    """An input that cannot be rendered into a safe unit or layout."""


@dataclass(frozen=True)
class LinuxLayout:
    """Run, probe, log and frozen-checkout paths of one attempt-3 launch."""

    home: PurePosixPath
    seal_date: str
    launch_commit: str

    @property
    def attempt_runs(self) -> PurePosixPath:
        return self.home / RUNS_DIRECTORY / ATTEMPT

    @property
    def root(self) -> PurePosixPath:
        return self.attempt_runs / f"root-{self.seal_date}"

    @property
    def probe_root(self) -> PurePosixPath:
        return self.attempt_runs / f"storage-probe-{self.seal_date}"

    @property
    def probe_report(self) -> PurePosixPath:
        return self.attempt_runs / f"storage-probe-{self.seal_date}.report.json"

    @property
    def log_directory(self) -> PurePosixPath:
        return self.attempt_runs / "logs"

    @property
    def supervisor_log(self) -> PurePosixPath:
        return self.log_directory / f"supervisor-{self.seal_date}.log"

    @property
    def checkout(self) -> PurePosixPath:
        return self.home / FROZEN_DIRECTORY / f"{ATTEMPT}-{self.launch_commit[:8]}"

    @property
    def python(self) -> PurePosixPath:
        return self.checkout / ".venv" / "bin" / "python"

    @property
    def predecessor_root(self) -> PurePosixPath:
        return self.home / RUNS_DIRECTORY / "attempt2" / PREDECESSOR_ROOT_NAME

    def to_dict(self) -> dict[str, str]:
        return dict(
            schema=LAYOUT_SCHEMA,
            home=str(self.home),
            seal_date=self.seal_date,
            launch_commit=self.launch_commit,
            attempt_runs=str(self.attempt_runs),
            root=str(self.root),
            probe_root=str(self.probe_root),
            probe_report=str(self.probe_report),
            log_directory=str(self.log_directory),
            supervisor_log=str(self.supervisor_log),
            checkout=str(self.checkout),
            python=str(self.python),
            predecessor_root=str(self.predecessor_root),
        )


def build_layout(home: str, seal_date: str, launch_commit: str) -> LinuxLayout:
    """Validate the three inputs and return the layout; refuse anything unsafe.

    ``home`` must be an absolute POSIX path made of unit-safe characters, so that no
    path ever needs systemd quoting, ``%`` specifier or ``$`` escaping.
    """
    if not isinstance(home, str) or _HOME.fullmatch(home) is None:
        raise LinuxServiceRefusal(
            "home_must_be_an_absolute_posix_path_of_safe_characters"
        )
    path = PurePosixPath(home)
    if str(path) != home or ".." in path.parts or "." in path.parts:
        raise LinuxServiceRefusal("home_must_be_a_normalised_path")
    if not isinstance(seal_date, str) or _DATE.fullmatch(seal_date) is None:
        raise LinuxServiceRefusal("seal_date_must_be_yyyymmdd")
    try:
        datetime.strptime(seal_date, "%Y%m%d")
    except ValueError as error:
        raise LinuxServiceRefusal("seal_date_is_not_a_calendar_date") from error
    if not isinstance(launch_commit, str) or _COMMIT.fullmatch(launch_commit) is None:
        raise LinuxServiceRefusal("launch_commit_must_be_a_full_lowercase_sha1")
    return LinuxLayout(path, seal_date, launch_commit)


def _argument(value: str) -> str:
    """One ExecStart word; a ``#`` is double-quoted, anything else unsafe is refused."""
    if _BARE.fullmatch(value) is not None:
        return value
    if _QUOTED.fullmatch(value) is not None:
        return f'"{value}"'
    raise LinuxServiceRefusal("argument_contains_a_character_unsafe_for_a_unit")


def _command(words: Sequence[str]) -> str:
    return " ".join(_argument(word) for word in words)


def supervisor_command(
    layout: LinuxLayout,
    *,
    start_at: datetime,
    authorization: str,
    protocol_sha256: str,
) -> list[str]:
    """The attempt-3 ``launch`` command, exactly as the procedure states it."""
    from trade_rl.evaluation.paper.attempt2 import authorization_reasons

    reasons = authorization_reasons(authorization)
    if start_at.tzinfo is None or start_at.utcoffset() is None:
        reasons.append("start_not_timezone_aware")
    if (
        not isinstance(protocol_sha256, str)
        or _DIGEST.fullmatch(protocol_sha256) is None
    ):
        reasons.append("protocol_sha256_must_be_a_lowercase_sha256")
    if reasons:
        raise LinuxServiceRefusal(",".join(reasons))
    return [
        str(layout.python),
        "-m",
        "trade_rl.evaluation.paper.attempt3",
        "launch",
        "--root",
        str(layout.root),
        "--start-at",
        start_at.astimezone(UTC).isoformat(),
        "--authorization",
        authorization,
        "--allow-network",
        "--protocol-sha256",
        protocol_sha256,
        "--log-file",
        str(layout.supervisor_log),
    ]


def check_command(layout: LinuxLayout, *, protocol_sha256: str) -> list[str]:
    """The stateless liveness check the Windows ``carry-a3-check`` task runs."""
    if (
        not isinstance(protocol_sha256, str)
        or _DIGEST.fullmatch(protocol_sha256) is None
    ):
        raise LinuxServiceRefusal("protocol_sha256_must_be_a_lowercase_sha256")
    return [
        str(layout.python),
        "-m",
        "trade_rl.evaluation.paper.cli",
        "check",
        "--root",
        str(layout.root),
        "--protocol-sha256",
        protocol_sha256,
    ]


_HEADER = (
    "# Rendered by trade_rl.evaluation.paper.linux_service for the attempt-3 carry\n"
    "# paper screen (public data only, no orders). Do not edit after the seal.\n"
)


def render_supervisor_unit(
    layout: LinuxLayout,
    *,
    start_at: datetime,
    authorization: str,
    protocol_sha256: str,
) -> str:
    command = _command(
        supervisor_command(
            layout,
            start_at=start_at,
            authorization=authorization,
            protocol_sha256=protocol_sha256,
        )
    )
    prevented = " ".join(str(code) for code in NO_RESTART_EXIT_CODES)
    return (
        f"{_HEADER}"
        "[Unit]\n"
        "Description=Attempt 3 carry paper screen supervisor "
        "(public data only, no orders)\n"
        f"StartLimitIntervalSec={START_LIMIT_INTERVAL_SECONDS}\n"
        f"StartLimitBurst={START_LIMIT_BURST}\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"WorkingDirectory={_argument(str(layout.checkout))}\n"
        f"ExecStart={command}\n"
        "Restart=on-failure\n"
        f"RestartSec={RESTART_SECONDS}\n"
        f"RestartPreventExitStatus={prevented}\n"
        "KillMode=control-group\n"
        "SendSIGHUP=no\n"
        f"TimeoutStopSec={STOP_TIMEOUT_SECONDS}\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def render_check_service(layout: LinuxLayout, *, protocol_sha256: str) -> str:
    command = _command(check_command(layout, protocol_sha256=protocol_sha256))
    return (
        f"{_HEADER}"
        "[Unit]\n"
        "Description=Attempt 3 carry paper screen liveness check "
        "(stateless, one run per timer tick)\n"
        "\n"
        "[Service]\n"
        "Type=oneshot\n"
        f"WorkingDirectory={_argument(str(layout.checkout))}\n"
        f"ExecStart={command}\n"
    )


def render_check_timer() -> str:
    return (
        f"{_HEADER}"
        "[Unit]\n"
        "Description=Attempt 3 carry paper screen liveness check, every minute\n"
        "\n"
        "[Timer]\n"
        "OnCalendar=minutely\n"
        "AccuracySec=1s\n"
        "Persistent=false\n"
        f"Unit={CHECK_UNIT}\n"
        "\n"
        "[Install]\n"
        "WantedBy=timers.target\n"
    )


def render_units(
    layout: LinuxLayout,
    *,
    start_at: datetime,
    authorization: str,
    protocol_sha256: str,
) -> dict[str, str]:
    """All three unit files by name."""
    return {
        SUPERVISOR_UNIT: render_supervisor_unit(
            layout,
            start_at=start_at,
            authorization=authorization,
            protocol_sha256=protocol_sha256,
        ),
        CHECK_UNIT: render_check_service(layout, protocol_sha256=protocol_sha256),
        CHECK_TIMER: render_check_timer(),
    }


def write_units(directory: Path, units: dict[str, str]) -> list[str]:
    """Create each unit file exclusively in an existing directory; never overwrite."""
    target = Path(directory)
    if target.is_symlink() or not target.is_dir():
        raise LinuxServiceRefusal("output_directory_must_be_an_existing_directory")
    names = sorted(units)
    for name in names:
        if (target / name).exists() or (target / name).is_symlink():
            raise LinuxServiceRefusal(f"unit_file_exists:{name}")
    written: list[str] = []
    for name in names:
        with (target / name).open("xb") as stream:
            stream.write(units[name].encode("utf-8"))
        written.append(name)
    return written


# ---------------------------------------------------------------- clock (read only)

Runner = Callable[[Sequence[str]], tuple[int, str]]

_CHRONY_LEAP = re.compile(r"^Leap status\s*:\s*(.+?)\s*$", re.MULTILINE)
_CHRONY_SYSTEM_TIME = re.compile(
    r"^System time\s*:\s*([0-9.]+) seconds (fast|slow) of NTP time\s*$",
    re.MULTILINE,
)
_CHRONY_STRATUM = re.compile(r"^Stratum\s*:\s*(\d+)\s*$", re.MULTILINE)
_TIMESYNC_OFFSET = re.compile(
    r"^\s*Offset:\s*([+-]?[0-9.]+)(us|ms|s)\s*$", re.MULTILINE
)
_UNIT_SECONDS = {"us": 1e-6, "ms": 1e-3, "s": 1.0}


def run_read_only(arguments: Sequence[str]) -> tuple[int, str]:
    """Run one allow-listed read-only command; a missing tool is exit status 127."""
    if tuple(arguments) not in READ_ONLY_CLOCK_COMMANDS:
        raise LinuxServiceRefusal("command_is_not_an_allowed_read_only_command")
    try:
        completed = subprocess.run(
            list(arguments),
            capture_output=True,
            text=True,
            check=False,
            timeout=CLOCK_COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return 127, ""
    return completed.returncode, completed.stdout


def _kernel_synchronized(run: Runner) -> bool | None:
    code, output = run(READ_ONLY_CLOCK_COMMANDS[0])
    if code != 0:
        return None
    for line in output.splitlines():
        if line.strip() == "NTPSynchronized=yes":
            return True
        if line.strip() == "NTPSynchronized=no":
            return False
    return None


def _chrony(run: Runner) -> tuple[bool, float | None] | None:
    """``(normal, signed offset in seconds)``; None when chrony does not answer."""
    code, output = run(READ_ONLY_CLOCK_COMMANDS[1])
    if code != 0:
        return None
    leap = _CHRONY_LEAP.search(output)
    stratum = _CHRONY_STRATUM.search(output)
    offset = _CHRONY_SYSTEM_TIME.search(output)
    normal = (
        leap is not None
        and leap.group(1) == "Normal"
        and stratum is not None
        and 0 < int(stratum.group(1)) < 16
    )
    if offset is None:
        return normal, None
    sign = 1.0 if offset.group(2) == "fast" else -1.0
    return normal, sign * float(offset.group(1))


def _timesyncd(run: Runner) -> float | None:
    code, output = run(READ_ONLY_CLOCK_COMMANDS[2])
    if code != 0:
        return None
    match = _TIMESYNC_OFFSET.search(output)
    if match is None:
        return None
    return float(match.group(1)) * _UNIT_SECONDS[match.group(2)]


def clock_report(
    run: Runner | None = None,
    *,
    max_offset_seconds: float = MAX_CLOCK_OFFSET_SECONDS,
) -> dict[str, Any]:
    """Read-only clock state from chrony or systemd-timesyncd; fail closed.

    ``ok`` needs the kernel clock marked synchronised, an answering NTP client
    (chrony first, then systemd-timesyncd), chrony reporting a normal leap status and a
    sane stratum, and a measured offset within ``max_offset_seconds``. Nothing is set,
    stepped or restarted, and no network request is made by this process.
    """
    if run is None:
        run = run_read_only
    reasons: list[str] = []
    synchronized = _kernel_synchronized(run)
    if synchronized is not True:
        reasons.append(
            "kernel_clock_not_synchronized"
            if synchronized is False
            else "kernel_clock_state_unavailable"
        )
    source: str | None = None
    offset: float | None = None
    chrony = _chrony(run)
    if chrony is not None:
        source = "chrony"
        normal, offset = chrony
        if not normal:
            reasons.append("chrony_leap_or_stratum_not_normal")
    else:
        offset = _timesyncd(run)
        if offset is not None:
            source = "systemd-timesyncd"
        else:
            reasons.append("no_ntp_client_answered")
    if source is not None:
        if offset is None:
            reasons.append("offset_unavailable")
        elif abs(offset) > max_offset_seconds:
            reasons.append("offset_exceeds_limit")
    return dict(
        schema=CLOCK_SCHEMA,
        ok=not reasons,
        source=source,
        kernel_synchronized=synchronized,
        offset_seconds=offset,
        max_offset_seconds=max_offset_seconds,
        reasons=reasons,
        commands=[list(command) for command in READ_ONLY_CLOCK_COMMANDS],
        production_eligible=False,
    )


# ---------------------------------------------------------------------------- CLI


def _print(value: dict[str, Any]) -> None:
    print(json.dumps(value, allow_nan=False, sort_keys=True), flush=True)


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must carry a UTC offset")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("paths", "render"):
        child = commands.add_parser(name)
        child.add_argument("--home", default=str(Path.home().as_posix()))
        child.add_argument("--seal-date", required=True)
        child.add_argument("--launch-commit", required=True)
        if name == "render":
            child.add_argument("--start-at", type=_timestamp, required=True)
            child.add_argument("--authorization", required=True)
            child.add_argument("--protocol-sha256", required=True)
            child.add_argument("--output-dir", type=Path, default=None)
    commands.add_parser("clock")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "clock":
            report = clock_report()
            _print(report)
            return 0 if report["ok"] else 1
        layout = build_layout(args.home, args.seal_date, args.launch_commit)
        if args.command == "paths":
            _print(layout.to_dict())
            return 0
        units = render_units(
            layout,
            start_at=args.start_at,
            authorization=args.authorization,
            protocol_sha256=args.protocol_sha256,
        )
        written = [] if args.output_dir is None else write_units(args.output_dir, units)
        _print(
            dict(
                schema=UNITS_SCHEMA,
                layout=layout.to_dict(),
                units=units,
                written=written,
                production_eligible=False,
            )
        )
        return 0
    except LinuxServiceRefusal as refusal:
        print(
            json.dumps(dict(refused=str(refusal), production_eligible=False)),
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
