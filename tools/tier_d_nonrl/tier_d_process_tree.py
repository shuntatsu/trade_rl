from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, TypeGuard, cast

MAX_PROCESS_ROWS = 16_384
MAX_TRACKED_IDENTITIES = 16_384
_WINDOWS_SNAPSHOT_TIMEOUT_SECONDS = 30
_WINDOWS_FILETIME_EPOCH_TICKS = 504_911_232_000_000_000
_WINDOWS_CIM_TIME_RESOLUTION_TICKS = 10


class ProcessSnapshotError(RuntimeError):
    """A process snapshot was incomplete or could not be validated."""


@dataclass(frozen=True, slots=True, order=True)
class ProcessIdentity:
    pid: int
    creation_time: int


@dataclass(frozen=True, slots=True)
class ProcessUsage:
    """One process row from a single OS snapshot.

    ``creation_time`` is an opaque, platform-local generation token. It is
    compared only with values from the same host and sampler instance.
    """

    pid: int
    parent_pid: int
    creation_time: int
    rss_bytes: int

    @property
    def identity(self) -> ProcessIdentity:
        return ProcessIdentity(self.pid, self.creation_time)


@dataclass(frozen=True, slots=True)
class ProcessTreeSample:
    root_rss_bytes: int
    descendants_rss_bytes: int
    rss_bytes: int
    process_count: int


class ProcessTreeSampler:
    """Measure a launched PID and its observed descendants by PID generation.

    A process remains owned after its parent exits if that exact PID generation
    was observed in an earlier sample. A reused PID is a different identity and
    is never included merely because its numeric PID matches an old process.
    Invalid or partial snapshots raise instead of returning a low RSS value.
    """

    def __init__(self, root_identity: ProcessIdentity) -> None:
        if (
            not isinstance(root_identity, ProcessIdentity)
            or not _is_int(root_identity.pid)
            or root_identity.pid <= 0
            or not _is_int(root_identity.creation_time)
            or root_identity.creation_time <= 0
        ):
            raise ValueError("root_identity must contain a positive PID generation")
        self._root_identity = root_identity
        self._owned_identities: set[ProcessIdentity] = {root_identity}
        self._has_sampled = False

    @classmethod
    def from_popen(cls, process: subprocess.Popen[Any]) -> ProcessTreeSampler:
        """Pin the launched process generation from its native process handle."""

        return cls(process_identity_from_popen(process))

    def sample(
        self, records: Iterable[ProcessUsage] | None = None
    ) -> ProcessTreeSample:
        """Return one complete process-tree RSS measurement.

        ``records`` is an injection point for deterministic tests. Production
        callers should omit it to take an OS-native snapshot.
        """

        rows = _validate_snapshot(
            capture_process_snapshot() if records is None else records
        )
        by_pid = {row.pid: row for row in rows}

        root = by_pid.get(self._root_identity.pid)
        if not self._has_sampled and (
            root is None or root.identity != self._root_identity
        ):
            raise ProcessSnapshotError(
                "pinned root process generation is absent from initial snapshot"
            )
        next_owned = set(self._owned_identities)

        # Only a current parent row whose exact creation identity is already
        # owned can introduce a new descendant. A queue walks each current
        # parent edge at most once, while retaining previously observed
        # processes that were reparented after an owned parent exited.
        children_by_parent: dict[int, list[ProcessUsage]] = {}
        for row in rows:
            children_by_parent.setdefault(row.parent_pid, []).append(row)
        active_owned = {row.identity for row in rows if row.identity in next_owned}
        pending = list(active_owned)
        visited: set[ProcessIdentity] = set()
        while pending:
            parent_identity = pending.pop()
            if parent_identity in visited:
                continue
            visited.add(parent_identity)
            for row in children_by_parent.get(parent_identity.pid, ()):
                # The parent identity is active in this exact snapshot. The
                # dictionary has only one row per PID, so a reused PID cannot
                # borrow ownership from its earlier generation.
                if row.identity in next_owned:
                    continue
                next_owned.add(row.identity)
                pending.append(row.identity)
                if len(next_owned) > MAX_TRACKED_IDENTITIES:
                    raise ProcessSnapshotError("owned process identity limit exceeded")

        active = [row for row in rows if row.identity in next_owned]
        root_rss_bytes = sum(
            row.rss_bytes for row in active if row.identity == self._root_identity
        )
        descendants_rss_bytes = sum(
            row.rss_bytes for row in active if row.identity != self._root_identity
        )

        self._owned_identities = next_owned
        self._has_sampled = True
        return ProcessTreeSample(
            root_rss_bytes=root_rss_bytes,
            descendants_rss_bytes=descendants_rss_bytes,
            rss_bytes=root_rss_bytes + descendants_rss_bytes,
            process_count=len(active),
        )


def capture_process_snapshot() -> tuple[ProcessUsage, ...]:
    """Capture all visible process rows using the current OS native interface."""

    if os.name == "nt":
        return _capture_windows_snapshot()
    if sys.platform.startswith("linux"):
        return _capture_linux_snapshot()
    raise ProcessSnapshotError("process snapshots are supported on Windows and Linux")


def process_identity_from_popen(
    process: subprocess.Popen[Any],
) -> ProcessIdentity:
    """Read the exact launched process generation before tracking descendants."""

    pid = process.pid
    if not _is_int(pid) or pid <= 0:
        raise ProcessSnapshotError("launched process has no valid PID")
    if os.name == "nt":
        handle_creation_time = _windows_creation_time_from_handle(process)
        matching_row = next(
            (row for row in capture_process_snapshot() if row.pid == pid), None
        )
        if matching_row is None or not _windows_creation_times_match(
            matching_row.creation_time, handle_creation_time
        ):
            raise ProcessSnapshotError(
                "Windows process table does not match the launched process handle"
            )
        # CIM DateTime has microsecond resolution (10 FILETIME ticks). Store
        # the exact table identity used by later snapshots after validating it
        # against the Popen-owned handle generation within that precision.
        creation_time = matching_row.creation_time
    elif sys.platform.startswith("linux"):
        # A live Popen child cannot have its PID reused. If it exited already,
        # poll() reaps it and identity pinning fails instead of binding a reused
        # PID from a later procfs read.
        if process.poll() is not None:
            raise ProcessSnapshotError("launched process exited before identity pin")
        try:
            observed_pid, _parent_pid, creation_time = _parse_linux_stat(
                _read_proc_text(Path("/proc") / str(pid) / "stat")
            )
        except (OSError, ProcessSnapshotError) as exc:
            raise ProcessSnapshotError(
                "cannot pin launched Linux process identity"
            ) from exc
        if observed_pid != pid:
            raise ProcessSnapshotError("launched Linux process PID was reused")
    else:
        raise ProcessSnapshotError("process identity is supported on Windows and Linux")
    if creation_time <= 0:
        raise ProcessSnapshotError("launched process creation time is invalid")
    return ProcessIdentity(pid=pid, creation_time=creation_time)


def _windows_creation_times_match(snapshot_ticks: int, handle_ticks: int) -> bool:
    expected_snapshot_ticks = (
        handle_ticks - handle_ticks % _WINDOWS_CIM_TIME_RESOLUTION_TICKS
    )
    return snapshot_ticks == expected_snapshot_ticks


def _windows_creation_time_from_handle(
    process: subprocess.Popen[Any],
) -> int:
    handle = getattr(process, "_handle", None)
    if handle is None:
        raise ProcessSnapshotError("Windows Popen process handle is unavailable")

    class _FileTime(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    try:
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except (AttributeError, OSError) as exc:
        raise ProcessSnapshotError("Windows process handle API is unavailable") from exc
    get_process_times = kernel32.GetProcessTimes
    get_process_times.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
        ctypes.POINTER(_FileTime),
    )
    get_process_times.restype = wintypes.BOOL
    creation = _FileTime()
    exit_time = _FileTime()
    kernel_time = _FileTime()
    user_time = _FileTime()
    try:
        success = get_process_times(
            wintypes.HANDLE(handle),
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ProcessSnapshotError("cannot read Windows process creation time") from exc
    if not success:
        raise ProcessSnapshotError("cannot read Windows process creation time")
    return (creation.high << 32) | creation.low


def _capture_linux_snapshot() -> tuple[ProcessUsage, ...]:
    sysconf = cast(Callable[[str], int] | None, getattr(os, "sysconf", None))
    if sysconf is None:
        raise ProcessSnapshotError("Linux page size query is unavailable")
    page_size = sysconf("SC_PAGE_SIZE")
    if not _is_int(page_size) or page_size <= 0:
        raise ProcessSnapshotError("invalid Linux page size")

    rows: list[ProcessUsage] = []
    try:
        entries = os.scandir("/proc")
    except OSError as exc:
        raise ProcessSnapshotError("cannot enumerate Linux process table") from exc

    with entries:
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                stat_path = Path(entry.path) / "stat"
                statm_path = Path(entry.path) / "statm"
                first = _parse_linux_stat(_read_proc_text(stat_path))
                statm_fields = _read_proc_text(statm_path).split()
                if len(statm_fields) < 2:
                    raise ProcessSnapshotError("malformed Linux statm row")
                resident_pages = _parse_nonnegative_integer(statm_fields[1])
                second = _parse_linux_stat(_read_proc_text(stat_path))
            except (FileNotFoundError, ProcessLookupError):
                # A process that exited while procfs was read no longer owns
                # resident memory at the end of this sample.
                continue
            except PermissionError as exc:
                raise ProcessSnapshotError(
                    "Linux process snapshot is incomplete"
                ) from exc
            except OSError as exc:
                raise ProcessSnapshotError(
                    "Linux process snapshot is incomplete"
                ) from exc

            if first[0] != second[0] or first[2] != second[2]:
                # The PID changed generation while the non-atomic procfs read
                # was in flight. Do not combine its old identity and new RSS.
                continue
            # Parentage may change during a reparenting event without the
            # process itself changing generation. Use the later parent value.
            pid, parent_pid, creation_time = second
            if pid != int(entry.name):
                raise ProcessSnapshotError("Linux process directory PID mismatch")
            rows.append(
                ProcessUsage(
                    pid=pid,
                    parent_pid=parent_pid,
                    creation_time=creation_time,
                    rss_bytes=resident_pages * page_size,
                )
            )
            if len(rows) > MAX_PROCESS_ROWS:
                raise ProcessSnapshotError("Linux process row limit exceeded")

    return _validate_snapshot(rows)


def _parse_linux_stat(text: str) -> tuple[int, int, int]:
    # comm is parenthesized but may itself contain spaces and closing parens.
    closing = text.rfind(")")
    opening = text.find("(")
    if opening <= 0 or closing <= opening:
        raise ProcessSnapshotError("malformed Linux stat row")
    try:
        pid = int(text[:opening].strip())
        fields = text[closing + 1 :].split()
        # The suffix begins with field 3 (state); ppid is field 4 and starttime
        # is field 22, hence offset 1 and 19 respectively.
        parent_pid = int(fields[1])
        creation_time = int(fields[19])
    except (IndexError, ValueError) as exc:
        raise ProcessSnapshotError("malformed Linux stat row") from exc
    return pid, parent_pid, creation_time


def _read_proc_text(path: Path) -> str:
    try:
        return path.read_bytes().decode("ascii", errors="replace")
    except FileNotFoundError:
        raise
    except PermissionError:
        raise
    except OSError:
        raise


def _capture_windows_snapshot() -> tuple[ProcessUsage, ...]:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    if executable is None:
        raise ProcessSnapshotError("Windows PowerShell is unavailable")

    script = r"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$rows = @(Get-CimInstance -ClassName Win32_Process -Property ProcessId,ParentProcessId,CreationDate,WorkingSetSize)
$payload = foreach ($row in $rows) {
    if ($null -eq $row.ProcessId -or $null -eq $row.ParentProcessId -or $null -eq $row.CreationDate -or $null -eq $row.WorkingSetSize) {
        throw 'Incomplete Win32_Process row'
    }
    [pscustomobject]@{
        pid = [long]$row.ProcessId
        parent_pid = [long]$row.ParentProcessId
        creation_time = [long]$row.CreationDate.ToUniversalTime().Ticks
        rss_bytes = [long]$row.WorkingSetSize
    }
}
ConvertTo-Json -InputObject @($payload) -Compress -Depth 3
"""
    try:
        completed = subprocess.run(
            [
                executable,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=_WINDOWS_SNAPSHOT_TIMEOUT_SECONDS,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProcessSnapshotError("Windows process snapshot command failed") from exc
    if completed.returncode != 0:
        raise ProcessSnapshotError("Windows process snapshot command failed")
    try:
        payload = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProcessSnapshotError("Windows process snapshot JSON is invalid") from exc
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ProcessSnapshotError("Windows process snapshot shape is invalid")
    if len(payload) > MAX_PROCESS_ROWS:
        raise ProcessSnapshotError("Windows process row limit exceeded")

    rows = []
    for item in payload:
        if not isinstance(item, dict):
            raise ProcessSnapshotError("Windows process snapshot row is invalid")
        try:
            pid = _parse_nonnegative_integer(item["pid"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProcessSnapshotError(
                "Windows process snapshot row is incomplete"
            ) from exc
        if pid == 0:
            # Win32_Process includes the PID 0 System Idle pseudo-process. It
            # is not an ordinary process and cannot be in an owned process tree.
            continue
        try:
            row = ProcessUsage(
                pid=pid,
                parent_pid=_parse_nonnegative_integer(item["parent_pid"]),
                creation_time=(
                    _parse_nonnegative_integer(item["creation_time"])
                    - _WINDOWS_FILETIME_EPOCH_TICKS
                ),
                rss_bytes=_parse_nonnegative_integer(item["rss_bytes"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProcessSnapshotError(
                "Windows process snapshot row is incomplete"
            ) from exc
        rows.append(row)
    return _validate_snapshot(rows)


def _validate_snapshot(records: Iterable[ProcessUsage]) -> tuple[ProcessUsage, ...]:
    rows = tuple(records)
    if not rows:
        raise ProcessSnapshotError("process snapshot is empty")
    if len(rows) > MAX_PROCESS_ROWS:
        raise ProcessSnapshotError("process row limit exceeded")
    pids: set[int] = set()
    for row in rows:
        if not isinstance(row, ProcessUsage):
            raise ProcessSnapshotError("process snapshot row has an invalid type")
        if (
            not _is_int(row.pid)
            or row.pid <= 0
            or not _is_int(row.parent_pid)
            or row.parent_pid < 0
            or not _is_int(row.creation_time)
            or row.creation_time <= 0
            or not _is_int(row.rss_bytes)
            or row.rss_bytes < 0
        ):
            raise ProcessSnapshotError("process snapshot row has invalid metrics")
        if row.pid in pids:
            raise ProcessSnapshotError("duplicate PID in process snapshot")
        pids.add(row.pid)
    return rows


def _parse_nonnegative_integer(value: object) -> int:
    if _is_int(value):
        parsed = value
    elif isinstance(value, str) and value.isascii() and value.isdecimal():
        parsed = int(value)
    else:
        raise ValueError("process metric must be an integer")
    if parsed < 0:
        raise ValueError("process metric must be nonnegative")
    return parsed


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)
