"""Bounded local process ownership for Windows venv redirector supervision."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

_WINDOWS_CREATE_NO_WINDOW: int = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass(frozen=True)
class ProcessUsage:
    pid: int
    parent_pid: int
    creation_ticks: int
    rss_bytes: int

    @property
    def identity(self) -> tuple[int, int]:
        return self.pid, self.creation_ticks


@dataclass(frozen=True)
class ProcessTreeSample:
    members: tuple[ProcessUsage, ...]
    root_pid: int

    @property
    def rss_bytes(self) -> int:
        return sum(member.rss_bytes for member in self.members)

    @property
    def root_rss_bytes(self) -> int:
        return sum(
            member.rss_bytes for member in self.members if member.pid == self.root_pid
        )


def windows_process_snapshot() -> tuple[ProcessUsage, ...]:
    # One snapshot supplies ancestry, birth identity and working set together.
    command = """@(Get-CimInstance Win32_Process | ForEach-Object {
      if ($_.CreationDate) { [pscustomobject]@{
        pid=[int]$_.ProcessId; parent_pid=[int]$_.ParentProcessId;
        creation_ticks=[long]$_.CreationDate.ToUniversalTime().Ticks;
        rss_bytes=[long]$_.WorkingSetSize
      } }
    }) | ConvertTo-Json -Compress"""
    output = subprocess.check_output(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        text=True,
        timeout=10,
        creationflags=_WINDOWS_CREATE_NO_WINDOW,
    )
    rows = json.loads(output)
    if type(rows) is not list or len(rows) > 4096:
        raise ValueError("process snapshot must contain at most4096 entries")
    return tuple(ProcessUsage(**row) for row in rows)


@dataclass
class ProcessTree:
    root_pid: int
    provider: Callable[[], tuple[ProcessUsage, ...]] = windows_process_snapshot
    root_identity: tuple[int, int] | None = None
    owned: dict[tuple[int, int], int] = field(default_factory=dict)
    last_sample: ProcessTreeSample | None = None

    def observe(self, snapshot: tuple[ProcessUsage, ...]) -> ProcessTreeSample:
        if len(snapshot) > 4096 or len({row.pid for row in snapshot}) != len(snapshot):
            raise ValueError("process snapshot is oversized or repeats process IDs")
        rows = {row.pid: row for row in snapshot}
        root = rows.get(self.root_pid)
        if self.root_identity is None and root is not None:
            self.root_identity = root.identity
            self.owned[root.identity] = 0
        selected = {
            row.pid: (row, self.owned[row.identity])
            for row in snapshot
            if row.identity in self.owned
        }
        while True:
            additions = [
                (row, selected[row.parent_pid][1] + 1)
                for row in snapshot
                if row.pid not in selected
                and row.parent_pid in selected
                and row.creation_ticks >= selected[row.parent_pid][0].creation_ticks
            ]
            if not additions:
                break
            for row, depth in additions:
                selected[row.pid] = row, depth
                self.owned[row.identity] = depth
            if len(self.owned) > 64:
                raise ValueError("owned process tree exceeds64 identities")
        members = tuple(
            row
            for row, _ in sorted(
                selected.values(), key=lambda pair: (pair[1], pair[0].pid)
            )
        )
        self.last_sample = ProcessTreeSample(members, self.root_pid)
        return self.last_sample

    def sample(self) -> ProcessTreeSample:
        return self.observe(self.provider())


def terminate_owned_tree(process: subprocess.Popen[Any], tree: ProcessTree) -> None:
    try:
        sample = tree.sample()
    except Exception:
        if tree.last_sample is None:
            raise
        sample = tree.last_sample
    # Kill deepest descendants before the redirector. A retained OS process
    # handle plus creation time prevents killing a reused PID from an old sample.
    # CIM birth clocks retain microseconds; <=10 ticks allows only1us precision loss.
    members = [asdict(member) for member in reversed(sample.members)]
    command = """$members=[Console]::In.ReadToEnd() | ConvertFrom-Json;
    foreach ($member in $members) {
      $ownedProcess=Get-Process -Id $member.pid -ErrorAction SilentlyContinue;
      if ($ownedProcess) {
        try {
          $null=$ownedProcess.Handle;
          $ticks=$ownedProcess.StartTime.ToUniversalTime().Ticks;
          if (-not $ownedProcess.HasExited -and [math]::Abs($ticks-[long]$member.creation_ticks) -le 10) {
            $ownedProcess.Kill(); $null=$ownedProcess.WaitForExit(1000)
          }
        } catch { if (-not $ownedProcess.HasExited) { throw } }
      }
    }; exit 0"""
    if members:
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            input=json.dumps(members),
            text=True,
            check=True,
            timeout=30,
            creationflags=_WINDOWS_CREATE_NO_WINDOW,
        )
    if process.poll() is None:
        process.terminate()
    process.wait(timeout=30)
    if tree.sample().members:
        raise RuntimeError("owned process descendants survived termination")
