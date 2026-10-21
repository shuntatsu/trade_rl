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
    """One sample separates RSS attribution from verified kill ownership."""

    members: tuple[ProcessUsage, ...]
    root_pid: int
    owned_members: tuple[ProcessUsage, ...]
    ambiguous_members: tuple[ProcessUsage, ...]

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
    rss_only: dict[tuple[int, int], int] = field(default_factory=dict)
    last_sample: ProcessTreeSample | None = None

    def observe(self, snapshot: tuple[ProcessUsage, ...]) -> ProcessTreeSample:
        if len(snapshot) > 4096 or len({row.pid for row in snapshot}) != len(snapshot):
            raise ValueError("process snapshot is oversized or repeats process IDs")
        rows = {row.pid: row for row in snapshot}
        root = rows.get(self.root_pid)
        if self.root_identity is None and root is not None:
            self.root_identity = root.identity
            self.owned[root.identity] = 0
        selected_owned = {
            row.pid: (row, self.owned[row.identity])
            for row in snapshot
            if row.identity in self.owned
        }
        selected_ambiguous = {
            row.pid: (row, self.rss_only[row.identity])
            for row in snapshot
            if row.identity in self.rss_only and row.pid not in selected_owned
        }
        historical_by_pid: dict[int, list[tuple[int, int, bool]]] = {}
        for identities, verified in ((self.owned, True), (self.rss_only, False)):
            for (pid, creation_ticks), depth in identities.items():
                historical_by_pid.setdefault(pid, []).append(
                    (creation_ticks, depth, verified)
                )

        while True:
            owned_additions: list[tuple[ProcessUsage, int]] = []
            ambiguous_additions: list[tuple[ProcessUsage, int]] = []
            for row in snapshot:
                if row.pid in selected_owned:
                    continue

                verified_parent = selected_owned.get(row.parent_pid)
                if verified_parent is not None:
                    if row.creation_ticks >= verified_parent[0].creation_ticks:
                        owned_additions.append((row, verified_parent[1] + 1))
                    continue

                # Existing RSS-only rows are already selected. Reconsider
                # them only through the verified-parent promotion above.
                if row.pid in selected_ambiguous:
                    continue

                ambiguous_parent = selected_ambiguous.get(row.parent_pid)
                if ambiguous_parent is not None:
                    if row.creation_ticks >= ambiguous_parent[0].creation_ticks:
                        ambiguous_additions.append((row, ambiguous_parent[1] + 1))
                    continue

                # A visible parent born no later than this child may own it.
                # If the PID's visible process was born later, it is a reused
                # parent ID; a historical owner can only attribute the child
                # ambiguously, never grant kill authority.
                visible_parent = rows.get(row.parent_pid)
                if (
                    visible_parent is not None
                    and visible_parent.creation_ticks <= row.creation_ticks
                ):
                    continue
                prior_parents = [
                    (created, depth, verified)
                    for created, depth, verified in historical_by_pid.get(
                        row.parent_pid, []
                    )
                    if created <= row.creation_ticks
                ]
                if prior_parents:
                    _, parent_depth, _ = max(prior_parents, key=lambda item: item[0])
                    ambiguous_additions.append((row, parent_depth + 1))

            if not owned_additions and not ambiguous_additions:
                break
            for row, depth in owned_additions:
                selected_owned[row.pid] = row, depth
                selected_ambiguous.pop(row.pid, None)
                self.owned[row.identity] = depth
                self.rss_only.pop(row.identity, None)
            for row, depth in ambiguous_additions:
                if row.pid in selected_owned or row.pid in selected_ambiguous:
                    continue
                selected_ambiguous[row.pid] = row, depth
                self.rss_only[row.identity] = depth
            if len(self.owned) + len(self.rss_only) > 64:
                raise ValueError("observed process tree exceeds64 identities")

            for identities, verified in ((self.owned, True), (self.rss_only, False)):
                for (pid, creation_ticks), depth in identities.items():
                    known = historical_by_pid.setdefault(pid, [])
                    entry = (creation_ticks, depth, verified)
                    if entry not in known:
                        known.append(entry)

        owned_members = tuple(
            row
            for row, _ in sorted(
                selected_owned.values(), key=lambda pair: (pair[1], pair[0].pid)
            )
        )
        ambiguous_members = tuple(
            row
            for row, _ in sorted(
                selected_ambiguous.values(), key=lambda pair: (pair[1], pair[0].pid)
            )
        )
        members = tuple(
            row
            for row, _ in sorted(
                (*selected_owned.values(), *selected_ambiguous.values()),
                key=lambda pair: (pair[1], pair[0].pid),
            )
        )
        self.last_sample = ProcessTreeSample(
            members, self.root_pid, owned_members, ambiguous_members
        )
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
    # Only identities seen in a verified current parent chain are killable.
    # Historical PID correlations contribute to RSS but never to cleanup.
    # Kill deepest descendants before the redirector; the fresh OS handle and
    # creation time check also refuse a PID that has since been reused.
    # CIM birth clocks retain microseconds; <=10 ticks allows only1us precision loss.
    members = [asdict(member) for member in reversed(sample.owned_members)]
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
    final_sample = tree.sample()
    if final_sample.owned_members:
        raise RuntimeError("verified owned process descendants survived termination")
    if final_sample.ambiguous_members:
        raise RuntimeError(
            "ambiguous RSS-attributed processes remain; refusing cleanup verification"
        )
