from __future__ import annotations

import ctypes
import dataclasses
import os
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest

import tools.tier_d_nonrl.tier_d_process_containment as containment
from tools.tier_d_nonrl.tier_d_process_containment import (
    ContainedProcessSample,
    ProcessContainmentError,
    ProcessContainmentUnavailable,
    ProcessIdentity,
    ProcessMember,
    launch_contained_process,
)


def _member(pid: int, creation_time: int, rss_bytes: int) -> ProcessMember:
    return ProcessMember(
        identity=ProcessIdentity(pid=pid, creation_time=creation_time),
        rss_bytes=rss_bytes,
    )


def test_containment_sample_is_immutable_and_sums_all_members() -> None:
    sample = ContainedProcessSample(
        root_identity=ProcessIdentity(pid=100, creation_time=1000),
        members=(
            _member(100, 1000, 4_000_000),
            _member(101, 1001, 64_000_000),
        ),
    )

    assert sample.rss_bytes == 68_000_000
    assert sample.process_count == 2

    with pytest.raises(dataclasses.FrozenInstanceError):
        sample.members = ()  # type: ignore[misc]


def test_live_root_missing_from_nonempty_job_membership_fails_closed() -> None:
    root = ProcessIdentity(pid=100, creation_time=1000)
    descendant_only = ContainedProcessSample(
        root_identity=root,
        members=(_member(101, 1001, 64_000_000),),
    )

    class LiveProcess:
        returncode = None

        def poll(self) -> None:
            return None

    class DescendantOnlyJob:
        def sample(self) -> ContainedProcessSample:
            return descendant_only

    owned = containment.ContainedProcess(LiveProcess(), DescendantOnlyJob())  # type: ignore[arg-type]

    with pytest.raises(ProcessContainmentError, match="root process is absent"):
        owned.sample()


def test_stable_membership_retries_when_job_members_change() -> None:
    membership_snapshots = iter(
        (
            (100,),
            (100, 101),
            (100, 101),
            (100, 101),
            (100, 101),
        )
    )
    identities = {
        100: ProcessIdentity(100, 1000),
        101: ProcessIdentity(101, 1001),
    }

    sample = containment._collect_stable_membership_snapshot(
        root_identity=identities[100],
        query_member_pids=lambda: next(membership_snapshots),
        read_member=lambda pid: _member(pid, identities[pid].creation_time, pid),
        verify_identity=lambda pid: identities[pid],
        attempts=2,
    )

    assert sample.members == (_member(100, 1000, 100), _member(101, 1001, 101))


def test_stable_membership_fails_closed_on_unreadable_member() -> None:
    membership_snapshots = iter(((100, 101), (100, 101)))

    with pytest.raises(ProcessContainmentError, match="incomplete"):
        containment._collect_stable_membership_snapshot(
            root_identity=ProcessIdentity(100, 1000),
            query_member_pids=lambda: next(membership_snapshots),
            read_member=lambda pid: (
                _member(100, 1000, 100) if pid == 100 else _raise_unreadable()
            ),
            verify_identity=lambda pid: ProcessIdentity(pid, pid + 900),
            attempts=1,
        )


def _raise_unreadable() -> NoReturn:
    raise ProcessContainmentError("incomplete member read")


def _windows_generation_is_running(identity: ProcessIdentity) -> bool:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(containment._FileTime),
        ctypes.POINTER(containment._FileTime),
        ctypes.POINTER(containment._FileTime),
        ctypes.POINTER(containment._FileTime),
    )
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(0x00100000 | 0x00001000, False, identity.pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: this PID no longer exists.
            return False
        raise OSError(error, "cannot open process to verify Job Object cleanup")
    try:
        created = containment._FileTime()
        exited = containment._FileTime()
        kernel = containment._FileTime()
        user = containment._FileTime()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        actual = ProcessIdentity(identity.pid, (created.high << 32) | created.low)
        if actual != identity:
            return False
        wait_status = kernel32.WaitForSingleObject(handle, 0)
        if wait_status == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())
        return wait_status == 258  # WAIT_TIMEOUT means this generation is alive.
    finally:
        if not kernel32.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def _terminate_windows_generation(identity: ProcessIdentity) -> None:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(containment._FileTime),
        ctypes.POINTER(containment._FileTime),
        ctypes.POINTER(containment._FileTime),
        ctypes.POINTER(containment._FileTime),
    )
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(0x00100000 | 0x00001000 | 0x0001, False, identity.pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:
            return
        raise OSError(error, "cannot open test process for exact-generation cleanup")
    try:
        created = containment._FileTime()
        exited = containment._FileTime()
        kernel = containment._FileTime()
        user = containment._FileTime()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        actual = ProcessIdentity(identity.pid, (created.high << 32) | created.low)
        if actual != identity:
            return
        if not kernel32.TerminateProcess(handle, 1) and ctypes.get_last_error() != 5:
            raise ctypes.WinError(ctypes.get_last_error())
        if kernel32.WaitForSingleObject(handle, 5000) != 0:
            raise TimeoutError("test child did not exit after exact-generation cleanup")
    finally:
        if not kernel32.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def test_linux_strict_launch_fails_before_starting_any_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(containment.os, "name", "posix")
    monkeypatch.setattr(containment.sys, "platform", "linux")

    def unexpected_spawn(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        raise AssertionError("strict Linux launch must fail before spawning")

    monkeypatch.setattr(containment.subprocess, "Popen", unexpected_spawn)

    with pytest.raises(ProcessContainmentUnavailable, match="delegated cgroup"):
        launch_contained_process(["not-started"])


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object integration test")
def test_assignment_failure_terminates_and_reaps_suspended_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_popen = subprocess.Popen
    spawned: list[subprocess.Popen[bytes]] = []

    class AssignmentFailingJob:
        assigned = False
        closed = False

        def assign(self, _process_handle: int) -> None:
            raise ProcessContainmentError("injected Job Object assignment failure")

        def close(self) -> None:
            self.closed = True

    job = AssignmentFailingJob()

    def capture_suspended_process(
        args: list[str], *, creationflags: int, **kwargs: object
    ) -> subprocess.Popen[bytes]:
        assert creationflags & containment._CREATE_SUSPENDED
        process = real_popen(args, creationflags=creationflags, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(containment, "_WindowsJob", lambda: job)
    monkeypatch.setattr(containment.subprocess, "Popen", capture_suspended_process)

    with pytest.raises(ProcessContainmentError, match="assignment failure"):
        containment._launch_windows_contained_process(
            [sys.executable, "-I", "-S", "-c", "import time; time.sleep(30)"],
            {
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            },
        )

    assert len(spawned) == 1
    assert spawned[0].poll() is not None
    assert job.closed


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object integration test")
def test_job_contains_live_child_after_root_exits_and_cleanup_kills_it(
    tmp_path: Path,
) -> None:
    marker = os.fspath(tmp_path / "child-ready")
    child_code = (
        "import os,pathlib,sys,time; "
        "data=bytearray(64*1024*1024); "
        "[data.__setitem__(i,1) for i in range(0,len(data),4096)]; "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()),encoding='ascii'); "
        "time.sleep(30)"
    )
    root_code = """
import os,subprocess,sys,time
subprocess.Popen([sys.executable,'-I','-S','-c',sys.argv[1],sys.argv[2]],
    stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
deadline=time.monotonic()+10
while not os.path.exists(sys.argv[2]) and time.monotonic()<deadline:
    time.sleep(.01)
if not os.path.exists(sys.argv[2]):
    raise SystemExit('contained child did not become ready')
"""

    with launch_contained_process(
        [sys.executable, "-I", "-S", "-c", root_code, child_code, marker],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) as owned:
        assert owned.process.wait(timeout=10) == 0

        sample = owned.sample()

        assert sample.process_count >= 1
        assert all(member.identity != sample.root_identity for member in sample.members)
        assert sample.rss_bytes >= 32 * 1024 * 1024
        child_pid = int((tmp_path / "child-ready").read_text(encoding="ascii"))
        child_identity = next(
            member.identity
            for member in sample.members
            if member.identity.pid == child_pid
        )

    assert owned.process.returncode == 0
    still_running = _windows_generation_is_running(child_identity)
    if still_running:
        _terminate_windows_generation(child_identity)
    assert not still_running
