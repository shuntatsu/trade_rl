from __future__ import annotations

import ctypes
import dataclasses
import os
import subprocess
import sys
import time
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
        "marker=pathlib.Path(sys.argv[1]); "
        "staged=marker.with_suffix('.tmp'); "
        "staged.write_text(str(os.getpid()),encoding='ascii'); "
        "staged.replace(marker); "
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
    # Closing a Windows Job Object triggers asynchronous kernel cleanup.
    # A bounded poll is evidence of termination, not an immediate race.
    deadline = time.monotonic() + 2.0
    still_running = _windows_generation_is_running(child_identity)
    while still_running and time.monotonic() < deadline:
        time.sleep(0.02)
        still_running = _windows_generation_is_running(child_identity)
    if still_running:
        _terminate_windows_generation(child_identity)
    assert not still_running


def test_windows_open_member_accepts_native_ctypes_handle() -> None:
    # A Windows OpenProcess HANDLE is c_void_p, not an int-convertible scalar.
    from ctypes import wintypes

    pid, raw_handle = 4321, 0x12345

    class Kernel:
        def OpenProcess(
            self, _flags: int, _inherit: bool, requested_pid: int
        ) -> object:
            assert requested_pid == pid
            return wintypes.HANDLE(raw_handle)

        def GetProcessId(self, handle: object) -> int:
            assert isinstance(handle, wintypes.HANDLE)
            assert handle.value == raw_handle
            return pid

        def IsProcessInJob(self, handle: object, job: object, output: object) -> int:
            assert isinstance(handle, wintypes.HANDLE)
            assert isinstance(job, wintypes.HANDLE)
            ctypes.cast(output, ctypes.POINTER(wintypes.BOOL)).contents.value = 1
            return 1

    api = object.__new__(containment._WindowsApi)
    api._wintypes = wintypes
    api._kernel32 = Kernel()
    assert api._open_member(999, pid) == raw_handle


def test_failed_windows_job_close_cannot_be_retried_as_success() -> None:
    class CloseErrorApi:
        calls = 0

        def close_handle(self, handle: int) -> None:
            assert handle == 999
            self.calls += 1
            if self.calls == 1:
                raise OSError("close status unknown")

    api = CloseErrorApi()
    job = object.__new__(containment._WindowsJob)
    job._handle = 999
    job._api = api
    job._closed = False
    job._close_failed = False
    with pytest.raises(OSError, match="close status unknown"):
        job.close()
    with pytest.raises(ProcessContainmentError, match="previously failed"):
        job.close()
    assert api.calls == 1
    assert not job._closed


def test_contained_process_close_after_failed_native_close_never_claims_success() -> (
    None
):
    class CloseErrorApi:
        calls = 0
        member_queries = 0

        def query_member_pids(self, handle: int) -> tuple[int, ...]:
            self.member_queries += 1
            return ()

        def query_active_process_count(self, handle: int) -> int:
            return 0

        def close_handle(self, handle: int) -> None:
            self.calls += 1
            if self.calls == 1:
                raise OSError("cannot prove job handle closed")

    class FinishedProcess:
        returncode = 0

        def wait(self, timeout: float) -> int:
            return 0

    job = object.__new__(containment._WindowsJob)
    api = CloseErrorApi()
    job._handle = 999
    job._api = api
    job._closed = False
    job._close_failed = False
    owned = containment.ContainedProcess(FinishedProcess(), job)  # type: ignore[arg-type]
    with pytest.raises(OSError, match="cannot prove"):
        owned.close()
    queries_after_failure = api.member_queries
    with pytest.raises(ProcessContainmentError, match="previously failed"):
        owned.close()
    for operation in (owned.sample, owned.terminate_and_wait, owned.__enter__):
        with pytest.raises(ProcessContainmentError, match="previously failed"):
            operation()
    assert api.member_queries == queries_after_failure
    assert not owned._closed
    assert api.calls == 1


def test_failed_windows_job_close_refuses_all_followup_native_job_uses() -> None:
    class UnknownCloseApi:
        def __init__(self) -> None:
            self.unsafe_native_calls: list[str] = []

        def close_handle(self, handle: int) -> None:
            assert handle == 999
            raise OSError("Job handle status unknown")

        def __getattr__(self, name: str):
            def untrusted_call(*args: object, **kwargs: object) -> None:
                self.unsafe_native_calls.append(name)
                raise AssertionError("native operation attempted after failed close")

            return untrusted_call

    api = UnknownCloseApi()
    job = object.__new__(containment._WindowsJob)
    job._handle = 999
    job._api = api
    job._root_identity = ProcessIdentity(pid=100, creation_time=1000)
    job._assigned = True
    job._closed = False
    job._close_failed = False

    with pytest.raises(OSError, match="status unknown"):
        job.close()

    operations = (
        lambda: job.assign(888),
        lambda: job.identity_from_handle(888, 100),
        lambda: job.resume(888),
        job.sample,
        job.terminate_members,
        lambda: job.wait_empty(time.monotonic() + 1),
        lambda: job.terminate_process_handle(888),
    )
    for operation in operations:
        with pytest.raises(ProcessContainmentError, match="previously failed"):
            operation()
    assert api.unsafe_native_calls == []


def test_successfully_closed_windows_job_is_idempotent() -> None:
    class CloseOkApi:
        calls = 0

        def close_handle(self, handle: int) -> None:
            assert handle == 999
            self.calls += 1

    api = CloseOkApi()
    job = object.__new__(containment._WindowsJob)
    job._handle = 999
    job._api = api
    job._closed = False
    job._close_failed = False
    job.close()
    job.close()
    assert api.calls == 1
    assert job._closed
    assert not job._close_failed


def test_contained_process_fallback_close_does_not_rewrite_failed_cleanup_as_success() -> (
    None
):
    class TerminationFailureJob:
        termination_calls = 0
        close_calls = 0

        def terminate_members(self, timeout: float) -> None:
            self.termination_calls += 1
            raise ProcessContainmentError("job termination cannot be confirmed")

        def close(self) -> None:
            self.close_calls += 1

    class ReapedProcess:
        returncode = 17
        wait_calls = 0

        def wait(self, timeout: float) -> int:
            self.wait_calls += 1
            return 17

    job, process = TerminationFailureJob(), ReapedProcess()
    owned = containment.ContainedProcess(process, job)  # type: ignore[arg-type]
    with pytest.raises(
        ProcessContainmentError, match="job termination cannot be confirmed"
    ):
        owned.close()
    with pytest.raises(ProcessContainmentError, match="cleanup previously failed"):
        owned.close()
    with pytest.raises(ProcessContainmentError, match="cleanup previously failed"):
        owned.terminate_and_wait()
    with pytest.raises(ProcessContainmentError, match="cleanup previously failed"):
        owned.sample()
    with pytest.raises(ProcessContainmentError, match="cleanup previously failed"):
        owned.__enter__()
    assert job.termination_calls == job.close_calls == process.wait_calls == 1


def test_contained_process_verified_close_is_idempotent() -> None:
    class VerifiedJob:
        termination_calls = 0
        close_calls = 0
        empty_calls = 0

        def terminate_members(self, timeout: float) -> None:
            self.termination_calls += 1

        def wait_empty(self, deadline: float) -> None:
            self.empty_calls += 1

        def close(self) -> None:
            self.close_calls += 1

    class ReapedProcess:
        returncode = 0
        wait_calls = 0

        def wait(self, timeout: float) -> int:
            self.wait_calls += 1
            return 0

    job, process = VerifiedJob(), ReapedProcess()
    owned = containment.ContainedProcess(process, job)  # type: ignore[arg-type]
    owned.close()
    owned.close()
    assert owned.terminate_and_wait() == 0
    assert (job.termination_calls, job.empty_calls, job.close_calls) == (1, 1, 1)
    assert process.wait_calls == 1
    assert owned._closed and not owned._cleanup_failed


@pytest.mark.parametrize("failure_phase", ("terminate", "root_wait", "empty"))
@pytest.mark.parametrize("exception_type", (ProcessContainmentError, KeyboardInterrupt))
def test_direct_stop_failure_never_becomes_verified_cleanup(
    failure_phase: str, exception_type: type[BaseException]
) -> None:
    calls: list[str] = []
    failed = False

    def record(phase: str) -> None:
        nonlocal failed
        calls.append(phase)
        if phase == failure_phase and not failed:
            failed = True
            raise exception_type("first stop verification failed")

    class Job:
        def terminate_members(self, timeout: float) -> None:
            record("terminate")

        def wait_empty(self, deadline: float) -> None:
            record("empty")

        def close(self) -> None:
            record("close")

        def sample(self) -> NoReturn:
            raise AssertionError("must not sample after failed stop")

    class Process:
        returncode = 17

        def wait(self, timeout: float) -> int:
            record("root_wait")
            return 17

    owned = containment.ContainedProcess(Process(), Job())  # type: ignore[arg-type]
    with pytest.raises(exception_type, match="first stop verification failed"):
        owned.terminate_and_wait()

    # Release the kill-on-close Job and reap the root as best effort, but
    # preserve the failed verification as the permanent disposition.
    expected = {
        "terminate": ["terminate", "close", "root_wait"],
        "root_wait": ["terminate", "root_wait", "close", "root_wait"],
        "empty": ["terminate", "root_wait", "empty", "close", "root_wait"],
    }
    for operation in (
        owned.terminate_and_wait,
        owned.close,
        owned.sample,
        owned.__enter__,
    ):
        with pytest.raises(ProcessContainmentError, match="cleanup previously failed"):
            operation()
    assert calls == expected[failure_phase]


@pytest.mark.parametrize("fallback_failure", ("close", "reap"))
def test_direct_stop_preserves_uncertain_fallback_disposition(
    fallback_failure: str,
) -> None:
    calls: list[str] = []

    class Job:
        def terminate_members(self, timeout: float) -> NoReturn:
            calls.append("terminate")
            raise ProcessContainmentError("stop not confirmed")

        def close(self) -> None:
            calls.append("close")
            if fallback_failure == "close":
                raise OSError("native close uncertain")

    class Process:
        returncode = None

        def wait(self, timeout: float) -> NoReturn:
            calls.append("wait")
            raise subprocess.TimeoutExpired("synthetic", timeout)

    owned = containment.ContainedProcess(Process(), Job())  # type: ignore[arg-type]
    with pytest.raises(
        ProcessContainmentError, match="could not be closed|could not be reaped"
    ):
        owned.terminate_and_wait()
    expected = ["terminate", "close"]
    if fallback_failure == "reap":
        expected.append("wait")
    assert calls == expected
    for operation in (owned.terminate_and_wait, owned.close, owned.sample):
        with pytest.raises(ProcessContainmentError, match="previously failed"):
            operation()
    assert calls == expected


def test_direct_successful_stop_can_close_normally() -> None:
    calls: list[str] = []

    class Job:
        def terminate_members(self, timeout: float) -> None:
            calls.append("terminate")

        def wait_empty(self, deadline: float) -> None:
            calls.append("empty")

        def close(self) -> None:
            calls.append("close")

    class Process:
        returncode = 0

        def wait(self, timeout: float) -> int:
            calls.append("wait")
            return 0

    owned = containment.ContainedProcess(Process(), Job())  # type: ignore[arg-type]
    assert owned.terminate_and_wait() == 0
    assert calls == ["terminate", "wait", "empty"]
    assert not owned._closed
    owned.close()
    assert calls == [
        "terminate",
        "wait",
        "empty",
        "terminate",
        "wait",
        "empty",
        "close",
    ]
    owned.close()
    assert owned.terminate_and_wait() == 0
    assert calls.count("close") == 1
    assert not owned._cleanup_failed


@pytest.mark.parametrize("first_phase", ("sample", "terminate", "close"))
@pytest.mark.parametrize(
    "second_operation", ("sample", "terminate_and_wait", "close", "__enter__")
)
@pytest.mark.parametrize("first_fails", (True, False))
def test_concurrent_containment_operations_cannot_share_native_handle(
    first_phase: str, second_operation: str, first_fails: bool
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    entered = Event()
    release = Event()
    calls: list[str] = []
    root = ProcessIdentity(100, 1000)
    native_failure = OSError if first_phase == "close" else ProcessContainmentError

    def phase(name: str) -> None:
        calls.append(name)
        if name == first_phase and not entered.is_set():
            entered.set()
            assert release.wait(20), "test did not release active native operation"
            if first_fails:
                raise native_failure("native operation failed")

    class Job:
        def sample(self) -> ContainedProcessSample:
            phase("sample")
            return ContainedProcessSample(root, ())

        def terminate_members(self, timeout: float) -> None:
            phase("terminate")

        def wait_empty(self, deadline: float) -> None:
            phase("empty")

        def close(self) -> None:
            phase("close")

    class Process:
        returncode = 0

        def wait(self, timeout: float) -> int:
            phase("wait")
            return 0

        def poll(self) -> int:
            return 0

    owned = containment.ContainedProcess(Process(), Job())  # type: ignore[arg-type]
    first_operation = {
        "sample": owned.sample,
        "terminate": owned.terminate_and_wait,
        "close": owned.close,
    }[first_phase]
    prefix = {
        "sample": ["sample"],
        "terminate": ["terminate"],
        "close": ["terminate", "wait", "empty", "close"],
    }[first_phase]
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(first_operation)
        try:
            assert entered.wait(5)
            second = pool.submit(getattr(owned, second_operation))
            # No blocking on another native call, and no native access while
            # the active call can change handle ownership or verification.
            with pytest.raises(
                ProcessContainmentError, match="operation is in progress"
            ):
                second.result(timeout=5)
            assert calls == prefix
        finally:
            release.set()
        if first_fails:
            with pytest.raises(native_failure, match="native operation failed"):
                first.result(timeout=5)
        else:
            first.result(timeout=5)

    if first_fails and first_phase != "sample":
        for retry in (
            owned.sample,
            owned.terminate_and_wait,
            owned.close,
            owned.__enter__,
        ):
            with pytest.raises(ProcessContainmentError, match="previously failed"):
                retry()
        expected = (
            ["terminate", "close", "wait"] if first_phase == "terminate" else prefix
        )
        assert calls == expected
    else:
        # A refused concurrent call or failed sample is not itself a failed
        # native cleanup. A verified sequential close is still idempotent.
        owned.close()
        owned.close()
        assert owned.terminate_and_wait() == 0
        for retry in (owned.sample, owned.__enter__):
            with pytest.raises(ProcessContainmentError, match="already closed"):
                retry()
        cleanup = ["terminate", "wait", "empty", "close"]
        expected = {
            "sample": ["sample", *cleanup],
            "terminate": ["terminate", "wait", "empty", *cleanup],
            "close": cleanup,
        }[first_phase]
        assert calls == expected
