from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from tools.tier_d_nonrl.tier_d_process_tree import ProcessIdentity

_MAX_JOB_MEMBERS = 16_384
_JOB_QUERY_ATTEMPTS = 3
_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
_JOB_OBJECT_BASIC_PROCESS_ID_LIST = 3
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_CREATE_SUSPENDED = 0x00000004
_PROCESS_QUERY_INFORMATION = 0x00000400
_PROCESS_VM_READ = 0x00000010
_SYNCHRONIZE = 0x00100000


class ProcessContainmentError(RuntimeError):
    """A strict process-containment snapshot or cleanup could not be proven."""


class ProcessContainmentUnavailable(ProcessContainmentError):
    """The current operating system has no supported strict containment backend."""


@dataclass(frozen=True, slots=True)
class ProcessMember:
    identity: ProcessIdentity
    rss_bytes: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.identity, ProcessIdentity)
            or not isinstance(self.identity.pid, int)
            or isinstance(self.identity.pid, bool)
            or self.identity.pid <= 0
            or not isinstance(self.identity.creation_time, int)
            or isinstance(self.identity.creation_time, bool)
            or self.identity.creation_time <= 0
            or not isinstance(self.rss_bytes, int)
            or isinstance(self.rss_bytes, bool)
            or self.rss_bytes < 0
        ):
            raise ValueError("process member identity and RSS must be valid")


@dataclass(frozen=True, slots=True)
class ContainedProcessSample:
    """One stable Job Object membership sample and each member's working set."""

    root_identity: ProcessIdentity
    members: tuple[ProcessMember, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.root_identity, ProcessIdentity):
            raise ValueError("root_identity must be a process identity")
        if not isinstance(self.members, tuple) or any(
            not isinstance(member, ProcessMember) for member in self.members
        ):
            raise ValueError("members must be an immutable tuple of process members")
        pids = [member.identity.pid for member in self.members]
        if len(pids) != len(set(pids)):
            raise ValueError("contained process sample has duplicate PIDs")

    @property
    def rss_bytes(self) -> int:
        return sum(member.rss_bytes for member in self.members)

    @property
    def process_count(self) -> int:
        return len(self.members)


class ContainedProcess:
    """A launched process whose descendants remain inside a native Job Object."""

    def __init__(self, process: subprocess.Popen[Any], job: _WindowsJob) -> None:
        self.process = process
        self._job = job
        self._closed = False

    def sample(self) -> ContainedProcessSample:
        if self._closed:
            raise ProcessContainmentError("process containment is already closed")
        sample = self._job.sample()
        root_is_member = any(
            member.identity == sample.root_identity for member in sample.members
        )
        if self.process.poll() is None and not root_is_member:
            raise ProcessContainmentError(
                "running root process is absent from its containment Job Object"
            )
        return sample

    def terminate_and_wait(self, timeout: float = 10.0) -> int:
        if self._closed:
            return (
                self.process.returncode if self.process.returncode is not None else -1
            )
        deadline = time.monotonic() + timeout
        self._job.terminate_members(timeout=max(0.0, deadline - time.monotonic()))
        try:
            return_code = self.process.wait(
                timeout=max(0.0, deadline - time.monotonic())
            )
            self._job.wait_empty(deadline)
            return return_code
        except (OSError, subprocess.TimeoutExpired, ProcessContainmentError):
            raise

    def close(self, timeout: float = 10.0) -> None:
        if self._closed:
            return
        try:
            self.terminate_and_wait(timeout)
        except BaseException:
            try:
                self._job.close()
            except BaseException as close_error:
                raise ProcessContainmentError(
                    "Job Object could not be closed after cleanup failed"
                ) from close_error
            self._closed = True
            try:
                self.process.wait(timeout=timeout)
            except BaseException as wait_error:
                raise ProcessContainmentError(
                    "root process could not be reaped after Job Object close"
                ) from wait_error
            raise
        else:
            self._job.close()
            self._closed = True

    def __enter__(self) -> ContainedProcess:
        if self._closed:
            raise ProcessContainmentError("process containment is already closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def launch_contained_process(
    args: Sequence[str | os.PathLike[str]], **popen_kwargs: Any
) -> ContainedProcess:
    """Launch only when strict native process-family containment is available.

    Windows starts the root suspended, assigns it to a kill-on-close Job Object,
    and resumes it only after assignment. Linux strict containment is currently
    unsupported; parent-PID sampling is not used as a budget fallback.
    """

    if not args:
        raise ValueError("args must contain an executable")
    if os.name == "nt":
        return _launch_windows_contained_process(args, popen_kwargs)
    if sys.platform.startswith("linux"):
        raise ProcessContainmentUnavailable(
            "strict Linux launch requires a tested delegated cgroup-v2 backend"
        )
    raise ProcessContainmentUnavailable(
        f"strict process containment is unsupported on {sys.platform}"
    )


def _collect_stable_membership_snapshot(
    *,
    root_identity: ProcessIdentity,
    query_member_pids: Callable[[], Iterable[int]],
    read_member: Callable[[int], ProcessMember],
    verify_identity: Callable[[int], ProcessIdentity],
    attempts: int = _JOB_QUERY_ATTEMPTS,
) -> ContainedProcessSample:
    """Bracket RSS reads with membership and generation checks, failing closed."""

    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts <= 0:
        raise ValueError("attempts must be a positive integer")
    for _ in range(attempts):
        before = _validated_member_pids(query_member_pids())
        try:
            members = tuple(read_member(pid) for pid in before)
        except ProcessContainmentError as exc:
            if before != _validated_member_pids(query_member_pids()):
                continue
            raise ProcessContainmentError(
                "containment member snapshot is incomplete"
            ) from exc

        after_read = _validated_member_pids(query_member_pids())
        if before != after_read:
            continue
        if tuple(member.identity.pid for member in members) != before:
            raise ProcessContainmentError(
                "containment member reads do not match the complete membership list"
            )

        if any(
            verify_identity(pid) != member.identity
            for pid, member in zip(before, members, strict=True)
        ):
            continue
        after_verify = _validated_member_pids(query_member_pids())
        if after_read != after_verify:
            continue
        return ContainedProcessSample(root_identity, members)

    raise ProcessContainmentError(
        "containment membership or process generations did not stabilize"
    )


def _validated_member_pids(values: Iterable[int]) -> tuple[int, ...]:
    try:
        pids = tuple(sorted(values))
    except (TypeError, ValueError) as exc:
        raise ProcessContainmentError("containment membership list is invalid") from exc
    if (
        len(pids) > _MAX_JOB_MEMBERS
        or any(
            not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0
            for pid in pids
        )
        or len(pids) != len(set(pids))
    ):
        raise ProcessContainmentError("containment membership list is invalid")
    return pids


def _launch_windows_contained_process(
    args: Sequence[str | os.PathLike[str]], popen_kwargs: dict[str, Any]
) -> ContainedProcess:
    if "creationflags" in popen_kwargs and not isinstance(
        popen_kwargs["creationflags"], int
    ):
        raise TypeError("creationflags must be an integer")
    job = _WindowsJob()
    process: subprocess.Popen[Any] | None = None
    try:
        creationflags = int(popen_kwargs.pop("creationflags", 0)) | _CREATE_SUSPENDED
        process = subprocess.Popen(args, creationflags=creationflags, **popen_kwargs)
        job.assign(process._handle)
        root_identity = job.identity_from_handle(process._handle, process.pid)
        job.set_root_identity(root_identity)
        job.resume(process._handle)
        return ContainedProcess(process, job)
    except BaseException as launch_error:
        cleanup_errors: list[BaseException] = []
        try:
            if process is not None:
                try:
                    if job.assigned:
                        job.terminate_members()
                    else:
                        _terminate_unassigned_process(process, job)
                except BaseException as exc:
                    cleanup_errors.append(exc)

                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        if job.assigned:
                            job.terminate_members()
                        else:
                            job.terminate_process_handle(process._handle)
                        process.wait(timeout=10)
                    except BaseException as exc:
                        cleanup_errors.append(exc)
                except BaseException as exc:
                    cleanup_errors.append(exc)

                if job.assigned:
                    try:
                        job.wait_empty(time.monotonic() + 10)
                    except BaseException as exc:
                        cleanup_errors.append(exc)
        except BaseException as exc:
            cleanup_errors.append(exc)
        finally:
            try:
                job.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
            if process is not None:
                try:
                    process.wait(timeout=10)
                except BaseException as exc:
                    cleanup_errors.append(exc)
        if cleanup_errors:
            raise ProcessContainmentError(
                "contained process launch failed and child cleanup was not confirmed"
            ) from launch_error
        raise


def _terminate_unassigned_process(
    process: subprocess.Popen[Any], job: _WindowsJob
) -> None:
    """Terminate and reap the one suspended child if Job assignment failed."""

    try:
        process.kill()
    except OSError:
        # TerminateProcess on the original Popen handle is the last-resort path;
        # this never targets a PID discovered through a process-tree scan.
        job.terminate_process_handle(process._handle)


class _WindowsJob:
    def __init__(self) -> None:
        if os.name != "nt":
            raise ProcessContainmentUnavailable("Windows Job Objects are unavailable")
        try:
            self._api = _WindowsApi()
            self._handle = self._api.create_job()
        except Exception as exc:
            raise ProcessContainmentError(
                "cannot create a strict Windows Job Object"
            ) from exc
        try:
            self._api.set_kill_on_close(self._handle)
        except Exception as exc:
            try:
                self._api.close_handle(self._handle)
            except Exception:
                pass
            raise ProcessContainmentError(
                "cannot enable Job Object kill-on-close"
            ) from exc
        self._root_identity: ProcessIdentity | None = None
        self._assigned = False
        self._closed = False

    @property
    def assigned(self) -> bool:
        return self._assigned

    def assign(self, process_handle: int) -> None:
        self._api.assign_process(self._handle, process_handle)
        self._assigned = True

    def identity_from_handle(
        self, process_handle: int, expected_pid: int
    ) -> ProcessIdentity:
        return self._api.identity_from_handle(process_handle, expected_pid)

    def set_root_identity(self, identity: ProcessIdentity) -> None:
        self._root_identity = identity

    def resume(self, process_handle: int) -> None:
        self._api.resume_process(process_handle)

    def sample(self) -> ContainedProcessSample:
        if self._root_identity is None or self._closed:
            raise ProcessContainmentError("Job Object is not ready for sampling")
        return _collect_stable_membership_snapshot(
            root_identity=self._root_identity,
            query_member_pids=lambda: self._api.query_member_pids(self._handle),
            read_member=lambda pid: self._api.read_member(self._handle, pid),
            verify_identity=lambda pid: self._api.verify_member_identity(
                self._handle, pid
            ),
        )

    def terminate_members(self, timeout: float = 10.0) -> None:
        if self._closed:
            return
        deadline = time.monotonic() + timeout
        for _ in range(_JOB_QUERY_ATTEMPTS):
            before = self._api.query_member_pids(self._handle)
            handles: list[int] = []
            termination_requested = False
            try:
                for pid in before:
                    handle = self._api.open_member_handle(self._handle, pid)
                    handles.append(handle)
                    self._api.identity_from_open_handle(handle, pid)
                after = self._api.query_member_pids(self._handle)
                if before != after:
                    continue
                if not handles:
                    return

                self._api.terminate_job(self._handle)
                termination_requested = True
                for handle in handles:
                    self._api.wait_process_exit(handle, deadline)
                return
            except ProcessContainmentError:
                if not termination_requested and before != self._api.query_member_pids(
                    self._handle
                ):
                    continue
                raise
            finally:
                for handle in handles:
                    self._api.close_handle(handle)
        raise ProcessContainmentError(
            "Job Object membership did not stabilize before termination"
        )

    def terminate_process_handle(self, process_handle: int) -> None:
        self._api.terminate_process(process_handle)

    def wait_empty(self, deadline: float) -> None:
        while True:
            member_pids = self._api.query_member_pids(self._handle)
            active_processes = self._api.query_active_process_count(self._handle)
            if active_processes == 0 and not member_pids:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProcessContainmentError(
                    "Job Object processes did not drain before cleanup deadline"
                )
            time.sleep(min(0.01, remaining))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._api.close_handle(self._handle)


class _WindowsApi:
    def __init__(self) -> None:
        from ctypes import wintypes

        self._wintypes = wintypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self._ntdll = ctypes.WinDLL("ntdll")
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        w = self._wintypes
        self._kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, w.LPCWSTR)
        self._kernel32.CreateJobObjectW.restype = w.HANDLE
        self._kernel32.SetInformationJobObject.argtypes = (
            w.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            w.DWORD,
        )
        self._kernel32.SetInformationJobObject.restype = w.BOOL
        self._kernel32.AssignProcessToJobObject.argtypes = (w.HANDLE, w.HANDLE)
        self._kernel32.AssignProcessToJobObject.restype = w.BOOL
        self._kernel32.QueryInformationJobObject.argtypes = (
            w.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            w.DWORD,
            ctypes.POINTER(w.DWORD),
        )
        self._kernel32.QueryInformationJobObject.restype = w.BOOL
        self._kernel32.WaitForSingleObject.argtypes = (w.HANDLE, w.DWORD)
        self._kernel32.WaitForSingleObject.restype = w.DWORD
        self._kernel32.TerminateJobObject.argtypes = (w.HANDLE, w.UINT)
        self._kernel32.TerminateJobObject.restype = w.BOOL
        self._kernel32.TerminateProcess.argtypes = (w.HANDLE, w.UINT)
        self._kernel32.TerminateProcess.restype = w.BOOL
        self._kernel32.OpenProcess.argtypes = (w.DWORD, w.BOOL, w.DWORD)
        self._kernel32.OpenProcess.restype = w.HANDLE
        self._kernel32.GetProcessId.argtypes = (w.HANDLE,)
        self._kernel32.GetProcessId.restype = w.DWORD
        self._kernel32.GetProcessTimes.argtypes = (
            w.HANDLE,
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
        )
        self._kernel32.GetProcessTimes.restype = w.BOOL
        self._kernel32.IsProcessInJob.argtypes = (
            w.HANDLE,
            w.HANDLE,
            ctypes.POINTER(w.BOOL),
        )
        self._kernel32.IsProcessInJob.restype = w.BOOL
        self._kernel32.CloseHandle.argtypes = (w.HANDLE,)
        self._kernel32.CloseHandle.restype = w.BOOL
        self._psapi.GetProcessMemoryInfo.argtypes = (
            w.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            w.DWORD,
        )
        self._psapi.GetProcessMemoryInfo.restype = w.BOOL
        self._ntdll.NtResumeProcess.argtypes = (w.HANDLE,)
        self._ntdll.NtResumeProcess.restype = ctypes.c_long

    def create_job(self) -> int:
        handle = self._kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise _last_winerror("CreateJobObjectW failed")
        return int(getattr(handle, "value", handle))

    def set_kill_on_close(self, handle: int) -> None:
        limits = _ExtendedLimitInformation()
        limits.basic_limit_information.limit_flags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self._kernel32.SetInformationJobObject(
            self._handle(handle),
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            raise _last_winerror("cannot enable Job Object kill-on-close")

    def assign_process(self, job_handle: int, process_handle: int) -> None:
        if not self._kernel32.AssignProcessToJobObject(
            self._handle(job_handle), self._handle(process_handle)
        ):
            raise _last_winerror("cannot assign suspended process to Job Object")

    def resume_process(self, process_handle: int) -> None:
        status = self._ntdll.NtResumeProcess(self._handle(process_handle))
        if status < 0:
            raise ProcessContainmentError(
                f"cannot resume process after Job Object assignment (NTSTATUS {status})"
            )

    def identity_from_handle(self, handle: int, expected_pid: int) -> ProcessIdentity:
        process_handle = self._handle(handle)
        if self._kernel32.GetProcessId(process_handle) != expected_pid:
            raise ProcessContainmentError("process handle PID does not match Popen PID")
        identity = self._identity(process_handle)
        if identity.pid != expected_pid:
            raise ProcessContainmentError("process handle generation is invalid")
        return identity

    def query_member_pids(self, job_handle: int) -> tuple[int, ...]:
        offset = ctypes.sizeof(ctypes.c_uint32) * 2
        buffer_size = offset + _MAX_JOB_MEMBERS * ctypes.sizeof(ctypes.c_size_t)
        buffer = ctypes.create_string_buffer(buffer_size)
        returned = self._wintypes.DWORD()
        if not self._kernel32.QueryInformationJobObject(
            self._handle(job_handle),
            _JOB_OBJECT_BASIC_PROCESS_ID_LIST,
            ctypes.byref(buffer),
            buffer_size,
            ctypes.byref(returned),
        ):
            raise _last_winerror("cannot query complete Job Object membership")
        assigned, listed = (
            ctypes.cast(
                ctypes.byref(buffer), ctypes.POINTER(_BasicProcessIdListHeader)
            ).contents.number_of_assigned,
            ctypes.cast(
                ctypes.byref(buffer), ctypes.POINTER(_BasicProcessIdListHeader)
            ).contents.number_of_listed,
        )
        if assigned != listed or listed > _MAX_JOB_MEMBERS:
            raise ProcessContainmentError("Job Object membership list is incomplete")
        array_type = ctypes.c_size_t * listed
        pids = ctypes.cast(
            ctypes.addressof(buffer) + offset, ctypes.POINTER(array_type)
        ).contents
        return _validated_member_pids(int(pid) for pid in pids)

    def query_active_process_count(self, job_handle: int) -> int:
        info = _BasicAccountingInformation()
        returned = self._wintypes.DWORD()
        if not self._kernel32.QueryInformationJobObject(
            self._handle(job_handle),
            _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
            ctypes.byref(returned),
        ):
            raise _last_winerror("cannot query Job Object active process count")
        if returned.value < ctypes.sizeof(info):
            raise ProcessContainmentError(
                "Job Object active process count is incomplete"
            )
        return int(info.active_processes)

    def read_member(self, job_handle: int, pid: int) -> ProcessMember:
        process_handle = self._open_member(job_handle, pid)
        try:
            identity = self._identity(process_handle)
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if not self._psapi.GetProcessMemoryInfo(
                process_handle, ctypes.byref(counters), ctypes.sizeof(counters)
            ):
                raise _last_winerror("cannot read contained process working set")
            return ProcessMember(identity, int(counters.working_set_size))
        finally:
            self.close_handle(process_handle)

    def verify_member_identity(self, job_handle: int, pid: int) -> ProcessIdentity:
        process_handle = self._open_member(job_handle, pid)
        try:
            return self._identity(process_handle)
        finally:
            self.close_handle(process_handle)

    def _open_member(self, job_handle: int, pid: int) -> int:
        process_handle = self._kernel32.OpenProcess(
            _PROCESS_QUERY_INFORMATION | _PROCESS_VM_READ | _SYNCHRONIZE,
            False,
            pid,
        )
        if not process_handle:
            raise _last_winerror("cannot open a contained process for RSS sampling")
        handle = int(process_handle)
        if self._kernel32.GetProcessId(self._handle(handle)) != pid:
            self.close_handle(handle)
            raise ProcessContainmentError(
                "contained process PID changed during sampling"
            )
        in_job = self._wintypes.BOOL()
        if not self._kernel32.IsProcessInJob(
            self._handle(handle), self._handle(job_handle), ctypes.byref(in_job)
        ):
            self.close_handle(handle)
            raise _last_winerror("cannot verify contained process membership")
        if not in_job.value:
            self.close_handle(handle)
            raise ProcessContainmentError("process is absent from its Job Object")
        return handle

    def open_member_handle(self, job_handle: int, pid: int) -> int:
        return self._open_member(job_handle, pid)

    def identity_from_open_handle(
        self, process_handle: int, expected_pid: int
    ) -> ProcessIdentity:
        identity = self._identity(process_handle)
        if identity.pid != expected_pid:
            raise ProcessContainmentError(
                "contained process generation changed before termination"
            )
        return identity

    def wait_process_exit(self, process_handle: int, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProcessContainmentError(
                "contained processes did not exit before cleanup deadline"
            )
        timeout_ms = min(0xFFFFFFFE, max(0, int(remaining * 1000)))
        result = self._kernel32.WaitForSingleObject(
            self._handle(process_handle), timeout_ms
        )
        if result == 0:
            return
        if result == 0x00000102:
            raise ProcessContainmentError(
                "contained processes did not exit before cleanup deadline"
            )
        raise _last_winerror("cannot wait for contained process termination")

    def _identity(self, handle: int) -> ProcessIdentity:
        creation = _FileTime()
        exit_time = _FileTime()
        kernel_time = _FileTime()
        user_time = _FileTime()
        if not self._kernel32.GetProcessTimes(
            self._handle(handle),
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            raise _last_winerror("cannot read contained process generation")
        creation_time = (creation.high << 32) | creation.low
        pid = int(self._kernel32.GetProcessId(self._handle(handle)))
        if pid <= 0 or creation_time <= 0:
            raise ProcessContainmentError("contained process identity is invalid")
        return ProcessIdentity(pid, creation_time)

    def terminate_job(self, job_handle: int) -> None:
        if not self._kernel32.TerminateJobObject(self._handle(job_handle), 1):
            raise _last_winerror("cannot terminate contained process family")

    def terminate_process(self, process_handle: int) -> None:
        if not self._kernel32.TerminateProcess(self._handle(process_handle), 1):
            error = ctypes.get_last_error()
            # A process that exited concurrently is already in the desired state.
            if error != 5:  # ERROR_ACCESS_DENIED is the normal exited-process result.
                raise _last_winerror("cannot terminate unassigned suspended process")

    def close_handle(self, handle: int) -> None:
        if handle and not self._kernel32.CloseHandle(self._handle(handle)):
            raise _last_winerror("cannot close native process-containment handle")

    def _handle(self, value: int) -> Any:
        raw_value = getattr(value, "value", value)
        return self._wintypes.HANDLE(raw_value)


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]


class _BasicProcessIdListHeader(ctypes.Structure):
    _fields_ = [
        ("number_of_assigned", ctypes.c_uint32),
        ("number_of_listed", ctypes.c_uint32),
    ]


class _BasicAccountingInformation(ctypes.Structure):
    _fields_ = [
        ("total_user_time", ctypes.c_int64),
        ("total_kernel_time", ctypes.c_int64),
        ("this_period_total_user_time", ctypes.c_int64),
        ("this_period_total_kernel_time", ctypes.c_int64),
        ("total_page_faults", ctypes.c_uint32),
        ("total_processes", ctypes.c_uint32),
        ("active_processes", ctypes.c_uint32),
        ("total_terminated_processes", ctypes.c_uint32),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("read_operation_count", ctypes.c_uint64),
        ("write_operation_count", ctypes.c_uint64),
        ("other_operation_count", ctypes.c_uint64),
        ("read_transfer_count", ctypes.c_uint64),
        ("write_transfer_count", ctypes.c_uint64),
        ("other_transfer_count", ctypes.c_uint64),
    ]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("per_process_user_time_limit", ctypes.c_int64),
        ("per_job_user_time_limit", ctypes.c_int64),
        ("limit_flags", ctypes.c_uint32),
        ("minimum_working_set_size", ctypes.c_size_t),
        ("maximum_working_set_size", ctypes.c_size_t),
        ("active_process_limit", ctypes.c_uint32),
        ("affinity", ctypes.c_size_t),
        ("priority_class", ctypes.c_uint32),
        ("scheduling_class", ctypes.c_uint32),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("basic_limit_information", _BasicLimitInformation),
        ("io_info", _IoCounters),
        ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory_used", ctypes.c_size_t),
        ("peak_job_memory_used", ctypes.c_size_t),
    ]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("page_fault_count", ctypes.c_uint32),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


def _last_winerror(message: str) -> ProcessContainmentError:
    error = ctypes.get_last_error()
    return ProcessContainmentError(f"{message} (Windows error {error})")
