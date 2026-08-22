from __future__ import annotations

import trade_rl.studio.jobs as studio_jobs


def test_windows_pid_start_token_uses_process_creation_time_provider(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        studio_jobs,
        "_windows_process_start_token",
        lambda pid: f"creation:{pid}",
        raising=False,
    )

    assert studio_jobs._pid_start_token(4321, platform_name="nt") == ("creation:4321")


def test_windows_process_start_token_closes_handle_and_uses_creation_time(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class Function:
        def __init__(self, name, implementation):
            self.name = name
            self.implementation = implementation
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            calls.append(self.name)
            return self.implementation(*args)

    def open_process(_access, _inherit, pid):
        assert pid == 4321
        return 99

    def get_process_times(_handle, creation, _exit, _kernel, _user):
        creation._obj.low = 1
        creation._obj.high = 2
        return 1

    kernel32 = type(
        "Kernel32",
        (),
        {
            "OpenProcess": Function("open", open_process),
            "GetProcessTimes": Function("times", get_process_times),
            "CloseHandle": Function("close", lambda _handle: 1),
        },
    )()
    monkeypatch.setattr(
        studio_jobs.ctypes,
        "WinDLL",
        lambda *_args, **_kwargs: kernel32,
        raising=False,
    )

    token = studio_jobs._windows_process_start_token(4321)

    assert token == "0000000200000001"
    assert calls == ["open", "times", "close"]


def test_windows_process_start_token_closes_handle_when_times_fail(
    monkeypatch,
) -> None:
    closed: list[int] = []

    class Function:
        def __init__(self, implementation):
            self.implementation = implementation
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.implementation(*args)

    kernel32 = type(
        "Kernel32",
        (),
        {
            "OpenProcess": Function(lambda *_args: 99),
            "GetProcessTimes": Function(lambda *_args: 0),
            "CloseHandle": Function(lambda handle: closed.append(handle) or 1),
        },
    )()
    monkeypatch.setattr(
        studio_jobs.ctypes,
        "WinDLL",
        lambda *_args, **_kwargs: kernel32,
        raising=False,
    )

    assert studio_jobs._windows_process_start_token(4321) is None
    assert closed == [99]
