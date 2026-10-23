"""Tests for the Windows-only CI process-observer readiness preflight."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "tools/windows_process_ci_readiness.py"


def readiness_module():
    assert HELPER_PATH.is_file(), "the CI readiness helper has not been added"
    spec = importlib.util.spec_from_file_location(
        "windows_process_ci_readiness", HELPER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timeout_is_retried_once_before_any_owned_child_and_result_is_discarded():
    module = readiness_module()
    calls = []
    sleeps = []
    messages = []
    snapshot_result = object()

    def snapshot():
        calls.append(True)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired("Get-CimInstance", 10)
        return snapshot_result

    result = module.ensure_windows_cim_ready(
        snapshot=snapshot, sleep=sleeps.append, log=messages.append
    )

    assert result is None
    assert calls == [True, True]
    assert sleeps == [1.0]
    assert any("retry" in message.lower() for message in messages)
    assert any("readiness" in message.lower() for message in messages)


@pytest.mark.parametrize(
    "failure",
    [
        OSError("CIM unavailable"),
        ValueError("invalid CIM response"),
        subprocess.CalledProcessError(1, "powershell.exe"),
    ],
)
def test_non_timeout_failure_is_not_retried(failure):
    module = readiness_module()
    calls = []
    sleeps = []

    def snapshot():
        calls.append(True)
        raise failure

    with pytest.raises(type(failure), match=str(failure)):
        module.ensure_windows_cim_ready(snapshot=snapshot, sleep=sleeps.append)

    assert calls == [True]
    assert sleeps == []


def test_second_timeout_fails_closed_after_one_retry():
    module = readiness_module()
    calls = []
    sleeps = []
    messages = []

    def snapshot():
        calls.append(True)
        raise subprocess.TimeoutExpired("Get-CimInstance", 10)

    with pytest.raises(subprocess.TimeoutExpired):
        module.ensure_windows_cim_ready(
            snapshot=snapshot, sleep=sleeps.append, log=messages.append
        )

    assert calls == [True, True]
    assert sleeps == [1.0]
    assert any("failed" in message.lower() for message in messages)


def test_readiness_cli_refuses_non_windows_without_querying(monkeypatch):
    module = readiness_module()
    monkeypatch.setattr(module.sys, "platform", "linux")

    def unexpected_snapshot():
        raise AssertionError("non-Windows invocation must not query CIM")

    monkeypatch.setattr(module, "windows_process_snapshot", unexpected_snapshot)

    with pytest.raises(SystemExit, match="Windows"):
        module.main()
