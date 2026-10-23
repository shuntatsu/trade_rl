"""Exit races for the repository-only Linux root-process RSS sampler."""

from __future__ import annotations

import errno
import importlib
from pathlib import Path

import pytest


class Child:
    pid = 314159

    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.poll_calls = 0

    def poll(self) -> int | None:
        self.poll_calls += 1
        return self.returncode


def proc_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[object, Path]:
    module = importlib.import_module("tools.tier_d_nonrl.tier_d_supervisor")
    repo_root = Path(__file__).resolve().parents[3]
    assert Path(module.__file__).resolve() == (
        repo_root / "tools/tier_d_nonrl/tier_d_supervisor.py"
    )
    real_path = module.Path
    proc_root = tmp_path / "proc"
    monkeypatch.setattr(
        module,
        "Path",
        lambda value: proc_root if str(value) == "/proc" else real_path(value),
    )
    status_path = proc_root / str(Child.pid) / "status"
    status_path.parent.mkdir(parents=True)
    return module, status_path


def raise_from_status(
    monkeypatch: pytest.MonkeyPatch, status_path: Path, error: type[OSError]
) -> None:
    read_text = Path.read_text

    def failing_read(path: Path, *args: object, **kwargs: object) -> str:
        if path == status_path:
            if error is ProcessLookupError:
                raise ProcessLookupError(errno.ESRCH, "process vanished")
            raise error("synthetic proc status access failure")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", failing_read)


def test_process_rss_reads_vm_rss_without_polling_a_live_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, status_path = proc_status(tmp_path, monkeypatch)
    status_path.write_text("Name:\tworker\nVmRSS:\t42 kB\n", encoding="ascii")
    child = Child()

    assert module.process_rss(child) == 42 * 1024
    assert child.poll_calls == 0


@pytest.mark.parametrize("failure", ["missing_file", "process_lookup", "missing_vmrss"])
def test_process_rss_returns_zero_for_expected_proc_exit_races_only_after_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    module, status_path = proc_status(tmp_path, monkeypatch)
    if failure == "process_lookup":
        status_path.write_text("Name:\tworker\nVmRSS:\t42 kB\n", encoding="ascii")
        raise_from_status(monkeypatch, status_path, ProcessLookupError)
    elif failure == "missing_vmrss":
        # A controlled incomplete proc status exercises the missing-field path.
        status_path.write_text("Name:\tworker\n", encoding="ascii")
    child = Child(returncode=0)

    assert module.process_rss(child) == 0
    assert child.poll_calls == 1


@pytest.mark.parametrize(
    ("failure", "status_text"),
    [
        ("missing_file", None),
        ("process_lookup", "Name:\tworker\nVmRSS:\t42 kB\n"),
        ("missing_vmrss", "Name:\tworker\n"),
        ("malformed_vmrss", "Name:\tworker\nVmRSS:\tbad kB\n"),
        ("permission", "Name:\tworker\nVmRSS:\t42 kB\n"),
    ],
)
def test_process_rss_preserves_telemetry_errors_while_child_is_live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    status_text: str | None,
) -> None:
    module, status_path = proc_status(tmp_path, monkeypatch)
    if status_text is not None:
        status_path.write_text(status_text, encoding="ascii")
    if failure in {"process_lookup", "permission"}:
        error = ProcessLookupError if failure == "process_lookup" else PermissionError
        raise_from_status(monkeypatch, status_path, error)
    expected_error = {
        "missing_file": FileNotFoundError,
        "process_lookup": ProcessLookupError,
        "missing_vmrss": StopIteration,
        "malformed_vmrss": ValueError,
        "permission": PermissionError,
    }[failure]
    child = Child()

    with pytest.raises(expected_error):
        module.process_rss(child)
    assert child.poll_calls == int(
        failure in {"missing_file", "process_lookup", "missing_vmrss"}
    )
