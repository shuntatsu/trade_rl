from __future__ import annotations

import pytest

from tools.tier_d_nonrl.tier_d_process_tree import (
    _WINDOWS_FILETIME_EPOCH_TICKS,
    ProcessIdentity,
    ProcessSnapshotError,
    ProcessTreeSampler,
    ProcessUsage,
    _capture_linux_snapshot,
    _capture_windows_snapshot,
    _parse_linux_stat,
    _windows_creation_times_match,
    capture_process_snapshot,
    process_identity_from_popen,
)


def _usage(
    pid: int,
    parent_pid: int,
    creation_time: int,
    rss_bytes: int,
) -> ProcessUsage:
    return ProcessUsage(
        pid=pid,
        parent_pid=parent_pid,
        creation_time=creation_time,
        rss_bytes=rss_bytes,
    )


def test_tree_rss_includes_python_descendant_instead_of_launcher_only() -> None:
    sampler = ProcessTreeSampler(ProcessIdentity(100, 100))

    sample = sampler.sample(
        (
            _usage(1, 0, 1, 10_000_000),
            _usage(100, 1, 100, 4_400_000),
            _usage(101, 100, 101, 248_800_000),
            _usage(102, 101, 102, 1_200_000),
            _usage(200, 1, 200, 900_000_000),
        )
    )

    assert sample.root_rss_bytes == 4_400_000
    assert sample.descendants_rss_bytes == 250_000_000
    assert sample.rss_bytes == 254_400_000
    assert sample.process_count == 3


def test_tree_sampler_remembers_owned_descendants_after_parent_exits() -> None:
    sampler = ProcessTreeSampler(ProcessIdentity(100, 100))
    sampler.sample(
        (
            _usage(100, 1, 100, 4_400_000),
            _usage(101, 100, 101, 248_800_000),
        )
    )

    sample = sampler.sample(
        (
            _usage(1, 0, 1, 10_000_000),
            _usage(101, 1, 101, 248_800_000),
            _usage(102, 101, 102, 1_200_000),
        )
    )

    assert sample.root_rss_bytes == 0
    assert sample.descendants_rss_bytes == 250_000_000
    assert sample.process_count == 2


def test_reused_root_pid_and_its_new_children_are_not_owned() -> None:
    sampler = ProcessTreeSampler(ProcessIdentity(100, 100))
    sampler.sample(
        (
            _usage(100, 1, 100, 4_400_000),
            _usage(101, 100, 101, 248_800_000),
        )
    )

    sample = sampler.sample(
        (
            _usage(100, 1, 999, 900_000_000),
            _usage(102, 100, 1000, 800_000_000),
            _usage(101, 1, 101, 248_800_000),
        )
    )

    assert sample.root_rss_bytes == 0
    assert sample.descendants_rss_bytes == 248_800_000
    assert sample.process_count == 1


def test_reused_child_pid_is_not_counted_as_the_old_child() -> None:
    sampler = ProcessTreeSampler(ProcessIdentity(100, 100))
    sampler.sample(
        (
            _usage(100, 1, 100, 4_400_000),
            _usage(101, 100, 101, 248_800_000),
        )
    )

    sample = sampler.sample(
        (
            _usage(100, 1, 100, 4_500_000),
            _usage(101, 1, 999, 900_000_000),
        )
    )

    assert sample.root_rss_bytes == 4_500_000
    assert sample.descendants_rss_bytes == 0
    assert sample.process_count == 1


def test_empty_snapshot_after_launch_fails_instead_of_reporting_zero() -> None:
    sampler = ProcessTreeSampler(ProcessIdentity(100, 100))
    sampler.sample((_usage(100, 1, 100, 4_400_000),))

    with pytest.raises(ProcessSnapshotError, match="empty"):
        sampler.sample(())


@pytest.mark.parametrize(
    "records",
    [
        (
            _usage(100, 1, 100, 4_400_000),
            _usage(101, 100, 101, 248_800_000),
            _usage(101, 100, 102, 1),
        ),
        (
            _usage(100, 1, 100, 4_400_000),
            _usage(101, 100, 101, -1),
        ),
        (
            _usage(100, 1, 100, 4_400_000),
            _usage(101, 100, 0, 248_800_000),
        ),
    ],
    ids=("duplicate-pid", "negative-rss", "invalid-creation-time"),
)
def test_invalid_snapshot_cannot_produce_a_low_rss_sample(
    records: tuple[ProcessUsage, ...],
) -> None:
    sampler = ProcessTreeSampler(ProcessIdentity(100, 100))

    with pytest.raises(ProcessSnapshotError):
        sampler.sample(records)


def test_partial_native_snapshot_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    sampler = ProcessTreeSampler(ProcessIdentity(100, 100))

    def fail_snapshot() -> tuple[ProcessUsage, ...]:
        raise ProcessSnapshotError("incomplete process snapshot")

    monkeypatch.setattr(
        "tools.tier_d_nonrl.tier_d_process_tree.capture_process_snapshot",
        fail_snapshot,
    )

    with pytest.raises(ProcessSnapshotError):
        sampler.sample()


def test_linux_stat_parser_handles_spaces_and_parentheses_in_command_name() -> None:
    fields = ["S", "10", *(["0"] * 17), "99"]

    assert _parse_linux_stat(f"123 (python ) worker) {' '.join(fields)}") == (
        123,
        10,
        99,
    )


def test_linux_proc_snapshot_uses_resident_pages_and_parent_links(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    import tools.tier_d_nonrl.tier_d_process_tree as process_tree

    real_scandir = os.scandir
    monkeypatch.setattr(process_tree.os, "sysconf", lambda _name: 4096, raising=False)
    monkeypatch.setattr(
        process_tree.os,
        "scandir",
        lambda _path: real_scandir(tmp_path),
    )
    for pid, parent, start_ticks, resident_pages in (
        (100, 1, 1000, 10),
        (101, 100, 1001, 20),
    ):
        directory = tmp_path / str(pid)
        directory.mkdir()
        fields = ["S", str(parent), *(["0"] * 17), str(start_ticks)]
        (directory / "stat").write_text(
            f"{pid} (python worker) {' '.join(fields)}", encoding="ascii"
        )
        (directory / "statm").write_text(
            f"100 {resident_pages} 1 0 0 0 0", encoding="ascii"
        )

    rows = _capture_linux_snapshot()

    assert rows == (
        _usage(100, 1, 1000, 40_960),
        _usage(101, 100, 1001, 81_920),
    )


def test_linux_proc_snapshot_fails_closed_on_permission_error(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    import tools.tier_d_nonrl.tier_d_process_tree as process_tree

    real_scandir = os.scandir
    monkeypatch.setattr(process_tree.os, "sysconf", lambda _name: 4096, raising=False)
    monkeypatch.setattr(
        process_tree.os,
        "scandir",
        lambda _path: real_scandir(tmp_path),
    )
    directory = tmp_path / "100"
    directory.mkdir()
    fields = ["S", "1", *(["0"] * 17), "1000"]
    (directory / "stat").write_text(
        f"100 (python worker) {' '.join(fields)}", encoding="ascii"
    )

    def unreadable(_path):
        raise PermissionError("unreadable proc entry")

    monkeypatch.setattr(process_tree, "_read_proc_text", unreadable)

    with pytest.raises(ProcessSnapshotError, match="incomplete"):
        _capture_linux_snapshot()


def test_windows_snapshot_rejects_nonzero_command_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(
        "tools.tier_d_nonrl.tier_d_process_tree.shutil.which",
        lambda _name: "powershell.exe",
    )
    monkeypatch.setattr(
        "tools.tier_d_nonrl.tier_d_process_tree.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr=""),
    )

    with pytest.raises(ProcessSnapshotError):
        _capture_windows_snapshot()


def test_windows_snapshot_discards_only_pid_zero_pseudo_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from types import SimpleNamespace

    payload = [
        {
            "pid": 0,
            "parent_pid": 0,
            "creation_time": _WINDOWS_FILETIME_EPOCH_TICKS + 1,
            "rss_bytes": 8192,
        },
        {
            "pid": 4,
            "parent_pid": 0,
            "creation_time": _WINDOWS_FILETIME_EPOCH_TICKS + 2,
            "rss_bytes": 2048,
        },
    ]
    monkeypatch.setattr(
        "tools.tier_d_nonrl.tier_d_process_tree.shutil.which",
        lambda _name: "powershell.exe",
    )
    monkeypatch.setattr(
        "tools.tier_d_nonrl.tier_d_process_tree.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(payload), stderr=""
        ),
    )

    rows = _capture_windows_snapshot()

    assert rows == (_usage(4, 0, 2, 2048),)


def test_windows_snapshot_rejects_fractional_rss_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from types import SimpleNamespace

    payload = [
        {
            "pid": 4,
            "parent_pid": 0,
            "creation_time": _WINDOWS_FILETIME_EPOCH_TICKS + 2,
            "rss_bytes": 1.5,
        }
    ]
    monkeypatch.setattr(
        "tools.tier_d_nonrl.tier_d_process_tree.shutil.which",
        lambda _name: "powershell.exe",
    )
    monkeypatch.setattr(
        "tools.tier_d_nonrl.tier_d_process_tree.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(payload), stderr=""
        ),
    )

    with pytest.raises(ProcessSnapshotError):
        _capture_windows_snapshot()


def test_native_snapshot_counts_a_live_direct_child() -> None:
    import os
    import subprocess
    import sys

    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        rows = capture_process_snapshot()
        by_pid = {row.pid: row for row in rows}
        parent = by_pid.get(os.getpid())
        child_row = by_pid.get(child.pid)
        assert parent is not None
        assert child_row is not None
        assert child_row.parent_pid == os.getpid()

        sampler = ProcessTreeSampler(ProcessIdentity(os.getpid(), parent.creation_time))
        sample = sampler.sample(rows)

        assert sample.root_rss_bytes == parent.rss_bytes
        assert sample.descendants_rss_bytes >= child_row.rss_bytes
        assert child_row.identity in sampler._owned_identities
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_from_popen_pins_a_real_launched_process_generation() -> None:
    import subprocess
    import sys

    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        identity = process_identity_from_popen(child)
        sampler = ProcessTreeSampler.from_popen(child)
        rows = capture_process_snapshot()
        sample = sampler.sample(rows)

        assert identity.pid == child.pid
        assert identity == next(row.identity for row in rows if row.pid == child.pid)
        assert sample.root_rss_bytes == next(
            row.rss_bytes for row in rows if row.identity == identity
        )
        assert sample.process_count >= 1
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_windows_handle_and_cim_creation_times_use_exact_microsecond_bucket() -> None:
    # CIM DateTime stores microseconds: 10 FILETIME ticks, rounded down.
    assert _windows_creation_times_match(1_000, 1_000)
    assert _windows_creation_times_match(1_000, 1_009)
    assert not _windows_creation_times_match(1_010, 1_009)
    assert not _windows_creation_times_match(990, 1_009)
    assert not _windows_creation_times_match(1_001, 1_009)


def test_reused_pid_before_first_snapshot_fails_generation_check() -> None:
    sampler = ProcessTreeSampler(ProcessIdentity(100, 100))

    with pytest.raises(ProcessSnapshotError, match="generation"):
        sampler.sample((_usage(100, 1, 999, 900_000_000),))


def test_windows_cim_skips_pid_zero_before_null_date_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from types import SimpleNamespace

    def fake_run(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        script = argv[-1]
        # The System Idle row may lack CreationDate. It must be discarded in
        # the PowerShell producer, not only after JSON reaches Python.
        skip = "if ($row.ProcessId -eq 0) { continue }"
        assert skip in script
        assert script.index(skip) < script.index("$null -eq $row.CreationDate")
        payload = [
            {
                "pid": 4,
                "parent_pid": 0,
                "creation_time": _WINDOWS_FILETIME_EPOCH_TICKS + 2,
                "rss_bytes": 2048,
            }
        ]
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(
        "tools.tier_d_nonrl.tier_d_process_tree.shutil.which",
        lambda _name: "powershell.exe",
    )
    monkeypatch.setattr(
        "tools.tier_d_nonrl.tier_d_process_tree.subprocess.run", fake_run
    )
    assert _capture_windows_snapshot() == (_usage(4, 0, 2, 2048),)
