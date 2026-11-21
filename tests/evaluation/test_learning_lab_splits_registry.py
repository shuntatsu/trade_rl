"""Synthetic tests of the learning-lab split, metrics and trial registry."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

from tests.evaluation import learning_lab_support as support
from tools import learning_lab as lab_tool
from trade_rl.evaluation.learning_lab import registry as reg
from trade_rl.evaluation.learning_lab import runner as lab_runner
from trade_rl.evaluation.learning_lab import splits
from trade_rl.evaluation.learning_lab.metrics import summarize_returns

HOUR4 = np.timedelta64(4, "h")


def _grid(first: str = "2021-06-01T00:00:00", last: str = "2026-07-31T20:00:00"):
    first_ns = np.datetime64(first, "ns")
    return np.arange(first_ns, np.datetime64(last, "ns") + HOUR4, HOUR4)


# --------------------------------------------------------------------- splits


def test_the_frozen_split_boundaries_are_exact_on_the_4h_grid() -> None:
    stamps = _grid()
    out = splits.split_indices(stamps)

    def at(i: int) -> str:
        return str(stamps[i])[:19]

    first, last = out[splits.TRAIN]
    assert at(first) == "2022-01-01T00:00:00" and at(last) == "2023-12-31T20:00:00"
    first, last = out[splits.VALIDATE]
    assert at(first) == "2024-01-01T00:00:00" and at(last) == "2025-06-30T20:00:00"
    first, last = out[splits.TEST]
    assert at(first) == "2025-07-01T00:00:00" and at(last) == "2026-07-31T20:00:00"
    assert (
        last == stamps.size - 1
    )  # the test window ends at the last row before 2026-08-01
    # adjacent, disjoint, in order
    assert out[splits.TRAIN][1] + 1 == out[splits.VALIDATE][0]
    assert out[splits.VALIDATE][1] + 1 == out[splits.TEST][0]


def test_no_row_at_or_after_the_cutoff_is_accepted() -> None:
    with_cutoff = _grid(last="2026-08-01T00:00:00")
    with pytest.raises(ValueError, match="cutoff"):
        splits.require_before_cutoff(with_cutoff)
    with pytest.raises(ValueError, match="cutoff"):
        splits.split_indices(with_cutoff)
    just_before = _grid(last="2026-07-31T20:00:00")
    splits.require_before_cutoff(just_before)
    assert str(splits.DATA_CUTOFF)[:10] == "2026-08-01"


def test_a_row_stamped_on_a_boundary_belongs_to_the_later_window() -> None:
    stamps = _grid()
    for name, instant in (
        (splits.VALIDATE, "2024-01-01T00:00:00"),
        (splits.TEST, "2025-07-01T00:00:00"),
    ):
        moment = np.datetime64(instant, "ns")
        assert splits.FROZEN_SPLITS[name].contains(moment)
    assert not splits.FROZEN_SPLITS[splits.TRAIN].contains(
        np.datetime64("2024-01-01T00:00:00", "ns")
    )
    assert stamps.size > 0


def test_split_sets_must_be_contiguous_ordered_and_complete() -> None:
    splits.validate_split_set(dict(splits.FROZEN_SPLITS))
    with pytest.raises(ValueError):
        splits.validate_split_set({splits.TRAIN: splits.FROZEN_SPLITS[splits.TRAIN]})
    gap = dict(splits.FROZEN_SPLITS)
    gap[splits.TEST] = splits.WalkForwardSplit(
        splits.TEST,
        np.datetime64("2025-08-01", "ns"),
        np.datetime64("2026-08-01", "ns"),
    )
    with pytest.raises(ValueError, match="contiguous"):
        splits.validate_split_set(gap)
    with pytest.raises(ValueError):
        splits.WalkForwardSplit(
            "x", np.datetime64("2024-01-01", "ns"), np.datetime64("2023-01-01", "ns")
        )


def test_an_empty_window_unsorted_or_nat_timestamps_are_refused() -> None:
    early = _grid(first="2021-01-01T00:00:00", last="2021-03-01T00:00:00")
    with pytest.raises(ValueError, match="no row"):
        splits.split_indices(early)
    with pytest.raises(ValueError, match="increasing"):
        splits.require_before_cutoff(early[::-1])
    with pytest.raises(ValueError, match="NaT"):
        splits.require_before_cutoff(np.array(["NaT"], dtype="datetime64[ns]"))


def test_validation_halves_partition_the_range() -> None:
    (a0, a1), (b0, b1) = splits.validation_halves(10, 20)
    assert (a0, b1) == (10, 20) and a1 + 1 == b0
    assert (b1 - b0 + 1) - (a1 - a0 + 1) in (0, 1)
    with pytest.raises(ValueError):
        splits.validation_halves(0, 2)


# -------------------------------------------------------------------- metrics


def test_summarize_returns_known_values() -> None:
    returns = [0.01, -0.005, 0.02, 0.0, 0.015]
    out = summarize_returns(returns, periods_per_year=2190)
    arr = np.asarray(returns)
    assert out["sharpe_per_period"] == pytest.approx(arr.mean() / arr.std(ddof=1))
    assert out["sharpe_annual"] == pytest.approx(
        out["sharpe_per_period"] * math.sqrt(2190)
    )
    assert out["total_return"] == pytest.approx(float(np.prod(1 + arr) - 1))
    # 1.01 -> 1.00495 : the only drawdown is the second bar
    assert out["max_drawdown"] == pytest.approx(0.005)
    flat = summarize_returns([0.0] * 5, periods_per_year=2190)
    assert flat["sharpe_per_period"] == 0.0 and flat["max_drawdown"] == 0.0
    with pytest.raises(ValueError):
        summarize_returns([float("nan")], periods_per_year=1)


# ------------------------------------------------------------------- registry


def _record(
    *,
    seed: int = 0,
    sharpe: float | None = 0.05,
    status: str = "completed",
    reason: str | None = None,
    method: str = "L1-PPO",
) -> reg.TrialRecord:
    metrics = {} if sharpe is None else {"sharpe_per_period": sharpe}
    return reg.TrialRecord(
        method_id=method,
        config_digest=hashlib.sha256(f"{method}{seed}".encode()).hexdigest(),
        seed=seed,
        split="validate",
        code_commit="a" * 40,
        metrics=metrics,
        status=status,
        stop_reason=reason,
    )


@pytest.mark.parametrize("status", ("shelved", "invalid", "failed"))
def test_deflated_last_refuses_ineligible_terminal_record_without_crashing(
    tmp_path: Path,
    status: str,
) -> None:
    path = tmp_path / "trials.jsonl"
    registry = reg.TrialRegistry(path)
    registry.append(
        reg.TrialRecord(
            method_id="L1-PPO",
            config_digest="a" * 64,
            seed=0,
            split="validate",
            code_commit="b" * 40,
            metrics={},
            status=status,
            stop_reason="no valid Sharpe",
        )
    )
    before = path.read_bytes()
    stdout, stderr = io.StringIO(), io.StringIO()
    code = lab_tool.main(
        ["registry", "--registry", str(path), "--deflated-for-last"],
        stdout=stdout,
        stderr=stderr,
    )
    assert code == lab_tool.EXIT_REFUSED
    assert stdout.getvalue() == ""
    assert "completed" in stderr.getvalue()
    assert path.read_bytes() == before and registry.verify() == 1
    stdout, stderr = io.StringIO(), io.StringIO()
    assert (
        lab_tool.main(
            ["registry", "--registry", str(path)],
            stdout=stdout,
            stderr=stderr,
        )
        == lab_tool.EXIT_OK
    )
    assert json.loads(stdout.getvalue())["trials"] == 1


def test_deflated_last_refuses_incomplete_success_metrics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trials.jsonl"
    registry = reg.TrialRegistry(path)
    registry.append(
        reg.TrialRecord(
            method_id="L1-PPO",
            config_digest="a" * 64,
            seed=0,
            split="validate",
            code_commit="b" * 40,
            metrics={"sharpe_per_period": 0.05},
            status="completed",
        )
    )
    before = path.read_bytes()
    stdout, stderr = io.StringIO(), io.StringIO()
    code = lab_tool.main(
        ["registry", "--registry", str(path), "--deflated-for-last"],
        stdout=stdout,
        stderr=stderr,
    )
    assert code == lab_tool.EXIT_REFUSED
    assert stdout.getvalue() == "" and "missing" in stderr.getvalue()
    assert path.read_bytes() == before and registry.verify() == 1


@pytest.mark.parametrize("invalid_count", (True, 4.5, "4", -1, 0, 1))
def test_deflated_last_does_not_silently_coerce_observation_count(
    tmp_path: Path,
    invalid_count: object,
) -> None:
    path = tmp_path / "trials.jsonl"
    registry = reg.TrialRegistry(path)
    registry.append(
        reg.TrialRecord(
            method_id="L1-PPO",
            config_digest="a" * 64,
            seed=0,
            split="validate",
            code_commit="b" * 40,
            metrics={
                "sharpe_per_period": 0.1,
                "n_obs": invalid_count,
                "skewness": 0.0,
                "kurtosis": 3.0,
            },
            status="completed",
        )
    )
    before = path.read_bytes()
    out, err = io.StringIO(), io.StringIO()
    code = lab_tool.main(
        ["registry", "--registry", str(path), "--deflated-for-last"],
        stdout=out,
        stderr=err,
    )
    assert code == lab_tool.EXIT_REFUSED
    assert "n_obs" in err.getvalue() and not out.getvalue()
    assert path.read_bytes() == before and registry.verify() == 1


def test_deflated_view_uses_only_one_verified_registry_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "trials.jsonl"
    registry = reg.TrialRegistry(path)
    metrics = {
        "sharpe_per_period": 0.05,
        "n_obs": 100,
        "skewness": 0.0,
        "kurtosis": 3.0,
    }
    registry.append(
        reg.TrialRecord(
            method_id="L1-PPO",
            config_digest="a" * 64,
            seed=0,
            split="validate",
            code_commit="b" * 40,
            metrics=metrics,
            status="completed",
        )
    )
    original_records = reg.TrialRegistry.records
    reads = 0

    def read_then_append(self: reg.TrialRegistry) -> list[reg.TrialRecord]:
        nonlocal reads
        snapshot = original_records(self)
        reads += 1
        if reads == 1:
            self.append(
                reg.TrialRecord(
                    method_id="L1-PPO",
                    config_digest="a" * 64,
                    seed=1,
                    split="validate",
                    code_commit="b" * 40,
                    metrics={},
                    status="failed",
                    stop_reason="concurrent append",
                )
            )
        return snapshot

    monkeypatch.setattr(reg.TrialRegistry, "records", read_then_append)
    out, err = io.StringIO(), io.StringIO()
    assert (
        lab_tool.main(
            ["registry", "--registry", str(path), "--deflated-for-last"],
            stdout=out,
            stderr=err,
        )
        == lab_tool.EXIT_OK
    ), err.getvalue()
    result = json.loads(out.getvalue())
    assert reads == 1
    assert result["trials"] == 1
    assert result["deflated_sharpe"]["n_trials"] == 1.0
    assert registry.verify() == 2


@pytest.mark.parametrize(
    "huge_count", (10**309, 10**1000), ids=("float_max_plus", "long_integer")
)
def test_deflated_last_refuses_unrepresentable_integer_observation_count(
    tmp_path: Path,
    huge_count: int,
) -> None:
    path = tmp_path / "trials.jsonl"
    registry = reg.TrialRegistry(path)
    registry.append(
        reg.TrialRecord(
            method_id="L1-PPO",
            config_digest="a" * 64,
            seed=0,
            split="validate",
            code_commit="b" * 40,
            metrics={
                "sharpe_per_period": 0.05,
                "n_obs": huge_count,
                "skewness": 0.0,
                "kurtosis": 3.0,
            },
            status="completed",
        )
    )
    before = path.read_bytes()
    out, err = io.StringIO(), io.StringIO()
    assert (
        lab_tool.main(
            ["registry", "--registry", str(path), "--deflated-for-last"],
            stdout=out,
            stderr=err,
        )
        == lab_tool.EXIT_REFUSED
    )
    assert not out.getvalue() and "n_obs" in err.getvalue()
    assert path.read_bytes() == before and registry.verify() == 1


@pytest.mark.parametrize("overflow_in", ("last_sharpe", "earlier_sharpe"))
def test_deflated_last_refuses_numeric_overflow_in_trial_series(
    tmp_path: Path,
    overflow_in: str,
) -> None:
    path = tmp_path / "trials.jsonl"
    registry = reg.TrialRegistry(path)
    if overflow_in == "earlier_sharpe":
        registry.append(
            reg.TrialRecord(
                method_id="L1-PPO",
                config_digest="a" * 64,
                seed=0,
                split="validate",
                code_commit="b" * 40,
                metrics={"sharpe_per_period": 10**1000},
                status="completed",
            )
        )
    registry.append(
        reg.TrialRecord(
            method_id="L1-PPO",
            config_digest="a" * 64,
            seed=1,
            split="validate",
            code_commit="b" * 40,
            metrics={
                "sharpe_per_period": 1e308 if overflow_in == "last_sharpe" else 0.05,
                "n_obs": 100,
                "skewness": 0.0,
                "kurtosis": 3.0,
            },
            status="completed",
        )
    )
    before = path.read_bytes()
    out, err = io.StringIO(), io.StringIO()
    assert (
        lab_tool.main(
            ["registry", "--registry", str(path), "--deflated-for-last"],
            stdout=out,
            stderr=err,
        )
        == lab_tool.EXIT_REFUSED
    )
    assert out.getvalue() == "" and "refused:" in err.getvalue()
    assert path.read_bytes() == before and registry.verify() == (
        2 if overflow_in == "earlier_sharpe" else 1
    )


@pytest.mark.parametrize("name", ("sharpe_per_period", "skewness", "kurtosis"))
@pytest.mark.parametrize("invalid_moment", (True, None, "0.03"))
def test_deflated_last_refuses_nonnumeric_moments_without_mutating_registry(
    tmp_path: Path,
    name: str,
    invalid_moment: object,
) -> None:
    path = tmp_path / "trials.jsonl"
    registry = reg.TrialRegistry(path)
    metrics: dict[str, object] = {
        "sharpe_per_period": 0.05,
        "n_obs": 100,
        "skewness": 0.0,
        "kurtosis": 3.0,
    }
    metrics[name] = invalid_moment
    registry.append(
        reg.TrialRecord(
            method_id="L1-PPO",
            config_digest="a" * 64,
            seed=0,
            split="validate",
            code_commit="b" * 40,
            metrics=metrics,
            status="completed",
        )
    )
    before = path.read_bytes()
    out, err = io.StringIO(), io.StringIO()
    assert (
        lab_tool.main(
            ["registry", "--registry", str(path), "--deflated-for-last"],
            stdout=out,
            stderr=err,
        )
        == lab_tool.EXIT_REFUSED
    )
    assert not out.getvalue() and name in err.getvalue()
    assert path.read_bytes() == before and registry.verify() == 1


@pytest.mark.parametrize("name", ("sharpe_per_period", "skewness", "kurtosis"))
@pytest.mark.parametrize("bad_value", (math.nan, math.inf, -math.inf, 10**1000))
def test_deflated_for_refuses_unpersistable_moments_from_mutated_record(
    name: str,
    bad_value: float | int,
) -> None:
    # The registry rejects nonfinite floats on append. Also defend the runner
    # against a mutable TrialRecord supplied directly by an in-process caller.
    metrics: dict[str, object] = {
        "sharpe_per_period": 0.05,
        "n_obs": 100,
        "skewness": 0.0,
        "kurtosis": 3.0,
    }
    record = reg.TrialRecord(
        method_id="L1-PPO",
        config_digest="a" * 64,
        seed=0,
        split="validate",
        code_commit="b" * 40,
        metrics=metrics,
        status="completed",
    )
    metrics[name] = bad_value
    with pytest.raises(ValueError, match=f"{name} must be a finite number"):
        lab_runner.deflated_for([record])


def test_deflated_last_works_for_completed_record_with_real_moments(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trials.jsonl"
    registry = reg.TrialRegistry(path)
    registry.append(
        reg.TrialRecord(
            method_id="L1-PPO",
            config_digest="a" * 64,
            seed=0,
            split="validate",
            code_commit="b" * 40,
            metrics={
                "sharpe_per_period": 0.02,
                "n_obs": 100,
                "skewness": 0.0,
                "kurtosis": 3.0,
            },
            status="completed",
        )
    )
    out, err = io.StringIO(), io.StringIO()
    code = lab_tool.main(
        ["registry", "--registry", str(path), "--deflated-for-last"],
        stdout=out,
        stderr=err,
    )
    assert code == lab_tool.EXIT_OK, err.getvalue()
    result = json.loads(out.getvalue())
    assert result["trials"] == 1 and result["chain_verified"] is True
    assert 0.0 <= result["deflated_sharpe"]["deflated_sharpe"] <= 1.0


@pytest.mark.parametrize(
    "malformed",
    (
        "null_root",
        "array_root",
        "null_record",
        "missing_seed",
        "null_metrics",
        "numeric_config_digest",
        "invalid_metrics_entries",
    ),
)
def test_malformed_registry_json_fails_closed_for_cli_verify_and_append(
    tmp_path: Path, malformed: str
) -> None:
    """Corrupt JSON must not escape with TypeError/AttributeError or mutate trials."""
    valid = _record(seed=0).to_dict()
    body: object = {
        "schema": reg.SCHEMA,
        "index": 0,
        "prev_sha256": reg.GENESIS,
        "record": valid,
    }
    if malformed == "null_root":
        body = None
    elif malformed == "array_root":
        body = []
    elif malformed == "null_record":
        body["record"] = None  # type: ignore[index]
    elif malformed == "missing_seed":
        del valid["seed"]
    elif malformed == "null_metrics":
        valid["metrics"] = None
    elif malformed == "numeric_config_digest":
        valid["config_digest"] = 17
    else:
        valid["metrics"] = [1, 2]
    path = tmp_path / "trials.jsonl"
    payload = (json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(payload)
    registry = reg.TrialRegistry(path)
    with pytest.raises(reg.RegistryError, match="line 0"):
        registry.verify()
    with pytest.raises(reg.RegistryError, match="line 0"):
        registry.append(_record(seed=2))
    out, err = io.StringIO(), io.StringIO()
    assert (
        lab_tool.main(["registry", "--registry", str(path)], stdout=out, stderr=err)
        == lab_tool.EXIT_REFUSED
    )
    assert not out.getvalue() and "refused" in err.getvalue()
    assert path.read_bytes() == payload


def test_the_registry_is_append_only_and_chained(tmp_path: Path) -> None:
    registry = reg.TrialRegistry(tmp_path / "trials.jsonl")
    assert registry.records() == []
    for seed in range(3):
        assert registry.append(_record(seed=seed)) == seed
    before = registry.path.read_bytes()
    assert registry.append(_record(seed=3)) == 3
    after = registry.path.read_bytes()
    assert after.startswith(before) and len(after) > len(before)  # only appended
    assert registry.verify() == 4
    assert [r.seed for r in registry.records()] == [0, 1, 2, 3]
    lines = after.split(b"\n")[:-1]
    assert json.loads(lines[0])["prev_sha256"] == reg.GENESIS
    assert json.loads(lines[1])["prev_sha256"] == hashlib.sha256(lines[0]).hexdigest()


def _registry_worker(
    path: Path, ready: Path, *, seed: int | None
) -> subprocess.Popen[str]:
    """Independent Python interpreter, not a thread sharing one registry object."""
    script = """
import sys
from pathlib import Path
from trade_rl.evaluation.learning_lab.registry import TrialRecord, TrialRegistry
path, ready = Path(sys.argv[1]), Path(sys.argv[2])
seed = sys.argv[3]
record = None if seed == 'verify' else TrialRecord(
    method_id='parallel', config_digest='a' * 64, seed=int(seed),
    split='validate', code_commit='b' * 40, metrics={}, status='completed',
)
ready.write_text('ready')
registry = TrialRegistry(path)
print(registry.verify() if record is None else registry.append(record), flush=True)
"""
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(path),
            str(ready),
            "verify" if seed is None else str(seed),
        ],
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _ready_all(paths: list[Path]) -> None:
    deadline = time.monotonic() + 10.0
    while not all(path.exists() for path in paths):
        if time.monotonic() > deadline:
            pytest.fail("the child Python interpreters did not start")
        time.sleep(0.01)


def test_two_registry_writers_wait_for_the_same_exclusive_file_lock(
    tmp_path: Path,
) -> None:
    fcntl = pytest.importorskip("fcntl")
    path = tmp_path / "trials.jsonl"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    children: list[subprocess.Popen[str]] = []
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        ready = [tmp_path / f"ready-{i}" for i in range(2)]
        children = [_registry_worker(path, ready[i], seed=i) for i in range(2)]
        _ready_all(ready)
        # Neither writer may read the old chain and append while an external
        # cooperating process holds the lock, even on independent descriptors.
        time.sleep(0.2)
        assert path.read_bytes() == b""
        assert all(child.poll() is None for child in children)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        for child in children:
            if child.poll() is None:
                try:
                    child.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5.0)
    results = [child.communicate(timeout=1.0) for child in children]
    assert all(child.returncode == 0 for child in children), results
    assert sorted(int(stdout.strip()) for stdout, _ in results) == [0, 1]
    assert reg.TrialRegistry(path).verify() == 2
    assert {record.seed for record in reg.TrialRegistry(path).records()} == {0, 1}


def test_registry_reader_waits_for_writer_lock_before_verifying(
    tmp_path: Path,
) -> None:
    fcntl = pytest.importorskip("fcntl")
    path = tmp_path / "trials.jsonl"
    registry = reg.TrialRegistry(path)
    registry.append(_record())
    descriptor = os.open(path, os.O_RDONLY)
    child: subprocess.Popen[str] | None = None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        ready = tmp_path / "reader-ready"
        child = _registry_worker(path, ready, seed=None)
        _ready_all([ready])
        time.sleep(0.2)
        assert child.poll() is None
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        if child is not None and child.poll() is None:
            try:
                child.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5.0)
    assert child is not None
    stdout, stderr = child.communicate(timeout=1.0)
    assert child.returncode == 0, stderr
    assert stdout.strip() == "1"


def test_registry_refuses_symlink_and_special_files_without_mutating_targets(
    tmp_path: Path,
) -> None:
    pytest.importorskip("fcntl")
    if not hasattr(os, "mkfifo") or not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("POSIX no-follow file descriptors required")
    target = tmp_path / "canonical.jsonl"
    registry = reg.TrialRegistry(target)
    registry.append(_record())
    original = target.read_bytes()
    link = tmp_path / "shortcut.jsonl"
    link.symlink_to(target)
    with pytest.raises(reg.RegistryError, match="safely open"):
        reg.TrialRegistry(link).append(_record(seed=1))
    with pytest.raises(reg.RegistryError, match="safely open"):
        reg.TrialRegistry(link).verify()
    assert target.read_bytes() == original

    fifo = tmp_path / "not-a-regular-registry.jsonl"
    os.mkfifo(fifo)
    with pytest.raises(reg.RegistryError, match="regular file"):
        reg.TrialRegistry(fifo).verify()
    with pytest.raises(reg.RegistryError, match="regular file"):
        reg.TrialRegistry(fifo).append(_record(seed=2))
    assert registry.verify() == 1


def test_read_snapshot_refuses_path_swapped_while_the_inode_is_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fcntl")
    path = tmp_path / "trials.jsonl"
    registry = reg.TrialRegistry(path)
    registry.append(_record())
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(path.read_bytes())
    rotated = tmp_path / "old-inode.jsonl"
    original = reg.TrialRegistry._read_lines

    def swap_after_read(descriptor: int) -> list[bytes]:
        lines = original(descriptor)
        path.rename(rotated)
        replacement.rename(path)
        return lines

    monkeypatch.setattr(reg.TrialRegistry, "_read_lines", staticmethod(swap_after_read))
    with pytest.raises(reg.RegistryError, match="locked regular file"):
        registry.verify()
    assert rotated.read_bytes() == path.read_bytes()


def test_an_edited_removed_or_reordered_line_breaks_the_chain(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    registry = reg.TrialRegistry(path)
    for seed in range(4):
        registry.append(_record(seed=seed))
    original = path.read_bytes()
    lines = original.split(b"\n")[:-1]

    path.write_bytes(original.replace(b'"seed":1', b'"seed":9', 1))
    with pytest.raises(reg.RegistryError):
        registry.verify()
    path.write_bytes(b"\n".join([lines[0], lines[2], lines[3]]) + b"\n")
    with pytest.raises(reg.RegistryError):
        registry.verify()
    path.write_bytes(b"\n".join([lines[1], lines[0], lines[2], lines[3]]) + b"\n")
    with pytest.raises(reg.RegistryError):
        registry.verify()
    # a broken registry refuses further appends and is left as found
    path.write_bytes(b"\n".join([lines[0], lines[2]]) + b"\n")
    broken = path.read_bytes()
    with pytest.raises(reg.RegistryError):
        registry.append(_record(seed=7))
    assert path.read_bytes() == broken


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("container", ["nested_dict", "nested_list"])
def test_registry_rejects_nested_nonfinite_metrics(
    tmp_path: Path, nonfinite: float, container: str
) -> None:
    """Nested statistics must not be stored as nonstandard JSON NaN/Infinity."""
    metrics = (
        {"halves": {"first": {"sharpe": nonfinite}}}
        if container == "nested_dict"
        else {"paths": [0.1, {"turnover": nonfinite}]}
    )
    with pytest.raises(reg.RegistryError, match="finite"):
        reg.TrialRecord(
            method_id="m",
            config_digest="c" * 64,
            seed=0,
            split="validate",
            code_commit="d" * 40,
            metrics=metrics,
            status="completed",
        )
    assert not (tmp_path / "trials.jsonl").exists()


def test_mutated_nested_metrics_cannot_break_an_existing_registry(
    tmp_path: Path,
) -> None:
    registry = reg.TrialRegistry(tmp_path / "trials.jsonl")
    assert registry.append(_record(seed=0)) == 0
    original = registry.path.read_bytes()
    metrics: dict[str, object] = {"halves": {"first": {"sharpe": 0.05}}}
    record = reg.TrialRecord(
        method_id="m",
        config_digest="c" * 64,
        seed=1,
        split="validate",
        code_commit="d" * 40,
        metrics=metrics,
        status="completed",
    )
    metrics["halves"] = {"first": {"sharpe": float("nan")}}
    with pytest.raises(reg.RegistryError, match="finite"):
        registry.append(record)
    assert registry.path.read_bytes() == original
    assert registry.verify() == 1


def test_nested_finite_metrics_roundtrip_under_v1_schema(tmp_path: Path) -> None:
    registry = reg.TrialRegistry(tmp_path / "trials.jsonl")
    record = reg.TrialRecord(
        method_id="m",
        config_digest="c" * 64,
        seed=1,
        split="validate",
        code_commit="d" * 40,
        metrics={"halves": {"first": {"sharpe": 0.05}}, "paths": [0.2, 0.3]},
        status="completed",
    )
    assert registry.append(record) == 0
    assert registry.verify() == 1
    assert registry.records()[0].metrics == record.metrics


def test_records_validate_their_fields(tmp_path: Path) -> None:
    with pytest.raises(reg.RegistryError, match="stop_reason"):
        _record(status="shelved")
    shelved = _record(
        status="shelved", reason="validation Sharpe below the best stride"
    )
    assert shelved.stop_reason
    with pytest.raises(reg.RegistryError):
        _record(status="bogus")
    with pytest.raises(reg.RegistryError):
        reg.TrialRecord(
            method_id="m",
            config_digest="nothex",
            seed=0,
            split="train",
            code_commit="c",
            metrics={},
            status="completed",
        )
    with pytest.raises(reg.RegistryError, match="finite"):
        _record(sharpe=float("inf"))
    registry = reg.TrialRegistry(tmp_path / "t.jsonl")
    registry.append(shelved)
    assert registry.records()[0].stop_reason == shelved.stop_reason


# ------------------------------------------------------------- deflated Sharpe


@pytest.mark.parametrize(
    "n_trials,variance",
    [
        (True, 0.01),
        (2.5, 0.01),
        (0, 0.01),
        (10**20, 0.01),
        (2, float("nan")),
        (2, float("inf")),
        (2, True),
    ],
)
def test_expected_max_sharpe_rejects_invalid_numeric_inputs(
    n_trials: object, variance: float
) -> None:
    with pytest.raises(ValueError):
        reg.expected_max_sharpe(n_trials, variance)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "updates",
    [
        {"sharpe": float("nan")},
        {"sharpe": 1e200},
        {"skewness": float("inf")},
        {"kurtosis": float("nan")},
        {"n_obs": True},
        {"n_obs": 2.5},
        {"n_obs": 10**400},
        {"n_trials": False},
        {"trial_sharpe_variance": float("inf")},
        {"kurtosis": float("inf")},
    ],
)
def test_deflated_sharpe_refuses_bad_statistics(updates: dict[str, object]) -> None:
    params: dict[str, object] = {
        "sharpe": 0.1,
        "n_obs": 300,
        "skewness": 0.0,
        "kurtosis": 3.0,
        "n_trials": 4,
        "trial_sharpe_variance": 0.01,
    }
    params.update(updates)
    with pytest.raises(ValueError):
        reg.deflated_sharpe_ratio(**params)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", (True, "nope", None, 10**400, float("nan")))
def test_trial_sharpes_refuses_malformed_registered_metric(bad: object) -> None:
    metrics: dict[str, object] = {"sharpe_per_period": 0.02}
    record = reg.TrialRecord(
        method_id="m",
        config_digest="a" * 64,
        seed=0,
        split="validate",
        code_commit="b" * 40,
        metrics=metrics,
        status="completed",
    )
    metrics["sharpe_per_period"] = bad
    with pytest.raises(ValueError):
        reg.trial_sharpes([record])


def test_registry_deflation_rejects_overflowed_sample_variance() -> None:
    records = [_record(seed=i, sharpe=s) for i, s in enumerate((1e200, -1e200))]
    with pytest.raises(ValueError):
        reg.registry_deflated_sharpe(
            records, sharpe=0.1, n_obs=200, skewness=0.0, kurtosis=3.0
        )


@pytest.mark.parametrize(
    "changed",
    [
        {"method_id": 1},
        {"split": 2},
        {"code_commit": True},
        {"stop_reason": 17},
        {"recorded_at": 123},
        {"status": False},
    ],
)
def test_trial_records_refuse_nontext_identity_and_metadata(
    changed: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "method_id": "L1-PPO",
        "config_digest": "a" * 64,
        "seed": 0,
        "split": "validate",
        "code_commit": "b" * 40,
        "metrics": {},
        "status": "completed",
    }
    payload.update(changed)
    with pytest.raises(reg.RegistryError):
        reg.TrialRecord(**payload)  # type: ignore[arg-type]


def test_config_digest_binds_actual_initial_capital() -> None:
    payload = lab_runner.protocol_payload()
    assert payload["observation_bar_clock"].startswith("uniform dataset-native")
    assert payload["run_identity"] == "config digest includes actual initial_capital"
    setup = support.synthetic_setup(n_bars=120)
    method = "BASE-STRIDE-6"
    baseline = lab_runner.config_digest(setup, None, method)
    richer = replace(setup, initial_capital=setup.initial_capital * 2)
    assert lab_runner.config_digest(richer, None, method) != baseline
    with pytest.raises(lab_runner.RunnerError, match="initial_capital"):
        replace(setup, initial_capital=0.0)


def test_dev_gate_refuses_nonfinite_agent_and_baseline_values() -> None:
    def row(*, sharpe: float = 0.03, dd: float = 0.05) -> dict[str, object]:
        return {
            "sharpe_per_period": sharpe,
            "max_drawdown": dd,
            "half1": {"sharpe_per_period": sharpe},
            "half2": {"sharpe_per_period": sharpe},
        }

    baseline = {"BASE-STRIDE-6": row(sharpe=0.01)}
    agents = {seed: row() for seed in range(5)}
    assert (
        lab_runner.dev_decision(agents, baseline)["label"] == lab_runner.LABEL_PROMISING
    )
    for bad_input in (
        ({**agents, 4: row(dd=float("nan"))}, baseline),
        (agents, {"BASE-STRIDE-6": row(sharpe=float("nan"))}),
        ({**agents, 0: row(sharpe=float("inf"))}, baseline),
        ({0: row(sharpe=1e308), 1: row(sharpe=1e308)}, baseline),
    ):
        with pytest.raises((lab_runner.RunnerError, ValueError)):
            lab_runner.dev_decision(*bad_input)


def test_expected_max_sharpe_known_values() -> None:
    # reference values computed independently with scipy.stats.norm
    assert reg.expected_max_sharpe(10, 0.01) == pytest.approx(0.157459830134575)
    assert reg.expected_max_sharpe(100, 0.04) == pytest.approx(0.506120578640337)
    assert reg.expected_max_sharpe(1, 0.5) == 0.0
    assert reg.expected_max_sharpe(50, 0.0) == 0.0
    assert reg.expected_max_sharpe(100, 0.04) > reg.expected_max_sharpe(10, 0.04)


def test_deflated_sharpe_known_values() -> None:
    value = reg.deflated_sharpe_ratio(
        0.1,
        n_obs=250,
        skewness=-0.3,
        kurtosis=5.0,
        n_trials=10,
        trial_sharpe_variance=0.01,
    )
    assert value == pytest.approx(0.18697646742160656)
    # one trial is the probabilistic Sharpe ratio against zero
    psr = reg.deflated_sharpe_ratio(
        0.1,
        n_obs=250,
        skewness=0.0,
        kurtosis=3.0,
        n_trials=1,
        trial_sharpe_variance=0.0,
    )
    z = 0.1 * math.sqrt(249) / math.sqrt(1 + 0.5 * 0.01)
    assert psr == pytest.approx(NormalDist().cdf(z))
    zero = reg.deflated_sharpe_ratio(
        0.0,
        n_obs=100,
        skewness=0.0,
        kurtosis=3.0,
        n_trials=1,
        trial_sharpe_variance=0.0,
    )
    assert zero == pytest.approx(0.5)
    # more trials can only lower the deflated Sharpe
    many = reg.deflated_sharpe_ratio(
        0.1,
        n_obs=250,
        skewness=0.0,
        kurtosis=3.0,
        n_trials=50,
        trial_sharpe_variance=0.01,
    )
    assert many < psr


def test_the_registry_deflated_sharpe_counts_every_trial_including_failures() -> None:
    sharpes = [0.01, 0.03, -0.02, 0.05]
    records = [_record(seed=i, sharpe=s) for i, s in enumerate(sharpes)]
    base = reg.registry_deflated_sharpe(
        records, sharpe=0.05, n_obs=300, skewness=0.0, kurtosis=3.0
    )
    variance = float(np.var(sharpes, ddof=1))
    assert base["n_trials"] == 4
    assert base["trial_sharpe_variance"] == pytest.approx(variance)
    assert base["expected_max_sharpe"] == pytest.approx(
        reg.expected_max_sharpe(4, variance)
    )
    failed = [
        *records,
        _record(seed=9, sharpe=None, status="failed", reason="diverged"),
    ]
    worse = reg.registry_deflated_sharpe(
        failed, sharpe=0.05, n_obs=300, skewness=0.0, kurtosis=3.0
    )
    assert worse["n_trials"] == 5  # a run without a metric still counts as a trial
    assert worse["deflated_sharpe"] < base["deflated_sharpe"]
    with pytest.raises(ValueError):
        reg.deflated_sharpe_ratio(
            9.0,
            n_obs=10,
            skewness=5.0,
            kurtosis=3.0,
            n_trials=2,
            trial_sharpe_variance=1.0,
        )
