"""Synthetic tests of the learning-lab split, metrics and trial registry."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

from trade_rl.evaluation.learning_lab import registry as reg
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
