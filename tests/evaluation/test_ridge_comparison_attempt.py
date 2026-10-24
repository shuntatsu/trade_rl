from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data import (
    MarketCalendarKind,
    MarketDataset,
    publish_market_dataset_artifact,
)
from trade_rl.evaluation.runs import (
    RidgeArtifactComparisonConfig,
    load_ridge_comparison_artifact,
)
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.forecasts import (
    RidgeForecastArtifactBinding,
    RidgeForecastModel,
    publish_ridge_forecast_artifact,
)

_CONTEXT = "d" * 64


def _dataset() -> MarketDataset:
    offsets_minutes = np.asarray([0, 30, 120, 180, 360, 390], dtype=np.int64)
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + (
        offsets_minutes * np.timedelta64(1, "m")
    )
    signal = np.asarray([-0.20, -0.15, 0.20, 0.25, -0.20, -0.25], dtype=np.float32)
    secondary = np.asarray([0.4, 0.3, 0.2, 0.1, -0.1, -0.2], dtype=np.float32)
    features = np.repeat(
        np.stack((signal, secondary), axis=1)[:, None, :],
        2,
        axis=1,
    )
    close = np.asarray(
        [
            [100.0, 200.0],
            [101.0, 198.0],
            [102.0, 196.0],
            [103.0, 194.0],
            [104.0, 192.0],
            [105.0, 190.0],
        ],
        dtype=np.float64,
    )
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((6, 1), dtype=np.float32),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((6, 2), 100_000.0, dtype=np.float64),
        funding_rate=np.zeros((6, 2), dtype=np.float64),
        tradable=np.ones((6, 2), dtype=np.bool_),
        feature_available=np.ones((6, 2, 2), dtype=np.bool_),
        feature_names=("momentum", "secondary"),
        global_feature_names=("regime",),
        periods_per_year=252,
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=1.0,
    )
    return dataset.with_content_identity({"fixture": "ridge-comparison-attempt"})


def _inputs(
    tmp_path: Path,
) -> tuple[
    Path, Path, str, RidgeForecastArtifactBinding, RidgeArtifactComparisonConfig
]:
    dataset = _dataset()
    dataset_root = tmp_path / "dataset"
    published = publish_market_dataset_artifact(dataset_root, dataset)
    model = RidgeForecastModel(
        feature_indices=(1, 0),
        feature_mean=np.asarray([0.0, 0.0], dtype=np.float64),
        feature_scale=np.asarray([2.0, 1.0], dtype=np.float64),
        coefficients=np.asarray([0.0, 1.0], dtype=np.float64),
        intercept=0.0,
        horizon_hours=2,
        alpha=0.5,
        n_samples=4,
        fit_cutoff=dataset.timestamps[2].astype("datetime64[ns]"),
    )
    symbol_indices = (1, 0)
    binding = RidgeForecastArtifactBinding(
        dataset_id=dataset.dataset_id,
        dataset_schema_version=published.schema_version,
        dataset_artifact_sha256=published.artifact_digest,
        fit_symbol_indices=symbol_indices,
        fit_symbol_names=tuple(dataset.symbols[index] for index in symbol_indices),
        feature_names=dataset.feature_names,
        feature_indices=model.feature_indices,
        fit_cutoff_ns=int(model.fit_cutoff.astype("int64")),
        horizon_hours=model.horizon_hours,
        alpha_hex=float(model.alpha).hex(),
        n_samples=model.n_samples,
        vintage_id="literal-comparison-attempt-fixture",
        source_commit="a" * 40,
        implementation_sha256="b" * 64,
        runtime_sha256="c" * 64,
    )
    ridge_root = tmp_path / "ridge-model"
    ridge_digest = publish_ridge_forecast_artifact(ridge_root, model, binding)
    config = RidgeArtifactComparisonConfig(
        start_index=2,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=10_000.0,
        entry_threshold=0.10,
        exit_threshold=0.02,
        one_way_switch_cost=0.01,
        execution_cost=ExecutionCostConfig(
            fee_rate=0.001,
            spread_rate=0.0005,
            impact_rate=0.0001,
            max_participation_rate=1.0,
        ),
        risk=PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=1.0,
            max_turnover=None,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        ),
        minimum_hold_bars=0,
    )
    return dataset_root, ridge_root, ridge_digest, binding, config


def _invoke(**arguments):
    from trade_rl.evaluation.runs import run_ridge_comparison_attempt

    return run_ridge_comparison_attempt(**arguments)


def _arguments(root: Path, tmp_path: Path, **overrides) -> dict[str, object]:
    dataset_root, ridge_root, ridge_digest, binding, config = _inputs(tmp_path)
    arguments: dict[str, object] = {
        "dataset_root": dataset_root,
        "ridge_artifact_root": ridge_root,
        "expected_ridge_digest": ridge_digest,
        "expected_binding": binding,
        "config": config,
        "attempt_root": root,
        "research_context_digest": _CONTEXT,
    }
    arguments.update(overrides)
    return arguments


def _run(root: Path, tmp_path: Path, **overrides):
    return _invoke(**_arguments(root, tmp_path, **overrides))


def _record(root: Path, name: str) -> dict[str, object]:
    return json.loads((root / name / "record.json").read_text(encoding="utf-8"))


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_ridge_comparison_attempt_is_exported_from_runs_facade() -> None:
    import trade_rl.evaluation.runs as runs

    assert callable(getattr(runs, "run_ridge_comparison_attempt", None))


def test_actual_synthetic_attempt_publishes_strict_artifact_and_is_write_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.evaluation.runs.candidate_suite as suite
    import trade_rl.evaluation.runs.ridge_comparison_attempt as owner

    calls = 0
    actual_compare = owner.compare_ridge_artifact_shared_cash

    def counted_compare(**kwargs):
        nonlocal calls
        calls += 1
        return actual_compare(**kwargs)

    def forbidden_fit(*_args, **_kwargs):
        raise AssertionError("comparison attempt must not fit a model")

    monkeypatch.setattr(owner, "compare_ridge_artifact_shared_cash", counted_compare)
    monkeypatch.setattr(suite, "fit_ridge_forecast", forbidden_fit)
    monkeypatch.setattr(suite, "fit_lightgbm_forecast", forbidden_fit)
    monkeypatch.setattr(suite, "fit_ppo_strategy", forbidden_fit)

    attempt_root = tmp_path / "attempt"
    arguments = _arguments(attempt_root, tmp_path)
    result = _invoke(**arguments)

    assert calls == 1
    assert result.root == attempt_root
    assert len(result.result_artifact_digest) == 64
    loaded = load_ridge_comparison_artifact(
        attempt_root / "result",
        expected_digest=result.result_artifact_digest,
    )
    assert loaded.artifact_digest == result.result_artifact_digest
    assert tuple(arm["name"] for arm in loaded.payload["arms"]) == (
        "cash",
        "legacy_hysteresis",
        "position_aware_zero_cost",
        "position_aware_cost",
    )

    started = _record(attempt_root, "started")
    before = _record(attempt_root, "before")
    after = _record(attempt_root, "after")
    completed = _record(attempt_root, "complete")
    assert started["research_context_digest"] == _CONTEXT
    assert before["provenance"] == after["provenance"]
    assert before["provenance"]["research_context_digest"] == _CONTEXT
    assert completed["result_artifact_digest"] == result.result_artifact_digest
    assert not (attempt_root / "failure").exists()
    with pytest.raises(FrozenInstanceError):
        result.result_artifact_digest = "0" * 64

    prior_tree = _tree_bytes(attempt_root)
    with pytest.raises(FileExistsError):
        _invoke(**arguments)
    assert calls == 1
    assert _tree_bytes(attempt_root) == prior_tree


def test_missing_dataset_after_reservation_keeps_failure_evidence(
    tmp_path: Path,
) -> None:
    dataset_root, ridge_root, ridge_digest, binding, config = _inputs(tmp_path)
    attempt_root = tmp_path / "attempt"
    with pytest.raises((FileNotFoundError, ValueError)):
        _invoke(
            dataset_root=tmp_path / "missing-dataset",
            ridge_artifact_root=ridge_root,
            expected_ridge_digest=ridge_digest,
            expected_binding=binding,
            config=config,
            attempt_root=attempt_root,
            research_context_digest=_CONTEXT,
        )

    assert _record(attempt_root, "started")["expected_ridge_digest"] == ridge_digest
    assert (
        _record(attempt_root, "before")["provenance"]["research_context_digest"]
        == _CONTEXT
    )
    failure = _record(attempt_root, "failure")
    assert failure["phase"] == "comparison"
    assert failure["comparison_returned"] is False
    assert "not exposed" in failure["ledger_evidence_note"]
    assert not (attempt_root / "complete").exists()
    assert not (attempt_root / "result").exists()


def test_invalid_context_is_rejected_before_attempt_reservation(
    tmp_path: Path,
) -> None:
    dataset_root, ridge_root, ridge_digest, binding, config = _inputs(tmp_path)
    attempt_root = tmp_path / "attempt"

    with pytest.raises(ValueError, match="research_context_digest"):
        _invoke(
            dataset_root=dataset_root,
            ridge_artifact_root=ridge_root,
            expected_ridge_digest=ridge_digest,
            expected_binding=binding,
            config=config,
            attempt_root=attempt_root,
            research_context_digest="not-a-digest",
        )

    assert not attempt_root.exists()


def test_before_after_provenance_drift_blocks_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.evaluation.runs.ridge_comparison_attempt as owner

    provenance_calls = 0
    comparison_calls = 0

    def changing_provenance(*, research_context_digest):
        nonlocal provenance_calls
        provenance_calls += 1
        return {
            "schema_version": "test-provenance",
            "implementation_digest": "a" * 64 if provenance_calls == 1 else "b" * 64,
            "runtime_environment_digest": "c" * 64,
            "research_context_digest": research_context_digest,
        }

    def returned_for_drift_test(**_kwargs):
        nonlocal comparison_calls
        comparison_calls += 1
        return None

    def forbidden_publish(*_args, **_kwargs):
        raise AssertionError("drift must block result publication")

    monkeypatch.setattr(owner, "build_candidate_run_provenance", changing_provenance)
    monkeypatch.setattr(
        owner, "compare_ridge_artifact_shared_cash", returned_for_drift_test
    )
    monkeypatch.setattr(owner, "publish_ridge_comparison_artifact", forbidden_publish)
    attempt_root = tmp_path / "attempt"
    arguments = _arguments(attempt_root, tmp_path)

    with pytest.raises(RuntimeError, match="provenance changed"):
        _invoke(**arguments)

    assert provenance_calls == 2
    assert comparison_calls == 1
    assert (
        _record(attempt_root, "before")["provenance"]["implementation_digest"]
        == "a" * 64
    )
    assert (
        _record(attempt_root, "after")["provenance"]["implementation_digest"]
        == "b" * 64
    )
    assert _record(attempt_root, "failure")["phase"] == "verify_provenance"
    assert not (attempt_root / "result").exists()
    assert not (attempt_root / "complete").exists()


def test_publisher_rejection_preserves_returned_ledgers_as_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.evaluation.runs.ridge_comparison_attempt as owner

    def reject(*_args, **_kwargs):
        raise ValueError("injected result rejection")

    monkeypatch.setattr(owner, "publish_ridge_comparison_artifact", reject)
    attempt_root = tmp_path / "attempt"
    with pytest.raises(ValueError, match="injected result rejection"):
        _run(attempt_root, tmp_path)

    partial = _record(attempt_root, "partial")
    assert tuple(arm["name"] for arm in partial["arms"]) == (
        "cash",
        "legacy_hysteresis",
        "position_aware_zero_cost",
        "position_aware_cost",
    )
    assert all("ledger" in arm for arm in partial["arms"])
    failure = _record(attempt_root, "failure")
    assert failure["phase"] == "publish_result"
    assert failure["comparison_returned"] is True
    assert failure["partial_ledger_record"] == "partial/record.json"
    assert not (attempt_root / "complete").exists()


def test_journal_error_preserves_original_failure_and_partial_ledgers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.evaluation.runs.ridge_comparison_attempt as owner

    actual_write = owner._publish_record

    def fail_after_journal(destination: Path, record: object) -> None:
        if destination.name == "after":
            raise OSError("injected after-provenance journal failure")
        actual_write(destination, record)

    monkeypatch.setattr(owner, "_publish_record", fail_after_journal)
    attempt_root = tmp_path / "attempt"

    with pytest.raises(OSError, match="after-provenance journal failure"):
        _run(attempt_root, tmp_path)

    assert _record(attempt_root, "partial")["arms"]
    failure = _record(attempt_root, "failure")
    assert failure["phase"] == "after_provenance_journal"
    assert failure["comparison_returned"] is True
    assert not (attempt_root / "complete").exists()


def test_failure_journal_error_does_not_mask_original_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.evaluation.runs.ridge_comparison_attempt as owner

    def compare_failure(**_kwargs):
        raise RuntimeError("original comparison failure")

    actual_write = owner._publish_record

    def fail_failure_record(destination: Path, record: object) -> None:
        if destination.name == "failure":
            raise OSError("secondary failure-journal failure")
        actual_write(destination, record)

    monkeypatch.setattr(owner, "compare_ridge_artifact_shared_cash", compare_failure)
    monkeypatch.setattr(owner, "_publish_record", fail_failure_record)
    attempt_root = tmp_path / "attempt"

    with pytest.raises(RuntimeError, match="original comparison failure") as captured:
        _run(attempt_root, tmp_path)

    assert any(
        "secondary failure-journal failure" in note for note in captured.value.__notes__
    )
    assert (attempt_root / "started" / "record.json").is_file()
    assert (attempt_root / "before" / "record.json").is_file()
    assert not (attempt_root / "complete").exists()


def test_existing_attempt_directory_is_never_modified(
    tmp_path: Path,
) -> None:
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    sentinel = attempt_root / "preserve.txt"
    sentinel.write_text("keep", encoding="utf-8")
    before = _tree_bytes(attempt_root)

    with pytest.raises(FileExistsError):
        _run(attempt_root, tmp_path)

    assert _tree_bytes(attempt_root) == before
