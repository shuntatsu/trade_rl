from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.comparison.strategies import UniversalStrategyComparison
from trade_rl.evaluation.experiments.contracts import ResolvedRunConfig
from trade_rl.evaluation.runs.artifact import _result_payload
from trade_rl.evaluation.runs.config import (
    CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY,
    LEGACY_DATASET_EXECUTION_OVERLAY,
    CandidateRunConfig,
    resolve_candidate_run_spec,
)
from trade_rl.evaluation.runs.execute import CandidateRunResult, execute_candidate_run


def _market() -> MarketDataset:
    n = 8
    close = np.full((n, 1), 100.0, dtype=np.float64)
    return MarketDataset(
        dataset_id="b" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        features=np.zeros((n, 1, 1), dtype=np.float32),
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n, 1), 1_000.0, dtype=np.float64),
        funding_rate=np.zeros((n, 1), dtype=np.float64),
        tradable=np.ones((n, 1), dtype=np.bool_),
        feature_available=np.ones((n, 1, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def _config() -> CandidateRunConfig:
    return CandidateRunConfig(
        signal_name="signal",
        feature_names=("signal",),
        fit_symbol_names=("BTCUSDT",),
        fit_cutoff=np.datetime64("2026-01-01T04:00:00", "ns"),
        evaluation_start=np.datetime64("2026-01-01T04:00:00", "ns"),
        evaluation_stop_exclusive=np.datetime64("2026-01-01T07:00:00", "ns"),
        rule_entry_threshold=0.10,
        rule_exit_threshold=0.02,
        forecast_entry_threshold=0.01,
        forecast_exit_threshold=0.002,
        ppo_total_timesteps=256,
        ppo_seed=7,
        gross_budget=0.5,
        initial_capital=1_000.0,
    )


def _resolve(*, execution_overlay: str = LEGACY_DATASET_EXECUTION_OVERLAY):
    dataset = _market()
    return dataset, resolve_candidate_run_spec(
        dataset,
        dataset_artifact_schema="market_dataset_artifact_v3",
        dataset_artifact_digest="d" * 64,
        config=_config(),
        execution_overlay=execution_overlay,
    )


def test_execution_overlay_is_identity_bound_without_changing_raw_candidate_config() -> None:
    _, legacy = _resolve()
    _, causal = _resolve(
        execution_overlay=CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY
    )

    assert legacy.execution_overlay == LEGACY_DATASET_EXECUTION_OVERLAY
    assert causal.execution_overlay == CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY
    assert legacy.config.to_json_payload() == causal.config.to_json_payload()

    legacy_contract = ResolvedRunConfig.from_candidate_spec(legacy)
    causal_contract = ResolvedRunConfig.from_candidate_spec(causal)
    assert legacy_contract.execution_overlay == LEGACY_DATASET_EXECUTION_OVERLAY
    assert (
        causal_contract.execution_overlay
        == CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY
    )
    assert legacy_contract.digest != causal_contract.digest

    with pytest.raises(ValueError, match="execution_overlay"):
        resolve_candidate_run_spec(
            _market(),
            dataset_artifact_schema="market_dataset_artifact_v3",
            dataset_artifact_digest="d" * 64,
            config=_config(),
            execution_overlay="unknown-overlay",
        )


def test_execution_overlay_controls_runtime_capacity_mode(monkeypatch) -> None:
    from trade_rl.evaluation.runs import execute as execute_module

    observed: list[bool] = []

    def fake_suite(dataset, lean_config, **kwargs):
        del dataset, lean_config
        observed.append(kwargs["execution_cost"].processing_bar_volume_capacity)
        return UniversalStrategyComparison(by_symbol=())

    monkeypatch.setattr(execute_module, "run_lean_candidate_suite", fake_suite)

    legacy_dataset, legacy = _resolve()
    execute_candidate_run(legacy_dataset, legacy)
    causal_dataset, causal = _resolve(
        execution_overlay=CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY
    )
    execute_candidate_run(causal_dataset, causal)

    assert observed == [True, False]


def test_candidate_summary_records_actual_execution_overlay() -> None:
    _, spec = _resolve(
        execution_overlay=CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY
    )
    result = CandidateRunResult(
        spec=spec,
        symbols=("BTCUSDT",),
        comparison=UniversalStrategyComparison(by_symbol=()),
    )

    summary, _ = _result_payload(result)

    assert summary["evaluation"] == {
        "start": str(spec.config.evaluation_start),
        "stop_exclusive": str(spec.config.evaluation_stop_exclusive),
        "gross_budget": spec.config.gross_budget,
        "initial_capital": spec.config.initial_capital,
        "execution_overlay": CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY,
    }
