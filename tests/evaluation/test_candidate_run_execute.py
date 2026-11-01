from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.comparison.strategies import UniversalStrategyComparison
from trade_rl.evaluation.runs.config import (
    CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY,
    LEGACY_DATASET_EXECUTION_OVERLAY,
    CandidateRunConfig,
    resolve_candidate_run_spec,
)
from trade_rl.evaluation.runs.execute import execute_candidate_run
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig


def market() -> MarketDataset:
    n = 8
    close = np.full((n, 2), 100.0, dtype=np.float64)
    return MarketDataset(
        dataset_id="b" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        features=np.zeros((n, 2, 1), dtype=np.float32),
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n, 2), 1_000_000.0),
        funding_rate=np.zeros((n, 2), dtype=np.float64),
        tradable=np.ones((n, 2), dtype=np.bool_),
        feature_available=np.ones((n, 2, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def resolved_spec(dataset: MarketDataset):
    config = CandidateRunConfig(
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
    return resolve_candidate_run_spec(
        dataset,
        dataset_artifact_schema="market_dataset_artifact_v3",
        dataset_artifact_digest="d" * 64,
        config=config,
    )


@pytest.mark.parametrize(
    ("execution_overlay", "expected_capacity"),
    [
        (LEGACY_DATASET_EXECUTION_OVERLAY, True),
        (CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY, False),
    ],
)
def test_execute_candidate_run_delegates_exactly_once_to_candidate_suite(
    monkeypatch,
    execution_overlay: str,
    expected_capacity: bool,
) -> None:
    from trade_rl.evaluation.runs import execute as execute_module

    dataset = market()
    spec = replace(resolved_spec(dataset), execution_overlay=execution_overlay)
    comparison = UniversalStrategyComparison(by_symbol=(), ppo_training_timesteps=2048)
    calls: list[tuple[object, object, dict[str, object]]] = []

    def fake_suite(loaded, lean_config, **kwargs):
        calls.append((loaded, lean_config, kwargs))
        return comparison

    monkeypatch.setattr(execute_module, "run_lean_candidate_suite", fake_suite)

    result = execute_candidate_run(dataset, spec)

    assert result.spec is spec
    assert result.symbols == dataset.symbols
    assert result.comparison is comparison
    assert result.ppo_training_timesteps == 2048
    assert len(calls) == 1
    loaded, lean_config, kwargs = calls[0]
    assert loaded is dataset
    assert lean_config == spec.lean_config
    assert kwargs["start_index"] == spec.evaluation_start_index
    assert kwargs["stop_index"] == spec.evaluation_stop_index
    assert kwargs["gross_budget"] == spec.config.gross_budget
    assert kwargs["initial_capital"] == spec.config.initial_capital
    execution_cost = kwargs["execution_cost"]
    assert isinstance(execution_cost, ExecutionCostConfig)
    assert execution_cost == replace(
        ExecutionCostConfig.zero(),
        borrow_rate_multiplier=1.0,
        processing_bar_volume_capacity=expected_capacity,
    )
    assert kwargs["risk"] is None
    assert kwargs["include_ppo_shared_cash_replay"] is False


@pytest.mark.parametrize(
    ("execution_overlay", "borrow_rate", "expected_borrow"),
    [
        (LEGACY_DATASET_EXECUTION_OVERLAY, 0.0, 0.0),
        (
            LEGACY_DATASET_EXECUTION_OVERLAY,
            0.365,
            500.0 * 0.365 / (365.0 * 24.0),
        ),
        (CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY, 0.0, 0.0),
        (
            CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY,
            0.365,
            500.0 * 0.365 / (365.0 * 24.0),
        ),
    ],
)
def test_execute_candidate_run_overlay_charges_dataset_borrow(
    monkeypatch: pytest.MonkeyPatch,
    execution_overlay: str,
    borrow_rate: float,
    expected_borrow: float,
) -> None:
    from trade_rl.evaluation.runs import execute as execute_module

    dataset = replace(
        market(),
        borrow_rate=np.full((8, 2), borrow_rate, dtype=np.float64),
        cash_rate=np.zeros(8, dtype=np.float64),
    )
    spec = replace(resolved_spec(dataset), execution_overlay=execution_overlay)
    comparison = UniversalStrategyComparison(by_symbol=(), ppo_training_timesteps=256)
    captured_costs: list[ExecutionCostConfig] = []

    def fake_suite(loaded, lean_config, **kwargs):
        assert loaded is dataset
        assert lean_config is spec.lean_config
        captured_costs.append(kwargs["execution_cost"])
        return comparison

    monkeypatch.setattr(execute_module, "run_lean_candidate_suite", fake_suite)
    execute_candidate_run(dataset, spec)

    assert len(captured_costs) == 1
    book = BookState.from_weights(
        weights=np.array([-0.5, 0.0]),
        capital=1_000.0,
        prices=dataset.close[0],
        contract_multipliers=dataset.contract_multipliers,
    )
    execution = MarketExecutor(dataset, captured_costs[0]).execute_interval(
        book,
        np.array([-0.5, 0.0]),
        start_index=0,
        bars=1,
    )

    assert execution.interval_borrow_cost == pytest.approx(expected_borrow)


def test_execute_candidate_run_rejects_dataset_identity_mismatch() -> None:
    dataset = market()
    spec = replace(resolved_spec(dataset), dataset_id="c" * 64)

    with pytest.raises(
        ValueError,
        match="dataset id does not match resolved candidate run spec",
    ):
        execute_candidate_run(dataset, spec)


def test_execute_candidate_run_uses_one_explicit_risk_for_training_and_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import execute as execute_module

    dataset = market()
    configured_risk = PreTradeRiskConfig(
        max_gross=0.5,
        max_abs_weight=0.1,
        max_turnover=None,
        drawdown_start=0.1,
        drawdown_stop=0.2,
    )
    original = resolved_spec(dataset)
    config = replace(
        original.config,
        ppo_minimum_hold_bars=0,
        ppo_observation_schema="ppo_observation_v3",
        ppo_settle_terminal_position=True,
        pretrade_risk_config=configured_risk,
    )
    spec = resolve_candidate_run_spec(
        dataset,
        dataset_artifact_schema="market_dataset_artifact_v3",
        dataset_artifact_digest="d" * 64,
        config=config,
    )
    comparison = UniversalStrategyComparison(
        by_symbol=(),
        ppo_training_timesteps=2048,
        ppo_training_minimum_hold_suppressed_count=0,
    )
    captured: dict[str, object] = {}

    def fake_suite(loaded, lean_config, **kwargs):
        captured["risk"] = kwargs["risk"]
        captured["settlement"] = lean_config.ppo_settle_terminal_position
        captured["shared_cash"] = kwargs["include_ppo_shared_cash_replay"]
        return comparison

    monkeypatch.setattr(execute_module, "run_lean_candidate_suite", fake_suite)

    execute_candidate_run(dataset, spec)

    assert isinstance(captured["risk"], PreTradeRisk)
    assert captured["risk"].config == configured_risk
    assert captured["settlement"] is True
    assert captured["shared_cash"] is True
