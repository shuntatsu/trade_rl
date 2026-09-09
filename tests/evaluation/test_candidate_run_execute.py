from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.runs.config import (
    CandidateRunConfig,
    resolve_candidate_run_spec,
)
from trade_rl.evaluation.runs.execute import execute_candidate_run


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


def test_execute_candidate_run_delegates_exactly_once_to_candidate_suite(
    monkeypatch,
) -> None:
    from trade_rl.evaluation.runs import execute as execute_module

    dataset = market()
    spec = resolved_spec(dataset)
    comparison = object()
    calls: list[tuple[object, object, dict[str, object]]] = []

    def fake_suite(loaded, lean_config, **kwargs):
        calls.append((loaded, lean_config, kwargs))
        return comparison

    monkeypatch.setattr(execute_module, "run_lean_candidate_suite", fake_suite)

    result = execute_candidate_run(dataset, spec)

    assert result.spec is spec
    assert result.symbols == dataset.symbols
    assert result.comparison is comparison
    assert calls == [
        (
            dataset,
            spec.lean_config,
            {
                "start_index": spec.evaluation_start_index,
                "stop_index": spec.evaluation_stop_index,
                "gross_budget": spec.config.gross_budget,
                "initial_capital": spec.config.initial_capital,
                "execution_cost": None,
                "risk": None,
            },
        )
    ]


def test_execute_candidate_run_rejects_dataset_identity_mismatch() -> None:
    dataset = market()
    spec = replace(resolved_spec(dataset), dataset_id="c" * 64)

    with pytest.raises(
        ValueError,
        match="dataset id does not match resolved candidate run spec",
    ):
        execute_candidate_run(dataset, spec)
