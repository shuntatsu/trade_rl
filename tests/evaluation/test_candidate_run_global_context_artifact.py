from __future__ import annotations

import json

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.comparison.strategies import compare_strategies_by_symbol
from trade_rl.evaluation.runs.artifact import (
    load_candidate_run_artifact,
    publish_candidate_run,
)
from trade_rl.evaluation.runs.config import (
    CandidateRunConfig,
    resolve_candidate_run_spec,
)
from trade_rl.evaluation.runs.execute import CandidateRunResult
from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.ppo import (
    PPO_GLOBAL_BTC_REGIME_CONTEXT,
    ppo_observation_contract_payload,
)


def _market() -> MarketDataset:
    n = 8
    close = np.asarray(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [105.0, 95.0],
            [110.0, 90.0],
            [115.0, 85.0],
            [120.0, 80.0],
            [125.0, 75.0],
            [130.0, 70.0],
        ]
    )
    signal = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    features = np.stack((signal, -signal), axis=1).reshape(n, 2, 1)
    return MarketDataset(
        dataset_id="b" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n, 2), 1_000_000.0),
        funding_rate=np.zeros((n, 2)),
        tradable=np.ones((n, 2), dtype=np.bool_),
        feature_available=np.ones((n, 2, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def _candidate_result() -> CandidateRunResult:
    dataset = _market()
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
        ppo_global_context=PPO_GLOBAL_BTC_REGIME_CONTEXT,
    )
    spec = resolve_candidate_run_spec(
        dataset,
        dataset_artifact_schema="market_dataset_artifact_v3",
        dataset_artifact_digest="d" * 64,
        config=config,
    )
    comparison = compare_strategies_by_symbol(
        dataset,
        {"cash": ConstantIntentStrategy(PositionIntent.FLAT)},
        start_index=4,
        stop_index=7,
        gross_budget=0.5,
        initial_capital=1_000.0,
    )
    return CandidateRunResult(
        spec=spec,
        symbols=tuple(dataset.symbols),
        comparison=comparison,
    )


def _publish(tmp_path):
    return publish_candidate_run(
        tmp_path / "candidate",
        _candidate_result(),
        build_candidate_run_provenance(research_context_digest="e" * 64),
    )


def test_global_context_candidate_artifact_persists_exact_observation_contract(
    tmp_path,
) -> None:
    artifact = _publish(tmp_path)
    summary = json.loads(artifact.summary_path.read_text(encoding="utf-8"))

    assert summary["candidate_config"]["ppo_global_context"] == (
        PPO_GLOBAL_BTC_REGIME_CONTEXT
    )
    assert summary["ppo_observation"] == ppo_observation_contract_payload(
        global_context=PPO_GLOBAL_BTC_REGIME_CONTEXT
    )

    loaded = load_candidate_run_artifact(artifact.root)
    assert loaded.summary == summary


def test_loader_rejects_global_context_candidate_with_baseline_observation_contract(
    tmp_path,
) -> None:
    artifact = _publish(tmp_path)
    summary = json.loads(artifact.summary_path.read_text(encoding="utf-8"))
    summary["candidate_config"]["ppo_global_context"] = PPO_GLOBAL_BTC_REGIME_CONTEXT
    summary["ppo_observation"] = ppo_observation_contract_payload()
    artifact.summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="PPO observation contract mismatch"):
        load_candidate_run_artifact(artifact.root)
