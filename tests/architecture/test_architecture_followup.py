from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.rl.environment_episode import EpisodeContractSampler
from trade_rl.rl.environment_execution import EnvironmentExecutionCoordinator
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.execution_adapter import StatefulCompatibilityMarketExecutor
from trade_rl.telemetry import TrainingTelemetryRecord, TrainingTelemetryWriter
from trade_rl.telemetry.indexed_training import (
    IndexedTrainingTelemetryWriter,
    StrictTrainingTelemetryRecord,
)

ROOT = Path(__file__).resolve().parents[2]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _market(*, global_available: bool) -> MarketDataset:
    n_bars = 12
    close = np.full((n_bars, 1), 100.0, dtype=np.float64)
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTC",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n_bars, 1), 1_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        global_feature_available=np.full(
            (n_bars, 1), global_available, dtype=np.bool_
        ),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_package_initializers_do_not_replace_runtime_symbols() -> None:
    for path in (
        "trade_rl/simulation/__init__.py",
        "trade_rl/telemetry/__init__.py",
        "trade_rl/studio/__init__.py",
        "trade_rl/catalog/__init__.py",
    ):
        assert "setattr(" not in _source(path), path

    assert MarketExecutor is StatefulCompatibilityMarketExecutor
    assert TrainingTelemetryRecord is StrictTrainingTelemetryRecord
    assert TrainingTelemetryWriter is IndexedTrainingTelemetryWriter


def test_environment_execution_delegates_to_shared_target_helper() -> None:
    source = inspect.getsource(EnvironmentExecutionCoordinator.execute_target)

    assert "execute_target_statefully(" in source
    assert "reconcile_target(" not in source
    assert ".execute_orders(" not in source


@pytest.mark.parametrize("mode", ["regime_balanced", "stress_tail"])
def test_regime_episode_sampling_fails_when_feature_is_never_available(
    mode: str,
) -> None:
    dataset = _market(global_available=False)
    sampler = EpisodeContractSampler(
        dataset,
        ResidualMarketEnvConfig(
            episode_bars=2,
            decision_every=1,
            episode_sampling_mode=mode,
            regime_feature_index=0,
        ),
        minimum_start_index=1,
    )

    with pytest.raises(
        ValueError,
        match="episode sampling feature is unavailable for every valid start",
    ):
        sampler.sample({}, np.random.default_rng(7))


def test_catalog_has_single_canonical_json_and_sealed_test_sql_owners() -> None:
    contracts = _source("trade_rl/catalog/contracts.py")
    postgres = _source("trade_rl/catalog/postgres.py")
    sealed_store = _source("trade_rl/catalog/postgres_sealed_test.py")

    assert (
        "from trade_rl.domain.canonical_json import canonical_json_bytes" in contracts
    )
    assert "def canonical_json_bytes(" not in contracts
    assert "INSERT INTO catalog_sealed_test_access" not in postgres
    assert "PostgresSealedTestReservationStore" in postgres
    assert "INSERT INTO catalog_sealed_test_access" in sealed_store


def test_postgres_workflow_runs_on_exact_pr_head_and_main_push() -> None:
    workflow = _source(".github/workflows/postgres-catalog.yml")

    assert "push:" in workflow
    assert "branches:" in workflow
    assert "- main" in workflow
    assert "trade_rl/evaluation/walk_forward/**" in workflow
    assert "trade_rl/workflows/**" in workflow
    assert "ref: ${{ github.event.pull_request.head.sha || github.sha }}" in workflow
    assert "persist-credentials: false" in workflow
