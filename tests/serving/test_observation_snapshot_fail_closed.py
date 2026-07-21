from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.serving.test_observation_parity import _RecordingModel, _bundle, _dataset
from trade_rl.integrations.sb3_serving import StableBaselines3PolicyLoader
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import ObservationBuilder, observation_passthrough_indices
from trade_rl.serving.runtime import RuntimeIdentityContract, ServingRuntime
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


class _DatasetPolicy:
    def __init__(self, action: np.ndarray) -> None:
        self.action = np.asarray(action, dtype=np.float32).reshape(-1)
        self.calls = 0

    def predict(self, observation):
        del observation
        return self.action.copy()

    def predict_from_dataset(self, dataset, *, index: int, current_flat: np.ndarray):
        assert dataset.dataset_id
        assert index >= 0
        assert np.asarray(current_flat).size > 0
        self.calls += 1
        return self.action.copy()


def _environment_snapshot():
    dataset = _dataset()
    builder = ObservationBuilder(action_size=3, n_factors=0, finite_horizon=False)
    layout = builder.layout(dataset)
    normalizer = ObservationNormalizer(
        mean=np.linspace(-0.2, 0.2, layout.size),
        scale=np.linspace(1.0, 2.0, layout.size),
        train_start=0,
        train_end=16,
        dataset_id=dataset.dataset_id,
        source_dataset_id=dataset.dataset_id,
        observation_schema_digest=builder.schema_digest(dataset),
        passthrough_indices=observation_passthrough_indices(
            dataset,
            action_size=3,
            n_factors=0,
            finite_horizon=False,
        ),
    )
    env = ResidualMarketEnv(
        dataset,
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=6)
        ),
        normalizer=normalizer,
        config=ResidualMarketEnvConfig(
            episode_bars=12,
            decision_every=1,
            signal_delay_decisions=1,
            initial_capital=10_000.0,
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )
    env.reset(seed=11, options={"start_idx": 16})
    env.step(np.asarray((0.4, -0.2, 0.1), dtype=np.float32))
    snapshot = env.observation_snapshot()
    contract = RuntimeIdentityContract(
        environment_digest=env.environment_digest,
        action_names=env.action_names,
        action_spec_digest=env.action_spec_digest,
        normalizer_digest=normalizer.digest,
    )
    return dataset, env, snapshot, normalizer, contract


def _activated_runtime(tmp_path: Path):
    dataset, env, snapshot, normalizer, contract = _environment_snapshot()
    bundle = _bundle(tmp_path / "bundle", env=env, normalizer=normalizer)
    actions = (
        np.asarray((0.2, -0.4, 0.6), dtype=np.float32),
        np.asarray((0.4, 0.2, 0.0), dtype=np.float32),
    )
    models = tuple(
        _RecordingModel(action, normalizer.size) for action in actions
    )
    iterator = iter(models)
    runtime = ServingRuntime(
        StableBaselines3PolicyLoader(
            model_loader=lambda algorithm, path: next(iterator)
        ),
        allow_unreleased=True,
        identity_contract=contract,
    )
    runtime.activate(bundle.root)
    return dataset, snapshot, runtime


def test_snapshot_prediction_requires_active_policy() -> None:
    dataset, _env, snapshot, _normalizer, contract = _environment_snapshot()
    runtime = ServingRuntime(
        allow_unreleased=True,
        identity_contract=contract,
    )

    with pytest.raises(RuntimeError, match="no active policy"):
        runtime.predict_from_observation_snapshot(dataset, snapshot)


def test_snapshot_prediction_rejects_normalization_mismatch(tmp_path: Path) -> None:
    dataset, snapshot, runtime = _activated_runtime(tmp_path)
    mismatched = replace(
        snapshot,
        normalized_observation=snapshot.normalized_observation + 0.25,
        snapshot_digest="",
    )

    with pytest.raises(ValueError, match="normalized observation parity"):
        runtime.predict_from_observation_snapshot(dataset, mismatched)


def test_snapshot_prediction_supports_bundle_without_normalizer(tmp_path: Path) -> None:
    dataset, snapshot, runtime = _activated_runtime(tmp_path)
    runtime._normalizer = None
    unnormalized = replace(
        snapshot,
        normalized_observation=snapshot.raw_observation.astype(np.float32),
        snapshot_digest="",
    )

    action = runtime.predict_from_observation_snapshot(dataset, unnormalized)

    assert action.shape == (3,)
    assert np.isfinite(action).all()


def test_snapshot_prediction_uses_dataset_native_policy(tmp_path: Path) -> None:
    dataset, snapshot, runtime = _activated_runtime(tmp_path)
    policy = _DatasetPolicy(np.asarray((0.1, -0.2, 0.3), dtype=np.float32))
    runtime._policy = policy

    action = runtime.predict_from_observation_snapshot(dataset, snapshot)

    np.testing.assert_allclose(action, (0.1, -0.2, 0.3), atol=1e-7)
    assert policy.calls == 1


def test_snapshot_prediction_rejects_invalid_dataset_native_action(
    tmp_path: Path,
) -> None:
    dataset, snapshot, runtime = _activated_runtime(tmp_path)
    runtime._policy = _DatasetPolicy(
        np.asarray((0.1, -0.2, 1.5), dtype=np.float32)
    )

    with pytest.raises(ValueError, match="action schema"):
        runtime.predict_from_observation_snapshot(dataset, snapshot)
