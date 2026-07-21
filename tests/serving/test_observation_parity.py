from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.domain.selection import PolicyMode
from trade_rl.integrations.sb3_serving import StableBaselines3PolicyLoader
from trade_rl.rl.actions import ACTION_SCHEMA
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import OBSERVATION_SCHEMA, ObservationBuilder
from trade_rl.serving.bundle import (
    ServingBundleManifest,
    load_serving_bundle,
    write_serving_bundle_manifest,
)
from trade_rl.serving.normalizer import write_observation_normalizer
from trade_rl.serving.runtime import RuntimeIdentityContract, ServingRuntime
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def _dataset() -> MarketDataset:
    n = 80
    timestamps = np.datetime64("2026-01-01", "ns") + np.arange(n) * np.timedelta64(
        1, "h"
    )
    close = np.column_stack(
        (
            100.0 * np.exp(np.arange(n) * 0.002),
            90.0 * np.exp(-np.arange(n) * 0.001),
        )
    )
    open_price = np.vstack((close[:1], close[:-1]))
    features = np.empty((n, 2, 2), dtype=np.float32)
    features[:, :, 0] = np.arange(n, dtype=np.float32)[:, None] / 100.0
    features[:, :, 1] = np.asarray((1.0, -1.0), dtype=np.float32)
    available = np.ones_like(features, dtype=np.bool_)
    available[19, 1, 1] = False
    staleness = np.zeros_like(features, dtype=np.float32)
    staleness[19, 1, 1] = 3.0
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("UP", "DOWN"),
        timestamps=timestamps,
        features=features,
        global_features=np.sin(np.arange(n, dtype=np.float32)[:, None] / 8.0),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=available,
        feature_staleness_hours=staleness,
        feature_names=("momentum", "direction"),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def _book_vector(book) -> np.ndarray:
    return np.concatenate(
        (
            book.quantities,
            book.mark_prices,
            book.contract_multipliers,
            book.weights,
            np.asarray(
                (
                    book.cash,
                    book.portfolio_value,
                    book.peak_value,
                    book.max_drawdown,
                    book.turnover_total,
                    book.total_cost,
                    book.funding_pnl,
                    book.borrow_cost,
                    book.margin_used,
                    book.maintenance_margin,
                    book.maintenance_requirement,
                    book.margin_deficit,
                    float(book.insolvent),
                ),
                dtype=np.float64,
            ),
        )
    )


class _RecordingModel:
    def __init__(self, action: np.ndarray, observation_size: int) -> None:
        self.action = action
        self.observation_size = observation_size
        self.observations: list[np.ndarray] = []

    def predict(self, observation, deterministic=True):
        assert deterministic is True
        vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        assert vector.shape == (self.observation_size,)
        self.observations.append(vector.copy())
        return self.action.copy(), None


def _bundle(
    root: Path,
    *,
    env: ResidualMarketEnv,
    normalizer: ObservationNormalizer,
):
    root.mkdir()
    members: list[str] = []
    for index in range(2):
        relative = f"members/member-{index:03d}/policy.zip"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"member-{index}".encode())
        members.append(relative)
    (root / "policy-loader.json").write_text(
        json.dumps(
            {
                "algorithm": "ppo",
                "members": members,
                "schema_version": "sb3_policy_loader_v1",
            }
        ),
        encoding="utf-8",
    )
    write_observation_normalizer(root, normalizer)
    manifest = ServingBundleManifest.build(
        root=root,
        dataset_id=env.dataset.dataset_id,
        action_schema=ACTION_SCHEMA,
        action_size=env.action_spec.size,
        action_names=env.action_names,
        action_spec_digest=env.action_spec_digest,
        observation_schema=OBSERVATION_SCHEMA,
        observation_size=normalizer.size,
        environment_digest=env.environment_digest,
        initial_capital=env.initial_capital,
        policy_mode=PolicyMode.RESIDUAL_POLICY,
        policy_digest="c" * 64,
        signal_digest="d" * 64,
        selection_digest="e" * 64,
        training_run_digest="f" * 64,
        selection_proposal_digest="1" * 64,
        selection_authorization_digest="2" * 64,
        walk_forward_run_digest="3" * 64,
        gate_evidence_digest="4" * 64,
        confirmation_evidence_digest="5" * 64,
        normalizer_digest=normalizer.digest,
        artifact_paths=tuple([*members, "policy-loader.json", "normalizer.json"]),
        created_at=datetime(2026, 7, 21, tzinfo=UTC),
    )
    write_serving_bundle_manifest(root, manifest)
    return load_serving_bundle(root)


def test_real_environment_observation_matches_serving_members_and_ensemble(
    tmp_path: Path,
) -> None:
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
    env.reset(seed=7, options={"start_idx": 16})
    actions = (
        np.array((0.5, -0.2, 0.1), dtype=np.float32),
        np.array((0.3, 0.4, -0.1), dtype=np.float32),
        np.array((-0.2, 0.6, 0.2), dtype=np.float32),
    )
    for action in actions:
        env.step(action)

    snapshot = env.observation_snapshot()
    assert snapshot.index == 19
    assert snapshot.feature_names == dataset.feature_names
    np.testing.assert_array_equal(
        snapshot.availability_mask.astype(bool),
        dataset.feature_available[19].reshape(-1),
    )
    np.testing.assert_array_equal(
        snapshot.staleness,
        dataset.resolved_array("feature_staleness")[19].reshape(-1),
    )
    np.testing.assert_allclose(snapshot.hybrid_book_state, _book_vector(env.hybrid))
    np.testing.assert_allclose(snapshot.shadow_book_state, _book_vector(env.shadow))
    np.testing.assert_allclose(snapshot.previous_action, actions[-1])
    assert np.count_nonzero(snapshot.pending_target) > 0
    np.testing.assert_allclose(
        snapshot.normalized_observation,
        normalizer.transform(snapshot.raw_observation),
        rtol=0.0,
        atol=0.0,
    )
    assert np.count_nonzero(snapshot.raw_observation) > 0

    bundle = _bundle(tmp_path / "bundle", env=env, normalizer=normalizer)
    member_actions = (
        np.array((0.2, -0.4, 0.6), dtype=np.float32),
        np.array((0.4, 0.2, 0.0), dtype=np.float32),
    )
    models = tuple(
        _RecordingModel(action, layout.size) for action in member_actions
    )
    iterator = iter(models)
    loader = StableBaselines3PolicyLoader(
        model_loader=lambda algorithm, path: next(iterator)
    )
    runtime = ServingRuntime(
        loader,
        allow_unreleased=True,
        identity_contract=RuntimeIdentityContract(
            environment_digest=env.environment_digest,
            action_names=env.action_names,
            action_spec_digest=env.action_spec_digest,
            normalizer_digest=normalizer.digest,
        ),
    )
    runtime.activate(bundle.root)
    for model in models:
        model.observations.clear()  # discard activation smoke input

    ensemble = runtime.predict_from_observation_snapshot(dataset, snapshot)

    for model in models:
        assert len(model.observations) == 1
        np.testing.assert_allclose(
            model.observations[0],
            snapshot.normalized_observation,
            rtol=0.0,
            atol=0.0,
        )
    np.testing.assert_allclose(
        ensemble,
        np.mean(np.stack(member_actions), axis=0),
        rtol=0.0,
        atol=1e-7,
    )
