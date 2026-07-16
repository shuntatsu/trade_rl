from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.selection import PolicyMode
from trade_rl.integrations.sb3_serving import StableBaselines3PolicyLoader
from trade_rl.rl.actions import ACTION_SCHEMA
from trade_rl.rl.observations import OBSERVATION_SCHEMA
from trade_rl.serving.bundle import (
    ServingBundleManifest,
    load_serving_bundle,
    write_serving_bundle_manifest,
)


class FakeModel:
    def __init__(self, action: np.ndarray) -> None:
        self.action = action

    def predict(self, observation, deterministic=True):
        assert deterministic is True
        assert np.asarray(observation).shape == (5,)
        return self.action.copy(), None


def _bundle(root: Path, member_count: int = 2):
    root.mkdir()
    members: list[str] = []
    for index in range(member_count):
        relative = f"members/member-{index:03d}/policy.zip"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"member-{index}".encode())
        members.append(relative)
    loader_payload = {
        "algorithm": "ppo",
        "members": members,
        "schema_version": "sb3_policy_loader_v1",
    }
    (root / "policy-loader.json").write_text(
        json.dumps(loader_payload), encoding="utf-8"
    )
    action_names = ("fast_tilt", "slow_tilt", "risk_tilt", "alpha_scale")
    action_digest = content_digest({"names": action_names})
    manifest = ServingBundleManifest.build(
        root=root,
        dataset_id="a" * 64,
        action_schema=ACTION_SCHEMA,
        action_size=4,
        action_names=action_names,
        action_spec_digest=action_digest,
        observation_schema=OBSERVATION_SCHEMA,
        observation_size=5,
        environment_digest="b" * 64,
        initial_capital=100_000.0,
        policy_mode=PolicyMode.RESIDUAL_POLICY,
        policy_digest="c" * 64,
        signal_digest="d" * 64,
        selection_digest="e" * 64,
        release_digest=None,
        normalizer_digest=None,
        artifact_paths=tuple([*members, "policy-loader.json"]),
        created_at=datetime(2026, 7, 14, tzinfo=UTC),
    )
    write_serving_bundle_manifest(root, manifest)
    return load_serving_bundle(root)


def test_sb3_loader_averages_all_dynamic_action_members(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    actions = {
        "member-000": np.array([0.2, -0.2, 0.4, 0.0], dtype=np.float32),
        "member-001": np.array([0.4, 0.2, 0.0, 0.6], dtype=np.float32),
    }

    def load(algorithm: str, path: Path):
        assert algorithm == "ppo"
        return FakeModel(actions[path.parent.name])

    policy = StableBaselines3PolicyLoader(model_loader=load).load(bundle)

    np.testing.assert_allclose(
        policy.predict(np.zeros(5, dtype=np.float32)),
        np.array([0.3, 0.0, 0.2, 0.3], dtype=np.float32),
    )


def test_sb3_loader_rejects_member_not_declared_by_bundle(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle", member_count=1)
    loader_path = bundle.root / "policy-loader.json"
    payload = json.loads(loader_path.read_text(encoding="utf-8"))
    payload["members"].append("members/member-999/policy.zip")
    loader_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="bundle|member"):
        StableBaselines3PolicyLoader(model_loader=lambda algorithm, path: None).load(
            load_serving_bundle(bundle.root)
        )


def test_sb3_ensemble_prediction_fails_closed_on_bad_member(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    models = iter(
        (
            FakeModel(np.zeros(4, dtype=np.float32)),
            FakeModel(np.array([0.0, np.nan, 0.0, 0.0], dtype=np.float32)),
        )
    )
    policy = StableBaselines3PolicyLoader(
        model_loader=lambda algorithm, path: next(models)
    ).load(bundle)

    with pytest.raises(ValueError, match="finite|action"):
        policy.predict(np.zeros(5, dtype=np.float32))


def _structured_dataset(*, symbol_suffix: str = ""):
    from trade_rl.data.market import MarketDataset

    n = 128
    symbols = (f"BTCUSDT{symbol_suffix}", f"ETHUSDT{symbol_suffix}")
    names = ("15m__ret", "1h__ret", "4h__ret", "1d__ret")
    timestamps = np.datetime64("2026-01-01T00:15", "ns") + np.arange(
        n
    ) * np.timedelta64(15, "m")
    features = np.zeros((n, 2, len(names)), dtype=np.float32)
    for index in range(n):
        features[index] = index + np.arange(len(names), dtype=np.float32)
    close = 100.0 + np.arange(n, dtype=np.float64)[:, None] + np.arange(2)[None, :]
    open_price = np.vstack((close[:1], close[:-1]))
    return MarketDataset(
        dataset_id=("f" if symbol_suffix else "a") * 64,
        symbols=symbols,
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) + 1.0,
        low=np.minimum(open_price, close) - 1.0,
        close=close,
        volume=np.full_like(close, 1_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones_like(features, dtype=np.bool_),
        feature_staleness_hours=np.zeros_like(features, dtype=np.float32),
        feature_names=names,
        global_feature_names=("regime",),
        periods_per_year=35_040,
    )


class StructuredFakeModel:
    def __init__(self, action: np.ndarray, observation_space) -> None:
        self.action = action
        self.observation_space = observation_space
        self.last_observation = None

    def predict(self, observation, deterministic=True):
        assert deterministic is True
        assert isinstance(observation, dict)
        self.last_observation = observation
        return self.action.copy(), None


def _structured_bundle(root: Path):
    from gymnasium import spaces

    from trade_rl.rl.actions import ACTION_SCHEMA
    from trade_rl.rl.normalization import ObservationNormalizer
    from trade_rl.rl.observations import ObservationBuilder
    from trade_rl.rl.sequence_normalization import SequenceFeatureNormalizer
    from trade_rl.rl.sequence_observations import (
        SEQUENCE_OBSERVATION_SCHEMA,
        SequenceObservationBuilder,
        SequenceWindowSpec,
    )
    from trade_rl.serving.normalizer import write_observation_normalizer
    from trade_rl.serving.sequence_normalizer import write_sequence_feature_normalizer

    dataset = _structured_dataset()
    builder = SequenceObservationBuilder(
        windows=(
            SequenceWindowSpec("15m", 2),
            SequenceWindowSpec("1h", 2),
            SequenceWindowSpec("4h", 2),
            SequenceWindowSpec("1d", 2),
        )
    )
    layout = ObservationBuilder(
        action_size=2, n_factors=0, finite_horizon=False
    ).layout(dataset)
    normalizer = ObservationNormalizer(
        mean=np.zeros(layout.size),
        scale=np.ones(layout.size),
        train_start=0,
        train_end=8,
        dataset_id=dataset.dataset_id,
        source_dataset_id=dataset.dataset_id,
        observation_schema_digest=ObservationBuilder(
            action_size=2, n_factors=0, finite_horizon=False
        ).schema_digest(dataset),
        action_spec_digest="b" * 64,
    )
    feature_names = {
        timeframe: (f"{timeframe}__ret",) for timeframe in ("15m", "1h", "4h", "1d")
    }
    sequence_normalizer = SequenceFeatureNormalizer(
        feature_names=feature_names,
        center={key: np.zeros(1) for key in feature_names},
        scale={key: np.ones(1) for key in feature_names},
        sample_count={key: np.full(1, 8, dtype=np.int64) for key in feature_names},
        train_start=0,
        train_end=8,
        dataset_id=dataset.dataset_id,
        source_dataset_id=dataset.dataset_id,
        sequence_schema_digest=builder.layout_digest(dataset),
    )
    root.mkdir()
    members = []
    for index in range(2):
        relative = f"members/member-{index:03d}/policy.zip"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"member-{index}".encode())
        members.append(relative)
    write_observation_normalizer(root, normalizer)
    write_sequence_feature_normalizer(root, sequence_normalizer)
    (root / "dataset-reference.json").write_text(
        json.dumps(
            {
                "bar_hours": dataset.bar_hours,
                "dataset_id": dataset.dataset_id,
                "feature_config_digest": dataset.feature_config_digest,
                "feature_names": list(dataset.feature_names),
                "global_feature_names": list(dataset.global_feature_names),
                "schema_version": "dataset_reference_v3",
                "symbols": list(dataset.symbols),
            }
        ),
        encoding="utf-8",
    )
    (root / "environment.json").write_text(
        json.dumps(
            {
                "action": {
                    "alpha_enabled": False,
                    "n_factors": 0,
                    "risk_tilt_enabled": False,
                    "mode": "target_weight",
                    "target_weight_count": 2,
                },
                "environment": {
                    "finite_horizon_observation": False,
                    "structured_sequence_observation": True,
                    "sequence_windows": [["15m", 2], ["1h", 2], ["4h", 2], ["1d", 2]],
                },
                "schema_version": "training_environment_v2",
            }
        ),
        encoding="utf-8",
    )
    (root / "policy-loader.json").write_text(
        json.dumps(
            {
                "algorithm": "ppo",
                "dataset_reference": "dataset-reference.json",
                "environment": "environment.json",
                "members": members,
                "normalizer": "normalizer.json",
                "observation_mode": "structured_sequence",
                "schema_version": "sb3_policy_loader_v2",
                "sequence_normalizer": "sequence-normalizer.json",
            }
        ),
        encoding="utf-8",
    )
    sequence_spaces: dict[str, spaces.Space] = {
        "decision_index": spaces.Box(
            0, n := dataset.n_bars - 1, shape=(1,), dtype=np.int64
        ),
        "current_snapshot": spaces.Box(
            -np.inf, np.inf, shape=(2, 4 * dataset.n_features), dtype=np.float32
        ),
        "asset_state": spaces.Box(
            -np.inf,
            np.inf,
            shape=(2, layout.per_symbol_width - 4 * dataset.n_features),
            dtype=np.float32,
        ),
        "global_state": spaces.Box(
            -np.inf, np.inf, shape=(layout.global_width,), dtype=np.float32
        ),
        "active": spaces.Box(0, 1, shape=(2,), dtype=np.float32),
    }
    del n
    for timeframe in ("15m", "1h", "4h", "1d"):
        shape = (2, 2, 1)
        sequence_spaces[f"sequence_{timeframe}_values"] = spaces.Box(
            -np.inf, np.inf, shape=shape, dtype=np.float16
        )
        sequence_spaces[f"sequence_{timeframe}_available"] = spaces.Box(
            0, 1, shape=shape, dtype=np.uint8
        )
        sequence_spaces[f"sequence_{timeframe}_staleness"] = spaces.Box(
            0, np.inf, shape=shape, dtype=np.float16
        )
    observation_space = spaces.Dict(sequence_spaces)
    action_names = ("target_weight:BTCUSDT", "target_weight:ETHUSDT")
    manifest = ServingBundleManifest.build(
        root=root,
        dataset_id=dataset.dataset_id,
        action_schema=ACTION_SCHEMA,
        action_size=2,
        action_names=action_names,
        action_spec_digest="b" * 64,
        observation_schema=SEQUENCE_OBSERVATION_SCHEMA,
        observation_size=sum(
            int(np.prod(space.shape)) for space in sequence_spaces.values()
        ),
        environment_digest="c" * 64,
        initial_capital=100_000.0,
        policy_mode=PolicyMode.RESIDUAL_POLICY,
        policy_digest="d" * 64,
        signal_digest="e" * 64,
        selection_digest="1" * 64,
        release_digest=None,
        normalizer_digest=normalizer.digest,
        artifact_paths=tuple(
            [
                *members,
                "dataset-reference.json",
                "environment.json",
                "normalizer.json",
                "policy-loader.json",
                "sequence-normalizer.json",
            ]
        ),
        created_at=datetime(2026, 7, 16, tzinfo=UTC),
    )
    write_serving_bundle_manifest(root, manifest)
    return load_serving_bundle(root), dataset, layout, observation_space


def test_structured_sb3_loader_rebuilds_native_sequence_observation(
    tmp_path: Path,
) -> None:
    bundle, dataset, layout, observation_space = _structured_bundle(
        tmp_path / "structured"
    )
    models = iter(
        (
            StructuredFakeModel(
                np.array([0.2, -0.2], dtype=np.float32), observation_space
            ),
            StructuredFakeModel(
                np.array([0.4, 0.2], dtype=np.float32), observation_space
            ),
        )
    )
    policy = StableBaselines3PolicyLoader(
        model_loader=lambda algorithm, path: next(models)
    ).load(bundle)
    action = policy.predict_from_dataset(
        dataset,
        index=100,
        current_flat=np.zeros(layout.size, dtype=np.float32),
    )
    np.testing.assert_allclose(action, np.array([0.3, 0.0], dtype=np.float32))
    assert policy.models[0].last_observation["decision_index"].dtype == np.int64
    assert policy.models[0].last_observation["sequence_1h_values"].shape == (2, 2, 1)


def test_structured_sb3_loader_accepts_new_content_but_rejects_schema_drift(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    bundle, dataset, layout, observation_space = _structured_bundle(
        tmp_path / "structured"
    )
    policy = StableBaselines3PolicyLoader(
        model_loader=lambda algorithm, path: StructuredFakeModel(
            np.zeros(2, dtype=np.float32), observation_space
        )
    ).load(bundle)
    rolling = replace(dataset, dataset_id="9" * 64, identity_payload_json=None)
    policy.predict_from_dataset(
        rolling,
        index=100,
        current_flat=np.zeros(layout.size, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="symbols|layout|feature"):
        policy.predict_from_dataset(
            _structured_dataset(symbol_suffix="-DRIFT"),
            index=100,
            current_flat=np.zeros(layout.size, dtype=np.float32),
        )


def test_serving_runtime_delegates_structured_dataset_prediction(
    tmp_path: Path,
) -> None:
    from trade_rl.serving.runtime import RuntimeIdentityContract, ServingRuntime

    bundle, dataset, layout, observation_space = _structured_bundle(
        tmp_path / "structured"
    )
    loader = StableBaselines3PolicyLoader(
        model_loader=lambda algorithm, path: StructuredFakeModel(
            np.array([0.1, -0.1], dtype=np.float32), observation_space
        )
    )
    runtime = ServingRuntime(
        loader,
        allow_unreleased=True,
        identity_contract=RuntimeIdentityContract(
            environment_digest=bundle.manifest.environment_digest,
            action_names=bundle.manifest.action_names,
            action_spec_digest=bundle.manifest.action_spec_digest,
            normalizer_digest=bundle.manifest.normalizer_digest,
        ),
    )
    runtime.activate(bundle.root)
    np.testing.assert_allclose(
        runtime.predict_from_dataset(
            dataset,
            index=100,
            current_flat=np.zeros(layout.size, dtype=np.float32),
        ),
        np.array([0.1, -0.1], dtype=np.float32),
    )
