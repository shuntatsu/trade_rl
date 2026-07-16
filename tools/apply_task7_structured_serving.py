from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing Task 7 anchor in {path}: {old[:160]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, block: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if marker not in text:
        target.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def add_tests() -> None:
    replace_once(
        "tests/workflows/test_training_run_config.py",
        "def test_sequence_training_rejects_flat_export_and_declares_serving_unsupported() -> (\n    None\n):",
        "def test_sequence_training_rejects_flat_export_and_declares_native_serving_supported() -> None:",
    )
    replace_once(
        "tests/workflows/test_training_run_config.py",
        '''    support = _serving_support_payload(config)
    assert support["status"] == "unsupported"
    assert "sequence" in str(support["reason"])
''',
        '''    support = _serving_support_payload(config)
    assert support == {
        "loader_schema": "sb3_policy_loader_v2",
        "observation_mode": "structured_sequence",
        "runtime": "native_sb3_structured_sequence_v1",
        "schema_version": "serving_support_v2",
        "status": "supported",
    }
''',
    )
    append_once(
        "tests/serving/test_sb3_loader.py",
        "test_structured_sb3_loader_rebuilds_native_sequence_observation",
        '''

def _structured_dataset(*, symbol_suffix: str = ""):
    from trade_rl.data.market import MarketDataset

    n = 16
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
        timeframe: (f"{timeframe}__ret",)
        for timeframe in ("15m", "1h", "4h", "1d")
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
                    "sequence_windows": [
                        ["15m", 2], ["1h", 2], ["4h", 2], ["1d", 2]
                    ],
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
        "decision_index": spaces.Box(0, n := dataset.n_bars - 1, shape=(1,), dtype=np.int64),
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
        observation_size=sum(int(np.prod(space.shape)) for space in sequence_spaces.values()),
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
            StructuredFakeModel(np.array([0.2, -0.2], dtype=np.float32), observation_space),
            StructuredFakeModel(np.array([0.4, 0.2], dtype=np.float32), observation_space),
        )
    )
    policy = StableBaselines3PolicyLoader(
        model_loader=lambda algorithm, path: next(models)
    ).load(bundle)
    action = policy.predict_from_dataset(
        dataset,
        index=8,
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
        index=8,
        current_flat=np.zeros(layout.size, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="symbols|layout|feature"):
        policy.predict_from_dataset(
            _structured_dataset(symbol_suffix="-DRIFT"),
            index=8,
            current_flat=np.zeros(layout.size, dtype=np.float32),
        )


def test_serving_runtime_delegates_structured_dataset_prediction(tmp_path: Path) -> None:
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
            index=8,
            current_flat=np.zeros(layout.size, dtype=np.float32),
        ),
        np.array([0.1, -0.1], dtype=np.float32),
    )
''',
    )


def add_implementation() -> None:
    (ROOT / "trade_rl/serving/sequence_normalizer.py").write_text(
        '''"""Canonical structured-sequence normalizer sidecars for serving."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.rl.sequence_normalization import SequenceFeatureNormalizer

SEQUENCE_NORMALIZER_ARTIFACT_NAME = "sequence-normalizer.json"


def write_sequence_feature_normalizer(
    root: Path,
    normalizer: SequenceFeatureNormalizer,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / SEQUENCE_NORMALIZER_ARTIFACT_NAME
    temporary = path.with_name(f".{path.name}.tmp")
    payload = {
        "center": {
            key: tuple(float(value) for value in normalizer.center[key])
            for key in normalizer.feature_names
        },
        "clip": normalizer.clip,
        "dataset_id": normalizer.dataset_id,
        "digest": normalizer.digest,
        "epsilon": normalizer.epsilon,
        "feature_names": dict(normalizer.feature_names),
        "minimum_samples_per_channel": normalizer.minimum_samples_per_channel,
        "sample_count": {
            key: tuple(int(value) for value in normalizer.sample_count[key])
            for key in normalizer.feature_names
        },
        "scale": {
            key: tuple(float(value) for value in normalizer.scale[key])
            for key in normalizer.feature_names
        },
        "schema_version": normalizer.schema_version,
        "sequence_schema_digest": normalizer.sequence_schema_digest,
        "source_dataset_id": normalizer.source_dataset_id,
        "train_range": [normalizer.train_start, normalizer.train_end],
    }
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(path)
    return path


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def load_sequence_feature_normalizer(root: Path) -> SequenceFeatureNormalizer:
    path = Path(root) / SEQUENCE_NORMALIZER_ARTIFACT_NAME
    if not path.is_file():
        raise ValueError("serving sequence normalizer sidecar is missing")
    raw = _mapping(json.loads(path.read_text(encoding="utf-8")), field="sequence normalizer")
    raw_names = _mapping(raw.get("feature_names"), field="feature_names")
    raw_center = _mapping(raw.get("center"), field="center")
    raw_scale = _mapping(raw.get("scale"), field="scale")
    raw_counts = _mapping(raw.get("sample_count"), field="sample_count")
    raw_range = raw.get("train_range")
    if not isinstance(raw_range, list) or len(raw_range) != 2:
        raise ValueError("sequence normalizer train_range must contain two integers")
    clocks = ("15m", "1h", "4h", "1d")
    try:
        normalizer = SequenceFeatureNormalizer(
            feature_names={
                key: tuple(str(value) for value in cast(list[object], raw_names[key]))
                for key in clocks
            },
            center={
                key: np.asarray(cast(list[float], raw_center[key]), dtype=np.float64)
                for key in clocks
            },
            scale={
                key: np.asarray(cast(list[float], raw_scale[key]), dtype=np.float64)
                for key in clocks
            },
            sample_count={
                key: np.asarray(cast(list[int], raw_counts[key]), dtype=np.int64)
                for key in clocks
            },
            train_start=int(raw_range[0]),
            train_end=int(raw_range[1]),
            dataset_id=str(raw["dataset_id"]),
            source_dataset_id=str(raw["source_dataset_id"]),
            sequence_schema_digest=str(raw["sequence_schema_digest"]),
            minimum_samples_per_channel=int(raw["minimum_samples_per_channel"]),
            clip=float(raw["clip"]),
            epsilon=float(raw.get("epsilon", 1e-8)),
            schema_version=str(raw["schema_version"]),
            digest=str(raw["digest"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"serving sequence normalizer sidecar is invalid: {error}") from error
    return normalizer


__all__ = [
    "SEQUENCE_NORMALIZER_ARTIFACT_NAME",
    "load_sequence_feature_normalizer",
    "write_sequence_feature_normalizer",
]
''',
        encoding="utf-8",
    )

    (ROOT / "trade_rl/integrations/sb3_serving.py").write_text(
        '''"""Stable-Baselines3 serving integration for flat and structured bundles."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np
from gymnasium import spaces

from trade_rl.data.market import MarketDataset
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import ObservationBuilder
from trade_rl.rl.sequence_observations import (
    SequenceObservationBuilder,
    SequenceWindowSpec,
    build_structured_policy_observation,
)
from trade_rl.serving.bundle import ServingBundle
from trade_rl.serving.sequence_normalizer import (
    SEQUENCE_NORMALIZER_ARTIFACT_NAME,
    load_sequence_feature_normalizer,
)

SB3_POLICY_LOADER_NAME: Final = "policy-loader.json"
SB3_POLICY_LOADER_SCHEMA: Final = "sb3_policy_loader_v1"
SB3_STRUCTURED_POLICY_LOADER_SCHEMA: Final = "sb3_policy_loader_v2"
_SUPPORTED_ALGORITHMS: Final = frozenset({"ppo", "sac", "td3", "tqc"})


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _safe_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field} contains an unsafe member path")
    return path.as_posix()


def _validated_action(raw: object, *, action_size: int, member_index: int) -> np.ndarray:
    action = np.asarray(raw, dtype=np.float32).reshape(-1)
    if action.shape != (action_size,):
        raise ValueError(f"SB3 ensemble member {member_index} action shape mismatch")
    if not np.isfinite(action).all():
        raise ValueError(f"SB3 ensemble member {member_index} action must be finite")
    if np.any(action < -1.0) or np.any(action > 1.0):
        raise ValueError(f"SB3 ensemble member {member_index} action violates bounds")
    return action


def _mean_action(actions: list[np.ndarray]) -> np.ndarray:
    averaged = np.mean(np.stack(actions, axis=0), axis=0, dtype=np.float64)
    if not np.isfinite(averaged).all():
        raise ValueError("SB3 ensemble mean action must be finite")
    return np.asarray(averaged, dtype=np.float32)


class _SB3EnsemblePolicy:
    def __init__(self, models: tuple[Any, ...], *, observation_size: int, action_size: int) -> None:
        if not models:
            raise ValueError("SB3 ensemble must contain at least one member")
        self.models = models
        self.observation_size = observation_size
        self.action_size = action_size

    def predict(self, observation: np.ndarray) -> np.ndarray:
        vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        if vector.shape != (self.observation_size,) or not np.isfinite(vector).all():
            raise ValueError("observation does not match the SB3 ensemble contract")
        actions: list[np.ndarray] = []
        for member_index, model in enumerate(self.models):
            try:
                raw, _ = model.predict(vector, deterministic=True)
            except Exception as error:
                raise ValueError(
                    f"SB3 ensemble member {member_index} prediction failed"
                ) from error
            actions.append(
                _validated_action(raw, action_size=self.action_size, member_index=member_index)
            )
        return _mean_action(actions)


class _SB3StructuredSequenceEnsemblePolicy:
    """Reconstruct the exact training Dict observation from a rolling dataset."""

    def __init__(
        self,
        models: tuple[Any, ...],
        *,
        action_size: int,
        dataset_reference: Mapping[str, object],
        builder: SequenceObservationBuilder,
        normalizer: ObservationNormalizer,
        sequence_normalizer: Any,
        n_factors: int,
        finite_horizon: bool,
    ) -> None:
        if not models:
            raise ValueError("SB3 structured ensemble must contain at least one member")
        first_space = getattr(models[0], "observation_space", None)
        if not isinstance(first_space, spaces.Dict):
            raise ValueError("structured SB3 member must expose a Dict observation space")
        self.models = models
        self.action_size = action_size
        self.dataset_reference = dict(dataset_reference)
        self.builder = builder
        self.normalizer = normalizer
        self.sequence_normalizer = sequence_normalizer
        self.n_factors = n_factors
        self.finite_horizon = finite_horizon
        self.observation_space = first_space
        expected = {
            key: (space.shape, np.dtype(space.dtype))
            for key, space in first_space.spaces.items()
        }
        for model in models[1:]:
            candidate = getattr(model, "observation_space", None)
            if not isinstance(candidate, spaces.Dict):
                raise ValueError("all structured SB3 members must expose Dict spaces")
            observed = {
                key: (space.shape, np.dtype(space.dtype))
                for key, space in candidate.spaces.items()
            }
            if observed != expected:
                raise ValueError("structured SB3 member observation spaces disagree")

    def _validate_dataset(self, dataset: MarketDataset) -> None:
        expected_symbols = tuple(self.dataset_reference.get("symbols", ()))
        expected_features = tuple(self.dataset_reference.get("feature_names", ()))
        expected_globals = tuple(self.dataset_reference.get("global_feature_names", ()))
        expected_bar_hours = self.dataset_reference.get("bar_hours")
        if dataset.symbols != expected_symbols:
            raise ValueError("serving rolling dataset symbols do not match training")
        if dataset.feature_names != expected_features:
            raise ValueError("serving rolling dataset feature order does not match training")
        if dataset.global_feature_names != expected_globals:
            raise ValueError("serving rolling dataset global feature order does not match training")
        if (
            isinstance(expected_bar_hours, bool)
            or not isinstance(expected_bar_hours, (int, float))
            or not math.isclose(
                dataset.bar_hours,
                float(expected_bar_hours),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("serving rolling dataset bar cadence does not match training")
        if self.builder.layout_digest(dataset) != self.sequence_normalizer.sequence_schema_digest:
            raise ValueError("serving rolling dataset sequence layout does not match training")

    def _validate_observation(
        self, observation: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        expected_keys = set(self.observation_space.spaces)
        if set(observation) != expected_keys:
            raise ValueError("structured observation keys do not match policy contract")
        result: dict[str, np.ndarray] = {}
        for key, space in self.observation_space.spaces.items():
            array = np.asarray(observation[key], dtype=space.dtype)
            if array.shape != space.shape or not np.isfinite(array).all():
                raise ValueError(f"structured observation component {key} is invalid")
            result[key] = array
        return result

    def smoke_observation(self) -> dict[str, np.ndarray]:
        return {
            key: np.zeros(space.shape, dtype=space.dtype)
            for key, space in self.observation_space.spaces.items()
        }

    def predict(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        structured = self._validate_observation(observation)
        actions: list[np.ndarray] = []
        for member_index, model in enumerate(self.models):
            try:
                raw, _ = model.predict(structured, deterministic=True)
            except Exception as error:
                raise ValueError(
                    f"SB3 ensemble member {member_index} prediction failed"
                ) from error
            actions.append(
                _validated_action(raw, action_size=self.action_size, member_index=member_index)
            )
        return _mean_action(actions)

    def predict_from_dataset(
        self,
        dataset: MarketDataset,
        *,
        index: int,
        current_flat: np.ndarray,
    ) -> np.ndarray:
        self._validate_dataset(dataset)
        layout = ObservationBuilder(
            action_size=self.action_size,
            n_factors=self.n_factors,
            finite_horizon=self.finite_horizon,
        ).layout(dataset)
        raw = np.asarray(current_flat, dtype=np.float32).reshape(-1)
        if raw.shape != (layout.size,) or not np.isfinite(raw).all():
            raise ValueError("current flat observation does not match serving layout")
        current = self.normalizer.transform(raw)
        sequence = self.builder.build(dataset, index=index)
        structured = build_structured_policy_observation(
            sequence=sequence,
            current_flat=current,
            layout=layout,
            n_features=dataset.n_features,
            sequence_normalizer=self.sequence_normalizer,
        )
        structured["decision_index"] = np.asarray([index], dtype=np.int64)
        return self.predict(structured)


class StableBaselines3PolicyLoader:
    """Load every declared SB3 member and fail closed on partial ensembles."""

    def __init__(self, *, model_loader: Callable[[str, Path], Any] | None = None) -> None:
        self.model_loader = model_loader or self._load_model

    @staticmethod
    def _load_model(algorithm: str, path: Path) -> Any:
        if algorithm == "ppo":
            from stable_baselines3 import PPO

            return PPO.load(str(path), device="cpu")
        if algorithm == "sac":
            from stable_baselines3 import SAC

            return SAC.load(str(path), device="cpu")
        if algorithm == "td3":
            from stable_baselines3 import TD3

            return TD3.load(str(path), device="cpu")
        if algorithm == "tqc":
            from sb3_contrib import TQC

            return TQC.load(str(path), device="cpu")
        raise ValueError("unsupported Stable-Baselines3 algorithm")

    def load(self, bundle: ServingBundle) -> Any:
        manifest_path = bundle.root / SB3_POLICY_LOADER_NAME
        declared_files = {item.path for item in bundle.manifest.files}
        if SB3_POLICY_LOADER_NAME not in declared_files:
            raise ValueError("serving bundle does not declare policy-loader.json")
        if not manifest_path.is_file():
            raise ValueError("serving bundle policy loader manifest is missing")
        payload = _mapping(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            field="SB3 policy loader manifest",
        )
        schema = payload.get("schema_version")
        if schema not in {SB3_POLICY_LOADER_SCHEMA, SB3_STRUCTURED_POLICY_LOADER_SCHEMA}:
            raise ValueError("unsupported SB3 policy loader schema")
        algorithm_value = payload.get("algorithm")
        if not isinstance(algorithm_value, str):
            raise ValueError("SB3 policy loader algorithm must be a string")
        algorithm = algorithm_value.lower()
        if algorithm not in _SUPPORTED_ALGORITHMS:
            raise ValueError("SB3 policy loader algorithm is unsupported")
        raw_members = payload.get("members")
        if not isinstance(raw_members, list) or not raw_members:
            raise ValueError("SB3 policy loader members must be a non-empty list")
        members = tuple(
            _safe_relative_path(item, field=f"members[{index}]")
            for index, item in enumerate(raw_members)
        )
        if len(set(members)) != len(members):
            raise ValueError("SB3 policy loader members must be unique")
        models: list[Any] = []
        for relative in members:
            if relative not in declared_files:
                raise ValueError(
                    f"SB3 policy member is not declared by the serving bundle: {relative}"
                )
            if not relative.endswith("/policy.zip") and relative != "policy.zip":
                raise ValueError("SB3 policy member path must end with policy.zip")
            path = bundle.root / relative
            if not path.is_file():
                raise ValueError(f"SB3 policy member is missing: {relative}")
            model = self.model_loader(algorithm, path)
            if model is None or not callable(getattr(model, "predict", None)):
                raise ValueError(f"SB3 policy member could not be loaded: {relative}")
            models.append(model)
        if schema == SB3_POLICY_LOADER_SCHEMA:
            return _SB3EnsemblePolicy(
                tuple(models),
                observation_size=bundle.manifest.observation_size,
                action_size=bundle.manifest.action_size,
            )
        if payload.get("observation_mode") != "structured_sequence":
            raise ValueError("SB3 v2 loader requires structured_sequence mode")
        required_paths = {
            "dataset_reference": "dataset-reference.json",
            "environment": "environment.json",
            "normalizer": "normalizer.json",
            "sequence_normalizer": SEQUENCE_NORMALIZER_ARTIFACT_NAME,
        }
        resolved_paths: dict[str, str] = {}
        for field, expected in required_paths.items():
            relative = _safe_relative_path(payload.get(field), field=field)
            if relative != expected or relative not in declared_files:
                raise ValueError(f"structured SB3 loader {field} is not declared")
            resolved_paths[field] = relative
        if not isinstance(bundle.normalizer, ObservationNormalizer):
            raise ValueError("structured SB3 loader requires the flat normalizer sidecar")
        dataset_reference = _mapping(
            json.loads(
                (bundle.root / resolved_paths["dataset_reference"]).read_text(
                    encoding="utf-8"
                )
            ),
            field="dataset reference",
        )
        environment = _mapping(
            json.loads(
                (bundle.root / resolved_paths["environment"]).read_text(encoding="utf-8")
            ),
            field="training environment",
        )
        action = _mapping(environment.get("action"), field="environment.action")
        env_config = _mapping(
            environment.get("environment"), field="environment.environment"
        )
        if env_config.get("structured_sequence_observation") is not True:
            raise ValueError("structured loader environment does not enable sequences")
        raw_windows = env_config.get("sequence_windows")
        if not isinstance(raw_windows, list) or not raw_windows:
            raise ValueError("structured loader sequence windows are missing")
        windows: list[SequenceWindowSpec] = []
        for index, raw in enumerate(raw_windows):
            if not isinstance(raw, list) or len(raw) != 2:
                raise ValueError(f"sequence_windows[{index}] is invalid")
            windows.append(SequenceWindowSpec(str(raw[0]), int(raw[1])))
        n_factors = action.get("n_factors", 0)
        if isinstance(n_factors, bool) or not isinstance(n_factors, int) or n_factors < 0:
            raise ValueError("structured loader factor count is invalid")
        finite_horizon = env_config.get("finite_horizon_observation", False)
        if not isinstance(finite_horizon, bool):
            raise ValueError("structured loader finite horizon flag is invalid")
        sequence_normalizer = load_sequence_feature_normalizer(bundle.root)
        return _SB3StructuredSequenceEnsemblePolicy(
            tuple(models),
            action_size=bundle.manifest.action_size,
            dataset_reference=dataset_reference,
            builder=SequenceObservationBuilder(windows=tuple(windows)),
            normalizer=bundle.normalizer,
            sequence_normalizer=sequence_normalizer,
            n_factors=n_factors,
            finite_horizon=finite_horizon,
        )


__all__ = [
    "SB3_POLICY_LOADER_NAME",
    "SB3_POLICY_LOADER_SCHEMA",
    "SB3_STRUCTURED_POLICY_LOADER_SCHEMA",
    "StableBaselines3PolicyLoader",
]
''',
        encoding="utf-8",
    )

    replace_once(
        "trade_rl/workflows/training_run.py",
        '''def _policy_loader_payload(
    ensemble: PolicyEnsembleManifest,
    *,
    algorithm: str,
) -> dict[str, object]:
    return {
        "algorithm": algorithm,
        "members": tuple(
            f"members/member-{index:03d}/policy.zip"
            for index in range(ensemble.expected_members)
        ),
        "schema_version": "sb3_policy_loader_v1",
    }
''',
        '''def _policy_loader_payload(
    ensemble: PolicyEnsembleManifest,
    *,
    algorithm: str,
    structured_sequence: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "algorithm": algorithm,
        "members": tuple(
            f"members/member-{index:03d}/policy.zip"
            for index in range(ensemble.expected_members)
        ),
        "schema_version": (
            "sb3_policy_loader_v2" if structured_sequence else "sb3_policy_loader_v1"
        ),
    }
    if structured_sequence:
        payload.update(
            {
                "dataset_reference": "dataset-reference.json",
                "environment": "environment.json",
                "normalizer": "normalizer.json",
                "observation_mode": "structured_sequence",
                "sequence_normalizer": "sequence-normalizer.json",
            }
        )
    return payload
''',
    )
    replace_once(
        "trade_rl/workflows/training_run.py",
        '''    if config.training.sequence_encoder:
        return {
            "reason": (
                "structured sequence observations require a runtime-native dataset "
                "and sequence builder; the flat serving/export path is disabled"
            ),
            "schema_version": "serving_support_v1",
            "status": "unsupported",
        }
    return {
        "loader_schema": "sb3_policy_loader_v1",
        "schema_version": "serving_support_v1",
        "status": "supported",
    }
''',
        '''    if config.training.sequence_encoder:
        return {
            "loader_schema": "sb3_policy_loader_v2",
            "observation_mode": "structured_sequence",
            "runtime": "native_sb3_structured_sequence_v1",
            "schema_version": "serving_support_v2",
            "status": "supported",
        }
    return {
        "loader_schema": "sb3_policy_loader_v1",
        "observation_mode": "flat",
        "runtime": "flat_vector_v1",
        "schema_version": "serving_support_v2",
        "status": "supported",
    }
''',
    )
    replace_once(
        "trade_rl/workflows/training_run.py",
        '''                "schema_version": "dataset_reference_v2",
''',
        '''                "schema_version": "dataset_reference_v3",
''',
    )
    replace_once(
        "trade_rl/workflows/training_run.py",
        '''        if not config.training.sequence_encoder:
            _write_json(
                stage / "policy-loader.json",
                _policy_loader_payload(ensemble, algorithm=config.training.algorithm),
            )
''',
        '''        _write_json(
            stage / "policy-loader.json",
            _policy_loader_payload(
                ensemble,
                algorithm=config.training.algorithm,
                structured_sequence=config.training.sequence_encoder,
            ),
        )
''',
    )

    runtime = ROOT / "trade_rl/serving/runtime.py"
    text = runtime.read_text(encoding="utf-8")
    text = text.replace("from typing import Protocol\n", "from collections.abc import Mapping\nfrom typing import Any, Protocol\n", 1)
    text = text.replace(
        "from trade_rl.rl.observations import OBSERVATION_SCHEMA\n",
        "from trade_rl.rl.observations import OBSERVATION_SCHEMA\nfrom trade_rl.rl.sequence_observations import SEQUENCE_OBSERVATION_SCHEMA\n",
        1,
    )
    text = text.replace(
        "class LoadedPolicy(Protocol):\n    def predict(self, observation: np.ndarray) -> np.ndarray: ...\n",
        '''PolicyObservation = np.ndarray | Mapping[str, np.ndarray]


class LoadedPolicy(Protocol):
    def predict(self, observation: PolicyObservation) -> np.ndarray: ...
''',
        1,
    )
    text = text.replace(
        '''    def _predict_action(
        policy: LoadedPolicy,
        snapshot: RuntimeSnapshot,
        normalizer: ObservationNormalizer | None,
        observation: np.ndarray,
    ) -> np.ndarray:
        vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        if (
            vector.shape != (snapshot.observation_size,)
            or not np.isfinite(vector).all()
        ):
            raise ValueError("observation violates the active observation schema")
        policy_input = vector if normalizer is None else normalizer.transform(vector)
        raw_action = np.asarray(
            policy.predict(policy_input),
            dtype=np.float32,
        ).reshape(-1)
''',
        '''    def _predict_action(
        policy: LoadedPolicy,
        snapshot: RuntimeSnapshot,
        normalizer: ObservationNormalizer | None,
        observation: PolicyObservation,
    ) -> np.ndarray:
        if isinstance(observation, Mapping):
            if snapshot.observation_schema != SEQUENCE_OBSERVATION_SCHEMA:
                raise ValueError("structured observation violates the active schema")
            if not observation or any(
                np.asarray(value).size == 0 or not np.isfinite(np.asarray(value)).all()
                for value in observation.values()
            ):
                raise ValueError("structured observation violates the active schema")
            policy_input: Any = dict(observation)
        else:
            vector = np.asarray(observation, dtype=np.float32).reshape(-1)
            if (
                snapshot.observation_schema != OBSERVATION_SCHEMA
                or vector.shape != (snapshot.observation_size,)
                or not np.isfinite(vector).all()
            ):
                raise ValueError("observation violates the active observation schema")
            policy_input = vector if normalizer is None else normalizer.transform(vector)
        raw_action = np.asarray(
            policy.predict(policy_input),
            dtype=np.float32,
        ).reshape(-1)
''',
        1,
    )
    text = text.replace(
        '''        if manifest.observation_schema != OBSERVATION_SCHEMA:
            raise ValueError(
                "serving bundle observation schema does not match runtime schema"
            )
''',
        '''        if manifest.observation_schema not in {
            OBSERVATION_SCHEMA,
            SEQUENCE_OBSERVATION_SCHEMA,
        }:
            raise ValueError(
                "serving bundle observation schema does not match runtime schema"
            )
''',
        1,
    )
    text = text.replace(
        '''        self._predict_action(
            candidate_policy,
            candidate_snapshot,
            candidate_normalizer,
            np.zeros(candidate_snapshot.observation_size, dtype=np.float32),
        )
''',
        '''        smoke_factory = getattr(candidate_policy, "smoke_observation", None)
        smoke = (
            smoke_factory()
            if callable(smoke_factory)
            else np.zeros(candidate_snapshot.observation_size, dtype=np.float32)
        )
        self._predict_action(
            candidate_policy,
            candidate_snapshot,
            candidate_normalizer,
            smoke,
        )
''',
        1,
    )
    text = text.replace(
        '''    def predict(self, observation: np.ndarray) -> np.ndarray:
        vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("observation must be a non-empty finite vector")
        with self._lock:
            policy = self._policy
            snapshot = self._snapshot
            normalizer = self._normalizer
        if policy is None or snapshot is None:
            raise RuntimeError("serving runtime has no active policy")
        return self._predict_action(policy, snapshot, normalizer, vector)
''',
        '''    def predict(self, observation: PolicyObservation) -> np.ndarray:
        if not isinstance(observation, Mapping):
            vector = np.asarray(observation, dtype=np.float32).reshape(-1)
            if vector.size == 0 or not np.isfinite(vector).all():
                raise ValueError("observation must be a non-empty finite vector")
            observation = vector
        with self._lock:
            policy = self._policy
            snapshot = self._snapshot
            normalizer = self._normalizer
        if policy is None or snapshot is None:
            raise RuntimeError("serving runtime has no active policy")
        return self._predict_action(policy, snapshot, normalizer, observation)

    def predict_from_dataset(
        self,
        dataset: Any,
        *,
        index: int,
        current_flat: np.ndarray,
    ) -> np.ndarray:
        with self._lock:
            policy = self._policy
            snapshot = self._snapshot
        if policy is None or snapshot is None:
            raise RuntimeError("serving runtime has no active policy")
        predictor = getattr(policy, "predict_from_dataset", None)
        if not callable(predictor):
            raise RuntimeError("active policy does not support structured dataset serving")
        raw = np.asarray(
            predictor(dataset, index=index, current_flat=current_flat),
            dtype=np.float32,
        ).reshape(-1)
        if (
            raw.shape != (snapshot.action_size,)
            or not np.isfinite(raw).all()
            or np.any(raw < -1.0)
            or np.any(raw > 1.0)
        ):
            raise ValueError("policy output violates the residual action schema")
        return raw.copy()
''',
        1,
    )
    text = text.replace(
        '''__all__ = [
''',
        '''__all__ = [
''',
        1,
    ) if "__all__" in text else text
    runtime.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: apply_task7_structured_serving.py tests|implementation")
    if sys.argv[1] == "tests":
        add_tests()
    else:
        add_implementation()
