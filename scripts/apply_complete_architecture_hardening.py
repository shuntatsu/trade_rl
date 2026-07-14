from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "tests/architecture/test_complete_architecture_hardening.py"


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}: {old[:80]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, addition: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if marker in text:
        return
    target.write_text(text.rstrip() + "\n\n" + addition.lstrip(), encoding="utf-8")


def write_tests() -> None:
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PATH.write_text(
        '''from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.serving.helpers import create_bundle
from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    VolumeUnit,
)
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.rl.actions import ActionValidationMode
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import observation_passthrough_indices
from trade_rl.serving.bundle import load_serving_bundle
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def dataset(
    *,
    dataset_id: str = "a" * 64,
    volume: float = 1_000.0,
    volume_unit: VolumeUnit = VolumeUnit.BASE_ASSET,
    contract_multiplier: float = 1.0,
    feature_staleness: float = 0.0,
) -> MarketDataset:
    n_bars = 40
    close = np.full((n_bars, 1), 100.0)
    return MarketDataset(
        dataset_id=dataset_id,
        symbols=("ASSET",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 4), dtype=np.float32),
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n_bars, 1), volume),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("feature",),
        global_feature_names=("a", "b", "c", "d"),
        periods_per_year=8_760,
        feature_staleness=np.full((n_bars, 1, 1), feature_staleness),
        feature_staleness_hours=np.zeros((n_bars, 1, 1), dtype=np.float32),
        volume_units=(volume_unit,),
        contract_multipliers=np.array([contract_multiplier]),
    )


def env(
    value: MarketDataset,
    *,
    mode: ActionValidationMode = ActionValidationMode.CLIP,
    normalizer: ObservationNormalizer | None = None,
) -> ResidualMarketEnv:
    return ResidualMarketEnv(
        value,
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
        ),
        normalizer=normalizer,
        config=ResidualMarketEnvConfig(
            initial_capital=1_000.0,
            episode_bars=8,
            decision_every=1,
            action_validation_mode=mode,
            accept_legacy_actions=False,
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )


def test_action_spec_digest_binds_validation_mode() -> None:
    value = dataset()
    assert env(value, mode=ActionValidationMode.CLIP).action_spec_digest != env(
        value, mode=ActionValidationMode.STRICT
    ).action_spec_digest


def test_normalizer_source_dataset_binding_accepts_fold_view_identity() -> None:
    value = dataset()
    reference = env(value)
    size = int(reference.observation_space.shape[0])
    passthrough = observation_passthrough_indices(
        value,
        action_size=reference.action_spec.size,
        n_factors=reference.action_spec.n_factors,
        finite_horizon=False,
    )
    normalizer = ObservationNormalizer.fit(
        np.zeros((2, size), dtype=np.float32),
        train_start=0,
        train_end=2,
        passthrough_indices=passthrough,
        dataset_id="b" * 64,
        source_dataset_id=value.dataset_id,
        absolute_train_start=4,
        absolute_train_end=20,
        observation_schema_digest=reference.observation_builder.schema_digest(value),
        action_spec_digest=reference.action_spec_digest,
    )
    resolved = env(value, normalizer=normalizer)
    assert resolved.normalizer is normalizer


def test_observation_uses_canonical_normalized_feature_staleness() -> None:
    value = dataset(feature_staleness=0.75)
    environment = env(value)
    observation, _ = environment.reset(seed=0, options={"start_idx": 8})
    assert observation[2] == pytest.approx(0.75)


def test_quote_notional_volume_is_not_multiplied_by_price_twice() -> None:
    value = dataset(volume=10.0, volume_unit=VolumeUnit.QUOTE_NOTIONAL)
    result = MarketExecutor(value, ExecutionCostConfig.zero()).execute_interval(
        BookState.zero(1, 1_000.0, value.close[0]),
        np.array([1.0]),
        start_index=0,
        bars=1,
    )
    assert result.filled_turnover == pytest.approx(0.01)
    assert result.unfilled_turnover == pytest.approx(0.99)


def test_non_unit_contract_multiplier_fails_closed() -> None:
    with pytest.raises(ValueError, match="contract multiplier"):
        MarketExecutor(dataset(contract_multiplier=0.1), ExecutionCostConfig.zero())


def test_bundle_rejects_undeclared_files(tmp_path: Path) -> None:
    root = create_bundle(tmp_path / "bundle")
    (root / "undeclared.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="undeclared"):
        load_serving_bundle(root)


def test_bundle_rejects_declared_symlink_even_when_content_matches(tmp_path: Path) -> None:
    root = create_bundle(tmp_path / "bundle")
    declared = root / "signal.json"
    content = declared.read_bytes()
    outside = tmp_path / "outside-signal.json"
    outside.write_bytes(content)
    declared.unlink()
    declared.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink|escapes"):
        load_serving_bundle(root)


def test_builder_marks_unavailable_market_return_globals() -> None:
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        4
    ) * np.timedelta64(1, "h")
    close = np.arange(100.0, 104.0)
    raw = RawMarketSeries(
        timestamps=timestamps,
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.ones(4),
        funding_rate=np.zeros(4),
        tradable=np.ones(4, dtype=np.bool_),
    )
    built = MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="1h",
            features=(FeatureSpec("return", FeatureKind.LOG_RETURN),),
        )
    ).build(
        InMemoryMarketDataSource({"ASSET": raw}),
        (InstrumentContract("ASSET"),),
    )
    assert not built.global_feature_available[0, 2]
    assert not built.global_feature_available[0, 3]
    assert built.global_feature_staleness_hours[0, 2] >= 1.0
''',
        encoding="utf-8",
    )


def apply_implementation() -> None:
    replace_once(
        "trade_rl/rl/environment.py",
        '''        if normalizer is not None:
            if normalizer.size != layout.size:
                raise ValueError("normalizer size does not match observation layout")
            if (
                normalizer.dataset_id is not None
                and normalizer.dataset_id != dataset.dataset_id
            ):
                raise ValueError(
                    "normalizer dataset identity does not match environment"
                )
            if normalizer.observation_schema != OBSERVATION_SCHEMA:
                raise ValueError(
                    "normalizer observation schema does not match environment"
                )
            required_passthrough = set(
                observation_passthrough_indices(
                    dataset,
                    action_size=self.action_spec.size,
                    n_factors=self.action_spec.n_factors,
                    finite_horizon=self.config.finite_horizon_observation,
                )
            )
            if not required_passthrough.issubset(normalizer.passthrough_indices):
                raise ValueError(
                    "normalizer must preserve observation mask and activity indices"
                )
''',
        '''        if normalizer is not None:
            if normalizer.size != layout.size:
                raise ValueError("normalizer size does not match observation layout")
            bound_dataset_ids = {
                identity
                for identity in (normalizer.dataset_id, normalizer.source_dataset_id)
                if identity is not None
            }
            if bound_dataset_ids and dataset.dataset_id not in bound_dataset_ids:
                raise ValueError(
                    "normalizer dataset identity does not match environment"
                )
            if normalizer.observation_schema != OBSERVATION_SCHEMA:
                raise ValueError(
                    "normalizer observation schema does not match environment"
                )
            observation_schema_digest = self.observation_builder.schema_digest(dataset)
            if (
                normalizer.observation_schema_digest is not None
                and normalizer.observation_schema_digest != observation_schema_digest
            ):
                raise ValueError(
                    "normalizer observation schema digest does not match environment"
                )
            if (
                normalizer.action_spec_digest is not None
                and normalizer.action_spec_digest != self.action_spec_digest
            ):
                raise ValueError(
                    "normalizer action identity does not match environment"
                )
            for field_name, expected, observed in (
                (
                    "alpha artifact",
                    self.alpha_artifact_digest,
                    normalizer.alpha_artifact_digest,
                ),
                (
                    "factor artifact",
                    self.factor_artifact_digest,
                    normalizer.factor_artifact_digest,
                ),
            ):
                if observed is not None and observed != expected:
                    raise ValueError(
                        f"normalizer {field_name} identity does not match environment"
                    )
            required_passthrough = set(
                observation_passthrough_indices(
                    dataset,
                    action_size=self.action_spec.size,
                    n_factors=self.action_spec.n_factors,
                    finite_horizon=self.config.finite_horizon_observation,
                )
            )
            if not required_passthrough.issubset(normalizer.passthrough_indices):
                raise ValueError(
                    "normalizer must preserve observation mask and activity indices"
                )
''',
    )
    replace_once(
        "trade_rl/rl/environment.py",
        '''                "n_factors": self.action_spec.n_factors,
                "names": self.action_spec.names,
            }
''',
        '''                "n_factors": self.action_spec.n_factors,
                "names": self.action_spec.names,
                "validation_mode": ActionValidationMode(
                    self.action_spec.validation_mode
                ).value,
            }
''',
    )
    replace_once(
        "trade_rl/rl/observations.py",
        '''def _feature_staleness(dataset: MarketDataset, index: int) -> np.ndarray:
    staleness = dataset.resolved_array("feature_staleness_hours")[index].astype(
        np.float64,
        copy=False,
    )
    available = dataset.feature_available[index]
    return np.where(available, staleness, np.maximum(staleness, 1.0))
''',
        '''def _feature_staleness(dataset: MarketDataset, index: int) -> np.ndarray:
    staleness = dataset.resolved_array("feature_staleness")[index].astype(
        np.float64,
        copy=False,
    )
    available = dataset.feature_available[index]
    return np.where(available, staleness, np.maximum(staleness, 1.0))
''',
    )
    replace_once(
        "trade_rl/data/builder.py",
        '''        global_features = np.zeros(
            (n_bars, len(self.config.global_feature_names)), dtype=np.float64
        )
        global_features[:, 0] = symbol_active.mean(axis=1)
        observable_tradable = tradable & information_available
        global_features[:, 1] = observable_tradable.mean(axis=1)
        for index in range(n_bars):
            sample = one_bar_returns[index, one_bar_available[index]]
            if sample.size:
                global_features[index, 2] = float(np.mean(sample))
                global_features[index, 3] = float(np.std(sample))
''',
        '''        global_features = np.zeros(
            (n_bars, len(self.config.global_feature_names)), dtype=np.float64
        )
        global_feature_available = np.ones_like(global_features, dtype=np.bool_)
        global_feature_staleness = np.zeros_like(global_features, dtype=np.float32)
        global_feature_missing_reason = np.zeros_like(global_features, dtype=np.int16)
        global_features[:, 0] = symbol_active.mean(axis=1)
        observable_tradable = tradable & information_available
        global_features[:, 1] = observable_tradable.mean(axis=1)
        for index in range(n_bars):
            sample = one_bar_returns[index, one_bar_available[index]]
            if sample.size:
                global_features[index, 2] = float(np.mean(sample))
                global_features[index, 3] = float(np.std(sample))
            else:
                global_feature_available[index, 2:4] = False
                global_feature_staleness[index, 2:4] = 1.0
                global_feature_missing_reason[index, 2:4] = 1
''',
    )
    replace_once(
        "trade_rl/data/builder.py",
        '''            features=features,
            global_features=global_features,
            open=open_price,
''',
        '''            features=features,
            global_features=global_features,
            global_feature_available=global_feature_available,
            global_feature_staleness_hours=global_feature_staleness,
            global_feature_missing_reason=global_feature_missing_reason,
            open=open_price,
''',
    )
    replace_once(
        "trade_rl/data/identity.py",
        '''    "global_features",
    "open",
''',
        '''    "global_features",
    "global_feature_available",
    "global_feature_staleness_hours",
    "global_feature_missing_reason",
    "open",
''',
    )
    replace_once(
        "trade_rl/data/builder.py",
        '''                "global_features": global_features,
                "open": open_price,
''',
        '''                "global_features": global_features,
                "global_feature_available": global_feature_available,
                "global_feature_staleness_hours": global_feature_staleness,
                "global_feature_missing_reason": global_feature_missing_reason,
                "open": open_price,
''',
    )
    replace_once(
        "trade_rl/data/market.py",
        '''                    "global_features": global_features,
                    "open": open_price,
''',
        '''                    "global_features": global_features,
                    "global_feature_available": global_available,
                    "global_feature_staleness_hours": global_staleness,
                    "global_feature_missing_reason": global_missing_reason,
                    "open": open_price,
''',
    )
    replace_once(
        "trade_rl/simulation/execution.py",
        '''from trade_rl.data.market import MarketDataset
''',
        '''from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
''',
    )
    replace_once(
        "trade_rl/simulation/execution.py",
        '''        self.dataset = dataset
        self.cost = cost or ExecutionCostConfig()
        self._rng = np.random.default_rng(self.cost.random_seed)
''',
        '''        self.dataset = dataset
        self.cost = cost or ExecutionCostConfig()
        if not np.allclose(dataset.contract_multipliers, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(
                "non-unit contract multiplier accounting is not supported; "
                "execution fails closed"
            )
        self._rng = np.random.default_rng(self.cost.random_seed)
''',
    )
    replace_once(
        "trade_rl/simulation/execution.py",
        '''    def _constrain_borrow(
''',
        '''    def _capacity_notional(
        self,
        prices: np.ndarray,
        capacity_volume: np.ndarray,
    ) -> np.ndarray:
        result = np.empty_like(prices, dtype=np.float64)
        for index, unit in enumerate(self.dataset.volume_units):
            resolved = VolumeUnit(unit)
            if resolved is VolumeUnit.QUOTE_NOTIONAL:
                result[index] = capacity_volume[index]
            else:
                result[index] = prices[index] * capacity_volume[index]
        return result

    def _constrain_borrow(
''',
    )
    replace_once(
        "trade_rl/simulation/execution.py",
        '''        capacity_notional = price_vector * capacity_volume_vector
''',
        '''        capacity_notional = self._capacity_notional(
            price_vector,
            capacity_volume_vector,
        )
''',
    )
    replace_once(
        "trade_rl/serving/bundle.py",
        '''def load_serving_bundle(root: Path) -> ServingBundle:
    manifest_path = root / BUNDLE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"serving bundle manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _parse_manifest(_mapping(payload, field="bundle manifest"))
    for file in manifest.files:
        path = root / file.path
        if not path.is_file():
            raise ValueError(f"bundle artifact is missing: {file.path}")
        if path.stat().st_size != file.size_bytes:
            raise ValueError(f"bundle artifact size mismatch: {file.path}")
        if _file_digest(path) != file.digest:
            raise ValueError(f"bundle artifact digest mismatch: {file.path}")
    return ServingBundle(root=root, manifest=manifest)
''',
        '''def load_serving_bundle(root: Path) -> ServingBundle:
    root = Path(root)
    manifest_path = root / BUNDLE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"serving bundle manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _parse_manifest(_mapping(payload, field="bundle manifest"))
    root_resolved = root.resolve()
    declared = {BUNDLE_MANIFEST_NAME}
    for file in manifest.files:
        path = root / file.path
        declared.add(file.path)
        if path.is_symlink():
            raise ValueError(f"bundle artifact cannot be a symlink: {file.path}")
        resolved = path.resolve()
        if not resolved.is_relative_to(root_resolved):
            raise ValueError(f"bundle artifact escapes bundle root: {file.path}")
        if not path.is_file():
            raise ValueError(f"bundle artifact is missing: {file.path}")
        if path.stat().st_size != file.size_bytes:
            raise ValueError(f"bundle artifact size mismatch: {file.path}")
        if _file_digest(path) != file.digest:
            raise ValueError(f"bundle artifact digest mismatch: {file.path}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    undeclared = sorted(actual - declared)
    missing = sorted(declared - actual)
    if undeclared:
        raise ValueError(f"serving bundle contains undeclared files: {undeclared}")
    if missing:
        raise ValueError(f"serving bundle is missing declared files: {missing}")
    return ServingBundle(root=root, manifest=manifest)
''',
    )
    replace_once(
        "trade_rl/workflows/market_walk_forward.py",
        '''        self._normalizers: dict[int, ObservationNormalizer] = {}
''',
        '''        self._normalizers: dict[tuple[int, str], ObservationNormalizer] = {}
''',
    )
    replace_once(
        "trade_rl/workflows/market_walk_forward.py",
        '''        existing = self._normalizers.get(request.fold_index)
        if existing is not None:
            return existing
        normalizer = _fit_normalizer(self.dataset, request.train, run)
        self._normalizers[request.fold_index] = normalizer
''',
        '''        key = (request.fold_index, request.configuration.name)
        existing = self._normalizers.get(key)
        if existing is not None:
            return existing
        normalizer = _fit_normalizer(self.dataset, request.train, run)
        self._normalizers[key] = normalizer
''',
    )
    replace_once(
        "trade_rl/workflows/market_walk_forward.py",
        '''    return ObservationNormalizer.fit(
        matrix,
        train_start=0,
        train_end=matrix.shape[0],
        passthrough_indices=passthrough,
        dataset_id=None,
    )
''',
        '''    return ObservationNormalizer.fit(
        matrix,
        train_start=0,
        train_end=matrix.shape[0],
        passthrough_indices=passthrough,
        dataset_id=training_dataset.dataset_id,
        source_dataset_id=dataset.dataset_id,
        absolute_train_start=train_range.start,
        absolute_train_end=train_range.stop,
        observation_schema_digest=env.observation_builder.schema_digest(
            training_dataset
        ),
        action_spec_digest=env.action_spec_digest,
        alpha_artifact_digest=(
            None if alpha_provider is None else alpha_provider.artifact_digest
        ),
        factor_artifact_digest=(
            None if factor_provider is None else factor_provider.artifact_digest
        ),
        candidate_config_digest=content_digest(run.digest_payload()),
    )
''',
    )
    replace_once(
        "trade_rl/workflows/market_walk_forward.py",
        '''        for fold, fold_result in zip(
            result.folds,
            result.fold_results,
            strict=True,
        ):
            payload = _fold_payload(
''',
        '''        for fold, fold_result in zip(
            result.folds,
            result.fold_results,
            strict=True,
        ):
            sealed_count = evaluator.outer_test_counts.get(fold.fold_index, 0)
            expected_count = (
                1 if fold_result.selection.selected_policy_digest is None else 2
            )
            if sealed_count != expected_count:
                raise RuntimeError(
                    "sealed outer test evaluation count violates the fold contract"
                )
            payload = _fold_payload(
''',
    )
    replace_once(
        "trade_rl/workflows/market_walk_forward.py",
        '''                sealed_test_evaluations=evaluator.outer_test_counts.get(
                    fold.fold_index, 0
                ),
''',
        '''                sealed_test_evaluations=sealed_count,
''',
    )
    replace_once(
        "trade_rl/workflows/market_walk_forward.py",
        '''        policy_digest = content_digest(
            {
                "policies": tuple(sorted(registry)),
                "schema_version": "walk_forward_policy_set_v1",
            }
        )
        environment_digest = content_digest(
            {
                "action": asdict(config.candidates[0].run.action),
                "environment": asdict(config.candidates[0].run.environment),
                "risk": asdict(config.candidates[0].run.risk),
                "reward": asdict(config.candidates[0].run.reward),
                "trend": asdict(config.candidates[0].run.trend),
            }
        )
''',
        '''        policy_digest = content_digest(
            {
                "policies": tuple(
                    {
                        "algorithm": record.algorithm,
                        "normalizer_digest": record.normalizer.digest,
                        "policy_digest": digest,
                        "run_config_digest": content_digest(record.run.digest_payload()),
                    }
                    for digest, record in sorted(registry.items())
                ),
                "schema_version": "walk_forward_policy_set_v2",
            }
        )
        environment_digest = content_digest(
            {
                "candidates": tuple(
                    {
                        "name": item.name,
                        "run_environment": {
                            "action": asdict(item.run.action),
                            "environment": asdict(item.run.environment),
                            "risk": asdict(item.run.risk),
                            "reward": asdict(item.run.reward),
                            "trend": asdict(item.run.trend),
                        },
                    }
                    for item in config.candidates
                ),
                "schema_version": "walk_forward_environment_set_v1",
            }
        )
''',
    )


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: apply_complete_architecture_hardening.py tests|implementation")
    if sys.argv[1] == "tests":
        write_tests()
    else:
        apply_implementation()


if __name__ == "__main__":
    main()
