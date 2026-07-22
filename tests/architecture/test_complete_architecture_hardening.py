from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
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
from trade_rl.domain.selection import PolicyMode
from trade_rl.rl.actions import ACTION_SCHEMA, ActionSpec, ActionValidationMode
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import OBSERVATION_SCHEMA, observation_passthrough_indices
from trade_rl.serving.bundle import (
    ServingBundleManifest,
    load_serving_bundle,
    write_serving_bundle_manifest,
)
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy

ACTION_NAMES = ("fast_tilt", "slow_tilt", "risk_tilt")
ACTION_SPEC_DIGEST = content_digest({"names": ACTION_NAMES})


def create_bundle(root: Path) -> Path:
    root.mkdir(parents=True)
    artifact_paths = ("dataset.json", "signal.json", "selection.json")
    (root / "dataset.json").write_text('{"dataset":"a"}', encoding="utf-8")
    (root / "signal.json").write_text('{"signal":"rejected"}', encoding="utf-8")
    (root / "selection.json").write_text(
        '{"selection":"baseline_only"}', encoding="utf-8"
    )
    manifest = ServingBundleManifest.build(
        root=root,
        dataset_id="a" * 64,
        action_schema=ACTION_SCHEMA,
        action_size=len(ACTION_NAMES),
        action_names=ACTION_NAMES,
        action_spec_digest=ACTION_SPEC_DIGEST,
        observation_schema=OBSERVATION_SCHEMA,
        observation_size=5,
        environment_digest="d" * 64,
        initial_capital=250_000.0,
        policy_mode=PolicyMode.BASELINE_ONLY,
        policy_digest=None,
        signal_digest="b" * 64,
        selection_digest="c" * 64,
        release_digest=None,
        normalizer_digest=None,
        artifact_paths=artifact_paths,
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )
    write_serving_bundle_manifest(root, manifest)
    return root


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
    action_spec: ActionSpec | None = None,
) -> ResidualMarketEnv:
    return ResidualMarketEnv(
        value,
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
        ),
        normalizer=normalizer,
        action_spec=action_spec,
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
    assert (
        env(value, mode=ActionValidationMode.CLIP).action_spec_digest
        != env(value, mode=ActionValidationMode.STRICT).action_spec_digest
    )


def test_action_spec_digest_binds_risk_tilt_layout() -> None:
    value = dataset()
    with_risk = env(value)
    without_risk = env(value, action_spec=ActionSpec(risk_tilt_enabled=False))

    assert without_risk.action_names == ("fast_tilt", "slow_tilt")
    assert with_risk.action_spec_digest != without_risk.action_spec_digest
    assert with_risk.environment_digest != without_risk.environment_digest


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


def test_non_unit_contract_multiplier_uses_quantity_semantics() -> None:
    value = dataset(contract_multiplier=0.1)
    book = BookState.zero(
        1,
        1_000.0,
        value.close[0],
        contract_multipliers=value.contract_multipliers,
    )
    result = MarketExecutor(value, ExecutionCostConfig.zero()).execute_interval(
        book,
        np.array([0.5]),
        start_index=0,
        bars=1,
    )
    assert result.book.weights[0] == pytest.approx(0.5)


def test_bundle_rejects_undeclared_files(tmp_path: Path) -> None:
    root = create_bundle(tmp_path / "bundle")
    (root / "undeclared.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="undeclared"):
        load_serving_bundle(root)


def test_bundle_rejects_declared_symlink_even_when_content_matches(
    tmp_path: Path,
) -> None:
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
