from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.rl.actions import ActionSpec
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.rl.environment_observation_contract import (
    EnvironmentObservationContractBuilder,
)
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import (
    OBSERVATION_SCHEMA,
    ObservationBuilder,
    observation_passthrough_indices,
)
from trade_rl.rl.sequence_normalization import SequenceFeatureNormalizer
from trade_rl.rl.sequence_observations import (
    SEQUENCE_OBSERVATION_SCHEMA,
    SequenceObservationBuilder,
)


def _market() -> MarketDataset:
    n_bars = 120
    n_symbols = 2
    n_features = 4
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(15, "m")
    close = np.column_stack(
        (
            np.linspace(100.0, 130.0, n_bars),
            np.linspace(80.0, 95.0, n_bars),
        )
    )
    open_price = np.vstack((close[0], close[:-1]))
    features = np.zeros((n_bars, n_symbols, n_features), dtype=np.float32)
    for feature_index in range(n_features):
        features[:, :, feature_index] = (
            np.arange(n_bars, dtype=np.float32)[:, None] + feature_index
        )
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("A", "B"),
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) + 1.0,
        low=np.minimum(open_price, close) - 1.0,
        close=close,
        volume=np.full((n_bars, n_symbols), 1_000.0),
        funding_rate=np.zeros((n_bars, n_symbols)),
        tradable=np.ones((n_bars, n_symbols), dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, n_features), dtype=np.bool_),
        feature_names=("15m__ret", "1h__ret", "4h__ret", "1d__ret"),
        global_feature_names=("regime",),
        periods_per_year=35_040,
    )


def _config(*, structured: bool = False) -> ResidualMarketEnvConfig:
    return ResidualMarketEnvConfig(
        initial_capital=100_000.0,
        structured_sequence_observation=structured,
        sequence_windows=(
            (("15m", 2), ("1h", 2), ("4h", 2), ("1d", 2)) if structured else ()
        ),
    )


def _builder(
    dataset: MarketDataset,
    *,
    config: ResidualMarketEnvConfig | None = None,
    normalizer: ObservationNormalizer | None = None,
    sequence_normalizer: SequenceFeatureNormalizer | None = None,
    alpha_artifact_digest: str | None = None,
    factor_artifact_digest: str | None = None,
) -> EnvironmentObservationContractBuilder:
    return EnvironmentObservationContractBuilder(
        dataset,
        config or _config(),
        action_spec=ActionSpec(),
        normalizer=normalizer,
        sequence_normalizer=sequence_normalizer,
        alpha_artifact_digest=alpha_artifact_digest,
        factor_artifact_digest=factor_artifact_digest,
        action_spec_digest="b" * 64,
    )


def _normalizer(
    dataset: MarketDataset,
    *,
    size_delta: int = 0,
    dataset_id: str | None = None,
    observation_schema: str = OBSERVATION_SCHEMA,
    observation_schema_digest: str | None = None,
    action_spec_digest: str | None = "b" * 64,
    alpha_artifact_digest: str | None = None,
    factor_artifact_digest: str | None = None,
    preserve_passthrough: bool = True,
) -> ObservationNormalizer:
    action_spec = ActionSpec()
    observation_builder = ObservationBuilder(
        action_size=action_spec.size,
        n_factors=action_spec.n_factors,
        finite_horizon=False,
    )
    layout = observation_builder.layout(dataset)
    size = layout.size + size_delta
    passthrough = (
        observation_passthrough_indices(
            dataset,
            action_size=action_spec.size,
            n_factors=action_spec.n_factors,
            finite_horizon=False,
        )
        if preserve_passthrough and size_delta == 0
        else ()
    )
    return ObservationNormalizer(
        mean=np.zeros(size),
        scale=np.ones(size),
        train_start=0,
        train_end=1,
        passthrough_indices=passthrough,
        dataset_id=dataset.dataset_id if dataset_id is None else dataset_id,
        observation_schema=observation_schema,
        observation_schema_digest=(
            observation_builder.schema_digest(dataset)
            if observation_schema_digest is None
            else observation_schema_digest
        ),
        action_spec_digest=action_spec_digest,
        alpha_artifact_digest=alpha_artifact_digest,
        factor_artifact_digest=factor_artifact_digest,
    )


def _sequence_normalizer(
    dataset: MarketDataset,
    *,
    dataset_id: str | None = None,
    source_dataset_id: str | None = None,
    sequence_schema_digest: str | None = None,
) -> SequenceFeatureNormalizer:
    sequence_builder = SequenceObservationBuilder(
        windows=tuple(
            __import__(
                "trade_rl.rl.sequence_observations",
                fromlist=["SequenceWindowSpec"],
            ).SequenceWindowSpec(timeframe, 2)
            for timeframe in ("15m", "1h", "4h", "1d")
        )
    )
    names = {
        timeframe: (f"{timeframe}__ret",) for timeframe in ("15m", "1h", "4h", "1d")
    }
    return SequenceFeatureNormalizer(
        feature_names=names,
        center={timeframe: np.zeros(1) for timeframe in names},
        scale={timeframe: np.ones(1) for timeframe in names},
        train_start=0,
        train_end=1,
        dataset_id=dataset.dataset_id if dataset_id is None else dataset_id,
        source_dataset_id=(
            dataset.dataset_id if source_dataset_id is None else source_dataset_id
        ),
        sequence_schema_digest=(
            sequence_builder.layout_digest(dataset)
            if sequence_schema_digest is None
            else sequence_schema_digest
        ),
    )


def test_flat_contract_preserves_layout_spaces_and_identity() -> None:
    dataset = _market()
    contract = _builder(dataset).build(minimum_start_index=7)

    assert contract.observation_schema == OBSERVATION_SCHEMA
    assert (
        contract.observation_contract_digest
        == contract.observation_builder.schema_digest(dataset)
    )
    assert contract.asset_active_column == 4 * dataset.n_features
    assert contract.minimum_start_index == 7
    assert contract.sequence_observation_builder is None
    assert contract.sequence_policy_plane is None
    assert contract.sequence_layout_metadata is None
    assert isinstance(contract.observation_space, gym.spaces.Box)
    assert contract.observation_space.shape == (contract.layout.size,)
    assert contract.observation_space.dtype == np.dtype(np.float32)
    assert isinstance(contract.action_space, gym.spaces.Box)
    assert contract.action_space.shape == (ActionSpec().size,)
    assert contract.action_space.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(contract.action_space.low, -1.0)
    np.testing.assert_array_equal(contract.action_space.high, 1.0)


def test_structured_contract_preserves_components_and_minimum_index() -> None:
    dataset = _market()
    contract = _builder(dataset, config=_config(structured=True)).build(
        minimum_start_index=3
    )

    assert contract.observation_schema == SEQUENCE_OBSERVATION_SCHEMA
    assert contract.sequence_observation_builder is not None
    assert contract.sequence_policy_plane is not None
    assert contract.minimum_start_index == 96
    assert contract.sequence_layout_metadata == {
        "feature_counts": {"15m": 1, "1h": 1, "4h": 1, "1d": 1},
        "window_lengths": {"15m": 2, "1h": 2, "4h": 2, "1d": 2},
        "snapshot_width": 4 * dataset.n_features,
        "asset_state_width": contract.layout.per_symbol_width - 4 * dataset.n_features,
        "global_width": contract.layout.global_width,
        "n_symbols": dataset.n_symbols,
    }
    assert isinstance(contract.observation_space, gym.spaces.Dict)
    assert contract.observation_space["decision_index"].dtype == np.dtype(np.int64)
    for timeframe in ("15m", "1h", "4h", "1d"):
        expected_shape = (dataset.n_symbols, 2, 1)
        assert (
            contract.observation_space[f"sequence_{timeframe}_values"].shape
            == expected_shape
        )
        assert contract.observation_space[
            f"sequence_{timeframe}_values"
        ].dtype == np.dtype(np.float16)
        assert contract.observation_space[
            f"sequence_{timeframe}_available"
        ].dtype == np.dtype(np.uint8)
        assert contract.observation_space[
            f"sequence_{timeframe}_staleness"
        ].dtype == np.dtype(np.float16)


@pytest.mark.parametrize(
    ("normalizer_factory", "message"),
    (
        (lambda dataset: _normalizer(dataset, size_delta=-1), "normalizer size"),
        (
            lambda dataset: _normalizer(dataset, dataset_id="c" * 64),
            "normalizer dataset identity does not match environment",
        ),
        (
            lambda dataset: _normalizer(dataset, observation_schema="wrong"),
            "normalizer observation schema does not match environment",
        ),
        (
            lambda dataset: _normalizer(dataset, observation_schema_digest="c" * 64),
            "normalizer observation schema digest does not match environment",
        ),
        (
            lambda dataset: _normalizer(dataset, action_spec_digest="c" * 64),
            "normalizer action identity does not match environment",
        ),
        (
            lambda dataset: _normalizer(dataset, alpha_artifact_digest="c" * 64),
            "normalizer alpha artifact identity does not match environment",
        ),
        (
            lambda dataset: _normalizer(dataset, factor_artifact_digest="c" * 64),
            "normalizer factor artifact identity does not match environment",
        ),
        (
            lambda dataset: _normalizer(dataset, preserve_passthrough=False),
            "normalizer must preserve observation mask and activity indices",
        ),
    ),
)
def test_flat_normalizer_validation_messages_are_preserved(
    normalizer_factory: Callable[[MarketDataset], ObservationNormalizer],
    message: str,
) -> None:
    dataset = _market()
    normalizer = normalizer_factory(dataset)

    with pytest.raises(ValueError, match=message):
        _builder(dataset, normalizer=normalizer).build(minimum_start_index=0)


@pytest.mark.parametrize(
    ("normalizer", "message"),
    (
        (
            lambda dataset: _sequence_normalizer(
                dataset,
                dataset_id="c" * 64,
                source_dataset_id="d" * 64,
            ),
            "sequence normalizer dataset identity does not match environment",
        ),
        (
            lambda dataset: _sequence_normalizer(
                dataset, sequence_schema_digest="c" * 64
            ),
            "sequence normalizer schema does not match environment",
        ),
    ),
)
def test_sequence_normalizer_validation_messages_are_preserved(
    normalizer: Callable[[MarketDataset], SequenceFeatureNormalizer],
    message: str,
) -> None:
    dataset = _market()

    with pytest.raises(ValueError, match=message):
        _builder(
            dataset,
            config=_config(structured=True),
            sequence_normalizer=normalizer(dataset),
        ).build(minimum_start_index=0)


def test_sequence_window_length_type_error_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _market()

    def invalid_payload(
        self: SequenceObservationBuilder, dataset: MarketDataset
    ) -> dict[str, object]:
        del self, dataset
        return {
            "windows": (
                {
                    "timeframe": "15m",
                    "length": True,
                    "feature_names": ("15m__ret",),
                },
            )
        }

    monkeypatch.setattr(SequenceObservationBuilder, "schema_payload", invalid_payload)
    with pytest.raises(ValueError, match="sequence window length must be an integer"):
        _builder(dataset, config=_config(structured=True)).build(minimum_start_index=0)


def test_sequence_feature_order_error_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _market()

    def invalid_payload(
        self: SequenceObservationBuilder, dataset: MarketDataset
    ) -> dict[str, object]:
        del self, dataset
        return {
            "windows": (
                {
                    "timeframe": "15m",
                    "length": 2,
                    "feature_names": {"15m__ret"},
                },
            )
        }

    monkeypatch.setattr(SequenceObservationBuilder, "schema_payload", invalid_payload)
    with pytest.raises(ValueError, match="sequence feature names must be ordered"):
        _builder(dataset, config=_config(structured=True)).build(minimum_start_index=0)
