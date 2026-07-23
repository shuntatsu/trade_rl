"""Observation and action-space construction for ``ResidualMarketEnv``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.rl.actions import ActionSpec
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import (
    OBSERVATION_SCHEMA,
    ObservationBuilder,
    ObservationLayout,
    observation_passthrough_indices,
)
from trade_rl.rl.sequence_normalization import SequenceFeatureNormalizer
from trade_rl.rl.sequence_observations import (
    SEQUENCE_OBSERVATION_SCHEMA,
    SequenceObservationBuilder,
    SequencePolicyPlane,
    SequenceWindowSpec,
    build_sequence_policy_plane,
)

ObservationValue = np.ndarray | dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class EnvironmentObservationContractRequest:
    dataset: MarketDataset
    config: ResidualMarketEnvConfig
    action_spec: ActionSpec
    action_spec_digest: str
    alpha_artifact_digest: str | None
    factor_artifact_digest: str | None
    normalizer: ObservationNormalizer | None
    sequence_normalizer: SequenceFeatureNormalizer | None
    minimum_start_index: int


@dataclass(frozen=True, slots=True)
class EnvironmentObservationContract:
    observation_builder: ObservationBuilder
    layout: ObservationLayout
    asset_active_column: int
    sequence_observation_builder: SequenceObservationBuilder | None
    sequence_policy_plane: SequencePolicyPlane | None
    sequence_layout_metadata: dict[str, object] | None
    observation_schema: str
    observation_contract_digest: str
    observation_space: spaces.Space[ObservationValue]
    action_space: spaces.Box
    minimum_start_index: int


class EnvironmentObservationContractFactory:
    """Build immutable observation metadata and public Gymnasium spaces."""

    @staticmethod
    def build(
        request: EnvironmentObservationContractRequest,
    ) -> EnvironmentObservationContract:
        dataset = request.dataset
        config = request.config
        action_spec = request.action_spec
        observation_builder = ObservationBuilder(
            action_size=action_spec.size,
            n_factors=action_spec.n_factors,
            finite_horizon=config.finite_horizon_observation,
        )
        layout = observation_builder.layout(dataset)
        _validate_flat_normalizer(request, observation_builder, layout)
        asset_active_column = 4 * dataset.n_features
        sequence_observation_builder: SequenceObservationBuilder | None = None
        sequence_policy_plane: SequencePolicyPlane | None = None
        sequence_layout_metadata: dict[str, object] | None = None
        minimum_start_index = request.minimum_start_index
        if config.structured_sequence_observation:
            sequence_observation_builder = SequenceObservationBuilder(
                windows=tuple(
                    SequenceWindowSpec(timeframe, length)
                    for timeframe, length in config.resolved_sequence_windows
                )
            )
            if request.sequence_normalizer is not None:
                if dataset.dataset_id not in {
                    request.sequence_normalizer.dataset_id,
                    request.sequence_normalizer.source_dataset_id,
                }:
                    raise ValueError(
                        "sequence normalizer dataset identity does not match environment"
                    )
                if (
                    request.sequence_normalizer.sequence_schema_digest
                    != sequence_observation_builder.layout_digest(dataset)
                ):
                    raise ValueError(
                        "sequence normalizer schema does not match environment"
                    )
            sequence_policy_plane = build_sequence_policy_plane(
                dataset,
                sequence_observation_builder,
                request.sequence_normalizer,
            )
            minimum_start_index = max(
                minimum_start_index,
                sequence_observation_builder.minimum_index(dataset),
            )
            (
                sequence_layout_metadata,
                observation_space,
                observation_contract_digest,
            ) = _structured_contract(
                dataset,
                config,
                observation_builder,
                layout,
                sequence_observation_builder,
            )
            observation_schema = SEQUENCE_OBSERVATION_SCHEMA
        else:
            observation_schema = OBSERVATION_SCHEMA
            observation_contract_digest = observation_builder.schema_digest(dataset)
            observation_space = cast(
                spaces.Space[ObservationValue],
                spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(layout.size,),
                    dtype=np.float32,
                ),
            )
        action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_spec.size,),
            dtype=np.float32,
        )
        return EnvironmentObservationContract(
            observation_builder=observation_builder,
            layout=layout,
            asset_active_column=asset_active_column,
            sequence_observation_builder=sequence_observation_builder,
            sequence_policy_plane=sequence_policy_plane,
            sequence_layout_metadata=sequence_layout_metadata,
            observation_schema=observation_schema,
            observation_contract_digest=observation_contract_digest,
            observation_space=observation_space,
            action_space=action_space,
            minimum_start_index=minimum_start_index,
        )


def _validate_flat_normalizer(
    request: EnvironmentObservationContractRequest,
    observation_builder: ObservationBuilder,
    layout: ObservationLayout,
) -> None:
    normalizer = request.normalizer
    if normalizer is None:
        return
    dataset = request.dataset
    if normalizer.size != layout.size:
        raise ValueError("normalizer size does not match observation layout")
    bound_dataset_ids = {
        identity
        for identity in (normalizer.dataset_id, normalizer.source_dataset_id)
        if identity is not None
    }
    if bound_dataset_ids and dataset.dataset_id not in bound_dataset_ids:
        raise ValueError("normalizer dataset identity does not match environment")
    if normalizer.observation_schema != OBSERVATION_SCHEMA:
        raise ValueError("normalizer observation schema does not match environment")
    observation_schema_digest = observation_builder.schema_digest(dataset)
    if (
        normalizer.observation_schema_digest is not None
        and normalizer.observation_schema_digest != observation_schema_digest
    ):
        raise ValueError(
            "normalizer observation schema digest does not match environment"
        )
    if (
        normalizer.action_spec_digest is not None
        and normalizer.action_spec_digest != request.action_spec_digest
    ):
        raise ValueError("normalizer action identity does not match environment")
    for field_name, expected, observed in (
        (
            "alpha artifact",
            request.alpha_artifact_digest,
            normalizer.alpha_artifact_digest,
        ),
        (
            "factor artifact",
            request.factor_artifact_digest,
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
            action_size=request.action_spec.size,
            n_factors=request.action_spec.n_factors,
            finite_horizon=request.config.finite_horizon_observation,
        )
    )
    if not required_passthrough.issubset(normalizer.passthrough_indices):
        raise ValueError(
            "normalizer must preserve observation mask and activity indices"
        )


def _structured_contract(
    dataset: MarketDataset,
    config: ResidualMarketEnvConfig,
    observation_builder: ObservationBuilder,
    layout: ObservationLayout,
    sequence_observation_builder: SequenceObservationBuilder,
) -> tuple[dict[str, object], spaces.Space[ObservationValue], str]:
    sequence_payload = sequence_observation_builder.schema_payload(dataset)
    sequence_spaces: dict[str, spaces.Space[np.ndarray]] = {
        "decision_index": spaces.Box(
            low=0,
            high=dataset.n_bars - 1,
            shape=(1,),
            dtype=np.int64,
        ),
        "current_snapshot": spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(dataset.n_symbols, 4 * dataset.n_features),
            dtype=np.float32,
        ),
        "asset_state": spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                dataset.n_symbols,
                layout.per_symbol_width - 4 * dataset.n_features,
            ),
            dtype=np.float32,
        ),
        "global_state": spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(layout.global_width,),
            dtype=np.float32,
        ),
        "active": spaces.Box(
            low=0.0,
            high=1.0,
            shape=(dataset.n_symbols,),
            dtype=np.float32,
        ),
    }
    feature_counts: dict[str, int] = {}
    window_lengths: dict[str, int] = {}
    sequence_windows = cast(tuple[dict[str, object], ...], sequence_payload["windows"])
    for window in sequence_windows:
        item = dict(window)
        timeframe = str(item["timeframe"])
        raw_length = item["length"]
        if isinstance(raw_length, bool) or not isinstance(raw_length, int):
            raise ValueError("sequence window length must be an integer")
        raw_feature_names = item["feature_names"]
        if not isinstance(raw_feature_names, (tuple, list)):
            raise ValueError("sequence feature names must be ordered")
        length = raw_length
        feature_count = len(raw_feature_names)
        feature_counts[timeframe] = feature_count
        window_lengths[timeframe] = length
        base_shape = (dataset.n_symbols, length, feature_count)
        sequence_spaces[f"sequence_{timeframe}_values"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=base_shape,
            dtype=np.float16,
        )
        sequence_spaces[f"sequence_{timeframe}_available"] = spaces.Box(
            low=0,
            high=1,
            shape=base_shape,
            dtype=np.uint8,
        )
        sequence_spaces[f"sequence_{timeframe}_staleness"] = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=base_shape,
            dtype=np.float16,
        )
    layout_metadata: dict[str, object] = {
        "feature_counts": feature_counts,
        "window_lengths": window_lengths,
        "snapshot_width": 4 * dataset.n_features,
        "asset_state_width": layout.per_symbol_width - 4 * dataset.n_features,
        "global_width": layout.global_width,
        "n_symbols": dataset.n_symbols,
    }
    component_dtypes = {
        key: str(np.dtype(space.dtype))
        for key, space in sorted(sequence_spaces.items())
    }
    contract_digest = content_digest(
        {
            "component_dtypes": component_dtypes,
            "current_schema_digest": observation_builder.schema_digest(dataset),
            "sequence_schema_digest": sequence_observation_builder.schema_digest(
                dataset
            ),
            "layout": layout_metadata,
            "schema_version": SEQUENCE_OBSERVATION_SCHEMA,
        }
    )
    observation_space = cast(
        spaces.Space[ObservationValue],
        spaces.Dict(sequence_spaces),
    )
    return layout_metadata, observation_space, contract_digest


__all__ = [
    "EnvironmentObservationContract",
    "EnvironmentObservationContractFactory",
    "EnvironmentObservationContractRequest",
]
