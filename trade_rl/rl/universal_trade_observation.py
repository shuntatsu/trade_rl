"""Strategy-prior-free causal observation surface for Universal Trade RL U1."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.rl.sequence_observations import (
    SequenceObservation,
    SequenceObservationBuilder,
    SequenceWindowSpec,
)
from trade_rl.rl.universal_normalization import UniversalTradeSequenceNormalizer
from trade_rl.rl.universal_trade_contract import (
    UNIVERSAL_TRADE_OBSERVATION_SCHEMA,
    UNIVERSAL_TRADE_SEQUENCE_WINDOWS,
    UNIVERSAL_TRADE_STATE_LAYOUT_SCHEMA,
    UniversalTradePolicyContract,
)
from trade_rl.rl.universal_trade_runtime import UniversalTradeRuntimeSnapshot

_POLICY_STATE_LAYOUT = (
    ("policy_requested_weight", "policy_requested_weight", "identity_v1"),
    ("pending_target_weight", "pending_target_weight", "identity_v1"),
    ("pending_target_active", "pending_target_active", "bool_to_float_v1"),
    ("risk_projected_weight", "risk_projected_weight", "identity_v1"),
    ("current_weight", "current_weight", "identity_v1"),
    ("previous_action", "previous_action", "identity_v1"),
    ("fill_ratio", "fill_ratio", "identity_v1"),
    ("unfilled_turnover_ratio", "unfilled_turnover_ratio", "identity_v1"),
    ("participation_ratio", "participation_ratio", "identity_v1"),
    ("execution_cost_rate", "execution_cost_rate", "identity_v1"),
    ("position_age_days", "position_age_hours", "log1p_hours_over_24_v1"),
    ("pending_notional_ratio", "pending_notional_ratio", "identity_v1"),
    ("pending_order_type_code", "pending_order_type_code", "identity_v1"),
    ("pending_order_status_code", "pending_order_status_code", "identity_v1"),
    (
        "pending_order_age_days",
        "pending_order_age_hours",
        "log1p_hours_over_24_v1",
    ),
    (
        "pending_order_eligible_delay_days",
        "pending_order_eligible_delay_hours",
        "log1p_hours_over_24_v1",
    ),
    ("pending_order_triggered", "pending_order_triggered", "bool_to_float_v1"),
    (
        "pending_order_expiry_distance_days",
        "pending_order_expiry_distance_hours",
        "log1p_hours_over_24_v1",
    ),
    ("asset_active", "asset_active", "bool_to_float_v1"),
    ("tradable", "tradable", "bool_to_float_v1"),
    ("borrow_available", "borrow_available", "bool_to_float_v1"),
    ("borrow_rate", "borrow_rate", "tanh_raw_v1"),
    ("mark_index_basis", "mark_index_basis", "tanh_100x_v1"),
    ("current_drawdown", "current_drawdown", "identity_v1"),
    ("current_gross_exposure", "current_gross_exposure", "identity_v1"),
    ("current_net_exposure", "current_net_exposure", "identity_v1"),
    ("cash_weight", "cash_weight", "identity_v1"),
    ("risk_scale", "risk_scale", "identity_v1"),
    ("margin_utilization", "margin_utilization", "identity_v1"),
)
UNIVERSAL_TRADE_POLICY_STATE_FIELDS = tuple(
    field_name for field_name, _source_name, _transform in _POLICY_STATE_LAYOUT
)
_POLICY_STATE_FIELDS = UNIVERSAL_TRADE_POLICY_STATE_FIELDS


def _transform_policy_state_value(transform: str, value: float | bool) -> float:
    resolved = float(value)
    if transform == "identity_v1":
        return resolved
    if transform == "bool_to_float_v1":
        return float(bool(value))
    if transform == "log1p_hours_over_24_v1":
        return float(np.log1p(resolved / 24.0))
    if transform == "tanh_raw_v1":
        return float(np.tanh(resolved))
    if transform == "tanh_100x_v1":
        return float(np.tanh(100.0 * resolved))
    raise AssertionError(f"unknown U1 policy-state transform: {transform}")


class UniversalTradeObservationBuilder:
    """Project maintained market/runtime state into the narrow U1 policy surface."""

    def __init__(
        self,
        *,
        contract: UniversalTradePolicyContract,
        normalizer: UniversalTradeSequenceNormalizer | None = None,
    ) -> None:
        if not isinstance(contract, UniversalTradePolicyContract):
            raise TypeError("U1 observation requires a Universal Trade policy contract")
        if normalizer is not None:
            if not isinstance(normalizer, UniversalTradeSequenceNormalizer):
                raise TypeError("U1 observation normalizer is invalid")
            if normalizer.contract_digest != contract.digest:
                raise ValueError(
                    "U1 observation normalizer contract digest does not match policy contract"
                )
        self._contract = contract
        self._normalizer = normalizer
        self._sequence_builder = SequenceObservationBuilder(
            windows=tuple(
                SequenceWindowSpec(timeframe=timeframe, length=length)
                for timeframe, length in UNIVERSAL_TRADE_SEQUENCE_WINDOWS
            )
        )
        self._feature_names_by_timeframe = {
            timeframe: tuple(
                spec.name
                for spec in contract.feature_specs
                if spec.resolved_timeframe("15m") == timeframe
            )
            for timeframe, _length in UNIVERSAL_TRADE_SEQUENCE_WINDOWS
        }
        feature_counts = {
            timeframe: len(self._feature_names_by_timeframe[timeframe])
            for timeframe, _length in UNIVERSAL_TRADE_SEQUENCE_WINDOWS
        }
        if any(count <= 0 for count in feature_counts.values()):
            raise ValueError(
                "U1 observation requires features for every sequence clock"
            )

        self._state_layout_digest = content_digest(
            {
                "schema": UNIVERSAL_TRADE_STATE_LAYOUT_SCHEMA,
                "dtype": "float32",
                "fields": tuple(
                    {
                        "name": field_name,
                        "source": source_name,
                        "transform": transform,
                    }
                    for field_name, source_name, transform in _POLICY_STATE_LAYOUT
                ),
            }
        )
        self._schema_digest = content_digest(
            {
                "schema": UNIVERSAL_TRADE_OBSERVATION_SCHEMA,
                "symbol_axis_semantics": "one_concrete_instrument_v1",
                "sequence_windows": UNIVERSAL_TRADE_SEQUENCE_WINDOWS,
                "feature_specs": tuple(
                    spec.canonical_payload() for spec in contract.feature_specs
                ),
                "sequence_channels": tuple(
                    {
                        "timeframe": timeframe,
                        "values_key": f"sequence_{timeframe}_values",
                        "available_key": f"sequence_{timeframe}_available",
                        "staleness_key": f"sequence_{timeframe}_staleness",
                        "shape": (1, length, feature_counts[timeframe]),
                        "feature_names": self._feature_names_by_timeframe[timeframe],
                        "values_dtype": "float32",
                        "available_dtype": "uint8",
                        "staleness_dtype": "float32",
                    }
                    for timeframe, length in UNIVERSAL_TRADE_SEQUENCE_WINDOWS
                ),
                "policy_state": {
                    "key": "policy_state",
                    "shape": (len(_POLICY_STATE_LAYOUT),),
                    "dtype": "float32",
                    "state_layout_digest": self._state_layout_digest,
                },
            }
        )

        observation_spaces: dict[str, spaces.Space[np.ndarray]] = {}
        for timeframe, length in UNIVERSAL_TRADE_SEQUENCE_WINDOWS:
            shape = (1, length, feature_counts[timeframe])
            observation_spaces[f"sequence_{timeframe}_values"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=shape,
                dtype=np.float32,
            )
            observation_spaces[f"sequence_{timeframe}_available"] = spaces.Box(
                low=0,
                high=1,
                shape=shape,
                dtype=np.uint8,
            )
            observation_spaces[f"sequence_{timeframe}_staleness"] = spaces.Box(
                low=0.0,
                high=np.inf,
                shape=shape,
                dtype=np.float32,
            )
        observation_spaces["policy_state"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(_POLICY_STATE_FIELDS),),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(observation_spaces)

    @property
    def policy_state_fields(self) -> tuple[str, ...]:
        return _POLICY_STATE_FIELDS

    @property
    def schema_digest(self) -> str:
        return self._schema_digest

    @property
    def state_layout_digest(self) -> str:
        return self._state_layout_digest

    def _validate_dataset(self, dataset: MarketDataset) -> None:
        if not isinstance(dataset, MarketDataset):
            raise TypeError("U1 observation dataset is invalid")
        if dataset.n_symbols != 1:
            raise ValueError("U1 observation requires exactly one symbol")
        expected = tuple(spec.name for spec in self._contract.feature_specs)
        if dataset.feature_names != expected:
            raise ValueError("U1 observation feature order does not match contract")

    @staticmethod
    def _policy_state(runtime: UniversalTradeRuntimeSnapshot) -> np.ndarray:
        if not isinstance(runtime, UniversalTradeRuntimeSnapshot):
            raise TypeError("U1 observation runtime snapshot is invalid")
        return np.asarray(
            tuple(
                _transform_policy_state_value(
                    transform,
                    getattr(runtime, source_name),
                )
                for _field_name, source_name, transform in _POLICY_STATE_LAYOUT
            ),
            dtype=np.float32,
        )

    def _sequence(self, *, dataset: MarketDataset, index: int) -> SequenceObservation:
        self._validate_dataset(dataset)
        return self._sequence_builder.build(dataset, index=index)

    def build(
        self,
        *,
        dataset: MarketDataset,
        index: int,
        runtime: UniversalTradeRuntimeSnapshot,
    ) -> dict[str, np.ndarray]:
        sequence = self._sequence(dataset=dataset, index=index)
        result: dict[str, np.ndarray] = {}
        for timeframe, _length in UNIVERSAL_TRADE_SEQUENCE_WINDOWS:
            values = np.asarray(sequence.values[timeframe], dtype=np.float32)
            available = np.asarray(sequence.available[timeframe], dtype=np.uint8)
            availability_mask = available.astype(np.bool_, copy=False)
            if self._normalizer is None:
                values = np.where(availability_mask, values, 0.0).astype(
                    np.float32, copy=False
                )
            else:
                values = self._normalizer.transform(
                    timeframe,
                    values,
                    availability_mask,
                    feature_names=self._feature_names_by_timeframe[timeframe],
                )
            result[f"sequence_{timeframe}_values"] = values
            result[f"sequence_{timeframe}_available"] = available
            result[f"sequence_{timeframe}_staleness"] = np.asarray(
                sequence.staleness[timeframe], dtype=np.float32
            )
        result["policy_state"] = self._policy_state(runtime)
        return result

    def source_indices(
        self,
        *,
        dataset: MarketDataset,
        index: int,
    ) -> Mapping[str, np.ndarray]:
        return self._sequence(dataset=dataset, index=index).source_indices


__all__ = [
    "UNIVERSAL_TRADE_POLICY_STATE_FIELDS",
    "UniversalTradeObservationBuilder",
]
