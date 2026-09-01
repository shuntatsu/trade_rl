"""Strategy-prior-free causal observation surface for Universal Trade RL U1."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from gymnasium import spaces

from trade_rl.data.market import MarketDataset
from trade_rl.rl.sequence_observations import (
    SequenceObservation,
    SequenceObservationBuilder,
    SequenceWindowSpec,
)
from trade_rl.rl.universal_normalization import UniversalTradeSequenceNormalizer
from trade_rl.rl.universal_trade_contract import (
    UNIVERSAL_TRADE_SEQUENCE_WINDOWS,
    UniversalTradePolicyContract,
)
from trade_rl.rl.universal_trade_runtime import UniversalTradeRuntimeSnapshot

_POLICY_STATE_FIELDS = (
    "policy_requested_weight",
    "pending_target_weight",
    "pending_target_active",
    "risk_projected_weight",
    "current_weight",
    "previous_action",
    "fill_ratio",
    "unfilled_turnover_ratio",
    "participation_ratio",
    "execution_cost_rate",
    "position_age_days",
    "pending_notional_ratio",
    "pending_order_type_code",
    "pending_order_status_code",
    "pending_order_age_days",
    "pending_order_eligible_delay_days",
    "pending_order_triggered",
    "pending_order_expiry_distance_days",
    "asset_active",
    "tradable",
    "borrow_available",
    "borrow_rate",
    "mark_index_basis",
    "current_drawdown",
    "current_gross_exposure",
    "current_net_exposure",
    "cash_weight",
    "risk_scale",
    "margin_utilization",
)


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
            (
                runtime.policy_requested_weight,
                runtime.pending_target_weight,
                float(runtime.pending_target_active),
                runtime.risk_projected_weight,
                runtime.current_weight,
                runtime.previous_action,
                runtime.fill_ratio,
                runtime.unfilled_turnover_ratio,
                runtime.participation_ratio,
                runtime.execution_cost_rate,
                np.log1p(runtime.position_age_hours / 24.0),
                runtime.pending_notional_ratio,
                runtime.pending_order_type_code,
                runtime.pending_order_status_code,
                np.log1p(runtime.pending_order_age_hours / 24.0),
                np.log1p(runtime.pending_order_eligible_delay_hours / 24.0),
                float(runtime.pending_order_triggered),
                np.log1p(runtime.pending_order_expiry_distance_hours / 24.0),
                float(runtime.asset_active),
                float(runtime.tradable),
                float(runtime.borrow_available),
                np.tanh(runtime.borrow_rate),
                np.tanh(100.0 * runtime.mark_index_basis),
                runtime.current_drawdown,
                runtime.current_gross_exposure,
                runtime.current_net_exposure,
                runtime.cash_weight,
                runtime.risk_scale,
                runtime.margin_utilization,
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
            if self._normalizer is not None:
                values = self._normalizer.transform(
                    timeframe,
                    values,
                    available.astype(np.bool_),
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


__all__ = ["UniversalTradeObservationBuilder"]
