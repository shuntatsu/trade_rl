"""Observation assembly for the residual market environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import (
    ObservationBuilder,
    ObservationExecutionState,
    ObservationInput,
    ObservationLayout,
    PendingOrderObservationState,
    PolicyObservationSnapshot,
    book_state_vector,
    observation_availability_mask,
    observation_staleness_vector,
)
from trade_rl.rl.sequence_normalization import SequenceFeatureNormalizer
from trade_rl.rl.sequence_observations import (
    SequenceObservationBuilder,
    SequencePolicyPlane,
    build_structured_current_observation,
    build_structured_policy_observation,
)
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.orders import OrderBookState
from trade_rl.strategies.trend import TrendTargets

_LIQUIDATION_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class EnvironmentObservationRuntime:
    """Mutable environment state copied into an observation-only boundary."""

    current_index: int
    start_index: int
    end_index: int
    hybrid: BookState
    shadow: BookState
    hybrid_order_book: OrderBookState
    execution_state: ObservationExecutionState
    previous_action: np.ndarray
    pending_hybrid_target: np.ndarray | None


class EnvironmentObservationAssembler:
    """Build flat, sequence and serving-parity observations from explicit state."""

    def __init__(
        self,
        dataset: MarketDataset,
        *,
        observation_builder: ObservationBuilder,
        layout: ObservationLayout,
        normalizer: ObservationNormalizer | None,
        sequence_observation_builder: SequenceObservationBuilder | None,
        sequence_policy_plane: SequencePolicyPlane | None,
        sequence_normalizer: SequenceFeatureNormalizer | None,
        action_size: int,
        n_factors: int,
        finite_horizon: bool,
    ) -> None:
        self.dataset = dataset
        self.observation_builder = observation_builder
        self.layout = layout
        self.normalizer = normalizer
        self.sequence_observation_builder = sequence_observation_builder
        self.sequence_policy_plane = sequence_policy_plane
        self.sequence_normalizer = sequence_normalizer
        self.action_size = action_size
        self.n_factors = n_factors
        self.finite_horizon = finite_horizon

    @staticmethod
    def _drawdown(book: BookState) -> float:
        value = max(book.portfolio_value, 0.0)
        return min(1.0, max(0.0, 1.0 - value / max(book.peak_value, value, 1e-12)))

    def pending_order_state(
        self,
        runtime: EnvironmentObservationRuntime,
    ) -> PendingOrderObservationState:
        multipliers = runtime.hybrid.contract_multipliers
        if multipliers is None:
            raise RuntimeError("hybrid book is missing contract multipliers")
        return PendingOrderObservationState.from_order_book(
            runtime.hybrid_order_book,
            n_symbols=self.dataset.n_symbols,
            current_index=runtime.current_index,
            reference_prices=self.dataset.resolved_array("mark_price")[
                runtime.current_index
            ],
            contract_multipliers=multipliers,
            portfolio_value=max(
                runtime.hybrid.portfolio_value,
                _LIQUIDATION_TOLERANCE,
            ),
        )

    def flat_pair(
        self,
        runtime: EnvironmentObservationRuntime,
        *,
        trends: TrendTargets,
        alpha: np.ndarray,
        factor_basis: np.ndarray,
        pre_trade_risk: PreTradeRisk,
    ) -> tuple[np.ndarray, np.ndarray]:
        raw = self.observation_builder.build(
            ObservationInput(
                dataset=self.dataset,
                index=runtime.current_index,
                trends=trends,
                alpha=alpha,
                factor_basis=factor_basis,
                hybrid=runtime.hybrid,
                shadow=runtime.shadow,
                start_index=runtime.start_index,
                end_index=runtime.end_index,
                hybrid_risk_scale=pre_trade_risk.risk_scale(
                    self._drawdown(runtime.hybrid)
                ),
                shadow_risk_scale=pre_trade_risk.risk_scale(
                    self._drawdown(runtime.shadow)
                ),
                execution_state=runtime.execution_state,
                pending_order_state=self.pending_order_state(runtime),
                previous_action=runtime.previous_action,
                action_size=self.action_size,
                finite_horizon=self.finite_horizon,
            )
        )
        current = raw if self.normalizer is None else self.normalizer.transform(raw)
        return raw, current

    def snapshot(
        self,
        runtime: EnvironmentObservationRuntime,
        *,
        trends: TrendTargets,
        alpha: np.ndarray,
        factor_basis: np.ndarray,
        pre_trade_risk: PreTradeRisk,
        execution_policy_digest: str,
    ) -> PolicyObservationSnapshot:
        raw, current = self.flat_pair(
            runtime,
            trends=trends,
            alpha=alpha,
            factor_basis=factor_basis,
            pre_trade_risk=pre_trade_risk,
        )
        pending = (
            np.zeros(self.dataset.n_symbols, dtype=np.float64)
            if runtime.pending_hybrid_target is None
            else runtime.pending_hybrid_target.copy()
        )
        order_state = self.pending_order_state(runtime)
        return PolicyObservationSnapshot(
            dataset_id=self.dataset.dataset_id,
            index=runtime.current_index,
            symbols=self.dataset.symbols,
            feature_names=self.dataset.feature_names,
            global_feature_names=self.dataset.global_feature_names,
            availability_mask=observation_availability_mask(
                self.dataset,
                runtime.current_index,
            ),
            staleness=observation_staleness_vector(
                self.dataset,
                runtime.current_index,
            ),
            hybrid_book_state=book_state_vector(runtime.hybrid),
            shadow_book_state=book_state_vector(runtime.shadow),
            pending_target=pending,
            previous_action=runtime.previous_action,
            pending_order_remaining=order_state.remaining_notional_ratio,
            pending_order_type=order_state.order_type_code,
            pending_order_status=order_state.status_code,
            pending_order_age_bars=order_state.age_bars,
            pending_order_eligible_delay=order_state.eligible_delay_bars,
            pending_order_triggered=order_state.triggered,
            pending_order_expiry_distance=order_state.expiry_distance_bars,
            execution_policy_digest=execution_policy_digest,
            raw_observation=raw,
            normalized_observation=current,
        )

    def observation(
        self,
        runtime: EnvironmentObservationRuntime,
        *,
        trends: TrendTargets,
        alpha: np.ndarray,
        factor_basis: np.ndarray,
        pre_trade_risk: PreTradeRisk,
    ) -> np.ndarray | dict[str, np.ndarray]:
        _, current = self.flat_pair(
            runtime,
            trends=trends,
            alpha=alpha,
            factor_basis=factor_basis,
            pre_trade_risk=pre_trade_risk,
        )
        if self.sequence_observation_builder is None:
            return current
        if self.sequence_policy_plane is not None:
            structured = build_structured_current_observation(
                current_flat=current,
                layout=self.layout,
                n_features=self.dataset.n_features,
            )
            structured.update(
                self.sequence_policy_plane.components(runtime.current_index)
            )
            structured["decision_index"] = np.asarray(
                [runtime.current_index],
                dtype=np.int64,
            )
            return structured
        sequence = self.sequence_observation_builder.build(
            self.dataset,
            index=runtime.current_index,
        )
        structured = build_structured_policy_observation(
            sequence=sequence,
            current_flat=current,
            layout=self.layout,
            n_features=self.dataset.n_features,
            sequence_normalizer=self.sequence_normalizer,
        )
        structured["decision_index"] = np.asarray(
            [runtime.current_index],
            dtype=np.int64,
        )
        return structured
