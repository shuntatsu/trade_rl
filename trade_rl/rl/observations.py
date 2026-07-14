"""Stable observation layout for baseline-anchored residual policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState
from trade_rl.strategies.trend import TrendTargets

OBSERVATION_SCHEMA = "baseline_residual_observation_v3"


@dataclass(frozen=True, slots=True)
class ObservationLayout:
    n_symbols: int
    n_features: int
    action_size: int
    n_factors: int
    per_symbol_width: int
    global_width: int
    finite_horizon: bool = False

    @property
    def size(self) -> int:
        return self.n_symbols * self.per_symbol_width + self.global_width


@dataclass(frozen=True, slots=True)
class ObservationExecutionState:
    """Previous requested and realized execution state visible to the policy."""

    requested_weights: np.ndarray
    fill_ratio: np.ndarray
    unfilled_turnover: np.ndarray
    participation: np.ndarray
    execution_cost: np.ndarray
    position_age: np.ndarray

    @classmethod
    def zero(
        cls,
        n_symbols: int,
        *,
        requested_weights: np.ndarray | None = None,
    ) -> ObservationExecutionState:
        requested = (
            np.zeros(n_symbols, dtype=np.float64)
            if requested_weights is None
            else np.asarray(requested_weights, dtype=np.float64).reshape(-1).copy()
        )
        if requested.shape != (n_symbols,) or not np.isfinite(requested).all():
            raise ValueError("requested_weights do not match symbols")
        zeros = np.zeros(n_symbols, dtype=np.float64)
        return cls(
            requested_weights=requested,
            fill_ratio=zeros.copy(),
            unfilled_turnover=zeros.copy(),
            participation=zeros.copy(),
            execution_cost=zeros.copy(),
            position_age=zeros.copy(),
        )

    def validated(self, n_symbols: int) -> ObservationExecutionState:
        arrays: dict[str, np.ndarray] = {}
        for field_name in (
            "requested_weights",
            "fill_ratio",
            "unfilled_turnover",
            "participation",
            "execution_cost",
            "position_age",
        ):
            vector = np.asarray(getattr(self, field_name), dtype=np.float64).reshape(-1)
            if vector.shape != (n_symbols,) or not np.isfinite(vector).all():
                raise ValueError(f"{field_name} do not match symbols")
            arrays[field_name] = vector
        if np.any(arrays["fill_ratio"] < 0.0) or np.any(arrays["fill_ratio"] > 1.0):
            raise ValueError("fill_ratio must be within [0, 1]")
        if any(
            np.any(arrays[name] < 0.0)
            for name in (
                "unfilled_turnover",
                "participation",
                "execution_cost",
                "position_age",
            )
        ):
            raise ValueError("execution diagnostics must be non-negative")
        return ObservationExecutionState(**arrays)


@dataclass(frozen=True, slots=True)
class ObservationInput:
    """Structured causal state consumed by the shared observation builder."""

    dataset: MarketDataset
    index: int
    trends: TrendTargets
    alpha: np.ndarray
    hybrid: BookState
    shadow: BookState
    start_index: int
    end_index: int
    hybrid_risk_scale: float
    shadow_risk_scale: float
    factor_basis: np.ndarray | None = None
    execution_state: ObservationExecutionState | None = None
    previous_action: np.ndarray | None = None
    action_size: int | None = None
    finite_horizon: bool = False


@dataclass(frozen=True, slots=True)
class ObservationBuilder:
    """Single causal observation implementation shared by training and adapters."""

    action_size: int = 3
    n_factors: int = 0
    finite_horizon: bool = False

    def layout(self, dataset: MarketDataset) -> ObservationLayout:
        return observation_layout(
            dataset,
            action_size=self.action_size,
            n_factors=self.n_factors,
            finite_horizon=self.finite_horizon,
        )

    def schema_digest(self, dataset: MarketDataset) -> str:
        layout = self.layout(dataset)
        return content_digest(
            {
                "schema_version": OBSERVATION_SCHEMA,
                "dataset_feature_names": dataset.feature_names,
                "dataset_global_feature_names": dataset.global_feature_names,
                "action_size": self.action_size,
                "n_factors": self.n_factors,
                "finite_horizon": self.finite_horizon,
                "layout": {
                    "n_symbols": layout.n_symbols,
                    "n_features": layout.n_features,
                    "per_symbol_width": layout.per_symbol_width,
                    "global_width": layout.global_width,
                    "size": layout.size,
                    "dtype": "float32",
                },
            }
        )

    def build(self, value: ObservationInput) -> np.ndarray:
        factor_basis = value.factor_basis
        resolved_factors = (
            self.n_factors
            if factor_basis is None
            else int(np.asarray(factor_basis).shape[0])
        )
        if resolved_factors != self.n_factors:
            raise ValueError("observation factor basis does not match builder")
        resolved_action_size = (
            self.action_size if value.action_size is None else value.action_size
        )
        if resolved_action_size != self.action_size:
            raise ValueError("observation action size does not match builder")
        finite_horizon = value.finite_horizon or self.finite_horizon
        if finite_horizon != self.finite_horizon:
            raise ValueError("observation horizon mode does not match builder")
        return build_observation(
            dataset=value.dataset,
            index=value.index,
            trends=value.trends,
            alpha=value.alpha,
            factor_basis=factor_basis,
            hybrid=value.hybrid,
            shadow=value.shadow,
            start_index=value.start_index,
            end_index=value.end_index,
            hybrid_risk_scale=value.hybrid_risk_scale,
            shadow_risk_scale=value.shadow_risk_scale,
            execution_state=value.execution_state,
            previous_action=value.previous_action,
            action_size=self.action_size,
            finite_horizon=self.finite_horizon,
        )


def observation_layout(
    dataset: MarketDataset,
    *,
    action_size: int = 2,
    n_factors: int = 0,
    finite_horizon: bool = False,
) -> ObservationLayout:
    if (
        isinstance(action_size, bool)
        or not isinstance(action_size, int)
        or action_size <= 0
    ):
        raise ValueError("action_size must be a positive integer")
    if isinstance(n_factors, bool) or not isinstance(n_factors, int) or n_factors < 0:
        raise ValueError("n_factors must be a non-negative integer")
    # Features, masks, staleness, missing reasons, factor loadings, and 18
    # scalar asset fields, including mark/index basis premium.
    per_symbol_width = 4 * dataset.n_features + n_factors + 18
    # global features, masks, staleness, reasons, 15 book/risk fields and action.
    global_width = 4 * len(dataset.global_feature_names) + 15 + action_size
    if finite_horizon:
        global_width += 1
    return ObservationLayout(
        n_symbols=dataset.n_symbols,
        n_features=dataset.n_features,
        action_size=action_size,
        n_factors=n_factors,
        per_symbol_width=per_symbol_width,
        global_width=global_width,
        finite_horizon=finite_horizon,
    )


def observation_trainable_indices(
    dataset: MarketDataset,
    *,
    action_size: int = 2,
    n_factors: int = 0,
    finite_horizon: bool = False,
) -> tuple[int, ...]:
    """Continuous exogenous fields whose statistics may be fitted on train data."""

    layout = observation_layout(
        dataset,
        action_size=action_size,
        n_factors=n_factors,
        finite_horizon=finite_horizon,
    )
    n_features = dataset.n_features
    indices: list[int] = []
    for symbol_index in range(dataset.n_symbols):
        base = symbol_index * layout.per_symbol_width
        indices.extend(range(base, base + n_features))
        indices.extend(range(base + 2 * n_features, base + 3 * n_features))
    global_base = dataset.n_symbols * layout.per_symbol_width
    n_global = len(dataset.global_feature_names)
    indices.extend(range(global_base, global_base + n_global))
    indices.extend(range(global_base + 2 * n_global, global_base + 3 * n_global))
    return tuple(indices)


def observation_passthrough_indices(
    dataset: MarketDataset,
    *,
    action_size: int = 2,
    n_factors: int = 0,
    finite_horizon: bool = False,
) -> tuple[int, ...]:
    """Fields excluded from statistical fitting.

    Masks, categorical values, signals, portfolio state, actions and execution
    state retain their native semantics.  Only raw market features and feature
    age fields are train-fitted.
    """

    layout = observation_layout(
        dataset,
        action_size=action_size,
        n_factors=n_factors,
        finite_horizon=finite_horizon,
    )
    trainable = set(
        observation_trainable_indices(
            dataset,
            action_size=action_size,
            n_factors=n_factors,
            finite_horizon=finite_horizon,
        )
    )
    return tuple(index for index in range(layout.size) if index not in trainable)


def observation_market_matrix(
    dataset: MarketDataset,
    *,
    start: int,
    stop: int,
    action_size: int = 2,
    n_factors: int = 0,
    finite_horizon: bool = False,
) -> np.ndarray:
    """Construct a policy-independent matrix for exogenous normalization."""

    if not 0 <= start < stop <= dataset.n_bars:
        raise ValueError("normalization range is outside the dataset")
    layout = observation_layout(
        dataset,
        action_size=action_size,
        n_factors=n_factors,
        finite_horizon=finite_horizon,
    )
    matrix = np.zeros((stop - start, layout.size), dtype=np.float64)
    n_features = dataset.n_features
    feature_age = dataset.resolved_array("feature_staleness_hours")[start:stop]
    for symbol_index in range(dataset.n_symbols):
        base = symbol_index * layout.per_symbol_width
        matrix[:, base : base + n_features] = dataset.features[
            start:stop, symbol_index
        ]
        matrix[:, base + 2 * n_features : base + 3 * n_features] = feature_age[
            :, symbol_index
        ]
    global_base = dataset.n_symbols * layout.per_symbol_width
    n_global = len(dataset.global_feature_names)
    matrix[:, global_base : global_base + n_global] = dataset.global_features[
        start:stop
    ]
    matrix[
        :, global_base + 2 * n_global : global_base + 3 * n_global
    ] = dataset.resolved_array("global_feature_staleness_hours")[start:stop]
    return matrix


def _drawdown(book: BookState) -> float:
    value = max(book.portfolio_value, 0.0)
    return min(1.0, max(0.0, 1.0 - value / max(book.peak_value, value, 1e-12)))


def _validate_book(book: BookState, dataset: MarketDataset, *, field_name: str) -> None:
    if book.weights.shape != (dataset.n_symbols,):
        raise ValueError(f"{field_name} weights do not match dataset symbols")


def _feature_staleness(dataset: MarketDataset, index: int) -> np.ndarray:
    staleness = dataset.resolved_array("feature_staleness")[index].astype(
        np.float64,
        copy=False,
    )
    available = dataset.feature_available[index]
    return np.where(available, staleness, np.maximum(staleness, 1.0))


def build_observation(
    *,
    dataset: MarketDataset,
    index: int,
    trends: TrendTargets,
    alpha: np.ndarray,
    factor_basis: np.ndarray | None = None,
    hybrid: BookState,
    shadow: BookState,
    start_index: int,
    end_index: int,
    hybrid_risk_scale: float,
    shadow_risk_scale: float,
    execution_state: ObservationExecutionState | None = None,
    previous_action: np.ndarray | None = None,
    action_size: int | None = None,
    finite_horizon: bool = False,
) -> np.ndarray:
    """Build causal market, execution, book and risk-state policy inputs."""

    if not 0 <= index < dataset.n_bars:
        raise ValueError("observation index is outside the dataset")
    if end_index <= start_index:
        raise ValueError("episode end_index must be greater than start_index")
    _validate_book(hybrid, dataset, field_name="hybrid")
    _validate_book(shadow, dataset, field_name="shadow")
    alpha_vector = np.asarray(alpha, dtype=np.float64).reshape(-1)
    if (
        alpha_vector.shape != (dataset.n_symbols,)
        or not np.isfinite(alpha_vector).all()
    ):
        raise ValueError("alpha vector does not match dataset symbols")
    basis = (
        np.empty((0, dataset.n_symbols), dtype=np.float64)
        if factor_basis is None
        else np.asarray(factor_basis, dtype=np.float64)
    )
    if basis.ndim != 2 or basis.shape[1] != dataset.n_symbols:
        raise ValueError("factor_basis must have shape (n_factors, n_symbols)")
    if not np.isfinite(basis).all():
        raise ValueError("factor_basis must be finite")
    n_factors = int(basis.shape[0])
    for field_name, value in (
        ("hybrid_risk_scale", hybrid_risk_scale),
        ("shadow_risk_scale", shadow_risk_scale),
    ):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{field_name} must be finite and within [0, 1]")

    if previous_action is None:
        resolved_action_size = 2 if action_size is None else action_size
        previous = np.zeros(resolved_action_size, dtype=np.float64)
    else:
        previous = np.asarray(previous_action, dtype=np.float64).reshape(-1)
        resolved_action_size = previous.size if action_size is None else action_size
    if (
        isinstance(resolved_action_size, bool)
        or not isinstance(resolved_action_size, int)
        or resolved_action_size <= 0
        or previous.shape != (resolved_action_size,)
        or not np.isfinite(previous).all()
    ):
        raise ValueError("previous_action does not match action_size")

    state = (
        ObservationExecutionState.zero(
            dataset.n_symbols,
            requested_weights=hybrid.weights,
        )
        if execution_state is None
        else execution_state.validated(dataset.n_symbols)
    )
    hybrid_weights = hybrid.weights
    shadow_weights = shadow.weights
    per_symbol = np.column_stack(
        (
            dataset.features[index],
            dataset.feature_available[index].astype(np.float64, copy=False),
            _feature_staleness(dataset, index),
            dataset.resolved_array("feature_missing_reason")[index].astype(
                np.float64, copy=False
            ),
            dataset.resolved_array("asset_active")[index].astype(
                np.float64, copy=False
            ),
            dataset.observable_tradable(index).astype(np.float64, copy=False),
            trends.fast,
            trends.base,
            trends.slow,
            alpha_vector,
            basis.T,
            hybrid_weights,
            shadow_weights,
            hybrid_weights - shadow_weights,
            state.requested_weights,
            state.fill_ratio,
            state.unfilled_turnover,
            state.participation,
            state.execution_cost,
            state.position_age,
            dataset.resolved_array("borrow_available")[index].astype(
                np.float64, copy=False
            ),
            dataset.resolved_array("borrow_rate")[index],
            dataset.resolved_array("mark_price")[index]
            / dataset.resolved_array("index_price")[index]
            - 1.0,
        )
    )

    hybrid_value = max(hybrid.portfolio_value, 1e-12)
    shadow_value = max(shadow.portfolio_value, 1e-12)
    hybrid_gross = hybrid.gross_exposure
    shadow_gross = shadow.gross_exposure
    hybrid_net = hybrid.net_exposure
    shadow_net = shadow.net_exposure
    book_values = [
        math_log_value(hybrid_value),
        math_log_value(shadow_value),
        _drawdown(hybrid),
        _drawdown(shadow),
        math_log_value(hybrid_value / shadow_value),
        hybrid_gross,
        shadow_gross,
        hybrid_net,
        shadow_net,
        hybrid.cash_weight,
        shadow.cash_weight,
        hybrid_risk_scale,
        shadow_risk_scale,
        hybrid.margin_utilization,
        shadow.margin_utilization,
    ]
    global_staleness = np.where(
        dataset.resolved_array("global_feature_available")[index],
        dataset.resolved_array("global_feature_staleness_hours")[index],
        np.maximum(
            dataset.resolved_array("global_feature_staleness_hours")[index], 1.0
        ),
    )
    global_parts = [
        dataset.global_features[index].astype(np.float64, copy=False),
        dataset.resolved_array("global_feature_available")[index].astype(
            np.float64, copy=False
        ),
        global_staleness.astype(np.float64, copy=False),
        dataset.resolved_array("global_feature_missing_reason")[index].astype(
            np.float64, copy=False
        ),
        np.asarray(book_values, dtype=np.float64),
        previous,
    ]
    if finite_horizon:
        remaining = max(0.0, min(1.0, (end_index - index) / (end_index - start_index)))
        global_parts.append(np.array([remaining], dtype=np.float64))
    global_values = np.concatenate(tuple(global_parts))
    observation = np.concatenate((per_symbol.reshape(-1), global_values)).astype(
        np.float32
    )
    expected = observation_layout(
        dataset,
        action_size=resolved_action_size,
        n_factors=n_factors,
        finite_horizon=finite_horizon,
    ).size
    if observation.shape != (expected,) or not np.isfinite(observation).all():
        raise ValueError("constructed observation does not match its schema")
    return observation


def math_log_value(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("portfolio value must be finite and positive")
    return float(np.log(value))
