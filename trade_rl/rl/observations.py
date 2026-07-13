"""Stable observation layout for baseline-anchored residual policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState
from trade_rl.strategies.trend import TrendTargets


@dataclass(frozen=True, slots=True)
class ObservationLayout:
    n_symbols: int
    per_symbol_width: int
    global_width: int

    @property
    def size(self) -> int:
        return self.n_symbols * self.per_symbol_width + self.global_width


def observation_layout(dataset: MarketDataset) -> ObservationLayout:
    return ObservationLayout(
        n_symbols=dataset.n_symbols,
        per_symbol_width=dataset.n_features + 9,
        global_width=len(dataset.global_feature_names) + 10,
    )


def _drawdown(book: BookState) -> float:
    return 1.0 - book.portfolio_value / max(book.peak_value, book.portfolio_value)


def _validate_book(book: BookState, dataset: MarketDataset, *, field_name: str) -> None:
    if book.weights.shape != (dataset.n_symbols,):
        raise ValueError(f"{field_name} weights do not match dataset symbols")


def build_observation(
    *,
    dataset: MarketDataset,
    index: int,
    trends: TrendTargets,
    alpha: np.ndarray,
    hybrid: BookState,
    shadow: BookState,
    start_index: int,
    end_index: int,
    hybrid_risk_scale: float,
    shadow_risk_scale: float,
) -> np.ndarray:
    """Build the explicit market, hybrid, shadow, and risk-state observation."""

    if not 0 <= index < dataset.n_bars:
        raise ValueError("observation index is outside the dataset")
    _validate_book(hybrid, dataset, field_name="hybrid")
    _validate_book(shadow, dataset, field_name="shadow")
    alpha_vector = np.asarray(alpha, dtype=np.float64).reshape(-1)
    if (
        alpha_vector.shape != (dataset.n_symbols,)
        or not np.isfinite(alpha_vector).all()
    ):
        raise ValueError("alpha vector does not match dataset symbols")
    if end_index <= start_index:
        raise ValueError("episode end_index must be greater than start_index")
    for field_name, value in (
        ("hybrid_risk_scale", hybrid_risk_scale),
        ("shadow_risk_scale", shadow_risk_scale),
    ):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{field_name} must be finite and within [0, 1]")

    availability_fraction = dataset.feature_available[index].mean(axis=1)
    next_index = min(index + 1, dataset.n_bars - 1)
    next_tradable = dataset.tradable[next_index].astype(np.float64, copy=False)
    hybrid_weights = hybrid.weights
    shadow_weights = shadow.weights
    per_symbol = np.column_stack(
        (
            dataset.features[index],
            availability_fraction,
            next_tradable,
            trends.fast,
            trends.base,
            trends.slow,
            alpha_vector,
            hybrid_weights,
            shadow_weights,
            hybrid_weights - shadow_weights,
        )
    )

    hybrid_value = hybrid.portfolio_value
    shadow_value = shadow.portfolio_value
    hybrid_drawdown = _drawdown(hybrid)
    shadow_drawdown = _drawdown(shadow)
    progress = (index - start_index) / (end_index - start_index)
    global_values = np.concatenate(
        (
            dataset.global_features[index].astype(np.float64, copy=False),
            np.array(
                [
                    math_log_value(hybrid_value),
                    math_log_value(shadow_value),
                    hybrid_drawdown,
                    shadow_drawdown,
                    math_log_value(hybrid_value / shadow_value),
                    float(np.abs(hybrid_weights).sum()),
                    float(np.abs(shadow_weights).sum()),
                    hybrid_risk_scale,
                    shadow_risk_scale,
                    float(np.clip(progress, 0.0, 1.0)),
                ],
                dtype=np.float64,
            ),
        )
    )
    observation = np.concatenate((per_symbol.reshape(-1), global_values)).astype(
        np.float32
    )
    expected = observation_layout(dataset).size
    if observation.shape != (expected,) or not np.isfinite(observation).all():
        raise ValueError("constructed observation does not match its schema")
    return observation


def math_log_value(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("portfolio value must be finite and positive")
    return float(np.log(value))
