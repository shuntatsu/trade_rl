"""Stable causal observation layout shared by training and serving."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState
from trade_rl.strategies.trend import TrendTargets

OBSERVATION_SCHEMA_VERSION = "causal_observation_v2"


@dataclass(frozen=True, slots=True)
class ObservationLayout:
    n_symbols: int
    per_symbol_width: int
    global_width: int

    @property
    def size(self) -> int:
        return self.n_symbols * self.per_symbol_width + self.global_width


@dataclass(frozen=True, slots=True)
class ObservationInput:
    """Complete structured state required to build one policy observation."""

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


def _drawdown(book: BookState) -> float:
    return 1.0 - book.portfolio_value / max(book.peak_value, book.portfolio_value)


def _validate_book(book: BookState, dataset: MarketDataset, *, field_name: str) -> None:
    if book.weights.shape != (dataset.n_symbols,):
        raise ValueError(f"{field_name} weights do not match dataset symbols")


class ObservationBuilder:
    """Build the exact current-time policy input without reading future rows."""

    def layout(self, dataset: MarketDataset) -> ObservationLayout:
        return ObservationLayout(
            n_symbols=dataset.n_symbols,
            per_symbol_width=dataset.n_features * 3 + 9,
            global_width=len(dataset.global_feature_names) + 10,
        )

    def schema_digest(self, dataset: MarketDataset) -> str:
        """Return the stable ordered observation contract for one dataset schema."""

        layout = self.layout(dataset)
        return content_digest(
            {
                "schema_version": OBSERVATION_SCHEMA_VERSION,
                "symbols": dataset.symbols,
                "feature_names": dataset.feature_names,
                "global_feature_names": dataset.global_feature_names,
                "feature_config_digest": dataset.feature_config_digest,
                "normalization_digest": dataset.normalization_digest,
                "per_symbol_fields": (
                    "features",
                    "feature_available",
                    "feature_staleness",
                    "tradable",
                    "symbol_active",
                    "trend_fast",
                    "trend_base",
                    "trend_slow",
                    "alpha",
                    "hybrid_weight",
                    "shadow_weight",
                    "hybrid_minus_shadow",
                ),
                "global_state_fields": (
                    "hybrid_log_value",
                    "shadow_log_value",
                    "hybrid_drawdown",
                    "shadow_drawdown",
                    "relative_log_value",
                    "hybrid_gross",
                    "shadow_gross",
                    "hybrid_risk_scale",
                    "shadow_risk_scale",
                    "episode_progress",
                ),
                "layout": {
                    "n_symbols": layout.n_symbols,
                    "per_symbol_width": layout.per_symbol_width,
                    "global_width": layout.global_width,
                    "size": layout.size,
                    "dtype": "float32",
                },
            }
        )

    def build(self, value: ObservationInput) -> np.ndarray:
        dataset = value.dataset
        index = value.index
        if not 0 <= index < dataset.n_bars:
            raise ValueError("observation index is outside the dataset")
        _validate_book(value.hybrid, dataset, field_name="hybrid")
        _validate_book(value.shadow, dataset, field_name="shadow")
        alpha_vector = np.asarray(value.alpha, dtype=np.float64).reshape(-1)
        if (
            alpha_vector.shape != (dataset.n_symbols,)
            or not np.isfinite(alpha_vector).all()
        ):
            raise ValueError("alpha vector does not match dataset symbols")
        if value.end_index <= value.start_index:
            raise ValueError("episode end_index must be greater than start_index")
        for field_name, scale in (
            ("hybrid_risk_scale", value.hybrid_risk_scale),
            ("shadow_risk_scale", value.shadow_risk_scale),
        ):
            if not np.isfinite(scale) or not 0.0 <= scale <= 1.0:
                raise ValueError(f"{field_name} must be finite and within [0, 1]")

        feature_staleness = dataset.feature_staleness
        symbol_active = dataset.symbol_active
        assert feature_staleness is not None
        assert symbol_active is not None
        hybrid_weights = value.hybrid.weights
        shadow_weights = value.shadow.weights
        per_symbol = np.column_stack(
            (
                dataset.features[index],
                dataset.feature_available[index].astype(np.float64, copy=False),
                feature_staleness[index].astype(np.float64, copy=False),
                dataset.tradable[index].astype(np.float64, copy=False),
                symbol_active[index].astype(np.float64, copy=False),
                value.trends.fast,
                value.trends.base,
                value.trends.slow,
                alpha_vector,
                hybrid_weights,
                shadow_weights,
                hybrid_weights - shadow_weights,
            )
        )

        hybrid_value = value.hybrid.portfolio_value
        shadow_value = value.shadow.portfolio_value
        hybrid_drawdown = _drawdown(value.hybrid)
        shadow_drawdown = _drawdown(value.shadow)
        progress = (index - value.start_index) / (value.end_index - value.start_index)
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
                        value.hybrid_risk_scale,
                        value.shadow_risk_scale,
                        float(np.clip(progress, 0.0, 1.0)),
                    ],
                    dtype=np.float64,
                ),
            )
        )
        observation = np.concatenate((per_symbol.reshape(-1), global_values)).astype(
            np.float32
        )
        expected = self.layout(dataset).size
        if observation.shape != (expected,) or not np.isfinite(observation).all():
            raise ValueError("constructed observation does not match its schema")
        return observation


def observation_layout(dataset: MarketDataset) -> ObservationLayout:
    """Compatibility wrapper for callers that only need the stable layout."""

    return ObservationBuilder().layout(dataset)


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
    builder: ObservationBuilder | None = None,
) -> np.ndarray:
    """Compatibility wrapper around :class:`ObservationBuilder`."""

    resolved = builder or ObservationBuilder()
    return resolved.build(
        ObservationInput(
            dataset=dataset,
            index=index,
            trends=trends,
            alpha=alpha,
            hybrid=hybrid,
            shadow=shadow,
            start_index=start_index,
            end_index=end_index,
            hybrid_risk_scale=hybrid_risk_scale,
            shadow_risk_scale=shadow_risk_scale,
        )
    )


def math_log_value(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("portfolio value must be finite and positive")
    return float(np.log(value))
