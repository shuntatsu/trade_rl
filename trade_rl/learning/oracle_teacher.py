"""Train-range-only cost-aware dynamic-programming oracle targets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.simulation.execution import ExecutionCostConfig

ORACLE_TEACHER_SCHEMA: Final = "dp_oracle_teacher_v1"


@dataclass(frozen=True, slots=True)
class OracleTeacherConfig:
    """Deterministic oracle state and execution-cost contract."""

    execution_cost: ExecutionCostConfig = field(default_factory=ExecutionCostConfig)
    positions: tuple[float, ...] = (-1.0, 0.0, 1.0)
    max_gross: float = 1.0
    reference_portfolio_value: float = 1_000_000.0
    schema_version: str = ORACLE_TEACHER_SCHEMA

    def __post_init__(self) -> None:
        positions = tuple(float(value) for value in self.positions)
        if (
            not positions
            or len(set(positions)) != len(positions)
            or positions != tuple(sorted(positions))
            or 0.0 not in positions
            or not np.isfinite(positions).all()
            or min(positions) < -1.0
            or max(positions) > 1.0
        ):
            raise ValueError(
                "oracle positions must be sorted unique finite values in [-1, 1] "
                "and include zero"
            )
        if not self.execution_cost.allow_short and min(positions) < 0.0:
            positions = tuple(value for value in positions if value >= 0.0)
        if not math.isfinite(self.max_gross) or self.max_gross <= 0.0:
            raise ValueError("oracle max_gross must be finite and positive")
        if (
            not math.isfinite(self.reference_portfolio_value)
            or self.reference_portfolio_value <= 0.0
        ):
            raise ValueError(
                "oracle reference_portfolio_value must be finite and positive"
            )
        cost = self.execution_cost
        if (
            cost.slippage_std != 0.0
            or cost.tail_slippage_probability != 0.0
            or cost.order_latency_bars != 0
            or cost.order_type != "market"
        ):
            raise ValueError(
                "oracle execution cost must be deterministic next-open market execution"
            )
        if self.schema_version != ORACLE_TEACHER_SCHEMA:
            raise ValueError("unsupported oracle teacher schema")
        object.__setattr__(self, "positions", positions)

    @property
    def digest(self) -> str:
        return content_digest(self)


def _validate_train_range(
    dataset: MarketDataset,
    train_range: tuple[int, int],
) -> tuple[int, int]:
    if (
        len(train_range) != 2
        or isinstance(train_range[0], bool)
        or isinstance(train_range[1], bool)
        or not isinstance(train_range[0], int)
        or not isinstance(train_range[1], int)
    ):
        raise ValueError("training range must be a pair of integer indices")
    start, stop = train_range
    if not 0 <= start < stop - 1 < dataset.n_bars:
        raise ValueError("training range must contain at least two in-dataset bars")
    return start, stop


def _transition_log_cost(
    dataset: MarketDataset,
    config: OracleTeacherConfig,
    *,
    execution_index: int,
    symbol_index: int,
    delta_weight: float,
) -> float:
    absolute_delta = abs(delta_weight)
    if absolute_delta == 0.0:
        return 0.0
    cost = config.execution_cost
    venue_fee = (
        cost.taker_fee_rate
        + dataset.resolved_array("taker_fee_rate")[execution_index, symbol_index]
    )
    linear_rate = cost.multiplier * (
        cost.fee_rate
        + dataset.resolved_array("fee_rate")[execution_index, symbol_index]
        + venue_fee
        + cost.spread_rate
        + dataset.resolved_array("spread_rate")[execution_index, symbol_index]
    )
    market_notional = float(
        dataset.market_notional(execution_index)[symbol_index]
    )
    requested_notional = absolute_delta * config.reference_portfolio_value
    participation = requested_notional / market_notional if market_notional > 0 else 1.0
    impact_rate = cost.multiplier * cost.impact_rate * math.sqrt(participation)
    cost_fraction = absolute_delta * (linear_rate + impact_rate)
    return math.log(max(1.0 - cost_fraction, 1e-12))


def _symbol_path(
    dataset: MarketDataset,
    config: OracleTeacherConfig,
    *,
    start: int,
    stop: int,
    symbol_index: int,
) -> np.ndarray:
    scale = config.max_gross / dataset.n_symbols
    states = np.asarray(config.positions, dtype=np.float64) * scale
    returns = (
        dataset.close[start + 1 : stop, symbol_index]
        / dataset.close[start : stop - 1, symbol_index]
        - 1.0
    )
    steps = len(returns)
    state_count = len(states)
    holding = np.log(
        np.clip(1.0 + returns[:, None] * states[None, :], 1e-12, None)
    )
    scores = np.full((steps, state_count), -np.inf, dtype=np.float64)
    pointers = np.zeros((steps, state_count), dtype=np.int64)
    for state_index, state in enumerate(states):
        scores[0, state_index] = _transition_log_cost(
            dataset,
            config,
            execution_index=start + 1,
            symbol_index=symbol_index,
            delta_weight=float(state),
        ) + holding[0, state_index]
    for step in range(1, steps):
        execution_index = start + step + 1
        for state_index, state in enumerate(states):
            candidates = np.empty(state_count, dtype=np.float64)
            for prior_index, prior in enumerate(states):
                candidates[prior_index] = scores[step - 1, prior_index] + (
                    _transition_log_cost(
                        dataset,
                        config,
                        execution_index=execution_index,
                        symbol_index=symbol_index,
                        delta_weight=float(state - prior),
                    )
                )
            best_prior = int(np.argmax(candidates))
            pointers[step, state_index] = best_prior
            scores[step, state_index] = candidates[best_prior] + holding[
                step, state_index
            ]
    state_path = np.zeros(steps, dtype=np.int64)
    state_path[-1] = int(np.argmax(scores[-1]))
    for step in range(steps - 1, 0, -1):
        state_path[step - 1] = pointers[step, state_path[step]]
    return states[state_path]


def oracle_target_path(
    dataset: MarketDataset,
    train_range: tuple[int, int],
    config: OracleTeacherConfig,
) -> np.ndarray:
    """Return raw direct-target labels for decisions wholly inside train range."""

    start, stop = _validate_train_range(dataset, train_range)
    targets = np.column_stack(
        [
            _symbol_path(
                dataset,
                config,
                start=start,
                stop=stop,
                symbol_index=symbol_index,
            )
            for symbol_index in range(dataset.n_symbols)
        ]
    )
    if not np.isfinite(targets).all():
        raise RuntimeError("oracle target path contains non-finite values")
    gross = np.abs(targets).sum(axis=1)
    if np.any(gross > config.max_gross + 1e-12):
        raise RuntimeError("oracle target path exceeds max_gross")
    result = np.asarray(targets, dtype=np.float32)
    result.setflags(write=False)
    return result


__all__ = [
    "ORACLE_TEACHER_SCHEMA",
    "OracleTeacherConfig",
    "oracle_target_path",
]
