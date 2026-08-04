"""Train-range-only bounded approximate portfolio teacher targets."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.learning.oracle_bellman_contracts import (
    OracleBellmanParameters,
    OracleEpisodeInputs,
    OracleSolverConfig,
)
from trade_rl.learning.oracle_market_tape import (
    build_oracle_market_tape,
    oracle_open_market_factors,
)
from trade_rl.learning.oracle_solver import solve_oracle_episodes
from trade_rl.learning.oracle_transition_numpy import (
    numpy_effective_target_matrix,
    numpy_execute_transition_step,
    numpy_open_state_step,
    project_portfolio_targets_numpy,
)
from trade_rl.risk.portfolio import PortfolioRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig

ORACLE_TEACHER_SCHEMA: Final = "approximate_portfolio_teacher_v3"
_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class OracleTeacherConfig:
    """Deterministic bounded-state approximation of the execution contract."""

    execution_cost: ExecutionCostConfig = field(default_factory=ExecutionCostConfig)
    portfolio_risk: PortfolioRiskConfig = field(default_factory=PortfolioRiskConfig)
    positions: tuple[float, ...] = (-1.0, 0.0, 1.0)
    max_gross: float = 1.0
    max_abs_weight: float = 0.45
    entry_threshold: float = 0.10
    exit_threshold: float = 0.03
    no_trade_band: float = 0.05
    reference_portfolio_value: float = 1_000_000.0
    maximum_states: int = 512
    signal_delay_decisions: int = 0
    approximation_contract: str = "bounded_state_partial_fill_v1"
    control_tie_break_penalty: float = 1e-9
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
        for name, value in (
            ("max_gross", self.max_gross),
            ("max_abs_weight", self.max_abs_weight),
            ("reference_portfolio_value", self.reference_portfolio_value),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"oracle {name} must be finite and positive")
        if self.max_abs_weight > self.max_gross:
            raise ValueError("oracle max_abs_weight cannot exceed max_gross")
        if (
            not 0.0
            <= self.exit_threshold
            <= self.entry_threshold
            <= self.max_abs_weight
        ):
            raise ValueError("oracle hysteresis thresholds are invalid")
        if not 0.0 <= self.no_trade_band <= 2.0 * self.max_abs_weight:
            raise ValueError("oracle no_trade_band is invalid")
        if (
            isinstance(self.maximum_states, bool)
            or not isinstance(self.maximum_states, int)
            or self.maximum_states <= 0
        ):
            raise ValueError("oracle maximum_states must be a positive integer")
        if (
            isinstance(self.signal_delay_decisions, bool)
            or not isinstance(self.signal_delay_decisions, int)
            or self.signal_delay_decisions not in {0, 1}
        ):
            raise ValueError(
                "oracle signal_delay_decisions must be exactly zero or one"
            )
        if self.approximation_contract != "bounded_state_partial_fill_v1":
            raise ValueError("unsupported oracle approximation contract")
        if (
            not math.isfinite(self.control_tie_break_penalty)
            or self.control_tie_break_penalty <= 0.0
        ):
            raise ValueError(
                "oracle control_tie_break_penalty must be finite and positive"
            )
        if not isinstance(self.portfolio_risk, PortfolioRiskConfig):
            raise ValueError("oracle portfolio_risk must be PortfolioRiskConfig")
        if any(
            value is not None
            for value in (
                self.portfolio_risk.volatility_target,
                self.portfolio_risk.max_abs_beta,
                self.portfolio_risk.max_stress_loss,
            )
        ):
            raise ValueError(
                "oracle portfolio risk does not support covariance, beta, or stress inputs"
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

    @property
    def bellman_parameters(self) -> OracleBellmanParameters:
        return OracleBellmanParameters(
            execution_cost=self.execution_cost,
            portfolio_risk=self.portfolio_risk,
            positions=self.positions,
            max_gross=self.max_gross,
            max_abs_weight=self.max_abs_weight,
            entry_threshold=self.entry_threshold,
            exit_threshold=self.exit_threshold,
            no_trade_band=self.no_trade_band,
            reference_portfolio_value=self.reference_portfolio_value,
            maximum_states=self.maximum_states,
            signal_delay_decisions=self.signal_delay_decisions,
            approximation_contract=self.approximation_contract,
            control_tie_break_penalty=self.control_tie_break_penalty,
        )


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


def portfolio_states(
    dataset: MarketDataset, config: OracleTeacherConfig
) -> np.ndarray:
    levels = tuple(value * config.max_abs_weight for value in config.positions)
    states = np.asarray(
        [
            state
            for state in itertools.product(levels, repeat=dataset.n_symbols)
            if float(np.abs(state).sum()) <= config.max_gross + _EPSILON
        ],
        dtype=np.float64,
    )
    if len(states) == 0 or len(states) > config.maximum_states:
        raise ValueError(
            f"oracle portfolio state count {len(states)} exceeds maintained bound"
        )
    if not np.any(np.all(np.isclose(states, 0.0), axis=1)):
        raise RuntimeError("oracle portfolio states do not contain cash")
    return states


_portfolio_states = portfolio_states


def _effective_target_matrix(
    config: OracleTeacherConfig,
    current_weights: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Compatibility wrapper over maintained batched NumPy controls."""

    current = np.asarray(current_weights, dtype=np.float64)
    if current.ndim != 2:
        raise ValueError("current_weights must be two-dimensional")
    return numpy_effective_target_matrix(
        config.bellman_parameters,
        current[None, :, :],
        targets,
    )[0]


def _open_state_matrix(
    dataset: MarketDataset,
    *,
    close_index: int,
    prior_close_weights: np.ndarray,
    prior_scores: np.ndarray,
    reference_portfolio_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper over the batched next-open kernel."""

    raw_factor, equity_factor, active = oracle_open_market_factors(dataset, close_index)
    result = numpy_open_state_step(
        raw_position_factor=raw_factor,
        equity_position_factor=equity_factor,
        active=active,
        prior_scores=np.asarray(prior_scores, dtype=np.float64)[None, :],
        prior_close_weights=np.asarray(prior_close_weights, dtype=np.float64)[
            None, :, :
        ],
        reference_portfolio_value=reference_portfolio_value,
    )
    return (
        result.gap_factor[0],
        result.open_weights[0],
        result.open_equity[0],
        result.valid_prior[0],
    )


def project_portfolio_targets(
    targets: np.ndarray,
    *,
    portfolio_value: np.ndarray,
    market_notional: np.ndarray,
    config: PortfolioRiskConfig,
) -> np.ndarray:
    """Compatibility wrapper over the batched risk projection."""

    weights = np.asarray(targets, dtype=np.float64)
    values = np.asarray(portfolio_value, dtype=np.float64).reshape(-1)
    if weights.ndim != 3 or weights.shape[0] != values.size:
        raise ValueError(
            "oracle portfolio target batch does not match portfolio values"
        )
    liquidity = np.asarray(market_notional, dtype=np.float64)
    if liquidity.ndim == 1:
        liquidity = liquidity[None, :]
    return project_portfolio_targets_numpy(
        weights[None, :, :, :],
        portfolio_value=values[None, :],
        market_notional=liquidity,
        config=config,
    )[0]


def _transition_matrices(
    dataset: MarketDataset,
    config: OracleTeacherConfig,
    *,
    close_index: int,
    current_weights: np.ndarray,
    open_equity: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper over the batched execution kernel."""

    tape = build_oracle_market_tape(
        dataset,
        (close_index, close_index + 2),
        config.bellman_parameters,
    )
    result = numpy_execute_transition_step(
        tape=tape,
        step=0,
        current_weights=np.asarray(current_weights, dtype=np.float64)[None, :, :],
        open_equity=np.asarray(open_equity, dtype=np.float64)[None, :],
        targets=targets,
        parameters=config.bellman_parameters,
    )
    return (
        result.valid[0],
        result.close_factor[0],
        result.close_weights[0],
        result.effective_targets[0],
    )


def oracle_target_path(
    dataset: MarketDataset,
    train_range: tuple[int, int],
    config: OracleTeacherConfig,
    *,
    solver_config: OracleSolverConfig | None = None,
) -> np.ndarray:
    """Return bounded approximate submitted target labels inside train range."""

    if config.execution_cost.margin_mode != "cross":
        raise ValueError("oracle currently supports cross margin only")
    start, stop = _validate_train_range(dataset, train_range)
    result = solve_oracle_episodes(
        dataset,
        states=portfolio_states(dataset, config),
        episode_inputs=OracleEpisodeInputs(
            episode_indices=np.array([0], dtype=np.int64),
            starts=np.array([start], dtype=np.int64),
            stops=np.array([stop], dtype=np.int64),
            initial_weights=np.zeros((1, dataset.n_symbols), dtype=np.float64),
        ),
        parameters=config.bellman_parameters,
        solver_config=solver_config or OracleSolverConfig(),
    )
    return result.targets[0]


__all__ = [
    "ORACLE_TEACHER_SCHEMA",
    "OracleTeacherConfig",
    "oracle_target_path",
    "portfolio_states",
    "project_portfolio_targets",
]
