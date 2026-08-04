"""Train-range-only bounded approximate portfolio teacher targets."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.learning.oracle_bellman_contracts import OracleBellmanParameters
from trade_rl.learning.oracle_market_tape import (
    build_oracle_market_tape,
    oracle_open_market_factors,
)
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


def _portfolio_states(
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
    return project_portfolio_targets_numpy(
        weights[None, :, :, :],
        portfolio_value=values[None, :],
        market_notional=market_notional,
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
) -> np.ndarray:
    """Return bounded approximate submitted target labels inside train range."""

    if config.execution_cost.margin_mode != "cross":
        raise ValueError("oracle currently supports cross margin only")
    start, stop = _validate_train_range(dataset, train_range)
    states = _portfolio_states(dataset, config)
    steps = stop - start - 1
    state_count = len(states)
    scores = np.full((steps, state_count), -np.inf, dtype=np.float64)
    pointers = np.full((steps, state_count), -1, dtype=np.int64)
    close_weights = np.zeros((steps, state_count, dataset.n_symbols), dtype=np.float64)
    cash_index = int(np.flatnonzero(np.all(np.isclose(states, 0.0), axis=1))[0])

    for step in range(steps):
        close_index = start + step
        if step == 0:
            prior_scores = np.full(state_count, -np.inf, dtype=np.float64)
            prior_scores[cash_index] = 0.0
            prior_close_weights = np.zeros_like(close_weights[0])
        else:
            prior_scores = scores[step - 1]
            prior_close_weights = close_weights[step - 1]
        gap_factor, open_weights, open_equity, valid_prior = _open_state_matrix(
            dataset,
            close_index=close_index,
            prior_close_weights=prior_close_weights,
            prior_scores=prior_scores,
            reference_portfolio_value=config.reference_portfolio_value,
        )
        if config.signal_delay_decisions == 0:
            (
                transition_valid,
                close_factor,
                candidate_close_weights,
                candidate_effective_targets,
            ) = _transition_matrices(
                dataset,
                config,
                close_index=close_index,
                current_weights=open_weights,
                open_equity=open_equity,
                targets=states,
            )
            transition_valid &= valid_prior[:, None]
            candidate_scores = (
                prior_scores[:, None]
                + np.log(np.where(valid_prior, gap_factor, 1.0))[:, None]
                + np.log(np.where(transition_valid, close_factor, 1.0))
            )
            control_projection = np.abs(
                states[None, :, :] - candidate_effective_targets
            ).sum(axis=2)
            candidate_scores -= config.control_tie_break_penalty * control_projection
            candidate_scores = np.where(transition_valid, candidate_scores, -np.inf)
            best_prior = np.argmax(candidate_scores, axis=0)
            best_scores = candidate_scores[best_prior, np.arange(state_count)]
            scores[step] = best_scores
            pointers[step] = np.where(np.isfinite(best_scores), best_prior, -1)
            close_weights[step] = candidate_close_weights[
                best_prior, np.arange(state_count)
            ]
        elif step == 0:
            hold = states[cash_index : cash_index + 1]
            transition_valid, close_factor, candidate_close_weights, _ = (
                _transition_matrices(
                    dataset,
                    config,
                    close_index=close_index,
                    current_weights=open_weights,
                    open_equity=open_equity,
                    targets=hold,
                )
            )
            transition_valid &= valid_prior[:, None]
            candidate_scores = (
                prior_scores[:, None]
                + np.log(np.where(valid_prior, gap_factor, 1.0))[:, None]
                + np.log(np.where(transition_valid, close_factor, 1.0))
            )
            candidate_scores = np.where(transition_valid, candidate_scores, -np.inf)
            best_prior = int(np.argmax(candidate_scores[:, 0]))
            best_score = float(candidate_scores[best_prior, 0])
            scores[step] = best_score
            pointers[step] = best_prior
            close_weights[step] = candidate_close_weights[best_prior, 0]
        else:
            transition_valid, close_factor, candidate_close_weights, _ = (
                _transition_matrices(
                    dataset,
                    config,
                    close_index=close_index,
                    current_weights=open_weights,
                    open_equity=open_equity,
                    targets=states,
                )
            )
            diagonal = np.arange(state_count)
            diagonal_valid = transition_valid[diagonal, diagonal] & valid_prior
            diagonal_scores = (
                prior_scores
                + np.log(np.where(valid_prior, gap_factor, 1.0))
                + np.log(
                    np.where(
                        diagonal_valid,
                        close_factor[diagonal, diagonal],
                        1.0,
                    )
                )
            )
            diagonal_scores = np.where(diagonal_valid, diagonal_scores, -np.inf)
            best_prior = int(np.argmax(diagonal_scores))
            best_score = float(diagonal_scores[best_prior])
            scores[step] = best_score
            pointers[step] = best_prior
            close_weights[step] = candidate_close_weights[best_prior, best_prior]

        invalid = ~np.isfinite(scores[step])
        close_weights[step, invalid] = 0.0

    final_state = (
        cash_index if config.signal_delay_decisions == 1 else int(np.argmax(scores[-1]))
    )
    if not math.isfinite(float(scores[-1, final_state])):
        raise RuntimeError("oracle found no executable portfolio path")
    state_path = np.zeros(steps, dtype=np.int64)
    state_path[-1] = final_state
    for step in range(steps - 1, 0, -1):
        prior = int(pointers[step, state_path[step]])
        if prior < 0:
            raise RuntimeError("oracle portfolio backpointer is missing")
        state_path[step - 1] = prior
    # Labels are bounded submitted targets. Realized partial/no-fill weights
    # remain in the DP transition state and may drift outside the target grid.
    targets = states[state_path]
    if not np.isfinite(targets).all():
        raise RuntimeError("oracle target path contains non-finite values")
    if np.any(np.abs(targets) > config.max_abs_weight + _EPSILON):
        raise RuntimeError("oracle target path exceeds max_abs_weight")
    if np.any(np.abs(targets).sum(axis=1) > config.max_gross + _EPSILON):
        raise RuntimeError("oracle target path exceeds max_gross")
    result = np.asarray(targets, dtype=np.float32)
    result.setflags(write=False)
    return result


__all__ = [
    "ORACLE_TEACHER_SCHEMA",
    "OracleTeacherConfig",
    "oracle_target_path",
    "project_portfolio_targets",
]
