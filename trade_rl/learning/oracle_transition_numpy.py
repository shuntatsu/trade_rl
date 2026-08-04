"""Batched NumPy transition kernel for bounded Oracle Bellman solvers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Final

import numpy as np

from trade_rl.learning.oracle_bellman_contracts import OracleBellmanParameters
from trade_rl.learning.oracle_market_tape import OracleMarketTape
from trade_rl.risk.portfolio import PortfolioRiskConfig

_EPSILON: Final = 1e-12
_MINIMUM_NOTIONAL_TOLERANCE: Final = 1e-9


class FillClassification(IntEnum):
    """Aggregate execution outcome for one prior/target transition."""

    INVALID = -1
    NO_REQUEST = 0
    MINIMUM_NOTIONAL_NOOP = 1
    EXECUTION_BLOCKED_NOOP = 2
    CAPACITY_NOOP = 3
    PARTIAL_FILL = 4
    FULL_FILL = 5


@dataclass(frozen=True, slots=True)
class NumPyOpenStateBatch:
    """Path-dependent state after the close-to-open gap."""

    gap_factor: np.ndarray
    open_weights: np.ndarray
    open_equity: np.ndarray
    valid_prior: np.ndarray


@dataclass(frozen=True, slots=True)
class NumPyExecutionBatch:
    """Candidate transition outputs after execution and close accounting."""

    valid: np.ndarray
    close_factor: np.ndarray
    close_weights: np.ndarray
    effective_targets: np.ndarray
    fill_classification: np.ndarray


@dataclass(frozen=True, slots=True)
class NumPyTransitionBatch:
    """Combined open-state and candidate execution outputs."""

    gap_factor: np.ndarray
    open_weights: np.ndarray
    open_equity: np.ndarray
    valid_prior: np.ndarray
    valid: np.ndarray
    close_factor: np.ndarray
    close_weights: np.ndarray
    effective_targets: np.ndarray
    fill_classification: np.ndarray


def _numeric_array(
    value: object,
    *,
    field: str,
    ndim: int,
    allow_negative_infinity: bool = False,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != ndim:
        raise ValueError(f"{field} must be {ndim}-dimensional")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
        raw.dtype, np.bool_
    ):
        raise ValueError(f"{field} must be numeric")
    array = np.asarray(raw, dtype=np.float64)
    if np.isnan(array).any() or np.isposinf(array).any():
        raise ValueError(f"{field} contains unsupported non-finite values")
    if not allow_negative_infinity and np.isneginf(array).any():
        raise ValueError(f"{field} must be finite")
    return array


def numpy_open_state_step(
    *,
    raw_position_factor: np.ndarray,
    equity_position_factor: np.ndarray,
    active: np.ndarray,
    prior_scores: np.ndarray,
    prior_close_weights: np.ndarray,
    reference_portfolio_value: float,
) -> NumPyOpenStateBatch:
    """Advance batched prior close states through the next open."""

    scores = _numeric_array(
        prior_scores,
        field="prior_scores",
        ndim=2,
        allow_negative_infinity=True,
    )
    weights = _numeric_array(
        prior_close_weights,
        field="prior_close_weights",
        ndim=3,
    )
    if scores.shape != weights.shape[:2]:
        raise ValueError("prior scores and close weights do not align")
    symbol_count = weights.shape[2]
    raw_factor = _numeric_array(
        raw_position_factor,
        field="raw_position_factor",
        ndim=1,
    )
    equity_factor = _numeric_array(
        equity_position_factor,
        field="equity_position_factor",
        ndim=1,
    )
    active_mask = np.asarray(active)
    if (
        raw_factor.shape != (symbol_count,)
        or equity_factor.shape != (symbol_count,)
        or active_mask.shape != (symbol_count,)
        or not np.issubdtype(active_mask.dtype, np.bool_)
    ):
        raise ValueError("open market inputs must match the symbol count")
    if np.any(raw_factor <= 0.0) or np.any(equity_factor < 0.0):
        raise ValueError("open position factors are outside their maintained bounds")
    if (
        not math.isfinite(reference_portfolio_value)
        or reference_portfolio_value <= 0.0
    ):
        raise ValueError("reference_portfolio_value must be finite and positive")

    gap_factor = 1.0 + np.sum(
        weights * (equity_factor[None, None, :] - 1.0),
        axis=2,
    )
    valid_prior = (
        np.isfinite(scores)
        & np.isfinite(gap_factor)
        & (gap_factor > _EPSILON)
    )
    safe_gap = np.where(valid_prior, gap_factor, 1.0)
    open_position_fractions = (
        weights
        * raw_factor[None, None, :]
        * active_mask[None, None, :].astype(np.float64)
    )
    open_weights = open_position_fractions / safe_gap[:, :, None]
    open_weights = np.where(valid_prior[:, :, None], open_weights, 0.0)
    open_equity = (
        reference_portfolio_value
        * np.exp(np.clip(scores, -50.0, 50.0))
        * safe_gap
    )
    open_equity = np.where(valid_prior, open_equity, 0.0)
    return NumPyOpenStateBatch(
        gap_factor=gap_factor,
        open_weights=open_weights,
        open_equity=open_equity,
        valid_prior=valid_prior,
    )


def numpy_effective_target_matrix(
    parameters: OracleBellmanParameters,
    current_weights: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Apply maintained rebalance controls and hard target limits."""

    current = _numeric_array(
        current_weights,
        field="current_weights",
        ndim=3,
    )[:, :, None, :]
    requested_targets = _numeric_array(targets, field="targets", ndim=2)
    if requested_targets.shape[1] != current.shape[3]:
        raise ValueError("targets must match the current symbol count")
    requested = np.broadcast_to(
        requested_targets[None, None, :, :],
        (*current.shape[:2], requested_targets.shape[0], current.shape[3]),
    )
    controlled = requested.copy()
    current_abs = np.abs(current)
    target_abs = np.abs(requested)
    target_nonzero = target_abs > _EPSILON
    current_zero = current_abs <= _EPSILON
    same_direction = current * requested > 0.0

    entry_suppressed = (
        current_zero
        & target_nonzero
        & (target_abs < parameters.entry_threshold)
    )
    controlled[entry_suppressed] = 0.0

    exit_suppressed = (
        ~current_zero
        & same_direction
        & (target_abs <= parameters.exit_threshold)
    )
    controlled[exit_suppressed] = 0.0

    hold_suppressed = (
        ~current_zero
        & same_direction
        & (target_abs > parameters.exit_threshold)
        & (target_abs < parameters.entry_threshold)
    )
    controlled[hold_suppressed] = np.broadcast_to(current, controlled.shape)[
        hold_suppressed
    ]

    reversal_suppressed = (
        ~current_zero
        & ~same_direction
        & (target_abs < parameters.entry_threshold)
    )
    controlled[reversal_suppressed] = 0.0

    small_change = np.abs(controlled - current) < parameters.no_trade_band
    controlled = np.where(small_change, current, controlled)
    controlled = np.clip(
        controlled,
        -parameters.max_abs_weight,
        parameters.max_abs_weight,
    )
    gross = np.abs(controlled).sum(axis=3, keepdims=True)
    scale = np.minimum(
        1.0,
        parameters.max_gross / np.maximum(gross, _EPSILON),
    )
    return controlled * scale


def project_portfolio_targets_numpy(
    targets: np.ndarray,
    *,
    portfolio_value: np.ndarray,
    market_notional: np.ndarray,
    config: PortfolioRiskConfig,
) -> np.ndarray:
    """Project batched targets through maintained portfolio-risk limits."""

    if not isinstance(config, PortfolioRiskConfig):
        raise ValueError("config must be PortfolioRiskConfig")
    weights = _numeric_array(targets, field="targets", ndim=4).copy()
    values = _numeric_array(
        portfolio_value,
        field="portfolio_value",
        ndim=2,
    )
    liquidity = _numeric_array(
        market_notional,
        field="market_notional",
        ndim=1,
    )
    if weights.shape[:2] != values.shape or weights.shape[3] != liquidity.size:
        raise ValueError("portfolio projection inputs do not align")
    if np.any(values <= 0.0) or np.any(liquidity < 0.0):
        raise ValueError("portfolio projection inputs are outside maintained bounds")
    if config.max_abs_weight is not None:
        weights = np.clip(weights, -config.max_abs_weight, config.max_abs_weight)
    if config.max_position_to_market_notional is not None:
        caps = (
            liquidity[None, None, None, :]
            * config.max_position_to_market_notional
            / values[:, :, None, None]
        )
        weights = np.clip(weights, -caps, caps)
    if config.max_net_exposure is not None:
        net = np.abs(weights.sum(axis=3, keepdims=True))
        scale = np.minimum(
            1.0,
            config.max_net_exposure / np.maximum(net, _EPSILON),
        )
        weights *= scale
    return weights


def _fill_classification(
    *,
    valid: np.ndarray,
    requested_trade: np.ndarray,
    executable: np.ndarray,
    requested: np.ndarray,
    minimum_notional: np.ndarray,
    capacity: np.ndarray,
    filled_notional: np.ndarray,
) -> np.ndarray:
    requested_any = np.any(requested_trade, axis=3)
    filled_any = np.any(filled_notional > _EPSILON, axis=3)
    below_minimum = (
        requested_trade
        & executable
        & (
            requested
            < minimum_notional[None, None, None, :]
            - _MINIMUM_NOTIONAL_TOLERANCE
        )
    )
    blocked = requested_trade & ~executable
    eligible = requested_trade & executable & ~below_minimum
    capacity_noop = eligible & (capacity[None, None, None, :] <= _EPSILON)
    fully_filled = np.all(
        ~requested_trade
        | (filled_notional >= requested - _MINIMUM_NOTIONAL_TOLERANCE),
        axis=3,
    )

    result = np.full(valid.shape, FillClassification.NO_REQUEST, dtype=np.int8)
    no_fill = requested_any & ~filled_any
    result[no_fill & np.any(below_minimum, axis=3)] = (
        FillClassification.MINIMUM_NOTIONAL_NOOP
    )
    result[
        no_fill & ~np.any(below_minimum, axis=3) & np.any(blocked, axis=3)
    ] = FillClassification.EXECUTION_BLOCKED_NOOP
    result[
        no_fill
        & ~np.any(below_minimum, axis=3)
        & ~np.any(blocked, axis=3)
        & np.any(capacity_noop, axis=3)
    ] = FillClassification.CAPACITY_NOOP
    result[filled_any & fully_filled] = FillClassification.FULL_FILL
    result[filled_any & ~fully_filled] = FillClassification.PARTIAL_FILL
    result[~valid] = FillClassification.INVALID
    return result


def numpy_execute_transition_step(
    *,
    tape: OracleMarketTape,
    step: int,
    current_weights: np.ndarray,
    open_equity: np.ndarray,
    targets: np.ndarray,
    parameters: OracleBellmanParameters,
) -> NumPyExecutionBatch:
    """Evaluate all batched prior-state to target-state candidates."""

    if not isinstance(tape, OracleMarketTape):
        raise ValueError("tape must be OracleMarketTape")
    if not isinstance(parameters, OracleBellmanParameters):
        raise ValueError("parameters must be OracleBellmanParameters")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 0 <= step < tape.steps
    ):
        raise ValueError("step is outside the market tape")
    weights = _numeric_array(current_weights, field="current_weights", ndim=3)
    equity = _numeric_array(open_equity, field="open_equity", ndim=2)
    if weights.shape[:2] != equity.shape or weights.shape[2] != tape.symbol_count:
        raise ValueError("current weights and equity do not align with the tape")

    requested_targets = numpy_effective_target_matrix(parameters, weights, targets)
    safe_portfolio_value = np.maximum(equity, _EPSILON)
    requested_targets = project_portfolio_targets_numpy(
        requested_targets,
        portfolio_value=safe_portfolio_value,
        market_notional=tape.market_notional[step],
        config=parameters.portfolio_risk,
    )
    current = weights[:, :, None, :]
    desired_delta = requested_targets - current
    requested_trade = np.abs(desired_delta) > _EPSILON
    valid_prior = np.isfinite(equity) & (equity > _EPSILON)
    valid = np.broadcast_to(
        valid_prior[:, :, None],
        desired_delta.shape[:3],
    ).copy()

    direction_allowed = np.where(
        desired_delta > _EPSILON,
        tape.buy_allowed[step][None, None, None, :],
        np.where(
            desired_delta < -_EPSILON,
            tape.sell_allowed[step][None, None, None, :],
            True,
        ),
    )
    executable = (
        tape.active[step][None, None, None, :]
        & tape.tradable[step][None, None, None, :]
        & direction_allowed
    )
    increasing_short = (desired_delta < -_EPSILON) & (
        requested_targets < -_EPSILON
    )
    executable &= (
        ~increasing_short
        | tape.borrow_available[step][None, None, None, :]
    )
    if not parameters.execution_cost.allow_short:
        executable &= requested_targets >= -_EPSILON

    requested = np.abs(desired_delta) * equity[:, :, None, None]
    minimum = tape.minimum_notional[step]
    eligible = (
        requested_trade
        & executable
        & (
            requested
            >= minimum[None, None, None, :]
            - _MINIMUM_NOTIONAL_TOLERANCE
        )
    )
    capacity = tape.participation_capacity[step]
    filled_notional = np.where(
        eligible,
        np.minimum(requested, capacity[None, None, None, :]),
        0.0,
    )
    safe_equity = np.maximum(equity[:, :, None, None], _EPSILON)
    filled_delta = np.sign(desired_delta) * filled_notional / safe_equity
    effective_targets = current + filled_delta
    absolute_delta = np.abs(filled_delta)

    participation = np.zeros_like(filled_notional)
    positive_liquidity = tape.market_notional[step] > _EPSILON
    participation[..., positive_liquidity] = (
        filled_notional[..., positive_liquidity]
        / tape.market_notional[step][
            None, None, None, positive_liquidity
        ]
    )
    unit_cost = tape.base_unit_cost[step][None, None, None, :] + (
        parameters.execution_cost.multiplier
        * parameters.execution_cost.impact_rate
        * np.sqrt(participation)
    )
    cost_fraction = np.sum(absolute_delta * unit_cost, axis=3)
    valid &= np.isfinite(cost_fraction) & (cost_fraction < 1.0 - _EPSILON)

    target_sum = np.sum(effective_targets, axis=3)
    cash_after_execution = 1.0 - target_sum - cost_fraction
    open_collateral = (
        cash_after_execution
        + np.sum(np.minimum(effective_targets, 0.0), axis=3)
        + parameters.execution_cost.collateral_haircut
        * np.sum(np.maximum(effective_targets, 0.0), axis=3)
    )
    open_maintenance = (
        parameters.execution_cost.maintenance_margin_rate
        * np.sum(np.abs(effective_targets), axis=3)
    )
    valid &= open_collateral + _EPSILON >= open_maintenance

    close_position = (
        effective_targets
        * tape.mark_open_ratio[step][None, None, None, :]
    )
    dividend_fraction = np.sum(
        effective_targets
        * tape.dividend_open_ratio[step][None, None, None, :],
        axis=3,
    )
    interest_base = cash_after_execution + dividend_fraction
    cash_interest_fraction = (
        interest_base
        * tape.cash_rate[step]
        * tape.elapsed_year_fraction[step]
    )
    funding_fraction = -np.sum(
        effective_targets
        * tape.funding_due_rate[step][None, None, None, :],
        axis=3,
    )
    borrow_fraction = (
        np.sum(
            np.maximum(-effective_targets, 0.0)
            * tape.borrow_rate[step][None, None, None, :],
            axis=3,
        )
        * tape.elapsed_year_fraction[step]
        * parameters.execution_cost.borrow_rate_multiplier
    )
    close_factor = (
        cash_after_execution
        + np.sum(close_position, axis=3)
        + dividend_fraction
        + cash_interest_fraction
        + funding_fraction
        - borrow_fraction
    )
    valid &= np.isfinite(close_factor) & (close_factor > _EPSILON)
    safe_factor = np.where(valid, close_factor, 1.0)
    close_weights = close_position / safe_factor[:, :, :, None]

    close_cash = close_factor - np.sum(close_position, axis=3)
    close_collateral = (
        close_cash
        + np.sum(np.minimum(close_position, 0.0), axis=3)
        + parameters.execution_cost.collateral_haircut
        * np.sum(np.maximum(close_position, 0.0), axis=3)
    )
    close_maintenance = (
        parameters.execution_cost.maintenance_margin_rate
        * np.sum(np.abs(close_position), axis=3)
    )
    valid &= close_collateral + _EPSILON >= close_maintenance
    classification = _fill_classification(
        valid=valid,
        requested_trade=requested_trade,
        executable=executable,
        requested=requested,
        minimum_notional=minimum,
        capacity=capacity,
        filled_notional=filled_notional,
    )
    close_weights = np.where(valid[:, :, :, None], close_weights, 0.0)
    effective_targets = np.where(
        valid[:, :, :, None],
        effective_targets,
        0.0,
    )
    return NumPyExecutionBatch(
        valid=valid,
        close_factor=close_factor,
        close_weights=close_weights,
        effective_targets=effective_targets,
        fill_classification=classification,
    )


def numpy_transition_step(
    *,
    tape: OracleMarketTape,
    step: int,
    prior_scores: np.ndarray,
    prior_close_weights: np.ndarray,
    targets: np.ndarray,
    parameters: OracleBellmanParameters,
) -> NumPyTransitionBatch:
    """Advance and evaluate one Bellman step for a batch of episodes."""

    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 0 <= step < tape.steps
    ):
        raise ValueError("step is outside the market tape")
    open_state = numpy_open_state_step(
        raw_position_factor=tape.raw_position_factor[step],
        equity_position_factor=tape.equity_position_factor[step],
        active=tape.active[step],
        prior_scores=prior_scores,
        prior_close_weights=prior_close_weights,
        reference_portfolio_value=parameters.reference_portfolio_value,
    )
    execution = numpy_execute_transition_step(
        tape=tape,
        step=step,
        current_weights=open_state.open_weights,
        open_equity=open_state.open_equity,
        targets=targets,
        parameters=parameters,
    )
    valid = execution.valid & open_state.valid_prior[:, :, None]
    close_weights = np.where(
        valid[:, :, :, None], execution.close_weights, 0.0
    )
    effective_targets = np.where(
        valid[:, :, :, None], execution.effective_targets, 0.0
    )
    fill_classification = np.where(
        valid,
        execution.fill_classification,
        FillClassification.INVALID,
    ).astype(np.int8, copy=False)
    return NumPyTransitionBatch(
        gap_factor=open_state.gap_factor,
        open_weights=open_state.open_weights,
        open_equity=open_state.open_equity,
        valid_prior=open_state.valid_prior,
        valid=valid,
        close_factor=execution.close_factor,
        close_weights=close_weights,
        effective_targets=effective_targets,
        fill_classification=fill_classification,
    )


__all__ = [
    "FillClassification",
    "NumPyExecutionBatch",
    "NumPyOpenStateBatch",
    "NumPyTransitionBatch",
    "numpy_effective_target_matrix",
    "numpy_execute_transition_step",
    "numpy_open_state_step",
    "numpy_transition_step",
    "project_portfolio_targets_numpy",
]
