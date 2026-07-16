"""Train-range-only bounded approximate portfolio teacher targets."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
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
    """Apply the maintained rebalance controls and hard limits vectorially.

    Suppressed target changes are executable no-ops, not invalid transitions.
    This mirrors ``PreTradeRisk.constrain(..., drawdown=0)`` with the maintained
    Oracle contract where per-step turnover slicing is disabled.
    """

    current = np.asarray(current_weights, dtype=np.float64)[:, None, :]
    requested = np.broadcast_to(
        np.asarray(targets, dtype=np.float64)[None, :, :],
        (len(current_weights), len(targets), targets.shape[1]),
    )
    controlled = requested.copy()
    current_abs = np.abs(current)
    target_abs = np.abs(requested)
    target_nonzero = target_abs > _EPSILON
    current_zero = current_abs <= _EPSILON
    same_direction = current * requested > 0.0

    entry_suppressed = (
        current_zero & target_nonzero & (target_abs < config.entry_threshold)
    )
    controlled[entry_suppressed] = 0.0

    exit_suppressed = (
        ~current_zero & same_direction & (target_abs <= config.exit_threshold)
    )
    controlled[exit_suppressed] = 0.0

    hold_suppressed = (
        ~current_zero
        & same_direction
        & (target_abs > config.exit_threshold)
        & (target_abs < config.entry_threshold)
    )
    controlled[hold_suppressed] = np.broadcast_to(current, controlled.shape)[
        hold_suppressed
    ]

    reversal_suppressed = (
        ~current_zero & ~same_direction & (target_abs < config.entry_threshold)
    )
    controlled[reversal_suppressed] = 0.0

    small_change = np.abs(controlled - current) < config.no_trade_band
    controlled = np.where(small_change, current, controlled)

    # Hard concentration and gross limits are applied after the soft rebalance
    # controls, exactly as in PreTradeRisk. They may force de-risking even when
    # a requested change was suppressed by hysteresis/no-trade.
    controlled = np.clip(controlled, -config.max_abs_weight, config.max_abs_weight)
    gross = np.abs(controlled).sum(axis=2, keepdims=True)
    scale = np.minimum(1.0, config.max_gross / np.maximum(gross, _EPSILON))
    return controlled * scale


def _open_state_matrix(
    dataset: MarketDataset,
    *,
    close_index: int,
    prior_close_weights: np.ndarray,
    prior_scores: np.ndarray,
    reference_portfolio_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance every prior close state through splits, delistings, and the next open."""

    execution_index = close_index + 1
    previous_mark = dataset.resolved_array("mark_price")[close_index]
    split = dataset.resolved_array("split_factor")[execution_index]
    next_open = dataset.open[execution_index]
    active = dataset.resolved_array("asset_active")[execution_index]
    recovery = dataset.resolved_array("delisting_recovery")[execution_index]
    raw_position_factor = next_open * split / previous_mark
    equity_position_factor = np.where(
        active,
        raw_position_factor,
        raw_position_factor * recovery,
    )
    gap_factor = 1.0 + np.sum(
        prior_close_weights * (equity_position_factor[None, :] - 1.0),
        axis=1,
    )
    valid = (
        np.isfinite(prior_scores) & np.isfinite(gap_factor) & (gap_factor > _EPSILON)
    )
    safe_gap = np.where(valid, gap_factor, 1.0)
    open_position_fractions = (
        prior_close_weights
        * raw_position_factor[None, :]
        * active[None, :].astype(np.float64)
    )
    open_weights = open_position_fractions / safe_gap[:, None]
    open_weights[~valid] = 0.0
    open_equity = (
        reference_portfolio_value
        * np.exp(np.clip(prior_scores, -50.0, 50.0))
        * safe_gap
    )
    open_equity[~valid] = 0.0
    return gap_factor, open_weights, open_equity, valid


def project_portfolio_targets(
    targets: np.ndarray,
    *,
    portfolio_value: np.ndarray,
    market_notional: np.ndarray,
    config: PortfolioRiskConfig,
) -> np.ndarray:
    """Vectorized maintained portfolio projection for oracle transitions."""

    weights = np.asarray(targets, dtype=np.float64).copy()
    values = np.asarray(portfolio_value, dtype=np.float64).reshape(-1)
    liquidity = np.asarray(market_notional, dtype=np.float64).reshape(-1)
    if weights.ndim != 3 or weights.shape[0] != values.size:
        raise ValueError(
            "oracle portfolio target batch does not match portfolio values"
        )
    if weights.shape[2] != liquidity.size:
        raise ValueError("oracle portfolio target batch does not match liquidity")
    if (
        not np.isfinite(weights).all()
        or not np.isfinite(values).all()
        or not np.isfinite(liquidity).all()
        or np.any(values <= 0.0)
        or np.any(liquidity < 0.0)
    ):
        raise ValueError("oracle portfolio projection inputs are invalid")
    if config.max_abs_weight is not None:
        weights = np.clip(weights, -config.max_abs_weight, config.max_abs_weight)
    if config.max_position_to_market_notional is not None:
        caps = (
            liquidity[None, None, :]
            * config.max_position_to_market_notional
            / values[:, None, None]
        )
        weights = np.clip(weights, -caps, caps)
    if config.max_net_exposure is not None:
        net = np.abs(weights.sum(axis=2, keepdims=True))
        scale = np.minimum(
            1.0,
            config.max_net_exposure / np.maximum(net, _EPSILON),
        )
        weights *= scale
    return weights


def _transition_matrices(
    dataset: MarketDataset,
    config: OracleTeacherConfig,
    *,
    close_index: int,
    current_weights: np.ndarray,
    open_equity: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return feasibility, equity factors, close weights, and effective targets."""

    execution_index = close_index + 1
    requested_targets = _effective_target_matrix(config, current_weights, targets)
    prices = dataset.open[execution_index]
    market_notional = dataset.market_notional(
        execution_index,
        prices,
        volume=dataset.volume[close_index],
    )
    requested_targets = project_portfolio_targets(
        requested_targets,
        portfolio_value=np.maximum(open_equity, _EPSILON),
        market_notional=market_notional,
        config=config.portfolio_risk,
    )
    desired_delta = requested_targets - current_weights[:, None, :]
    requested_trade = np.abs(desired_delta) > _EPSILON
    valid_prior = np.isfinite(open_equity) & (open_equity > _EPSILON)
    valid = np.broadcast_to(valid_prior[:, None], desired_delta.shape[:2]).copy()

    active = dataset.resolved_array("asset_active")[execution_index]
    tradable = dataset.tradable[execution_index]
    buy_allowed = dataset.resolved_array("buy_allowed")[execution_index]
    sell_allowed = dataset.resolved_array("sell_allowed")[execution_index]
    borrow_available = dataset.resolved_array("borrow_available")[execution_index]
    direction_allowed = np.where(
        desired_delta > _EPSILON,
        buy_allowed[None, None, :],
        np.where(desired_delta < -_EPSILON, sell_allowed[None, None, :], True),
    )
    executable = active[None, None, :] & tradable[None, None, :] & direction_allowed
    increasing_short = (desired_delta < -_EPSILON) & (requested_targets < -_EPSILON)
    executable &= ~increasing_short | borrow_available[None, None, :]
    if not config.execution_cost.allow_short:
        executable &= requested_targets >= -_EPSILON

    requested = np.abs(desired_delta) * open_equity[:, None, None]
    participation_limit = np.minimum(
        dataset.resolved_array("max_participation_rate")[execution_index],
        config.execution_cost.max_participation_rate,
    )
    capacity = participation_limit * market_notional
    minimum_notional = np.maximum(
        dataset.resolved_array("minimum_notional")[execution_index],
        config.execution_cost.minimum_notional,
    )
    eligible = (
        requested_trade
        & executable
        & (requested >= minimum_notional[None, None, :] - 1e-9)
    )
    filled_notional = np.where(
        eligible,
        np.minimum(requested, capacity[None, None, :]),
        0.0,
    )
    safe_equity = np.maximum(open_equity[:, None, None], _EPSILON)
    filled_delta = np.sign(desired_delta) * filled_notional / safe_equity
    effective_targets = current_weights[:, None, :] + filled_delta
    absolute_delta = np.abs(filled_delta)

    participation = np.zeros_like(filled_notional)
    positive_liquidity = market_notional > _EPSILON
    participation[:, :, positive_liquidity] = (
        filled_notional[:, :, positive_liquidity]
        / market_notional[None, None, positive_liquidity]
    )
    venue_fee = (
        config.execution_cost.taker_fee_rate
        + dataset.resolved_array("taker_fee_rate")[execution_index]
    )
    base_unit_cost = config.execution_cost.multiplier * (
        config.execution_cost.fee_rate
        + dataset.resolved_array("fee_rate")[execution_index]
        + venue_fee
        + config.execution_cost.spread_rate
        + dataset.resolved_array("spread_rate")[execution_index]
    )
    unit_cost = base_unit_cost[None, None, :] + (
        config.execution_cost.multiplier
        * config.execution_cost.impact_rate
        * np.sqrt(participation)
    )
    cost_fraction = np.sum(absolute_delta * unit_cost, axis=2)
    valid &= np.isfinite(cost_fraction) & (cost_fraction < 1.0 - _EPSILON)

    target_sum = np.sum(effective_targets, axis=2)
    cash_after_execution = 1.0 - target_sum - cost_fraction
    open_position = effective_targets
    open_collateral = (
        cash_after_execution
        + np.sum(np.minimum(open_position, 0.0), axis=2)
        + config.execution_cost.collateral_haircut
        * np.sum(np.maximum(open_position, 0.0), axis=2)
    )
    open_maintenance = config.execution_cost.maintenance_margin_rate * np.sum(
        np.abs(open_position), axis=2
    )
    valid &= open_collateral + _EPSILON >= open_maintenance

    mark_ratio = (
        dataset.resolved_array("mark_price")[execution_index]
        / dataset.open[execution_index]
    )
    close_position = effective_targets * mark_ratio[None, None, :]
    dividend_fraction = np.sum(
        effective_targets
        * dataset.resolved_array("dividend")[execution_index][None, None, :]
        / dataset.open[execution_index][None, None, :],
        axis=2,
    )
    year_fraction = dataset.elapsed_year_fraction(close_index, execution_index)
    interest_base = cash_after_execution + dividend_fraction
    cash_interest_fraction = (
        interest_base
        * float(dataset.resolved_array("cash_rate")[execution_index])
        * year_fraction
    )
    funding_fraction = -np.sum(
        effective_targets
        * dataset.funding_rate[execution_index][None, None, :]
        * dataset.resolved_array("funding_due")[execution_index][None, None, :].astype(
            np.float64
        ),
        axis=2,
    )
    borrow_fraction = (
        np.sum(
            np.maximum(-effective_targets, 0.0)
            * dataset.resolved_array("borrow_rate")[execution_index][None, None, :],
            axis=2,
        )
        * year_fraction
        * config.execution_cost.borrow_rate_multiplier
    )
    close_equity_factor = (
        cash_after_execution
        + np.sum(close_position, axis=2)
        + dividend_fraction
        + cash_interest_fraction
        + funding_fraction
        - borrow_fraction
    )
    valid &= np.isfinite(close_equity_factor) & (close_equity_factor > _EPSILON)
    safe_factor = np.where(valid, close_equity_factor, 1.0)
    close_weights = close_position / safe_factor[:, :, None]

    close_cash = close_equity_factor - np.sum(close_position, axis=2)
    close_collateral = (
        close_cash
        + np.sum(np.minimum(close_position, 0.0), axis=2)
        + config.execution_cost.collateral_haircut
        * np.sum(np.maximum(close_position, 0.0), axis=2)
    )
    close_maintenance = config.execution_cost.maintenance_margin_rate * np.sum(
        np.abs(close_position), axis=2
    )
    valid &= close_collateral + _EPSILON >= close_maintenance
    close_weights = np.where(valid[:, :, None], close_weights, 0.0)
    effective_targets = np.where(valid[:, :, None], effective_targets, 0.0)
    return valid, close_equity_factor, close_weights, effective_targets


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
