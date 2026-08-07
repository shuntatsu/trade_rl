"""Batched Torch transition kernel for bounded Oracle Bellman solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol

import torch

from trade_rl.learning.oracle_bellman_contracts import OracleBellmanParameters
from trade_rl.risk.portfolio import PortfolioRiskConfig

_EPSILON: Final = 1e-12
_MINIMUM_NOTIONAL_TOLERANCE: Final = 1e-9


class TorchMarketTapeLike(Protocol):
    @property
    def raw_position_factor(self) -> torch.Tensor: ...

    @property
    def equity_position_factor(self) -> torch.Tensor: ...

    @property
    def mark_open_ratio(self) -> torch.Tensor: ...

    @property
    def active(self) -> torch.Tensor: ...

    @property
    def tradable(self) -> torch.Tensor: ...

    @property
    def buy_allowed(self) -> torch.Tensor: ...

    @property
    def sell_allowed(self) -> torch.Tensor: ...

    @property
    def borrow_available(self) -> torch.Tensor: ...

    @property
    def market_notional(self) -> torch.Tensor: ...

    @property
    def participation_capacity(self) -> torch.Tensor: ...

    @property
    def minimum_notional(self) -> torch.Tensor: ...

    @property
    def base_unit_cost(self) -> torch.Tensor: ...

    @property
    def funding_due_rate(self) -> torch.Tensor: ...

    @property
    def borrow_rate(self) -> torch.Tensor: ...

    @property
    def dividend_open_ratio(self) -> torch.Tensor: ...

    @property
    def cash_rate(self) -> torch.Tensor: ...

    @property
    def elapsed_year_fraction(self) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class TorchTransitionBatch:
    gap_factor: torch.Tensor
    open_weights: torch.Tensor
    open_equity: torch.Tensor
    valid_prior: torch.Tensor
    valid: torch.Tensor
    close_factor: torch.Tensor
    close_weights: torch.Tensor
    effective_targets: torch.Tensor
    fill_classification: torch.Tensor


def _batch_targets(
    targets: torch.Tensor,
    *,
    batch_size: int,
    symbol_count: int,
) -> torch.Tensor:
    if targets.ndim == 2:
        if targets.shape[1] != symbol_count:
            raise ValueError("targets must match the current symbol count")
        return targets.unsqueeze(0).expand(batch_size, -1, -1)
    if targets.ndim == 3:
        if targets.shape[0] != batch_size or targets.shape[2] != symbol_count:
            raise ValueError("targets must match the batch and symbol counts")
        return targets
    raise ValueError("targets must be two- or three-dimensional")


def torch_effective_target_matrix(
    parameters: OracleBellmanParameters,
    current_weights: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Apply maintained rebalance controls using only device operations."""

    if current_weights.ndim != 3 or current_weights.dtype != torch.float64:
        raise ValueError("current_weights must be float64 [B,P,N]")
    batch_size, prior_count, symbol_count = current_weights.shape
    requested = _batch_targets(
        targets,
        batch_size=batch_size,
        symbol_count=symbol_count,
    )
    current = current_weights.unsqueeze(2)
    requested = requested.unsqueeze(1).expand(
        batch_size, prior_count, requested.shape[1], symbol_count
    )
    controlled = requested
    current_abs = current.abs()
    target_abs = requested.abs()
    target_nonzero = target_abs > _EPSILON
    current_zero = current_abs <= _EPSILON
    same_direction = current * requested > 0.0

    entry_suppressed = (
        current_zero & target_nonzero & (target_abs < parameters.entry_threshold)
    )
    controlled = torch.where(entry_suppressed, torch.zeros_like(controlled), controlled)

    exit_suppressed = (
        ~current_zero & same_direction & (target_abs <= parameters.exit_threshold)
    )
    controlled = torch.where(exit_suppressed, torch.zeros_like(controlled), controlled)

    hold_suppressed = (
        ~current_zero
        & same_direction
        & (target_abs > parameters.exit_threshold)
        & (target_abs < parameters.entry_threshold)
    )
    controlled = torch.where(hold_suppressed, current.expand_as(controlled), controlled)

    reversal_suppressed = (
        ~current_zero & ~same_direction & (target_abs < parameters.entry_threshold)
    )
    controlled = torch.where(
        reversal_suppressed, torch.zeros_like(controlled), controlled
    )

    controlled = torch.where(
        (controlled - current).abs() < parameters.no_trade_band,
        current.expand_as(controlled),
        controlled,
    )
    controlled = controlled.clamp(
        min=-parameters.max_abs_weight,
        max=parameters.max_abs_weight,
    )
    gross = controlled.abs().sum(dim=3, keepdim=True)
    scale = torch.minimum(
        torch.ones_like(gross),
        parameters.max_gross / gross.clamp_min(_EPSILON),
    )
    return controlled * scale


def project_portfolio_targets_torch(
    targets: torch.Tensor,
    *,
    portfolio_value: torch.Tensor,
    market_notional: torch.Tensor,
    config: PortfolioRiskConfig,
) -> torch.Tensor:
    if not isinstance(config, PortfolioRiskConfig):
        raise ValueError("config must be PortfolioRiskConfig")
    weights = targets
    if config.max_abs_weight is not None:
        weights = weights.clamp(-config.max_abs_weight, config.max_abs_weight)
    if config.max_position_to_market_notional is not None:
        caps = (
            market_notional[:, None, None, :]
            * config.max_position_to_market_notional
            / portfolio_value[:, :, None, None]
        )
        weights = torch.minimum(torch.maximum(weights, -caps), caps)
    if config.max_net_exposure is not None:
        net = weights.sum(dim=3, keepdim=True).abs()
        scale = torch.minimum(
            torch.ones_like(net),
            config.max_net_exposure / net.clamp_min(_EPSILON),
        )
        weights = weights * scale
    return weights


def _fill_classification(
    *,
    valid: torch.Tensor,
    requested_trade: torch.Tensor,
    executable: torch.Tensor,
    requested: torch.Tensor,
    minimum_notional: torch.Tensor,
    capacity: torch.Tensor,
    filled_notional: torch.Tensor,
) -> torch.Tensor:
    requested_any = requested_trade.any(dim=3)
    filled_any = (filled_notional > _EPSILON).any(dim=3)
    below_minimum = (
        requested_trade
        & executable
        & (requested < minimum_notional[:, None, None, :] - _MINIMUM_NOTIONAL_TOLERANCE)
    )
    blocked = requested_trade & ~executable
    eligible = requested_trade & executable & ~below_minimum
    capacity_noop = eligible & (capacity[:, None, None, :] <= _EPSILON)
    fully_filled = (
        ~requested_trade | (filled_notional >= requested - _MINIMUM_NOTIONAL_TOLERANCE)
    ).all(dim=3)

    result = torch.zeros(valid.shape, dtype=torch.int8, device=valid.device)
    no_fill = requested_any & ~filled_any
    result = torch.where(
        no_fill & below_minimum.any(dim=3),
        torch.full_like(result, 1),
        result,
    )
    result = torch.where(
        no_fill & ~below_minimum.any(dim=3) & blocked.any(dim=3),
        torch.full_like(result, 2),
        result,
    )
    result = torch.where(
        no_fill
        & ~below_minimum.any(dim=3)
        & ~blocked.any(dim=3)
        & capacity_noop.any(dim=3),
        torch.full_like(result, 3),
        result,
    )
    result = torch.where(
        filled_any & fully_filled,
        torch.full_like(result, 5),
        result,
    )
    result = torch.where(
        filled_any & ~fully_filled,
        torch.full_like(result, 4),
        result,
    )
    return torch.where(valid, result, torch.full_like(result, -1))


def torch_transition_step(
    *,
    tape: TorchMarketTapeLike,
    step: torch.Tensor,
    prior_scores: torch.Tensor,
    prior_close_weights: torch.Tensor,
    targets: torch.Tensor,
    parameters: OracleBellmanParameters,
) -> TorchTransitionBatch:
    """Advance and evaluate one Bellman step without host transfers."""

    if prior_scores.ndim != 2 or prior_scores.dtype != torch.float64:
        raise ValueError("prior_scores must be float64 [B,P]")
    if prior_close_weights.ndim != 3 or prior_close_weights.dtype != torch.float64:
        raise ValueError("prior_close_weights must be float64 [B,P,N]")
    if prior_scores.shape != prior_close_weights.shape[:2]:
        raise ValueError("prior state tensors do not align")
    if step.ndim == 0:
        step = step.expand(prior_scores.shape[0])
    if step.ndim != 1 or step.shape[0] != prior_scores.shape[0]:
        raise ValueError("step must align with the episode batch")
    step = step.to(device=prior_scores.device, dtype=torch.int64)

    raw_position_factor = tape.raw_position_factor.index_select(0, step)
    equity_position_factor = tape.equity_position_factor.index_select(0, step)
    active = tape.active.index_select(0, step)
    weights = prior_close_weights
    gap_factor = 1.0 + (weights * (equity_position_factor[:, None, :] - 1.0)).sum(dim=2)
    valid_prior = (
        torch.isfinite(prior_scores)
        & torch.isfinite(gap_factor)
        & (gap_factor > _EPSILON)
    )
    safe_gap = torch.where(valid_prior, gap_factor, torch.ones_like(gap_factor))
    open_position_fractions = (
        weights * raw_position_factor[:, None, :] * active[:, None, :].to(torch.float64)
    )
    open_weights = open_position_fractions / safe_gap[:, :, None]
    open_weights = torch.where(
        valid_prior[:, :, None], open_weights, torch.zeros_like(open_weights)
    )
    clipped_scores = prior_scores.clamp(-50.0, 50.0)
    open_equity = parameters.reference_portfolio_value * clipped_scores.exp() * safe_gap
    open_equity = torch.where(valid_prior, open_equity, torch.zeros_like(open_equity))

    requested_targets = torch_effective_target_matrix(parameters, open_weights, targets)
    safe_portfolio_value = open_equity.clamp_min(_EPSILON)
    market_notional = tape.market_notional.index_select(0, step)
    requested_targets = project_portfolio_targets_torch(
        requested_targets,
        portfolio_value=safe_portfolio_value,
        market_notional=market_notional,
        config=parameters.portfolio_risk,
    )
    current = open_weights.unsqueeze(2)
    desired_delta = requested_targets - current
    requested_trade = desired_delta.abs() > _EPSILON
    valid = valid_prior.unsqueeze(2).expand(desired_delta.shape[:3]).clone()

    buy_allowed = tape.buy_allowed.index_select(0, step)
    sell_allowed = tape.sell_allowed.index_select(0, step)
    direction_allowed = torch.where(
        desired_delta > _EPSILON,
        buy_allowed[:, None, None, :],
        torch.where(
            desired_delta < -_EPSILON,
            sell_allowed[:, None, None, :],
            torch.ones_like(desired_delta, dtype=torch.bool),
        ),
    )
    executable = (
        tape.active.index_select(0, step)[:, None, None, :]
        & tape.tradable.index_select(0, step)[:, None, None, :]
        & direction_allowed
    )
    increasing_short = (desired_delta < -_EPSILON) & (requested_targets < -_EPSILON)
    executable = executable & (
        ~increasing_short
        | tape.borrow_available.index_select(0, step)[:, None, None, :]
    )
    if not parameters.execution_cost.allow_short:
        executable = executable & (requested_targets >= -_EPSILON)

    requested = desired_delta.abs() * open_equity[:, :, None, None]
    minimum = tape.minimum_notional.index_select(0, step)
    eligible = (
        requested_trade
        & executable
        & (requested >= minimum[:, None, None, :] - _MINIMUM_NOTIONAL_TOLERANCE)
    )
    capacity = tape.participation_capacity.index_select(0, step)
    filled_notional = torch.where(
        eligible,
        torch.minimum(requested, capacity[:, None, None, :]),
        torch.zeros_like(requested),
    )
    safe_equity = open_equity[:, :, None, None].clamp_min(_EPSILON)
    filled_delta = desired_delta.sign() * filled_notional / safe_equity
    effective_targets = current + filled_delta
    absolute_delta = filled_delta.abs()

    positive_liquidity = market_notional > _EPSILON
    participation = torch.where(
        positive_liquidity[:, None, None, :],
        filled_notional / market_notional[:, None, None, :].clamp_min(_EPSILON),
        torch.zeros_like(filled_notional),
    )
    unit_cost = tape.base_unit_cost.index_select(0, step)[:, None, None, :] + (
        parameters.execution_cost.multiplier
        * parameters.execution_cost.impact_rate
        * participation.sqrt()
    )
    cost_fraction = (absolute_delta * unit_cost).sum(dim=3)
    valid = valid & torch.isfinite(cost_fraction) & (cost_fraction < 1.0 - _EPSILON)

    target_sum = effective_targets.sum(dim=3)
    cash_after_execution = 1.0 - target_sum - cost_fraction
    open_collateral = (
        cash_after_execution
        + torch.minimum(effective_targets, torch.zeros_like(effective_targets)).sum(
            dim=3
        )
        + parameters.execution_cost.collateral_haircut
        * torch.maximum(effective_targets, torch.zeros_like(effective_targets)).sum(
            dim=3
        )
    )
    open_maintenance = parameters.execution_cost.maintenance_margin_rate * (
        effective_targets.abs().sum(dim=3)
    )
    valid = valid & (open_collateral + _EPSILON >= open_maintenance)

    close_position = (
        effective_targets * tape.mark_open_ratio.index_select(0, step)[:, None, None, :]
    )
    dividend_fraction = (
        effective_targets
        * tape.dividend_open_ratio.index_select(0, step)[:, None, None, :]
    ).sum(dim=3)
    interest_base = cash_after_execution + dividend_fraction
    cash_interest_fraction = (
        interest_base
        * tape.cash_rate.index_select(0, step)[:, None, None]
        * tape.elapsed_year_fraction.index_select(0, step)[:, None, None]
    )
    funding_fraction = -(
        effective_targets
        * tape.funding_due_rate.index_select(0, step)[:, None, None, :]
    ).sum(dim=3)
    borrow_fraction = (
        (
            torch.maximum(-effective_targets, torch.zeros_like(effective_targets))
            * tape.borrow_rate.index_select(0, step)[:, None, None, :]
        ).sum(dim=3)
        * tape.elapsed_year_fraction.index_select(0, step)[:, None, None]
        * parameters.execution_cost.borrow_rate_multiplier
    )
    close_factor = (
        cash_after_execution
        + close_position.sum(dim=3)
        + dividend_fraction
        + cash_interest_fraction
        + funding_fraction
        - borrow_fraction
    )
    valid = valid & torch.isfinite(close_factor) & (close_factor > _EPSILON)
    safe_factor = torch.where(valid, close_factor, torch.ones_like(close_factor))
    close_weights = close_position / safe_factor[:, :, :, None]

    close_cash = close_factor - close_position.sum(dim=3)
    close_collateral = (
        close_cash
        + torch.minimum(close_position, torch.zeros_like(close_position)).sum(dim=3)
        + parameters.execution_cost.collateral_haircut
        * torch.maximum(close_position, torch.zeros_like(close_position)).sum(dim=3)
    )
    close_maintenance = parameters.execution_cost.maintenance_margin_rate * (
        close_position.abs().sum(dim=3)
    )
    valid = valid & (close_collateral + _EPSILON >= close_maintenance)
    classification = _fill_classification(
        valid=valid,
        requested_trade=requested_trade,
        executable=executable,
        requested=requested,
        minimum_notional=minimum,
        capacity=capacity,
        filled_notional=filled_notional,
    )
    close_weights = torch.where(
        valid[:, :, :, None], close_weights, torch.zeros_like(close_weights)
    )
    effective_targets = torch.where(
        valid[:, :, :, None],
        effective_targets,
        torch.zeros_like(effective_targets),
    )
    return TorchTransitionBatch(
        gap_factor=gap_factor,
        open_weights=open_weights,
        open_equity=open_equity,
        valid_prior=valid_prior,
        valid=valid,
        close_factor=close_factor,
        close_weights=close_weights,
        effective_targets=effective_targets,
        fill_classification=classification,
    )


__all__ = [
    "TorchMarketTapeLike",
    "TorchTransitionBatch",
    "project_portfolio_targets_torch",
    "torch_effective_target_matrix",
    "torch_transition_step",
]
