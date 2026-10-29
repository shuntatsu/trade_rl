"""Result-blind paired action comparison over the canonical shared portfolio replay.

This module compares one decision with two fresh complete native replays. It
never clones a partially advanced BookState, estimates a future price, or
supplies a second execution / accounting authority.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real

from trade_rl.data.market import MarketDataset
from trade_rl.data.market_order_rules import MarketOrderProfile
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.decision_value import preserve_posture_aware_proposal
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioPolicy,
    PortfolioTargetProposal,
)
from trade_rl.strategies.portfolio_policy_codec import (
    decode_portfolio_action_codes,
    portfolio_action_to_proposal,
)


@dataclass(frozen=True, slots=True)
class PairedPortfolioActionResult:
    """Both native arms and a label only when both endpoints can be compared.

    An eligible label is a realized *simulation-path* delta, not a future
    conditional expectation and not a claim about unused market data.
    """

    baseline_result: SharedCashPortfolioReplayResult
    candidate_result: SharedCashPortfolioReplayResult
    baseline_snapshot: PortfolioDecisionSnapshot
    candidate_snapshot: PortfolioDecisionSnapshot
    intervention_index: int
    eligible_for_label: bool
    delta_per_initial_capital: float | None


class _OneDecisionOverride:
    """Update the reference actor on every bar but override one native proposal."""

    def __init__(
        self,
        reference: PortfolioPolicy,
        *,
        intervention_index: int,
        action: tuple[int, ...],
        target_abs_weights: tuple[float, ...],
    ) -> None:
        self._reference = reference
        self._intervention_index = intervention_index
        self._action = action
        self._target_abs_weights = target_abs_weights
        self.prefix: list[
            tuple[PortfolioDecisionSnapshot, PortfolioTargetProposal]
        ] = []
        self.intervention_snapshot: PortfolioDecisionSnapshot | None = None

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        proposal = self._reference.decide_portfolio(snapshot)
        if not isinstance(proposal, PortfolioTargetProposal):
            raise TypeError("reference policy must return PortfolioTargetProposal")
        if snapshot.index < self._intervention_index:
            self.prefix.append((snapshot, proposal))
        elif snapshot.index == self._intervention_index:
            self.intervention_snapshot = snapshot
            if snapshot.active_orders:
                return preserve_posture_aware_proposal(
                    self._action, snapshot, self._target_abs_weights
                )
            return portfolio_action_to_proposal(
                self._action, snapshot, self._target_abs_weights
            )
        return proposal


def _settled_label_eligible(
    result: SharedCashPortfolioReplayResult,
    *,
    expected_intervals: int,
) -> bool:
    if result.ledger_evidence is None:
        raise RuntimeError("paired replay must retain native ledger evidence")
    book = result.book
    mapping = result.ledger_evidence.to_mapping()
    intervals = mapping["intervals"]
    if not isinstance(intervals, (tuple, list)):
        raise RuntimeError("native ledger intervals must be an ordered sequence")
    return (
        len(intervals) == expected_intervals
        and not book.insolvent
        and book.termination_reason is None
        and all(quantity == 0 for quantity in book.exact_quantities)
        and not mapping["active_order_remainders"]
    )


def run_paired_portfolio_action(
    dataset: MarketDataset,
    *,
    reference_policy_factory: Callable[[], PortfolioPolicy],
    start_index: int,
    stop_index: int,
    intervention_index: int,
    baseline_action: tuple[int, ...],
    candidate_action: tuple[int, ...],
    target_abs_weights: tuple[object, ...],
    initial_capital: object,
    execution_cost: ExecutionCostConfig,
    risk: PreTradeRisk,
    market_order_profile: MarketOrderProfile | None = None,
) -> PairedPortfolioActionResult:
    """Compare two one-bar joint interventions under the same reference policy.

    Both arms start with a new account/executor/risk and a distinct policy
    instance; the same prefix is replayed rather than copying BookState. The
    reference policy receives each branch's *own* state after the override.
    The supported execution regime is deterministic; identical seeds alone do
    not provide common random numbers when the two arms use different orders.
    """

    if not isinstance(dataset, MarketDataset):
        raise TypeError("dataset must be a MarketDataset")
    if not isinstance(execution_cost, ExecutionCostConfig):
        raise TypeError("execution_cost must be explicit ExecutionCostConfig")
    if type(risk) is not PreTradeRisk:
        raise TypeError("risk must be an ordinary PreTradeRisk")
    if market_order_profile is not None and not isinstance(
        market_order_profile, MarketOrderProfile
    ):
        raise TypeError("market_order_profile must be verified or None")
    if (
        execution_cost.slippage_std != 0
        or execution_cost.tail_slippage_probability != 0
    ):
        raise ValueError("stochastic execution is not supported by paired replay v1")
    if any(type(x) is not int for x in (start_index, stop_index, intervention_index)):
        raise TypeError("start, stop and intervention indices must be native integers")
    if not 0 <= start_index < stop_index < dataset.n_bars:
        raise ValueError("replay range must satisfy 0 <= start < stop < n_bars")
    agent_stop = stop_index - execution_cost.order_latency_bars - 1
    if not start_index <= intervention_index < agent_stop:
        raise ValueError("intervention must precede reserved terminal settlement")
    if isinstance(initial_capital, bool) or not isinstance(initial_capital, Real):
        raise TypeError("initial_capital must be a positive finite real")
    try:
        capital = float(initial_capital)
    except OverflowError as error:
        raise ValueError("initial_capital must be finite") from error
    if not math.isfinite(capital) or capital <= 0:
        raise ValueError("initial_capital must be finite and positive")
    if (
        type(target_abs_weights) is not tuple
        or len(target_abs_weights) != dataset.n_symbols
    ):
        raise ValueError("target weights must match the symbol roster")
    weights: list[float] = []
    for weight in target_abs_weights:
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise TypeError("target weights must contain non-boolean real values")
        try:
            value = float(weight)
        except OverflowError as error:
            raise ValueError("target weights must be finite") from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("target weights must be finite and non-negative")
        weights.append(value)
    baseline_codes = decode_portfolio_action_codes(
        baseline_action, n_symbols=dataset.n_symbols
    )
    candidate_codes = decode_portfolio_action_codes(
        candidate_action, n_symbols=dataset.n_symbols
    )
    if not callable(reference_policy_factory):
        raise TypeError("reference_policy_factory must be callable")
    baseline_actor = reference_policy_factory()
    candidate_actor = reference_policy_factory()
    if baseline_actor is candidate_actor:
        raise ValueError("reference_policy_factory must return distinct actors")
    for actor in (baseline_actor, candidate_actor):
        if not callable(getattr(actor, "decide_portfolio", None)):
            raise TypeError("reference policy must implement decide_portfolio")
    weights_tuple = tuple(weights)
    baseline = _OneDecisionOverride(
        baseline_actor,
        intervention_index=intervention_index,
        action=baseline_codes,
        target_abs_weights=weights_tuple,
    )
    candidate = _OneDecisionOverride(
        candidate_actor,
        intervention_index=intervention_index,
        action=candidate_codes,
        target_abs_weights=weights_tuple,
    )

    def replay(actor: _OneDecisionOverride) -> SharedCashPortfolioReplayResult:
        return run_shared_portfolio_replay(
            dataset,
            actor,
            start_index=start_index,
            stop_index=stop_index,
            initial_capital=capital,
            execution_cost=execution_cost,
            risk=PreTradeRisk(risk.config),
            market_order_profile=market_order_profile,
            settle_terminal_position=True,
            capture_ledger_evidence=True,
        )

    baseline_result = replay(baseline)
    candidate_result = replay(candidate)
    if (
        baseline.intervention_snapshot is None
        or candidate.intervention_snapshot is None
    ):
        raise RuntimeError("paired replay did not reach the intervention decision")
    if (
        len(baseline.prefix) != intervention_index - start_index
        or baseline.prefix != candidate.prefix
        or baseline.intervention_snapshot != candidate.intervention_snapshot
    ):
        raise RuntimeError("paired reference prefix is not identical")
    if (
        baseline_result.ledger_evidence is None
        or candidate_result.ledger_evidence is None
    ):
        raise RuntimeError("paired replay did not retain native ledger evidence")
    baseline_ledger = baseline_result.ledger_evidence.to_mapping()
    candidate_ledger = candidate_result.ledger_evidence.to_mapping()
    prefix_length = intervention_index - start_index
    baseline_intervals = baseline_ledger["intervals"]
    candidate_intervals = candidate_ledger["intervals"]
    if not isinstance(baseline_intervals, (tuple, list)) or not isinstance(
        candidate_intervals, (tuple, list)
    ):
        raise RuntimeError("paired native intervals must be ordered sequences")
    if baseline_intervals[:prefix_length] != candidate_intervals[:prefix_length]:
        raise RuntimeError("paired native ledger prefix is not identical")
    if baseline_codes == candidate_codes and baseline_ledger != candidate_ledger:
        raise RuntimeError("null intervention changed the native outcome")
    eligible = _settled_label_eligible(
        baseline_result, expected_intervals=stop_index - start_index
    ) and _settled_label_eligible(
        candidate_result, expected_intervals=stop_index - start_index
    )
    delta = (
        (candidate_result.book.portfolio_value - baseline_result.book.portfolio_value)
        / capital
        if eligible
        else None
    )
    return PairedPortfolioActionResult(
        baseline_result=baseline_result,
        candidate_result=candidate_result,
        baseline_snapshot=baseline.intervention_snapshot,
        candidate_snapshot=candidate.intervention_snapshot,
        intervention_index=intervention_index,
        eligible_for_label=eligible,
        delta_per_initial_capital=delta,
    )


__all__ = ["PairedPortfolioActionResult", "run_paired_portfolio_action"]
