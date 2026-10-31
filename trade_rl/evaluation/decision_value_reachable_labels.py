"""Native paired labels from reachable (usually non-flat) states (#1025 TD8).

The TD5-TD7 teacher started every branch from a flat 100k account with a
quantity-HOLD reference, so every training state was flat and the label was
really "enter at ``t``, hold, liquidate". Issue #1025 asks for HOLD-relative
increments from states the policy can actually reach.

For each branch time ``t`` this module replays a frozen reference policy ``pi0``
(the B0 cost-aware policy recipe) from a flat account at
``start = max(train_start, t - prefix_bars)`` so the account at ``t`` is whatever
``pi0`` built during that prefix, intervenes once at ``t`` with each candidate
(the baseline is exact-quantity HOLD) and lets ``pi0`` continue to
``t + horizon_bars`` with forced settlement. This is a *truncated-prefix
approximation* of reachability: the book at ``t`` depends on the prefix length
and on ``pi0`` having been run from a flat start, not on the full history that
precedes ``t``.

``pi0`` is supplied per branch by a *provider* given ``(t, start)``. The B0
ridge strategy fail-closes on decisions earlier than its ``fit_cutoff``, so a
single model fitted on the whole train window cannot legally act inside it; the
provider therefore returns a policy factory whose model was fitted causally
before ``start`` (or ``None`` when no sufficiently informed causal reference
exists, in which case the branch is skipped and counted).

Pairs that cannot yield a settled native label retain original branch outcomes,
termination reasons and residual orders/positions for audit. The study runner
blocks survivor-only fitting and may publish an invalid write-once receipt.
Losing and zero eligible increments are kept, never fabricated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.decision_value_ledger_oracle import (
    verify_paired_economic_ledger,
)
from trade_rl.evaluation.paired_portfolio_action import run_paired_portfolio_action
from trade_rl.evaluation.paired_portfolio_label import (
    SyntheticPairedLabel,
    build_synthetic_paired_label,
)
from trade_rl.evaluation.replay import SharedCashPortfolioReplayResult
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.accounting import EconomicTerminationReason
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.portfolio import PortfolioPolicy


@dataclass(frozen=True, slots=True)
class IneligiblePairOutcome:
    """Versioned, audit-only financial endpoint for a pair not fitted as a label.

    A terminated/unsettled branch may have a marked equity and real losses, but
    it is not a comparable, fully settled action-value target.
    """

    intervention_index: int
    prefix_start_index: int
    terminal_index: int
    candidate_action: tuple[int, ...]
    reason_codes: tuple[str, ...]
    baseline: dict[str, object]
    candidate: dict[str, object]

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema": "decision_value_ineligible_pair_v1",
            "intervention_index": self.intervention_index,
            "prefix_start_index": self.prefix_start_index,
            "terminal_index": self.terminal_index,
            "candidate_action": list(self.candidate_action),
            "reason_codes": list(self.reason_codes),
            "baseline": dict(self.baseline),
            "candidate": dict(self.candidate),
            "admissible_for_fit": False,
        }


def _ineligible_arm_endpoint(
    result: SharedCashPortfolioReplayResult,
    *,
    expected_intervals: int,
    arm: str,
    initial_capital: float,
) -> tuple[dict[str, object], list[str]]:
    """Record the original account/ledger endpoint, including losses and failures."""
    ledger = result.ledger_evidence
    if ledger is None:
        raise RuntimeError("ineligible paired replay lost its native ledger evidence")
    mapping = ledger.to_mapping()
    intervals = mapping["intervals"]
    if not isinstance(intervals, (list, tuple)):
        raise RuntimeError("ineligible paired ledger intervals are not ordered")
    pending = mapping["active_order_remainders"]
    if not isinstance(pending, (tuple, list)):
        raise RuntimeError("ineligible pending order remainders are not ordered")
    pending_records: list[list[object]] = []
    for item in pending:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise RuntimeError("invalid ineligible pending order remainder")
        pending_records.append([item[0], item[1]])
    book = result.book
    reason = book.termination_reason
    record: dict[str, object] = {
        "terminal_marked_equity": str(book.portfolio_value),
        "observed_equity_change": str(float(book.portfolio_value) - initial_capital),
        "initial_capital": str(initial_capital),
        "observed_stop_index": ledger.intervals.stop_index,
        "terminal_cash": str(book.cash),
        "terminal_exact_quantities": [str(q) for q in book.exact_quantities],
        "total_cost": str(book.total_cost),
        "funding_pnl": str(book.funding_pnl),
        "insolvent": bool(book.insolvent),
        "termination_reason": (
            reason.value
            if isinstance(reason, EconomicTerminationReason)
            else (str(reason) if reason is not None else None)
        ),
        "observed_intervals": len(intervals),
        "expected_intervals": expected_intervals,
        "pending_order_remainders": pending_records,
        "settled_label_eligible": False,
    }
    problems = []
    if book.insolvent:
        problems.append(f"{arm}_insolvent")
    if reason is not None:
        problems.append(f"{arm}_terminated")
    if len(intervals) != expected_intervals:
        problems.append(f"{arm}_intervals_incomplete")
    if any(q != 0 for q in book.exact_quantities):
        problems.append(f"{arm}_not_flat")
    if pending_records:
        problems.append(f"{arm}_pending_orders")
    record["settled_label_eligible"] = not problems
    return record, problems


@dataclass(frozen=True, slots=True)
class ReachableLabelCollection:
    """Labels plus the accounting needed to audit the training population."""

    samples: tuple[SyntheticPairedLabel, ...]
    branch_windows: tuple[tuple[int, int, int], ...]
    paired_replay_runs: int
    ineligible_pairs: int
    nonflat_branch_states: int
    skipped_branches_without_reference: int
    ineligible_outcomes: tuple[IneligiblePairOutcome, ...]


def collect_reachable_state_labels(
    dataset: MarketDataset,
    *,
    reference_policy_provider: Callable[
        [int, int], Callable[[], PortfolioPolicy] | None
    ],
    reference_policy_digest: str,
    branch_indices: tuple[int, ...],
    train_start_index: int,
    prefix_bars: int,
    horizon_bars: int,
    candidate_actions: tuple[tuple[int, ...], ...],
    target_abs_weights: tuple[float, ...],
    initial_capital: float,
    execution_cost: ExecutionCostConfig,
    risk_config: PreTradeRiskConfig,
    fit_cutoff: np.datetime64,
    source_scope: str,
) -> ReachableLabelCollection:
    """Collect one HOLD-relative native label per (branch time, candidate)."""
    if type(branch_indices) is not tuple or not branch_indices:
        raise ValueError("branch_indices must be a non-empty tuple")
    if type(prefix_bars) is not int or prefix_bars < 0:
        raise ValueError("prefix_bars must be a non-negative native integer")
    if type(horizon_bars) is not int or horizon_bars <= 0:
        raise ValueError("horizon_bars must be a positive native integer")
    if type(candidate_actions) is not tuple or len(candidate_actions) < 2:
        raise ValueError("candidate roster requires HOLD and non-HOLD actions")
    baseline_action = candidate_actions[0]
    samples: list[SyntheticPairedLabel] = []
    invalid_outcomes: list[IneligiblePairOutcome] = []
    windows: list[tuple[int, int, int]] = []
    runs = 0
    ineligible = 0
    nonflat = 0
    skipped = 0
    for t in branch_indices:
        if type(t) is not int or t < train_start_index:
            raise ValueError("branch index precedes the train window")
        start = max(train_start_index, t - prefix_bars)
        stop = t + horizon_bars
        factory = reference_policy_provider(t, start)
        if factory is None:
            skipped += 1
            continue
        windows.append((t, start, stop))
        branch_counted = False
        for candidate in candidate_actions[1:]:
            pair = run_paired_portfolio_action(
                dataset,
                reference_policy_factory=factory,
                start_index=start,
                stop_index=stop,
                intervention_index=t,
                baseline_action=baseline_action,
                candidate_action=candidate,
                target_abs_weights=target_abs_weights,
                initial_capital=initial_capital,
                execution_cost=execution_cost,
                risk=PreTradeRisk(risk_config),
                intervention_projection="posture_v2",
            )
            runs += 2
            if not branch_counted:
                branch_counted = True
                if any(
                    Fraction(q) != 0
                    for q in pair.baseline_snapshot.account.exact_quantities
                ):
                    nonflat += 1
            if not pair.eligible_for_label:
                baseline, baseline_reasons = _ineligible_arm_endpoint(
                    pair.baseline_result,
                    expected_intervals=stop - start,
                    arm="baseline",
                    initial_capital=initial_capital,
                )
                candidate_endpoint, candidate_reasons = _ineligible_arm_endpoint(
                    pair.candidate_result,
                    expected_intervals=stop - start,
                    arm="candidate",
                    initial_capital=initial_capital,
                )
                problems = (*baseline_reasons, *candidate_reasons)
                invalid_outcomes.append(
                    IneligiblePairOutcome(
                        intervention_index=t,
                        prefix_start_index=start,
                        terminal_index=stop,
                        candidate_action=candidate,
                        reason_codes=tuple(problems or ("invalid_endpoint",)),
                        baseline=baseline,
                        candidate=candidate_endpoint,
                    )
                )
                ineligible += 1
                continue
            # Validate the money of both teacher branches *before* fitting.
            # A matching BookState/observer terminal total is not independent.
            if any(
                not verify_paired_economic_ledger(
                    arm,
                    dataset,
                    expected_intervals=stop - start,
                    initial_capital=initial_capital,
                    expected_start_index=start,
                    execution_cost=execution_cost,
                )
                for arm in (pair.baseline_result, pair.candidate_result)
            ):
                raise ValueError("TD8 independent training ledger accounting failed")
            samples.append(
                build_synthetic_paired_label(
                    dataset,
                    pair,
                    baseline_action=baseline_action,
                    candidate_action=candidate,
                    target_abs_weights=target_abs_weights,
                    reference_policy_digest=reference_policy_digest,
                    expected_execution_policy_digest=(
                        execution_cost.execution_policy_digest
                    ),
                    observation_record_available_at=dataset.timestamps[t],
                    settlement_record_available_at=dataset.timestamps[stop],
                    fit_cutoff=fit_cutoff,
                    source_scope=source_scope,
                )
            )
    return ReachableLabelCollection(
        samples=tuple(samples),
        branch_windows=tuple(windows),
        paired_replay_runs=runs,
        ineligible_pairs=ineligible,
        nonflat_branch_states=nonflat,
        skipped_branches_without_reference=skipped,
        ineligible_outcomes=tuple(invalid_outcomes),
    )


__all__ = [
    "IneligiblePairOutcome",
    "ReachableLabelCollection",
    "collect_reachable_state_labels",
]
