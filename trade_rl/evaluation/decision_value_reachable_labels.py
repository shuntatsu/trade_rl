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

Pairs that cannot yield a settled native label are counted, never silently
dropped or fabricated. Losing and zero increments are kept.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.paired_portfolio_action import run_paired_portfolio_action
from trade_rl.evaluation.paired_portfolio_label import (
    SyntheticPairedLabel,
    build_synthetic_paired_label,
)
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.portfolio import PortfolioPolicy


@dataclass(frozen=True, slots=True)
class ReachableLabelCollection:
    """Labels plus the accounting needed to audit the training population."""

    samples: tuple[SyntheticPairedLabel, ...]
    branch_windows: tuple[tuple[int, int, int], ...]
    paired_replay_runs: int
    ineligible_pairs: int
    nonflat_branch_states: int
    skipped_branches_without_reference: int


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
                ineligible += 1
                continue
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
    )


__all__ = ["ReachableLabelCollection", "collect_reachable_state_labels"]
