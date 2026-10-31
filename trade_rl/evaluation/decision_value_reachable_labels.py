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
from dataclasses import asdict, dataclass, replace
from fractions import Fraction

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.decision_value_ledger_oracle import (
    verify_paired_economic_ledger,
)
from trade_rl.evaluation.paired_portfolio_action import (
    PairedPortfolioActionResult,
    run_paired_portfolio_action,
)
from trade_rl.evaluation.paired_portfolio_label import (
    ReferenceVintage,
    SyntheticPairedLabel,
    build_synthetic_paired_label,
)
from trade_rl.evaluation.replay import SharedCashPortfolioReplayResult
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.accounting import EconomicTerminationReason
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.portfolio import PortfolioPolicy


@dataclass(frozen=True, slots=True)
class ReferencePolicyFactory:
    """Fresh reference instances with content-addressed teacher vintage."""

    build: Callable[[], PortfolioPolicy]
    vintage: ReferenceVintage

    def __call__(self) -> PortfolioPolicy:
        return self.build()


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
    reference_vintage: ReferenceVintage | None = None

    def to_mapping(self) -> dict[str, object]:
        value: dict[str, object] = {
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
        if self.reference_vintage is not None:
            value["reference_vintage"] = self.reference_vintage.to_mapping()
        return value


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
        "requested_stop_index": ledger.intervals.stop_index,
        "observed_stop_index": (
            ledger.intervals.intervals[-1].next_index
            if ledger.intervals.intervals
            else ledger.intervals.start_index
        ),
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


def _eligible_pair_evidence(
    pair: PairedPortfolioActionResult,
    label: SyntheticPairedLabel,
    *,
    vintage: ReferenceVintage,
    reference_policy_digest: str,
    prefix_start_index: int,
    initial_capital: float,
    study_protocol_digest: str | None = None,
    risk_config_digest: str | None = None,
    execution_policy_digest: str | None = None,
    horizon_bars: int | None = None,
    candidate_roster_digest: str | None = None,
    observation_available_at_ns: int | None = None,
) -> dict[str, object]:
    """Persist both native endpoints and a common pre-intervention ledger root.

    The full ledgers can be regenerated and checked against their digests. This
    compact record preserves actual losses and native outcomes rather than
    replacing them with a signed label alone.
    """
    stop = pair.baseline_result.ledger_evidence
    if stop is None:
        raise ValueError("eligible paired replay lacks native ledger evidence")
    terminal = stop.intervals.stop_index
    expected = terminal - prefix_start_index
    endpoints: list[dict[str, object]] = []
    prefix_digests: list[str] = []
    for arm, result in (
        ("baseline", pair.baseline_result),
        ("candidate", pair.candidate_result),
    ):
        endpoint, problems = _ineligible_arm_endpoint(
            result,
            expected_intervals=expected,
            arm=arm,
            initial_capital=initial_capital,
        )
        if problems:
            raise ValueError("eligible paired evidence contains an invalid endpoint")
        ledger = result.ledger_evidence
        assert ledger is not None
        mapping = ledger.to_mapping()
        interval_rows = mapping["intervals"]
        assert isinstance(interval_rows, (tuple, list))
        prefix = interval_rows[: pair.intervention_index - prefix_start_index]
        prefix_digests.append(content_digest(prefix))
        endpoint["ledger_digest"] = content_digest(mapping)
        endpoints.append(endpoint)
    if prefix_digests[0] != prefix_digests[1]:
        raise ValueError("paired native prefix ledgers diverged before intervention")
    prefix_identity = content_digest(
        {
            "dataset_id": label.dataset_id,
            "prefix_start_index": prefix_start_index,
            "intervention_index": pair.intervention_index,
            "native_prefix_digest": prefix_digests[0],
            "reference_vintage_digest": vintage.vintage_digest,
        }
    )
    payload: dict[str, object] = {
        "schema": "decision_value_eligible_pair_evidence_v1",
        "dataset_id": label.dataset_id,
        "prefix_start_index": prefix_start_index,
        "prefix_identity_digest": prefix_identity,
        "intervention_index": pair.intervention_index,
        "decision_time_ns": label.decision_time_ns,
        "terminal_index": terminal,
        "candidate_action": list(label.candidate_action),
        "baseline_action": list(label.baseline_action),
        "reference_policy_digest": reference_policy_digest,
        "reference_vintage": vintage.to_mapping(),
        "label_delta": label.label_delta,
        "baseline": endpoints[0],
        "candidate": endpoints[1],
    }
    if study_protocol_digest is not None:
        if (
            not all(
                type(value) is str and len(value) == 64
                for value in (
                    study_protocol_digest,
                    risk_config_digest,
                    execution_policy_digest,
                    candidate_roster_digest,
                )
            )
            or horizon_bars is None
            or observation_available_at_ns is None
        ):
            raise ValueError("TD8 v4 requires a complete economic/protocol contract")
        payload.update(
            {
                "schema": "decision_value_eligible_pair_evidence_v2",
                "study_protocol_digest": study_protocol_digest,
                "risk_config_digest": risk_config_digest,
                "execution_policy_digest": execution_policy_digest,
                "source_runtime_digest": vintage.source_runtime_digest,
                "initial_capital": float(initial_capital),
                "horizon_bars": horizon_bars,
                "candidate_roster_digest": candidate_roster_digest,
                "observation_available_at_ns": observation_available_at_ns,
                # Bind the native pre-intervention replay to the teacher-vintage
                # identity. A single opaque prefix_identity_digest cannot be
                # reconstructed or checked at the batch boundary.
                "native_prefix_digest": prefix_digests[0],
            }
        )
    return {**payload, "evidence_digest": content_digest(payload)}


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
    eligible_pair_outcomes: tuple[dict[str, object], ...] = ()


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
    require_reference_vintage: bool = False,
    require_v4_evidence: bool = False,
    study_protocol_digest: str | None = None,
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
    if require_v4_evidence:
        if source_scope != "synthetic_only" or not require_reference_vintage:
            raise ValueError(
                "TD8 v4 evidence requires synthetic source and teacher vintage"
            )
        if (
            type(study_protocol_digest) is not str
            or len(study_protocol_digest) != 64
            or any(c not in "0123456789abcdef" for c in study_protocol_digest)
        ):
            raise ValueError("TD8 v4 study protocol digest is required")
    risk_digest = content_digest(asdict(risk_config))
    roster_digest = content_digest([list(action) for action in candidate_actions])
    baseline_action = candidate_actions[0]
    samples: list[SyntheticPairedLabel] = []
    invalid_outcomes: list[IneligiblePairOutcome] = []
    eligible_outcomes: list[dict[str, object]] = []
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
        vintage = (
            factory.vintage if isinstance(factory, ReferencePolicyFactory) else None
        )
        if require_reference_vintage and vintage is None:
            raise ValueError(
                "TD8 reference vintage must be bound to its policy factory"
            )
        if vintage is not None and (
            vintage.source_dataset_id != dataset.dataset_id
            or vintage.prefix_start_index != start
            or vintage.fit_cutoff_ns != int(dataset.timestamps[start].astype(np.int64))
        ):
            raise ValueError("reference vintage/source/prefix identity mismatch")
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
                        reference_vintage=vintage,
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
            label = build_synthetic_paired_label(
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
            if vintage is not None:
                evidence = _eligible_pair_evidence(
                    pair,
                    label,
                    vintage=vintage,
                    reference_policy_digest=reference_policy_digest,
                    prefix_start_index=start,
                    initial_capital=initial_capital,
                    study_protocol_digest=study_protocol_digest
                    if require_v4_evidence
                    else None,
                    risk_config_digest=risk_digest,
                    execution_policy_digest=execution_cost.execution_policy_digest,
                    horizon_bars=horizon_bars,
                    candidate_roster_digest=roster_digest,
                    observation_available_at_ns=int(
                        np.max(
                            dataset.resolved_array("available_at")[t].astype(np.int64)
                        )
                    ),
                )
                label = replace(
                    label,
                    reference_vintage=vintage,
                    pair_evidence_digest=str(evidence["evidence_digest"]),
                    pair_evidence=evidence,
                )
                eligible_outcomes.append(evidence)
            samples.append(label)
    return ReachableLabelCollection(
        samples=tuple(samples),
        branch_windows=tuple(windows),
        paired_replay_runs=runs,
        ineligible_pairs=ineligible,
        nonflat_branch_states=nonflat,
        skipped_branches_without_reference=skipped,
        ineligible_outcomes=tuple(invalid_outcomes),
        eligible_pair_outcomes=tuple(eligible_outcomes),
    )


__all__ = [
    "IneligiblePairOutcome",
    "ReferencePolicyFactory",
    "ReachableLabelCollection",
    "collect_reachable_state_labels",
]
