"""Synthetic-only causal admission for native paired portfolio action labels.

A completed native simulation-path label is neither an expected action value
nor a point-in-time market evidence receipt. No real-market admission is offered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from trade_rl.data._timestamps import _datetime64_values_to_ns_exact
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.paired_portfolio_action import PairedPortfolioActionResult
from trade_rl.strategies.portfolio import PortfolioDecisionSnapshot
from trade_rl.strategies.portfolio_policy_codec import (
    decode_portfolio_action_codes,
    portfolio_action_to_proposal,
)

_ALLOWED_SOURCE_SCOPES = ("synthetic_only", "real_development")


@dataclass(frozen=True, slots=True)
class SyntheticPairedLabel:
    """A mature synthetic decision-row and one signed native equity delta.

    No native BookState, price suffix, or future-observation array is retained.
    Source arrival and reference-policy digest remain caller declarations,
    not cryptographically verified provider/actor attestations.
    """

    source_scope: str
    dataset_id: str
    feature_config_digest: str
    normalization_digest: str
    execution_policy_digest: str
    reference_policy_digest: str
    observation: PortfolioDecisionSnapshot
    baseline_action: tuple[int, ...]
    candidate_action: tuple[int, ...]
    target_abs_weights: tuple[float, ...]
    decision_time_ns: int
    terminal_time_ns: int
    label_first_available_ns: int
    fit_cutoff_ns: int
    label_delta: float


@dataclass(frozen=True, slots=True)
class SyntheticPairedLabelBatch:
    """Rows sorted chronologically; group IDs disclose overlap, not independence."""

    samples: tuple[SyntheticPairedLabel, ...]
    group_ids: tuple[int, ...]
    fit_cutoff_ns: int
    evaluation_start_ns: int


def _time_ns(value: np.datetime64, *, field: str) -> int:
    if not isinstance(value, np.datetime64):
        raise TypeError(f"{field} must be an explicit numpy datetime64 scalar")
    if np.isnat(value):
        raise ValueError(f"{field} cannot be NaT")
    try:
        normalized = _datetime64_values_to_ns_exact(np.asarray(value))
        return int(normalized.astype(np.int64))
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            f"{field} must be exactly representable as datetime64[ns]"
        ) from error


def _digest(value: str, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _native_ledger(
    pair: PairedPortfolioActionResult,
    dataset: MarketDataset,
    expected_execution_policy_digest: str,
) -> tuple[int, int, float]:
    if not pair.eligible_for_label or pair.delta_per_initial_capital is None:
        raise ValueError("paired result lacks an eligible settled label")
    if pair.baseline_snapshot != pair.candidate_snapshot:
        raise ValueError("paired decision snapshot differs between arms")
    idx = pair.intervention_index
    if type(idx) is not int or not 0 <= idx < dataset.n_bars:
        raise ValueError("paired intervention index is invalid")
    at_decision = int(dataset.timestamps[idx].astype(np.int64))
    if (
        pair.baseline_snapshot.index != idx
        or pair.baseline_snapshot.timestamp_ns != at_decision
        or pair.baseline_snapshot.symbols != dataset.symbols
        or pair.baseline_snapshot.feature_names != dataset.feature_names
        or pair.baseline_snapshot.global_feature_names != dataset.global_feature_names
    ):
        raise ValueError("paired decision snapshot is inconsistent with the Dataset")
    observation = pair.baseline_snapshot
    if (
        not np.array_equal(
            np.asarray(observation.features, dtype=np.float64),
            dataset.features[idx].astype(np.float64),
        )
        or not np.array_equal(
            np.asarray(observation.feature_available, dtype=np.bool_),
            dataset.feature_available[idx],
        )
        or not np.array_equal(
            np.asarray(observation.feature_staleness, dtype=np.float64),
            dataset.resolved_array("feature_staleness")[idx].astype(np.float64),
        )
        or not np.array_equal(
            np.asarray(observation.global_features, dtype=np.float64),
            dataset.global_features[idx].astype(np.float64),
        )
        or not np.array_equal(
            np.asarray(observation.global_feature_available, dtype=np.bool_),
            dataset.resolved_array("global_feature_available")[idx],
        )
    ):
        raise ValueError("paired decision snapshot differs from source Dataset row")
    if any(not ready for row in observation.feature_available for ready in row) or any(
        not ready for ready in observation.global_feature_available
    ):
        raise ValueError("unavailable decision feature must not enter a fit sample")
    native: list[tuple[int, int, float]] = []
    for arm in (pair.baseline_result, pair.candidate_result):
        evidence = arm.ledger_evidence
        if evidence is None:
            raise ValueError("paired native ledger evidence is missing")
        ledger = evidence.to_mapping()
        if ledger["dataset_id"] != dataset.dataset_id:
            raise ValueError("native ledger Dataset identity mismatch")
        if ledger["execution_policy_digest"] != expected_execution_policy_digest:
            raise ValueError("native ledger execution_policy_digest mismatch")
        if ledger["schema_version"] != "shared_cash_portfolio_replay_ledger_v1":
            raise ValueError("unexpected native ledger schema")
        start, stop = ledger["start_index"], ledger["stop_index"]
        intervals = ledger["intervals"]
        if (
            type(start) is not int
            or type(stop) is not int
            or not 0 <= start <= idx < stop < dataset.n_bars
            or not isinstance(intervals, tuple)
            or len(intervals) != stop - start
        ):
            raise ValueError("native ledger window is malformed")
        if (
            arm.book.insolvent
            or arm.book.termination_reason is not None
            or any(quantity != 0 for quantity in arm.book.exact_quantities)
            or ledger["active_order_remainders"]
            or ledger["termination_reason"] is not None
        ):
            raise ValueError("native result is not a settled endpoint")
        raw_terminal = ledger["final_portfolio_value"]
        if (
            isinstance(raw_terminal, bool)
            or not isinstance(raw_terminal, (int, float))
            or not math.isfinite(raw_terminal)
        ):
            raise ValueError("native terminal ledger value is invalid")
        if not math.isclose(
            raw_terminal,
            arm.book.portfolio_value,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError("native terminal ledger differs from the account")
        first = intervals[0]
        if not isinstance(first, dict):
            raise ValueError("native first interval must be a mapping")
        initial_capital = first.get("portfolio_value_before")
        if (
            isinstance(initial_capital, bool)
            or not isinstance(initial_capital, (float, int))
            or not math.isfinite(initial_capital)
            or initial_capital <= 0
        ):
            raise ValueError("native initial capital evidence is invalid")
        native.append((start, stop, float(initial_capital)))
    if native[0] != native[1]:
        raise ValueError("paired arm clock/capital contract differs")
    start, stop, capital = native[0]
    baseline_evidence = pair.baseline_result.ledger_evidence
    assert baseline_evidence is not None
    interval_rows = baseline_evidence.to_mapping()["intervals"]
    if not isinstance(interval_rows, tuple):
        raise ValueError("native interval evidence must be a tuple")
    offset = idx - start
    current_row = interval_rows[offset - 1] if offset else interval_rows[0]
    if not isinstance(current_row, dict):
        raise ValueError("native boundary ledger interval must be a mapping")
    phase = "after" if offset else "before"
    if (
        observation.account.exact_quantities != current_row[f"exact_quantities_{phase}"]
        or not math.isclose(
            observation.account.cash,
            float(current_row[f"cash_{phase}"]),
            rel_tol=0.0,
            abs_tol=1e-10,
        )
        or not math.isclose(
            observation.account.portfolio_value,
            float(current_row[f"portfolio_value_{phase}"]),
            rel_tol=0.0,
            abs_tol=1e-10,
        )
    ):
        raise ValueError("paired decision account differs from native boundary ledger")
    delta = (
        pair.candidate_result.book.portfolio_value
        - pair.baseline_result.book.portfolio_value
    ) / capital
    if not math.isfinite(delta) or not math.isclose(
        delta, pair.delta_per_initial_capital, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("paired native equity delta does not match the ledger")
    return start, stop, delta


def _validate_intervention_actions(
    pair: PairedPortfolioActionResult,
    start: int,
    baseline_action: tuple[int, ...],
    candidate_action: tuple[int, ...],
    target_abs_weights: tuple[float, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    n_symbols = len(pair.baseline_snapshot.symbols)
    baseline = decode_portfolio_action_codes(baseline_action, n_symbols=n_symbols)
    candidate = decode_portfolio_action_codes(candidate_action, n_symbols=n_symbols)
    for arm, action in (
        (pair.baseline_result, baseline),
        (pair.candidate_result, candidate),
    ):
        decisions = arm.decisions
        offset = pair.intervention_index - start
        if offset >= len(decisions):
            raise ValueError("native arm did not record its intervention")
        actual = decisions[offset]
        expected = portfolio_action_to_proposal(
            action, pair.baseline_snapshot, target_abs_weights
        )
        if (
            actual.index != pair.intervention_index
            or actual.proposal_weights != expected.target_weights
            or actual.requested_hold_symbols != expected.hold_current_quantity_symbols
        ):
            raise ValueError("declared action differs from native intervention")
    return baseline, candidate


def build_synthetic_paired_label(
    dataset: MarketDataset,
    pair: PairedPortfolioActionResult,
    *,
    baseline_action: tuple[int, ...],
    candidate_action: tuple[int, ...],
    target_abs_weights: tuple[object, ...],
    reference_policy_digest: str,
    expected_execution_policy_digest: str,
    observation_record_available_at: np.datetime64,
    settlement_record_available_at: np.datetime64,
    fit_cutoff: np.datetime64,
    source_scope: str = "synthetic_only",
) -> SyntheticPairedLabel:
    """Admit one completed native paired label only after it truly matures.

    The two arrival times are explicit fixture/declared assumptions, not
    historical source receipts.
    """
    if source_scope not in _ALLOWED_SOURCE_SCOPES:
        raise ValueError("source_scope must be synthetic_only or real_development")
    if not isinstance(dataset, MarketDataset):
        raise TypeError("dataset must be a MarketDataset")
    if not isinstance(pair, PairedPortfolioActionResult):
        raise TypeError("pair must be a PairedPortfolioActionResult")
    ref = _digest(reference_policy_digest, field="reference_policy_digest")
    execution = _digest(
        expected_execution_policy_digest, field="expected_execution_policy_digest"
    )
    start, stop, delta = _native_ledger(pair, dataset, execution)
    if type(target_abs_weights) is not tuple:
        raise TypeError("target_abs_weights must be a tuple")
    weights_list: list[float] = []
    for weight in target_abs_weights:
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise TypeError("target_abs_weights must contain non-boolean real values")
        value = float(weight)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("target_abs_weights must be finite and non-negative")
        weights_list.append(value)
    weights = tuple(weights_list)
    base, candidate = _validate_intervention_actions(
        pair, start, baseline_action, candidate_action, weights
    )
    decision = int(dataset.timestamps[pair.intervention_index].astype(np.int64))
    terminal = int(dataset.timestamps[stop].astype(np.int64))
    inputs_arrive = _time_ns(
        observation_record_available_at, field="observation_record_available_at"
    )
    settled_arrive = _time_ns(
        settlement_record_available_at, field="settlement_record_available_at"
    )
    cutoff = _time_ns(fit_cutoff, field="fit_cutoff")
    if inputs_arrive > decision:
        raise ValueError("observation source arrived after the decision")
    if settled_arrive < terminal:
        raise ValueError("settlement record cannot precede the terminal event")
    as_of_data = dataset.resolved_array("available_at")
    decision_source_max = int(
        np.max(as_of_data[pair.intervention_index].astype(np.int64))
    )
    if decision_source_max > decision:
        raise ValueError("market source was unavailable at the decision")
    future_source_max = int(
        np.max(as_of_data[pair.intervention_index + 1 : stop + 1].astype(np.int64))
    )
    maturity = max(terminal, settled_arrive, future_source_max)
    if maturity >= cutoff:
        raise ValueError("label maturity must be strictly before fit cutoff")
    return SyntheticPairedLabel(
        source_scope=source_scope,
        dataset_id=dataset.dataset_id,
        feature_config_digest=dataset.feature_config_digest,
        normalization_digest=dataset.normalization_digest,
        execution_policy_digest=execution,
        reference_policy_digest=ref,
        observation=pair.baseline_snapshot,
        baseline_action=base,
        candidate_action=candidate,
        target_abs_weights=weights,
        decision_time_ns=decision,
        terminal_time_ns=terminal,
        label_first_available_ns=maturity,
        fit_cutoff_ns=cutoff,
        label_delta=delta,
    )


def build_synthetic_paired_label_batch(
    samples: tuple[SyntheticPairedLabel, ...],
    *,
    fit_cutoff: np.datetime64,
    evaluation_start: np.datetime64,
    expected_reference_policy_digest: str,
) -> SyntheticPairedLabelBatch:
    """Bind a small fit-only roster and disclose overlapping market horizons."""
    if type(samples) is not tuple or not samples:
        raise ValueError("samples must be a non-empty tuple")
    cutoff = _time_ns(fit_cutoff, field="fit_cutoff")
    evaluation = _time_ns(evaluation_start, field="evaluation_start")
    if evaluation <= cutoff:
        raise ValueError("evaluation_start must be strictly after fit_cutoff")
    ref = _digest(
        expected_reference_policy_digest, field="expected_reference_policy_digest"
    )
    first = samples[0]
    if not isinstance(first, SyntheticPairedLabel):
        raise TypeError("samples must contain SyntheticPairedLabel rows")
    if first.source_scope not in _ALLOWED_SOURCE_SCOPES:
        raise ValueError(
            "only synthetic or real_development paired labels with matching source_scope may be batched"
        )
    roster = (
        first.observation.symbols,
        first.observation.feature_names,
        first.observation.global_feature_names,
        first.feature_config_digest,
        first.normalization_digest,
        first.execution_policy_digest,
        first.target_abs_weights,
    )
    rows: list[SyntheticPairedLabel] = []
    used: set[tuple[str, int, int, tuple[int, ...], tuple[int, ...]]] = set()
    for sample in samples:
        if not isinstance(sample, SyntheticPairedLabel):
            raise TypeError("samples must contain SyntheticPairedLabel rows")
        if sample.source_scope != first.source_scope:
            raise ValueError(
                "only synthetic or real_development paired labels with matching source_scope may be batched"
            )
        if sample.reference_policy_digest != ref:
            raise ValueError("reference_policy_digest does not match the batch")
        if sample.fit_cutoff_ns != cutoff or sample.label_first_available_ns >= cutoff:
            raise ValueError("label maturity/fit cutoff contract mismatch")
        current_roster = (
            sample.observation.symbols,
            sample.observation.feature_names,
            sample.observation.global_feature_names,
            sample.feature_config_digest,
            sample.normalization_digest,
            sample.execution_policy_digest,
            sample.target_abs_weights,
        )
        if current_roster != roster:
            raise ValueError("incompatible action/feature/economic roster in batch")
        if not (
            sample.decision_time_ns
            == sample.observation.timestamp_ns
            < sample.terminal_time_ns
            <= sample.label_first_available_ns
        ):
            raise ValueError("invalid time ordering in synthetic sample")
        if not math.isfinite(sample.label_delta):
            raise ValueError("synthetic label delta must be finite")
        key = (
            sample.dataset_id,
            sample.decision_time_ns,
            sample.terminal_time_ns,
            sample.baseline_action,
            sample.candidate_action,
        )
        if key in used:
            raise ValueError("duplicate paired training sample")
        used.add(key)
        rows.append(sample)
    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.decision_time_ns,
                row.terminal_time_ns,
                row.dataset_id,
                row.candidate_action,
            ),
        )
    )
    group_ids: list[int] = []
    group = -1
    current_end = -(2**63)
    for row in ordered:
        if row.decision_time_ns > current_end:
            group += 1
            current_end = row.terminal_time_ns
        else:
            current_end = max(current_end, row.terminal_time_ns)
        group_ids.append(group)
    return SyntheticPairedLabelBatch(
        samples=ordered,
        group_ids=tuple(group_ids),
        fit_cutoff_ns=cutoff,
        evaluation_start_ns=evaluation,
    )


__all__ = [
    "SyntheticPairedLabel",
    "SyntheticPairedLabelBatch",
    "build_synthetic_paired_label",
    "build_synthetic_paired_label_batch",
]
