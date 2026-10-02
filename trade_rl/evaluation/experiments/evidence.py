"""Study-owned multi-seed candidate evidence sets."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import cast

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import open_regular_binary
from trade_rl.data import (
    MarketDataset,
    PublishedDatasetArtifact,
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)
from trade_rl.evaluation.experiments.contracts import ResolvedRunConfig, StudyPlan
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.store import StudyStore
from trade_rl.evaluation.runs import (
    CandidateRunConfig,
    LoadedCandidateRun,
    build_candidate_run_provenance,
    execute_candidate_run,
    inspect_candidate_run_artifact,
    load_candidate_run_artifact,
    publish_candidate_run,
    resolve_candidate_run_spec,
)

_EVIDENCE_SCHEMA = "controlled_evidence_set_v1"
_SOURCE_TIME_TOLERANCE_HOURS = 1e-12


@dataclass(frozen=True, slots=True)
class EvidenceSet:
    """Raw Study-owned multi-seed evidence identity."""

    fingerprint: str
    semantic_config_digest: str
    ppo_seeds: tuple[int, ...]
    run_digests: tuple[tuple[int, str], ...]
    research_context_digest: str
    schema_version: str = _EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        try:
            require_sha256(self.fingerprint, field="fingerprint")
            require_sha256(
                self.semantic_config_digest,
                field="semantic_config_digest",
            )
            require_sha256(
                self.research_context_digest,
                field="research_context_digest",
            )
        except ValueError as error:
            raise ArtifactIntegrityError(str(error)) from error
        if self.schema_version != _EVIDENCE_SCHEMA:
            raise ArtifactIntegrityError("unsupported EvidenceSet schema")
        if len(self.ppo_seeds) < 2 or len(set(self.ppo_seeds)) != len(self.ppo_seeds):
            raise ArtifactIntegrityError("EvidenceSet seed roster must be unique")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in self.ppo_seeds
        ):
            raise ArtifactIntegrityError(
                "EvidenceSet seeds must be non-negative integers"
            )
        if tuple(seed for seed, _ in self.run_digests) != self.ppo_seeds:
            raise ArtifactIntegrityError(
                "EvidenceSet run digests must follow the seed roster"
            )
        for _, digest in self.run_digests:
            try:
                require_sha256(digest, field="run artifact digest")
            except ValueError as error:
                raise ArtifactIntegrityError(str(error)) from error
        expected = _evidence_fingerprint(
            semantic_config_digest=self.semantic_config_digest,
            ppo_seeds=self.ppo_seeds,
            run_digests=self.run_digests,
            research_context_digest=self.research_context_digest,
        )
        if self.fingerprint != expected:
            raise ArtifactIntegrityError("EvidenceSet fingerprint mismatch")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "semantic_config_digest": self.semantic_config_digest,
            "ppo_seeds": list(self.ppo_seeds),
            "run_digests": [
                {"ppo_seed": seed, "artifact_digest": digest}
                for seed, digest in self.run_digests
            ],
            "research_context_digest": self.research_context_digest,
        }


@dataclass(frozen=True, slots=True)
class LoadedEvidenceSet:
    """Verified EvidenceSet plus its Study-owned Candidate Runs."""

    evidence: EvidenceSet
    semantic_config: dict[str, object]
    runs: Mapping[int, LoadedCandidateRun]


def _evidence_fingerprint(
    *,
    semantic_config_digest: str,
    ppo_seeds: tuple[int, ...],
    run_digests: tuple[tuple[int, str], ...],
    research_context_digest: str,
) -> str:
    return content_digest(
        {
            "schema_version": _EVIDENCE_SCHEMA,
            "semantic_config_digest": semantic_config_digest,
            "ppo_seeds": list(ppo_seeds),
            "run_digests": [
                {"ppo_seed": seed, "artifact_digest": digest}
                for seed, digest in run_digests
            ],
            "research_context_digest": research_context_digest,
        }
    )


def _without_seed(config: ResolvedRunConfig) -> dict[str, object]:
    payload = config.to_payload()
    payload.pop("ppo_seed")
    return payload


def _check_study_fixed_config(plan: StudyPlan, resolved: ResolvedRunConfig) -> None:
    baseline = plan.baseline_config
    for field in plan.FIXED_RESOLVED_FIELDS:
        actual = getattr(resolved, field)
        expected = getattr(baseline, field)
        if actual != expected:
            raise ArtifactIntegrityError(f"Study-fixed {field} drifted")


def _verify_plan_inputs(
    *,
    dataset_root: str | Path,
    plan: StudyPlan,
    research_context_digest: str,
) -> tuple[PublishedDatasetArtifact, MarketDataset]:
    try:
        require_sha256(
            research_context_digest,
            field="research_context_digest",
        )
    except ValueError as error:
        raise ArtifactIntegrityError(str(error)) from error

    artifact = inspect_published_market_dataset_artifact(dataset_root)
    dataset = load_market_dataset_artifact(dataset_root)
    if dataset.dataset_id != plan.dataset_id:
        raise ArtifactIntegrityError("Study dataset identity mismatch")
    if artifact.schema_version != plan.dataset_artifact_schema:
        raise ArtifactIntegrityError("Study dataset artifact schema mismatch")
    if artifact.artifact_digest != plan.dataset_artifact_digest:
        raise ArtifactIntegrityError("Study dataset artifact digest mismatch")
    if tuple(dataset.symbols) != plan.symbols:
        raise ArtifactIntegrityError("Study dataset symbol roster mismatch")

    provenance = build_candidate_run_provenance(
        research_context_digest=research_context_digest,
    )
    if provenance.get("implementation_digest") != plan.implementation_digest:
        raise ArtifactIntegrityError("Study implementation provenance mismatch")
    if provenance.get("runtime_environment_digest") != plan.runtime_environment_digest:
        raise ArtifactIntegrityError("Study runtime provenance mismatch")
    return artifact, dataset


def _candidate_source_index(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactIntegrityError(f"candidate source ledger {field} is invalid")
    return value


def _require_candidate_source_prices(
    actual: object,
    expected: np.ndarray,
    *,
    source_row: str,
    processing_index: int,
) -> None:
    _require_candidate_source_values(
        actual,
        expected,
        source_row=source_row,
        processing_index=processing_index,
    )


def _require_candidate_source_values(
    actual: object,
    expected: np.ndarray,
    *,
    source_row: str,
    processing_index: int,
) -> None:
    try:
        actual_values = np.asarray(actual)
    except (TypeError, ValueError) as error:
        raise ArtifactIntegrityError(
            f"candidate ledger source {source_row} is invalid at index "
            f"{processing_index}"
        ) from error
    if expected.dtype.kind == "b":
        if actual_values.dtype.kind != "b":
            raise ArtifactIntegrityError(
                f"candidate ledger source {source_row} is invalid at index "
                f"{processing_index}"
            )
    elif actual_values.dtype.kind not in "iuf":
        raise ArtifactIntegrityError(
            f"candidate ledger source {source_row} is invalid at index "
            f"{processing_index}"
        )
    try:
        actual_values = np.asarray(actual_values, dtype=expected.dtype)
        expected_values = np.asarray(expected, dtype=expected.dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise ArtifactIntegrityError(
            f"candidate ledger source {source_row} is invalid at index "
            f"{processing_index}"
        ) from error
    if actual_values.shape != expected_values.shape or not np.array_equal(
        actual_values, expected_values
    ):
        raise ArtifactIntegrityError(
            f"candidate ledger does not match source {source_row} at index "
            f"{processing_index}"
        )


def _require_candidate_source_scalar(
    actual: object,
    expected: float,
    *,
    source_row: str,
    processing_index: int,
) -> None:
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not np.isfinite(actual)
        or float(actual) != expected
    ):
        raise ArtifactIntegrityError(
            f"candidate ledger does not match source {source_row} at index "
            f"{processing_index}"
        )


def _candidate_source_year_fractions(
    dataset: MarketDataset,
    *,
    start_index: int,
    processing_index: int,
) -> tuple[float, float]:
    elapsed_hours = dataset.elapsed_hours(start_index, processing_index)
    elapsed_year_fraction = dataset.elapsed_year_fraction(
        start_index,
        processing_index,
    )
    if elapsed_hours <= dataset.bar_hours + _SOURCE_TIME_TOLERANCE_HOURS:
        return elapsed_year_fraction, 0.0
    processing_year_fraction = elapsed_year_fraction * dataset.bar_hours / elapsed_hours
    return processing_year_fraction, elapsed_year_fraction - processing_year_fraction


def _candidate_source_exact_quantities(
    interval: Mapping[str, object],
    *,
    dataset: MarketDataset,
) -> tuple[Fraction, ...]:
    raw_quantities = interval.get("exact_quantities_before")
    if (
        not isinstance(raw_quantities, Sequence)
        or isinstance(raw_quantities, (str, bytes, bytearray))
        or len(raw_quantities) != dataset.n_symbols
    ):
        raise ArtifactIntegrityError(
            "candidate source ledger interval quantities are invalid"
        )
    try:
        return tuple(Fraction(value) for value in raw_quantities)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ArtifactIntegrityError(
            "candidate source ledger interval quantities are invalid"
        ) from error


def _candidate_shared_cash_ledger(
    candidate_summary: Mapping[str, object],
    *,
    dataset: MarketDataset,
    expected_dataset_artifact_digest: str,
) -> Mapping[str, object]:
    """Validate the identities and schema needed to bind candidate prices."""
    try:
        require_sha256(
            expected_dataset_artifact_digest,
            field="expected_dataset_artifact_digest",
        )
    except ValueError as error:
        raise ArtifactIntegrityError(str(error)) from error

    if candidate_summary.get("dataset_id") != dataset.dataset_id:
        raise ArtifactIntegrityError("candidate source Dataset identity mismatch")
    dataset_artifact = candidate_summary.get("dataset_artifact")
    if (
        not isinstance(dataset_artifact, Mapping)
        or dataset_artifact.get("artifact_digest") != expected_dataset_artifact_digest
    ):
        raise ArtifactIntegrityError(
            "candidate source Dataset artifact digest mismatch"
        )

    portfolio = candidate_summary.get("shared_cash_ppo")
    if not isinstance(portfolio, Mapping):
        raise ArtifactIntegrityError("shared-cash candidate source ledger is missing")
    ledger_evidence = portfolio.get("ledger_evidence")
    if not isinstance(ledger_evidence, Mapping):
        raise ArtifactIntegrityError("shared-cash candidate source ledger is missing")
    if ledger_evidence.get("schema_version") != "shared_cash_replay_ledger_v3":
        raise ArtifactIntegrityError("candidate source ledger schema is unsupported")
    ledger = ledger_evidence.get("payload")
    if not isinstance(ledger, Mapping):
        raise ArtifactIntegrityError("candidate source ledger payload is invalid")
    if (
        ledger.get("schema_version") != "shared_cash_replay_ledger_v3"
        or ledger.get("dataset_id") != dataset.dataset_id
    ):
        raise ArtifactIntegrityError("candidate source ledger identity mismatch")
    return ledger


def _validate_candidate_source_mark_transition(
    transition: Mapping[str, object],
    *,
    dataset: MarketDataset,
    mark_prices: np.ndarray,
    expected_index: int,
) -> bool:
    transition_type = transition.get("transition_type")
    if transition_type not in ("mark_revaluation", "funding_mark"):
        return False

    processing_index = _candidate_source_index(
        transition.get("processing_index"),
        field="transition processing_index",
    )
    if processing_index != expected_index:
        raise ArtifactIntegrityError(
            "candidate source ledger mark index does not match its interval"
        )
    transition_evidence = transition.get("evidence")
    state_after = transition.get("state_after")
    if not isinstance(transition_evidence, Mapping) or not isinstance(
        state_after, Mapping
    ):
        raise ArtifactIntegrityError(
            "candidate source ledger mark evidence is incomplete"
        )

    is_open_mark = transition_type == "mark_revaluation"
    if is_open_mark:
        if transition_evidence.get("mark_phase") != "open":
            raise ArtifactIntegrityError("candidate source open mark phase is invalid")
        source_row = "open row"
        expected_prices = dataset.open[processing_index]
    else:
        source_row = "mark_price row"
        expected_prices = mark_prices[processing_index]

    for actual in (
        transition_evidence.get("mark_prices"),
        state_after.get("mark_prices"),
    ):
        _require_candidate_source_prices(
            actual,
            expected_prices,
            source_row=source_row,
            processing_index=processing_index,
        )
    return is_open_mark


def _validate_candidate_source_funding_events(
    interval: Mapping[str, object],
    *,
    dataset: MarketDataset,
    mark_prices: np.ndarray,
    expected_index: int,
) -> None:
    raw_events = interval.get("funding_events")
    if not isinstance(raw_events, Sequence) or isinstance(
        raw_events, (str, bytes, bytearray)
    ):
        raise ArtifactIntegrityError(
            "candidate source ledger funding events are missing"
        )
    expected_due = dataset.resolved_array("funding_due")[expected_index]
    expected_count = 1 if np.any(expected_due) else 0
    if len(raw_events) != expected_count:
        raise ArtifactIntegrityError(
            "candidate source ledger funding boundaries do not match source due row"
        )
    if expected_count == 0:
        return

    event = raw_events[0]
    if not isinstance(event, Mapping):
        raise ArtifactIntegrityError(
            "candidate source ledger funding boundary is invalid"
        )
    if (
        _candidate_source_index(
            event.get("processing_index"), field="funding processing_index"
        )
        != expected_index
    ):
        raise ArtifactIntegrityError(
            "candidate source ledger funding index does not match its interval"
        )
    timestamp_ns = int(
        dataset.timestamps[expected_index].astype("datetime64[ns]").astype(np.int64)
    )
    if event.get("timestamp_ns") != timestamp_ns:
        raise ArtifactIntegrityError(
            "candidate source ledger funding timestamp does not match source time row"
        )
    source_vectors = (
        ("funding_due", expected_due, "funding_due row"),
        (
            "funding_rates",
            dataset.funding_rate[expected_index],
            "funding_rate row",
        ),
        ("mark_prices", mark_prices[expected_index], "mark_price row"),
        (
            "contract_multipliers",
            dataset.resolved_array("contract_multipliers"),
            "contract_multipliers row",
        ),
    )
    for field, expected, source_row in source_vectors:
        _require_candidate_source_values(
            event.get(field),
            np.asarray(expected),
            source_row=source_row,
            processing_index=expected_index,
        )


def _validate_candidate_source_transition(
    transition: Mapping[str, object],
    *,
    dataset: MarketDataset,
    mark_prices: np.ndarray,
    expected_index: int,
    processing_year_fraction: float,
    gap_year_fraction: float,
    last_mark_source: tuple[np.ndarray, str] | None,
) -> tuple[str, str | None, tuple[np.ndarray, str] | None]:
    transition_type = transition.get("transition_type")
    if not isinstance(transition_type, str):
        raise ArtifactIntegrityError(
            "candidate source ledger accounting transition type is invalid"
        )
    if (
        _candidate_source_index(
            transition.get("processing_index"),
            field="transition processing_index",
        )
        != expected_index
    ):
        raise ArtifactIntegrityError(
            "candidate source ledger accounting transition index is invalid"
        )
    evidence = transition.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ArtifactIntegrityError(
            "candidate source ledger accounting transition evidence is missing"
        )

    carry_phase: str | None = None
    if transition_type == "mark_revaluation":
        _validate_candidate_source_mark_transition(
            transition,
            dataset=dataset,
            mark_prices=mark_prices,
            expected_index=expected_index,
        )
        last_mark_source = (dataset.open[expected_index], "open row")
    elif transition_type == "fill":
        # Fill events are cross-checked against ledger events by artifact validation;
        # this validator binds Dataset-backed accounting inputs.
        pass
    elif transition_type == "funding_mark":
        _validate_candidate_source_mark_transition(
            transition,
            dataset=dataset,
            mark_prices=mark_prices,
            expected_index=expected_index,
        )
        last_mark_source = (mark_prices[expected_index], "mark_price row")
    elif transition_type == "split":
        _require_candidate_source_values(
            evidence.get("split_factors"),
            dataset.resolved_array("split_factor")[expected_index],
            source_row="split_factor row",
            processing_index=expected_index,
        )
    elif transition_type == "delisting_settlement":
        _require_candidate_source_values(
            evidence.get("inactive_mask"),
            ~dataset.resolved_array("asset_active")[expected_index],
            source_row="asset_active row",
            processing_index=expected_index,
        )
        _require_candidate_source_prices(
            evidence.get("open_prices"),
            dataset.open[expected_index],
            source_row="open row",
            processing_index=expected_index,
        )
        _require_candidate_source_values(
            evidence.get("delisting_recovery"),
            dataset.resolved_array("delisting_recovery")[expected_index],
            source_row="delisting_recovery row",
            processing_index=expected_index,
        )
    elif transition_type == "dividend":
        _require_candidate_source_values(
            evidence.get("dividend_per_unit"),
            dataset.resolved_array("dividend")[expected_index],
            source_row="dividend row",
            processing_index=expected_index,
        )
    elif transition_type == "cash_interest":
        raw_phase = evidence.get("carry_phase")
        if raw_phase not in {"gap", "processing"}:
            raise ArtifactIntegrityError(
                "candidate source ledger cash-interest phase is invalid"
            )
        carry_phase = cast(str, raw_phase)
        expected_fraction = (
            gap_year_fraction if carry_phase == "gap" else processing_year_fraction
        )
        _require_candidate_source_scalar(
            evidence.get("annual_rate"),
            float(dataset.resolved_array("cash_rate")[expected_index]),
            source_row="cash_rate row",
            processing_index=expected_index,
        )
        _require_candidate_source_scalar(
            evidence.get("year_fraction"),
            expected_fraction,
            source_row="elapsed-time row",
            processing_index=expected_index,
        )
    elif transition_type == "borrow_charge":
        raw_phase = evidence.get("carry_phase")
        if raw_phase not in {"gap", "processing"}:
            raise ArtifactIntegrityError(
                "candidate source ledger borrow-charge phase is invalid"
            )
        carry_phase = cast(str, raw_phase)
        expected_fraction = (
            gap_year_fraction if carry_phase == "gap" else processing_year_fraction
        )
        _require_candidate_source_values(
            evidence.get("borrow_rate"),
            dataset.resolved_array("borrow_rate")[expected_index],
            source_row="borrow_rate row",
            processing_index=expected_index,
        )
        _require_candidate_source_scalar(
            evidence.get("year_fraction"),
            expected_fraction,
            source_row="elapsed-time row",
            processing_index=expected_index,
        )
    elif transition_type == "termination_flatten":
        if last_mark_source is None:
            raise ArtifactIntegrityError(
                "candidate source ledger termination has no source mark"
            )
        expected_prices, source_row = last_mark_source
        _require_candidate_source_prices(
            evidence.get("liquidation_prices"),
            expected_prices,
            source_row=source_row,
            processing_index=expected_index,
        )
    else:
        raise ArtifactIntegrityError(
            "candidate source ledger accounting transition type is unsupported"
        )
    return transition_type, carry_phase, last_mark_source


def _validate_candidate_source_interval(
    interval: Mapping[str, object],
    *,
    dataset: MarketDataset,
    mark_prices: np.ndarray,
    expected_start: int,
) -> None:
    expected_next = expected_start + 1
    if (
        _candidate_source_index(
            interval.get("start_index"), field="interval start_index"
        )
        != expected_start
        or _candidate_source_index(
            interval.get("next_index"), field="interval next_index"
        )
        != expected_next
    ):
        raise ArtifactIntegrityError(
            "candidate source ledger interval order is invalid"
        )

    transitions = interval.get("accounting_transitions")
    if not isinstance(transitions, Sequence) or isinstance(
        transitions, (str, bytes, bytearray)
    ):
        raise ArtifactIntegrityError(
            "candidate source ledger accounting transitions are missing"
        )
    processing_year_fraction, gap_year_fraction = _candidate_source_year_fractions(
        dataset,
        start_index=expected_start,
        processing_index=expected_next,
    )
    counts: dict[str, int] = {}
    cash_phases: dict[str, int] = {}
    borrow_phases: dict[str, int] = {}
    last_mark_source: tuple[np.ndarray, str] | None = None
    for transition in transitions:
        if not isinstance(transition, Mapping):
            raise ArtifactIntegrityError(
                "candidate source ledger accounting transition is invalid"
            )
        transition_type, carry_phase, last_mark_source = (
            _validate_candidate_source_transition(
                transition,
                dataset=dataset,
                mark_prices=mark_prices,
                expected_index=expected_next,
                processing_year_fraction=processing_year_fraction,
                gap_year_fraction=gap_year_fraction,
                last_mark_source=last_mark_source,
            )
        )
        counts[transition_type] = counts.get(transition_type, 0) + 1
        if transition_type == "cash_interest" and carry_phase is not None:
            cash_phases[carry_phase] = cash_phases.get(carry_phase, 0) + 1
        if transition_type == "borrow_charge" and carry_phase is not None:
            borrow_phases[carry_phase] = borrow_phases.get(carry_phase, 0) + 1

    split_factors = dataset.resolved_array("split_factor")[expected_next]
    expected_split_count = int(np.any(np.abs(split_factors - 1.0) > 1e-12))
    exact_quantities = _candidate_source_exact_quantities(interval, dataset=dataset)
    inactive = ~dataset.resolved_array("asset_active")[expected_next]
    expected_delisting_count = int(
        any(
            inactive[index] and quantity != 0
            for index, quantity in enumerate(exact_quantities)
        )
    )
    expected_gap_phases = int(gap_year_fraction > 0.0)
    if (
        counts.get("mark_revaluation", 0) != 1
        or counts.get("funding_mark", 0) != 1
        or counts.get("dividend", 0) != 1
        or counts.get("split", 0) != expected_split_count
        or counts.get("delisting_settlement", 0) != expected_delisting_count
        or counts.get("cash_interest", 0) != 1 + expected_gap_phases
        or counts.get("borrow_charge", 0) != 1 + expected_gap_phases
        or cash_phases.get("processing", 0) != 1
        or cash_phases.get("gap", 0) != expected_gap_phases
        or borrow_phases.get("processing", 0) != 1
        or borrow_phases.get("gap", 0) != expected_gap_phases
    ):
        raise ArtifactIntegrityError(
            "candidate source ledger accounting transitions do not cover source rows"
        )
    _validate_candidate_source_funding_events(
        interval,
        dataset=dataset,
        mark_prices=mark_prices,
        expected_index=expected_next,
    )


def _validate_shared_cash_candidate_source_binding(
    *,
    candidate_summary: Mapping[str, object],
    dataset: MarketDataset,
    expected_dataset_artifact_digest: str,
) -> None:
    """Bind shared-cash accounting evidence to a content-verified Dataset."""
    ledger = _candidate_shared_cash_ledger(
        candidate_summary,
        dataset=dataset,
        expected_dataset_artifact_digest=expected_dataset_artifact_digest,
    )
    start_index = _candidate_source_index(
        ledger.get("start_index"), field="start_index"
    )
    stop_index = _candidate_source_index(ledger.get("stop_index"), field="stop_index")
    if not 0 <= start_index < stop_index < dataset.n_bars:
        raise ArtifactIntegrityError("candidate source ledger window is invalid")

    mark_prices = dataset.resolved_array("mark_price")
    _require_candidate_source_prices(
        ledger.get("initial_mark_prices"),
        mark_prices[start_index],
        source_row="mark_price row",
        processing_index=start_index,
    )
    intervals = ledger.get("intervals")
    if (
        not isinstance(intervals, Sequence)
        or isinstance(intervals, (str, bytes, bytearray))
        or len(intervals) != stop_index - start_index
    ):
        raise ArtifactIntegrityError("candidate source ledger intervals are incomplete")
    for offset, interval in enumerate(intervals):
        if not isinstance(interval, Mapping):
            raise ArtifactIntegrityError("candidate source ledger interval is invalid")
        _validate_candidate_source_interval(
            interval,
            dataset=dataset,
            mark_prices=mark_prices,
            expected_start=start_index + offset,
        )
    if not dataset.identity_verified:
        raise ArtifactIntegrityError(
            "candidate source Dataset content identity mismatch"
        )


def _run_return_map(
    run: LoadedCandidateRun,
    *,
    expected_symbols: tuple[str, ...],
) -> dict[tuple[str, str], np.ndarray]:
    symbols = run.summary.get("symbols")
    if (
        not isinstance(symbols, Sequence)
        or isinstance(symbols, (str, bytes, bytearray))
        or tuple(symbols) != expected_symbols
    ):
        raise ArtifactIntegrityError("EvidenceSet symbol roster mismatch")
    by_symbol = run.summary.get("by_symbol")
    if (
        not isinstance(by_symbol, Sequence)
        or isinstance(by_symbol, (str, bytes, bytearray))
        or len(by_symbol) != len(expected_symbols)
    ):
        raise ArtifactIntegrityError("EvidenceSet symbol results are incomplete")

    result: dict[tuple[str, str], np.ndarray] = {}
    for expected_symbol, entry in zip(expected_symbols, by_symbol, strict=True):
        if not isinstance(entry, Mapping) or entry.get("symbol") != expected_symbol:
            raise ArtifactIntegrityError("EvidenceSet symbol result ordering mismatch")
        strategies = entry.get("strategies")
        if not isinstance(strategies, Sequence) or isinstance(
            strategies, (str, bytes, bytearray)
        ):
            raise ArtifactIntegrityError("EvidenceSet strategy results are malformed")
        names: set[str] = set()
        for strategy in strategies:
            if not isinstance(strategy, Mapping):
                raise ArtifactIntegrityError("EvidenceSet strategy entry is malformed")
            name = strategy.get("name")
            return_key = strategy.get("return_key")
            if not isinstance(name, str) or not isinstance(return_key, str):
                raise ArtifactIntegrityError(
                    "EvidenceSet strategy evidence is malformed"
                )
            if name in names:
                raise ArtifactIntegrityError(
                    "EvidenceSet strategy roster has duplicates"
                )
            names.add(name)
            values = run.returns.get(return_key)
            if values is None:
                raise ArtifactIntegrityError("EvidenceSet return key is missing")
            result[(expected_symbol, name)] = values
        if names != frozenset(StudyPlan.STRATEGY_NAMES):
            raise ArtifactIntegrityError("EvidenceSet strategy roster mismatch")
    return result


def _run_summary_seed(run: LoadedCandidateRun) -> int:
    candidate_config = run.summary.get("candidate_config")
    if not isinstance(candidate_config, Mapping):
        raise ArtifactIntegrityError("EvidenceSet candidate_config is malformed")
    seed = candidate_config.get("ppo_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ArtifactIntegrityError("EvidenceSet PPO seed evidence is malformed")
    return seed


def _verify_seed_invariance(
    runs: Mapping[int, LoadedCandidateRun],
    *,
    seeds: tuple[int, ...],
    symbols: tuple[str, ...],
    research_context_digest: str,
) -> None:
    first_seed = seeds[0]
    first_map = _run_return_map(runs[first_seed], expected_symbols=symbols)
    for seed in seeds:
        run = runs[seed]
        if run.provenance.get("research_context_digest") != research_context_digest:
            raise ArtifactIntegrityError("EvidenceSet research context mismatch")
        if _run_summary_seed(run) != seed:
            raise ArtifactIntegrityError("EvidenceSet PPO seed evidence mismatch")
        current_map = _run_return_map(run, expected_symbols=symbols)
        if seed == first_seed:
            continue
        for symbol in symbols:
            for strategy in StudyPlan.PPO_SEED_INVARIANT_STRATEGY_NAMES:
                if not np.array_equal(
                    first_map[(symbol, strategy)],
                    current_map[(symbol, strategy)],
                ):
                    raise ArtifactIntegrityError(
                        f"deterministic strategy {strategy} drifted across PPO seeds"
                    )


def execute_evidence_set(
    *,
    store: StudyStore,
    target: str | Path,
    dataset_root: str | Path,
    plan: StudyPlan,
    config: CandidateRunConfig,
    research_context_digest: str,
) -> EvidenceSet:
    """Generate, verify, and atomically publish one Study-owned EvidenceSet."""

    artifact, dataset = _verify_plan_inputs(
        dataset_root=dataset_root,
        plan=plan,
        research_context_digest=research_context_digest,
    )
    completed: EvidenceSet | None = None

    def builder(staging: Path) -> None:
        nonlocal completed
        loaded_runs: dict[int, LoadedCandidateRun] = {}
        run_digests: list[tuple[int, str]] = []
        normalized_contract: ResolvedRunConfig | None = None
        seedless_contract: dict[str, object] | None = None

        for seed in plan.ppo_seeds:
            seed_config = replace(config, ppo_seed=seed)
            spec = resolve_candidate_run_spec(
                dataset,
                dataset_artifact_schema=artifact.schema_version,
                dataset_artifact_digest=artifact.artifact_digest,
                config=seed_config,
                execution_overlay=plan.baseline_config.execution_overlay,
            )
            resolved = ResolvedRunConfig.from_candidate_spec(spec)
            _check_study_fixed_config(plan, resolved)
            if normalized_contract is None:
                normalized_contract = resolved
                seedless_contract = _without_seed(resolved)
            elif _without_seed(resolved) != seedless_contract:
                raise ArtifactIntegrityError(
                    "EvidenceSet resolved config changed beyond ppo_seed"
                )

            before = build_candidate_run_provenance(
                research_context_digest=research_context_digest,
            )
            if before.get("implementation_digest") != plan.implementation_digest:
                raise ArtifactIntegrityError(
                    "Study implementation provenance changed during EvidenceSet"
                )
            if (
                before.get("runtime_environment_digest")
                != plan.runtime_environment_digest
            ):
                raise ArtifactIntegrityError(
                    "Study runtime provenance changed during EvidenceSet"
                )
            result = execute_candidate_run(dataset, spec)
            after = build_candidate_run_provenance(
                research_context_digest=research_context_digest,
            )
            if (
                before.get("implementation_digest"),
                before.get("runtime_environment_digest"),
            ) != (
                after.get("implementation_digest"),
                after.get("runtime_environment_digest"),
            ):
                raise ArtifactIntegrityError("Run provenance changed during execution")

            run_root = staging / "runs" / f"seed-{seed}"
            publish_candidate_run(run_root, result, after)
            loaded = load_candidate_run_artifact(run_root)
            if plan.is_ppo_shared_cash_holding_duration_study:
                _validate_shared_cash_candidate_source_binding(
                    candidate_summary=loaded.summary,
                    dataset=dataset,
                    expected_dataset_artifact_digest=artifact.artifact_digest,
                )
            identity = inspect_candidate_run_artifact(run_root)
            loaded_runs[seed] = loaded
            run_digests.append((seed, identity.artifact_digest))

        if normalized_contract is None or seedless_contract is None:
            raise ArtifactIntegrityError("EvidenceSet contains no seed Runs")
        _verify_seed_invariance(
            loaded_runs,
            seeds=plan.ppo_seeds,
            symbols=plan.symbols,
            research_context_digest=research_context_digest,
        )
        run_digest_tuple = tuple(run_digests)
        semantic_config_digest = content_digest(seedless_contract)
        fingerprint = _evidence_fingerprint(
            semantic_config_digest=semantic_config_digest,
            ppo_seeds=plan.ppo_seeds,
            run_digests=run_digest_tuple,
            research_context_digest=research_context_digest,
        )
        evidence = EvidenceSet(
            fingerprint=fingerprint,
            semantic_config_digest=semantic_config_digest,
            ppo_seeds=plan.ppo_seeds,
            run_digests=run_digest_tuple,
            research_context_digest=research_context_digest,
        )
        manifest = evidence.to_payload()
        manifest["semantic_config"] = seedless_contract
        atomic_write_bytes(staging / "manifest.json", canonical_json_bytes(manifest))
        completed = evidence

    with store.mutation_lock():
        store.publish_directory_once(target, builder)
    if completed is None:
        raise ArtifactIntegrityError("EvidenceSet publication did not complete")
    return completed


def _read_manifest(root: Path) -> dict[str, object]:
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ArtifactIntegrityError("EvidenceSet manifest must not be a symlink")
    try:
        with open_regular_binary(manifest_path, field="EvidenceSet manifest") as handle:
            payload = handle.read()
        raw = cast(object, json.loads(payload.decode("utf-8")))
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError("EvidenceSet manifest is malformed") from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ArtifactIntegrityError("EvidenceSet manifest must be a JSON object")
    return cast(dict[str, object], raw)


def _parse_evidence(
    manifest: dict[str, object],
) -> tuple[EvidenceSet, dict[str, object]]:
    semantic_config = manifest.get("semantic_config")
    seeds_raw = manifest.get("ppo_seeds")
    run_digests_raw = manifest.get("run_digests")
    if not isinstance(semantic_config, dict) or any(
        not isinstance(key, str) for key in semantic_config
    ):
        raise ArtifactIntegrityError("EvidenceSet semantic_config is malformed")
    if "ppo_seed" in semantic_config:
        raise ArtifactIntegrityError(
            "EvidenceSet semantic config must not contain ppo_seed"
        )
    if not isinstance(seeds_raw, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds_raw
    ):
        raise ArtifactIntegrityError("EvidenceSet seed roster is malformed")
    if not isinstance(run_digests_raw, list):
        raise ArtifactIntegrityError("EvidenceSet run digests are malformed")

    run_digests: list[tuple[int, str]] = []
    for entry in run_digests_raw:
        if not isinstance(entry, dict):
            raise ArtifactIntegrityError("EvidenceSet run digest entry is malformed")
        seed = entry.get("ppo_seed")
        digest = entry.get("artifact_digest")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(digest, str)
        ):
            raise ArtifactIntegrityError("EvidenceSet run digest entry is malformed")
        run_digests.append((seed, digest))

    fingerprint = manifest.get("fingerprint")
    semantic_digest = manifest.get("semantic_config_digest")
    context = manifest.get("research_context_digest")
    schema = manifest.get("schema_version")
    if not all(
        isinstance(value, str)
        for value in (fingerprint, semantic_digest, context, schema)
    ):
        raise ArtifactIntegrityError("EvidenceSet identity fields are malformed")
    if content_digest(semantic_config) != semantic_digest:
        raise ArtifactIntegrityError("EvidenceSet semantic config digest mismatch")

    evidence = EvidenceSet(
        fingerprint=cast(str, fingerprint),
        semantic_config_digest=cast(str, semantic_digest),
        ppo_seeds=tuple(cast(list[int], seeds_raw)),
        run_digests=tuple(run_digests),
        research_context_digest=cast(str, context),
        schema_version=cast(str, schema),
    )
    return evidence, cast(dict[str, object], semantic_config)


def load_evidence_set(root: str | Path) -> LoadedEvidenceSet:
    """Load and re-verify one complete Study-owned EvidenceSet."""

    evidence_root = Path(root)
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        raise ArtifactIntegrityError("EvidenceSet root must be a regular directory")
    names = {entry.name for entry in evidence_root.iterdir()}
    if names != {"manifest.json", "runs"}:
        raise ArtifactIntegrityError(
            "EvidenceSet root must contain exactly manifest.json and runs"
        )

    manifest = _read_manifest(evidence_root)
    evidence, semantic_config = _parse_evidence(manifest)
    runs_root = evidence_root / "runs"
    if runs_root.is_symlink() or not runs_root.is_dir():
        raise ArtifactIntegrityError("EvidenceSet runs must be a regular directory")
    expected_dirs = {f"seed-{seed}" for seed in evidence.ppo_seeds}
    actual_dirs = {entry.name for entry in runs_root.iterdir()}
    if actual_dirs != expected_dirs:
        raise ArtifactIntegrityError("EvidenceSet seed Run roster mismatch")

    expected_digests = dict(evidence.run_digests)
    runs: dict[int, LoadedCandidateRun] = {}
    for seed in evidence.ppo_seeds:
        run_root = runs_root / f"seed-{seed}"
        try:
            loaded = load_candidate_run_artifact(run_root)
            identity = inspect_candidate_run_artifact(run_root)
        except ValueError as error:
            raise ArtifactIntegrityError(str(error)) from error
        if identity.artifact_digest != expected_digests[seed]:
            raise ArtifactIntegrityError("EvidenceSet Run artifact digest mismatch")
        if (
            loaded.provenance.get("research_context_digest")
            != evidence.research_context_digest
        ):
            raise ArtifactIntegrityError("EvidenceSet Run research context mismatch")
        if _run_summary_seed(loaded) != seed:
            raise ArtifactIntegrityError("EvidenceSet Run seed mismatch")
        runs[seed] = loaded

    _verify_seed_invariance(
        runs,
        seeds=evidence.ppo_seeds,
        symbols=tuple(
            cast(
                Sequence[str],
                runs[evidence.ppo_seeds[0]].summary.get("symbols"),
            )
        ),
        research_context_digest=evidence.research_context_digest,
    )
    return LoadedEvidenceSet(
        evidence=evidence,
        semantic_config=semantic_config,
        runs=runs,
    )


__all__ = [
    "EvidenceSet",
    "LoadedEvidenceSet",
    "execute_evidence_set",
    "load_evidence_set",
]
