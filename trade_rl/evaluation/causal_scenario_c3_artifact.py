"""Deterministic artifacts for C3 aggregate reports and Phase A gates."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.evaluation.causal_scenario_c3_contracts import (
    CausalScenarioQueryComparison,
    PerfectInformationComparison,
    PerfectInformationComparisonStatus,
    RealizedPolicyOutcome,
)
from trade_rl.evaluation.causal_scenario_c3_gate import (
    GateConditionResult,
    PhaseAEntryGateEvidence,
)
from trade_rl.evaluation.causal_scenario_c3_report import (
    CausalScenarioAggregateReport,
    build_c3_aggregate_report,
    build_c3_fold_report,
)

C3_AGGREGATE_REPORT_ARTIFACT_SCHEMA: Final = (
    "causal_scenario_c3_aggregate_report_artifact_v1"
)
PHASE_A_GATE_ARTIFACT_SCHEMA: Final = "phase_a_entry_gate_artifact_v1"
_REPORT_FILE: Final = "report.json"
_GATE_FILE: Final = "gate.json"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _verify_exact_file(root: Path, filename: str, *, label: str) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"{label} artifact directory is missing: {root}")
    entries = tuple(root.iterdir())
    if len(entries) != 1:
        raise ValueError(f"{label} artifact file closure mismatch")
    entry = entries[0]
    if entry.name != filename or entry.is_symlink() or not entry.is_file():
        raise ValueError(f"{label} artifact file closure mismatch")
    return entry


def _strict_object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _strict_list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _strict_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _strict_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _strict_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _strict_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _outcome_payload(outcome: RealizedPolicyOutcome) -> dict[str, object]:
    return {
        "borrow_paid": outcome.borrow_paid,
        "fees": outcome.fees,
        "fill_count": outcome.fill_count,
        "filled_turnover": outcome.filled_turnover,
        "funding_paid": outcome.funding_paid,
        "gross_log_return": outcome.gross_log_return,
        "impact_cost": outcome.impact_cost,
        "max_drawdown": outcome.max_drawdown,
        "outcome_digest": outcome.outcome_digest,
        "pending_order_events": outcome.pending_order_events,
        "policy_kind": outcome.policy_kind,
        "schema_version": outcome.schema_version,
        "spread_cost": outcome.spread_cost,
        "terminal_equity": outcome.terminal_equity,
        "termination_reason": outcome.termination_reason,
    }


def _load_outcome(payload: object, *, field: str) -> RealizedPolicyOutcome:
    item = _strict_object(payload, field=field)
    expected = {
        "borrow_paid",
        "fees",
        "fill_count",
        "filled_turnover",
        "funding_paid",
        "gross_log_return",
        "impact_cost",
        "max_drawdown",
        "outcome_digest",
        "pending_order_events",
        "policy_kind",
        "schema_version",
        "spread_cost",
        "terminal_equity",
        "termination_reason",
    }
    if set(item) != expected:
        raise ValueError(f"{field} field closure mismatch")
    return RealizedPolicyOutcome(
        policy_kind=_strict_string(item["policy_kind"], field=f"{field}.policy_kind"),
        gross_log_return=_strict_number(
            item["gross_log_return"], field=f"{field}.gross_log_return"
        ),
        filled_turnover=_strict_number(
            item["filled_turnover"], field=f"{field}.filled_turnover"
        ),
        fees=_strict_number(item["fees"], field=f"{field}.fees"),
        spread_cost=_strict_number(item["spread_cost"], field=f"{field}.spread_cost"),
        impact_cost=_strict_number(item["impact_cost"], field=f"{field}.impact_cost"),
        funding_paid=_strict_number(
            item["funding_paid"], field=f"{field}.funding_paid"
        ),
        borrow_paid=_strict_number(item["borrow_paid"], field=f"{field}.borrow_paid"),
        fill_count=_strict_int(item["fill_count"], field=f"{field}.fill_count"),
        pending_order_events=_strict_int(
            item["pending_order_events"], field=f"{field}.pending_order_events"
        ),
        max_drawdown=_strict_number(
            item["max_drawdown"], field=f"{field}.max_drawdown"
        ),
        terminal_equity=_strict_number(
            item["terminal_equity"], field=f"{field}.terminal_equity"
        ),
        termination_reason=_strict_string(
            item["termination_reason"], field=f"{field}.termination_reason"
        ),
        outcome_digest=_strict_string(
            item["outcome_digest"], field=f"{field}.outcome_digest"
        ),
        schema_version=_strict_string(
            item["schema_version"], field=f"{field}.schema_version"
        ),
    )


def _perfect_information_payload(
    comparison: PerfectInformationComparison,
) -> dict[str, object]:
    return {
        "bound_log_return": comparison.bound_log_return,
        "causal_log_return": comparison.causal_log_return,
        "gap": comparison.gap,
        "reason": comparison.reason,
        "status": comparison.status.value,
    }


def _load_perfect_information(
    payload: object, *, field: str
) -> PerfectInformationComparison:
    item = _strict_object(payload, field=field)
    if set(item) != {
        "bound_log_return",
        "causal_log_return",
        "gap",
        "reason",
        "status",
    }:
        raise ValueError(f"{field} field closure mismatch")
    status = PerfectInformationComparisonStatus(
        _strict_string(item["status"], field=f"{field}.status")
    )
    if status is PerfectInformationComparisonStatus.COMPARABLE:
        return PerfectInformationComparison(
            status=status,
            reason=_strict_string(item["reason"], field=f"{field}.reason"),
            bound_log_return=_strict_number(
                item["bound_log_return"], field=f"{field}.bound_log_return"
            ),
            causal_log_return=_strict_number(
                item["causal_log_return"], field=f"{field}.causal_log_return"
            ),
            gap=_strict_number(item["gap"], field=f"{field}.gap"),
        )
    if any(
        item[name] is not None
        for name in ("bound_log_return", "causal_log_return", "gap")
    ):
        raise ValueError(f"{field} non-comparable values must be null")
    return PerfectInformationComparison(
        status=status,
        reason=_strict_string(item["reason"], field=f"{field}.reason"),
        bound_log_return=None,
        causal_log_return=None,
        gap=None,
    )


def _comparison_payload(
    comparison: CausalScenarioQueryComparison,
) -> dict[str, object]:
    return {
        "candidate_outcomes": [
            _outcome_payload(outcome) for outcome in comparison.candidate_outcomes
        ],
        "comparison_digest": comparison.digest,
        "decision_digest": comparison.decision_digest,
        "perfect_information": _perfect_information_payload(
            comparison.perfect_information
        ),
        "ppo_mean": _outcome_payload(comparison.ppo_mean),
        "predicted_realized_spearman": comparison.predicted_realized_spearman,
        "random_candidate": _outcome_payload(comparison.random_candidate),
        "random_candidate_indices": comparison.random_candidate_indices,
        "random_candidate_outcomes": [
            _outcome_payload(outcome)
            for outcome in comparison.random_candidate_outcomes
        ],
        "random_realized_regret": comparison.random_realized_regret,
        "random_realized_regrets": comparison.random_realized_regrets.tolist(),
        "realized_candidate_advantages": comparison.realized_candidate_advantages.tolist(),
        "scenario_oracle": _outcome_payload(comparison.scenario_oracle),
        "schema_version": comparison.schema_version,
        "selected_realized_regret": comparison.selected_realized_regret,
        "trend": _outcome_payload(comparison.trend),
    }


def _load_comparison(payload: object, *, field: str) -> CausalScenarioQueryComparison:
    item = _strict_object(payload, field=field)
    expected = {
        "candidate_outcomes",
        "comparison_digest",
        "decision_digest",
        "perfect_information",
        "ppo_mean",
        "predicted_realized_spearman",
        "random_candidate",
        "random_candidate_indices",
        "random_candidate_outcomes",
        "random_realized_regret",
        "random_realized_regrets",
        "realized_candidate_advantages",
        "scenario_oracle",
        "schema_version",
        "selected_realized_regret",
        "trend",
    }
    if set(item) != expected:
        raise ValueError(f"{field} field closure mismatch")
    candidates = tuple(
        _load_outcome(value, field=f"{field}.candidate_outcomes[{index}]")
        for index, value in enumerate(
            _strict_list(
                item["candidate_outcomes"], field=f"{field}.candidate_outcomes"
            )
        )
    )
    random_indices = tuple(
        _strict_int(value, field=f"{field}.random_candidate_indices[{index}]")
        for index, value in enumerate(
            _strict_list(
                item["random_candidate_indices"],
                field=f"{field}.random_candidate_indices",
            )
        )
    )
    random_outcomes = tuple(
        _load_outcome(value, field=f"{field}.random_candidate_outcomes[{index}]")
        for index, value in enumerate(
            _strict_list(
                item["random_candidate_outcomes"],
                field=f"{field}.random_candidate_outcomes",
            )
        )
    )
    advantages = np.asarray(
        [
            _strict_number(
                value, field=f"{field}.realized_candidate_advantages[{index}]"
            )
            for index, value in enumerate(
                _strict_list(
                    item["realized_candidate_advantages"],
                    field=f"{field}.realized_candidate_advantages",
                )
            )
        ],
        dtype=np.float64,
    )
    random_regrets = np.asarray(
        [
            _strict_number(value, field=f"{field}.random_realized_regrets[{index}]")
            for index, value in enumerate(
                _strict_list(
                    item["random_realized_regrets"],
                    field=f"{field}.random_realized_regrets",
                )
            )
        ],
        dtype=np.float64,
    )
    comparison = CausalScenarioQueryComparison(
        decision_digest=_strict_string(
            item["decision_digest"], field=f"{field}.decision_digest"
        ),
        trend=_load_outcome(item["trend"], field=f"{field}.trend"),
        scenario_oracle=_load_outcome(
            item["scenario_oracle"], field=f"{field}.scenario_oracle"
        ),
        ppo_mean=_load_outcome(item["ppo_mean"], field=f"{field}.ppo_mean"),
        random_candidate=_load_outcome(
            item["random_candidate"], field=f"{field}.random_candidate"
        ),
        random_candidate_indices=random_indices,
        random_candidate_outcomes=random_outcomes,
        random_realized_regrets=random_regrets,
        candidate_outcomes=candidates,
        realized_candidate_advantages=advantages,
        predicted_realized_spearman=_strict_number(
            item["predicted_realized_spearman"],
            field=f"{field}.predicted_realized_spearman",
        ),
        selected_realized_regret=_strict_number(
            item["selected_realized_regret"],
            field=f"{field}.selected_realized_regret",
        ),
        random_realized_regret=_strict_number(
            item["random_realized_regret"],
            field=f"{field}.random_realized_regret",
        ),
        perfect_information=_load_perfect_information(
            item["perfect_information"], field=f"{field}.perfect_information"
        ),
        schema_version=_strict_string(
            item["schema_version"], field=f"{field}.schema_version"
        ),
    )
    if comparison.digest != _strict_string(
        item["comparison_digest"], field=f"{field}.comparison_digest"
    ):
        raise ValueError(f"{field} digest mismatch")
    return comparison


def _report_base_payload(report: CausalScenarioAggregateReport) -> dict[str, object]:
    return {
        "bootstrap_block_days": report.bootstrap_block_days,
        "bootstrap_resamples": report.bootstrap_resamples,
        "folds": [
            {
                "comparisons": [
                    _comparison_payload(comparison) for comparison in fold.comparisons
                ],
                "failure_reasons": fold.failure_reasons,
                "fold_digest": fold.digest,
                "fold_id": fold.fold_id,
                "required_adverse_passed": fold.required_adverse_passed,
                "selection_days": fold.selection_days,
            }
            for fold in report.folds
        ],
        "report_digest": report.digest,
        "schema_version": C3_AGGREGATE_REPORT_ARTIFACT_SCHEMA,
    }


@dataclass(frozen=True, slots=True)
class LoadedC3AggregateReport:
    report: CausalScenarioAggregateReport
    artifact_digest: str
    root: Path

    def __post_init__(self) -> None:
        if not isinstance(self.report, CausalScenarioAggregateReport):
            raise ValueError("report must be CausalScenarioAggregateReport")
        object.__setattr__(
            self,
            "artifact_digest",
            require_sha256(self.artifact_digest, field="artifact_digest"),
        )
        object.__setattr__(self, "root", Path(self.root))


@dataclass(frozen=True, slots=True)
class LoadedPhaseAGate:
    gate: PhaseAEntryGateEvidence
    artifact_digest: str
    root: Path

    def __post_init__(self) -> None:
        if not isinstance(self.gate, PhaseAEntryGateEvidence):
            raise ValueError("gate must be PhaseAEntryGateEvidence")
        object.__setattr__(
            self,
            "artifact_digest",
            require_sha256(self.artifact_digest, field="artifact_digest"),
        )
        object.__setattr__(self, "root", Path(self.root))


def _write_single_json_artifact(
    root: Path,
    *,
    filename: str,
    base_payload: dict[str, object],
    label: str,
) -> str:
    artifact_digest = content_digest(base_payload)
    payload = dict(base_payload)
    payload["artifact_digest"] = artifact_digest
    encoded = canonical_json_bytes(payload)
    if root.exists() and any(root.iterdir()):
        existing = _verify_exact_file(root, filename, label=label).read_bytes()
        if existing != encoded:
            raise FileExistsError(
                f"conflicting {label} artifact already exists: {root}"
            )
        return artifact_digest
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write(root / filename, encoded)
    return artifact_digest


def write_c3_aggregate_report_artifact(
    root: str | Path, report: CausalScenarioAggregateReport
) -> str:
    if not isinstance(report, CausalScenarioAggregateReport):
        raise TypeError("report must be CausalScenarioAggregateReport")
    destination = Path(root)
    digest = _write_single_json_artifact(
        destination,
        filename=_REPORT_FILE,
        base_payload=_report_base_payload(report),
        label="C3 aggregate report",
    )
    loaded = load_c3_aggregate_report_artifact(destination)
    if loaded.artifact_digest != digest or loaded.report.digest != report.digest:
        raise ValueError("published C3 aggregate report artifact failed verification")
    return digest


def load_c3_aggregate_report_artifact(
    root: str | Path,
) -> LoadedC3AggregateReport:
    source = Path(root)
    path = _verify_exact_file(source, _REPORT_FILE, label="C3 aggregate report")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("C3 aggregate report manifest is invalid") from error
    manifest = _strict_object(raw, field="manifest")
    if set(manifest) != {
        "artifact_digest",
        "bootstrap_block_days",
        "bootstrap_resamples",
        "folds",
        "report_digest",
        "schema_version",
    }:
        raise ValueError("C3 aggregate report manifest field closure mismatch")
    if manifest["schema_version"] != C3_AGGREGATE_REPORT_ARTIFACT_SCHEMA:
        raise ValueError("unsupported C3 aggregate report artifact schema")
    artifact_digest = require_sha256(
        _strict_string(manifest["artifact_digest"], field="artifact_digest"),
        field="artifact_digest",
    )
    base = dict(manifest)
    del base["artifact_digest"]
    if content_digest(base) != artifact_digest:
        raise ValueError("C3 aggregate report artifact digest mismatch")
    if canonical_json_bytes(manifest) != path.read_bytes():
        raise ValueError("C3 aggregate report manifest is not canonical JSON")

    folds = []
    for fold_index, fold_value in enumerate(
        _strict_list(manifest["folds"], field="folds")
    ):
        fold = _strict_object(fold_value, field=f"folds[{fold_index}]")
        if set(fold) != {
            "comparisons",
            "failure_reasons",
            "fold_digest",
            "fold_id",
            "required_adverse_passed",
            "selection_days",
        }:
            raise ValueError("C3 fold artifact field closure mismatch")
        comparisons = tuple(
            _load_comparison(
                value,
                field=f"folds[{fold_index}].comparisons[{comparison_index}]",
            )
            for comparison_index, value in enumerate(
                _strict_list(
                    fold["comparisons"],
                    field=f"folds[{fold_index}].comparisons",
                )
            )
        )
        reasons = tuple(
            _strict_string(value, field=f"folds[{fold_index}].failure_reasons")
            for value in _strict_list(
                fold["failure_reasons"],
                field=f"folds[{fold_index}].failure_reasons",
            )
        )
        rebuilt = build_c3_fold_report(
            fold_id=_strict_string(
                fold["fold_id"], field=f"folds[{fold_index}].fold_id"
            ),
            selection_days=_strict_int(
                fold["selection_days"],
                field=f"folds[{fold_index}].selection_days",
            ),
            comparisons=comparisons,
            required_adverse_passed=_strict_bool(
                fold["required_adverse_passed"],
                field=f"folds[{fold_index}].required_adverse_passed",
            ),
            failure_reasons=reasons,
        )
        if rebuilt.digest != _strict_string(
            fold["fold_digest"], field=f"folds[{fold_index}].fold_digest"
        ):
            raise ValueError("C3 fold report digest mismatch")
        folds.append(rebuilt)
    report = build_c3_aggregate_report(
        tuple(folds),
        bootstrap_resamples=_strict_int(
            manifest["bootstrap_resamples"], field="bootstrap_resamples"
        ),
        bootstrap_block_days=_strict_int(
            manifest["bootstrap_block_days"], field="bootstrap_block_days"
        ),
    )
    if report.digest != _strict_string(
        manifest["report_digest"], field="report_digest"
    ):
        raise ValueError("C3 aggregate report digest mismatch")
    return LoadedC3AggregateReport(
        report=report,
        artifact_digest=artifact_digest,
        root=source,
    )


def _gate_base_payload(gate: PhaseAEntryGateEvidence) -> dict[str, object]:
    return {
        "conditions": [
            {
                "detail": condition.detail,
                "name": condition.name,
                "passed": condition.passed,
            }
            for condition in gate.conditions
        ],
        "config_digest": gate.config_digest,
        "gate_digest": gate.digest,
        "gate_schema_version": gate.schema_version,
        "passed": gate.passed,
        "report_digest": gate.report_digest,
        "schema_version": PHASE_A_GATE_ARTIFACT_SCHEMA,
    }


def write_phase_a_gate_artifact(root: str | Path, gate: PhaseAEntryGateEvidence) -> str:
    if not isinstance(gate, PhaseAEntryGateEvidence):
        raise TypeError("gate must be PhaseAEntryGateEvidence")
    destination = Path(root)
    digest = _write_single_json_artifact(
        destination,
        filename=_GATE_FILE,
        base_payload=_gate_base_payload(gate),
        label="Phase A gate",
    )
    loaded = load_phase_a_gate_artifact(destination)
    if loaded.artifact_digest != digest or loaded.gate.digest != gate.digest:
        raise ValueError("published Phase A gate artifact failed verification")
    return digest


def load_phase_a_gate_artifact(root: str | Path) -> LoadedPhaseAGate:
    source = Path(root)
    path = _verify_exact_file(source, _GATE_FILE, label="Phase A gate")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Phase A gate manifest is invalid") from error
    manifest = _strict_object(raw, field="manifest")
    if set(manifest) != {
        "artifact_digest",
        "conditions",
        "config_digest",
        "gate_digest",
        "gate_schema_version",
        "passed",
        "report_digest",
        "schema_version",
    }:
        raise ValueError("Phase A gate manifest field closure mismatch")
    if manifest["schema_version"] != PHASE_A_GATE_ARTIFACT_SCHEMA:
        raise ValueError("unsupported Phase A gate artifact schema")
    artifact_digest = require_sha256(
        _strict_string(manifest["artifact_digest"], field="artifact_digest"),
        field="artifact_digest",
    )
    base = dict(manifest)
    del base["artifact_digest"]
    if content_digest(base) != artifact_digest:
        raise ValueError("Phase A gate artifact digest mismatch")
    if canonical_json_bytes(manifest) != path.read_bytes():
        raise ValueError("Phase A gate manifest is not canonical JSON")
    condition_values = _strict_list(manifest["conditions"], field="conditions")
    condition_objects = tuple(
        _strict_object(value, field=f"conditions[{index}]")
        for index, value in enumerate(condition_values)
    )
    if any(set(value) != {"detail", "name", "passed"} for value in condition_objects):
        raise ValueError("Phase A gate condition field closure mismatch")
    conditions = tuple(
        GateConditionResult(
            name=_strict_string(value["name"], field=f"conditions[{index}].name"),
            passed=_strict_bool(
                value["passed"], field=f"conditions[{index}].passed"
            ),
            detail=_strict_string(
                value["detail"], field=f"conditions[{index}].detail"
            ),
        )
        for index, value in enumerate(condition_objects)
    )
    gate = PhaseAEntryGateEvidence(
        report_digest=_strict_string(manifest["report_digest"], field="report_digest"),
        config_digest=_strict_string(manifest["config_digest"], field="config_digest"),
        conditions=conditions,
        passed=_strict_bool(manifest["passed"], field="passed"),
        schema_version=_strict_string(
            manifest["gate_schema_version"], field="gate_schema_version"
        ),
    )
    if gate.digest != _strict_string(manifest["gate_digest"], field="gate_digest"):
        raise ValueError("Phase A gate digest mismatch")
    return LoadedPhaseAGate(
        gate=gate,
        artifact_digest=artifact_digest,
        root=source,
    )
