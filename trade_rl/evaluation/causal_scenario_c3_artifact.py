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


def _object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_digest(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return require_sha256(_string(value, field=field), field=field)


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _require_fields(
    payload: dict[str, object], expected: set[str], *, label: str
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{label} field closure mismatch")


def _outcome_payload(outcome: RealizedPolicyOutcome) -> dict[str, object]:
    return {
        "borrow_paid": outcome.borrow_paid,
        "cancel_replace_events": outcome.cancel_replace_events,
        "fees": outcome.fees,
        "fill_count": outcome.fill_count,
        "fill_ratio": outcome.fill_ratio,
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


def _load_outcome(value: object, *, field: str) -> RealizedPolicyOutcome:
    payload = _object(value, field=field)
    _require_fields(
        payload,
        {
            "borrow_paid",
            "cancel_replace_events",
            "fees",
            "fill_count",
            "fill_ratio",
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
        },
        label=field,
    )
    return RealizedPolicyOutcome(
        policy_kind=_string(payload["policy_kind"], field=f"{field}.policy_kind"),
        gross_log_return=_number(
            payload["gross_log_return"], field=f"{field}.gross_log_return"
        ),
        filled_turnover=_number(
            payload["filled_turnover"], field=f"{field}.filled_turnover"
        ),
        fees=_number(payload["fees"], field=f"{field}.fees"),
        spread_cost=_number(payload["spread_cost"], field=f"{field}.spread_cost"),
        impact_cost=_number(payload["impact_cost"], field=f"{field}.impact_cost"),
        funding_paid=_number(payload["funding_paid"], field=f"{field}.funding_paid"),
        borrow_paid=_number(payload["borrow_paid"], field=f"{field}.borrow_paid"),
        fill_ratio=_number(payload["fill_ratio"], field=f"{field}.fill_ratio"),
        fill_count=_integer(payload["fill_count"], field=f"{field}.fill_count"),
        pending_order_events=_integer(
            payload["pending_order_events"], field=f"{field}.pending_order_events"
        ),
        cancel_replace_events=_integer(
            payload["cancel_replace_events"], field=f"{field}.cancel_replace_events"
        ),
        max_drawdown=_number(payload["max_drawdown"], field=f"{field}.max_drawdown"),
        terminal_equity=_number(
            payload["terminal_equity"], field=f"{field}.terminal_equity"
        ),
        termination_reason=_string(
            payload["termination_reason"], field=f"{field}.termination_reason"
        ),
        outcome_digest=_string(
            payload["outcome_digest"], field=f"{field}.outcome_digest"
        ),
        schema_version=_string(
            payload["schema_version"], field=f"{field}.schema_version"
        ),
    )


def _perfect_payload(value: PerfectInformationComparison) -> dict[str, object]:
    return {
        "bound_log_return": value.bound_log_return,
        "causal_log_return": value.causal_log_return,
        "compatibility_evidence_digest": value.compatibility_evidence_digest,
        "gap": value.gap,
        "reason": value.reason,
        "status": value.status.value,
    }


def _load_perfect(value: object, *, field: str) -> PerfectInformationComparison:
    payload = _object(value, field=field)
    _require_fields(
        payload,
        {
            "bound_log_return",
            "causal_log_return",
            "compatibility_evidence_digest",
            "gap",
            "reason",
            "status",
        },
        label=field,
    )
    status = PerfectInformationComparisonStatus(
        _string(payload["status"], field=f"{field}.status")
    )
    reason = _string(payload["reason"], field=f"{field}.reason")
    evidence_digest = _optional_digest(
        payload["compatibility_evidence_digest"],
        field=f"{field}.compatibility_evidence_digest",
    )
    if status is PerfectInformationComparisonStatus.COMPARABLE:
        return PerfectInformationComparison(
            status=status,
            reason=reason,
            bound_log_return=_number(
                payload["bound_log_return"], field=f"{field}.bound_log_return"
            ),
            causal_log_return=_number(
                payload["causal_log_return"], field=f"{field}.causal_log_return"
            ),
            gap=_number(payload["gap"], field=f"{field}.gap"),
            compatibility_evidence_digest=evidence_digest,
        )
    if any(
        payload[name] is not None
        for name in ("bound_log_return", "causal_log_return", "gap")
    ):
        raise ValueError(f"{field} non-comparable values must be null")
    return PerfectInformationComparison(
        status=status,
        reason=reason,
        bound_log_return=None,
        causal_log_return=None,
        gap=None,
        compatibility_evidence_digest=evidence_digest,
    )


def _comparison_payload(value: CausalScenarioQueryComparison) -> dict[str, object]:
    return {
        "candidate_outcomes": [
            _outcome_payload(outcome) for outcome in value.candidate_outcomes
        ],
        "comparison_digest": value.digest,
        "decision_digest": value.decision_digest,
        "execution_scenario": value.execution_scenario,
        "perfect_information": _perfect_payload(value.perfect_information),
        "ppo_mean": _outcome_payload(value.ppo_mean),
        "predicted_expected_turnover": value.predicted_expected_turnover.tolist(),
        "predicted_loss_cvar": value.predicted_loss_cvar.tolist(),
        "predicted_mean_advantage": value.predicted_mean_advantage.tolist(),
        "predicted_realized_spearman": value.predicted_realized_spearman,
        "predicted_score": value.predicted_score.tolist(),
        "prediction_result_digest": value.prediction_result_digest,
        "query_timestamp_ns": value.query_timestamp_ns,
        "random_candidate": _outcome_payload(value.random_candidate),
        "random_candidate_indices": value.random_candidate_indices,
        "random_candidate_outcomes": [
            _outcome_payload(outcome) for outcome in value.random_candidate_outcomes
        ],
        "random_realized_regret": value.random_realized_regret,
        "random_realized_regrets": value.random_realized_regrets.tolist(),
        "realized_candidate_advantages": value.realized_candidate_advantages.tolist(),
        "replay_identity_digest": value.replay_identity_digest,
        "scenario_anchor_indices": value.scenario_anchor_indices.tolist(),
        "scenario_distances": value.scenario_distances.tolist(),
        "scenario_oracle": _outcome_payload(value.scenario_oracle),
        "schema_version": value.schema_version,
        "selected_realized_regret": value.selected_realized_regret,
        "trend": _outcome_payload(value.trend),
    }


def _float_vector(value: object, *, field: str) -> np.ndarray:
    return np.asarray(
        [
            _number(item, field=f"{field}[{index}]")
            for index, item in enumerate(_list(value, field=field))
        ],
        dtype=np.float64,
    )


def _int_vector(value: object, *, field: str) -> np.ndarray:
    return np.asarray(
        [
            _integer(item, field=f"{field}[{index}]")
            for index, item in enumerate(_list(value, field=field))
        ],
        dtype=np.int64,
    )


def _load_comparison(value: object, *, field: str) -> CausalScenarioQueryComparison:
    payload = _object(value, field=field)
    _require_fields(
        payload,
        {
            "candidate_outcomes",
            "comparison_digest",
            "decision_digest",
            "execution_scenario",
            "perfect_information",
            "ppo_mean",
            "predicted_expected_turnover",
            "predicted_loss_cvar",
            "predicted_mean_advantage",
            "predicted_realized_spearman",
            "predicted_score",
            "prediction_result_digest",
            "query_timestamp_ns",
            "random_candidate",
            "random_candidate_indices",
            "random_candidate_outcomes",
            "random_realized_regret",
            "random_realized_regrets",
            "realized_candidate_advantages",
            "replay_identity_digest",
            "scenario_anchor_indices",
            "scenario_distances",
            "scenario_oracle",
            "schema_version",
            "selected_realized_regret",
            "trend",
        },
        label=field,
    )
    candidates = tuple(
        _load_outcome(item, field=f"{field}.candidate_outcomes[{index}]")
        for index, item in enumerate(
            _list(payload["candidate_outcomes"], field=f"{field}.candidate_outcomes")
        )
    )
    random_indices = tuple(
        _integer(item, field=f"{field}.random_candidate_indices[{index}]")
        for index, item in enumerate(
            _list(
                payload["random_candidate_indices"],
                field=f"{field}.random_candidate_indices",
            )
        )
    )
    random_outcomes = tuple(
        _load_outcome(item, field=f"{field}.random_candidate_outcomes[{index}]")
        for index, item in enumerate(
            _list(
                payload["random_candidate_outcomes"],
                field=f"{field}.random_candidate_outcomes",
            )
        )
    )
    comparison = CausalScenarioQueryComparison(
        decision_digest=_string(
            payload["decision_digest"], field=f"{field}.decision_digest"
        ),
        query_timestamp_ns=_integer(
            payload["query_timestamp_ns"], field=f"{field}.query_timestamp_ns"
        ),
        replay_identity_digest=_string(
            payload["replay_identity_digest"],
            field=f"{field}.replay_identity_digest",
        ),
        execution_scenario=_string(
            payload["execution_scenario"], field=f"{field}.execution_scenario"
        ),
        prediction_result_digest=_string(
            payload["prediction_result_digest"],
            field=f"{field}.prediction_result_digest",
        ),
        predicted_score=_float_vector(
            payload["predicted_score"], field=f"{field}.predicted_score"
        ),
        predicted_mean_advantage=_float_vector(
            payload["predicted_mean_advantage"],
            field=f"{field}.predicted_mean_advantage",
        ),
        predicted_loss_cvar=_float_vector(
            payload["predicted_loss_cvar"], field=f"{field}.predicted_loss_cvar"
        ),
        predicted_expected_turnover=_float_vector(
            payload["predicted_expected_turnover"],
            field=f"{field}.predicted_expected_turnover",
        ),
        scenario_anchor_indices=_int_vector(
            payload["scenario_anchor_indices"],
            field=f"{field}.scenario_anchor_indices",
        ),
        scenario_distances=_float_vector(
            payload["scenario_distances"], field=f"{field}.scenario_distances"
        ),
        trend=_load_outcome(payload["trend"], field=f"{field}.trend"),
        scenario_oracle=_load_outcome(
            payload["scenario_oracle"], field=f"{field}.scenario_oracle"
        ),
        ppo_mean=_load_outcome(payload["ppo_mean"], field=f"{field}.ppo_mean"),
        random_candidate=_load_outcome(
            payload["random_candidate"], field=f"{field}.random_candidate"
        ),
        random_candidate_indices=random_indices,
        random_candidate_outcomes=random_outcomes,
        random_realized_regrets=_float_vector(
            payload["random_realized_regrets"],
            field=f"{field}.random_realized_regrets",
        ),
        candidate_outcomes=candidates,
        realized_candidate_advantages=_float_vector(
            payload["realized_candidate_advantages"],
            field=f"{field}.realized_candidate_advantages",
        ),
        predicted_realized_spearman=_number(
            payload["predicted_realized_spearman"],
            field=f"{field}.predicted_realized_spearman",
        ),
        selected_realized_regret=_number(
            payload["selected_realized_regret"],
            field=f"{field}.selected_realized_regret",
        ),
        random_realized_regret=_number(
            payload["random_realized_regret"],
            field=f"{field}.random_realized_regret",
        ),
        perfect_information=_load_perfect(
            payload["perfect_information"], field=f"{field}.perfect_information"
        ),
        schema_version=_string(
            payload["schema_version"], field=f"{field}.schema_version"
        ),
    )
    if comparison.digest != _string(
        payload["comparison_digest"], field=f"{field}.comparison_digest"
    ):
        raise ValueError(f"{field} digest mismatch")
    return comparison


def _report_payload(report: CausalScenarioAggregateReport) -> dict[str, object]:
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
                "required_adverse_evidence_digest": (
                    fold.required_adverse_evidence_digest
                ),
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


def _write_artifact(
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
    digest = _write_artifact(
        destination,
        filename=_REPORT_FILE,
        base_payload=_report_payload(report),
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
    manifest = _object(raw, field="manifest")
    _require_fields(
        manifest,
        {
            "artifact_digest",
            "bootstrap_block_days",
            "bootstrap_resamples",
            "folds",
            "report_digest",
            "schema_version",
        },
        label="C3 aggregate report manifest",
    )
    if manifest["schema_version"] != C3_AGGREGATE_REPORT_ARTIFACT_SCHEMA:
        raise ValueError("unsupported C3 aggregate report artifact schema")
    artifact_digest = require_sha256(
        _string(manifest["artifact_digest"], field="artifact_digest"),
        field="artifact_digest",
    )
    base = dict(manifest)
    del base["artifact_digest"]
    if content_digest(base) != artifact_digest:
        raise ValueError("C3 aggregate report artifact digest mismatch")
    if canonical_json_bytes(manifest) != path.read_bytes():
        raise ValueError("C3 aggregate report manifest is not canonical JSON")
    folds = []
    for fold_index, raw_fold in enumerate(_list(manifest["folds"], field="folds")):
        field = f"folds[{fold_index}]"
        fold = _object(raw_fold, field=field)
        _require_fields(
            fold,
            {
                "comparisons",
                "failure_reasons",
                "fold_digest",
                "fold_id",
                "required_adverse_evidence_digest",
                "required_adverse_passed",
                "selection_days",
            },
            label=field,
        )
        comparisons = tuple(
            _load_comparison(item, field=f"{field}.comparisons[{index}]")
            for index, item in enumerate(
                _list(fold["comparisons"], field=f"{field}.comparisons")
            )
        )
        reasons = tuple(
            _string(item, field=f"{field}.failure_reasons[{index}]")
            for index, item in enumerate(
                _list(fold["failure_reasons"], field=f"{field}.failure_reasons")
            )
        )
        rebuilt = build_c3_fold_report(
            fold_id=_string(fold["fold_id"], field=f"{field}.fold_id"),
            selection_days=_integer(
                fold["selection_days"], field=f"{field}.selection_days"
            ),
            comparisons=comparisons,
            required_adverse_passed=_boolean(
                fold["required_adverse_passed"],
                field=f"{field}.required_adverse_passed",
            ),
            required_adverse_evidence_digest=_string(
                fold["required_adverse_evidence_digest"],
                field=f"{field}.required_adverse_evidence_digest",
            ),
            failure_reasons=reasons,
        )
        if rebuilt.digest != _string(fold["fold_digest"], field=f"{field}.fold_digest"):
            raise ValueError("C3 fold report digest mismatch")
        folds.append(rebuilt)
    report = build_c3_aggregate_report(
        tuple(folds),
        bootstrap_resamples=_integer(
            manifest["bootstrap_resamples"], field="bootstrap_resamples"
        ),
        bootstrap_block_days=_integer(
            manifest["bootstrap_block_days"], field="bootstrap_block_days"
        ),
    )
    if report.digest != _string(manifest["report_digest"], field="report_digest"):
        raise ValueError("C3 aggregate report digest mismatch")
    return LoadedC3AggregateReport(
        report=report,
        artifact_digest=artifact_digest,
        root=source,
    )


def _gate_payload(gate: PhaseAEntryGateEvidence) -> dict[str, object]:
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
    digest = _write_artifact(
        destination,
        filename=_GATE_FILE,
        base_payload=_gate_payload(gate),
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
    manifest = _object(raw, field="manifest")
    _require_fields(
        manifest,
        {
            "artifact_digest",
            "conditions",
            "config_digest",
            "gate_digest",
            "gate_schema_version",
            "passed",
            "report_digest",
            "schema_version",
        },
        label="Phase A gate manifest",
    )
    if manifest["schema_version"] != PHASE_A_GATE_ARTIFACT_SCHEMA:
        raise ValueError("unsupported Phase A gate artifact schema")
    artifact_digest = require_sha256(
        _string(manifest["artifact_digest"], field="artifact_digest"),
        field="artifact_digest",
    )
    base = dict(manifest)
    del base["artifact_digest"]
    if content_digest(base) != artifact_digest:
        raise ValueError("Phase A gate artifact digest mismatch")
    if canonical_json_bytes(manifest) != path.read_bytes():
        raise ValueError("Phase A gate manifest is not canonical JSON")
    raw_conditions = tuple(
        _object(item, field=f"conditions[{index}]")
        for index, item in enumerate(_list(manifest["conditions"], field="conditions"))
    )
    for index, condition in enumerate(raw_conditions):
        _require_fields(
            condition,
            {"detail", "name", "passed"},
            label=f"conditions[{index}]",
        )
    conditions = tuple(
        GateConditionResult(
            name=_string(condition["name"], field=f"conditions[{index}].name"),
            passed=_boolean(condition["passed"], field=f"conditions[{index}].passed"),
            detail=_string(condition["detail"], field=f"conditions[{index}].detail"),
        )
        for index, condition in enumerate(raw_conditions)
    )
    gate = PhaseAEntryGateEvidence(
        report_digest=_string(manifest["report_digest"], field="report_digest"),
        config_digest=_string(manifest["config_digest"], field="config_digest"),
        conditions=conditions,
        passed=_boolean(manifest["passed"], field="passed"),
        schema_version=_string(
            manifest["gate_schema_version"], field="gate_schema_version"
        ),
    )
    if gate.digest != _string(manifest["gate_digest"], field="gate_digest"):
        raise ValueError("Phase A gate digest mismatch")
    return LoadedPhaseAGate(
        gate=gate,
        artifact_digest=artifact_digest,
        root=source,
    )
