"""Evaluation-only C3 batch workflow above frozen C1/C2 evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from trade_rl.evaluation.causal_scenario_c3_adverse import C3AdverseFoldEvidence
from trade_rl.evaluation.causal_scenario_c3_artifact import (
    write_c3_aggregate_report_artifact,
    write_phase_a_gate_artifact,
)
from trade_rl.evaluation.causal_scenario_c3_contracts import (
    C3ReplayIdentity,
    CausalScenarioC3Config,
    CausalScenarioQueryComparison,
    PerfectInformationComparison,
)
from trade_rl.evaluation.causal_scenario_c3_decision_artifact import (
    load_c3_decision_artifact,
)
from trade_rl.evaluation.causal_scenario_c3_gate import (
    PhaseAEntryGateEvidence,
    evaluate_phase_a_entry_gate,
)
from trade_rl.evaluation.causal_scenario_c3_prediction import C3PredictionEvidence
from trade_rl.evaluation.causal_scenario_c3_report import (
    CausalScenarioAggregateReport,
    build_c3_aggregate_report,
    build_c3_fold_report,
)
from trade_rl.evaluation.causal_scenario_c3_runner import (
    C3RealizedReplay,
    run_c3_query_comparison,
)

PRODUCTION_STATUS: Final = "NO-GO"


def _readonly_action(value: object) -> np.ndarray:
    try:
        action = np.asarray(value, dtype=np.float64).copy(order="C")
    except (TypeError, ValueError) as error:
        raise ValueError("ppo_mean_action must be numeric") from error
    if action.ndim != 1 or action.size == 0:
        raise ValueError("ppo_mean_action must be a non-empty vector")
    if not np.isfinite(action).all() or np.any(np.abs(action) > 1.0):
        raise ValueError("ppo_mean_action must be finite and within [-1, 1]")
    action[action == 0.0] = 0.0
    action.setflags(write=False)
    return action


def _label(value: str, *, field: str) -> str:
    result = value.strip()
    if not result:
        raise ValueError(f"{field} must be non-empty")
    return result


@dataclass(frozen=True, slots=True)
class C3BatchQuery:
    fold_id: str
    decision_root: Path
    replay: C3RealizedReplay
    ppo_mean_action: np.ndarray
    prediction_evidence: C3PredictionEvidence
    execution_scenario: str
    perfect_information: PerfectInformationComparison

    def __post_init__(self) -> None:
        object.__setattr__(self, "fold_id", _label(self.fold_id, field="fold_id"))
        object.__setattr__(
            self,
            "execution_scenario",
            _label(self.execution_scenario, field="execution_scenario"),
        )
        object.__setattr__(self, "decision_root", Path(self.decision_root))
        object.__setattr__(
            self, "ppo_mean_action", _readonly_action(self.ppo_mean_action)
        )
        if not isinstance(self.prediction_evidence, C3PredictionEvidence):
            raise ValueError("prediction_evidence must be C3PredictionEvidence")
        if not isinstance(self.perfect_information, PerfectInformationComparison):
            raise ValueError("perfect_information must be PerfectInformationComparison")
        identity = getattr(self.replay, "identity", None)
        if not isinstance(identity, C3ReplayIdentity):
            raise ValueError("replay must expose a C3ReplayIdentity")
        clone = getattr(self.replay, "clone_for_replay", None)
        if clone is None or not callable(clone):
            raise ValueError("replay must expose a callable clone_for_replay method")
        run = getattr(self.replay, "run", None)
        if run is None or not callable(run):
            raise ValueError("replay must expose a callable run method")


@dataclass(frozen=True, slots=True)
class C3BatchResult:
    report: CausalScenarioAggregateReport
    gate: PhaseAEntryGateEvidence
    report_artifact_root: Path
    gate_artifact_root: Path
    report_artifact_digest: str
    gate_artifact_digest: str
    comparison_count: int
    production_status: str = PRODUCTION_STATUS

    def __post_init__(self) -> None:
        if not isinstance(self.report, CausalScenarioAggregateReport):
            raise ValueError("report must be CausalScenarioAggregateReport")
        if not isinstance(self.gate, PhaseAEntryGateEvidence):
            raise ValueError("gate must be PhaseAEntryGateEvidence")
        if self.gate.report_digest != self.report.digest:
            raise ValueError("gate does not bind the aggregate report")
        object.__setattr__(
            self, "report_artifact_root", Path(self.report_artifact_root)
        )
        object.__setattr__(self, "gate_artifact_root", Path(self.gate_artifact_root))
        if self.comparison_count <= 0:
            raise ValueError("comparison_count must be positive")
        if self.production_status != PRODUCTION_STATUS:
            raise ValueError("C3 batch production status must remain NO-GO")


def _mapping_keys(mapping: Mapping[str, object], *, field: str) -> set[str]:
    keys = set(mapping)
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError(f"{field} keys must be non-empty strings")
    return keys


def execute_c3_batch(
    queries: tuple[C3BatchQuery, ...],
    *,
    output_root: str | Path,
    fold_selection_days: Mapping[str, int],
    required_adverse_evidence: Mapping[str, C3AdverseFoldEvidence],
    config: CausalScenarioC3Config,
) -> C3BatchResult:
    """Evaluate verified decisions, aggregate by reset fold, and publish gate evidence."""

    items = tuple(queries)
    if not items:
        raise ValueError("queries must not be empty")
    if any(not isinstance(item, C3BatchQuery) for item in items):
        raise ValueError("queries must contain C3BatchQuery values")
    if not isinstance(config, CausalScenarioC3Config):
        raise TypeError("config must be CausalScenarioC3Config")

    fold_ids = {item.fold_id for item in items}
    if _mapping_keys(fold_selection_days, field="fold_selection_days") != fold_ids:
        raise ValueError("fold_selection_days must exactly match query folds")
    if (
        _mapping_keys(required_adverse_evidence, field="required_adverse_evidence")
        != fold_ids
    ):
        raise ValueError("required_adverse_evidence must exactly match query folds")
    if any(
        not isinstance(item, C3AdverseFoldEvidence)
        for item in required_adverse_evidence.values()
    ):
        raise ValueError(
            "required_adverse_evidence must contain C3 adverse fold evidence"
        )

    by_fold: dict[str, list[CausalScenarioQueryComparison]] = defaultdict(list)
    seen_comparison_keys: set[tuple[str, str]] = set()
    for item in items:
        loaded = load_c3_decision_artifact(item.decision_root)
        comparison_key = (
            loaded.decision.decision_digest,
            item.execution_scenario,
        )
        if comparison_key in seen_comparison_keys:
            raise ValueError("duplicate C3 decision and execution scenario in batch")
        if loaded.decision.fold_digest != item.replay.identity.fold_digest:
            raise ValueError("C3 batch fold identity does not match replay")
        item.prediction_evidence.validate_for_decision(loaded.decision)
        seen_comparison_keys.add(comparison_key)
        by_fold[item.fold_id].append(
            run_c3_query_comparison(
                loaded,
                replay=item.replay,
                ppo_mean_action=item.ppo_mean_action,
                config=config,
                prediction_evidence=item.prediction_evidence,
                execution_scenario=item.execution_scenario,
                perfect_information=item.perfect_information,
            )
        )

    folds = tuple(
        build_c3_fold_report(
            fold_id=fold_id,
            selection_days=fold_selection_days[fold_id],
            comparisons=tuple(by_fold[fold_id]),
            required_adverse_passed=required_adverse_evidence[fold_id].passed,
            required_adverse_evidence_digest=required_adverse_evidence[fold_id].digest,
        )
        for fold_id in sorted(by_fold)
    )
    report = build_c3_aggregate_report(
        folds,
        bootstrap_resamples=config.bootstrap_resamples,
        bootstrap_block_days=config.bootstrap_block_days,
    )
    gate = evaluate_phase_a_entry_gate(report, config=config)
    destination = Path(output_root)
    report_root = destination / "report"
    gate_root = destination / "gate"
    report_artifact_digest = write_c3_aggregate_report_artifact(report_root, report)
    gate_artifact_digest = write_phase_a_gate_artifact(gate_root, gate)
    return C3BatchResult(
        report=report,
        gate=gate,
        report_artifact_root=report_root,
        gate_artifact_root=gate_root,
        report_artifact_digest=report_artifact_digest,
        gate_artifact_digest=gate_artifact_digest,
        comparison_count=len(items),
    )


__all__ = ["C3BatchQuery", "C3BatchResult", "execute_c3_batch"]
