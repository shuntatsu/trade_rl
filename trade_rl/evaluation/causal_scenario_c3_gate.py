"""Pure Phase A entry gate for frozen C3 aggregate evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.causal_scenario_c3_contracts import CausalScenarioC3Config
from trade_rl.evaluation.causal_scenario_c3_report import (
    CausalScenarioAggregateReport,
)

PHASE_A_ENTRY_GATE_SCHEMA: Final = "phase_a_entry_gate_evidence_v1"


@dataclass(frozen=True, slots=True)
class GateConditionResult:
    name: str
    passed: bool
    detail: str

    def __post_init__(self) -> None:
        name = self.name.strip()
        detail = self.detail.strip()
        if not name or not detail:
            raise ValueError("gate condition name and detail must be non-empty")
        if not isinstance(self.passed, bool):
            raise ValueError("gate condition passed must be boolean")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "detail", detail)


@dataclass(frozen=True, slots=True)
class PhaseAEntryGateEvidence:
    report_digest: str
    config_digest: str
    conditions: tuple[GateConditionResult, ...]
    passed: bool
    schema_version: str = PHASE_A_ENTRY_GATE_SCHEMA

    def __post_init__(self) -> None:
        conditions = tuple(self.conditions)
        if len(conditions) != 9:
            raise ValueError("Phase A gate must contain exactly nine conditions")
        if len({condition.name for condition in conditions}) != len(conditions):
            raise ValueError("Phase A gate condition names must be unique")
        if not isinstance(self.passed, bool):
            raise ValueError("Phase A gate passed must be boolean")
        if self.passed != all(condition.passed for condition in conditions):
            raise ValueError("Phase A gate pass state does not match conditions")
        if self.schema_version != PHASE_A_ENTRY_GATE_SCHEMA:
            raise ValueError("unsupported Phase A entry gate schema")
        object.__setattr__(self, "conditions", conditions)

    @property
    def failed_condition_names(self) -> tuple[str, ...]:
        return tuple(
            condition.name for condition in self.conditions if not condition.passed
        )

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "conditions": tuple(
                    {
                        "detail": condition.detail,
                        "name": condition.name,
                        "passed": condition.passed,
                    }
                    for condition in self.conditions
                ),
                "config_digest": self.config_digest,
                "passed": self.passed,
                "report_digest": self.report_digest,
                "schema_version": self.schema_version,
            }
        )


def _condition(name: str, passed: bool, detail: str) -> GateConditionResult:
    return GateConditionResult(name=name, passed=passed, detail=detail)


def evaluate_phase_a_entry_gate(
    report: CausalScenarioAggregateReport,
    *,
    config: CausalScenarioC3Config | None = None,
) -> PhaseAEntryGateEvidence:
    """Evaluate all approved C3 evidence requirements without external access."""

    if not isinstance(report, CausalScenarioAggregateReport):
        raise TypeError("report must be CausalScenarioAggregateReport")
    resolved = CausalScenarioC3Config() if config is None else config
    if not isinstance(resolved, CausalScenarioC3Config):
        raise TypeError("config must be CausalScenarioC3Config")

    diagnostics_complete = (
        bool(report.calibration_buckets)
        and bool(report.execution_summaries)
        and report.unique_anchor_count > 0
        and report.effective_anchor_count > 0.0
        and report.historical_coverage_fraction > 0.0
    )
    no_failures = not report.failure_reasons and diagnostics_complete
    support = (
        report.fold_count >= resolved.required_folds
        and report.total_selection_days >= resolved.required_selection_days
        and report.total_effective_days >= resolved.required_selection_days
    )
    required_positive_folds = min(4, resolved.required_folds)
    positive_folds = report.positive_uplift_folds >= required_positive_folds
    positive_uplift_ci = report.uplift_lower_ci > 0.0
    drawdown = (
        report.worst_scenario_oracle_drawdown <= 0.20
        and report.worst_scenario_oracle_drawdown <= report.worst_trend_drawdown + 0.02
    )
    regret = report.regret_margin_lower_ci > 0.0
    ranking = report.mean_spearman > 0.0 and report.spearman_lower_ci > 0.0
    perfect_information = report.all_perfect_information_valid
    scenarios = report.execution_scenario_names
    adverse_names = tuple(name for name in scenarios if name != "nominal")
    adverse = (
        report.all_required_adverse_passed
        and "nominal" in scenarios
        and bool(adverse_names)
    )

    conditions = (
        _condition(
            "integrity_and_determinism",
            no_failures,
            (
                "no leakage, identity, replay, artifact, determinism, or diagnostic failures"
                if no_failures
                else (
                    f"failures={report.failure_reasons}; "
                    f"diagnostics_complete={diagnostics_complete}"
                )
            ),
        ),
        _condition(
            "fold_and_day_support",
            support,
            (
                f"folds={report.fold_count}/{resolved.required_folds}; "
                f"selection_days={report.total_selection_days}/"
                f"{resolved.required_selection_days}; "
                f"effective_days={report.total_effective_days}/"
                f"{resolved.required_selection_days}"
            ),
        ),
        _condition(
            "positive_uplift_folds",
            positive_folds,
            (
                f"positive={report.positive_uplift_folds}; "
                f"required={required_positive_folds}"
            ),
        ),
        _condition(
            "aggregate_uplift_confidence",
            positive_uplift_ci,
            f"paired_95pct_lower={report.uplift_lower_ci:.12g}",
        ),
        _condition(
            "worst_fold_drawdown",
            drawdown,
            (
                f"oracle={report.worst_scenario_oracle_drawdown:.12g}; "
                f"trend={report.worst_trend_drawdown:.12g}"
            ),
        ),
        _condition(
            "realized_regret_vs_random",
            regret,
            f"paired_95pct_lower={report.regret_margin_lower_ci:.12g}",
        ),
        _condition(
            "predicted_realized_ranking",
            ranking,
            (
                f"mean={report.mean_spearman:.12g}; "
                f"lower={report.spearman_lower_ci:.12g}"
            ),
        ),
        _condition(
            "perfect_information_compatibility",
            perfect_information,
            "all asserted bounds are compatible and ordered"
            if perfect_information
            else "one or more bounds are not compatible or ordered",
        ),
        _condition(
            "required_adverse_execution",
            adverse,
            (
                f"nominal=true; adverse={adverse_names}; all_passed=true"
                if adverse
                else (
                    f"scenarios={scenarios}; adverse={adverse_names}; "
                    f"all_passed={report.all_required_adverse_passed}"
                )
            ),
        ),
    )
    return PhaseAEntryGateEvidence(
        report_digest=report.digest,
        config_digest=resolved.digest,
        conditions=conditions,
        passed=all(condition.passed for condition in conditions),
    )
