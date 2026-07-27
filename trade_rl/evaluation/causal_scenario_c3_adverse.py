"""Source-bound adverse execution evidence for the C3 Phase A gate."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

C3_ADVERSE_THRESHOLDS_SCHEMA: Final = "causal_scenario_c3_adverse_thresholds_v1"
C3_ADVERSE_FOLD_EVIDENCE_SCHEMA: Final = "causal_scenario_c3_adverse_fold_evidence_v1"


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise ValueError(f"{field} must be a sequence")


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _number(value: object, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class C3AdverseThresholds:
    required_scenario: str
    decision_hours: float
    minimum_selected_return: float
    minimum_baseline_uplift: float
    maximum_cost_fraction: float
    maximum_turnover_per_day: float
    maximum_drawdown: float
    config_digest: str
    schema_version: str = C3_ADVERSE_THRESHOLDS_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_scenario",
            _string(self.required_scenario, field="required_scenario"),
        )
        for field in (
            "decision_hours",
            "maximum_cost_fraction",
            "maximum_turnover_per_day",
            "maximum_drawdown",
        ):
            value = _number(getattr(self, field), field=field, minimum=0.0)
            object.__setattr__(self, field, value)
        if self.decision_hours <= 0.0:
            raise ValueError("decision_hours must be positive")
        if self.maximum_drawdown > 1.0:
            raise ValueError("maximum_drawdown must not exceed one")
        for field in ("minimum_selected_return", "minimum_baseline_uplift"):
            object.__setattr__(
                self,
                field,
                _number(getattr(self, field), field=field),
            )
        object.__setattr__(
            self,
            "config_digest",
            require_sha256(self.config_digest, field="config_digest"),
        )
        if self.schema_version != C3_ADVERSE_THRESHOLDS_SCHEMA:
            raise ValueError("unsupported C3 adverse threshold schema")
        if self.config_digest != content_digest(self.digest_payload()):
            raise ValueError("C3 adverse threshold digest mismatch")

    def digest_payload(self) -> dict[str, object]:
        return {
            "decision_hours": self.decision_hours,
            "maximum_cost_fraction": self.maximum_cost_fraction,
            "maximum_drawdown": self.maximum_drawdown,
            "maximum_turnover_per_day": self.maximum_turnover_per_day,
            "minimum_baseline_uplift": self.minimum_baseline_uplift,
            "minimum_selected_return": self.minimum_selected_return,
            "required_scenario": self.required_scenario,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class C3AdverseFoldEvidence:
    fold_index: int
    source_artifact_digest: str
    thresholds_digest: str
    required_scenario: str
    selected_return: float
    baseline_uplift: float
    cost_fraction: float
    turnover_per_day: float
    maximum_drawdown: float
    failed_conditions: tuple[str, ...]
    schema_version: str = C3_ADVERSE_FOLD_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fold_index",
            _integer(self.fold_index, field="fold_index"),
        )
        for field in ("source_artifact_digest", "thresholds_digest"):
            object.__setattr__(
                self,
                field,
                require_sha256(str(getattr(self, field)), field=field),
            )
        object.__setattr__(
            self,
            "required_scenario",
            _string(self.required_scenario, field="required_scenario"),
        )
        for field in (
            "selected_return",
            "baseline_uplift",
            "cost_fraction",
            "turnover_per_day",
            "maximum_drawdown",
        ):
            object.__setattr__(self, field, _number(getattr(self, field), field=field))
        if self.cost_fraction < 0.0 or self.turnover_per_day < 0.0:
            raise ValueError("adverse cost and turnover must be non-negative")
        if not 0.0 <= self.maximum_drawdown <= 1.0:
            raise ValueError("adverse maximum_drawdown must be in [0, 1]")
        failures = tuple(
            _string(item, field="failed_conditions") for item in self.failed_conditions
        )
        if len(set(failures)) != len(failures):
            raise ValueError("failed_conditions must be unique")
        object.__setattr__(self, "failed_conditions", failures)
        if self.schema_version != C3_ADVERSE_FOLD_EVIDENCE_SCHEMA:
            raise ValueError("unsupported C3 adverse fold evidence schema")

    @property
    def passed(self) -> bool:
        return not self.failed_conditions

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "baseline_uplift": self.baseline_uplift,
                "cost_fraction": self.cost_fraction,
                "failed_conditions": self.failed_conditions,
                "fold_index": self.fold_index,
                "maximum_drawdown": self.maximum_drawdown,
                "required_scenario": self.required_scenario,
                "schema_version": self.schema_version,
                "selected_return": self.selected_return,
                "source_artifact_digest": self.source_artifact_digest,
                "thresholds_digest": self.thresholds_digest,
                "turnover_per_day": self.turnover_per_day,
            }
        )


def build_c3_adverse_thresholds(
    walk_forward_config: Mapping[str, object],
) -> C3AdverseThresholds:
    """Resolve predeclared adverse limits from a validated walk-forward config."""

    config = _mapping(walk_forward_config, field="walk_forward_config")
    if config.get("schema_version") != "market_walk_forward_config_v1":
        raise ValueError("unsupported market walk-forward configuration schema")
    candidates = _sequence(config.get("candidates"), field="candidates")
    if not candidates:
        raise ValueError("candidates must not be empty")
    first = _mapping(candidates[0], field="candidates[0]")
    run = _mapping(first.get("run"), field="candidates[0].run")
    environment = _mapping(
        run.get("environment"), field="candidates[0].run.environment"
    )
    decision_hours = _number(
        environment.get("decision_hours"),
        field="candidates[0].run.environment.decision_hours",
        minimum=0.0,
    )
    if decision_hours <= 0.0:
        raise ValueError("decision_hours must be positive")

    sensitivity = _mapping(
        config.get("execution_sensitivity"), field="execution_sensitivity"
    )
    if sensitivity.get("schema_version") != "execution_sensitivity_config_v1":
        raise ValueError("unsupported execution sensitivity configuration schema")
    required_scenario = _string(
        sensitivity.get("required_scenario"),
        field="execution_sensitivity.required_scenario",
    )
    scenarios = _sequence(
        sensitivity.get("scenarios"), field="execution_sensitivity.scenarios"
    )
    matching = [
        _mapping(item, field="execution_sensitivity.scenarios[]")
        for item in scenarios
        if _mapping(item, field="execution_sensitivity.scenarios[]").get("name")
        == required_scenario
    ]
    if len(matching) != 1:
        raise ValueError(
            "required execution sensitivity scenario is missing or duplicated"
        )
    if matching[0].get("report_only") is True:
        raise ValueError(
            "required execution sensitivity scenario cannot be report-only"
        )

    selection_cost = _number(
        config.get("maximum_selection_cost_fraction"),
        field="maximum_selection_cost_fraction",
        minimum=0.0,
    )
    selection_turnover = _number(
        config.get("maximum_selection_turnover_per_day"),
        field="maximum_selection_turnover_per_day",
        minimum=0.0,
    )
    selection_drawdown = _number(
        config.get("maximum_selection_drawdown"),
        field="maximum_selection_drawdown",
        minimum=0.0,
    )
    sensitivity_drawdown = _number(
        sensitivity.get("maximum_drawdown"),
        field="execution_sensitivity.maximum_drawdown",
        minimum=0.0,
    )
    minimum_uplift = max(
        _number(
            config.get("minimum_selection_uplift"),
            field="minimum_selection_uplift",
        ),
        _number(
            sensitivity.get("minimum_baseline_uplift"),
            field="execution_sensitivity.minimum_baseline_uplift",
        ),
    )
    minimum_return = _number(
        sensitivity.get("minimum_selected_return"),
        field="execution_sensitivity.minimum_selected_return",
    )
    payload = {
        "decision_hours": decision_hours,
        "maximum_cost_fraction": selection_cost,
        "maximum_drawdown": min(selection_drawdown, sensitivity_drawdown),
        "maximum_turnover_per_day": selection_turnover,
        "minimum_baseline_uplift": minimum_uplift,
        "minimum_selected_return": minimum_return,
        "required_scenario": required_scenario,
        "schema_version": C3_ADVERSE_THRESHOLDS_SCHEMA,
    }
    return C3AdverseThresholds(
        required_scenario=required_scenario,
        decision_hours=decision_hours,
        minimum_selected_return=minimum_return,
        minimum_baseline_uplift=minimum_uplift,
        maximum_cost_fraction=selection_cost,
        maximum_turnover_per_day=selection_turnover,
        maximum_drawdown=min(selection_drawdown, sensitivity_drawdown),
        config_digest=content_digest(payload),
    )


def selection_days_from_source_fold(
    source_fold: Mapping[str, object],
    *,
    thresholds: C3AdverseThresholds,
) -> int:
    """Derive declared selection support from the source fold and decision clock."""

    fold = _mapping(source_fold, field="source_fold")
    selection_range = _sequence(fold.get("selection_range"), field="selection_range")
    if len(selection_range) != 2:
        raise ValueError("selection_range must contain start and stop")
    start = _integer(selection_range[0], field="selection_range.start")
    stop = _integer(selection_range[1], field="selection_range.stop", minimum=1)
    if stop <= start:
        raise ValueError("selection_range stop must be greater than start")
    days = (stop - start) * thresholds.decision_hours / 24.0
    rounded = round(days)
    if rounded <= 0 or not math.isclose(
        days, float(rounded), rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            "selection range does not resolve to a positive whole day count"
        )
    return int(rounded)


def evaluate_c3_adverse_fold(
    *,
    fold_index: int,
    scenario_result: Mapping[str, object],
    thresholds: C3AdverseThresholds,
    source_artifact_digest: str,
) -> C3AdverseFoldEvidence:
    """Recompute one required adverse fold gate from manifest-bound source metrics."""

    result = dict(_mapping(scenario_result, field="scenario_result"))
    stored_digest = require_sha256(
        _string(
            result.pop("scenario_result_digest", None), field="scenario_result_digest"
        ),
        field="scenario_result_digest",
    )
    if content_digest(result) != stored_digest:
        raise ValueError("adverse scenario result digest mismatch")
    scenario = _mapping(result.get("scenario"), field="scenario_result.scenario")
    name = _string(scenario.get("name"), field="scenario_result.scenario.name")
    if name != thresholds.required_scenario:
        raise ValueError(
            "adverse scenario does not match the predeclared required scenario"
        )
    if result.get("report_only") is True or scenario.get("report_only") is True:
        raise ValueError("required adverse scenario cannot be report-only")
    selected = _mapping(result.get("selected"), field="scenario_result.selected")
    selected_return = _number(
        selected.get("total_return"), field="scenario_result.selected.total_return"
    )
    baseline_uplift = _number(
        result.get("baseline_uplift"), field="scenario_result.baseline_uplift"
    )
    cost_fraction = _number(
        selected.get("cost_fraction"),
        field="scenario_result.selected.cost_fraction",
        minimum=0.0,
    )
    turnover = _number(
        selected.get("turnover_per_day"),
        field="scenario_result.selected.turnover_per_day",
        minimum=0.0,
    )
    drawdown = _number(
        selected.get("maximum_drawdown"),
        field="scenario_result.selected.maximum_drawdown",
        minimum=0.0,
    )
    failures: list[str] = []
    if not selected_return > thresholds.minimum_selected_return:
        failures.append("selected_return")
    if baseline_uplift < thresholds.minimum_baseline_uplift:
        failures.append("baseline_uplift")
    if cost_fraction > thresholds.maximum_cost_fraction:
        failures.append("cost_fraction")
    if turnover > thresholds.maximum_turnover_per_day:
        failures.append("turnover_per_day")
    if drawdown > thresholds.maximum_drawdown:
        failures.append("maximum_drawdown")
    return C3AdverseFoldEvidence(
        fold_index=fold_index,
        source_artifact_digest=source_artifact_digest,
        thresholds_digest=thresholds.config_digest,
        required_scenario=name,
        selected_return=selected_return,
        baseline_uplift=baseline_uplift,
        cost_fraction=cost_fraction,
        turnover_per_day=turnover,
        maximum_drawdown=drawdown,
        failed_conditions=tuple(failures),
    )


__all__ = [
    "C3_ADVERSE_FOLD_EVIDENCE_SCHEMA",
    "C3_ADVERSE_THRESHOLDS_SCHEMA",
    "C3AdverseFoldEvidence",
    "C3AdverseThresholds",
    "build_c3_adverse_thresholds",
    "evaluate_c3_adverse_fold",
    "selection_days_from_source_fold",
]
