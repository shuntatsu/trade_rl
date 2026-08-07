"""Load source-bound adverse evidence from a published walk-forward run."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.evaluation.causal_scenario_c3_adverse import (
    C3AdverseFoldEvidence,
    C3AdverseThresholds,
    build_c3_adverse_thresholds,
    evaluate_c3_adverse_fold,
    selection_days_from_source_fold,
)

C3_SOURCE_ADVERSE_EVIDENCE_SCHEMA: Final = (
    "causal_scenario_c3_source_adverse_evidence_v1"
)


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
    if isinstance(value, bool) or not isinstance(value, (int, float)):
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


def _range_pair(value: object, *, field: str) -> tuple[int, int]:
    values = _sequence(value, field=field)
    if len(values) != 2:
        raise ValueError(f"{field} must contain start and stop")
    start = _integer(values[0], field=f"{field}.start")
    stop = _integer(values[1], field=f"{field}.stop", minimum=1)
    if stop <= start:
        raise ValueError(f"{field} stop must be greater than start")
    return start, stop


@dataclass(frozen=True, slots=True)
class C3SourceAdverseEvidence:
    source_artifact_digest: str
    thresholds: C3AdverseThresholds
    folds: tuple[C3AdverseFoldEvidence, ...]
    selection_days: tuple[tuple[int, int], ...]
    schema_version: str = C3_SOURCE_ADVERSE_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_artifact_digest",
            require_sha256(self.source_artifact_digest, field="source_artifact_digest"),
        )
        if not isinstance(self.thresholds, C3AdverseThresholds):
            raise ValueError("thresholds must be C3AdverseThresholds")
        folds = tuple(self.folds)
        if not folds or any(
            not isinstance(item, C3AdverseFoldEvidence) for item in folds
        ):
            raise ValueError("folds must contain C3 adverse fold evidence")
        indices = tuple(item.fold_index for item in folds)
        if tuple(sorted(indices)) != indices or len(set(indices)) != len(indices):
            raise ValueError("adverse fold indices must be sorted and unique")
        if any(
            item.source_artifact_digest != self.source_artifact_digest
            or item.thresholds_digest != self.thresholds.config_digest
            for item in folds
        ):
            raise ValueError("adverse fold evidence identity mismatch")
        object.__setattr__(self, "folds", folds)
        days = tuple(
            (
                _integer(index, field="selection_days.fold_index"),
                _integer(value, field="selection_days.days", minimum=1),
            )
            for index, value in self.selection_days
        )
        if tuple(sorted(days)) != days or tuple(index for index, _ in days) != indices:
            raise ValueError("selection days must match adverse fold indices")
        object.__setattr__(self, "selection_days", days)
        if self.schema_version != C3_SOURCE_ADVERSE_EVIDENCE_SCHEMA:
            raise ValueError("unsupported C3 source adverse evidence schema")

    @property
    def required_scenario(self) -> str:
        return self.thresholds.required_scenario

    @property
    def by_fold_index(self) -> dict[int, C3AdverseFoldEvidence]:
        return {item.fold_index: item for item in self.folds}

    @property
    def selection_days_by_fold(self) -> dict[int, int]:
        return dict(self.selection_days)

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "fold_evidence_digests": tuple(item.digest for item in self.folds),
                "schema_version": self.schema_version,
                "selection_days": self.selection_days,
                "source_artifact_digest": self.source_artifact_digest,
                "thresholds_digest": self.thresholds.config_digest,
            }
        )


def _validated_access(
    value: object,
    *,
    fold_index: int,
    source_fold: Mapping[str, object],
    dataset_id: str,
    experiment_plan_digest: str,
    scenario_pack_digest: str,
) -> None:
    access = dict(_mapping(value, field="execution_sensitivity.fold.access"))
    digest = require_sha256(
        _string(access.pop("access_digest", None), field="access_digest"),
        field="access_digest",
    )
    if content_digest(access) != digest:
        raise ValueError("execution sensitivity access digest mismatch")
    if access.get("dataset_id") != dataset_id:
        raise ValueError("execution sensitivity access dataset mismatch")
    if access.get("experiment_plan_digest") != experiment_plan_digest:
        raise ValueError("execution sensitivity experiment plan mismatch")
    if access.get("scenario_pack_digest") != scenario_pack_digest:
        raise ValueError("execution sensitivity scenario pack mismatch")
    if access.get("fold_index") != fold_index:
        raise ValueError("execution sensitivity access fold mismatch")
    if access.get("purpose") != "post_selection_execution_sensitivity":
        raise ValueError("execution sensitivity access purpose mismatch")
    expected_range = _range_pair(
        source_fold.get("test_range"), field="source test range"
    )
    actual_range = _range_pair(access.get("test_range"), field="access test range")
    if actual_range != expected_range:
        raise ValueError("execution sensitivity access test range mismatch")


def _validated_top_level_gate(
    value: object,
    *,
    sensitivity_config: Mapping[str, object],
    required_scenario: str,
) -> None:
    gate = _mapping(value, field="execution_sensitivity.gate")
    if gate.get("required_scenario") != required_scenario:
        raise ValueError("execution sensitivity gate required scenario mismatch")
    expected_drawdown = _number(
        sensitivity_config.get("maximum_drawdown"),
        field="execution_sensitivity.maximum_drawdown",
        minimum=0.0,
    )
    expected_uplift = _number(
        sensitivity_config.get("minimum_baseline_uplift"),
        field="execution_sensitivity.minimum_baseline_uplift",
    )
    expected_return = _number(
        sensitivity_config.get("minimum_selected_return"),
        field="execution_sensitivity.minimum_selected_return",
    )
    observed_drawdown = _number(
        gate.get("maximum_drawdown_threshold"),
        field="gate.maximum_drawdown_threshold",
    )
    observed_uplift = _number(
        gate.get("minimum_baseline_uplift"), field="gate.minimum_baseline_uplift"
    )
    observed_return = _number(
        gate.get("minimum_selected_return"), field="gate.minimum_selected_return"
    )
    if not math.isclose(
        observed_drawdown, expected_drawdown, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("execution sensitivity drawdown threshold mismatch")
    if not math.isclose(observed_uplift, expected_uplift, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("execution sensitivity uplift threshold mismatch")
    if not math.isclose(observed_return, expected_return, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("execution sensitivity return threshold mismatch")
    recomputed = (
        _number(gate.get("selected_total_return"), field="gate.selected_total_return")
        > expected_return
        and _number(gate.get("baseline_uplift"), field="gate.baseline_uplift")
        >= expected_uplift
        and _number(
            gate.get("maximum_fold_drawdown"),
            field="gate.maximum_fold_drawdown",
            minimum=0.0,
        )
        <= expected_drawdown
    )
    passed = gate.get("passed")
    if not isinstance(passed, bool) or passed is not recomputed:
        raise ValueError("execution sensitivity gate pass state mismatch")


def load_c3_source_adverse_evidence(
    source_root: str | Path,
    *,
    walk_forward_config: Mapping[str, object],
    source_folds: Mapping[int, Mapping[str, object]],
    dataset_id: str,
) -> C3SourceAdverseEvidence:
    """Load source sensitivity evidence and recompute every required fold gate."""

    root = Path(source_root)
    path = root / "execution-sensitivity.json"
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"execution sensitivity artifact is missing: {path}")
    try:
        manifest = dict(
            _mapping(
                json.loads(path.read_text(encoding="utf-8")),
                field="execution_sensitivity",
            )
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("execution sensitivity artifact is invalid") from error
    if canonical_json_bytes(manifest) != path.read_bytes():
        raise ValueError("execution sensitivity artifact is not canonical JSON")
    expected_fields = {
        "artifact_digest",
        "dataset_id",
        "experiment_plan_digest",
        "folds",
        "gate",
        "production_status",
        "scenario_pack_digest",
        "schema_version",
    }
    if set(manifest) != expected_fields:
        raise ValueError("execution sensitivity artifact field closure mismatch")
    artifact_digest = require_sha256(
        _string(manifest.pop("artifact_digest"), field="artifact_digest"),
        field="artifact_digest",
    )
    if content_digest(manifest) != artifact_digest:
        raise ValueError("execution sensitivity artifact digest mismatch")
    resolved_dataset_id = require_sha256(dataset_id, field="dataset_id")
    if manifest.get("dataset_id") != resolved_dataset_id:
        raise ValueError("execution sensitivity artifact dataset mismatch")
    if manifest.get("schema_version") != "execution_sensitivity_v1":
        raise ValueError("unsupported execution sensitivity artifact schema")
    if manifest.get("production_status") != "NO-GO":
        raise ValueError("execution sensitivity artifact must remain NO-GO")

    config = _mapping(walk_forward_config, field="walk_forward_config")
    sensitivity = _mapping(
        config.get("execution_sensitivity"), field="execution_sensitivity"
    )
    thresholds = build_c3_adverse_thresholds(config)
    scenario_pack_digest = require_sha256(
        _string(manifest.get("scenario_pack_digest"), field="scenario_pack_digest"),
        field="scenario_pack_digest",
    )
    if scenario_pack_digest != content_digest(sensitivity):
        raise ValueError("execution sensitivity scenario pack digest mismatch")
    experiment_plan_digest = require_sha256(
        _string(manifest.get("experiment_plan_digest"), field="experiment_plan_digest"),
        field="experiment_plan_digest",
    )
    _validated_top_level_gate(
        manifest.get("gate"),
        sensitivity_config=sensitivity,
        required_scenario=thresholds.required_scenario,
    )

    normalized_source_folds = {
        _integer(index, field="source_fold_index"): _mapping(
            value, field=f"source_folds[{index}]"
        )
        for index, value in source_folds.items()
    }
    fold_evidence: list[C3AdverseFoldEvidence] = []
    selection_days: list[tuple[int, int]] = []
    seen: set[int] = set()
    raw_folds = _sequence(manifest.get("folds"), field="execution_sensitivity.folds")
    for raw_fold in raw_folds:
        fold = _mapping(raw_fold, field="execution_sensitivity.fold")
        if set(fold) != {"access", "fold_index", "scenarios"}:
            raise ValueError("execution sensitivity fold field closure mismatch")
        fold_index = _integer(fold.get("fold_index"), field="fold_index")
        if fold_index in seen:
            raise ValueError("execution sensitivity fold indices must be unique")
        source_fold = normalized_source_folds.get(fold_index)
        if source_fold is None:
            raise ValueError("execution sensitivity fold is absent from source run")
        _validated_access(
            fold.get("access"),
            fold_index=fold_index,
            source_fold=source_fold,
            dataset_id=resolved_dataset_id,
            experiment_plan_digest=experiment_plan_digest,
            scenario_pack_digest=scenario_pack_digest,
        )
        required_results: list[Mapping[str, object]] = []
        for raw_scenario in _sequence(fold.get("scenarios"), field="fold.scenarios"):
            scenario_result = _mapping(raw_scenario, field="fold.scenario")
            scenario = _mapping(
                scenario_result.get("scenario"), field="fold.scenario.scenario"
            )
            if scenario.get("name") == thresholds.required_scenario:
                required_results.append(scenario_result)
        if len(required_results) != 1:
            raise ValueError(
                "required scenario is missing or duplicated in source fold"
            )
        fold_evidence.append(
            evaluate_c3_adverse_fold(
                fold_index=fold_index,
                scenario_result=required_results[0],
                thresholds=thresholds,
                source_artifact_digest=artifact_digest,
            )
        )
        selection_days.append(
            (
                fold_index,
                selection_days_from_source_fold(source_fold, thresholds=thresholds),
            )
        )
        seen.add(fold_index)
    if seen != set(normalized_source_folds):
        raise ValueError("execution sensitivity folds do not match source run folds")
    return C3SourceAdverseEvidence(
        source_artifact_digest=artifact_digest,
        thresholds=thresholds,
        folds=tuple(sorted(fold_evidence, key=lambda item: item.fold_index)),
        selection_days=tuple(sorted(selection_days)),
    )


__all__ = [
    "C3_SOURCE_ADVERSE_EVIDENCE_SCHEMA",
    "C3SourceAdverseEvidence",
    "load_c3_source_adverse_evidence",
]
