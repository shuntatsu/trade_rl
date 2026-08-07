"""Narrow migration of archived research evidence into typed classifications.

This module is intentionally not a runtime compatibility layer. It exists only to
read archived evidence, validate its internal identities, and classify what the
historical run actually proved.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from trade_rl.domain.common import require_non_empty, require_sha256
from trade_rl.domain.signals import SignalStatus


class ResearchRunStatus(StrEnum):
    """Whether the archived research workflow reached a reportable end state."""

    COMPLETED = "completed"


class PolicyCandidateStatus(StrEnum):
    """Whether a residual policy was actually selected from archived evidence."""

    NOT_SELECTED = "not_selected"
    SELECTED = "selected"


class BaselineFallbackStatus(StrEnum):
    """How an identity baseline was used after candidate selection."""

    NOT_SELECTED = "not_selected"
    SELECTED_FOR_ANALYSIS = "selected_for_analysis"


class ReleaseStatus(StrEnum):
    """Whether the evidence may form a production release."""

    BLOCKED = "blocked"
    ELIGIBLE = "eligible"


@dataclass(frozen=True, slots=True)
class MigratedResearchRun:
    """Typed classification extracted from one archived legacy run."""

    research_run_status: ResearchRunStatus
    signal_status: SignalStatus
    policy_candidate_status: PolicyCandidateStatus
    baseline_fallback_status: BaselineFallbackStatus
    release_status: ReleaseStatus
    selected_configuration: str
    selected_policy_digest: str | None
    policy_ensemble_members: tuple[str, ...]
    holdout_total_return: float
    cost2x_total_return: float
    positive_return_p_value: float
    signal_model_kind: str
    signal_dataset_id: str
    notes: tuple[str, ...]


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise ValueError(f"{field} must use string keys")
    return value


def _nested(root: Mapping[str, object], *path: str) -> Mapping[str, object]:
    current = root
    traversed: list[str] = []
    for key in path:
        traversed.append(key)
        if key not in current:
            joined = ".".join(traversed)
            raise ValueError(f"missing required mapping: {joined}")
        current = _mapping(current[key], field=".".join(traversed))
    return current


def _value(root: Mapping[str, object], key: str, *, field: str) -> object:
    if key not in root:
        raise ValueError(f"missing required field: {field}")
    return root[key]


def _string(root: Mapping[str, object], key: str, *, field: str) -> str:
    value = _value(root, key, field=field)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return require_non_empty(value, field=field)


def _optional_string(
    root: Mapping[str, object],
    key: str,
    *,
    field: str,
) -> str | None:
    value = _value(root, key, field=field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return require_non_empty(value, field=field)


def _boolean(root: Mapping[str, object], key: str, *, field: str) -> bool:
    value = _value(root, key, field=field)
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _number(root: Mapping[str, object], key: str, *, field: str) -> float:
    value = _value(root, key, field=field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    return resolved


def _require_equal(left: str, right: str, *, field: str) -> None:
    if left != right:
        raise ValueError(f"{field} mismatch: {left!r} != {right!r}")


def migrate_legacy_research_run(
    *,
    report: Mapping[str, object],
    model_manifest: Mapping[str, object],
    signal_metadata: Mapping[str, object],
) -> MigratedResearchRun:
    """Validate and classify the archived baseline-residual research evidence."""

    selection = _nested(report, "selection")
    hyperparameters = _nested(model_manifest, "hyperparameters")
    signal_gate = _nested(report, "signal_gate")
    final_gate = _nested(report, "gate")
    holdout_hybrid = _nested(report, "relative", "hybrid")
    cost2x_hybrid = _nested(report, "cost2x", "hybrid")

    report_mode = _string(report, "mode", field="report.mode")
    manifest_mode = _string(
        model_manifest,
        "policy_mode",
        field="model_manifest.policy_mode",
    )
    selection_mode = _string(
        selection,
        "policy_mode",
        field="report.selection.policy_mode",
    )
    _require_equal(report_mode, manifest_mode, field="policy mode")
    _require_equal(report_mode, selection_mode, field="policy mode")

    selected_configuration = _string(
        report,
        "selected_configuration",
        field="report.selected_configuration",
    )
    manifest_configuration = _string(
        hyperparameters,
        "selected_configuration",
        field="model_manifest.hyperparameters.selected_configuration",
    )
    selection_configuration = _string(
        selection,
        "selected",
        field="report.selection.selected",
    )
    _require_equal(
        selected_configuration,
        manifest_configuration,
        field="selected configuration",
    )
    _require_equal(
        selected_configuration,
        selection_configuration,
        field="selected configuration",
    )

    manifest_dataset_id = require_sha256(
        _string(
            model_manifest,
            "alpha_dataset_identity",
            field="model_manifest.alpha_dataset_identity",
        ),
        field="model_manifest.alpha_dataset_identity",
    )
    signal_dataset_id = require_sha256(
        _string(
            signal_metadata,
            "dataset_identity",
            field="signal_metadata.dataset_identity",
        ),
        field="signal_metadata.dataset_identity",
    )
    if manifest_dataset_id != signal_dataset_id:
        raise ValueError("dataset identity mismatch between manifest and signal")

    selected_model_path = _optional_string(
        report,
        "selected_model_path",
        field="report.selected_model_path",
    )
    signal_passed = _boolean(
        signal_gate,
        "passed",
        field="report.signal_gate.passed",
    )
    alpha_enabled = _boolean(
        report,
        "alpha_enabled",
        field="report.alpha_enabled",
    )
    if not signal_passed and alpha_enabled:
        raise ValueError("rejected signal evidence cannot enable alpha")
    signal_status = SignalStatus.ACCEPTED if signal_passed else SignalStatus.REJECTED

    if report_mode == "baseline_only":
        if selected_model_path is not None:
            raise ValueError("baseline_only evidence cannot select a model path")
        policy_candidate_status = PolicyCandidateStatus.NOT_SELECTED
        baseline_fallback_status = BaselineFallbackStatus.SELECTED_FOR_ANALYSIS
        selected_policy_digest = None
        policy_ensemble_members: tuple[str, ...] = ()
    else:
        raise ValueError(
            "archived migration supports only evidence explicitly classified as "
            "baseline_only"
        )

    final_gate_passed = _boolean(
        final_gate,
        "passed",
        field="report.gate.passed",
    )
    release_status = (
        ReleaseStatus.ELIGIBLE
        if final_gate_passed and report_mode != "baseline_only"
        else ReleaseStatus.BLOCKED
    )

    selection_frozen = _boolean(
        model_manifest,
        "selection_frozen_before_test",
        field="model_manifest.selection_frozen_before_test",
    )
    checkpoint_separated = _boolean(
        model_manifest,
        "checkpoint_selection_separated",
        field="model_manifest.checkpoint_selection_separated",
    )
    if not selection_frozen:
        raise ValueError("selection must be frozen before sealed holdout evaluation")
    if not checkpoint_separated:
        raise ValueError("checkpoint and configuration selection must be separated")

    signal_model_kind = _string(
        signal_metadata,
        "model",
        field="signal_metadata.model",
    )
    gate_model_kind = _string(
        signal_gate,
        "model",
        field="report.signal_gate.model",
    )
    _require_equal(signal_model_kind, gate_model_kind, field="signal model kind")

    return MigratedResearchRun(
        research_run_status=ResearchRunStatus.COMPLETED,
        signal_status=signal_status,
        policy_candidate_status=policy_candidate_status,
        baseline_fallback_status=baseline_fallback_status,
        release_status=release_status,
        selected_configuration=selected_configuration,
        selected_policy_digest=selected_policy_digest,
        policy_ensemble_members=policy_ensemble_members,
        holdout_total_return=_number(
            holdout_hybrid,
            "total_return",
            field="report.relative.hybrid.total_return",
        ),
        cost2x_total_return=_number(
            cost2x_hybrid,
            "total_return",
            field="report.cost2x.hybrid.total_return",
        ),
        positive_return_p_value=_number(
            final_gate,
            "positive_return_p_value",
            field="report.gate.positive_return_p_value",
        ),
        signal_model_kind=signal_model_kind,
        signal_dataset_id=signal_dataset_id,
        notes=(
            "signal metadata is not a PPO ensemble artifact",
            "baseline identity result is analysis evidence, not a selected policy",
            "production release remains blocked",
        ),
    )
