"""Resolved one-factor verification for controlled development experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.contracts import (
    ControlledFactor,
    ExperimentDefinition,
    StudyPlan,
)
from trade_rl.evaluation.experiments.evidence import LoadedEvidenceSet
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.runs.artifact import LoadedCandidateRun

_STRATEGIES = (
    "cash",
    "constant_long",
    "constant_short",
    "trend",
    "mean_reversion",
    "ridge24",
    "lightgbm24",
    "ppo",
)


@dataclass(frozen=True, slots=True)
class FactorRule:
    """Resolved fields that may change and strategies that must not change."""

    allowed_paths: frozenset[tuple[str, ...]]
    unaffected_strategies: frozenset[str]


FACTOR_RULES: Mapping[ControlledFactor, FactorRule] = MappingProxyType(
    {
        ControlledFactor.FEATURE_SET: FactorRule(
            allowed_paths=frozenset(
                {
                    ("feature_names",),
                    ("feature_indices",),
                }
            ),
            unaffected_strategies=frozenset(
                {
                    "cash",
                    "constant_long",
                    "constant_short",
                    "trend",
                    "mean_reversion",
                }
            ),
        ),
        ControlledFactor.RULE_SIGNAL: FactorRule(
            allowed_paths=frozenset(
                {
                    ("signal_name",),
                    ("signal_index",),
                }
            ),
            unaffected_strategies=frozenset(
                {
                    "cash",
                    "constant_long",
                    "constant_short",
                    "ridge24",
                    "lightgbm24",
                    "ppo",
                }
            ),
        ),
        ControlledFactor.RULE_THRESHOLDS: FactorRule(
            allowed_paths=frozenset(
                {
                    ("rule_entry_threshold",),
                    ("rule_exit_threshold",),
                }
            ),
            unaffected_strategies=frozenset(
                {
                    "cash",
                    "constant_long",
                    "constant_short",
                    "ridge24",
                    "lightgbm24",
                    "ppo",
                }
            ),
        ),
        ControlledFactor.FORECAST_THRESHOLDS: FactorRule(
            allowed_paths=frozenset(
                {
                    ("forecast_entry_threshold",),
                    ("forecast_exit_threshold",),
                }
            ),
            unaffected_strategies=frozenset(
                {
                    "cash",
                    "constant_long",
                    "constant_short",
                    "trend",
                    "mean_reversion",
                    "ppo",
                }
            ),
        ),
        ControlledFactor.FIT_SYMBOL_SCOPE: FactorRule(
            allowed_paths=frozenset(
                {
                    ("fit_symbol_names",),
                    ("fit_symbol_indices",),
                }
            ),
            unaffected_strategies=frozenset(
                {
                    "cash",
                    "constant_long",
                    "constant_short",
                    "trend",
                    "mean_reversion",
                }
            ),
        ),
        ControlledFactor.PPO_TRAINING_BUDGET: FactorRule(
            allowed_paths=frozenset({("ppo_total_timesteps",)}),
            unaffected_strategies=frozenset(
                {
                    "cash",
                    "constant_long",
                    "constant_short",
                    "trend",
                    "mean_reversion",
                    "ridge24",
                    "lightgbm24",
                }
            ),
        ),
        ControlledFactor.GROSS_BUDGET: FactorRule(
            allowed_paths=frozenset({("gross_budget",)}),
            unaffected_strategies=frozenset({"cash"}),
        ),
    }
)


class ControlledVerificationStatus(StrEnum):
    """Terminal classification of trustworthy controlled-delta evidence."""

    CONTROLLED = "CONTROLLED"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class ControlledVerification:
    """Immutable result of verifying one declared semantic factor."""

    study_digest: str
    experiment_digest: str
    baseline_evidence_digest: str
    candidate_evidence_digest: str
    factor: ControlledFactor
    status: ControlledVerificationStatus
    changed_paths: tuple[tuple[str, ...], ...]
    violations: tuple[str, ...]
    schema_version: str = "controlled_verification_v1"

    def __post_init__(self) -> None:
        for value, field in (
            (self.study_digest, "study_digest"),
            (self.experiment_digest, "experiment_digest"),
            (self.baseline_evidence_digest, "baseline_evidence_digest"),
            (self.candidate_evidence_digest, "candidate_evidence_digest"),
        ):
            try:
                require_sha256(value, field=field)
            except ValueError as error:
                raise ArtifactIntegrityError(str(error)) from error
        if not isinstance(self.factor, ControlledFactor):
            raise ArtifactIntegrityError("verification factor is unsupported")
        if not isinstance(self.status, ControlledVerificationStatus):
            raise ArtifactIntegrityError("verification status is unsupported")
        normalized_paths = tuple(sorted(set(self.changed_paths)))
        if normalized_paths != self.changed_paths or any(
            not path or any(not part for part in path) for path in self.changed_paths
        ):
            raise ArtifactIntegrityError("verification changed paths are not canonical")
        if any(not violation for violation in self.violations):
            raise ArtifactIntegrityError("verification violations must be non-empty")
        if self.status is ControlledVerificationStatus.CONTROLLED and self.violations:
            raise ArtifactIntegrityError("CONTROLLED verification cannot contain violations")
        if self.status is ControlledVerificationStatus.INVALID and not self.violations:
            raise ArtifactIntegrityError("INVALID verification requires violations")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "study_digest": self.study_digest,
            "experiment_digest": self.experiment_digest,
            "baseline_evidence_digest": self.baseline_evidence_digest,
            "candidate_evidence_digest": self.candidate_evidence_digest,
            "factor": self.factor.value,
            "status": self.status.value,
            "changed_paths": [list(path) for path in self.changed_paths],
            "violations": list(self.violations),
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def _canonical_value(value: object) -> str:
    """Return a stable equality token for JSON-compatible semantic config values."""

    return content_digest({"value": value})


def _classify_resolved_delta(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    rule: FactorRule,
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
    """Partition actual top-level resolved changes into all and forbidden paths."""

    missing = object()
    changed: list[tuple[str, ...]] = []
    for key in sorted(set(baseline) | set(candidate)):
        baseline_value = baseline.get(key, missing)
        candidate_value = candidate.get(key, missing)
        if baseline_value is missing or candidate_value is missing:
            changed.append((key,))
            continue
        if _canonical_value(baseline_value) != _canonical_value(candidate_value):
            changed.append((key,))
    changed_paths = tuple(changed)
    forbidden = tuple(path for path in changed_paths if path not in rule.allowed_paths)
    return changed_paths, forbidden


def _without_seed(payload: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    normalized.pop("ppo_seed", None)
    return normalized


def _strategy_matrix(
    run: LoadedCandidateRun,
    *,
    expected_symbols: tuple[str, ...],
    label: str,
) -> tuple[dict[tuple[str, str], np.ndarray], tuple[str, ...]]:
    by_symbol = run.summary.get("by_symbol")
    if not isinstance(by_symbol, list):
        raise ArtifactIntegrityError(f"{label} by_symbol evidence is malformed")
    matrix: dict[tuple[str, str], np.ndarray] = {}
    observed_roster: tuple[str, ...] | None = None
    if len(by_symbol) != len(expected_symbols):
        return matrix, ()
    for expected_symbol, symbol_entry in zip(expected_symbols, by_symbol, strict=True):
        if not isinstance(symbol_entry, dict):
            raise ArtifactIntegrityError(f"{label} symbol evidence is malformed")
        if symbol_entry.get("symbol") != expected_symbol:
            return matrix, ()
        strategies = symbol_entry.get("strategies")
        if not isinstance(strategies, list):
            raise ArtifactIntegrityError(f"{label} strategy evidence is malformed")
        names: list[str] = []
        for strategy_entry in strategies:
            if not isinstance(strategy_entry, dict):
                raise ArtifactIntegrityError(f"{label} strategy entry is malformed")
            name = strategy_entry.get("name")
            return_key = strategy_entry.get("return_key")
            if not isinstance(name, str) or not name or not isinstance(return_key, str):
                raise ArtifactIntegrityError(f"{label} strategy entry is malformed")
            names.append(name)
            values = run.returns.get(return_key)
            if values is None:
                raise ArtifactIntegrityError(f"{label} raw return evidence is missing")
            key = (expected_symbol, name)
            if key in matrix:
                raise ArtifactIntegrityError(f"{label} strategy evidence is duplicated")
            matrix[key] = values
        roster = tuple(names)
        if observed_roster is None:
            observed_roster = roster
        elif roster != observed_roster:
            return matrix, ()
    return matrix, () if observed_roster is None else observed_roster


def _evidence_violations(
    *,
    plan: StudyPlan,
    evidence: LoadedEvidenceSet,
    label: str,
) -> tuple[list[str], dict[int, dict[tuple[str, str], np.ndarray]]]:
    violations: list[str] = []
    if evidence.evidence.ppo_seeds != plan.ppo_seeds:
        violations.append(f"{label} PPO seed roster differs from frozen Study plan")
    if (
        content_digest(evidence.semantic_config)
        != evidence.evidence.semantic_config_digest
    ):
        violations.append(f"{label} semantic config digest does not match evidence")
    if set(evidence.runs) != set(evidence.evidence.ppo_seeds):
        violations.append(f"{label} Run seed roster does not match EvidenceSet")

    matrices: dict[int, dict[tuple[str, str], np.ndarray]] = {}
    for seed in evidence.evidence.ppo_seeds:
        run = evidence.runs.get(seed)
        if run is None:
            continue
        if run.summary.get("dataset_id") != plan.dataset_id:
            violations.append(
                f"{label} dataset identity differs from frozen Study plan"
            )
        dataset_artifact = run.summary.get("dataset_artifact")
        if not isinstance(dataset_artifact, dict):
            raise ArtifactIntegrityError(
                f"{label} dataset artifact evidence is malformed"
            )
        if (
            dataset_artifact.get("schema_version") != plan.dataset_artifact_schema
            or dataset_artifact.get("artifact_digest") != plan.dataset_artifact_digest
        ):
            violations.append(
                f"{label} dataset artifact differs from frozen Study plan"
            )
        symbols = run.summary.get("symbols")
        if symbols != list(plan.symbols):
            violations.append(f"{label} symbol roster differs from frozen Study plan")
        if run.provenance.get("implementation_digest") != plan.implementation_digest:
            violations.append(
                f"{label} implementation provenance differs from frozen Study plan"
            )
        if (
            run.provenance.get("runtime_environment_digest")
            != plan.runtime_environment_digest
        ):
            violations.append(
                f"{label} runtime provenance differs from frozen Study plan"
            )
        matrix, roster = _strategy_matrix(
            run,
            expected_symbols=plan.symbols,
            label=label,
        )
        matrices[seed] = matrix
        if roster != _STRATEGIES:
            violations.append(
                f"{label} strategy roster differs from frozen Study roster"
            )
    return violations, matrices


def _unaffected_strategy_violations(
    *,
    plan: StudyPlan,
    rule: FactorRule,
    baseline_matrices: Mapping[int, Mapping[tuple[str, str], np.ndarray]],
    candidate_matrices: Mapping[int, Mapping[tuple[str, str], np.ndarray]],
) -> list[str]:
    violations: list[str] = []
    for seed in plan.ppo_seeds:
        baseline = baseline_matrices.get(seed)
        candidate = candidate_matrices.get(seed)
        if baseline is None or candidate is None:
            continue
        for symbol in plan.symbols:
            for strategy in sorted(rule.unaffected_strategies):
                key = (symbol, strategy)
                baseline_values = baseline.get(key)
                candidate_values = candidate.get(key)
                if baseline_values is None or candidate_values is None:
                    violations.append(
                        f"unaffected strategy evidence is missing: {symbol}/{strategy}/seed-{seed}"
                    )
                    continue
                if not np.array_equal(baseline_values, candidate_values):
                    violations.append(
                        f"unaffected strategy raw returns changed: {symbol}/{strategy}/seed-{seed}"
                    )
    return violations


def verify_controlled_delta(
    *,
    plan: StudyPlan,
    definition: ExperimentDefinition,
    baseline: LoadedEvidenceSet,
    candidate: LoadedEvidenceSet,
) -> ControlledVerification:
    """Classify trustworthy baseline/candidate evidence as CONTROLLED or INVALID."""

    violations: list[str] = []
    if definition.study_digest != plan.digest:
        violations.append("Experiment definition does not belong to the frozen Study")
    if definition.baseline_evidence_digest != baseline.evidence.fingerprint:
        violations.append("Experiment definition baseline evidence reference is stale")
    if definition.factor not in plan.allowed_factors:
        violations.append("declared factor is not allowed by the frozen Study plan")
    if definition.candidate_config.ppo_seed != plan.ppo_seeds[0]:
        violations.append("frozen definition seed does not match the Study seed policy")

    # Baseline lineage eligibility is owned by the workflow state machine. The
    # frozen definition binds the exact baseline EvidenceSet fingerprint, so a
    # prior ACCEPT_CANDIDATE may legitimately differ from StudyPlan.baseline_config.
    defined_candidate = _without_seed(definition.candidate_config.to_payload())
    if candidate.semantic_config != defined_candidate:
        violations.append("candidate evidence does not match the frozen definition")

    baseline_violations, baseline_matrices = _evidence_violations(
        plan=plan,
        evidence=baseline,
        label="baseline",
    )
    candidate_violations, candidate_matrices = _evidence_violations(
        plan=plan,
        evidence=candidate,
        label="candidate",
    )
    violations.extend(baseline_violations)
    violations.extend(candidate_violations)

    rule = FACTOR_RULES[definition.factor]
    changed_paths, forbidden_paths = _classify_resolved_delta(
        baseline.semantic_config,
        candidate.semantic_config,
        rule,
    )
    if not changed_paths:
        violations.append("declared factor is a no-op")
    if forbidden_paths:
        rendered = ", ".join(".".join(path) for path in forbidden_paths)
        violations.append(f"uncontrolled resolved delta: {rendered}")

    violations.extend(
        _unaffected_strategy_violations(
            plan=plan,
            rule=rule,
            baseline_matrices=baseline_matrices,
            candidate_matrices=candidate_matrices,
        )
    )

    unique_violations = tuple(dict.fromkeys(violations))
    status = (
        ControlledVerificationStatus.INVALID
        if unique_violations
        else ControlledVerificationStatus.CONTROLLED
    )
    return ControlledVerification(
        study_digest=plan.digest,
        experiment_digest=definition.digest,
        baseline_evidence_digest=baseline.evidence.fingerprint,
        candidate_evidence_digest=candidate.evidence.fingerprint,
        factor=definition.factor,
        status=status,
        changed_paths=changed_paths,
        violations=unique_violations,
    )


__all__ = [
    "FACTOR_RULES",
    "ControlledVerification",
    "ControlledVerificationStatus",
    "FactorRule",
    "verify_controlled_delta",
]
