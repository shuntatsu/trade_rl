from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.rl.universal_architecture import UniversalArchitectureName


class FullResearchAlgorithm(StrEnum):
    PPO = "ppo"
    LAGRANGIAN = "lagrangian"
    DISCOUNTED = "discounted"


@dataclass(frozen=True)
class UniversalFullResearchPlan:
    selected_architecture: UniversalArchitectureName
    algorithms: tuple[FullResearchAlgorithm, ...]
    zero_shot_gate_passed: bool

    @classmethod
    def create(
        cls,
        *,
        selected_architecture: UniversalArchitectureName | str,
        zero_shot_gate_passed: bool,
        algorithms: Sequence[FullResearchAlgorithm | str],
    ) -> "UniversalFullResearchPlan":
        if not zero_shot_gate_passed:
            raise ValueError(
                "U6 requires the zero-shot gate to pass before full research"
            )
        resolved = tuple(FullResearchAlgorithm(value) for value in algorithms)
        if len(set(resolved)) != len(resolved):
            raise ValueError("full-research algorithms must be unique")
        required = set(FullResearchAlgorithm)
        if set(resolved) != required:
            missing = sorted(value.value for value in required.difference(resolved))
            extra = sorted(value.value for value in set(resolved).difference(required))
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if extra:
                details.append("extra=" + ",".join(extra))
            raise ValueError(
                "U6 requires the closed PPO/Lagrangian/discounted comparison: "
                + "; ".join(details)
            )
        return cls(
            selected_architecture=UniversalArchitectureName(selected_architecture),
            algorithms=resolved,
            zero_shot_gate_passed=True,
        )


@dataclass(frozen=True)
class UniversalResearchManifest:
    catalog_digest: str
    split_manifest_digest: str
    normalizer_digest: str
    feature_schema_digest: str
    seed_manifest_digest: str
    architecture_name: UniversalArchitectureName
    checkpoint_digest: str
    cost_model_digest: str
    required_pairs: tuple[str, ...]
    completed_pairs: tuple[str, ...]
    bc_teacher_digest: str | None = None
    software_identity: str | None = None
    universe_manifest_digest: str | None = None
    normalization_fit_scope_digest: str | None = None
    observation_contract_digest: str | None = None
    architecture_evidence_digest: str | None = None
    zero_shot_gate_digest: str | None = None
    zero_shot_gate_passed: bool = False
    paired_baseline_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "architecture_name",
            UniversalArchitectureName(self.architecture_name),
        )
        if not isinstance(self.zero_shot_gate_passed, bool):
            raise ValueError("zero_shot_gate_passed must be a boolean")

    @property
    def manifest_digest(self) -> str:
        return content_digest(
            {
                "version": "universal_full_research_manifest_v2",
                "catalog_digest": self.catalog_digest,
                "split_manifest_digest": self.split_manifest_digest,
                "normalizer_digest": self.normalizer_digest,
                "feature_schema_digest": self.feature_schema_digest,
                "seed_manifest_digest": self.seed_manifest_digest,
                "architecture_name": self.architecture_name.value,
                "checkpoint_digest": self.checkpoint_digest,
                "cost_model_digest": self.cost_model_digest,
                "required_pairs": self.required_pairs,
                "completed_pairs": self.completed_pairs,
                "bc_teacher_digest": self.bc_teacher_digest,
                "software_identity": self.software_identity,
                "universe_manifest_digest": self.universe_manifest_digest,
                "normalization_fit_scope_digest": self.normalization_fit_scope_digest,
                "observation_contract_digest": self.observation_contract_digest,
                "architecture_evidence_digest": self.architecture_evidence_digest,
                "zero_shot_gate_digest": self.zero_shot_gate_digest,
                "zero_shot_gate_passed": self.zero_shot_gate_passed,
                "paired_baseline_digest": self.paired_baseline_digest,
            }
        )


def _legacy_identity(manifest: UniversalResearchManifest) -> dict[str, str]:
    return {
        "catalog_digest": manifest.catalog_digest,
        "split_manifest_digest": manifest.split_manifest_digest,
        "normalizer_digest": manifest.normalizer_digest,
        "feature_schema_digest": manifest.feature_schema_digest,
        "seed_manifest_digest": manifest.seed_manifest_digest,
        "checkpoint_digest": manifest.checkpoint_digest,
        "cost_model_digest": manifest.cost_model_digest,
    }


def _validate_legacy_identity(manifest: UniversalResearchManifest) -> None:
    missing_identity = sorted(
        key for key, value in _legacy_identity(manifest).items() if not value
    )
    if missing_identity:
        raise ValueError(
            f"missing required research identity: {', '.join(missing_identity)}"
        )
    if not manifest.required_pairs:
        raise ValueError("required paired deliverables must not be empty")
    if len(set(manifest.required_pairs)) != len(manifest.required_pairs):
        raise ValueError("required paired deliverables must be unique")
    if len(set(manifest.completed_pairs)) != len(manifest.completed_pairs):
        raise ValueError("completed paired deliverables must be unique")


def validate_full_research_inputs(manifest: UniversalResearchManifest) -> None:
    """Validate the legacy completion contract used by existing U6 callers."""

    _validate_legacy_identity(manifest)
    completed = set(manifest.completed_pairs)
    missing_pairs = [pair for pair in manifest.required_pairs if pair not in completed]
    if missing_pairs:
        raise ValueError(
            "missing paired deliverables: " + ", ".join(sorted(missing_pairs))
        )
    unexpected = sorted(completed.difference(manifest.required_pairs))
    if unexpected:
        raise ValueError(
            "completed paired deliverables are outside the manifest closure: "
            + ", ".join(unexpected)
        )


def validate_full_research_start_inputs(manifest: UniversalResearchManifest) -> None:
    """Validate immutable U3-U5 evidence before a U6 research run starts."""

    _validate_legacy_identity(manifest)
    strict_identity = {
        **_legacy_identity(manifest),
        "universe_manifest_digest": manifest.universe_manifest_digest,
        "normalization_fit_scope_digest": manifest.normalization_fit_scope_digest,
        "observation_contract_digest": manifest.observation_contract_digest,
        "architecture_evidence_digest": manifest.architecture_evidence_digest,
        "zero_shot_gate_digest": manifest.zero_shot_gate_digest,
        "paired_baseline_digest": manifest.paired_baseline_digest,
        "bc_teacher_digest": manifest.bc_teacher_digest,
        "software_identity": manifest.software_identity,
    }
    missing_identity = sorted(
        key for key, value in strict_identity.items() if value is None or not value
    )
    if missing_identity:
        raise ValueError(
            "missing required U6 immutable identity: " + ", ".join(missing_identity)
        )
    for field_name, raw_value in strict_identity.items():
        if raw_value is None:
            raise RuntimeError("validated U6 identity unexpectedly became unavailable")
        require_sha256(raw_value, field=field_name)
    if not manifest.zero_shot_gate_passed:
        raise ValueError("U6 requires a passed zero-shot gate")

    required = set(manifest.required_pairs)
    completed = set(manifest.completed_pairs)
    unexpected = sorted(completed.difference(required))
    if unexpected:
        raise ValueError(
            "completed paired deliverables are outside the manifest closure: "
            + ", ".join(unexpected)
        )


def validate_full_research_completion(manifest: UniversalResearchManifest) -> None:
    """Require strict U6 identity and complete paired algorithm/baseline closure."""

    validate_full_research_start_inputs(manifest)
    completed = set(manifest.completed_pairs)
    missing_pairs = [pair for pair in manifest.required_pairs if pair not in completed]
    if missing_pairs:
        raise ValueError(
            "missing paired deliverables: " + ", ".join(sorted(missing_pairs))
        )


def _validated_dimension(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    resolved = tuple(str(value).strip() for value in values)
    if not resolved or any(not value for value in resolved):
        raise ValueError(f"{field} must contain non-empty values")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field} must contain unique values")
    return resolved


def _validated_int_dimension(values: Sequence[int], *, field: str) -> tuple[int, ...]:
    resolved = tuple(values)
    if not resolved:
        raise ValueError(f"{field} must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in resolved
    ):
        raise ValueError(f"{field} must contain non-negative integers")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field} must contain unique values")
    return resolved


def build_pair_closure(
    *,
    candidate_names: Sequence[str],
    baseline_names: Sequence[str],
    folds: Sequence[int],
    seeds: Sequence[int],
) -> tuple[str, ...]:
    candidates = _validated_dimension(candidate_names, field="candidate_names")
    baselines = _validated_dimension(baseline_names, field="baseline_names")
    resolved_folds = _validated_int_dimension(folds, field="folds")
    resolved_seeds = _validated_int_dimension(seeds, field="seeds")
    return tuple(
        f"{candidate}:{baseline}:fold{fold}:seed{seed}"
        for candidate in candidates
        for baseline in baselines
        for fold in resolved_folds
        for seed in resolved_seeds
    )


def build_full_research_pair_closure(
    *,
    algorithms: Sequence[FullResearchAlgorithm | str],
    baseline_names: Sequence[str],
    folds: Sequence[int],
    seeds: Sequence[int],
) -> tuple[str, ...]:
    resolved_algorithms = tuple(FullResearchAlgorithm(value) for value in algorithms)
    if not resolved_algorithms:
        raise ValueError("algorithms must not be empty")
    if len(set(resolved_algorithms)) != len(resolved_algorithms):
        raise ValueError("algorithms must contain unique values")
    baselines = _validated_dimension(baseline_names, field="baseline_names")
    resolved_folds = _validated_int_dimension(folds, field="folds")
    resolved_seeds = _validated_int_dimension(seeds, field="seeds")
    return tuple(
        f"{algorithm.value}:{baseline}:fold{fold}:seed{seed}"
        for algorithm in resolved_algorithms
        for baseline in baselines
        for fold in resolved_folds
        for seed in resolved_seeds
    )
