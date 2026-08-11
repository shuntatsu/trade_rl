from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from typing import Sequence

from trade_rl.rl.universal_architecture import UniversalArchitectureName


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
            raise ValueError("U6 requires the zero-shot gate to pass before full research")
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

    @property
    def manifest_digest(self) -> str:
        return _digest(
            {
                "version": "universal_full_research_manifest_v1",
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
            }
        )


def validate_full_research_inputs(manifest: UniversalResearchManifest) -> None:
    required_identity = {
        "catalog_digest": manifest.catalog_digest,
        "split_manifest_digest": manifest.split_manifest_digest,
        "normalizer_digest": manifest.normalizer_digest,
        "feature_schema_digest": manifest.feature_schema_digest,
        "seed_manifest_digest": manifest.seed_manifest_digest,
        "checkpoint_digest": manifest.checkpoint_digest,
        "cost_model_digest": manifest.cost_model_digest,
    }
    missing_identity = sorted(key for key, value in required_identity.items() if not value)
    if missing_identity:
        raise ValueError(f"missing required research identity: {', '.join(missing_identity)}")
    if len(set(manifest.required_pairs)) != len(manifest.required_pairs):
        raise ValueError("required paired deliverables must be unique")
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


def build_pair_closure(
    *,
    candidate_names: Sequence[str],
    baseline_names: Sequence[str],
    folds: Sequence[int],
    seeds: Sequence[int],
) -> tuple[str, ...]:
    if not candidate_names or not baseline_names or not folds or not seeds:
        raise ValueError("candidate, baseline, fold, and seed dimensions must be non-empty")
    return tuple(
        f"{candidate}:{baseline}:fold{fold}:seed{seed}"
        for candidate in candidate_names
        for baseline in baseline_names
        for fold in folds
        for seed in seeds
    )
