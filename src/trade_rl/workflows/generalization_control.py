"""Immutable comparison control for Stage A and Stage B generalization work."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_git_sha, require_non_empty, require_sha256

GENERALIZATION_CONTROL_SCHEMA: Final = "generalization_control_manifest_v1"
GeneralizationStageScope = Literal["stage_a", "stage_a_b"]
_ALLOWED_STAGE_SCOPES: Final = frozenset({"stage_a", "stage_a_b"})


def _normalize_non_negative_ints(
    values: tuple[int, ...], *, field: str
) -> tuple[int, ...]:
    if not values:
        raise ValueError(f"{field} must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ValueError(f"{field} must contain non-negative integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must contain unique values")
    return tuple(sorted(values))


@dataclass(frozen=True, slots=True)
class GeneralizationControlManifest:
    """Content-addressed identities required for an apples-to-apples control."""

    control_name: str
    base_commit: str
    action_schema: str
    policy_identity: str
    dataset_identity: str
    feature_identity: str
    execution_identity: str
    evaluation_identity: str
    seeds: tuple[int, ...]
    folds: tuple[int, ...]
    stage_scope: GeneralizationStageScope
    schema_version: str = GENERALIZATION_CONTROL_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != GENERALIZATION_CONTROL_SCHEMA:
            raise ValueError("unsupported generalization control schema")
        if self.stage_scope not in _ALLOWED_STAGE_SCOPES:
            raise ValueError("generalization control stage scope is invalid")

        control_name = require_non_empty(
            self.control_name, field="generalization_control.control_name"
        )
        action_schema = require_non_empty(
            self.action_schema, field="generalization_control.action_schema"
        )
        base_commit = require_git_sha(
            self.base_commit, field="generalization_control.base_commit"
        )
        for field, value in self.comparison_identities.items():
            require_sha256(value, field=f"generalization_control.{field}_identity")
        seeds = _normalize_non_negative_ints(
            self.seeds, field="generalization_control.seeds"
        )
        folds = _normalize_non_negative_ints(
            self.folds, field="generalization_control.folds"
        )

        object.__setattr__(self, "control_name", control_name)
        object.__setattr__(self, "action_schema", action_schema)
        object.__setattr__(self, "base_commit", base_commit)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "folds", folds)

        expected_digest = content_digest(self.digest_payload())
        if self.digest and self.digest != expected_digest:
            raise ValueError("generalization control digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    @property
    def comparison_identities(self) -> dict[str, str]:
        return {
            "dataset": self.dataset_identity,
            "evaluation": self.evaluation_identity,
            "execution": self.execution_identity,
            "feature": self.feature_identity,
            "policy": self.policy_identity,
        }

    def digest_payload(self) -> dict[str, object]:
        return {
            "action_schema": self.action_schema,
            "base_commit": self.base_commit,
            "control_name": self.control_name,
            "dataset_identity": self.dataset_identity,
            "evaluation_identity": self.evaluation_identity,
            "execution_identity": self.execution_identity,
            "feature_identity": self.feature_identity,
            "folds": self.folds,
            "policy_identity": self.policy_identity,
            "schema_version": self.schema_version,
            "seeds": self.seeds,
            "stage_scope": self.stage_scope,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}


def build_generalization_control_manifest(
    *,
    control_name: str,
    base_commit: str,
    action_schema: str,
    policy_identity: str,
    dataset_identity: str,
    feature_identity: str,
    execution_identity: str,
    evaluation_identity: str,
    seeds: tuple[int, ...],
    folds: tuple[int, ...],
    stage_scope: GeneralizationStageScope,
) -> GeneralizationControlManifest:
    return GeneralizationControlManifest(
        control_name=control_name,
        base_commit=base_commit,
        action_schema=action_schema,
        policy_identity=policy_identity,
        dataset_identity=dataset_identity,
        feature_identity=feature_identity,
        execution_identity=execution_identity,
        evaluation_identity=evaluation_identity,
        seeds=seeds,
        folds=folds,
        stage_scope=stage_scope,
    )


def write_generalization_control_manifest(
    path: str | Path, manifest: GeneralizationControlManifest
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(manifest.to_json_dict()))
    return output


def load_generalization_control_manifest(
    path: str | Path,
) -> GeneralizationControlManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("generalization control manifest must be a JSON object")
    required = {
        "action_schema",
        "base_commit",
        "control_name",
        "dataset_identity",
        "digest",
        "evaluation_identity",
        "execution_identity",
        "feature_identity",
        "folds",
        "policy_identity",
        "schema_version",
        "seeds",
        "stage_scope",
    }
    if set(payload) != required:
        raise ValueError("generalization control field closure mismatch")
    raw_seeds = payload["seeds"]
    raw_folds = payload["folds"]
    if not isinstance(raw_seeds, list) or not isinstance(raw_folds, list):
        raise ValueError("generalization control seeds and folds must be lists")
    return GeneralizationControlManifest(
        control_name=payload["control_name"],
        base_commit=payload["base_commit"],
        action_schema=payload["action_schema"],
        policy_identity=payload["policy_identity"],
        dataset_identity=payload["dataset_identity"],
        feature_identity=payload["feature_identity"],
        execution_identity=payload["execution_identity"],
        evaluation_identity=payload["evaluation_identity"],
        seeds=tuple(raw_seeds),
        folds=tuple(raw_folds),
        stage_scope=cast(GeneralizationStageScope, payload["stage_scope"]),
        schema_version=payload["schema_version"],
        digest=payload["digest"],
    )


__all__ = [
    "GENERALIZATION_CONTROL_SCHEMA",
    "GeneralizationControlManifest",
    "GeneralizationStageScope",
    "build_generalization_control_manifest",
    "load_generalization_control_manifest",
    "write_generalization_control_manifest",
]
