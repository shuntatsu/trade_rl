"""External, non-circular approval attestations for immutable bundles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import (
    require_aware_datetime,
    require_git_sha,
    require_non_empty,
    require_sha256,
)

RELEASE_ATTESTATION_SCHEMA = "release_attestation_v1"


@dataclass(frozen=True, slots=True)
class ReleaseAttestation:
    attestation_digest: str
    bundle_digest: str
    dataset_id: str
    selection_evaluation_digest: str
    gate_evaluation_digest: str
    gate_evidence_digest: str
    selected_policy_digest: str | None
    git_commit: str
    dependency_digest: str
    approver: str
    approved_at: datetime
    schema_version: str = RELEASE_ATTESTATION_SCHEMA

    def __post_init__(self) -> None:
        for field_name, value in (
            ("attestation_digest", self.attestation_digest),
            ("bundle_digest", self.bundle_digest),
            ("dataset_id", self.dataset_id),
            ("selection_evaluation_digest", self.selection_evaluation_digest),
            ("gate_evaluation_digest", self.gate_evaluation_digest),
            ("gate_evidence_digest", self.gate_evidence_digest),
            ("dependency_digest", self.dependency_digest),
        ):
            require_sha256(value, field=field_name)
        if self.selected_policy_digest is not None:
            require_sha256(self.selected_policy_digest, field="selected_policy_digest")
        require_git_sha(self.git_commit)
        require_non_empty(self.approver, field="approver")
        require_aware_datetime(self.approved_at, field="approved_at")
        if self.schema_version != RELEASE_ATTESTATION_SCHEMA:
            raise ValueError("unsupported release attestation schema")
        if self.attestation_digest != content_digest(self.digest_payload()):
            raise ValueError("release attestation digest mismatch")

    def digest_payload(self) -> dict[str, object]:
        return {
            "approved_at": self.approved_at,
            "approver": self.approver,
            "bundle_digest": self.bundle_digest,
            "dataset_id": self.dataset_id,
            "dependency_digest": self.dependency_digest,
            "gate_evaluation_digest": self.gate_evaluation_digest,
            "gate_evidence_digest": self.gate_evidence_digest,
            "git_commit": self.git_commit,
            "schema_version": self.schema_version,
            "selected_policy_digest": self.selected_policy_digest,
            "selection_evaluation_digest": self.selection_evaluation_digest,
        }

    @classmethod
    def create(
        cls,
        *,
        bundle_digest: str,
        dataset_id: str,
        selection_evaluation_digest: str,
        gate_evaluation_digest: str,
        gate_evidence_digest: str,
        selected_policy_digest: str | None,
        git_commit: str,
        dependency_digest: str,
        approver: str,
        approved_at: datetime,
    ) -> ReleaseAttestation:
        payload = {
            "approved_at": approved_at,
            "approver": approver,
            "bundle_digest": bundle_digest,
            "dataset_id": dataset_id,
            "dependency_digest": dependency_digest,
            "gate_evaluation_digest": gate_evaluation_digest,
            "gate_evidence_digest": gate_evidence_digest,
            "git_commit": git_commit,
            "schema_version": RELEASE_ATTESTATION_SCHEMA,
            "selected_policy_digest": selected_policy_digest,
            "selection_evaluation_digest": selection_evaluation_digest,
        }
        return cls(attestation_digest=content_digest(payload), **payload)


def default_attestation_path(bundle_root: str | Path) -> Path:
    root = Path(bundle_root)
    return root.with_name(f"{root.name}.release.json")


def write_release_attestation(path: str | Path, value: ReleaseAttestation) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(asdict(value)))
    return output


def load_release_attestation(path: str | Path) -> ReleaseAttestation:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("release attestation must be a mapping")
    try:
        approved_at = datetime.fromisoformat(
            str(raw["approved_at"]).replace("Z", "+00:00")
        )
        return ReleaseAttestation(
            attestation_digest=str(raw["attestation_digest"]),
            bundle_digest=str(raw["bundle_digest"]),
            dataset_id=str(raw["dataset_id"]),
            selection_evaluation_digest=str(raw["selection_evaluation_digest"]),
            gate_evaluation_digest=str(raw["gate_evaluation_digest"]),
            gate_evidence_digest=str(raw["gate_evidence_digest"]),
            selected_policy_digest=(
                None
                if raw.get("selected_policy_digest") is None
                else str(raw["selected_policy_digest"])
            ),
            git_commit=str(raw["git_commit"]),
            dependency_digest=str(raw["dependency_digest"]),
            approver=str(raw["approver"]),
            approved_at=approved_at,
            schema_version=str(raw["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("release attestation is invalid") from error


__all__ = [
    "RELEASE_ATTESTATION_SCHEMA",
    "ReleaseAttestation",
    "default_attestation_path",
    "load_release_attestation",
    "write_release_attestation",
]
