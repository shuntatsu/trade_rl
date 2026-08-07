"""Verification-only Ed25519 release attestations for immutable bundles."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import (
    require_aware_datetime,
    require_git_sha,
    require_non_empty,
    require_sha256,
)
from trade_rl.release.asymmetric import (
    PublicVerificationKey,
    SignedEvidenceEnvelope,
    verify_signed_payload,
)

RELEASE_ATTESTATION_SCHEMA = "release_attestation_ed25519_v3"
RELEASE_PURPOSE = "release-verification"
_SELECTED_FINAL = "research_selected_final"
_BASELINE_RELEASE = "baseline_release"


def _utc(value: datetime, *, field: str) -> datetime:
    return require_aware_datetime(value, field=field).astimezone(UTC)


def _optional_digest(value: str | None, *, field: str) -> None:
    if value is not None:
        require_sha256(value, field=field)


@dataclass(frozen=True, slots=True)
class ReleaseAttestation:
    """Detached external approval bound to one complete serving identity."""

    attestation_digest: str
    bundle_digest: str
    dataset_id: str
    training_run_digest: str | None
    run_kind: str
    selection_proposal_digest: str | None
    selection_authorization_digest: str | None
    walk_forward_run_digest: str | None
    gate_evidence_digest: str | None
    confirmation_evidence_digest: str | None
    selected_policy_digest: str | None
    git_commit: str
    dependency_digest: str
    approver: str
    approved_at: datetime
    expires_at: datetime
    key_id: str
    signature: str
    schema_version: str = RELEASE_ATTESTATION_SCHEMA

    @property
    def digest(self) -> str:
        return self.attestation_digest

    def __post_init__(self) -> None:
        require_sha256(self.attestation_digest, field="attestation_digest")
        require_sha256(self.bundle_digest, field="bundle_digest")
        require_sha256(self.dataset_id, field="dataset_id")
        for field, value in (
            ("training_run_digest", self.training_run_digest),
            ("selection_proposal_digest", self.selection_proposal_digest),
            ("selection_authorization_digest", self.selection_authorization_digest),
            ("walk_forward_run_digest", self.walk_forward_run_digest),
            ("gate_evidence_digest", self.gate_evidence_digest),
            ("confirmation_evidence_digest", self.confirmation_evidence_digest),
            ("selected_policy_digest", self.selected_policy_digest),
        ):
            _optional_digest(value, field=field)
        require_git_sha(self.git_commit)
        require_sha256(self.dependency_digest, field="dependency_digest")
        require_non_empty(self.approver, field="approver")
        require_non_empty(self.key_id, field="key_id")
        approved = _utc(self.approved_at, field="approved_at")
        expires = _utc(self.expires_at, field="expires_at")
        object.__setattr__(self, "approved_at", approved)
        object.__setattr__(self, "expires_at", expires)
        if expires <= approved:
            raise ValueError("release attestation expires_at must follow approved_at")
        if self.schema_version != RELEASE_ATTESTATION_SCHEMA:
            raise ValueError("unsupported release attestation schema")
        if self.run_kind == _SELECTED_FINAL:
            required = (
                self.training_run_digest,
                self.selection_proposal_digest,
                self.selection_authorization_digest,
                self.walk_forward_run_digest,
                self.gate_evidence_digest,
                self.confirmation_evidence_digest,
                self.selected_policy_digest,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "selected-final release attestation requires the complete evidence chain"
                )
        elif self.run_kind == _BASELINE_RELEASE:
            if any(
                value is not None
                for value in (
                    self.training_run_digest,
                    self.selection_proposal_digest,
                    self.selection_authorization_digest,
                    self.walk_forward_run_digest,
                    self.gate_evidence_digest,
                    self.confirmation_evidence_digest,
                    self.selected_policy_digest,
                )
            ):
                raise ValueError(
                    "baseline release attestation cannot contain training evidence"
                )
        else:
            raise ValueError("release attestation run_kind is unsupported")
        if self.attestation_digest != content_digest(self.digest_payload()):
            raise ValueError("release attestation digest mismatch")

    def digest_payload(self) -> dict[str, object]:
        return {
            "approved_at": self.approved_at,
            "approver": self.approver,
            "bundle_digest": self.bundle_digest,
            "confirmation_evidence_digest": self.confirmation_evidence_digest,
            "dataset_id": self.dataset_id,
            "dependency_digest": self.dependency_digest,
            "expires_at": self.expires_at,
            "gate_evidence_digest": self.gate_evidence_digest,
            "git_commit": self.git_commit,
            "key_id": self.key_id,
            "run_kind": self.run_kind,
            "schema_version": self.schema_version,
            "selected_policy_digest": self.selected_policy_digest,
            "selection_authorization_digest": self.selection_authorization_digest,
            "selection_proposal_digest": self.selection_proposal_digest,
            "training_run_digest": self.training_run_digest,
            "walk_forward_run_digest": self.walk_forward_run_digest,
        }

    def envelope(self) -> SignedEvidenceEnvelope:
        return SignedEvidenceEnvelope(
            key_id=self.key_id,
            purpose=RELEASE_PURPOSE,
            payload_digest=self.attestation_digest,
            signed_at=self.approved_at,
            signature=self.signature,
        )

    def verify(
        self,
        trusted_keys: Mapping[str, PublicVerificationKey],
        *,
        trusted_at: datetime,
    ) -> None:
        now = _utc(trusted_at, field="trusted_at")
        if now > self.expires_at:
            raise ValueError("release attestation is expired")
        verify_signed_payload(
            self.digest_payload(),
            self.envelope(),
            trusted_keys=trusted_keys,
            trusted_at=now,
            required_purpose=RELEASE_PURPOSE,
        )

    def require_bundle_identity(
        self,
        *,
        bundle_digest: str,
        dataset_id: str,
        training_run_digest: str | None,
        run_kind: str,
        selection_proposal_digest: str | None,
        selection_authorization_digest: str | None,
        walk_forward_run_digest: str | None,
        gate_evidence_digest: str | None,
        confirmation_evidence_digest: str | None,
        selected_policy_digest: str | None,
    ) -> None:
        observed = (
            self.bundle_digest,
            self.dataset_id,
            self.training_run_digest,
            self.run_kind,
            self.selection_proposal_digest,
            self.selection_authorization_digest,
            self.walk_forward_run_digest,
            self.gate_evidence_digest,
            self.confirmation_evidence_digest,
            self.selected_policy_digest,
        )
        expected = (
            bundle_digest,
            dataset_id,
            training_run_digest,
            run_kind,
            selection_proposal_digest,
            selection_authorization_digest,
            walk_forward_run_digest,
            gate_evidence_digest,
            confirmation_evidence_digest,
            selected_policy_digest,
        )
        if observed != expected:
            raise ValueError(
                "release attestation identity does not match serving bundle"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            **self.digest_payload(),
            "attestation_digest": self.attestation_digest,
            "signature": self.signature,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> ReleaseAttestation:
        expected = {
            "approved_at",
            "approver",
            "attestation_digest",
            "bundle_digest",
            "confirmation_evidence_digest",
            "dataset_id",
            "dependency_digest",
            "expires_at",
            "gate_evidence_digest",
            "git_commit",
            "key_id",
            "run_kind",
            "schema_version",
            "selected_policy_digest",
            "selection_authorization_digest",
            "selection_proposal_digest",
            "signature",
            "training_run_digest",
            "walk_forward_run_digest",
        }
        if set(raw) != expected:
            raise ValueError("release attestation fields are invalid")
        try:
            return cls(
                attestation_digest=_string(raw, "attestation_digest"),
                bundle_digest=_string(raw, "bundle_digest"),
                dataset_id=_string(raw, "dataset_id"),
                training_run_digest=_optional_string(raw, "training_run_digest"),
                run_kind=_string(raw, "run_kind"),
                selection_proposal_digest=_optional_string(
                    raw, "selection_proposal_digest"
                ),
                selection_authorization_digest=_optional_string(
                    raw, "selection_authorization_digest"
                ),
                walk_forward_run_digest=_optional_string(
                    raw, "walk_forward_run_digest"
                ),
                gate_evidence_digest=_optional_string(raw, "gate_evidence_digest"),
                confirmation_evidence_digest=_optional_string(
                    raw, "confirmation_evidence_digest"
                ),
                selected_policy_digest=_optional_string(raw, "selected_policy_digest"),
                git_commit=_string(raw, "git_commit"),
                dependency_digest=_string(raw, "dependency_digest"),
                approver=_string(raw, "approver"),
                approved_at=_datetime(raw, "approved_at"),
                expires_at=_datetime(raw, "expires_at"),
                key_id=_string(raw, "key_id"),
                signature=_string(raw, "signature"),
                schema_version=_string(raw, "schema_version"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("release attestation is invalid") from error


def _string(raw: Mapping[str, object], field: str) -> str:
    value = raw[field]
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _optional_string(raw: Mapping[str, object], field: str) -> str | None:
    value = raw[field]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return value


def _datetime(raw: Mapping[str, object], field: str) -> datetime:
    try:
        return datetime.fromisoformat(_string(raw, field).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 datetime") from error


def default_attestation_path(bundle_root: str | Path) -> Path:
    root = Path(bundle_root)
    return root.with_name(f"{root.name}.release.json")


def write_release_attestation(path: str | Path, value: ReleaseAttestation) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("release attestation is immutable")
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(value.to_mapping()))
    temporary.replace(output)
    return output


def load_release_attestation(path: str | Path) -> ReleaseAttestation:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("release attestation must be a mapping")
    return ReleaseAttestation.from_mapping(raw)


__all__ = [
    "RELEASE_ATTESTATION_SCHEMA",
    "RELEASE_PURPOSE",
    "ReleaseAttestation",
    "default_attestation_path",
    "load_release_attestation",
    "write_release_attestation",
]
