"""Signed authorization for one walk-forward-selected final training run."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
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

SELECTION_PROPOSAL_SCHEMA = "selection_proposal_v2"
SELECTION_AUTHORIZATION_SCHEMA = "selection_authorization_ed25519_v2"
_SELECTION_PURPOSE = "selection-authorization"


def _utc(value: datetime, *, field: str) -> datetime:
    return require_aware_datetime(value, field=field).astimezone(UTC)


def _seeds(value: tuple[int, ...]) -> tuple[int, ...]:
    seeds = tuple(value)
    if (
        len(seeds) < 2
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in seeds
        )
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("selection proposal requires unique non-negative seeds")
    return seeds


def _resume_digests(value: tuple[tuple[int, str], ...]) -> tuple[tuple[int, str], ...]:
    items = tuple(value)
    seen: set[int] = set()
    normalized: list[tuple[int, str]] = []
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("resume checkpoint digests must contain seed/digest pairs")
        seed, digest = item
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("resume checkpoint seed must be a non-negative integer")
        if seed in seen:
            raise ValueError("resume checkpoint seeds must be unique")
        if not isinstance(digest, str):
            raise ValueError("resume checkpoint digest must be a string")
        require_sha256(digest, field="resume_checkpoint_digest")
        seen.add(seed)
        normalized.append((seed, digest))
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class SelectionProposal:
    """Immutable identity of the exact candidate proposed for final training."""

    digest: str
    walk_forward_run_digest: str
    gate_evidence_digest: str
    execution_sensitivity_digest: str
    dataset_id: str
    selected_configuration: str
    candidate_config_digest: str
    seeds: tuple[int, ...]
    git_commit: str
    dependency_digest: str
    resume_checkpoint_digests: tuple[tuple[int, str], ...]
    schema_version: str = SELECTION_PROPOSAL_SCHEMA

    def __post_init__(self) -> None:
        for name, value in (
            ("digest", self.digest),
            ("walk_forward_run_digest", self.walk_forward_run_digest),
            ("gate_evidence_digest", self.gate_evidence_digest),
            ("execution_sensitivity_digest", self.execution_sensitivity_digest),
            ("dataset_id", self.dataset_id),
            ("candidate_config_digest", self.candidate_config_digest),
            ("dependency_digest", self.dependency_digest),
        ):
            require_sha256(value, field=name)
        require_git_sha(self.git_commit)
        object.__setattr__(
            self,
            "selected_configuration",
            require_non_empty(
                self.selected_configuration,
                field="selected_configuration",
            ),
        )
        object.__setattr__(self, "seeds", _seeds(self.seeds))
        object.__setattr__(
            self,
            "resume_checkpoint_digests",
            _resume_digests(self.resume_checkpoint_digests),
        )
        if self.schema_version != SELECTION_PROPOSAL_SCHEMA:
            raise ValueError("unsupported selection proposal schema")
        if self.digest != content_digest(self.digest_payload()):
            raise ValueError("selection proposal digest mismatch")

    def digest_payload(self) -> dict[str, object]:
        return {
            "candidate_config_digest": self.candidate_config_digest,
            "dataset_id": self.dataset_id,
            "dependency_digest": self.dependency_digest,
            "execution_sensitivity_digest": self.execution_sensitivity_digest,
            "git_commit": self.git_commit,
            "resume_checkpoint_digests": self.resume_checkpoint_digests,
            "schema_version": self.schema_version,
            "seeds": self.seeds,
            "selected_configuration": self.selected_configuration,
            "walk_forward_run_digest": self.walk_forward_run_digest,
            "gate_evidence_digest": self.gate_evidence_digest,
        }

    @classmethod
    def create(
        cls,
        *,
        walk_forward_run_digest: str,
        gate_evidence_digest: str,
        execution_sensitivity_digest: str,
        dataset_id: str,
        selected_configuration: str,
        candidate_config_digest: str,
        seeds: tuple[int, ...],
        git_commit: str,
        dependency_digest: str,
        resume_checkpoint_digests: tuple[tuple[int, str], ...],
    ) -> SelectionProposal:
        resolved_seeds = _seeds(seeds)
        resolved_resume = _resume_digests(resume_checkpoint_digests)
        payload = {
            "candidate_config_digest": candidate_config_digest,
            "dataset_id": dataset_id,
            "dependency_digest": dependency_digest,
            "execution_sensitivity_digest": execution_sensitivity_digest,
            "git_commit": git_commit,
            "resume_checkpoint_digests": resolved_resume,
            "schema_version": SELECTION_PROPOSAL_SCHEMA,
            "seeds": resolved_seeds,
            "selected_configuration": selected_configuration,
            "walk_forward_run_digest": walk_forward_run_digest,
            "gate_evidence_digest": gate_evidence_digest,
        }
        return cls(
            digest=content_digest(payload),
            walk_forward_run_digest=walk_forward_run_digest,
            gate_evidence_digest=gate_evidence_digest,
            execution_sensitivity_digest=execution_sensitivity_digest,
            dataset_id=dataset_id,
            selected_configuration=selected_configuration,
            candidate_config_digest=candidate_config_digest,
            seeds=resolved_seeds,
            git_commit=git_commit,
            dependency_digest=dependency_digest,
            resume_checkpoint_digests=resolved_resume,
            schema_version=SELECTION_PROPOSAL_SCHEMA,
        )

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> SelectionProposal:
        try:
            return cls(
                digest=_string(raw, "digest"),
                walk_forward_run_digest=_string(raw, "walk_forward_run_digest"),
                gate_evidence_digest=_string(raw, "gate_evidence_digest"),
                execution_sensitivity_digest=_string(
                    raw,
                    "execution_sensitivity_digest",
                ),
                dataset_id=_string(raw, "dataset_id"),
                selected_configuration=_string(raw, "selected_configuration"),
                candidate_config_digest=_string(raw, "candidate_config_digest"),
                seeds=_strict_seeds(raw["seeds"]),
                git_commit=_string(raw, "git_commit"),
                dependency_digest=_string(raw, "dependency_digest"),
                resume_checkpoint_digests=_strict_resume(
                    raw["resume_checkpoint_digests"]
                ),
                schema_version=_string(raw, "schema_version"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("selection proposal is invalid") from error


@dataclass(frozen=True, slots=True)
class SelectionAuthorization:
    """External approver signature over one exact selection proposal."""

    proposal_digest: str
    approver: str
    approved_at: datetime
    expires_at: datetime
    key_id: str
    signature: str
    schema_version: str = SELECTION_AUTHORIZATION_SCHEMA

    def __post_init__(self) -> None:
        require_sha256(self.proposal_digest, field="proposal_digest")
        object.__setattr__(
            self, "approver", require_non_empty(self.approver, field="approver")
        )
        approved = _utc(self.approved_at, field="approved_at")
        expires = _utc(self.expires_at, field="expires_at")
        if expires <= approved:
            raise ValueError("selection authorization must expire after approval")
        object.__setattr__(self, "approved_at", approved)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(
            self, "key_id", require_non_empty(self.key_id, field="key_id")
        )
        if not isinstance(self.signature, str) or not self.signature:
            raise ValueError("selection authorization signature must be non-empty")
        if self.schema_version != SELECTION_AUTHORIZATION_SCHEMA:
            raise ValueError("unsupported selection authorization schema")

    @property
    def authorization_digest(self) -> str:
        return content_digest(self.to_mapping())

    def signed_payload(self) -> dict[str, object]:
        return {
            "approved_at": self.approved_at,
            "approver": self.approver,
            "expires_at": self.expires_at,
            "proposal_digest": self.proposal_digest,
            "schema_version": self.schema_version,
        }

    def envelope(self) -> SignedEvidenceEnvelope:
        return SignedEvidenceEnvelope(
            key_id=self.key_id,
            purpose=_SELECTION_PURPOSE,
            payload_digest=content_digest(self.signed_payload()),
            signed_at=self.approved_at,
            signature=self.signature,
        )

    def verify(
        self,
        proposal: SelectionProposal,
        *,
        trusted_keys: Mapping[str, PublicVerificationKey],
        trusted_at: datetime,
    ) -> None:
        now = _utc(trusted_at, field="trusted_at")
        if proposal.digest != self.proposal_digest:
            raise ValueError("selection authorization proposal digest mismatch")
        if now < self.approved_at:
            raise ValueError("selection authorization approval is from the future")
        if now > self.expires_at:
            raise ValueError("selection authorization has expired")
        verify_signed_payload(
            self.signed_payload(),
            self.envelope(),
            trusted_keys=trusted_keys,
            trusted_at=now,
            required_purpose=_SELECTION_PURPOSE,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "approved_at": self.approved_at,
            "approver": self.approver,
            "expires_at": self.expires_at,
            "key_id": self.key_id,
            "proposal_digest": self.proposal_digest,
            "schema_version": self.schema_version,
            "signature": self.signature,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> SelectionAuthorization:
        try:
            return cls(
                proposal_digest=_string(raw, "proposal_digest"),
                approver=_string(raw, "approver"),
                approved_at=_datetime(raw, "approved_at"),
                expires_at=_datetime(raw, "expires_at"),
                key_id=_string(raw, "key_id"),
                signature=_string(raw, "signature"),
                schema_version=_string(raw, "schema_version"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("selection authorization is invalid") from error


def _string(raw: Mapping[str, object], field: str) -> str:
    value = raw[field]
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _datetime(raw: Mapping[str, object], field: str) -> datetime:
    value = _string(raw, field)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 datetime") from error


def _strict_seeds(raw: object) -> tuple[int, ...]:
    if not isinstance(raw, list):
        raise ValueError("selection proposal seeds must be a list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
        raise ValueError("selection proposal seeds must contain integers")
    return tuple(raw)


def _strict_resume(raw: object) -> tuple[tuple[int, str], ...]:
    if not isinstance(raw, list):
        raise ValueError("resume checkpoint digests must be a list")
    result: list[tuple[int, str]] = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("resume checkpoint digests must contain pairs")
        seed, digest = item
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(digest, str)
        ):
            raise ValueError("resume checkpoint digest pair is invalid")
        result.append((seed, digest))
    return tuple(result)


def _write_once(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload)
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to overwrite immutable evidence: {path}")
        return path
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_selection_proposal(path: str | Path, proposal: SelectionProposal) -> Path:
    return _write_once(Path(path), proposal.to_mapping())


def load_selection_proposal(path: str | Path) -> SelectionProposal:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("selection proposal must be an object")
    return SelectionProposal.from_mapping(raw)


def write_selection_authorization(
    path: str | Path,
    authorization: SelectionAuthorization,
) -> Path:
    return _write_once(Path(path), authorization.to_mapping())


def load_selection_authorization(path: str | Path) -> SelectionAuthorization:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("selection authorization must be an object")
    return SelectionAuthorization.from_mapping(raw)


__all__ = [
    "SELECTION_AUTHORIZATION_SCHEMA",
    "SELECTION_PROPOSAL_SCHEMA",
    "SelectionAuthorization",
    "SelectionProposal",
    "load_selection_authorization",
    "load_selection_proposal",
    "write_selection_authorization",
    "write_selection_proposal",
]
