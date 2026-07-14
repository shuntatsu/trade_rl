"""Production release attestation identity and fail-closed construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trade_rl.domain.common import (
    domain_content_digest,
    require_aware_datetime,
    require_git_sha,
    require_non_empty,
    require_sha256,
)
from trade_rl.domain.datasets import DatasetManifest
from trade_rl.domain.evaluation import GateDecision
from trade_rl.domain.selection import SelectionDecision
from trade_rl.domain.signals import SignalArtifactManifest


@dataclass(frozen=True, slots=True)
class ReleaseManifest:
    """Immutable attestation created after a candidate bundle and gates exist."""

    version: str
    git_commit: str
    dataset_id: str
    signal_digest: str
    selection_digest: str
    selection_evaluation_digest: str
    gate_evaluation_digest: str
    selected_policy_digest: str | None
    bundle_digest: str
    created_at: datetime
    schema_version: str = "release_manifest_v3"

    def __post_init__(self) -> None:
        require_non_empty(self.version, field="version")
        require_git_sha(self.git_commit)
        require_sha256(self.dataset_id, field="dataset_id")
        require_sha256(self.signal_digest, field="signal_digest")
        require_sha256(self.selection_digest, field="selection_digest")
        require_sha256(
            self.selection_evaluation_digest,
            field="selection_evaluation_digest",
        )
        require_sha256(
            self.gate_evaluation_digest,
            field="gate_evaluation_digest",
        )
        if self.selected_policy_digest is not None:
            require_sha256(
                self.selected_policy_digest,
                field="selected_policy_digest",
            )
        require_sha256(self.bundle_digest, field="bundle_digest")
        require_aware_datetime(self.created_at, field="created_at")
        if self.schema_version != "release_manifest_v3":
            raise ValueError("unsupported release manifest schema")

    @property
    def digest(self) -> str:
        return domain_content_digest(self.digest_payload())

    def digest_payload(self) -> dict[str, object]:
        return {
            "bundle_digest": self.bundle_digest,
            "created_at": self.created_at,
            "dataset_id": self.dataset_id,
            "gate_evaluation_digest": self.gate_evaluation_digest,
            "git_commit": self.git_commit,
            "schema_version": self.schema_version,
            "selected_policy_digest": self.selected_policy_digest,
            "selection_digest": self.selection_digest,
            "selection_evaluation_digest": self.selection_evaluation_digest,
            "signal_digest": self.signal_digest,
            "version": self.version,
        }

    def validate_bundle_identity(
        self,
        *,
        bundle_digest: str,
        dataset_id: str,
        signal_digest: str,
        selection_digest: str,
        selected_policy_digest: str | None,
    ) -> None:
        comparisons = (
            (self.bundle_digest, bundle_digest, "bundle"),
            (self.dataset_id, dataset_id, "dataset"),
            (self.signal_digest, signal_digest, "signal"),
            (self.selection_digest, selection_digest, "selection"),
            (self.selected_policy_digest, selected_policy_digest, "policy"),
        )
        for attested, observed, label in comparisons:
            if attested != observed:
                raise ValueError(f"release attestation {label} identity mismatch")

    @classmethod
    def create(
        cls,
        *,
        version: str,
        git_commit: str,
        dataset: DatasetManifest,
        signal: SignalArtifactManifest,
        selection: SelectionDecision,
        gate: GateDecision,
        bundle_digest: str,
        created_at: datetime,
    ) -> ReleaseManifest:
        """Build a release only after bundle identity and mandatory gates exist."""

        if gate.failed_mandatory_checks:
            failed = ", ".join(check.name for check in gate.failed_mandatory_checks)
            raise ValueError(f"mandatory gate checks failed: {failed}")
        if not gate.passed:
            raise ValueError("mandatory gate decision did not pass")
        if dataset.dataset_id != signal.dataset_id:
            raise ValueError("dataset identity mismatch between dataset and signal")
        if dataset.dataset_id != selection.dataset_id:
            raise ValueError("dataset identity mismatch between dataset and selection")
        if signal.digest != selection.signal_digest:
            raise ValueError("signal digest mismatch between signal and selection")
        if gate.dataset_id != selection.dataset_id:
            raise ValueError("gate dataset identity mismatch")
        if gate.selected_policy_digest != selection.selected_policy_digest:
            raise ValueError("gate selected policy identity mismatch")
        return cls(
            version=version,
            git_commit=git_commit,
            dataset_id=dataset.dataset_id,
            signal_digest=signal.digest,
            selection_digest=selection.digest,
            selection_evaluation_digest=selection.evaluation_digest,
            gate_evaluation_digest=gate.evaluation_digest,
            selected_policy_digest=selection.selected_policy_digest,
            bundle_digest=bundle_digest,
            created_at=created_at,
        )
