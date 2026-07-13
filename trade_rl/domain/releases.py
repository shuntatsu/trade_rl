"""Production release identity and fail-closed construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trade_rl.domain.common import (
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
    """Immutable deployment identity created only after mandatory gates pass."""

    version: str
    git_commit: str
    dataset_id: str
    signal_digest: str
    selection_evaluation_digest: str
    selected_policy_digest: str | None
    bundle_digest: str
    created_at: datetime
    schema_version: str = "release_manifest_v1"

    def __post_init__(self) -> None:
        require_non_empty(self.version, field="version")
        require_git_sha(self.git_commit)
        require_sha256(self.dataset_id, field="dataset_id")
        require_sha256(self.signal_digest, field="signal_digest")
        require_sha256(
            self.selection_evaluation_digest,
            field="selection_evaluation_digest",
        )
        if self.selected_policy_digest is not None:
            require_sha256(
                self.selected_policy_digest,
                field="selected_policy_digest",
            )
        require_sha256(self.bundle_digest, field="bundle_digest")
        require_aware_datetime(self.created_at, field="created_at")
        require_non_empty(self.schema_version, field="schema_version")

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
        """Build a release only when identities agree and mandatory gates pass."""

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
        return cls(
            version=version,
            git_commit=git_commit,
            dataset_id=dataset.dataset_id,
            signal_digest=signal.digest,
            selection_evaluation_digest=selection.evaluation_digest,
            selected_policy_digest=selection.selected_policy_digest,
            bundle_digest=bundle_digest,
            created_at=created_at,
        )
