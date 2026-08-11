"""State-machine adapter for the universal U6 full-research integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trade_rl.domain.common import require_sha256
from trade_rl.workflows.full_research_state import (
    FullResearchStatus,
    ResearchPhase,
    ResearchPhaseOutcome,
)
from trade_rl.workflows.universal_research import (
    UniversalFullResearchPlan,
    UniversalResearchManifest,
    validate_full_research_completion,
    validate_full_research_start_inputs,
)


@dataclass(frozen=True, slots=True)
class UniversalResearchStages:
    """Bind one immutable U3-U5 evidence set to the maintained U6 state machine."""

    manifest: UniversalResearchManifest
    plan: UniversalFullResearchPlan
    selection_authorization_digest: str | None
    fresh_confirmation_digest: str | None

    def __post_init__(self) -> None:
        validate_full_research_start_inputs(self.manifest)
        if self.manifest.architecture_name != self.plan.selected_architecture:
            raise ValueError(
                "U6 selected architecture differs from the research manifest"
            )
        if not self.plan.zero_shot_gate_passed or not self.manifest.zero_shot_gate_passed:
            raise ValueError("U6 selected architecture requires a passed zero-shot gate")
        for field_name, value in (
            ("selection_authorization_digest", self.selection_authorization_digest),
            ("fresh_confirmation_digest", self.fresh_confirmation_digest),
        ):
            if value is not None:
                require_sha256(value, field=field_name)

    def _summary(self) -> dict[str, object]:
        return {
            "production_status": "NO-GO",
            "manifest_digest": self.manifest.manifest_digest,
            "selected_architecture": self.plan.selected_architecture.value,
            "algorithms": tuple(algorithm.value for algorithm in self.plan.algorithms),
            "zero_shot_gate_digest": self.manifest.zero_shot_gate_digest,
            "architecture_evidence_digest": self.manifest.architecture_evidence_digest,
            "required_pair_count": len(self.manifest.required_pairs),
            "completed_pair_count": len(self.manifest.completed_pairs),
            "software_identity": self.manifest.software_identity,
        }

    def run(self, phase: ResearchPhase, work_root: Path) -> ResearchPhaseOutcome:
        del work_root
        validate_full_research_start_inputs(self.manifest)
        summary = self._summary()

        if phase is ResearchPhase.DEVELOP:
            return ResearchPhaseOutcome(
                status=FullResearchStatus.AWAITING_SELECTION_AUTHORIZATION,
                summary=summary,
            )

        if self.selection_authorization_digest is None:
            raise ValueError("U6 train-selected requires selection authorization")
        require_sha256(
            self.selection_authorization_digest,
            field="selection_authorization_digest",
        )
        validate_full_research_completion(self.manifest)
        summary["selection_authorization_digest"] = self.selection_authorization_digest

        if phase is ResearchPhase.TRAIN_SELECTED:
            return ResearchPhaseOutcome(
                status=FullResearchStatus.AWAITING_FRESH_CONFIRMATION,
                summary=summary,
            )

        if phase is not ResearchPhase.FINALIZE:
            raise ValueError(f"unsupported U6 research phase: {phase}")
        if self.fresh_confirmation_digest is None:
            raise ValueError("U6 finalize requires fresh confirmation")
        require_sha256(
            self.fresh_confirmation_digest,
            field="fresh_confirmation_digest",
        )
        summary["fresh_confirmation_digest"] = self.fresh_confirmation_digest
        return ResearchPhaseOutcome(
            status=FullResearchStatus.AWAITING_RELEASE_APPROVAL,
            summary=summary,
        )


__all__ = ["UniversalResearchStages"]
