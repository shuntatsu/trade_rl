"""Deployment stage gate for Shadow -> Canary -> Production promotion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DeploymentStage = Literal["shadow", "canary", "production"]


@dataclass(frozen=True)
class DeploymentEvidence:
    stage: DeploymentStage
    shadow_passed: bool = False
    canary_passed: bool = False
    approval_ticket: str | None = None


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str


class DeploymentGate:
    """Enforce ordered promotion and human approval evidence."""

    def evaluate(self, evidence: DeploymentEvidence) -> GateDecision:
        if evidence.stage == "shadow":
            return GateDecision(True, "shadow deployment is the first stage")
        if evidence.stage == "canary":
            if not evidence.shadow_passed:
                return GateDecision(False, "canary requires passing shadow evidence")
            return GateDecision(True, "shadow evidence accepted")
        if evidence.stage == "production":
            if not evidence.shadow_passed:
                return GateDecision(False, "production requires shadow evidence")
            if not evidence.canary_passed:
                return GateDecision(False, "production requires canary evidence")
            if not evidence.approval_ticket:
                return GateDecision(False, "production requires approval ticket")
            return GateDecision(True, "production evidence accepted")
        return GateDecision(False, f"unknown deployment stage: {evidence.stage}")
