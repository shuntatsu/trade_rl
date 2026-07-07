"""Deployment stage gate for Shadow -> Canary -> Production promotion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

DeploymentStage = Literal["shadow", "canary", "production"]


@dataclass(frozen=True)
class DeploymentEvidence:
    stage: DeploymentStage
    shadow_passed: bool = False
    canary_passed: bool = False
    approval_ticket: str | None = None
    model_version: str | None = None
    git_commit: str | None = None
    drift_report_passed: bool = False
    active_incidents: bool = False


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str


class DeploymentGate:
    """Enforce ordered promotion and human approval evidence."""

    def evaluate(self, evidence: DeploymentEvidence) -> GateDecision:
        if evidence.model_version is not None and len(evidence.model_version) > 50:
            return GateDecision(
                False, "model version exceeds maximum length of 50 characters"
            )
        if evidence.approval_ticket is not None and len(evidence.approval_ticket) > 20:
            return GateDecision(
                False, "approval ticket exceeds maximum length of 20 characters"
            )

        if evidence.active_incidents:
            return GateDecision(False, "deployment blocked due to active incidents")

        if evidence.stage == "shadow":
            return GateDecision(True, "shadow deployment is the first stage")
        if evidence.stage == "canary":
            if not evidence.shadow_passed:
                return GateDecision(False, "canary requires passing shadow evidence")
            if not evidence.model_version or not re.fullmatch(
                r"[a-zA-Z0-9_\-\.]+", evidence.model_version
            ):
                return GateDecision(False, "canary requires valid model version")
            if not evidence.git_commit or not re.fullmatch(
                r"[a-fA-F0-9]{40}", evidence.git_commit
            ):
                return GateDecision(
                    False, "canary requires valid 40-character SHA-1 git commit hash"
                )
            if not evidence.drift_report_passed:
                return GateDecision(False, "canary requires passing drift report")
            return GateDecision(True, "shadow evidence accepted")
        if evidence.stage == "production":
            if not evidence.shadow_passed:
                return GateDecision(False, "production requires shadow evidence")
            if not evidence.canary_passed:
                return GateDecision(False, "production requires canary evidence")
            if not evidence.model_version or not re.fullmatch(
                r"[a-zA-Z0-9_\-\.]+", evidence.model_version
            ):
                return GateDecision(False, "production requires valid model version")
            if not evidence.git_commit or not re.fullmatch(
                r"[a-fA-F0-9]{40}", evidence.git_commit
            ):
                return GateDecision(
                    False,
                    "production requires valid 40-character SHA-1 git commit hash",
                )
            if not evidence.drift_report_passed:
                return GateDecision(False, "production requires passing drift report")
            if not evidence.approval_ticket or not re.fullmatch(
                r"PROD-\d+", evidence.approval_ticket
            ):
                return GateDecision(
                    False,
                    "production requires approval ticket with format PROD-<digits>",
                )
            return GateDecision(True, "production evidence accepted")
        return GateDecision(False, f"unknown deployment stage: {evidence.stage}")
