"""Deployment stage gate for Shadow -> Canary -> Production promotion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

DeploymentStage = Literal["shadow", "canary", "production"]


@dataclass(frozen=True)
class ShadowEvidenceReport:
    run_id: str
    oos_sharpe: float
    baseline_sharpe: float
    max_drawdown: float
    min_sharpe_diff: float = -0.3
    max_allowed_drawdown: float = 0.20

    def is_passed(self) -> tuple[bool, str]:
        diff = self.oos_sharpe - self.baseline_sharpe
        if diff < self.min_sharpe_diff:
            return False, f"shadow sharpe difference {diff:.2f} below threshold {self.min_sharpe_diff}"
        if self.max_drawdown > self.max_allowed_drawdown:
            return False, f"shadow drawdown {self.max_drawdown:.1%} exceeds allowed {self.max_allowed_drawdown:.1%}"
        return True, "shadow report passed"


@dataclass(frozen=True)
class CanaryEvidenceReport:
    run_id: str
    capital_cap_usd: float
    duration_days: int
    max_loss_pct: float
    mean_slippage_bps: float
    max_allowed_capital_usd: float = 10000.0
    min_duration_days: int = 7
    max_allowed_loss_pct: float = 0.05
    max_allowed_slippage_bps: float = 15.0

    def is_passed(self) -> tuple[bool, str]:
        if self.capital_cap_usd > self.max_allowed_capital_usd:
            return False, f"canary capital {self.capital_cap_usd} exceeds cap {self.max_allowed_capital_usd}"
        if self.duration_days < self.min_duration_days:
            return False, f"canary duration {self.duration_days} days below minimum {self.min_duration_days} days"
        if self.max_loss_pct > self.max_allowed_loss_pct:
            return False, f"canary max loss {self.max_loss_pct:.1%} exceeds allowed {self.max_allowed_loss_pct:.1%}"
        if self.mean_slippage_bps > self.max_allowed_slippage_bps:
            return False, f"canary slippage {self.mean_slippage_bps:.1f}bps exceeds allowed {self.max_allowed_slippage_bps:.1f}bps"
        return True, "canary report passed"


@dataclass(frozen=True)
class DriftEvidenceReport:
    report_id: str
    psi_score: float
    ks_p_value: float
    signature_sha256: str
    max_allowed_psi: float = 0.25
    min_allowed_ks_p: float = 0.01

    def is_passed(self) -> tuple[bool, str]:
        if self.psi_score > self.max_allowed_psi:
            return False, f"drift PSI {self.psi_score:.3f} exceeds maximum {self.max_allowed_psi:.3f}"
        if self.ks_p_value < self.min_allowed_ks_p:
            return False, f"drift KS p-value {self.ks_p_value:.4f} below minimum {self.min_allowed_ks_p:.4f}"
        if not re.fullmatch(r"[a-fA-F0-9]{64}", self.signature_sha256):
            return False, "drift report requires valid 64-character SHA-256 signature"
        return True, "drift report passed"


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
    # -- 証拠連動型拡張フィールド --
    model_artifact_sha256: str | None = None
    shadow_report: ShadowEvidenceReport | None = None
    canary_report: CanaryEvidenceReport | None = None
    drift_report: DriftEvidenceReport | None = None
    environment_approver: str | None = None


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

        if evidence.model_artifact_sha256 is not None:
            if not re.fullmatch(r"[a-fA-F0-9]{64}", evidence.model_artifact_sha256):
                return GateDecision(
                    False, "model artifact sha256 must be a valid 64-character hex string"
                )

        if evidence.stage == "shadow":
            return GateDecision(True, "shadow deployment is the first stage")

        # -- Canary 評価 --
        if evidence.stage == "canary":
            # Shadow 証拠検証 (レポートがある場合はレポート評価を優先)
            if evidence.shadow_report is not None:
                passed, reason = evidence.shadow_report.is_passed()
                if not passed:
                    return GateDecision(False, f"canary blocked: {reason}")
            elif not evidence.shadow_passed:
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

            # Drift レポート検証
            if evidence.drift_report is not None:
                passed, reason = evidence.drift_report.is_passed()
                if not passed:
                    return GateDecision(False, f"canary blocked: {reason}")
            elif not evidence.drift_report_passed:
                return GateDecision(False, "canary requires passing drift report")

            return GateDecision(True, "shadow evidence accepted")

        # -- Production 評価 --
        if evidence.stage == "production":
            # Shadow 証拠検証
            if evidence.shadow_report is not None:
                passed, reason = evidence.shadow_report.is_passed()
                if not passed:
                    return GateDecision(False, f"production blocked: {reason}")
            elif not evidence.shadow_passed:
                return GateDecision(False, "production requires shadow evidence")

            # Canary 証拠検証
            if evidence.canary_report is not None:
                passed, reason = evidence.canary_report.is_passed()
                if not passed:
                    return GateDecision(False, f"production blocked: {reason}")
            elif not evidence.canary_passed:
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

            # Drift レポート検証
            if evidence.drift_report is not None:
                passed, reason = evidence.drift_report.is_passed()
                if not passed:
                    return GateDecision(False, f"production blocked: {reason}")
            elif not evidence.drift_report_passed:
                return GateDecision(False, "production requires passing drift report")

            if not evidence.approval_ticket or not re.fullmatch(
                r"PROD-\d+", evidence.approval_ticket
            ):
                return GateDecision(
                    False,
                    "production requires approval ticket with format PROD-<digits>",
                )
            if evidence.environment_approver is not None and not evidence.environment_approver.strip():
                return GateDecision(False, "production requires valid environment approver")

            return GateDecision(True, "production evidence accepted")

        return GateDecision(False, f"unknown deployment stage: {evidence.stage}")

