"""Evidence-backed deployment gate for Shadow -> Canary -> Production.

Canary and Production promotion require immutable evidence downloaded from a trusted
prior run. Evidence is bound to the exact ServingBundle manifest; the gate revalidates
the complete bundle rather than trusting a standalone model file or self-reported flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mars_lite.serving.bundle import load_bundle

DeploymentStage = Literal["shadow", "canary", "production"]
_HEX_40 = re.compile(r"[a-fA-F0-9]{40}")
_HEX_64 = re.compile(r"[a-fA-F0-9]{64}")
_MODEL_VERSION = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,49}")
_APPROVAL_TICKET = re.compile(r"PROD-\d+")
_APPROVER = re.compile(r"[a-zA-Z0-9_.@-]+")
_SERVING_MANIFEST_PATH = "serving_candidate/manifest.json"
_SHADOW_MIN_SHARPE_DIFF = -0.3
_SHADOW_MAX_DRAWDOWN = 0.20
_CANARY_MAX_CAPITAL_USD = 10_000.0
_CANARY_MIN_DURATION_DAYS = 7
_CANARY_MAX_LOSS_PCT = 0.05
_CANARY_MAX_SLIPPAGE_BPS = 15.0
_DRIFT_MAX_PSI = 0.25
_DRIFT_MIN_KS_P = 0.01


@dataclass(frozen=True)
class CandidateArtifact:
    model_version: str
    git_commit: str
    artifact_path: str
    artifact_sha256: str
    shadow_report_sha256: str
    drift_report_sha256: str
    incident_report_sha256: str
    canary_report_sha256: str | None = None
    config_sha256: str | None = None
    data_sha256: str | None = None


@dataclass(frozen=True)
class ShadowEvidenceReport:
    run_id: str
    model_version: str
    git_commit: str
    artifact_sha256: str
    oos_sharpe: float
    baseline_sharpe: float
    max_drawdown: float

    def is_passed(self) -> tuple[bool, str]:
        metrics = (self.oos_sharpe, self.baseline_sharpe, self.max_drawdown)
        if not all(math.isfinite(float(value)) for value in metrics):
            return False, "shadow report metrics must be finite"
        if not 0.0 <= float(self.max_drawdown) <= 1.0:
            return False, "shadow drawdown must be between 0 and 1"
        difference = float(self.oos_sharpe) - float(self.baseline_sharpe)
        if difference < _SHADOW_MIN_SHARPE_DIFF:
            return False, (
                f"shadow sharpe difference {difference:.2f} below threshold "
                f"{_SHADOW_MIN_SHARPE_DIFF}"
            )
        if float(self.max_drawdown) > _SHADOW_MAX_DRAWDOWN:
            return False, (
                f"shadow drawdown {self.max_drawdown:.1%} exceeds allowed "
                f"{_SHADOW_MAX_DRAWDOWN:.1%}"
            )
        return True, "shadow report passed"


@dataclass(frozen=True)
class CanaryEvidenceReport:
    run_id: str
    parent_shadow_run_id: str
    model_version: str
    git_commit: str
    artifact_sha256: str
    capital_cap_usd: float
    duration_days: int
    max_loss_pct: float
    mean_slippage_bps: float

    def is_passed(self) -> tuple[bool, str]:
        metrics = (self.capital_cap_usd, self.max_loss_pct, self.mean_slippage_bps)
        if not all(math.isfinite(float(value)) for value in metrics):
            return False, "canary report metrics must be finite"
        if isinstance(self.duration_days, bool) or not isinstance(
            self.duration_days, int
        ):
            return False, "canary duration_days must be an integer"
        if float(self.capital_cap_usd) < 0.0:
            return False, "canary capital must be non-negative"
        if not 0.0 <= float(self.max_loss_pct) <= 1.0:
            return False, "canary max loss must be between 0 and 1"
        if float(self.mean_slippage_bps) < 0.0:
            return False, "canary slippage must be non-negative"
        if float(self.capital_cap_usd) > _CANARY_MAX_CAPITAL_USD:
            return False, (
                f"canary capital {self.capital_cap_usd} exceeds cap "
                f"{_CANARY_MAX_CAPITAL_USD}"
            )
        if self.duration_days < _CANARY_MIN_DURATION_DAYS:
            return False, (
                f"canary duration {self.duration_days} days below minimum "
                f"{_CANARY_MIN_DURATION_DAYS} days"
            )
        if float(self.max_loss_pct) > _CANARY_MAX_LOSS_PCT:
            return False, (
                f"canary max loss {self.max_loss_pct:.1%} exceeds allowed "
                f"{_CANARY_MAX_LOSS_PCT:.1%}"
            )
        if float(self.mean_slippage_bps) > _CANARY_MAX_SLIPPAGE_BPS:
            return False, (
                f"canary slippage {self.mean_slippage_bps:.1f}bps exceeds allowed "
                f"{_CANARY_MAX_SLIPPAGE_BPS:.1f}bps"
            )
        return True, "canary report passed"


@dataclass(frozen=True)
class DriftEvidenceReport:
    report_id: str
    model_version: str
    git_commit: str
    artifact_sha256: str
    psi_score: float
    ks_p_value: float

    def is_passed(self) -> tuple[bool, str]:
        metrics = (self.psi_score, self.ks_p_value)
        if not all(math.isfinite(float(value)) for value in metrics):
            return False, "drift report metrics must be finite"
        if float(self.psi_score) < 0.0:
            return False, "drift PSI must be non-negative"
        if not 0.0 <= float(self.ks_p_value) <= 1.0:
            return False, "drift KS p-value must be between 0 and 1"
        if float(self.psi_score) > _DRIFT_MAX_PSI:
            return False, (
                f"drift PSI {self.psi_score:.3f} exceeds maximum {_DRIFT_MAX_PSI:.3f}"
            )
        if float(self.ks_p_value) < _DRIFT_MIN_KS_P:
            return False, (
                f"drift KS p-value {self.ks_p_value:.4f} below minimum "
                f"{_DRIFT_MIN_KS_P:.4f}"
            )
        return True, "drift report passed"


@dataclass(frozen=True)
class IncidentEvidenceReport:
    report_id: str
    model_version: str
    git_commit: str
    artifact_sha256: str
    active_incidents: bool

    def is_passed(self) -> tuple[bool, str]:
        if type(self.active_incidents) is not bool:
            return False, "active_incidents must be a boolean"
        if self.active_incidents:
            return False, "deployment blocked due to active incidents"
        return True, "incident report passed"


@dataclass(frozen=True)
class DeploymentEvidence:
    stage: DeploymentStage
    artifact_root: Path | None = None
    candidate: CandidateArtifact | None = None
    shadow_report: ShadowEvidenceReport | None = None
    drift_report: DriftEvidenceReport | None = None
    incident_report: IncidentEvidenceReport | None = None
    canary_report: CanaryEvidenceReport | None = None
    approval_ticket: str | None = None
    environment_approver: str | None = None


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reason": self.reason}


class DeploymentGate:
    """Validate immutable evidence and enforce ordered promotion."""

    def evaluate(self, evidence: DeploymentEvidence) -> GateDecision:
        if evidence.stage == "shadow":
            return GateDecision(True, "shadow deployment is the first stage")
        if evidence.stage not in ("canary", "production"):
            return GateDecision(False, f"unknown deployment stage: {evidence.stage}")
        if evidence.artifact_root is None or evidence.candidate is None:
            return GateDecision(
                False, "evidence bundle and candidate artifact are required"
            )

        candidate_error = self._validate_candidate(
            evidence.candidate, evidence.artifact_root
        )
        if candidate_error:
            return GateDecision(False, candidate_error)

        required = {
            "shadow": evidence.shadow_report,
            "drift": evidence.drift_report,
            "incident": evidence.incident_report,
        }
        missing = [name for name, report in required.items() if report is None]
        if missing:
            return GateDecision(
                False, f"missing required evidence reports: {', '.join(missing)}"
            )

        assert evidence.shadow_report is not None
        assert evidence.drift_report is not None
        assert evidence.incident_report is not None
        for label, report in required.items():
            assert report is not None
            identity_error = self._validate_identity(evidence.candidate, report, label)
            if identity_error:
                return GateDecision(False, identity_error)

        for report in (
            evidence.shadow_report,
            evidence.drift_report,
            evidence.incident_report,
        ):
            passed, reason = report.is_passed()
            if not passed:
                return GateDecision(False, f"{evidence.stage} blocked: {reason}")

        if evidence.stage == "canary":
            return GateDecision(True, "verified shadow evidence accepted")
        if evidence.canary_report is None:
            return GateDecision(False, "production requires verified canary evidence")
        if evidence.candidate.canary_report_sha256 is None:
            return GateDecision(False, "candidate manifest lacks canary report digest")

        identity_error = self._validate_identity(
            evidence.candidate, evidence.canary_report, "canary"
        )
        if identity_error:
            return GateDecision(False, identity_error)
        if evidence.canary_report.parent_shadow_run_id != evidence.shadow_report.run_id:
            return GateDecision(
                False, "canary report does not reference the verified shadow run"
            )
        passed, reason = evidence.canary_report.is_passed()
        if not passed:
            return GateDecision(False, f"production blocked: {reason}")

        if not evidence.approval_ticket or not _APPROVAL_TICKET.fullmatch(
            evidence.approval_ticket
        ):
            return GateDecision(
                False, "production requires approval ticket with format PROD-<digits>"
            )
        if len(evidence.approval_ticket) > 20:
            return GateDecision(
                False, "approval ticket exceeds maximum length of 20 characters"
            )
        if not evidence.environment_approver or not _APPROVER.fullmatch(
            evidence.environment_approver
        ):
            return GateDecision(False, "production requires valid environment approver")
        return GateDecision(True, "verified production evidence accepted")

    @staticmethod
    def _validate_candidate(
        candidate: CandidateArtifact, artifact_root: Path
    ) -> str | None:
        if not _MODEL_VERSION.fullmatch(candidate.model_version):
            return "candidate requires valid model version"
        if not _HEX_40.fullmatch(candidate.git_commit):
            return "candidate requires valid 40-character SHA-1 git commit hash"

        digest_fields = [
            candidate.artifact_sha256,
            candidate.shadow_report_sha256,
            candidate.drift_report_sha256,
            candidate.incident_report_sha256,
        ]
        if candidate.canary_report_sha256 is not None:
            digest_fields.append(candidate.canary_report_sha256)
        if candidate.config_sha256 is not None:
            digest_fields.append(candidate.config_sha256)
        if candidate.data_sha256 is not None:
            digest_fields.append(candidate.data_sha256)
        if any(not _is_sha256(value) for value in digest_fields):
            return "candidate manifest contains invalid SHA-256 digest"

        if candidate.artifact_path != _SERVING_MANIFEST_PATH:
            return (
                "candidate artifact_path must reference "
                f"{_SERVING_MANIFEST_PATH}"
            )
        try:
            artifact_path = _resolve_inside(artifact_root, candidate.artifact_path)
        except (TypeError, ValueError):
            return "serving manifest path escapes evidence bundle"
        if not artifact_path.is_file():
            return f"serving manifest not found: {candidate.artifact_path}"
        if sha256_file(artifact_path) != candidate.artifact_sha256.lower():
            return "serving manifest SHA-256 mismatch"

        try:
            bundle = load_bundle(artifact_path.parent)
        except (OSError, TypeError, ValueError) as exc:
            return f"serving bundle validation failed: {exc}"
        if bundle.version != candidate.model_version:
            return "serving bundle model version does not match candidate"
        if bundle.manifest.git_sha != candidate.git_commit:
            return "serving bundle git commit does not match candidate"
        return None

    @staticmethod
    def _validate_identity(
        candidate: CandidateArtifact, report: Any, label: str
    ) -> str | None:
        if report.model_version != candidate.model_version:
            return f"{label} report model version does not match candidate"
        if report.git_commit != candidate.git_commit:
            return f"{label} report git commit does not match candidate"
        if report.artifact_sha256.lower() != candidate.artifact_sha256.lower():
            return f"{label} report artifact digest does not match candidate"
        return None


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_evidence_bundle(
    bundle_dir: str | Path,
    stage: DeploymentStage,
    approval_ticket: str | None = None,
    environment_approver: str | None = None,
) -> DeploymentEvidence:
    root = Path(bundle_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"evidence bundle does not exist: {root}")

    candidate_path = root / "candidate.json"
    shadow_path = root / "shadow.json"
    drift_path = root / "drift.json"
    incident_path = root / "incident.json"
    canary_path = root / "canary.json"

    candidate = _construct(CandidateArtifact, _read_json(candidate_path), "candidate")
    _verify_report_digest(shadow_path, candidate.shadow_report_sha256, "shadow")
    _verify_report_digest(drift_path, candidate.drift_report_sha256, "drift")
    _verify_report_digest(incident_path, candidate.incident_report_sha256, "incident")

    canary_report = None
    if stage == "production":
        if candidate.canary_report_sha256 is None:
            raise ValueError("production candidate manifest lacks canary report digest")
        _verify_report_digest(canary_path, candidate.canary_report_sha256, "canary")
        canary_report = _construct(
            CanaryEvidenceReport, _read_json(canary_path), "canary"
        )

    return DeploymentEvidence(
        stage=stage,
        artifact_root=root,
        candidate=candidate,
        shadow_report=_construct(
            ShadowEvidenceReport, _read_json(shadow_path), "shadow"
        ),
        drift_report=_construct(DriftEvidenceReport, _read_json(drift_path), "drift"),
        incident_report=_construct(
            IncidentEvidenceReport, _read_json(incident_path), "incident"
        ),
        canary_report=canary_report,
        approval_ticket=approval_ticket,
        environment_approver=environment_approver,
    )


def _construct(cls: type[Any], payload: dict[str, Any], label: str) -> Any:
    try:
        return cls(**payload)
    except TypeError as exc:
        raise ValueError(f"invalid {label} evidence schema: {exc}") from exc


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _HEX_64.fullmatch(value) is not None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required evidence file not found: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"evidence file must contain a JSON object: {path.name}")
    return payload


def _verify_report_digest(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"required evidence file not found: {path.name}")
    if not _is_sha256(expected):
        raise ValueError(f"{label} report digest is invalid")
    if sha256_file(path) != expected.lower():
        raise ValueError(f"{label} report SHA-256 mismatch")


def _resolve_inside(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("relative path is required")
    root_resolved = root.resolve()
    candidate = (root_resolved / relative_path).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("path traversal") from exc
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate deployment evidence bundle")
    parser.add_argument(
        "--stage", required=True, choices=["shadow", "canary", "production"]
    )
    parser.add_argument("--bundle-dir")
    parser.add_argument("--approval-ticket")
    parser.add_argument("--environment-approver")
    args = parser.parse_args(argv)

    try:
        if args.stage == "shadow":
            evidence = DeploymentEvidence(stage="shadow")
        else:
            if not args.bundle_dir:
                raise ValueError("canary and production require --bundle-dir")
            evidence = load_evidence_bundle(
                args.bundle_dir,
                args.stage,
                approval_ticket=args.approval_ticket,
                environment_approver=args.environment_approver,
            )
        decision = DeploymentGate().evaluate(evidence)
    except (OSError, TypeError, ValueError) as exc:
        decision = GateDecision(False, str(exc))

    print(json.dumps(decision.to_dict(), ensure_ascii=False))
    return 0 if decision.allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
