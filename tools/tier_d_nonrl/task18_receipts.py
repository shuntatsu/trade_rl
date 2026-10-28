"""Task18 declarations and report-binding checks, not verification authority.

This module compares receipt fields with supplied report claims. It does not
authenticate report producers or verify Git, imports, arrays, ledgers, software
completion, or profitability. Diagnostic arithmetic must not be treated as
authoritative evidence or an adoption result.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Literal, TypeGuard

from tools.tier_d_nonrl.task18_recipe import (
    ArmSpec,
    SourceFilePin,
    Task18Protocol,
)

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
ArmAttemptStatus = Literal[
    "NOT_ATTEMPTED", "RUNNING", "INTERRUPTED", "FAILED", "COMPLETED"
]
ArmAttemptReason = Literal[
    "NOT_STARTED",
    "RUN_INTERRUPTED_BEFORE_START",
    "RUN_IN_PROGRESS",
    "PROCESS_INTERRUPTED",
    "PROCESS_FAILED",
]
_STATUS_REASONS: dict[str, frozenset[str]] = {
    "NOT_ATTEMPTED": frozenset({"NOT_STARTED", "RUN_INTERRUPTED_BEFORE_START"}),
    "RUNNING": frozenset({"RUN_IN_PROGRESS"}),
    "INTERRUPTED": frozenset({"PROCESS_INTERRUPTED"}),
    "FAILED": frozenset({"PROCESS_FAILED"}),
}
_RECEIPT_STATUSES = frozenset((*_STATUS_REASONS, "COMPLETED"))


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _HEX_64.fullmatch(value) is not None


def _finite_number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


@dataclass(frozen=True)
class SourceBinding:
    """Run-declared source pins; these alone do not verify source identity."""

    commit_sha: str
    files: tuple[SourceFilePin, ...]
    import_origin_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.commit_sha, str) or not _HEX_40.fullmatch(
            self.commit_sha
        ):
            raise ValueError(
                "source commit must be 40 lowercase hexadecimal characters"
            )
        if (
            not isinstance(self.files, tuple)
            or not self.files
            or any(not isinstance(pin, SourceFilePin) for pin in self.files)
        ):
            raise ValueError("source binding must contain immutable source file pins")
        paths = [pin.path for pin in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("source binding contains a duplicate source path")
        if self.import_origin_manifest_sha256 is not None and not _valid_sha256(
            self.import_origin_manifest_sha256
        ):
            raise ValueError("import-origin manifest digest must be lowercase SHA-256")

    @property
    def manifest_sha256(self) -> str:
        return _canonical_digest(
            {
                "commit_sha": self.commit_sha,
                "files": [
                    {"path": pin.path, "sha256": pin.sha256}
                    for pin in sorted(self.files, key=lambda item: item.path)
                ],
                "import_origin_manifest_sha256": self.import_origin_manifest_sha256,
            }
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "commit_sha": self.commit_sha,
            "files": [asdict(pin) for pin in self.files],
            "import_origin_manifest_sha256": self.import_origin_manifest_sha256,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True)
class ExternalSourceVerificationReport:
    """Fields carried from a future upstream source-verifier report.

    ``binding_sha256`` hashes supplied claims; it is not a signature or proof
    of producer authenticity. A trusted future producer must create this report
    after checking exact Git head, clean tree, complete source closure, and
    imported module origins.
    """

    commit_sha: str
    tracked_tree_clean: bool
    source_files: tuple[SourceFilePin, ...]
    import_origin_manifest_sha256: str
    imported_module_count: int
    verification_report_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.commit_sha, str) or not _HEX_40.fullmatch(
            self.commit_sha
        ):
            raise ValueError("verified source head must be a 40-character commit SHA")
        if not isinstance(self.tracked_tree_clean, bool):
            raise ValueError("tracked-tree status must be boolean")
        if (
            not isinstance(self.source_files, tuple)
            or not self.source_files
            or any(not isinstance(pin, SourceFilePin) for pin in self.source_files)
        ):
            raise ValueError("verified source snapshot needs immutable source pins")
        if len({pin.path for pin in self.source_files}) != len(self.source_files):
            raise ValueError("verified source snapshot has duplicate paths")
        if not _valid_sha256(self.import_origin_manifest_sha256):
            raise ValueError("import-origin manifest digest must be lowercase SHA-256")
        if (
            not isinstance(self.imported_module_count, int)
            or isinstance(self.imported_module_count, bool)
            or self.imported_module_count <= 0
        ):
            raise ValueError("imported module count must be a positive integer")
        if not _valid_sha256(self.verification_report_sha256):
            raise ValueError("source verifier report must have a SHA-256 digest")

    @property
    def binding_sha256(self) -> str:
        """Digest reported fields for equality checks, not authenticity."""
        return _canonical_digest(
            {
                "commit_sha": self.commit_sha,
                "tracked_tree_clean": self.tracked_tree_clean,
                "source_files": [
                    {"path": pin.path, "sha256": pin.sha256}
                    for pin in sorted(self.source_files, key=lambda item: item.path)
                ],
                "import_origin_manifest_sha256": self.import_origin_manifest_sha256,
                "imported_module_count": self.imported_module_count,
                "verification_report_sha256": self.verification_report_sha256,
            }
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "commit_sha": self.commit_sha,
            "tracked_tree_clean": self.tracked_tree_clean,
            "source_files": [asdict(pin) for pin in self.source_files],
            "import_origin_manifest_sha256": self.import_origin_manifest_sha256,
            "imported_module_count": self.imported_module_count,
            "verification_report_sha256": self.verification_report_sha256,
            "binding_sha256": self.binding_sha256,
        }


@dataclass(frozen=True)
class ExternalArmVerificationReport:
    """Claims carried from a future native-ledger verifier for one arm.

    The evaluator compares these claims with the arm record and recipe but
    cannot authenticate their producer. A trusted future producer must inspect
    the native result and ledger before creating this report.
    """

    run_id: str
    protocol_digest: str
    source_report_binding_sha256: str
    arm_id: str
    pair_id: str
    role: str
    configuration: str
    symbol: str
    scenario: str
    dataset_id: str
    native_ledger_sha256: str
    arm_result_sha256: str
    verification_report_sha256: str
    expected_advances: int
    observed_advances: int
    initial_equity_usdt: float
    terminal_equity_usdt: float
    max_drawdown: float
    terminal_settlement_complete: bool
    terminal_flat: bool
    pending_order_count: int
    cost_oracle_passed: bool

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "protocol_digest",
            "source_report_binding_sha256",
            "arm_id",
            "pair_id",
            "role",
            "configuration",
            "symbol",
            "scenario",
            "dataset_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"arm verifier evidence {name} must be nonempty")
        for name in (
            "native_ledger_sha256",
            "arm_result_sha256",
            "verification_report_sha256",
        ):
            if not _valid_sha256(getattr(self, name)):
                raise ValueError(f"arm verifier evidence {name} must be SHA-256")
        for name in ("expected_advances", "observed_advances", "pending_order_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"arm verifier evidence {name} must be nonnegative")
        for name in ("initial_equity_usdt", "terminal_equity_usdt", "max_drawdown"):
            if not _finite_number(getattr(self, name)):
                raise ValueError(f"arm verifier evidence {name} must be finite")
        if self.max_drawdown < 0:
            raise ValueError("arm verifier evidence max drawdown cannot be negative")
        for name in (
            "terminal_settlement_complete",
            "terminal_flat",
            "cost_oracle_passed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"arm verifier evidence {name} must be boolean")

    @property
    def binding_sha256(self) -> str:
        return _canonical_digest(asdict(self))

    def to_mapping(self) -> dict[str, object]:
        result = asdict(self)
        result["binding_sha256"] = self.binding_sha256
        return result


@dataclass(frozen=True)
class ArmAttemptProgress:
    """Optional facts observed before an arm attempt stopped or was captured.

    ``None`` means the runner did not observe that fact. In particular, an
    unknown count or equity must not be represented as zero.
    """

    native_advances_observed: int | None = None
    last_native_index: int | None = None
    last_observed_equity_usdt: float | None = None
    max_drawdown_observed: float | None = None

    def __post_init__(self) -> None:
        for name in ("native_advances_observed", "last_native_index"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        for name in ("last_observed_equity_usdt", "max_drawdown_observed"):
            value = getattr(self, name)
            if value is not None and not _finite_number(value):
                raise ValueError(f"{name} must be finite or None")
        if self.max_drawdown_observed is not None and self.max_drawdown_observed < 0:
            raise ValueError("observed maximum drawdown cannot be negative")


@dataclass(frozen=True)
class ArmReceipt:
    arm_id: str
    pair_id: str
    role: str
    configuration: str
    symbol: str
    scenario: str
    protocol_digest: str
    dataset_id: str
    manifest_sha256: str
    arrays_sha256: str
    source_binding_sha256: str
    status: ArmAttemptStatus
    initial_equity_usdt: float | None
    terminal_equity_usdt: float | None
    max_drawdown: float | None
    terminal_flat: bool | None
    pending_order_count: int | None
    native_ledger_sha256: str | None = None
    arm_result_sha256: str | None = None
    external_report: ExternalArmVerificationReport | None = None
    error_code: str | None = None
    reason_code: ArmAttemptReason | None = None
    progress: ArmAttemptProgress | None = None

    def to_mapping(self) -> dict[str, object]:
        result = asdict(self)
        result["external_report"] = (
            self.external_report.to_mapping()
            if self.external_report is not None
            else None
        )
        return result


@dataclass(frozen=True)
class RunReceipt:
    run_id: str
    protocol_digest: str
    historical_protocol_digest: str
    dataset_id: str
    manifest_sha256: str
    arrays_sha256: str
    source_binding: SourceBinding
    source_report: ExternalSourceVerificationReport | None
    runtime_completed: bool
    arms: tuple[ArmReceipt, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run receipt needs a nonempty run id")
        if not isinstance(self.source_binding, SourceBinding):
            raise ValueError("run receipt must contain a source binding")
        if self.source_report is not None and not isinstance(
            self.source_report, ExternalSourceVerificationReport
        ):
            raise ValueError("source report must use the external report schema")
        if not isinstance(self.runtime_completed, bool):
            raise ValueError("runtime completion status must be boolean")
        if not isinstance(self.arms, tuple) or any(
            not isinstance(arm, ArmReceipt) for arm in self.arms
        ):
            raise ValueError("run receipt arms must be an immutable tuple")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema": "task18_run_receipt_v3",
            "run_id": self.run_id,
            "protocol_digest": self.protocol_digest,
            "historical_protocol_digest": self.historical_protocol_digest,
            "dataset_id": self.dataset_id,
            "manifest_sha256": self.manifest_sha256,
            "arrays_sha256": self.arrays_sha256,
            "source_binding": self.source_binding.to_mapping(),
            "source_report": (
                self.source_report.to_mapping()
                if self.source_report is not None
                else None
            ),
            "runtime_completed": self.runtime_completed,
            "arms": [arm.to_mapping() for arm in self.arms],
        }


@dataclass(frozen=True)
class RunEvaluation:
    declaration_consistent: bool
    source_report_matches: bool
    arm_reports_match: bool
    roster_complete: bool
    required_scenarios_covered: bool
    declared_runtime_completed: bool
    declared_arm_statuses_complete: bool
    declared_metrics_well_formed: bool
    terminal_claims_pass_diagnostic: bool
    drawdown_claims_pass_diagnostic: bool
    diagnostic_pair_arithmetic_available: bool
    winner_status: str
    adoption_status: str
    issues: tuple[str, ...]


def _expected_arm_by_id(protocol: Task18Protocol) -> dict[str, ArmSpec]:
    return {arm.arm_id: arm for arm in protocol.arms}


def _same_source_files(
    left: tuple[SourceFilePin, ...], right: tuple[SourceFilePin, ...]
) -> bool:
    return tuple(sorted(left, key=lambda pin: pin.path)) == tuple(
        sorted(right, key=lambda pin: pin.path)
    )


def _source_report_matches(receipt: RunReceipt) -> bool:
    report = receipt.source_report
    binding = receipt.source_binding
    return bool(
        report is not None
        and report.tracked_tree_clean
        and report.commit_sha == binding.commit_sha
        and _same_source_files(report.source_files, binding.files)
        and binding.import_origin_manifest_sha256 is not None
        and report.import_origin_manifest_sha256
        == binding.import_origin_manifest_sha256
        and report.imported_module_count > 0
        and _valid_sha256(report.verification_report_sha256)
    )


def _arm_report_matches(
    report: ExternalArmVerificationReport,
    arm: ArmReceipt,
    receipt: RunReceipt,
    protocol: Task18Protocol,
) -> bool:
    source_report = receipt.source_report
    expected_advances = protocol.index_plan.native_advances_per_arm
    expected_capital = protocol.economics.initial_capital_per_symbol_usdt
    return bool(
        source_report is not None
        and report.run_id == receipt.run_id
        and report.protocol_digest == receipt.protocol_digest
        and report.source_report_binding_sha256 == source_report.binding_sha256
        and report.arm_id == arm.arm_id
        and report.pair_id == arm.pair_id
        and report.role == arm.role
        and report.configuration == arm.configuration
        and report.symbol == arm.symbol
        and report.scenario == arm.scenario
        and report.dataset_id == arm.dataset_id == receipt.dataset_id
        and report.native_ledger_sha256 == arm.native_ledger_sha256
        and report.arm_result_sha256 == arm.arm_result_sha256
        and report.expected_advances == expected_advances
        and report.observed_advances == expected_advances
        and _finite_number(arm.initial_equity_usdt)
        and math.isclose(
            report.initial_equity_usdt,
            float(arm.initial_equity_usdt),
            rel_tol=0.0,
            abs_tol=1e-8,
        )
        and math.isclose(
            report.initial_equity_usdt,
            expected_capital,
            rel_tol=0.0,
            abs_tol=1e-8,
        )
        and _finite_number(arm.terminal_equity_usdt)
        and math.isclose(
            report.terminal_equity_usdt,
            float(arm.terminal_equity_usdt),
            rel_tol=0.0,
            abs_tol=1e-8,
        )
        and _finite_number(arm.max_drawdown)
        and math.isclose(
            report.max_drawdown,
            float(arm.max_drawdown),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and report.terminal_flat is arm.terminal_flat
        and report.pending_order_count == arm.pending_order_count
        and _valid_sha256(report.verification_report_sha256)
    )


def evaluate_run_receipt(
    receipt: RunReceipt, protocol: Task18Protocol
) -> RunEvaluation:
    """Compare declaration and report fields without asserting authenticity."""
    issues: list[str] = []
    declaration_consistent = True
    expected_dataset = protocol.historical_reference
    if receipt.protocol_digest != protocol.adapted_protocol_digest:
        declaration_consistent = False
        issues.append("run adapted protocol digest does not match recipe")
    if receipt.historical_protocol_digest != expected_dataset.protocol_digest:
        declaration_consistent = False
        issues.append("run historical reference digest does not match recipe")
    if receipt.dataset_id != expected_dataset.dataset_id:
        declaration_consistent = False
        issues.append("run dataset id does not match historical reference")
    if receipt.manifest_sha256 != expected_dataset.manifest_sha256:
        declaration_consistent = False
        issues.append("run manifest digest does not match historical reference")
    if receipt.arrays_sha256 != expected_dataset.arrays_sha256:
        declaration_consistent = False
        issues.append("run array digest does not match historical reference")

    expected_arms = _expected_arm_by_id(protocol)
    seen: set[str] = set()
    roster_valid = True
    identity_valid = True
    declared_metrics_well_formed = True
    completed_arms: list[ArmReceipt] = []
    arm_reports_valid = True
    for arm in receipt.arms:
        if not isinstance(arm.arm_id, str) or not arm.arm_id:
            roster_valid = False
            declared_metrics_well_formed = False
            arm_reports_valid = False
            issues.append("arm receipt has an invalid arm id")
            continue
        if arm.arm_id in seen:
            roster_valid = False
            arm_reports_valid = False
            issues.append(f"duplicate arm receipt: {arm.arm_id}")
            continue
        seen.add(arm.arm_id)
        expected = expected_arms.get(arm.arm_id)
        if expected is None:
            roster_valid = False
            arm_reports_valid = False
            issues.append(f"unexpected arm receipt: {arm.arm_id}")
            continue
        if not _arm_matches_spec(arm, expected):
            identity_valid = False
            declaration_consistent = False
            issues.append(f"arm identity does not match declared roster: {arm.arm_id}")
        if (
            arm.protocol_digest != protocol.adapted_protocol_digest
            or arm.dataset_id != expected_dataset.dataset_id
            or arm.manifest_sha256 != expected_dataset.manifest_sha256
            or arm.arrays_sha256 != expected_dataset.arrays_sha256
            or arm.source_binding_sha256 != receipt.source_binding.manifest_sha256
        ):
            declaration_consistent = False
            issues.append(f"arm declaration does not match run binding: {arm.arm_id}")
        if not isinstance(arm.status, str) or arm.status not in _RECEIPT_STATUSES:
            declared_metrics_well_formed = False
            arm_reports_valid = False
            issues.append(f"invalid arm status: {arm.arm_id}")
            continue
        progress_object: object = arm.progress
        progress_valid = progress_object is None or isinstance(
            progress_object, ArmAttemptProgress
        )
        if not progress_valid:
            declared_metrics_well_formed = False
            issues.append(f"arm progress record is invalid: {arm.arm_id}")
        if arm.status != "COMPLETED":
            allowed_reasons = _STATUS_REASONS[arm.status]
            if (
                not isinstance(arm.reason_code, str)
                or arm.reason_code not in allowed_reasons
            ):
                declared_metrics_well_formed = False
                issues.append(f"arm attempt reason is invalid: {arm.arm_id}")
            if (
                arm.status == "NOT_ATTEMPTED"
                and isinstance(progress_object, ArmAttemptProgress)
                and any(
                    value is not None
                    for value in (
                        progress_object.native_advances_observed,
                        progress_object.last_native_index,
                        progress_object.last_observed_equity_usdt,
                        progress_object.max_drawdown_observed,
                    )
                )
            ):
                declared_metrics_well_formed = False
                issues.append(f"unattempted arm has observed progress: {arm.arm_id}")
            if arm.status == "NOT_ATTEMPTED" and any(
                value is not None
                for value in (
                    arm.initial_equity_usdt,
                    arm.terminal_equity_usdt,
                    arm.max_drawdown,
                    arm.terminal_flat,
                    arm.pending_order_count,
                    arm.native_ledger_sha256,
                    arm.arm_result_sha256,
                    arm.external_report,
                    arm.error_code,
                )
            ):
                declared_metrics_well_formed = False
                issues.append(f"unattempted arm has completion facts: {arm.arm_id}")
            if arm.status == "FAILED" and (
                not isinstance(arm.error_code, str) or not arm.error_code.strip()
            ):
                declared_metrics_well_formed = False
                issues.append(f"failed arm has no failure code: {arm.arm_id}")
            continue
        if arm.reason_code is not None or arm.error_code is not None:
            declared_metrics_well_formed = False
            issues.append(f"completed arm has attempt reason or error: {arm.arm_id}")
        completed_arms.append(arm)
        report = arm.external_report
        if report is None or not _arm_report_matches(report, arm, receipt, protocol):
            arm_reports_valid = False
            issues.append(f"arm report is absent or mismatched: {arm.arm_id}")
        elif not report.cost_oracle_passed:
            declared_metrics_well_formed = False
            issues.append(f"reported native cost oracle failed: {arm.arm_id}")
        if not _finite_number(arm.initial_equity_usdt) or not math.isclose(
            float(arm.initial_equity_usdt),
            protocol.economics.initial_capital_per_symbol_usdt,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            declared_metrics_well_formed = False
            issues.append(f"initial equity is invalid: {arm.arm_id}")
        if not _finite_number(arm.terminal_equity_usdt):
            declared_metrics_well_formed = False
            issues.append(f"terminal equity is not finite: {arm.arm_id}")
        if not _finite_number(arm.max_drawdown) or float(arm.max_drawdown) < 0:
            declared_metrics_well_formed = False
            issues.append(f"maximum drawdown is invalid: {arm.arm_id}")
        if not isinstance(arm.terminal_flat, bool):
            declared_metrics_well_formed = False
            issues.append(f"terminal-flat value is invalid: {arm.arm_id}")
        if (
            not isinstance(arm.pending_order_count, int)
            or isinstance(arm.pending_order_count, bool)
            or arm.pending_order_count < 0
        ):
            declared_metrics_well_formed = False
            issues.append(f"pending-order count is invalid: {arm.arm_id}")
        if not _valid_sha256(arm.native_ledger_sha256):
            declared_metrics_well_formed = False
            issues.append(f"native ledger digest is invalid: {arm.arm_id}")
        if not _valid_sha256(arm.arm_result_sha256):
            declared_metrics_well_formed = False
            issues.append(f"arm result digest is invalid: {arm.arm_id}")

    missing = set(expected_arms) - seen
    if missing:
        roster_valid = False
        arm_reports_valid = False
        issues.append(f"missing {len(missing)} declared arm receipts")
    roster_complete = roster_valid and not missing
    declaration_consistent = declaration_consistent and identity_valid and roster_valid

    source_report_matches = _source_report_matches(receipt)
    if not source_report_matches:
        issues.append("source report is absent or does not match its declaration")
    arm_reports_match = (
        arm_reports_valid
        and bool(completed_arms)
        and all(
            arm.external_report is not None
            and _arm_report_matches(arm.external_report, arm, receipt, protocol)
            for arm in completed_arms
        )
    )
    if not arm_reports_match:
        issues.append(
            "one or more completed arm reports are absent or do not match their declarations"
        )

    all_arms_completed = (
        roster_complete
        and len(completed_arms) == len(expected_arms)
        and all(arm.status == "COMPLETED" for arm in receipt.arms)
    )
    runtime_completed = receipt.runtime_completed
    if not runtime_completed:
        issues.append("run runtime did not complete")
    report_bound_diagnostics = (
        declaration_consistent
        and roster_complete
        and runtime_completed
        and all_arms_completed
        and source_report_matches
        and arm_reports_match
        and declared_metrics_well_formed
    )
    terminal_claims_pass_diagnostic = report_bound_diagnostics and all(
        arm.external_report is not None
        and arm.external_report.terminal_settlement_complete
        and arm.external_report.terminal_flat
        and arm.external_report.pending_order_count == 0
        and arm.terminal_flat is True
        and arm.pending_order_count == 0
        for arm in completed_arms
    )
    drawdown_claims_pass_diagnostic = report_bound_diagnostics and all(
        _finite_number(arm.max_drawdown)
        and float(arm.max_drawdown) <= protocol.gates.maximum_drawdown_per_book
        for arm in completed_arms
    )

    required_scenarios = set(protocol.gates.required_scenarios)
    roster_scenarios = {pair.scenario for pair in protocol.pairs}
    required_scenarios_covered = (
        roster_complete and required_scenarios == roster_scenarios
    )
    diagnostic_pair_arithmetic_available = (
        report_bound_diagnostics and terminal_claims_pass_diagnostic
    )
    return RunEvaluation(
        declaration_consistent=declaration_consistent,
        source_report_matches=source_report_matches,
        arm_reports_match=arm_reports_match,
        roster_complete=roster_complete,
        required_scenarios_covered=required_scenarios_covered,
        declared_runtime_completed=runtime_completed,
        declared_arm_statuses_complete=all_arms_completed,
        declared_metrics_well_formed=declared_metrics_well_formed,
        terminal_claims_pass_diagnostic=terminal_claims_pass_diagnostic,
        drawdown_claims_pass_diagnostic=drawdown_claims_pass_diagnostic,
        diagnostic_pair_arithmetic_available=diagnostic_pair_arithmetic_available,
        winner_status=protocol.gates.winner_status,
        adoption_status=protocol.gates.adoption_status,
        issues=tuple(issues),
    )


def _arm_matches_spec(arm: ArmReceipt, expected: ArmSpec) -> bool:
    return (
        arm.pair_id == expected.pair_id
        and arm.role == expected.role
        and arm.configuration == expected.configuration
        and arm.symbol == expected.symbol
        and arm.scenario == expected.scenario
    )


def diagnostic_pair_delta(
    receipt: RunReceipt,
    protocol: Task18Protocol,
    pair_id: str,
) -> float:
    """Return unverified receipt arithmetic; this is not an authoritative metric."""
    evaluation = evaluate_run_receipt(receipt, protocol)
    if not evaluation.diagnostic_pair_arithmetic_available:
        raise ValueError(
            "diagnostic arithmetic requires a complete declaration and matching reports"
        )
    pair = next((item for item in protocol.pairs if item.pair_id == pair_id), None)
    if pair is None:
        raise ValueError(f"unknown pair id: {pair_id}")
    arms = {arm.arm_id: arm for arm in receipt.arms}
    nonrl = arms[f"{pair_id}-NONRL"]
    cash = arms[f"{pair_id}-CASH"]
    nonrl_equity = nonrl.terminal_equity_usdt
    cash_equity = cash.terminal_equity_usdt
    if not _finite_number(nonrl_equity) or not _finite_number(cash_equity):
        raise ValueError("pair terminal equity is not finite")
    return (
        float(nonrl_equity) - float(cash_equity)
    ) / protocol.economics.initial_capital_per_symbol_usdt


def diagnostic_combined_delta(
    receipt: RunReceipt,
    protocol: Task18Protocol,
    configuration: str,
    scenario: str,
) -> float:
    """Return unverified two-book arithmetic; this is not an authoritative metric."""
    pair_ids = [
        pair.pair_id
        for pair in protocol.pairs
        if pair.configuration == configuration and pair.scenario == scenario
    ]
    if len(pair_ids) != len(protocol.dataset.symbols):
        raise ValueError(
            "combined metric requires exactly one pair per declared symbol"
        )
    return sum(
        diagnostic_pair_delta(receipt, protocol, pair_id) for pair_id in pair_ids
    ) / len(protocol.dataset.symbols)
