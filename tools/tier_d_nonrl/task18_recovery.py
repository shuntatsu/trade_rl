"""Task18 stopped-attempt handoff, never runtime or research authorization.

The supervisor must first establish exclusive ownership and actual process-tree
termination, then supply an externally pinned recovery bundle. This module
preserves and binds bytes; it does not authenticate verifier producers or
interpret the opaque frozen execution plan, native ledger or verifier report.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from tools.tier_d_nonrl.task18_receipts import (
    ArmReceipt,
    ExternalArmVerificationReport,
    RunReceipt,
    _finite_number,
    evaluate_run_receipt,
)
from tools.tier_d_nonrl.task18_recipe import build_task18_protocol
from trade_rl._validation import require_git_sha, require_sha256
from trade_rl.artifacts.atomic_write import atomic_rename_directory, atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.verified_file import open_regular_binary

_SCHEMA = "task18_stopped_recovery_handoff_v1"


@dataclass(frozen=True, slots=True)
class RecoveryFilePin:
    """Exact byte identity and bounded read size from the sealed bundle."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("recovery path must be a nonempty string")
        require_sha256(self.sha256, field="recovery file digest")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise ValueError("recovery file size must be a positive integer")


@dataclass(frozen=True, slots=True)
class RecoveryHandoff:
    original_run_id: str
    source_commit: str
    protocol_digest: str
    completed_arm_ids: tuple[str, ...]
    recovered_arm_ids: tuple[str, ...]
    pending_arm_ids: tuple[str, ...]
    files: tuple[RecoveryFilePin, ...]

    @property
    def execution_authorized(self) -> bool:
        return False

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "schema": _SCHEMA,
                **asdict(self),
                "execution_authorized": False,
            }
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _check_receipt(run: RunReceipt) -> None:
    result = evaluate_run_receipt(run, build_task18_protocol())
    if not (
        result.declaration_consistent
        and result.roster_complete
        and result.required_scenarios_covered
        and result.source_report_matches
        and result.declared_metrics_well_formed
    ):
        raise ValueError("recovery run declaration or report is inconsistent")
    if any(a.status == "COMPLETED" for a in run.arms) and not result.arm_reports_match:
        raise ValueError("completed arm reports do not match the stopped run")
    # Also reject malformed partial numeric claims on maintained sources
    # predating the independent receipt admission correction in #1362.
    for arm in run.arms:
        if arm.initial_equity_usdt is not None and not _finite_number(
            arm.initial_equity_usdt
        ):
            raise ValueError("invalid recovery initial equity")


def _receipt_bytes(arm: ArmReceipt) -> bytes:
    return canonical_json_bytes(arm.to_mapping())


def _expected_receipt_pin(
    expected: dict[str, tuple[str, int | None]], path: str, raw: bytes
) -> None:
    expected[path] = (hashlib.sha256(raw).hexdigest(), len(raw))


def plan_stopped_recovery(
    stopped: RunReceipt,
    candidates: tuple[ArmReceipt, ...],
    pins: tuple[RecoveryFilePin, ...],
    *,
    expected_source_commit: str,
    expected_frozen_plan_sha256: str,
) -> RecoveryHandoff:
    """Reconcile exact roster claims without granting reuse or execution authority.

    Every already-completed arm must be retained unchanged. Only RUNNING or
    INTERRUPTED arms may acquire a late completed receipt. FAILED and
    NOT_ATTEMPTED arms cannot be promoted by an unaccounted artifact.
    The frozen plan digest must come from the pre-stop freeze, not the recovery
    bundle. The resulting canonical handoff digest must be separately pinned
    by the owner.
    """
    require_git_sha(expected_source_commit)
    require_sha256(expected_frozen_plan_sha256, field="pre-stop frozen plan")
    if (
        type(stopped) is not RunReceipt
        or stopped.runtime_completed
        or stopped.source_binding.commit_sha != expected_source_commit
    ):
        raise ValueError("recovery requires an incomplete exact-source run")
    if type(candidates) is not tuple or any(
        type(a) is not ArmReceipt or a.status != "COMPLETED" for a in candidates
    ):
        raise ValueError("recovery candidates must be completed receipt records")
    if type(pins) is not tuple or any(type(p) is not RecoveryFilePin for p in pins):
        raise ValueError("recovery pins must be immutable file records")
    _check_receipt(stopped)
    source_report = stopped.source_report
    if source_report is None:
        raise ValueError("recovery requires a matching source report")

    by_id: dict[str, ArmReceipt] = {}
    for candidate in candidates:
        if not isinstance(candidate.arm_id, str) or candidate.arm_id in by_id:
            raise ValueError("duplicate or invalid completed arm identity")
        by_id[candidate.arm_id] = candidate
    original = {a.arm_id: a for a in stopped.arms}
    if not by_id.keys() <= original.keys():
        raise ValueError("foreign completed arm")
    completed = []
    recovered = []
    pending = []
    reconciled = []
    expected: dict[str, tuple[str, int | None]] = {
        "frozen-plan.bin": (expected_frozen_plan_sha256, None),
        "source-verification.json": (source_report.verification_report_sha256, None),
    }
    _expected_receipt_pin(
        expected, "stopped-run.json", canonical_json_bytes(stopped.to_mapping())
    )

    # Use recipe order, not caller order or filesystem enumeration order.
    for spec in build_task18_protocol().arms:
        old = original[spec.arm_id]
        new = by_id.get(spec.arm_id)
        if new is None:
            if old.status == "COMPLETED":
                raise ValueError("previously completed arm is missing")
            pending.append(spec.arm_id)
            reconciled.append(old)
            continue
        if old.status == "COMPLETED":
            if _receipt_bytes(old) != _receipt_bytes(new):
                raise ValueError("previously completed receipt changed")
        elif old.status in {"RUNNING", "INTERRUPTED"}:
            recovered.append(spec.arm_id)
        else:
            raise ValueError("unaccounted completion for an ineligible arm")
        report = new.external_report
        if (
            type(report) is not ExternalArmVerificationReport
            or not report.terminal_settlement_complete
            or not report.terminal_flat
            or report.pending_order_count != 0
            or not report.cost_oracle_passed
        ):
            raise ValueError("completed arm lacks a matching terminal verifier claim")
        prefix = f"arms/{spec.arm_id}/"
        _expected_receipt_pin(expected, prefix + "receipt.json", _receipt_bytes(new))
        expected[prefix + "native-ledger.json"] = (report.native_ledger_sha256, None)
        expected[prefix + "result.json"] = (report.arm_result_sha256, None)
        expected[prefix + "verification.json"] = (
            report.verification_report_sha256,
            None,
        )
        completed.append(spec.arm_id)
        reconciled.append(new)

    _check_receipt(replace(stopped, arms=tuple(reconciled)))
    by_path = {p.path: p for p in pins}
    if len(by_path) != len(pins) or by_path.keys() != expected.keys():
        raise ValueError("recovery file roster is missing, duplicate or foreign")
    for path, (digest, size) in expected.items():
        pin = by_path[path]
        if path == "frozen-plan.bin" and pin.sha256 != expected_frozen_plan_sha256:
            raise ValueError("frozen plan differs from pre-stop digest")
        if pin.sha256 != digest or (size is not None and pin.size_bytes != size):
            raise ValueError(f"recovery file pin differs from receipt: {path}")
    return RecoveryHandoff(
        original_run_id=stopped.run_id,
        source_commit=expected_source_commit,
        protocol_digest=stopped.protocol_digest,
        completed_arm_ids=tuple(completed),
        recovered_arm_ids=tuple(recovered),
        pending_arm_ids=tuple(pending),
        files=tuple(by_path[p] for p in sorted(by_path)),
    )


def _plain_directory(path: Path) -> Path:
    if ".." in path.parts:
        raise ValueError("recovery directory must not contain parent traversal")
    path = path.absolute()
    if (
        any(p.is_symlink() or p.is_junction() for p in (*reversed(path.parents), path))
        or not path.is_dir()
    ):
        raise ValueError("recovery directories must be regular and non-aliased")
    return path


def _check_bundle_roster(root: Path, pins: tuple[RecoveryFilePin, ...]) -> None:
    # Traverse only the exact known directory tree, never an unexpected child.
    expected: dict[Path, set[str]] = {root: set()}
    for pin in pins:
        path = root / pin.path
        for parent in path.parents:
            if parent == root.parent:
                break
            child = path.relative_to(parent).parts[0]
            expected.setdefault(parent, set()).add(child)
    for directory in sorted(expected, key=lambda p: len(p.parts)):
        names = expected[directory]
        if directory.is_symlink() or directory.is_junction() or not directory.is_dir():
            raise ValueError("invalid recovery bundle directory")
        if {p.name for p in directory.iterdir()} != names:
            raise ValueError("recovery bundle contains missing or unexpected entries")
    for pin in pins:
        path = root / pin.path
        if path.is_symlink() or path.is_junction() or not path.is_file():
            raise ValueError("recovery bundle file must be a regular file")


def _copy_pinned_file(source: Path, destination: Path, pin: RecoveryFilePin) -> None:
    """Stream the exact bytes into private staging; no unbounded decode or read."""
    digest = hashlib.sha256()
    remaining = pin.size_bytes
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        open_regular_binary(source, field=pin.path) as reader,
        destination.open("xb") as writer,
    ):
        while remaining:
            chunk = reader.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"truncated recovery file: {pin.path}")
            digest.update(chunk)
            writer.write(chunk)
            remaining -= len(chunk)
        if reader.read(1) or digest.hexdigest() != pin.sha256:
            raise ValueError(f"recovery file size or digest mismatch: {pin.path}")
        writer.flush()
        os.fsync(writer.fileno())


def stage_stopped_recovery(
    source: Path,
    destination: Path,
    stopped: RunReceipt,
    candidates: tuple[ArmReceipt, ...],
    pins: tuple[RecoveryFilePin, ...],
    *,
    expected_source_commit: str,
    expected_frozen_plan_sha256: str,
    expected_handoff_sha256: str,
) -> RecoveryHandoff:
    """Publish a byte-identical recovery input bundle at a fresh destination.

    Source and destination parent must be controlled by the exclusive stopped
    attempt owner. This is not a process-stop detector, concurrent-writer
    security boundary, plan decoder, native ledger verifier or resume runner.
    No existing receipt is rewritten and no new completion is manufactured.
    """
    require_sha256(expected_handoff_sha256, field="expected recovery handoff")
    plan = plan_stopped_recovery(
        stopped,
        candidates,
        pins,
        expected_source_commit=expected_source_commit,
        expected_frozen_plan_sha256=expected_frozen_plan_sha256,
    )
    if plan.sha256 != expected_handoff_sha256:
        raise ValueError("recovery handoff digest mismatch")
    source = _plain_directory(Path(source))
    destination = Path(destination)
    if ".." in destination.parts or not destination.name:
        raise ValueError("invalid recovery destination")
    parent = _plain_directory(destination.parent)
    destination = parent / destination.name
    if destination.exists() or destination.is_symlink() or destination.is_junction():
        raise FileExistsError("recovery destination already exists")
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("recovery destination must be outside the stopped bundle")
    _check_bundle_roster(source, plan.files)

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=parent))
    try:
        for pin in plan.files:
            _copy_pinned_file(source / pin.path, staging / pin.path, pin)
        _check_bundle_roster(source, plan.files)
        atomic_write_bytes(staging / "recovery.json", plan.canonical_bytes())
        atomic_rename_directory(staging, destination)
    finally:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
    return plan


__all__ = [
    "RecoveryFilePin",
    "RecoveryHandoff",
    "plan_stopped_recovery",
    "stage_stopped_recovery",
]
