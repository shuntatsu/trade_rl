"""Synthetic-only stopped native attempt composition; never market authorization.

The historical receipt entrypoints remain fixed to their original recipe.
This adapter derives its separate receipt recipe from the pinned artificial
Dataset, frozen plan and externally expected source binding. No caller
supplied arbitrary recipe or test monkeypatch is used in this path.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from tools.tier_d_nonrl import task18_recovery as recovery
from tools.tier_d_nonrl.task18_frozen_inputs import FrozenInputBinding, FrozenInputsPin
from tools.tier_d_nonrl.task18_native import (
    SyntheticNativeAttempt,
    open_synthetic_native_attempt,
)
from tools.tier_d_nonrl.task18_native_artifact import (
    _MAX_LEDGER,
    _MAX_METADATA,
    NativeArmFilePin,
    VerifiedNativeArmArtifact,
    _validate_run,
    load_completed_native_arm,
    new_synthetic_run_receipt,
    publish_completed_native_arm,
    synthetic_receipt_protocol,
)
from tools.tier_d_nonrl.task18_receipts import ArmReceipt, RunReceipt, SourceBinding
from tools.tier_d_nonrl.task18_recipe import Task18Protocol
from trade_rl.artifacts.atomic_write import atomic_rename_directory, atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.data.market import MarketDataset

_SUCCESSOR_PREFIX = "task18-synthetic-successor:"


@dataclass(frozen=True, slots=True)
class RecoveredSyntheticAttempt:
    attempt: SyntheticNativeAttempt
    stopped: RunReceipt
    handoff: recovery.RecoveryHandoff
    completed: tuple[VerifiedNativeArmArtifact, ...]

    @property
    def execution_authorized(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class SyntheticSuccessor:
    run: RunReceipt
    predecessor_handoff_sha256: str
    lineage_sha256: str
    completed: tuple[VerifiedNativeArmArtifact, ...]

    @property
    def execution_authorized(self) -> bool:
        return False


def _limits(protocol: Task18Protocol, pin: FrozenInputsPin) -> dict[str, int]:
    limits = {
        "frozen-plan.bin": pin.size_bytes,
        "stopped-run.json": _MAX_METADATA,
        "source-verification.json": _MAX_METADATA,
    }
    for arm in protocol.arms:
        prefix = f"arms/{arm.arm_id}/"
        limits.update(
            {
                prefix + "native-ledger.json": _MAX_LEDGER,
                prefix + "result.json": _MAX_METADATA,
                prefix + "verification.json": _MAX_METADATA,
                prefix + "receipt.json": _MAX_METADATA,
            }
        )
    return limits


def _verify_completed(
    root: Path,
    attempt: SyntheticNativeAttempt,
    source: SourceBinding,
    stopped: RunReceipt,
    handoff: recovery.RecoveryHandoff,
) -> RecoveredSyntheticAttempt:
    _validate_run(attempt, stopped)
    # A successor contains only its own executed arms; its retained ancestors
    # deliberately keep their original receipts. Until complete lineage-chain
    # verification exists, never treat a successor as a new original attempt.
    if stopped.run_id.startswith(_SUCCESSOR_PREFIX):
        raise ValueError(
            "multi-hop successor recovery requires ancestor lineage verification"
        )
    if stopped.source_binding != source:
        raise ValueError("stopped source differs from the expected binding")
    plan_pin = next(pin for pin in handoff.files if pin.path == "frozen-plan.bin")
    if (plan_pin.sha256, plan_pin.size_bytes) != (
        attempt.plan_pin.sha256,
        attempt.plan_pin.size_bytes,
    ):
        raise ValueError("stopped plan differs from the exact pre-stop pin")
    completed = tuple(
        load_completed_native_arm(
            root,
            attempt=attempt,
            run=stopped,
            arm_id=arm_id,
            expected_pins=tuple(
                NativeArmFilePin(pin.path, pin.sha256, pin.size_bytes)
                for pin in handoff.files
                if pin.path.startswith(f"arms/{arm_id}/")
            ),
        )
        for arm_id in handoff.completed_arm_ids
    )
    return RecoveredSyntheticAttempt(attempt, stopped, handoff, completed)


def inspect_synthetic_stopped_recovery(
    root: Path,
    pins: tuple[recovery.RecoveryFilePin, ...],
    *,
    attempt: SyntheticNativeAttempt,
    expected_source_binding: SourceBinding,
) -> recovery.RecoveryHandoff:
    """Reverify each retained native arm as well as its stopped-bundle pins."""
    protocol = synthetic_receipt_protocol(attempt, expected_source_binding)
    limits = _limits(protocol, attempt.plan_pin)
    if (
        type(pins) is not tuple
        or len(pins) > len(limits)
        or any(
            type(pin) is not recovery.RecoveryFilePin
            or pin.path not in limits
            or pin.size_bytes > limits[pin.path]
            for pin in pins
        )
    ):
        raise ValueError("synthetic stopped file roster or size exceeds its bound")
    plan, stopped = recovery._inspect_stopped_recovery(
        protocol,
        root,
        pins,
        expected_source_commit=expected_source_binding.commit_sha,
        expected_frozen_plan_sha256=attempt.plan_pin.sha256,
    )
    _verify_completed(root, attempt, expected_source_binding, stopped, plan)
    return plan


def stage_synthetic_stopped_recovery(
    source: Path,
    destination: Path,
    pins: tuple[recovery.RecoveryFilePin, ...],
    *,
    attempt: SyntheticNativeAttempt,
    expected_source_binding: SourceBinding,
    expected_handoff_sha256: str,
) -> recovery.RecoveryHandoff:
    """Copy exact verified evidence; no replay or completion promotion."""
    plan = inspect_synthetic_stopped_recovery(
        source, pins, attempt=attempt, expected_source_binding=expected_source_binding
    )
    if plan.sha256 != expected_handoff_sha256:
        raise ValueError("recovery handoff digest mismatch")
    return recovery._stage_stopped_recovery(
        synthetic_receipt_protocol(attempt, expected_source_binding),
        source,
        destination,
        pins,
        expected_source_commit=expected_source_binding.commit_sha,
        expected_frozen_plan_sha256=attempt.plan_pin.sha256,
        expected_handoff_sha256=expected_handoff_sha256,
    )


def open_synthetic_staged_recovery(
    root: Path,
    *,
    dataset: MarketDataset,
    expected_binding: FrozenInputBinding,
    expected_pin: FrozenInputsPin,
    expected_source_binding: SourceBinding,
    expected_handoff_sha256: str,
) -> RecoveredSyntheticAttempt:
    """Reopen a staged handoff with fresh objects, without fit, recost or replay."""
    root = recovery._plain_directory(Path(root))
    attempt = open_synthetic_native_attempt(
        dataset,
        root / "frozen-plan.bin",
        expected_binding=expected_binding,
        expected_pin=expected_pin,
    )
    protocol = synthetic_receipt_protocol(attempt, expected_source_binding)
    plan, stopped = recovery._load_staged_recovery(
        protocol,
        root,
        expected_source_commit=expected_source_binding.commit_sha,
        expected_frozen_plan_sha256=expected_pin.sha256,
        expected_handoff_sha256=expected_handoff_sha256,
        file_size_limits=_limits(protocol, expected_pin),
    )
    return _verify_completed(root, attempt, expected_source_binding, stopped, plan)


def _replace_arm(run: RunReceipt, arm: ArmReceipt) -> RunReceipt:
    return replace(
        run, arms=tuple(arm if a.arm_id == arm.arm_id else a for a in run.arms)
    )


def _write_run(root: Path, run: RunReceipt) -> None:
    atomic_write_bytes(
        root / "stopped-run.json", canonical_json_bytes(run.to_mapping())
    )


def resume_synthetic_pending(
    root: Path,
    destination: Path,
    *,
    dataset: MarketDataset,
    expected_binding: FrozenInputBinding,
    expected_pin: FrozenInputsPin,
    expected_source_binding: SourceBinding,
    expected_handoff_sha256: str,
    successor_run_label: str,
    arm_ids: tuple[str, ...],
) -> SyntheticSuccessor:
    """Execute an explicitly selected pending subset in a fresh synthetic run.

    Original completed receipts stay under their original run ID. A separate
    immutable lineage record binds the successor to the original handoff.
    Write RUNNING before dispatch and promote only after publishing and
    independently reopening the actual arm artifact. Execution exceptions
    preserve a FAILED record with unknown progress (not fabricated counts).
    Publication errors leave the slot RUNNING for later stopped recovery.
    The exclusive-owner assumption still applies. This is not a stop detector,
    power-loss proof, full-run aggregation, or historical execution admission.
    The namespace binds the successor ID to the predecessor digest and
    rejects multi-hop recovery until verified ancestor lineage is available.
    """
    opened = open_synthetic_staged_recovery(
        root,
        dataset=dataset,
        expected_binding=expected_binding,
        expected_pin=expected_pin,
        expected_source_binding=expected_source_binding,
        expected_handoff_sha256=expected_handoff_sha256,
    )
    plan = opened.handoff
    if (
        type(arm_ids) is not tuple
        or not arm_ids
        or any(type(a) is not str or a not in plan.pending_arm_ids for a in arm_ids)
        or len(set(arm_ids)) != len(arm_ids)
        or tuple(a for a in plan.pending_arm_ids if a in arm_ids) != arm_ids
    ):
        raise ValueError(
            "successor requires an ordered nonempty subset of pending arms"
        )
    if (
        type(successor_run_label) is not str
        or not successor_run_label.strip()
        or successor_run_label == plan.original_run_id
    ):
        raise ValueError("successor requires a distinct nonempty run label")
    successor_run_id = f"{_SUCCESSOR_PREFIX}{plan.sha256}:{successor_run_label}"
    root = recovery._plain_directory(Path(root))
    destination = Path(destination)
    if ".." in destination.parts or not destination.name:
        raise ValueError("invalid successor destination")
    destination = recovery._plain_directory(destination.parent) / destination.name
    if destination.exists() or destination.is_symlink() or destination.is_junction():
        raise FileExistsError("successor destination already exists")
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError("successor must be outside the staged handoff")
    source_report = opened.stopped.source_report
    if source_report is None:
        raise ValueError("stopped attempt has no source report")
    run = new_synthetic_run_receipt(
        opened.attempt,
        run_id=successor_run_id,
        source_binding=expected_source_binding,
        source_report=source_report,
    )
    lineage = canonical_json_bytes(
        {
            "schema": "task18_synthetic_successor_lineage_v1",
            "scope": "SYNTHETIC_NATIVE_CONFORMANCE_ONLY",
            "original_run_id": plan.original_run_id,
            "successor_run_id": successor_run_id,
            "predecessor_handoff_sha256": plan.sha256,
            "protocol_digest": run.protocol_digest,
            "source_commit": plan.source_commit,
            "plan_pin": asdict(expected_pin),
            "retained_arm_ids": plan.completed_arm_ids,
            "selected_arm_ids": arm_ids,
            "execution_authorized": False,
        }
    )
    if len(lineage) > _MAX_METADATA:
        raise ValueError("successor lineage exceeds metadata limit")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        attempt_dir = staging / "attempt"
        attempt_dir.mkdir()
        for pin in plan.files:
            if pin.path in ("frozen-plan.bin", "source-verification.json"):
                recovery._copy_pinned_file(root / pin.path, attempt_dir / pin.path, pin)
        _write_run(attempt_dir, run)
        atomic_write_bytes(staging / "lineage.json", lineage)
        atomic_rename_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    attempt_dir = destination / "attempt"
    completed = []
    for arm_id in arm_ids:
        old = next(a for a in run.arms if a.arm_id == arm_id)
        running = replace(old, status="RUNNING", reason_code="RUN_IN_PROGRESS")
        run = _replace_arm(run, running)
        _write_run(attempt_dir, run)
        try:
            observation = next(opened.attempt.iter_arms((arm_id,)))
        except Exception:
            failed = replace(
                running,
                status="FAILED",
                reason_code="PROCESS_FAILED",
                error_code="NATIVE_ARM_EXECUTION_FAILED",
            )
            run = _replace_arm(run, failed)
            _write_run(attempt_dir, run)
            raise
        (attempt_dir / "arms").mkdir(exist_ok=True)
        published = publish_completed_native_arm(
            attempt_dir,
            attempt=opened.attempt,
            run=run,
            observation=observation,
        )
        verified = load_completed_native_arm(
            attempt_dir,
            attempt=opened.attempt,
            run=run,
            arm_id=arm_id,
            expected_pins=published.pins,
        )
        run = _replace_arm(run, verified.receipt)
        _validate_run(opened.attempt, run)
        _write_run(attempt_dir, run)
        completed.append(verified)
    return SyntheticSuccessor(
        run, plan.sha256, hashlib.sha256(lineage).hexdigest(), tuple(completed)
    )
