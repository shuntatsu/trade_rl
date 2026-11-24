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
from dataclasses import dataclass, replace
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
from tools.tier_d_nonrl.task18_native_lineage import (
    SUCCESSOR_PREFIX,
    StagedSyntheticGeneration,
    lineage_bytes,
    ordered_union,
    read_generation,
    remaining,
    require_pending_subset,
    validate_generation_paths,
    verify_successor_lineage,
)
from tools.tier_d_nonrl.task18_native_progress import NativeProgressTracker
from tools.tier_d_nonrl.task18_receipts import (
    ArmAttemptProgress,
    ArmReceipt,
    RunReceipt,
    SourceBinding,
)
from tools.tier_d_nonrl.task18_recipe import Task18Protocol
from trade_rl.artifacts.atomic_write import atomic_rename_directory, atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.data.market import MarketDataset


@dataclass(frozen=True, slots=True)
class RecoveredSyntheticAttempt:
    attempt: SyntheticNativeAttempt
    stopped: RunReceipt
    handoff: recovery.RecoveryHandoff
    completed: tuple[VerifiedNativeArmArtifact, ...]
    handoff_sha256s: tuple[str, ...]
    run_ids: tuple[str, ...]

    @property
    def completed_arm_ids(self) -> tuple[str, ...]:
        return tuple(a.receipt.arm_id for a in self.completed)

    @property
    def pending_arm_ids(self) -> tuple[str, ...]:
        return remaining(self.completed_arm_ids)

    @property
    def execution_authorized(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class SyntheticSuccessor:
    run: RunReceipt
    predecessor_handoff_sha256: str
    lineage_sha256: str
    completed: tuple[VerifiedNativeArmArtifact, ...]
    inherited_completed_arm_ids: tuple[str, ...]

    @property
    def completed_arm_ids(self) -> tuple[str, ...]:
        return ordered_union(
            self.inherited_completed_arm_ids,
            tuple(a.receipt.arm_id for a in self.completed),
        )

    @property
    def pending_arm_ids(self) -> tuple[str, ...]:
        return remaining(self.completed_arm_ids)

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
    *,
    predecessor: RecoveredSyntheticAttempt | None,
    lineage: bytes | None,
) -> RecoveredSyntheticAttempt:
    _validate_run(attempt, stopped)
    if stopped.source_binding != source:
        raise ValueError("stopped source differs from the expected binding")
    plan_pin = next(pin for pin in handoff.files if pin.path == "frozen-plan.bin")
    if (plan_pin.sha256, plan_pin.size_bytes) != (
        attempt.plan_pin.sha256,
        attempt.plan_pin.size_bytes,
    ):
        raise ValueError("stopped plan differs from the exact pre-stop pin")
    if predecessor is None:
        if lineage is not None or stopped.run_id.startswith(SUCCESSOR_PREFIX):
            raise ValueError("successor requires complete ordered ancestor lineage")
        inherited: tuple[VerifiedNativeArmArtifact, ...] = ()
        handoffs: tuple[str, ...] = ()
        run_ids: tuple[str, ...] = ()
    else:
        if lineage is None:
            raise ValueError("successor requires complete ordered ancestor lineage")
        if (
            stopped.run_id in predecessor.run_ids
            or handoff.sha256 in predecessor.handoff_sha256s
        ):
            raise ValueError("duplicate generation identity in ancestor lineage")
        verify_successor_lineage(
            lineage,
            predecessor=predecessor.handoff,
            predecessor_run=predecessor.stopped,
            ancestor_handoff_sha256s=predecessor.handoff_sha256s,
            retained=predecessor.completed_arm_ids,
            stopped=stopped,
            current=handoff,
            plan_pin=attempt.plan_pin,
        )
        inherited = predecessor.completed
        handoffs = predecessor.handoff_sha256s
        run_ids = predecessor.run_ids
    all_ids = ordered_union(
        tuple(a.receipt.arm_id for a in inherited), handoff.completed_arm_ids
    )
    completed = {a.receipt.arm_id: a for a in inherited}
    for arm_id in handoff.completed_arm_ids:
        completed[arm_id] = load_completed_native_arm(
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
    return RecoveredSyntheticAttempt(
        attempt,
        stopped,
        handoff,
        tuple(completed[a] for a in all_ids),
        (*handoffs, handoff.sha256),
        (*run_ids, stopped.run_id),
    )


def _open_one(
    root: Path,
    lineage_sha256: str | None,
    handoff_sha256: str,
    attempt: SyntheticNativeAttempt,
    source: SourceBinding,
    predecessor: RecoveredSyntheticAttempt | None,
) -> RecoveredSyntheticAttempt:
    bundle, lineage = read_generation(root, lineage_sha256)
    protocol = synthetic_receipt_protocol(attempt, source)
    plan, stopped = recovery._load_staged_recovery(
        protocol,
        bundle,
        expected_source_commit=source.commit_sha,
        expected_frozen_plan_sha256=attempt.plan_pin.sha256,
        expected_handoff_sha256=handoff_sha256,
        file_size_limits=_limits(protocol, attempt.plan_pin),
    )
    return _verify_completed(
        bundle, attempt, source, stopped, plan, predecessor=predecessor, lineage=lineage
    )


def _open_ancestors(
    ancestors: tuple[StagedSyntheticGeneration, ...],
    attempt: SyntheticNativeAttempt,
    source: SourceBinding,
) -> RecoveredSyntheticAttempt | None:
    predecessor = None
    for pin in ancestors:
        predecessor = _open_one(
            pin.root,
            pin.lineage_sha256,
            pin.handoff_sha256,
            attempt,
            source,
            predecessor,
        )
    return predecessor


def _inspect_one(
    root: Path,
    pins: tuple[recovery.RecoveryFilePin, ...],
    *,
    attempt: SyntheticNativeAttempt,
    source: SourceBinding,
    predecessor: RecoveredSyntheticAttempt | None,
    lineage: bytes | None,
) -> RecoveredSyntheticAttempt:
    protocol = synthetic_receipt_protocol(attempt, source)
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
        expected_source_commit=source.commit_sha,
        expected_frozen_plan_sha256=attempt.plan_pin.sha256,
    )
    return _verify_completed(
        root, attempt, source, stopped, plan, predecessor=predecessor, lineage=lineage
    )


def inspect_synthetic_stopped_recovery(
    root: Path,
    pins: tuple[recovery.RecoveryFilePin, ...],
    *,
    attempt: SyntheticNativeAttempt,
    expected_source_binding: SourceBinding,
    ancestors: tuple[StagedSyntheticGeneration, ...] = (),
    expected_lineage_sha256: str | None = None,
) -> RecoveredSyntheticAttempt:
    """Return the verified cumulative frontier, never local-only pending IDs."""
    validate_generation_paths(ancestors, root, expected_lineage_sha256)
    bundle, lineage = read_generation(root, expected_lineage_sha256)
    prior = _open_ancestors(ancestors, attempt, expected_source_binding)
    return _inspect_one(
        bundle,
        pins,
        attempt=attempt,
        source=expected_source_binding,
        predecessor=prior,
        lineage=lineage,
    )


def _fresh_destination(destination: Path, roots: tuple[Path, ...]) -> Path:
    destination = Path(destination)
    if ".." in destination.parts or not destination.name:
        raise ValueError("invalid successor destination")
    destination = recovery._plain_directory(destination.parent) / destination.name
    if destination.exists() or destination.is_symlink() or destination.is_junction():
        raise FileExistsError("successor destination already exists")
    if any(
        destination.is_relative_to(root) or root.is_relative_to(destination)
        for root in roots
    ):
        raise ValueError("successor must be outside the staged handoff and ancestors")
    return destination


def stage_synthetic_stopped_recovery(
    source: Path,
    destination: Path,
    pins: tuple[recovery.RecoveryFilePin, ...],
    *,
    attempt: SyntheticNativeAttempt,
    expected_source_binding: SourceBinding,
    expected_handoff_sha256: str,
    ancestors: tuple[StagedSyntheticGeneration, ...] = (),
    expected_lineage_sha256: str | None = None,
) -> RecoveredSyntheticAttempt:
    """Copy a verified generation, including exact lineage bytes when present."""
    validate_generation_paths(ancestors, source, expected_lineage_sha256)
    bundle, lineage = read_generation(source, expected_lineage_sha256)
    destination = _fresh_destination(
        destination, (source.absolute(), *(pin.root.absolute() for pin in ancestors))
    )
    prior = _open_ancestors(ancestors, attempt, expected_source_binding)
    opened = _inspect_one(
        bundle,
        pins,
        attempt=attempt,
        source=expected_source_binding,
        predecessor=prior,
        lineage=lineage,
    )
    if opened.handoff.sha256 != expected_handoff_sha256:
        raise ValueError("recovery handoff digest mismatch")
    protocol = synthetic_receipt_protocol(attempt, expected_source_binding)

    def stage(target: Path) -> recovery.RecoveryHandoff:
        return recovery._stage_stopped_recovery(
            protocol,
            bundle,
            target,
            pins,
            expected_source_commit=expected_source_binding.commit_sha,
            expected_frozen_plan_sha256=attempt.plan_pin.sha256,
            expected_handoff_sha256=expected_handoff_sha256,
        )

    if lineage is None:
        if stage(destination) != opened.handoff:
            raise ValueError("staged recovery differs from verified generation")
        return opened
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        if stage(staging / "attempt") != opened.handoff:
            raise ValueError("staged recovery differs from verified generation")
        atomic_write_bytes(staging / "lineage.json", lineage)
        atomic_rename_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return opened


def open_synthetic_staged_recovery(
    root: Path,
    *,
    dataset: MarketDataset,
    expected_binding: FrozenInputBinding,
    expected_pin: FrozenInputsPin,
    expected_source_binding: SourceBinding,
    expected_handoff_sha256: str,
    ancestors: tuple[StagedSyntheticGeneration, ...] = (),
    expected_lineage_sha256: str | None = None,
) -> RecoveredSyntheticAttempt:
    """Reopen a complete ordered chain; return the cumulative pending frontier."""
    validate_generation_paths(ancestors, root, expected_lineage_sha256)
    bundle, _ = read_generation(root, expected_lineage_sha256)
    attempt = open_synthetic_native_attempt(
        dataset,
        bundle / "frozen-plan.bin",
        expected_binding=expected_binding,
        expected_pin=expected_pin,
    )
    prior = _open_ancestors(ancestors, attempt, expected_source_binding)
    return _open_one(
        root,
        expected_lineage_sha256,
        expected_handoff_sha256,
        attempt,
        expected_source_binding,
        prior,
    )


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
    ancestors: tuple[StagedSyntheticGeneration, ...] = (),
    expected_lineage_sha256: str | None = None,
) -> SyntheticSuccessor:
    """Execute an explicitly selected pending subset in a fresh synthetic run.

    Original completed receipts stay under their original run ID. A separate
    immutable lineage record binds the successor to the original handoff.
    Write RUNNING before dispatch and promote only after publishing and
    independently reopening the actual arm artifact. Execution exceptions
    preserve a FAILED record with the latest actually observed progress, or
    unknown before the first returned native advance. Forced stop preserves
    only the last durable checkpoint, not an inferred actual stop position.
    Publication errors leave the slot RUNNING for later stopped recovery.
    The exclusive-owner assumption still applies. This is not a stop detector,
    power-loss proof, distributed single-writer lease, or historical execution
    admission. Every earlier generation is externally pinned and verified.
    Callers must retain the non-forking history and must not reuse a stale head.
    """
    opened = open_synthetic_staged_recovery(
        root,
        dataset=dataset,
        expected_binding=expected_binding,
        expected_pin=expected_pin,
        expected_source_binding=expected_source_binding,
        expected_handoff_sha256=expected_handoff_sha256,
        ancestors=ancestors,
        expected_lineage_sha256=expected_lineage_sha256,
    )
    plan = opened.handoff
    require_pending_subset(arm_ids, opened.pending_arm_ids)
    if (
        type(successor_run_label) is not str
        or not successor_run_label.strip()
        or successor_run_label in opened.run_ids
    ):
        raise ValueError("successor requires a distinct nonempty run label")
    successor_run_id = f"{SUCCESSOR_PREFIX}{plan.sha256}:{successor_run_label}"
    root = recovery._plain_directory(root)
    destination = _fresh_destination(
        destination, (root, *(pin.root.absolute() for pin in ancestors))
    )
    bundle, _ = read_generation(root, expected_lineage_sha256)
    source_report = opened.stopped.source_report
    if source_report is None:
        raise ValueError("stopped attempt has no source report")
    run = new_synthetic_run_receipt(
        opened.attempt,
        run_id=successor_run_id,
        source_binding=expected_source_binding,
        source_report=source_report,
    )
    lineage = lineage_bytes(
        plan,
        opened.handoff_sha256s,
        opened.completed_arm_ids,
        arm_ids,
        successor_run_id,
        expected_pin,
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        attempt_dir = staging / "attempt"
        attempt_dir.mkdir()
        for pin in plan.files:
            if pin.path in ("frozen-plan.bin", "source-verification.json"):
                recovery._copy_pinned_file(
                    bundle / pin.path, attempt_dir / pin.path, pin
                )
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
        tracker = NativeProgressTracker()

        def observe_progress(observed_id: str, progress: ArmAttemptProgress) -> None:
            nonlocal run
            if observed_id != arm_id:
                raise ValueError("native progress belongs to a different arm")
            if tracker.observe(progress):
                run = _replace_arm(run, replace(running, progress=progress))
                _write_run(attempt_dir, run)

        try:
            observation = next(
                opened.attempt.iter_arms((arm_id,), on_progress=observe_progress)
            )
            if (
                tracker.latest is None
                or tracker.latest.native_advances_observed
                != observation.verification.advances
            ):
                raise ValueError(
                    "native completion lacks the observed progress frontier"
                )
        except Exception:
            failed = replace(
                running,
                status="FAILED",
                reason_code="PROCESS_FAILED",
                error_code="NATIVE_ARM_EXECUTION_FAILED",
                progress=tracker.latest,
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
        run,
        plan.sha256,
        hashlib.sha256(lineage).hexdigest(),
        tuple(completed),
        opened.completed_arm_ids,
    )
