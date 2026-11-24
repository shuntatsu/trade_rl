"""Task18 stopped-attempt handoff, never runtime or research authorization.

The supervisor must first establish exclusive ownership and actual process-tree
termination, then supply an externally pinned recovery bundle. This module
preserves and binds bytes; it does not authenticate verifier producers or
interpret the opaque frozen execution plan, native ledger or verifier report.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO

from tools.tier_d_nonrl.task18_receipts import (
    ArmAttemptProgress,
    ArmReceipt,
    ExternalArmVerificationReport,
    ExternalSourceVerificationReport,
    RunReceipt,
    SourceBinding,
    _finite_number,
    evaluate_run_receipt,
)
from tools.tier_d_nonrl.task18_recipe import SourceFilePin, build_task18_protocol
from trade_rl._validation import require_git_sha, require_sha256
from trade_rl.artifacts.atomic_write import atomic_rename_directory, atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.verified_file import open_regular_binary, read_verified_bytes

_SCHEMA = "task18_stopped_recovery_handoff_v1"
_MAX_METADATA_BYTES = 4 * 1024 * 1024


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
    if len(raw) > _MAX_METADATA_BYTES:
        raise ValueError("recovery receipt exceeds metadata byte limit")
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
    plan = RecoveryHandoff(
        original_run_id=stopped.run_id,
        source_commit=expected_source_commit,
        protocol_digest=stopped.protocol_digest,
        completed_arm_ids=tuple(completed),
        recovered_arm_ids=tuple(recovered),
        pending_arm_ids=tuple(pending),
        files=tuple(by_path[p] for p in sorted(by_path)),
    )
    if len(plan.canonical_bytes()) > _MAX_METADATA_BYTES:
        raise ValueError("recovery handoff exceeds metadata byte limit")
    return plan


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


def _stream_pinned_file(
    source: Path, pin: RecoveryFilePin, writer: BinaryIO | None = None
) -> None:
    """Validate a bounded stream, optionally copying the exact verified bytes."""
    digest = hashlib.sha256()
    remaining = pin.size_bytes
    with open_regular_binary(source, field=pin.path) as reader:
        while remaining:
            chunk = reader.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"truncated recovery file: {pin.path}")
            digest.update(chunk)
            if writer is not None:
                writer.write(chunk)
            remaining -= len(chunk)
        if reader.read(1) or digest.hexdigest() != pin.sha256:
            raise ValueError(f"recovery file size or digest mismatch: {pin.path}")


def _copy_pinned_file(source: Path, destination: Path, pin: RecoveryFilePin) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as writer:
        _stream_pinned_file(source, pin, writer)
        writer.flush()
        os.fsync(writer.fileno())


def stage_stopped_recovery(
    source: Path,
    destination: Path,
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
    Read declarations from pinned files, not the stopped process's memory.
    No existing receipt is rewritten and no new completion is manufactured.
    """
    require_sha256(expected_handoff_sha256, field="expected recovery handoff")
    source = _plain_directory(Path(source))
    plan, _ = _read_bundle_plan(
        source,
        pins,
        expected_source_commit=expected_source_commit,
        expected_frozen_plan_sha256=expected_frozen_plan_sha256,
    )
    if plan.sha256 != expected_handoff_sha256:
        raise ValueError("recovery handoff digest mismatch")
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


def _object(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("recovery metadata requires a JSON object")
    return dict(value)


def _array(value: Any) -> list[Any]:
    if type(value) is not list:
        raise ValueError("recovery metadata requires a JSON array")
    return value


def _decode_metadata(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > _MAX_METADATA_BYTES:
        raise ValueError("recovery metadata exceeds byte limit or is empty")
    try:
        value = _object(json.loads(raw.decode("utf-8")))
        # Noncanonical whitespace, duplicate keys and nonfinite numbers cannot
        # round-trip. Only the exact canonical format produced by stage is read.
        if canonical_json_bytes(value) != raw:
            raise ValueError("recovery metadata is not canonical")
    except (UnicodeError, RecursionError, TypeError, OverflowError) as exc:
        raise ValueError("invalid recovery metadata") from exc
    return value


def _parse_arm(value: object) -> ArmReceipt:
    data = _object(value)
    report = data["external_report"]
    if report is not None:
        fields = _object(report)
        fields.pop("binding_sha256")
        data["external_report"] = ExternalArmVerificationReport(**fields)
    progress = data["progress"]
    if progress is not None:
        data["progress"] = ArmAttemptProgress(**_object(progress))
    arm = ArmReceipt(**data)
    if canonical_json_bytes(arm.to_mapping()) != canonical_json_bytes(value):
        raise ValueError("arm receipt fields or binding digest differ")
    return arm


def _parse_run(value: dict[str, Any]) -> RunReceipt:
    data = dict(value)
    if data.pop("schema") != "task18_run_receipt_v3":
        raise ValueError("unsupported stopped-run schema")
    binding = _object(data["source_binding"])
    binding.pop("manifest_sha256")
    binding["files"] = tuple(
        SourceFilePin(**_object(item)) for item in _array(binding["files"])
    )
    data["source_binding"] = SourceBinding(**binding)
    report = data["source_report"]
    if report is not None:
        fields = _object(report)
        fields.pop("binding_sha256")
        fields["source_files"] = tuple(
            SourceFilePin(**_object(item)) for item in _array(fields["source_files"])
        )
        data["source_report"] = ExternalSourceVerificationReport(**fields)
    data["arms"] = tuple(_parse_arm(item) for item in _array(data["arms"]))
    run = RunReceipt(**data)
    if canonical_json_bytes(run.to_mapping()) != canonical_json_bytes(value):
        raise ValueError("stopped-run fields or binding digests differ")
    return run


def _read_pinned_metadata(root: Path, pin: RecoveryFilePin) -> dict[str, Any]:
    if pin.size_bytes > _MAX_METADATA_BYTES:
        raise ValueError("recovery metadata exceeds byte limit")
    raw = read_verified_bytes(
        root / pin.path,
        expected_digest=pin.sha256,
        expected_size_bytes=pin.size_bytes,
        field=pin.path,
    )
    return _decode_metadata(raw)


def _bundle_pins(
    pins: tuple[RecoveryFilePin, ...],
) -> tuple[dict[str, RecoveryFilePin], tuple[str, ...]]:
    protocol = build_task18_protocol()
    if (
        type(pins) is not tuple
        or len(pins) > 3 + 4 * len(protocol.arms)
        or any(type(pin) is not RecoveryFilePin for pin in pins)
    ):
        raise ValueError("invalid recovery file pin roster")
    by_path = {pin.path: pin for pin in pins}
    paths = {"frozen-plan.bin", "source-verification.json", "stopped-run.json"}
    receipts: list[str] = []
    for arm in protocol.arms:
        prefix = f"arms/{arm.arm_id}/"
        receipt_path = prefix + "receipt.json"
        if receipt_path in by_path:
            receipts.append(receipt_path)
            paths.update(
                prefix + name
                for name in (
                    "receipt.json",
                    "native-ledger.json",
                    "result.json",
                    "verification.json",
                )
            )
    # No supplied path is used for I/O until it matches the recipe-derived roster.
    if len(by_path) != len(pins) or by_path.keys() != paths:
        raise ValueError(
            "recovery handoff contains missing, duplicate or foreign file paths"
        )
    return by_path, tuple(receipts)


def _read_bundle_plan(
    root: Path,
    pins: tuple[RecoveryFilePin, ...],
    *,
    expected_source_commit: str,
    expected_frozen_plan_sha256: str,
    manifest_pin: RecoveryFilePin | None = None,
) -> tuple[RecoveryHandoff, frozenset[str]]:
    require_git_sha(expected_source_commit)
    require_sha256(expected_frozen_plan_sha256, field="pre-stop frozen plan")
    by_path, receipt_paths = _bundle_pins(pins)
    if by_path["frozen-plan.bin"].sha256 != expected_frozen_plan_sha256:
        raise ValueError("frozen plan differs from pre-stop digest")
    _check_bundle_roster(root, pins if manifest_pin is None else (*pins, manifest_pin))
    try:
        stopped = _parse_run(_read_pinned_metadata(root, by_path["stopped-run.json"]))
        candidates = tuple(
            _parse_arm(_read_pinned_metadata(root, by_path[path]))
            for path in receipt_paths
        )
        # Bind each recovered receipt to its actual slot filename before any
        # plan is constructed, even when two files were swapped and re-pinned.
        if any(
            path != f"arms/{candidate.arm_id}/receipt.json"
            for path, candidate in zip(receipt_paths, candidates, strict=True)
        ):
            raise ValueError("recovery receipt differs from its file slot")
        plan = plan_stopped_recovery(
            stopped,
            candidates,
            pins,
            expected_source_commit=expected_source_commit,
            expected_frozen_plan_sha256=expected_frozen_plan_sha256,
        )
    except (TypeError, KeyError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError("invalid recovery handoff records") from exc
    return plan, frozenset(("stopped-run.json", *receipt_paths))


def _verify_remaining_files(
    root: Path,
    plan: RecoveryHandoff,
    metadata_paths: frozenset[str],
    manifest_pin: RecoveryFilePin | None = None,
) -> None:
    for pin in plan.files:
        if pin.path not in metadata_paths:
            _stream_pinned_file(root / pin.path, pin)
    if manifest_pin is not None:
        _stream_pinned_file(root / manifest_pin.path, manifest_pin)
    _check_bundle_roster(
        root, plan.files if manifest_pin is None else (*plan.files, manifest_pin)
    )


def inspect_stopped_recovery(
    root: Path,
    pins: tuple[RecoveryFilePin, ...],
    *,
    expected_source_commit: str,
    expected_frozen_plan_sha256: str,
) -> RecoveryHandoff:
    """Read-only inspection of an externally pinned stopped bundle.

    The exclusive stopped-attempt owner supplies the pins. No provenance,
    native ledger interpretation or execution authority is inferred from them.
    """
    root = _plain_directory(Path(root))
    plan, metadata_paths = _read_bundle_plan(
        root,
        pins,
        expected_source_commit=expected_source_commit,
        expected_frozen_plan_sha256=expected_frozen_plan_sha256,
    )
    _verify_remaining_files(root, plan, metadata_paths)
    return plan


def load_staged_recovery(
    root: Path,
    *,
    expected_source_commit: str,
    expected_frozen_plan_sha256: str,
    expected_handoff_sha256: str,
) -> RecoveryHandoff:
    """Reopen a published handoff without the original run's Python objects.

    The owner supplies independent pre-stop/source/handoff pins. Reconstruct
    the typed declarations and replay the same admission checks as stage;
    stream-verify every retained artifact before returning the pending roster.
    The stopped bundle must still be exclusive-owner controlled. This does
    not authorize an execution or authenticate a verifier producer.
    """
    require_sha256(expected_handoff_sha256, field="expected recovery handoff")
    root = _plain_directory(Path(root))
    with open_regular_binary(
        root / "recovery.json", field="recovery handoff"
    ) as reader:
        raw = reader.read(_MAX_METADATA_BYTES + 1)
    if len(raw) > _MAX_METADATA_BYTES:
        raise ValueError("recovery handoff exceeds metadata byte limit")
    if hashlib.sha256(raw).hexdigest() != expected_handoff_sha256:
        raise ValueError("recovery handoff digest mismatch")
    manifest_pin = RecoveryFilePin("recovery.json", expected_handoff_sha256, len(raw))
    try:
        data = _decode_metadata(raw)
        pins = tuple(RecoveryFilePin(**_object(item)) for item in _array(data["files"]))
    except (TypeError, KeyError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError("invalid recovery handoff records") from exc
    plan, metadata_paths = _read_bundle_plan(
        root,
        pins,
        expected_source_commit=expected_source_commit,
        expected_frozen_plan_sha256=expected_frozen_plan_sha256,
        manifest_pin=manifest_pin,
    )
    # The complete canonical declaration, including all roster fields and the
    # non-authority flag, must match the independently reconstructed plan.
    if plan.canonical_bytes() != raw:
        raise ValueError("reconstructed handoff does not match published manifest")
    _verify_remaining_files(root, plan, metadata_paths, manifest_pin)
    return plan


__all__ = [
    "RecoveryFilePin",
    "RecoveryHandoff",
    "inspect_stopped_recovery",
    "load_staged_recovery",
    "plan_stopped_recovery",
    "stage_stopped_recovery",
]
