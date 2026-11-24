"""Exact synthetic generation-chain declarations, not execution authority.

Every generation keeps its own immutable receipts. A successor is admitted
only with the externally pinned ordered predecessor handoffs and its original
lineage bytes. The caller must own one non-forking history; these local records
are not a distributed single-writer lease or producer authentication.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from tools.tier_d_nonrl.task18_frozen_inputs import FrozenInputsPin
from tools.tier_d_nonrl.task18_native_artifact import _MAX_METADATA
from tools.tier_d_nonrl.task18_native_decode import parse_json
from tools.tier_d_nonrl.task18_receipts import RunReceipt
from tools.tier_d_nonrl.task18_recipe import build_task18_protocol
from tools.tier_d_nonrl.task18_recovery import RecoveryHandoff, _plain_directory
from trade_rl._validation import require_sha256
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.verified_file import open_regular_binary

SUCCESSOR_PREFIX = "task18-synthetic-successor:"
MAX_GENERATIONS = 64
_ARM_IDS = tuple(a.arm_id for a in build_task18_protocol().arms)


@dataclass(frozen=True, slots=True)
class StagedSyntheticGeneration:
    """Owner-supplied external pins and location of one staged generation."""

    root: Path
    handoff_sha256: str
    lineage_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path) or ".." in self.root.parts:
            raise ValueError("generation root must be a non-traversing Path")
        require_sha256(self.handoff_sha256, field="generation handoff digest")
        if self.lineage_sha256 is not None:
            require_sha256(self.lineage_sha256, field="generation lineage digest")


def validate_generation_paths(
    ancestors: tuple[StagedSyntheticGeneration, ...],
    root: Path,
    lineage_sha256: str | None,
) -> None:
    if (
        type(ancestors) is not tuple
        or len(ancestors) >= MAX_GENERATIONS
        or any(type(pin) is not StagedSyntheticGeneration for pin in ancestors)
    ):
        raise ValueError("invalid bounded ancestor lineage roster")
    if bool(ancestors) != (lineage_sha256 is not None) or any(
        (i == 0) != (pin.lineage_sha256 is None) for i, pin in enumerate(ancestors)
    ):
        raise ValueError("successor requires complete ordered ancestor lineage pins")
    digests = tuple(pin.handoff_sha256 for pin in ancestors)
    if len(digests) != len(set(digests)):
        raise ValueError("duplicate ancestor handoff digest")
    roots = tuple(_plain_directory(pin.root) for pin in ancestors) + (
        _plain_directory(root),
    )
    for i, left in enumerate(roots):
        if any(
            left.is_relative_to(right) or right.is_relative_to(left)
            for right in roots[:i]
        ):
            raise ValueError("ancestor generation roots overlap")


def read_generation(
    root: Path, expected_lineage_sha256: str | None
) -> tuple[Path, bytes | None]:
    root = _plain_directory(root)
    if expected_lineage_sha256 is None:
        return root, None
    require_sha256(expected_lineage_sha256, field="expected lineage digest")
    if {p.name for p in root.iterdir()} != {"attempt", "lineage.json"}:
        raise ValueError("successor directory roster differs")
    with open_regular_binary(
        root / "lineage.json", field="successor lineage"
    ) as reader:
        raw = reader.read(_MAX_METADATA + 1)
    if not raw or len(raw) > _MAX_METADATA:
        raise ValueError("successor lineage exceeds metadata bound")
    if hashlib.sha256(raw).hexdigest() != expected_lineage_sha256:
        raise ValueError("successor lineage digest mismatch")
    return _plain_directory(root / "attempt"), raw


def ordered_union(*completed: tuple[str, ...]) -> tuple[str, ...]:
    merged = tuple(a for group in completed for a in group)
    if len(merged) != len(set(merged)) or any(a not in _ARM_IDS for a in merged):
        raise ValueError("duplicate or foreign completed arm in ancestor lineage")
    return tuple(a for a in _ARM_IDS if a in merged)


def remaining(completed: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(a for a in _ARM_IDS if a not in completed)


def require_pending_subset(arm_ids: tuple[str, ...], pending: tuple[str, ...]) -> None:
    if (
        type(arm_ids) is not tuple
        or not arm_ids
        or any(type(a) is not str or a not in pending for a in arm_ids)
        or len(set(arm_ids)) != len(arm_ids)
        or tuple(a for a in pending if a in arm_ids) != arm_ids
    ):
        raise ValueError(
            "successor requires an ordered nonempty subset of pending arms"
        )


def lineage_bytes(
    predecessor: RecoveryHandoff,
    ancestor_handoff_sha256s: tuple[str, ...],
    retained: tuple[str, ...],
    selected: tuple[str, ...],
    successor_run_id: str,
    plan_pin: FrozenInputsPin,
) -> bytes:
    if retained != ordered_union(retained):
        raise ValueError("retained ancestor roster is not in recipe order")
    require_pending_subset(selected, remaining(retained))
    if (
        not ancestor_handoff_sha256s
        or ancestor_handoff_sha256s[-1] != predecessor.sha256
        or len(ancestor_handoff_sha256s) >= MAX_GENERATIONS
        or len(set(ancestor_handoff_sha256s)) != len(ancestor_handoff_sha256s)
    ):
        raise ValueError("invalid successor ancestor digest frontier")
    for digest in ancestor_handoff_sha256s:
        require_sha256(digest, field="ancestor handoff digest")
    prefix = f"{SUCCESSOR_PREFIX}{predecessor.sha256}:"
    if (
        not successor_run_id.startswith(prefix)
        or not successor_run_id[len(prefix) :].strip()
    ):
        raise ValueError("successor run identity differs from ancestor lineage")
    raw = canonical_json_bytes(
        {
            "schema": "task18_synthetic_successor_lineage_v2",
            "scope": "SYNTHETIC_NATIVE_CONFORMANCE_ONLY",
            "original_run_id": predecessor.original_run_id,
            "successor_run_id": successor_run_id,
            "predecessor_handoff_sha256": predecessor.sha256,
            "ancestor_handoff_sha256s": ancestor_handoff_sha256s,
            "protocol_digest": predecessor.protocol_digest,
            "source_commit": predecessor.source_commit,
            "plan_pin": asdict(plan_pin),
            "retained_arm_ids": retained,
            "selected_arm_ids": selected,
            "execution_authorized": False,
        }
    )
    if len(raw) > _MAX_METADATA:
        raise ValueError("successor lineage exceeds metadata limit")
    return raw


def verify_successor_lineage(
    raw: bytes,
    *,
    predecessor: RecoveryHandoff,
    predecessor_run: RunReceipt,
    ancestor_handoff_sha256s: tuple[str, ...],
    retained: tuple[str, ...],
    stopped: RunReceipt,
    current: RecoveryHandoff,
    plan_pin: FrozenInputsPin,
) -> None:
    value = parse_json(raw)
    if type(value) is not dict or type(value.get("selected_arm_ids")) is not list:
        raise ValueError("invalid successor lineage declaration")
    selected = tuple(value["selected_arm_ids"])
    if raw != lineage_bytes(
        predecessor,
        ancestor_handoff_sha256s,
        retained,
        selected,
        stopped.run_id,
        plan_pin,
    ):
        raise ValueError("successor lineage differs from verified ancestor frontier")
    if (
        stopped.source_report != predecessor_run.source_report
        or stopped.source_binding != predecessor_run.source_binding
        or current.source_commit != predecessor.source_commit
    ):
        raise ValueError("successor source report or binding differs from ancestor")
    if (
        predecessor.original_run_id != predecessor_run.run_id
        or current.original_run_id != stopped.run_id
        or current.protocol_digest != predecessor.protocol_digest
        or stopped.protocol_digest != predecessor_run.protocol_digest
        or stopped.protocol_digest != current.protocol_digest
    ):
        raise ValueError("successor receipt run identity or protocol differs")
    if any(a not in selected for a in current.completed_arm_ids):
        raise ValueError("successor completion outside selected pending roster")
    if any(
        a.arm_id not in selected
        and (
            a.status != "NOT_ATTEMPTED"
            or a.reason_code != "NOT_STARTED"
            or a.progress is not None
        )
        for a in stopped.arms
    ):
        raise ValueError("successor activity outside selected pending roster")


__all__ = ["StagedSyntheticGeneration"]
