"""Metadata-only adversarial lineage tests; native E2E is a separate witness."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from tools.tier_d_nonrl.task18_frozen_inputs import (
    _ARRAY_BYTES,
    _FRAME,
    FrozenInputsPin,
)
from tools.tier_d_nonrl.task18_native_lineage import (
    MAX_GENERATIONS,
    StagedSyntheticGeneration,
    lineage_bytes,
    ordered_union,
    read_generation,
    remaining,
    require_pending_subset,
    validate_generation_paths,
    verify_successor_lineage,
)
from tools.tier_d_nonrl.task18_receipts import (
    ArmReceipt,
    ExternalSourceVerificationReport,
    RunReceipt,
    SourceBinding,
    SourceFilePin,
)
from tools.tier_d_nonrl.task18_recipe import build_task18_protocol
from tools.tier_d_nonrl.task18_recovery import RecoveryHandoff
from trade_rl.artifacts.canonical import canonical_json_bytes

ARM_IDS = tuple(a.arm_id for a in build_task18_protocol().arms)


@pytest.fixture
def link():
    source = SourceBinding(
        "a" * 40, (SourceFilePin("tools/artificial.py", "b" * 64),), "c" * 64
    )
    report = ExternalSourceVerificationReport(
        source.commit_sha, True, source.files, "c" * 64, 1, "d" * 64
    )
    digest = "e" * 64
    root = RunReceipt(
        "artificial-root",
        digest,
        "f" * 64,
        "1" * 64,
        "2" * 64,
        "3" * 64,
        source,
        report,
        False,
        tuple(
            ArmReceipt(
                a.arm_id,
                a.pair_id,
                a.role,
                a.configuration,
                a.symbol,
                a.scenario,
                digest,
                "1" * 64,
                "2" * 64,
                "3" * 64,
                source.manifest_sha256,
                "NOT_ATTEMPTED",
                None,
                None,
                None,
                None,
                None,
                reason_code="NOT_STARTED",
            )
            for a in build_task18_protocol().arms
        ),
    )
    predecessor = RecoveryHandoff(
        root.run_id, source.commit_sha, digest, (ARM_IDS[0],), (), ARM_IDS[1:], ()
    )
    child = replace(
        root, run_id=f"task18-synthetic-successor:{predecessor.sha256}:artificial-next"
    )
    child = replace(
        child,
        arms=tuple(
            replace(a, status="RUNNING", reason_code="RUN_IN_PROGRESS")
            if a.arm_id == ARM_IDS[1]
            else a
            for a in child.arms
        ),
    )
    current = RecoveryHandoff(
        child.run_id,
        source.commit_sha,
        digest,
        (ARM_IDS[1],),
        (ARM_IDS[1],),
        tuple(a for a in ARM_IDS if a != ARM_IDS[1]),
        (),
    )
    pin = FrozenInputsPin("9" * 64, _ARRAY_BYTES + _FRAME.size + 1)
    kwargs = dict(
        predecessor=predecessor,
        predecessor_run=root,
        ancestor_handoff_sha256s=(predecessor.sha256,),
        retained=(ARM_IDS[0],),
        stopped=child,
        current=current,
        plan_pin=pin,
    )
    raw = lineage_bytes(
        predecessor,
        (predecessor.sha256,),
        (ARM_IDS[0],),
        (ARM_IDS[1],),
        child.run_id,
        pin,
    )
    return raw, kwargs


def test_lineage_reconstructs_canonical_expected_link(link):
    raw, kwargs = link
    verify_successor_lineage(raw, **kwargs)


@pytest.mark.parametrize(
    "field",
    (
        "ancestor_handoff_sha256s",
        "retained_arm_ids",
        "selected_arm_ids",
        "original_run_id",
        "successor_run_id",
        "predecessor_handoff_sha256",
        "protocol_digest",
        "source_commit",
        "plan_pin",
        "execution_authorized",
        "schema",
        "unknown",
    ),
)
def test_resealed_lineage_does_not_authorize_different_frontier(link, field):
    raw, kwargs = link
    payload = json.loads(raw)
    payload[field] = {
        "ancestor_handoff_sha256s": [],
        "retained_arm_ids": [],
        "selected_arm_ids": [ARM_IDS[0]],
        "original_run_id": "foreign",
        "successor_run_id": "foreign",
        "predecessor_handoff_sha256": "0" * 64,
        "protocol_digest": "0" * 64,
        "source_commit": "0" * 40,
        "plan_pin": {"sha256": "0" * 64, "size_bytes": 1},
        "execution_authorized": True,
        "schema": "historical",
        "unknown": 0,
    }[field]
    with pytest.raises(ValueError):
        verify_successor_lineage(canonical_json_bytes(payload), **kwargs)


@pytest.mark.parametrize(
    "change",
    (
        "source_report",
        "foreign_protocol",
        "foreign_run",
        "extra_completed",
        "unselected_running",
        "unselected_progress",
    ),
)
def test_successor_local_evidence_cannot_escape_selection(link, change):
    from tools.tier_d_nonrl.task18_receipts import ArmAttemptProgress

    raw, kwargs = link
    stopped, current = kwargs["stopped"], kwargs["current"]
    if change == "source_report":
        kwargs["stopped"] = replace(
            stopped,
            source_report=replace(
                stopped.source_report, verification_report_sha256="7" * 64
            ),
        )
    elif change == "foreign_protocol":
        kwargs["current"] = replace(current, protocol_digest="7" * 64)
    elif change == "foreign_run":
        kwargs["current"] = replace(current, original_run_id="foreign")
    elif change == "extra_completed":
        kwargs["current"] = replace(current, completed_arm_ids=(ARM_IDS[0], ARM_IDS[1]))
    else:
        old = stopped.arms[0]
        altered = (
            replace(old, status="RUNNING", reason_code="RUN_IN_PROGRESS")
            if change == "unselected_running"
            else replace(old, progress=ArmAttemptProgress(native_advances_observed=1))
        )
        kwargs["stopped"] = replace(stopped, arms=(altered, *stopped.arms[1:]))
    with pytest.raises(ValueError):
        verify_successor_lineage(raw, **kwargs)


def test_all_48_completed_slots_are_unique_and_canonical():
    completed = ordered_union(ARM_IDS[::2], ARM_IDS[1::2])
    assert completed == ARM_IDS and not remaining(completed)
    with pytest.raises(ValueError, match="duplicate"):
        ordered_union(completed, (ARM_IDS[0],))
    with pytest.raises(ValueError, match="foreign"):
        ordered_union(("foreign",))
    for bad in ((), (ARM_IDS[1], ARM_IDS[0]), (ARM_IDS[1], ARM_IDS[1]), ("foreign",)):
        with pytest.raises(ValueError, match="pending"):
            require_pending_subset(bad, ARM_IDS)
    with pytest.raises(ValueError, match="pending"):
        require_pending_subset((ARM_IDS[0],), remaining(completed))


def test_generation_path_closure_and_bounded_ancestor_roster(tmp_path):
    base = tmp_path.resolve()
    a, b, c = (base / name for name in ("a", "b", "c"))
    for path in (a, b, c):
        path.mkdir()
    first = StagedSyntheticGeneration(a, "1" * 64)
    second = StagedSyntheticGeneration(b, "2" * 64, "3" * 64)
    validate_generation_paths((first, second), c, "4" * 64)
    for chain, lineage in (
        ((), "4" * 64),
        ((second, first), "4" * 64),
        ((first,), "4" * 64),
    ):
        root = c if len(chain) != 1 else a
        with pytest.raises(ValueError):
            validate_generation_paths(chain, root, lineage)
    with pytest.raises(ValueError, match="bounded"):
        validate_generation_paths((first,) * MAX_GENERATIONS, c, "4" * 64)
    with pytest.raises(ValueError, match="duplicate"):
        validate_generation_paths(
            (first, replace(second, handoff_sha256=first.handoff_sha256)), c, "4" * 64
        )


def test_lineage_pin_before_decode_and_exact_directory_roster(tmp_path):
    root = tmp_path.resolve()
    (root / "attempt").mkdir()
    raw = b"not json; only the verified byte reader runs here"
    (root / "lineage.json").write_bytes(raw)
    with pytest.raises(ValueError, match="digest mismatch"):
        read_generation(root, "0" * 64)
    assert read_generation(root, hashlib.sha256(raw).hexdigest()) == (
        root / "attempt",
        raw,
    )
    (root / "foreign").write_bytes(b"x")
    with pytest.raises(ValueError, match="roster"):
        read_generation(root, hashlib.sha256(raw).hexdigest())
