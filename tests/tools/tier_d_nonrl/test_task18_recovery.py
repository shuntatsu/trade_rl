from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from tools.tier_d_nonrl.task18_receipts import (
    ArmReceipt,
    ExternalArmVerificationReport,
    ExternalSourceVerificationReport,
    RunReceipt,
    SourceBinding,
    SourceFilePin,
)
from tools.tier_d_nonrl.task18_recipe import build_task18_protocol
from tools.tier_d_nonrl.task18_recovery import (
    RecoveryFilePin,
    plan_stopped_recovery,
    stage_stopped_recovery,
)
from trade_rl.artifacts.canonical import canonical_json_bytes


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _write(root: Path, path: str, raw: bytes) -> RecoveryFilePin:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return RecoveryFilePin(path, _sha(raw), len(raw))


def _bundle(root: Path):
    # Every byte is artificial. The reported ledger verification is a fixture,
    # not a native ledger verifier or authorization to run Task18.
    protocol = build_task18_protocol()
    binding = SourceBinding(
        "a" * 40, (SourceFilePin("tools/runner.py", "b" * 64),), "c" * 64
    )
    source_bytes = b"synthetic source verifier"
    source_report = ExternalSourceVerificationReport(
        binding.commit_sha,
        True,
        binding.files,
        "c" * 64,
        1,
        _sha(source_bytes),
    )
    ref = protocol.historical_reference
    pins = [
        _write(root, "frozen-plan.bin", b"synthetic frozen inputs"),
        _write(root, "source-verification.json", source_bytes),
    ]
    arms = []
    completed = []
    last_nonrl = protocol.arms[-2].arm_id
    last_cash = protocol.arms[-1].arm_id
    for spec in protocol.arms:
        arm = ArmReceipt(
            arm_id=spec.arm_id,
            pair_id=spec.pair_id,
            role=spec.role,
            configuration=spec.configuration,
            symbol=spec.symbol,
            scenario=spec.scenario,
            protocol_digest=protocol.adapted_protocol_digest,
            dataset_id=ref.dataset_id,
            manifest_sha256=ref.manifest_sha256,
            arrays_sha256=ref.arrays_sha256,
            source_binding_sha256=binding.manifest_sha256,
            status="NOT_ATTEMPTED",
            initial_equity_usdt=None,
            terminal_equity_usdt=None,
            max_drawdown=None,
            terminal_flat=None,
            pending_order_count=None,
            reason_code="NOT_STARTED",
        )
        if spec.arm_id == last_cash:
            arms.append(arm)
            continue
        prefix = f"arms/{arm.arm_id}/"
        ledger = _write(
            root, prefix + "native-ledger.json", f"ledger:{arm.arm_id}".encode()
        )
        result = _write(root, prefix + "result.json", f"result:{arm.arm_id}".encode())
        verification = _write(
            root, prefix + "verification.json", f"verified:{arm.arm_id}".encode()
        )
        evidence = ExternalArmVerificationReport(
            run_id="stopped-synthetic",
            protocol_digest=arm.protocol_digest,
            source_report_binding_sha256=source_report.binding_sha256,
            arm_id=arm.arm_id,
            pair_id=arm.pair_id,
            role=arm.role,
            configuration=arm.configuration,
            symbol=arm.symbol,
            scenario=arm.scenario,
            dataset_id=arm.dataset_id,
            native_ledger_sha256=ledger.sha256,
            arm_result_sha256=result.sha256,
            verification_report_sha256=verification.sha256,
            expected_advances=protocol.index_plan.native_advances_per_arm,
            observed_advances=protocol.index_plan.native_advances_per_arm,
            initial_equity_usdt=protocol.economics.initial_capital_per_symbol_usdt,
            terminal_equity_usdt=protocol.economics.initial_capital_per_symbol_usdt,
            max_drawdown=0.0,
            terminal_settlement_complete=True,
            terminal_flat=True,
            pending_order_count=0,
            cost_oracle_passed=True,
        )
        complete = replace(
            arm,
            status="COMPLETED",
            reason_code=None,
            initial_equity_usdt=evidence.initial_equity_usdt,
            terminal_equity_usdt=evidence.terminal_equity_usdt,
            max_drawdown=0.0,
            terminal_flat=True,
            pending_order_count=0,
            native_ledger_sha256=ledger.sha256,
            arm_result_sha256=result.sha256,
            external_report=evidence,
        )
        completed.append(complete)
        pins.extend(
            (
                ledger,
                result,
                verification,
                _write(
                    root,
                    prefix + "receipt.json",
                    canonical_json_bytes(complete.to_mapping()),
                ),
            )
        )
        arms.append(
            replace(arm, status="RUNNING", reason_code="RUN_IN_PROGRESS")
            if spec.arm_id == last_nonrl
            else complete
        )
    stopped = RunReceipt(
        run_id="stopped-synthetic",
        protocol_digest=protocol.adapted_protocol_digest,
        historical_protocol_digest=ref.protocol_digest,
        dataset_id=ref.dataset_id,
        manifest_sha256=ref.manifest_sha256,
        arrays_sha256=ref.arrays_sha256,
        source_binding=binding,
        source_report=source_report,
        runtime_completed=False,
        arms=tuple(arms),
    )
    pins.append(
        _write(root, "stopped-run.json", canonical_json_bytes(stopped.to_mapping()))
    )
    return stopped, tuple(completed), tuple(pins)


@pytest.fixture
def bundle(tmp_path):
    base = tmp_path.resolve()
    source = base / "stopped"
    source.mkdir()
    stopped, candidates, pins = _bundle(source)
    return source, base / "successor", stopped, candidates, pins


def _plan(stopped, candidates, pins):
    return plan_stopped_recovery(
        stopped,
        candidates,
        pins,
        expected_source_commit="a" * 40,
        expected_frozen_plan_sha256=_sha(b"synthetic frozen inputs"),
    )


def _stage(bundle, *, digest=None):
    source, destination, stopped, candidates, pins = bundle
    plan = _plan(stopped, candidates, pins)
    return stage_stopped_recovery(
        source,
        destination,
        pins,
        expected_source_commit="a" * 40,
        expected_frozen_plan_sha256=_sha(b"synthetic frozen inputs"),
        expected_handoff_sha256=plan.sha256 if digest is None else digest,
    )


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_stopped_nonrl_is_recovered_and_only_missing_cash_is_listed(bundle):
    source, destination, stopped, candidates, pins = bundle
    before = _snapshot(source)
    plan = _stage(bundle)
    assert plan.completed_arm_ids == tuple(a.arm_id for a in candidates)
    assert plan.recovered_arm_ids == (stopped.arms[-2].arm_id,)
    assert plan.pending_arm_ids == (stopped.arms[-1].arm_id,)
    assert plan.original_run_id == stopped.run_id
    assert plan.execution_authorized is False
    assert _snapshot(source) == before
    after = _snapshot(destination)
    assert after.pop("recovery.json") == plan.canonical_bytes()
    assert after == before
    assert _sha(plan.canonical_bytes()) == plan.sha256
    # A new destination is exclusive even for an identical retry.
    with pytest.raises(FileExistsError):
        _stage(bundle)


@pytest.mark.parametrize(
    "failure",
    (
        "duplicate",
        "missing_completed",
        "foreign_run",
        "not_attempted",
        "changed_completed",
        "incomplete_settlement",
        "missing_pin",
        "duplicate_pin",
        "failed_arm",
        "foreign_arm",
        "extra_pin",
        "wrong_ledger_pin",
        "wrong_source",
        "malformed_partial",
        "runtime_complete",
    ),
)
def test_refuses_inconsistent_recovery_claims_before_publication(bundle, failure):
    source, destination, stopped, candidates, pins = bundle
    last = candidates[-1]
    if failure == "duplicate":
        candidates += (last,)
    elif failure == "missing_completed":
        candidates = candidates[1:]
    elif failure == "foreign_run":
        candidates = candidates[:-1] + (
            replace(
                last, external_report=replace(last.external_report, run_id="foreign")
            ),
        )
    elif failure == "not_attempted":
        stopped = replace(
            stopped,
            arms=stopped.arms[:-2]
            + (
                replace(
                    stopped.arms[-2], status="NOT_ATTEMPTED", reason_code="NOT_STARTED"
                ),
                stopped.arms[-1],
            ),
        )
    elif failure == "changed_completed":
        first = candidates[0]
        candidates = (
            replace(first, terminal_equity_usdt=first.terminal_equity_usdt + 1),
        ) + candidates[1:]
    elif failure == "incomplete_settlement":
        candidates = candidates[:-1] + (
            replace(
                last,
                external_report=replace(
                    last.external_report, terminal_settlement_complete=False
                ),
            ),
        )
    elif failure == "failed_arm":
        stopped = replace(
            stopped,
            arms=stopped.arms[:-2]
            + (
                replace(
                    stopped.arms[-2],
                    status="FAILED",
                    reason_code="PROCESS_FAILED",
                    error_code="synthetic",
                ),
                stopped.arms[-1],
            ),
        )
    elif failure == "foreign_arm":
        candidates = candidates[:-1] + (replace(last, arm_id="PAIR-99-NONRL"),)
    elif failure == "duplicate_pin":
        pins += (pins[0],)
    elif failure == "missing_pin":
        pins = pins[:-1]
    elif failure == "extra_pin":
        pins += (RecoveryFilePin("foreign.json", "f" * 64, 1),)
    elif failure == "wrong_ledger_pin":
        pins = tuple(
            replace(p, sha256="f" * 64) if p.path.endswith("native-ledger.json") else p
            for p in pins
        )
    elif failure == "wrong_source":
        with pytest.raises(ValueError):
            plan_stopped_recovery(
                stopped,
                candidates,
                pins,
                expected_source_commit="f" * 40,
                expected_frozen_plan_sha256=_sha(b"synthetic frozen inputs"),
            )
        return
    elif failure == "malformed_partial":
        stopped = replace(
            stopped,
            arms=stopped.arms[:-2]
            + (replace(stopped.arms[-2], initial_equity_usdt=True), stopped.arms[-1]),
        )
    elif failure == "runtime_complete":
        stopped = replace(stopped, runtime_completed=True)
    # Refuse typed claims, not merely a file mismatch. For mutations of
    # the stopped declaration update its byte pin so that this check cannot
    # succeed solely by detecting the old stopped-run digest.
    raw = canonical_json_bytes(stopped.to_mapping())
    pins = tuple(
        replace(p, sha256=_sha(raw), size_bytes=len(raw))
        if p.path == "stopped-run.json"
        else p
        for p in pins
    )
    # Re-pin changed candidate bytes too; inconsistent report/identity
    # must be refused even if the caller re-seals that changed receipt.
    changed = {
        f"arms/{a.arm_id}/receipt.json": canonical_json_bytes(a.to_mapping())
        for a in candidates
    }
    pins = tuple(
        replace(p, sha256=_sha(changed[p.path]), size_bytes=len(changed[p.path]))
        if p.path in changed
        else p
        for p in pins
    )
    with pytest.raises(ValueError):
        _plan(stopped, candidates, pins)
    assert not destination.exists()


@pytest.mark.parametrize(
    "failure", ("tamper", "oversize", "missing", "extra", "link", "wrong_digest")
)
def test_file_failures_never_publish_partial_destination(bundle, failure):
    source, destination, stopped, candidates, pins = bundle
    path = source / f"arms/{candidates[-1].arm_id}/native-ledger.json"
    digest = None
    if failure == "tamper":
        raw = path.read_bytes()
        path.write_bytes(b"x" + raw[1:])
    elif failure == "oversize":
        path.write_bytes(path.read_bytes() + b"x")
    elif failure == "missing":
        path.unlink()
    elif failure == "extra":
        (source / "arms" / "unaccounted.json").write_bytes(b"unexpected")
    elif failure == "link":
        external = source.parent / "foreign"
        external.write_bytes(path.read_bytes())
        path.unlink()
        try:
            path.symlink_to(external)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation unsupported")
    else:
        digest = "f" * 64
    before = _snapshot(source)
    with pytest.raises((ValueError, OSError)):
        _stage(bundle, digest=digest)
    assert not destination.exists()
    assert not list(destination.parent.glob(".successor.staging-*"))
    assert _snapshot(source) == before


def test_pre_stop_frozen_plan_pin_cannot_be_resealed_after_substitution(bundle):
    source, destination, stopped, candidates, pins = bundle
    original_pin = next(p for p in pins if p.path == "frozen-plan.bin")
    replaced = _write(source, "frozen-plan.bin", b"substituted frozen inputs")
    pins = tuple(replaced if p.path == original_pin.path else p for p in pins)
    with pytest.raises(ValueError, match="frozen plan"):
        plan_stopped_recovery(
            stopped,
            candidates,
            pins,
            expected_source_commit="a" * 40,
            expected_frozen_plan_sha256=original_pin.sha256,
        )
    assert not destination.exists()


def test_roster_order_is_canonical_and_interrupted_completion_is_recoverable(bundle):
    _, _, stopped, candidates, pins = bundle
    arms = stopped.arms[:-2] + (
        replace(
            stopped.arms[-2], status="INTERRUPTED", reason_code="PROCESS_INTERRUPTED"
        ),
        stopped.arms[-1],
    )
    stopped = replace(stopped, arms=arms)
    raw = canonical_json_bytes(stopped.to_mapping())
    pins = tuple(
        replace(p, sha256=_sha(raw), size_bytes=len(raw))
        if p.path == "stopped-run.json"
        else p
        for p in pins
    )
    forward = _plan(stopped, candidates, pins)
    reverse = _plan(stopped, tuple(reversed(candidates)), tuple(reversed(pins)))
    assert forward.canonical_bytes() == reverse.canonical_bytes()
    assert forward.recovered_arm_ids == (stopped.arms[-2].arm_id,)
    assert forward.pending_arm_ids == (stopped.arms[-1].arm_id,)


def test_all_partial_run_has_no_invented_receipt(bundle):
    source, destination, stopped, _, _ = bundle
    source = source.parent / "all-partial"
    source.mkdir()
    arms = tuple(
        replace(
            a,
            status="NOT_ATTEMPTED",
            initial_equity_usdt=None,
            terminal_equity_usdt=None,
            max_drawdown=None,
            terminal_flat=None,
            pending_order_count=None,
            native_ledger_sha256=None,
            arm_result_sha256=None,
            external_report=None,
            reason_code="NOT_STARTED",
        )
        for a in stopped.arms
    )
    stopped = replace(stopped, arms=arms)
    pins = (
        _write(source, "frozen-plan.bin", b"synthetic frozen inputs"),
        _write(source, "source-verification.json", b"synthetic source verifier"),
        _write(source, "stopped-run.json", canonical_json_bytes(stopped.to_mapping())),
    )
    plan = _stage((source, destination, stopped, (), pins))
    assert plan.completed_arm_ids == plan.recovered_arm_ids == ()
    assert plan.pending_arm_ids == tuple(a.arm_id for a in arms)
    assert not (destination / "arms").exists()


def test_destination_inside_stopped_bundle_is_refused(bundle):
    source, _, stopped, candidates, pins = bundle
    before = _snapshot(source)
    with pytest.raises(ValueError, match="outside"):
        _stage((source, source / "new-attempt", stopped, candidates, pins))
    assert _snapshot(source) == before
    assert not (source / "new-attempt").exists()


def test_concurrent_publication_never_replaces_winner(bundle, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from tools.tier_d_nonrl import task18_recovery as recovery

    source, destination, _, _, _ = bundle
    before = _snapshot(source)
    barrier = Barrier(2)
    publish = recovery.atomic_rename_directory

    def synchronized_publish(staging, target):
        barrier.wait(timeout=15)
        return publish(staging, target)

    monkeypatch.setattr(recovery, "atomic_rename_directory", synchronized_publish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_stage, bundle) for _ in range(2)]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=25))
            except FileExistsError:
                outcomes.append(None)
    winners = [outcome for outcome in outcomes if outcome is not None]
    assert len(winners) == 1
    assert _snapshot(source) == before
    after = _snapshot(destination)
    assert after.pop("recovery.json") == winners[0].canonical_bytes()
    assert after == before
    assert not list(destination.parent.glob(".successor.staging-*"))


def test_symlinked_bundle_directory_refused_before_descendant_enumeration(
    bundle, monkeypatch
):
    source, destination, _, _, _ = bundle
    arms = source / "arms"
    foreign = source.parent / "foreign-arms"
    arms.rename(foreign)
    try:
        arms.symlink_to(foreign, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unsupported")
    original = Path.iterdir

    def guarded_iterdir(path):
        if path.is_relative_to(arms):
            raise AssertionError("must not enumerate a symlinked directory descendant")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    with pytest.raises(ValueError, match="directory"):
        _stage(bundle)
    assert not destination.exists()


def _reopen(destination: Path, expected_digest: str):
    from tools.tier_d_nonrl.task18_recovery import load_staged_recovery

    return load_staged_recovery(
        destination,
        expected_handoff_sha256=expected_digest,
        expected_source_commit="a" * 40,
        expected_frozen_plan_sha256=_sha(b"synthetic frozen inputs"),
    )


def test_reopen_published_handoff_without_original_python_objects(bundle):
    source, destination, _, _, _ = bundle
    expected = _stage(bundle)
    # The original attempt is not consulted by a fresh process.
    source.rename(source.with_name("archived"))
    before = _snapshot(destination)
    reopened = _reopen(destination, expected.sha256)
    assert reopened.canonical_bytes() == expected.canonical_bytes()
    assert reopened.pending_arm_ids == expected.pending_arm_ids
    assert reopened.execution_authorized is False
    assert _snapshot(destination) == before


@pytest.mark.parametrize(
    "failure",
    (
        "digest",
        "ledger",
        "missing",
        "foreign",
        "frozen",
        "receipt",
        "noncanonical",
        "duplicate_json",
        "foreign_path",
        "changed_roster",
        "receipt_report",
        "unknown_field",
        "oversized_metadata",
    ),
)
def test_reopen_refuses_mutated_published_handoff(bundle, failure):
    import json

    source, destination, _, candidates, _ = bundle
    expected = _stage(bundle)
    before_source = _snapshot(source)
    manifest = destination / "recovery.json"
    payload = json.loads(manifest.read_bytes())
    expected_digest = expected.sha256
    last = candidates[-1].arm_id
    ledger = destination / f"arms/{last}/native-ledger.json"
    receipt = destination / f"arms/{last}/receipt.json"

    def repin(path: Path, raw: bytes) -> None:
        path.write_bytes(raw)
        relative = path.relative_to(destination).as_posix()
        pin = next(p for p in payload["files"] if p["path"] == relative)
        pin.update(sha256=_sha(raw), size_bytes=len(raw))

    if failure == "digest":
        expected_digest = "f" * 64
    elif failure == "ledger":
        ledger.write_bytes(b"bad")
    elif failure == "missing":
        receipt.unlink()
    elif failure == "foreign":
        (destination / "foreign").write_bytes(b"foreign")
    elif failure == "frozen":
        repin(destination / "frozen-plan.bin", b"modified frozen plan")
    elif failure == "receipt":
        receipt.write_bytes(b"tampered")
    elif failure == "noncanonical":
        raw = manifest.read_bytes() + b"\n"
        manifest.write_bytes(raw)
        expected_digest = _sha(raw)
    elif failure == "duplicate_json":
        raw = manifest.read_bytes().replace(
            b'"execution_authorized":false',
            b'"execution_authorized":true,"execution_authorized":false',
        )
        manifest.write_bytes(raw)
        expected_digest = _sha(raw)
    elif failure == "foreign_path":
        payload["files"][0]["path"] = "../outside"
    elif failure == "changed_roster":
        payload["completed_arm_ids"].pop()
        payload["pending_arm_ids"].insert(0, last)
    elif failure == "receipt_report":
        record = json.loads(receipt.read_bytes())
        record["external_report"]["run_id"] = "foreign"
        # Recalculate its self-declared binding as well, so the reader must
        # actually reject the cross-run report, not only its checksum.
        report = record["external_report"]
        report.pop("binding_sha256")
        report["binding_sha256"] = _sha(
            json.dumps(
                report,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode()
        )
        repin(receipt, canonical_json_bytes(record))
    elif failure == "unknown_field":
        payload["unexpected"] = True
    elif failure == "oversized_metadata":
        raw = b" " * (4 * 1024 * 1024 + 1)
        repin(receipt, raw)
    if failure in {
        "frozen",
        "foreign_path",
        "changed_roster",
        "receipt_report",
        "unknown_field",
        "oversized_metadata",
    }:
        raw = canonical_json_bytes(payload)
        manifest.write_bytes(raw)
        expected_digest = _sha(raw)
    before = _snapshot(destination)
    with pytest.raises((ValueError, OSError)):
        _reopen(destination, expected_digest)
    assert _snapshot(destination) == before
    assert _snapshot(source) == before_source


def test_inspect_reads_disk_without_run_objects_and_refuses_swapped_slot(bundle):
    from tools.tier_d_nonrl.task18_recovery import inspect_stopped_recovery

    source, _, stopped, candidates, pins = bundle
    expected = _plan(stopped, candidates, pins)
    before = _snapshot(source)
    actual = inspect_stopped_recovery(
        source,
        pins,
        expected_source_commit="a" * 40,
        expected_frozen_plan_sha256=_sha(b"synthetic frozen inputs"),
    )
    assert actual.canonical_bytes() == expected.canonical_bytes()
    assert _snapshot(source) == before
    first = f"arms/{candidates[0].arm_id}/receipt.json"
    second = f"arms/{candidates[1].arm_id}/receipt.json"
    swapped = {first: before[second], second: before[first]}
    for path, raw in swapped.items():
        (source / path).write_bytes(raw)
    pins = tuple(
        replace(p, sha256=_sha(swapped[p.path]), size_bytes=len(swapped[p.path]))
        if p.path in swapped
        else p
        for p in pins
    )
    with pytest.raises(ValueError, match="file slot"):
        inspect_stopped_recovery(
            source,
            pins,
            expected_source_commit="a" * 40,
            expected_frozen_plan_sha256=_sha(b"synthetic frozen inputs"),
        )


def test_staged_handoff_never_grants_execution_authority_even_if_resealed(bundle):
    import json

    _, destination, _, _, _ = bundle
    _stage(bundle)
    path = destination / "recovery.json"
    payload = json.loads(path.read_bytes())
    payload["execution_authorized"] = True
    raw = canonical_json_bytes(payload)
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="reconstructed handoff"):
        _reopen(destination, _sha(raw))
