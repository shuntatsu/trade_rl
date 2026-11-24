"""Actual full-size artificial stop/reopen, with no historical recipe substitution."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.tools.tier_d_nonrl.test_task18_frozen_inputs import (
    _binding,
    _identity,
    _market,
)
from tools.tier_d_nonrl.task18_frozen_inputs import FrozenInputsPin
from tools.tier_d_nonrl.task18_receipts import (
    ExternalSourceVerificationReport,
    SourceBinding,
    SourceFilePin,
)
from tools.tier_d_nonrl.task18_recovery import RecoveryFilePin
from tools.tier_d_nonrl.tier_d_process_containment import launch_contained_process
from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.verified_file import file_digest_and_size, read_verified_bytes

_MODULE = "tests.tools.tier_d_nonrl.test_task18_native_recovery"
_ROOT = Path(__file__).resolve().parents[3]
_NONRL = "PAIR-01-NONRL"
_CASH = "PAIR-01-CASH"
_RUN = "synthetic-native-stopped"
_SUCCESSOR = "synthetic-native-successor"


def _fixture():
    original = _market()
    market = _identity(
        replace(
            original,
            dataset_id="0" * 64,
            identity_payload_json=None,
            lot_size=np.full_like(original.close, 0.001),
        )
    )
    source = SourceBinding(
        "d" * 40,
        (
            SourceFilePin(
                "tests/tools/tier_d_nonrl/test_task18_native_recovery.py",
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            ),
        ),
        hashlib.sha256(b"closed artificial fixture only").hexdigest(),
    )
    source_raw = canonical_json_bytes(
        {"scope": "closed artificial fixture only", "source_authentication": False}
    )
    report = ExternalSourceVerificationReport(
        source.commit_sha,
        True,
        source.files,
        source.import_origin_manifest_sha256,
        1,
        hashlib.sha256(source_raw).hexdigest(),
    )
    binding = replace(
        _binding(market),
        source_commit=source.commit_sha,
        source_manifest_sha256=source.manifest_sha256,
    )
    return market, binding, source, report, source_raw


def _pin(root, name):
    return RecoveryFilePin(name, *file_digest_and_size(root / name))


def _run(attempt, source, report):
    from tools.tier_d_nonrl.task18_native_artifact import new_synthetic_run_receipt

    run = new_synthetic_run_receipt(
        attempt, run_id=_RUN, source_binding=source, source_report=report
    )
    return replace(
        run,
        arms=tuple(
            replace(
                arm,
                status="RUNNING",
                reason_code="RUN_IN_PROGRESS",
                initial_equity_usdt=100000.0,
            )
            if arm.arm_id == _NONRL
            else arm
            for arm in run.arms
        ),
    )


def _publish_then_wait(base: Path, pin_data: str) -> None:
    from tools.tier_d_nonrl.task18_native import open_synthetic_native_attempt
    from tools.tier_d_nonrl.task18_native_artifact import publish_completed_native_arm

    market, binding, source, report, _ = _fixture()
    pins = json.loads(pin_data)
    root = base / "original"
    pin = FrozenInputsPin(**pins["plan"])
    stopped = RecoveryFilePin(**pins["stopped"])
    # No fit, new cost table or prediction is allowed in the stopped producer.
    with (
        patch(
            "tools.tier_d_nonrl.task18_frozen_inputs.fit_simple_return_ridge",
            side_effect=AssertionError("refit"),
        ),
        patch(
            "tools.tier_d_nonrl.task18_frozen_inputs.build_cost_rates",
            side_effect=AssertionError("recost"),
        ),
        patch(
            "trade_rl.strategies.forecasts.simple_return.SimpleReturnRidgeModel.predict_simple_return",
            side_effect=AssertionError("repredict"),
        ),
    ):
        attempt = open_synthetic_native_attempt(
            market, root / "frozen-plan.bin", expected_binding=binding, expected_pin=pin
        )
        run = _run(attempt, source, report)
        assert canonical_json_bytes(run.to_mapping()) == read_verified_bytes(
            root / stopped.path,
            expected_digest=stopped.sha256,
            expected_size_bytes=stopped.size_bytes,
            field="stopped declaration",
        )
        observed = next(attempt.iter_arms((_NONRL,)))
        assert observed.verification.passed and observed.result.book.n_trades > 0
        (root / "arms").mkdir()
        published = publish_completed_native_arm(
            root, attempt=attempt, run=run, observation=observed
        )
        assert published.receipt.status == "COMPLETED"
    atomic_write_bytes(
        base / "ready",
        canonical_json_bytes(
            {"arm_id": _NONRL, "files": [asdict(p) for p in published.pins]}
        ),
    )
    # Parent actually terminates this owned process after the receipt rename,
    # before updating its still-RUNNING slot or dispatching CASH.
    while True:
        time.sleep(0.1)


def _resume(base: Path, pin_data: str) -> None:
    from tools.tier_d_nonrl import task18_native as native
    from tools.tier_d_nonrl.task18_native_recovery import resume_synthetic_pending

    market, binding, source, _, _ = _fixture()
    pins = json.loads(pin_data)
    calls = []
    original = native.run_shared_portfolio_replay

    def checked_dispatch(dataset, policy, **kwargs):
        calls.append(policy._arm.arm_id)
        assert calls == [_CASH], "completed or unrequested arm dispatched"
        return original(dataset, policy, **kwargs)

    with (
        patch(
            "tools.tier_d_nonrl.task18_frozen_inputs.fit_simple_return_ridge",
            side_effect=AssertionError("refit"),
        ),
        patch(
            "tools.tier_d_nonrl.task18_frozen_inputs.build_cost_rates",
            side_effect=AssertionError("recost"),
        ),
        patch(
            "trade_rl.strategies.forecasts.simple_return.SimpleReturnRidgeModel.predict_simple_return",
            side_effect=AssertionError("repredict"),
        ),
        patch.object(
            native, "run_shared_portfolio_replay", side_effect=checked_dispatch
        ),
    ):
        resumed = resume_synthetic_pending(
            base / "handoff",
            base / "successor",
            dataset=market,
            expected_binding=binding,
            expected_pin=FrozenInputsPin(**pins["plan"]),
            expected_source_binding=source,
            expected_handoff_sha256=pins["handoff"],
            successor_run_label=_SUCCESSOR,
            arm_ids=(_CASH,),
        )
    assert calls == [_CASH]
    assert (
        resumed.run.run_id
        == f"task18-synthetic-successor:{pins['handoff']}:{_SUCCESSOR}"
    )
    assert resumed.predecessor_handoff_sha256 == pins["handoff"]
    assert len(resumed.completed) == 1 and resumed.completed[0].receipt.arm_id == _CASH
    assert resumed.completed[0].receipt.terminal_equity_usdt == 100000.0
    assert resumed.completed[0].verification.advances == 6599
    assert not resumed.execution_authorized


def _snapshot(root):
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_full_size_native_late_receipt_stop_recovery_and_successor(tmp_path):
    from tools.tier_d_nonrl.task18_frozen_inputs import (
        prepare_frozen_inputs,
        publish_frozen_inputs,
    )
    from tools.tier_d_nonrl.task18_native import open_synthetic_native_attempt
    from tools.tier_d_nonrl.task18_native_recovery import (
        inspect_synthetic_stopped_recovery,
        open_synthetic_staged_recovery,
        resume_synthetic_pending,
        stage_synthetic_stopped_recovery,
    )
    from tools.tier_d_nonrl.task18_recovery import inspect_stopped_recovery

    base = tmp_path.resolve()
    market, binding, source, report, source_raw = _fixture()
    frozen = prepare_frozen_inputs(market, binding=binding)
    root = base / "original"
    pin = publish_frozen_inputs(root, frozen)
    attempt = open_synthetic_native_attempt(
        market, root / "frozen-plan.bin", expected_binding=binding, expected_pin=pin
    )
    run = _run(attempt, source, report)
    atomic_write_bytes(root / "source-verification.json", source_raw)
    atomic_write_bytes(
        root / "stopped-run.json", canonical_json_bytes(run.to_mapping())
    )
    original_pins = tuple(
        _pin(root, n)
        for n in ("frozen-plan.bin", "source-verification.json", "stopped-run.json")
    )
    context = {"plan": asdict(pin), "stopped": asdict(original_pins[-1])}
    args = [sys.executable, "-m", _MODULE, "prepare", str(base), json.dumps(context)]
    with (base / "producer.log").open("wb") as log:
        owned = (
            launch_contained_process(
                args, cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT
            )
            if os.name == "nt"
            else None
        )
        child = (
            owned.process
            if owned is not None
            else subprocess.Popen(args, cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT)
        )
        try:
            deadline = time.monotonic() + 240
            while not (base / "ready").is_file():
                assert child.poll() is None, (base / "producer.log").read_text(
                    errors="replace"
                )
                assert time.monotonic() < deadline, (
                    "native producer did not reach published-receipt stop point"
                )
                time.sleep(0.05)
            assert child.poll() is None
        finally:
            if owned is not None:
                owned.close(timeout=15)
            else:
                child.kill()
                child.wait(timeout=15)
        assert child.returncode not in (None, 0)
    # Pin the actual files only after the owned child is reaped. Pre-stop
    # plan, source and enrollment bytes must still be the original pins.
    assert original_pins == tuple(_pin(root, p.path) for p in original_pins)
    before = _snapshot(root)
    pins = tuple(_pin(root, n) for n in sorted(before))
    assert len(pins) == 7
    # Historical public loader continues to reject this synthetic receipt.
    with pytest.raises(ValueError):
        inspect_stopped_recovery(
            root,
            pins,
            expected_source_commit=source.commit_sha,
            expected_frozen_plan_sha256=pin.sha256,
        )
    plan = inspect_synthetic_stopped_recovery(
        root, pins, attempt=attempt, expected_source_binding=source
    )
    assert plan.completed_arm_ids == plan.recovered_arm_ids == (_NONRL,)
    assert len(plan.pending_arm_ids) == 47 and plan.pending_arm_ids[0] == _CASH
    stage_synthetic_stopped_recovery(
        root,
        base / "handoff",
        pins,
        attempt=attempt,
        expected_source_binding=source,
        expected_handoff_sha256=plan.sha256,
    )
    root.rename(base / "retained")
    handoff = _snapshot(base / "handoff")
    assert {n: handoff[n] for n in before} == before

    kwargs = dict(
        dataset=market,
        expected_binding=binding,
        expected_pin=pin,
        expected_source_binding=source,
        expected_handoff_sha256=plan.sha256,
    )
    with pytest.raises(ValueError, match="pending"):
        resume_synthetic_pending(
            base / "handoff",
            base / "refused",
            successor_run_label=_SUCCESSOR,
            arm_ids=(_NONRL,),
            **kwargs,
        )
    assert not (base / "refused").exists()
    with pytest.raises(ValueError, match="handoff digest"):
        open_synthetic_staged_recovery(
            base / "handoff", **(kwargs | {"expected_handoff_sha256": "f" * 64})
        )
    context["handoff"] = plan.sha256
    resumed = subprocess.run(
        [sys.executable, "-m", _MODULE, "resume", str(base), json.dumps(context)],
        cwd=_ROOT,
        capture_output=True,
        timeout=240,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr.decode(errors="replace")
    assert (
        _snapshot(base / "retained") == before
        and _snapshot(base / "handoff") == handoff
    )
    successor = base / "successor"
    assert {p.name for p in (successor / "attempt" / "arms").iterdir()} == {_CASH}
    lineage = json.loads((successor / "lineage.json").read_bytes())
    assert lineage["predecessor_handoff_sha256"] == plan.sha256
    assert lineage["original_run_id"] == _RUN
    successor_id = f"task18-synthetic-successor:{plan.sha256}:{_SUCCESSOR}"
    assert lineage["successor_run_id"] == successor_id
    assert (
        lineage["selected_arm_ids"] == [_CASH]
        and lineage["execution_authorized"] is False
    )
    stopped = json.loads((successor / "attempt" / "stopped-run.json").read_bytes())
    assert stopped["run_id"] == successor_id and stopped["runtime_completed"] is False
    statuses = {a["arm_id"]: a["status"] for a in stopped["arms"]}
    assert statuses[_NONRL] == "NOT_ATTEMPTED" and statuses[_CASH] == "COMPLETED"
    assert list(statuses.values()).count("COMPLETED") == 1
    receipt = json.loads(
        (successor / "attempt" / "arms" / _CASH / "receipt.json").read_bytes()
    )
    assert receipt["external_report"]["run_id"] == successor_id
    assert receipt["external_report"]["observed_advances"] == 6599
    assert receipt["terminal_equity_usdt"] == 100000.0
    assert (successor / "attempt" / "frozen-plan.bin").read_bytes() == before[
        "frozen-plan.bin"
    ]

    # A copied successor attempt cannot erase the inherited completed arm
    # by pretending to be a new original; its bound run namespace is refused.
    successor_attempt = successor / "attempt"
    successor_pins = tuple(
        _pin(successor_attempt, n) for n in sorted(_snapshot(successor_attempt))
    )
    with pytest.raises(ValueError, match="ancestor lineage"):
        inspect_synthetic_stopped_recovery(
            successor_attempt,
            successor_pins,
            attempt=attempt,
            expected_source_binding=source,
        )

    # Fault injection after the actual handoff: the RUNNING declaration must
    # be persisted before calling the native executor. Failure must stop
    # further dispatch and cannot produce a completion artifact.
    failed_root = base / "failed-successor"

    def fail_after_enrollment(*args):
        pending = json.loads(
            (failed_root / "attempt" / "stopped-run.json").read_bytes()
        )
        active = [arm for arm in pending["arms"] if arm["status"] == "RUNNING"]
        assert len(active) == 1 and active[0]["arm_id"] == _CASH
        raise RuntimeError("injected execution failure")

    with patch(
        "tools.tier_d_nonrl.task18_native_recovery.SyntheticNativeAttempt.iter_arms",
        side_effect=fail_after_enrollment,
    ) as dispatch:
        with pytest.raises(RuntimeError, match="injected execution failure"):
            resume_synthetic_pending(
                base / "handoff",
                failed_root,
                successor_run_label="failed-artificial-successor",
                arm_ids=(_CASH, "PAIR-02-NONRL"),
                **kwargs,
            )
    assert dispatch.call_count == 1
    failed = json.loads((failed_root / "attempt" / "stopped-run.json").read_bytes())
    slots = {arm["arm_id"]: arm for arm in failed["arms"]}
    assert slots[_CASH]["status"] == "FAILED"
    assert slots[_CASH]["reason_code"] == "PROCESS_FAILED"
    assert slots[_CASH]["error_code"] == "NATIVE_ARM_EXECUTION_FAILED"
    assert slots[_CASH]["progress"] is None
    assert slots[_CASH]["external_report"] is None
    assert slots["PAIR-02-NONRL"]["status"] == "NOT_ATTEMPTED"
    assert not (failed_root / "attempt" / "arms").exists()
    assert (
        _snapshot(base / "retained") == before
        and _snapshot(base / "handoff") == handoff
    )

    # An occupied output cannot be overwritten or executed again.
    occupied = _snapshot(successor)
    with patch(
        "tools.tier_d_nonrl.task18_native_recovery.SyntheticNativeAttempt.iter_arms",
        side_effect=AssertionError("duplicate dispatch"),
    ):
        with pytest.raises(FileExistsError):
            resume_synthetic_pending(
                base / "handoff",
                successor,
                successor_run_label=_SUCCESSOR,
                arm_ids=(_CASH,),
                **kwargs,
            )
    assert _snapshot(successor) == occupied


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "prepare":
        _publish_then_wait(Path(sys.argv[2]), sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == "resume":
        _resume(Path(sys.argv[2]), sys.argv[3])
    else:
        raise SystemExit("closed synthetic native recovery test worker only")
