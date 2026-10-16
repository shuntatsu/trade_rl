"""Bounded resume and retained-receipt binding contracts using artificial inputs."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from tests.tools.tier_d_nonrl.test_native_equivalence import output_snapshot, runner
from tools.tier_d_nonrl.tier_d_records import (
    load_gzip,
    output_manifest,
    read_json,
    save_gzip,
    write_json,
)
from trade_rl.evaluation.allocation_terminal_evidence import (
    TerminalAllocationExecutionReceipt,
)


def _restored_matrix(module, root: Path):
    state = read_json(root / "attempt.json")
    frozen = read_json(root / "freeze.json")
    payload = load_gzip(root / frozen["matrix_file"]["path"])
    restore = importlib.import_module("tools.tier_d_nonrl.restore")
    return restore.restore_matrix(payload, expected_digest=state["matrix_digest"])


def test_manufactured_interrupted_arm_resumes_from_frozen_source_without_rework(
    tmp_path, monkeypatch
):
    """Manufacture an interrupted slot after an artificial CASH failure; no process is killed."""
    module = runner()
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    original_native = module.run_terminal_allocation_execution
    first_run_calls = 0

    def interrupt_cash(*args, **kwargs):
        nonlocal first_run_calls
        first_run_calls += 1
        if first_run_calls == 2:
            raise OSError("manufactured interruption before CASH receipt")
        return original_native(*args, **kwargs)

    with monkeypatch.context() as interrupted_run:
        interrupted_run.setattr(
            module, "run_terminal_allocation_execution", interrupt_cash
        )
        with pytest.raises(RuntimeError, match="manufactured interruption"):
            module.run_synthetic(source, max_workers=1)

    assert first_run_calls == 2
    state = read_json(source / "attempt.json")
    assert [arm["state"] for arm in state["arms"]] == ["COMPLETED", "FAILED"]
    # The source did not suffer an OS interruption; only its arm disposition is
    # manufactured to exercise the existing INTERRUPTED resume contract.
    cash = state["arms"][1]
    cash.update(
        state="INTERRUPTED",
        receipt=None,
        failure={
            "type": "ManufacturedInterruption",
            "message": "test-only state edit; no process was killed",
        },
    )
    write_json(source / "attempt.json", state, replace=True)
    write_json(source / "output-manifest.json", output_manifest(source), replace=True)
    source_before = output_snapshot(source)
    cash_path = f"receipts/{cash['account']}_{cash['scenario']}_{cash['role']}.json.gz"
    assert cash_path not in read_json(source / "output-manifest.json")

    calls = {
        "cost_rows": 0,
        "prepare_allocation_market_inputs": 0,
        "declare_matrix": 0,
        "run_terminal_allocation_execution": 0,
    }
    for name in calls:
        original = getattr(module, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, spy)

    result = module.run_synthetic(destination, resume_from=source, max_workers=1)

    assert result["status"] == "COMPLETED"
    assert result["completed_arms"] == 2
    assert result["reused_arms"] == 1
    assert calls == {
        "cost_rows": 0,
        "prepare_allocation_market_inputs": 0,
        "declare_matrix": 0,
        "run_terminal_allocation_execution": 1,
    }
    resumed = read_json(destination / "attempt.json")
    by_role = {arm["role"]: arm for arm in resumed["arms"]}
    assert by_role["nonrl"]["state"] == "COMPLETED"
    assert by_role["nonrl"]["reused"] is True
    assert by_role["cash"]["state"] == "COMPLETED"
    assert by_role["cash"]["reused"] is False
    assert (destination / cash_path).is_file()

    link = read_json(destination / "resume.json")
    assert link["source"] == str(source.resolve())
    assert link["source_manifest_sha256"] == module.file_sha256(
        source / "output-manifest.json"
    )
    assert state["arms"][1]["state"] == "INTERRUPTED"
    assert output_snapshot(source) == source_before


def test_valid_foreign_receipt_is_rejected_before_worker_environment(
    tmp_path, monkeypatch
):
    module = runner()
    source = tmp_path / "complete-source"
    module.run_synthetic(source, max_workers=1)

    protocol = read_json(source / "protocol.json")
    state = read_json(source / "attempt.json")
    matrix = _restored_matrix(module, source)
    controls = matrix.controls[0]
    assert controls.nonrl_plan.digest != controls.cash_plan.digest

    account, scenario = protocol["accounts"][0], protocol["scenarios"][0]
    cash_relative = f"receipts/{account}_{scenario}_cash.json.gz"
    foreign_payload = load_gzip(source / cash_relative)
    valid_foreign = TerminalAllocationExecutionReceipt.from_payload(
        foreign_payload, expected_plan=controls.cash_plan
    )
    cash_slot = next(arm for arm in state["arms"] if arm["role"] == "cash")
    assert valid_foreign.digest == cash_slot["receipt_digest"]

    # Put a genuine CASH receipt at the NONRL worker path. It is valid for its
    # own plan; only the worker's frozen NONRL plan should reject this binding.
    worker_root = tmp_path / "injected-worker"
    nonrl_relative = f"receipts/{account}_{scenario}_nonrl.json.gz"
    save_gzip(worker_root, nonrl_relative, foreign_payload)
    pair = module.PairDeclaration(
        account,
        scenario,
        protocol["configurations"][0]["id"],
        0,
        controls,
    )
    calls = {"carrier": 0, "native": 0}

    def forbidden(name):
        def fail(*args, **kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} ran before receipt binding rejection")

        return fail

    monkeypatch.setattr(module, "carrier", forbidden("carrier"))
    monkeypatch.setattr(
        module, "run_terminal_allocation_execution", forbidden("native")
    )

    source_before = output_snapshot(source)
    worker_before = output_snapshot(worker_root)
    expected = "terminal content differs from its declared or reconstructed value"
    with pytest.raises(ValueError, match=re.escape(expected)) as error:
        module._worker_execute_pair(worker_root, pair, object(), (), protocol)

    assert str(error.value) == expected
    assert calls == {"carrier": 0, "native": 0}
    assert output_snapshot(source) == source_before
    assert output_snapshot(worker_root) == worker_before
