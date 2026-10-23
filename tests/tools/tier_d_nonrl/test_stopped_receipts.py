"""Read-only recovery of completed receipts after an owned stop."""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from trade_rl.artifacts import content_digest
from trade_rl.evaluation.allocation_terminal_evidence import (
    TerminalAllocationExecutionReceipt,
)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture(scope="module")
def stopped_attempt(tmp_path_factory):
    """Build one genuine artificial NONRL receipt and stop CASH after one step."""
    runner = importlib.import_module("tools.tier_d_nonrl.tier_d_runner")
    supervisor = importlib.import_module("tools.tier_d_nonrl.tier_d_supervisor")
    records = importlib.import_module("tools.tier_d_nonrl.tier_d_records")
    source = tmp_path_factory.mktemp("stopped-tier-d") / "attempt"
    actual_run = runner.run_terminal_allocation_execution
    cash_advances: list[int] = []

    def stop_cash_after_one_advance(folds, env, plan, **kwargs):
        if plan.native_plan.payload["cell"]["kind"] != "cash":
            return actual_run(folds, env, plan, **kwargs)

        actual_step = env.step
        step_count = 0

        def advance_then_interrupt(action):
            nonlocal step_count
            result = actual_step(action)
            step_count += 1
            if step_count == 1:
                cash_advances.append(env.index)
                raise KeyboardInterrupt("manufactured abrupt CASH stop")
            return result

        env.step = advance_then_interrupt
        return actual_run(folds, env, plan, **kwargs)

    with patch.object(
        runner,
        "run_terminal_allocation_execution",
        stop_cash_after_one_advance,
    ):
        with pytest.raises(KeyboardInterrupt, match="manufactured abrupt CASH stop"):
            runner.run_synthetic(source)

    assert cash_advances == [129]
    supervisor.mark_attempt_failure(
        source, failure="manufactured abrupt CASH stop", returncode=None
    )
    records.write_json(
        source / "output-manifest.json",
        records.output_manifest(source),
        replace=True,
    )
    return source, cash_advances


def test_stopped_receipt_reconciliation_is_read_only_and_preserves_strict_resume(
    tmp_path, monkeypatch, stopped_attempt
):
    source, cash_advances = stopped_attempt
    resume = importlib.import_module("tools.tier_d_nonrl.resume")
    records = importlib.import_module("tools.tier_d_nonrl.tier_d_records")
    runner = importlib.import_module("tools.tier_d_nonrl.tier_d_runner")
    protocol = records.read_json(source / "protocol.json")
    provenance = records.read_json(source / "provenance.json")
    before = _snapshot(source)

    state = records.read_json(source / "attempt.json")
    assert state["status"] == "FAILED"
    assert [(row["role"], row["state"]) for row in state["arms"]] == [
        ("nonrl", "INTERRUPTED"),
        ("cash", "INTERRUPTED"),
    ]
    nonrl_path = source / "receipts/ARTIFICIAL_H72_A10_A_BASEL2_nonrl.json.gz"
    cash_path = source / "receipts/ARTIFICIAL_H72_A10_A_BASEL2_cash.json.gz"
    assert nonrl_path.is_file()
    assert not cash_path.exists()
    assert cash_advances == [129]

    destination = tmp_path / "strict-resume"
    with pytest.raises(ValueError, match="ambiguous|unaccounted"):
        resume.preflight_resume(source, destination, protocol, provenance)
    assert not destination.exists()
    assert _snapshot(source) == before

    reconcile = getattr(resume, "reconcile_stopped_receipts", None)
    assert callable(reconcile), (
        "read-only stopped-receipt reconciliation API is required; public resume "
        "must continue rejecting ambiguous interrupted attempts"
    )

    frozen = records.read_json(source / "freeze.json")
    matrix = resume.restore_matrix(
        records.load_gzip(source / frozen["matrix_file"]["path"]),
        expected_digest=state["matrix_digest"],
    )
    plan = matrix.controls[0].nonrl_plan
    raw_receipt = records.load_gzip(nonrl_path)
    receipt = TerminalAllocationExecutionReceipt.from_payload(
        raw_receipt, expected_plan=plan
    )
    expected = (
        {
            "account": "ARTIFICIAL_H72_A10_A",
            "scenario": "BASEL2",
            "role": "nonrl",
            "state": "COMPLETED",
            "plan_digest": plan.digest,
            "receipt_digest": receipt.digest,
            "receipt": {
                "path": nonrl_path.relative_to(source).as_posix(),
                "bytes": nonrl_path.stat().st_size,
                "sha256": runner.file_sha256(nonrl_path),
            },
            "reconciled_after_stop": True,
        },
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("reconciliation must not rebuild or execute an arm")

    with monkeypatch.context() as context:
        for name in (
            "build_candidate_run_provenance",
            "carrier",
            "cost_rows",
            "declare_matrix",
            "prepare_inputs",
            "run_synthetic",
            "run_terminal_allocation_execution",
        ):
            context.setattr(runner, name, forbidden)
        updates = reconcile(source, protocol, provenance)

    assert updates == expected
    assert content_digest(records.load_gzip(nonrl_path)) == receipt.digest
    assert not cash_path.exists()
    assert _snapshot(source) == before


@pytest.mark.parametrize(
    "case",
    [
        "corrupt_gzip",
        "foreign_cash",
        "unattempted",
        "protocol",
        "provenance",
        "matrix",
        "forecast",
        "costs",
        "extra_receipt",
        "stale_completed",
        "symlink",
    ],
)
def test_stopped_receipt_reconciliation_refuses_untrusted_or_ineligible_receipts(
    tmp_path, stopped_attempt, case
):
    source, _ = stopped_attempt
    resume = importlib.import_module("tools.tier_d_nonrl.resume")
    records = importlib.import_module("tools.tier_d_nonrl.tier_d_records")
    protocol = records.read_json(source / "protocol.json")
    provenance = records.read_json(source / "provenance.json")
    before_source = _snapshot(source)
    candidate = tmp_path / case
    shutil.copytree(source, candidate)
    state = records.read_json(candidate / "attempt.json")
    nonrl_path = candidate / "receipts/ARTIFICIAL_H72_A10_A_BASEL2_nonrl.json.gz"
    cash_path = candidate / "receipts/ARTIFICIAL_H72_A10_A_BASEL2_cash.json.gz"
    requested_protocol = protocol
    requested_provenance = provenance
    publish_manifest = False

    if case == "corrupt_gzip":
        nonrl_path.write_bytes(b"manufactured invalid gzip bytes")
        publish_manifest = True
    elif case == "foreign_cash":
        # This is a genuine NONRL receipt placed at the CASH role's path.
        cash_path.write_bytes(nonrl_path.read_bytes())
        publish_manifest = True
    elif case == "unattempted":
        state["arms"][0]["state"] = "UNATTEMPTED"
        records.write_json(candidate / "attempt.json", state, replace=True)
        publish_manifest = True
    elif case == "protocol":
        requested_protocol = protocol | {"scope": "FOREIGN_PROTOCOL"}
    elif case == "provenance":
        requested_provenance = provenance | {"source_head": "0" * 40}
    elif case in {"matrix", "forecast", "costs"}:
        frozen = records.read_json(candidate / "freeze.json")
        relative = {
            "matrix": frozen["matrix_file"]["path"],
            "forecast": frozen["forecasts"]["H72_RIDGE_A10"]["path"],
            "costs": frozen["costs_file"]["path"],
        }[case]
        artifact = candidate / relative
        artifact.write_bytes(artifact.read_bytes() + b"changed after freeze")
        publish_manifest = True
    elif case == "extra_receipt":
        (candidate / "receipts/unexpected.json.gz").write_bytes(b"unaccounted receipt")
        publish_manifest = True
    elif case == "stale_completed":
        frozen = records.read_json(candidate / "freeze.json")
        matrix = resume.restore_matrix(
            records.load_gzip(candidate / frozen["matrix_file"]["path"]),
            expected_digest=state["matrix_digest"],
        )
        metadata = records.read_json(candidate / "output-manifest.json")[
            nonrl_path.relative_to(candidate).as_posix()
        ]
        state["arms"][0].update(
            state="COMPLETED",
            plan_digest=matrix.controls[0].nonrl_plan.digest,
            receipt_digest="0" * 64,
            receipt={"path": nonrl_path.relative_to(candidate).as_posix(), **metadata},
        )
        records.write_json(candidate / "attempt.json", state, replace=True)
        publish_manifest = True
    elif case == "symlink":
        try:
            (candidate / "foreign-link.json").symlink_to(candidate / "protocol.json")
        except OSError as error:
            pytest.skip(f"Windows symlink creation is unavailable: {error}")
    if publish_manifest:
        records.write_json(
            candidate / "output-manifest.json",
            records.output_manifest(candidate),
            replace=True,
        )
    before_candidate = _snapshot(candidate)

    reconcile = getattr(resume, "reconcile_stopped_receipts", None)
    assert callable(reconcile), "stopped-receipt reconciliation API is required"
    with pytest.raises(ValueError):
        reconcile(candidate, requested_protocol, requested_provenance)

    assert _snapshot(candidate) == before_candidate
    if case == "symlink":
        assert (candidate / "foreign-link.json").is_symlink()
    assert _snapshot(source) == before_source


def test_stopped_receipt_reconciliation_detects_source_change_during_read(
    tmp_path, monkeypatch, stopped_attempt
):
    source, _ = stopped_attempt
    resume = importlib.import_module("tools.tier_d_nonrl.resume")
    records = importlib.import_module("tools.tier_d_nonrl.tier_d_records")
    protocol = records.read_json(source / "protocol.json")
    provenance = records.read_json(source / "provenance.json")
    candidate = tmp_path / "mutated-during-read"
    shutil.copytree(source, candidate)
    before_source = _snapshot(source)
    original_output_manifest = resume.output_manifest
    output_manifest_calls = 0
    after_injected_change: dict[str, bytes] = {}

    def change_attempt_on_final_read(root):
        nonlocal output_manifest_calls
        output_manifest_calls += 1
        if output_manifest_calls == 2:
            attempt = root / "attempt.json"
            attempt.write_bytes(attempt.read_bytes() + b" ")
            after_injected_change.update(_snapshot(root))
        return original_output_manifest(root)

    monkeypatch.setattr(resume, "output_manifest", change_attempt_on_final_read)
    reconcile = getattr(resume, "reconcile_stopped_receipts", None)
    assert callable(reconcile), "stopped-receipt reconciliation API is required"
    with pytest.raises(ValueError, match="identity changed|differs"):
        reconcile(candidate, protocol, provenance)

    assert output_manifest_calls == 2
    assert after_injected_change
    assert _snapshot(candidate) == after_injected_change
    assert _snapshot(source) == before_source
