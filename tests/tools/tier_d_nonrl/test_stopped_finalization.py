"""Supervisor publication of validated receipts after a verified stop."""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path

import pytest

from tests.tools.tier_d_nonrl.test_stopped_receipts import (
    stopped_attempt as _source_stopped_attempt,
)


@pytest.fixture(scope="module")
def stopped_attempt_fixture(tmp_path_factory):
    return _source_stopped_attempt.__wrapped__(tmp_path_factory)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_supervisor_reconciles_only_after_owned_tree_stop(
    tmp_path, monkeypatch, stopped_attempt_fixture
):
    fixture_source, _ = stopped_attempt_fixture
    source = tmp_path / "stopped-attempt"
    shutil.copytree(fixture_source, source)
    supervisor = importlib.import_module("tools.tier_d_nonrl.tier_d_supervisor")
    resume = importlib.import_module("tools.tier_d_nonrl.resume")
    records = importlib.import_module("tools.tier_d_nonrl.tier_d_records")
    runner = importlib.import_module("tools.tier_d_nonrl.tier_d_runner")
    protocol = records.read_json(source / "protocol.json")
    provenance = records.read_json(source / "provenance.json")
    original = _snapshot(source)

    publish = getattr(supervisor, "_publish_stopped_receipts", None)
    assert callable(publish), (
        "supervisor stopped-attempt finalization API is required; receipt "
        "reconciliation must follow verified owned-tree termination"
    )

    reconciliation_calls = []
    actual_reconcile = resume.reconcile_stopped_receipts

    def record_reconciliation(*args, **kwargs):
        reconciliation_calls.append((args, kwargs))
        return actual_reconcile(*args, **kwargs)

    monkeypatch.setattr(resume, "reconcile_stopped_receipts", record_reconciliation)
    skipped = publish(source, protocol, provenance, owned_tree_stopped=False)
    assert skipped == {"state": "SKIPPED_NO_VERIFIED_OWNED_TREE_STOP"}
    assert reconciliation_calls == []
    assert _snapshot(source) == original

    reconciled = publish(source, protocol, provenance, owned_tree_stopped=True)
    assert reconciled == {"state": "RECONCILED", "completed_arms": 1}
    assert len(reconciliation_calls) == 1

    state = records.read_json(source / "attempt.json")
    assert state["status"] == "FAILED"
    assert [(slot["role"], slot["state"]) for slot in state["arms"]] == [
        ("nonrl", "COMPLETED"),
        ("cash", "INTERRUPTED"),
    ]
    nonrl_relative = "receipts/ARTIFICIAL_H72_A10_A_BASEL2_nonrl.json.gz"
    cash_relative = "receipts/ARTIFICIAL_H72_A10_A_BASEL2_cash.json.gz"
    original_receipt = original[nonrl_relative]
    assert (source / nonrl_relative).read_bytes() == original_receipt
    assert not (source / cash_relative).exists()
    after_publish = _snapshot(source)
    assert set(after_publish) == set(original)
    assert {
        name for name in after_publish if after_publish[name] != original[name]
    } == {"attempt.json"}

    # The real finalizer publishes the new manifest after this read-only API.
    records.write_json(
        source / "output-manifest.json",
        records.output_manifest(source),
        replace=True,
    )
    before_resume = _snapshot(source)
    destination = tmp_path / "fresh-resumed-attempt"
    native_roles = []
    actual_native = runner.run_terminal_allocation_execution

    def count_native(folds, env, plan, **kwargs):
        native_roles.append(plan.native_plan.payload["cell"]["kind"])
        return actual_native(folds, env, plan, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("resume must use frozen costs and forecasts")

    monkeypatch.setattr(runner, "run_terminal_allocation_execution", count_native)
    for name in ("cost_rows", "declare_matrix", "prepare_inputs"):
        monkeypatch.setattr(runner, name, forbidden)
    result = runner.run_synthetic(destination, resume_from=source)

    assert result["completed_arms"] == 2
    assert result["reused_arms"] == 1
    assert result["attempted_arms"] == 1
    assert result["public_calls"]["cost_producers"] == 0
    assert result["public_calls"]["prepare_calls"] == 0
    assert result["public_calls"]["ridge_vintages"] == 0
    assert native_roles == ["cash"]
    assert (destination / nonrl_relative).read_bytes() == original_receipt
    assert _snapshot(source) == before_resume


def test_supervisor_refuses_corrupt_receipt_without_changing_its_path(
    tmp_path, stopped_attempt_fixture
):
    fixture_source, _ = stopped_attempt_fixture
    source = tmp_path / "corrupt-stopped-attempt"
    shutil.copytree(fixture_source, source)
    supervisor = importlib.import_module("tools.tier_d_nonrl.tier_d_supervisor")
    records = importlib.import_module("tools.tier_d_nonrl.tier_d_records")
    protocol = records.read_json(source / "protocol.json")
    provenance = records.read_json(source / "provenance.json")
    relative = "receipts/ARTIFICIAL_H72_A10_A_BASEL2_nonrl.json.gz"
    receipt_path = source / relative
    receipt_path.write_bytes(b"manufactured corrupt gzip")
    records.write_json(
        source / "output-manifest.json",
        records.output_manifest(source),
        replace=True,
    )
    before = _snapshot(source)

    publish = getattr(supervisor, "_publish_stopped_receipts", None)
    assert callable(publish), "supervisor stopped-attempt finalization API is required"
    result = publish(source, protocol, provenance, owned_tree_stopped=True)

    assert result["state"] == "REFUSED"
    assert result["failure"]["type"] == "ValueError"
    assert _snapshot(source) == before
    assert receipt_path.read_bytes() == b"manufactured corrupt gzip"
