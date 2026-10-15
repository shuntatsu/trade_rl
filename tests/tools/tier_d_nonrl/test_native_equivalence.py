"""Artificial-only runner scheduling and resume equivalence checks."""

from __future__ import annotations

import gzip
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from tools.tier_d_nonrl.tier_d_fixture import write_artificial_dataset
from tools.tier_d_nonrl.tier_d_records import load_gzip
from trade_rl.artifacts import canonical_json_bytes


def runner():
    assert importlib.util.find_spec("tools.tier_d_nonrl.tier_d_runner") is not None
    return importlib.import_module("tools.tier_d_nonrl.tier_d_runner")


def artificial_inputs(tmp_path, module, *, max_workers=1):
    dataset_root = tmp_path / "shared-artificial-dataset"
    dataset = write_artificial_dataset(dataset_root)
    protocol = module.synthetic_protocol(
        module.market_protocol(max_workers=max_workers), dataset
    )
    protocol["scenarios"] = ["BASEL0", "BASEL2"]
    protocol["budget"]["native_arms"] = 4
    protocol["dataset"].update(
        root=str(dataset_root),
        input_sha256={
            name: module.file_sha256(dataset_root / name) for name in module.INPUT_PINS
        },
    )
    protocol["economics"].update(
        borrow_rate=0.0876,
        declaration="Artificial manufactured carry for COST2 invariance oracle",
    )
    protocol["funding"].update(
        source="Artificial scheduled 8h events at rate .0001",
        marks="Manufactured rate times artificial close; no venue data",
    )
    return dataset_root, dataset, protocol


def output_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def progress_journals(root: Path) -> dict[str, bytes]:
    directory = root / "native-progress"
    if not directory.exists():
        return {}
    return {path.name: path.read_bytes() for path in sorted(directory.glob("*.jsonl"))}


def assert_completed_progress_journal(raw: bytes) -> None:
    records = [json.loads(line) for line in raw.splitlines()]
    advances = [record for record in records if record["event"] == "native_advanced"]
    assert len(records) == 42
    assert records[0]["event"] == "native_started"
    assert records[0]["count"] == 0
    assert [record["count"] for record in advances] == list(range(1, 41))
    assert sum(record["phase"] == "actor" for record in advances) == 32
    assert sum(record["phase"] == "reserve" for record in advances) == 8
    assert records[-1]["event"] == "native_returned"
    assert records[-1]["count"] == records[-1]["total"] == 40


def install_call_spies(module, monkeypatch):
    calls = {
        "cost_rows": 0,
        "prepare_allocation_market_inputs": 0,
        "run_terminal_allocation_execution": 0,
    }
    for name in calls:
        original = getattr(module, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, spy)
    return calls


def test_worker_count_changes_scheduling_only_for_canonical_artifacts(
    tmp_path, monkeypatch
):
    module = runner()
    dataset_root, dataset, protocol = artificial_inputs(tmp_path, module, max_workers=2)
    assert protocol["concurrency"]["max_workers"] == 2
    assert protocol["resource_proposal"]["maximum_inflight_pairs"] == 2

    calls = {"cost_rows": 0, "prepare_allocation_market_inputs": 0}
    for name in calls:
        original = getattr(module, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, spy)

    frozen_root = tmp_path / "frozen-source"

    class PreparedOnly(RuntimeError):
        pass

    def stop_after_freeze(*args, **kwargs):
        raise PreparedOnly("stop after the artificial matrix is frozen")

    with monkeypatch.context() as preparation_only:
        preparation_only.setattr(module, "execute_pairs", stop_after_freeze)
        with pytest.raises(PreparedOnly, match="matrix is frozen"):
            module.execute(frozen_root, dataset_root, dataset, protocol)

    source_state = json.loads(
        (frozen_root / "attempt.json").read_text(encoding="utf-8")
    )
    assert source_state["status"] == "FAILED"
    assert source_state["matrix_digest"]
    assert source_state["freeze_digest"]
    assert all(arm["state"] == "UNATTEMPTED" for arm in source_state["arms"])
    assert not list((frozen_root / "receipts").glob("*.json.gz"))
    preparation_counts = calls.copy()
    assert all(count > 0 for count in preparation_counts.values())

    frozen_snapshot = output_snapshot(frozen_root)
    dataset_snapshot = output_snapshot(dataset_root)
    frozen_protocol = (frozen_root / "protocol.json").read_bytes()
    source_pins = module.source_pins()
    original_pool = module.concurrent.futures.ProcessPoolExecutor

    def execute_from_frozen_source(destination, actual_workers):
        pool_sizes = []

        def pool_factory(*, max_workers):
            assert max_workers == protocol["concurrency"]["max_workers"]
            assert (
                actual_workers
                <= protocol["resource_proposal"]["maximum_inflight_pairs"]
            )
            pool_sizes.append(actual_workers)
            return original_pool(max_workers=actual_workers)

        with monkeypatch.context() as worker_limit:
            worker_limit.setattr(
                module.concurrent.futures,
                "ProcessPoolExecutor",
                pool_factory,
            )
            result = module.execute(
                destination,
                dataset_root,
                dataset,
                protocol,
                resume_from=frozen_root,
            )
        assert pool_sizes == [actual_workers]
        assert calls == preparation_counts
        assert output_snapshot(frozen_root) == frozen_snapshot
        assert output_snapshot(dataset_root) == dataset_snapshot
        assert (destination / "protocol.json").read_bytes() == frozen_protocol
        return result

    one_worker_root = tmp_path / "one-process-worker"
    two_worker_root = tmp_path / "two-process-workers"
    one_worker = execute_from_frozen_source(one_worker_root, actual_workers=1)
    two_workers = execute_from_frozen_source(two_worker_root, actual_workers=2)

    for result in (one_worker, two_workers):
        assert result["scope"] == "ARTIFICIAL_SOFTWARE_INTEGRATION_ONLY"
        assert result["status"] == "COMPLETED"
        assert result["completed_arms"] == result["attempted_arms"] == 4
        assert result["reused_arms"] == 0
        assert result["public_calls"]["cost_producers"] == 0
        assert result["public_calls"]["prepare_calls"] == 0
        assert len(result["pairs"]) == 2
    assert module.source_pins() == source_pins

    one_worker_plan = one_worker_root / "native-plan.json.gz"
    two_workers_plan = two_worker_root / "native-plan.json.gz"
    frozen_plan = frozen_root / "native-plan.json.gz"
    assert gzip.decompress(one_worker_plan.read_bytes()) == gzip.decompress(
        frozen_plan.read_bytes()
    )
    assert gzip.decompress(two_workers_plan.read_bytes()) == gzip.decompress(
        frozen_plan.read_bytes()
    )
    assert canonical_json_bytes(load_gzip(one_worker_plan)) == canonical_json_bytes(
        load_gzip(two_workers_plan)
    )

    one_worker_receipts = {
        path.name: path for path in (one_worker_root / "receipts").glob("*.json.gz")
    }
    two_worker_receipts = {
        path.name: path for path in (two_worker_root / "receipts").glob("*.json.gz")
    }
    assert one_worker_receipts.keys() == two_worker_receipts.keys()
    assert len(one_worker_receipts) == 4
    for name, one_worker_path in one_worker_receipts.items():
        two_worker_path = two_worker_receipts[name]
        assert gzip.decompress(one_worker_path.read_bytes()) == gzip.decompress(
            two_worker_path.read_bytes()
        )
        assert canonical_json_bytes(load_gzip(one_worker_path)) == canonical_json_bytes(
            load_gzip(two_worker_path)
        )

    assert (one_worker_root / "pairs.json").read_bytes() == (
        two_worker_root / "pairs.json"
    ).read_bytes()
    assert (one_worker_root / "result.json").read_bytes() == (
        two_worker_root / "result.json"
    ).read_bytes()

    one_worker_journals = progress_journals(one_worker_root)
    two_worker_journals = progress_journals(two_worker_root)
    assert one_worker_journals.keys() == two_worker_journals.keys()
    assert len(one_worker_journals) == 4
    assert one_worker_journals == two_worker_journals
    for raw in one_worker_journals.values():
        assert_completed_progress_journal(raw)


def test_completed_resume_skips_real_prepare_cost_and_native_calls(
    tmp_path, monkeypatch
):
    module = runner()
    calls = install_call_spies(module, monkeypatch)
    source, destination = tmp_path / "first", tmp_path / "resumed"

    fresh = module.run_synthetic(source, max_workers=1)
    assert fresh["completed_arms"] == 2
    assert calls == {
        "cost_rows": 1,
        "prepare_allocation_market_inputs": 1,
        "run_terminal_allocation_execution": 2,
    }
    source_before = output_snapshot(source)

    for name in calls:
        calls[name] = 0
    resumed = module.run_synthetic(destination, resume_from=source, max_workers=1)

    assert resumed["status"] == "COMPLETED"
    assert resumed["completed_arms"] == resumed["reused_arms"] == 2
    assert calls == {
        "cost_rows": 0,
        "prepare_allocation_market_inputs": 0,
        "run_terminal_allocation_execution": 0,
    }
    assert resumed["public_calls"]["cost_producers"] == 0
    assert resumed["public_calls"]["prepare_calls"] == 0
    assert progress_journals(destination) == {}
    assert output_snapshot(source) == source_before


def test_partial_resume_reuses_nonrl_receipt_without_refit_or_recost(
    tmp_path, monkeypatch
):
    module = runner()
    calls = {"cost_rows": 0, "prepare": 0, "native": 0}
    original_cost_rows = module.cost_rows
    original_prepare = module.prepare_allocation_market_inputs
    original_native = module.run_terminal_allocation_execution
    fail_cash = True

    def cost_spy(*args, **kwargs):
        calls["cost_rows"] += 1
        return original_cost_rows(*args, **kwargs)

    def prepare_spy(*args, **kwargs):
        calls["prepare"] += 1
        return original_prepare(*args, **kwargs)

    def native_spy(*args, **kwargs):
        calls["native"] += 1
        if fail_cash and calls["native"] == 2:
            raise OSError("artificial CASH-arm interruption")
        return original_native(*args, **kwargs)

    monkeypatch.setattr(module, "cost_rows", cost_spy)
    monkeypatch.setattr(module, "prepare_allocation_market_inputs", prepare_spy)
    monkeypatch.setattr(module, "run_terminal_allocation_execution", native_spy)
    source, destination = tmp_path / "partial", tmp_path / "resumed"

    with pytest.raises(RuntimeError, match="artificial CASH-arm interruption"):
        module.run_synthetic(source, max_workers=1)
    assert calls == {"cost_rows": 1, "prepare": 1, "native": 2}
    failed_state = json.loads((source / "attempt.json").read_text(encoding="utf-8"))
    assert [row["role"] for row in failed_state["arms"]] == ["nonrl", "cash"]
    assert [row["state"] for row in failed_state["arms"]] == [
        "COMPLETED",
        "FAILED",
    ]
    source_before = output_snapshot(source)

    fail_cash = False
    calls.update(cost_rows=0, prepare=0, native=0)
    resumed = module.run_synthetic(destination, resume_from=source, max_workers=1)

    assert resumed["status"] == "COMPLETED"
    assert resumed["completed_arms"] == 2
    assert resumed["reused_arms"] == 1
    assert calls == {"cost_rows": 0, "prepare": 0, "native": 1}
    destination_state = json.loads(
        (destination / "attempt.json").read_text(encoding="utf-8")
    )
    by_role = {row["role"]: row for row in destination_state["arms"]}
    assert by_role["nonrl"]["reused"] is True
    assert by_role["cash"]["reused"] is False
    roster = json.loads(
        (destination / "native-progress-roster.json").read_text(encoding="utf-8")
    )
    cash_arm_ids = {arm["arm_id"] for arm in roster["arms"] if arm["role"] == "cash"}
    resumed_journals = progress_journals(destination)
    assert len(cash_arm_ids) == len(resumed_journals) == 1
    assert set(resumed_journals) == {f"{next(iter(cash_arm_ids))}.jsonl"}
    assert_completed_progress_journal(next(iter(resumed_journals.values())))
    assert output_snapshot(source) == source_before
