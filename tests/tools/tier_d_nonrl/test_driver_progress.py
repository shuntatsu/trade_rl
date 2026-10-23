"""Native progress integration; manufactured orchestration is never market evidence."""

from __future__ import annotations

import concurrent.futures
import importlib
import inspect
import json
import threading
from types import SimpleNamespace

import pytest

from tests.tools.tier_d_nonrl.test_outcomes import pair


def api():
    return importlib.import_module("tools.tier_d_nonrl.tier_d_runner")


def protocol():
    value = api().market_protocol()
    value.update(
        scope="ARTIFICIAL_PROGRESS_ORCHESTRATION_ONLY",
        accounts=["A"],
        scenarios=["BASEL0"],
        configurations=[{"id": "fixture", "horizon_hours": 24}],
        actor={"start": 1, "stop": 2, "rows": 1, "vintage_starts": [1]},
        terminal={"stop_inclusive": 3, "advances": 1},
    )
    value["budget"]["native_arms"] = 2
    return value


def test_source_protocol_freezes_per_arm_no_progress_before_execution():
    value = api().market_protocol()
    assert value["schema"] == "issue810_tier_d_nonrl48_result_blind_v3"
    assert value["progress_policy"] == {
        "schema": "issue810_native_progress_v1",
        "no_progress_seconds": 600,
        "sample_seconds": 1,
    }
    assert value["actor"]["rows"] + value["terminal"]["advances"] == 6599
    assert value["budget"]["native_arms"] == 48


@pytest.mark.parametrize(
    "scope",
    ["CONSUMED_2025_DEVELOPMENT_NONRL48_ONLY", "ARTIFICIAL_SOFTWARE_INTEGRATION_ONLY"],
)
def test_ordinary_execution_requires_progress_policy_before_any_io(
    tmp_path, monkeypatch, scope
):
    module = api()
    value = {"scope": scope}
    monkeypatch.setattr(
        module,
        "build_candidate_run_provenance",
        lambda: pytest.fail("late policy validation"),
    )
    root = tmp_path / "attempt"
    with pytest.raises(ValueError, match="progress"):
        module.execute(root, tmp_path / "absent", object(), value)
    assert not root.exists()


def install_native_stub(module, monkeypatch):
    calls = []

    def native(*args, on_advance=None, **kwargs):
        assert on_advance is not None, "driver omitted actual native progress callback"
        calls.append(True)
        on_advance("actor", 1, 2)
        on_advance("reserve", 2, 3)
        return SimpleNamespace(payload={"scope": "MANUFACTURED_RETURNED_FACTS_ONLY"})

    monkeypatch.setattr(module, "execution_folds", lambda *args: ())
    monkeypatch.setattr(
        module, "carrier", lambda *args, **kwargs: SimpleNamespace(execution_cost=None)
    )
    monkeypatch.setattr(module, "run_terminal_allocation_execution", native)
    monkeypatch.setattr(
        module,
        "TerminalAllocationExecutionReceipt",
        SimpleNamespace(
            from_payload=lambda *args, **kwargs: SimpleNamespace(digest="c" * 64)
        ),
    )
    monkeypatch.setattr(module, "native_closure", lambda *args: {})
    monkeypatch.setattr(module, "ledger_decomposition", lambda *args: {})
    monkeypatch.setattr(module, "SettledNonRLComparisonPlan", lambda *args: None)
    monkeypatch.setattr(
        module,
        "build_settled_nonrl_comparison",
        lambda *args: SimpleNamespace(payload={"groups": [{}]}, digest="d" * 64),
    )
    return calls


def setup_progress(module, root, value, declaration):
    assert (
        "progress_enabled" in inspect.signature(module._worker_execute_pair).parameters
    )
    progress = importlib.import_module("tools.tier_d_nonrl.progress")
    io = importlib.import_module("tools.tier_d_nonrl.progress_io")
    module.initialize_attempt(root, value)
    specs = tuple(
        progress.ArmProgressSpec("fixture", "A", "BASEL0", role, plan.digest, 1, 2, 3)
        for role, plan in zip(
            ("nonrl", "cash"), declaration.controls.plans, strict=True
        )
    )
    io.publish_roster(root, value, specs)
    return io.ProgressReader(root, expected_protocol=value)


def test_direct_worker_publishes_both_native_phases_and_never_replays_reuse(
    tmp_path, monkeypatch
):
    module = api()
    declaration, value, root = pair("A"), protocol(), tmp_path / "attempt"
    reader = setup_progress(module, root, value, declaration)
    calls = install_native_stub(module, monkeypatch)

    def publish():
        for event in reader.poll():
            record = dict(event)
            module.append_log(root, record.pop("event"), **record)

    arguments = (
        root,
        declaration,
        SimpleNamespace(dataset=SimpleNamespace(symbols=("A",))),
        (),
        value,
    )
    result = module._worker_execute_pair(
        *arguments, progress_enabled=True, on_progress=publish
    )
    assert result.failure is None
    assert len(calls) == 2
    rows = [
        json.loads(line) for line in (root / "progress.jsonl").read_text().splitlines()
    ]
    advances = [row for row in rows if row["event"] == "native_advanced"]
    assert [
        (row["role"], row["phase"], row["count"], row["total"]) for row in advances
    ] == [
        ("nonrl", "actor", 1, 2),
        ("nonrl", "reserve", 2, 2),
        ("cash", "actor", 1, 2),
        ("cash", "reserve", 2, 2),
    ]
    snapshot = {path: path.read_bytes() for path in root.rglob("*.jsonl")}
    reused = module._worker_execute_pair(
        *arguments, progress_enabled=True, on_progress=publish
    )
    assert reused.failure is None and len(calls) == 2
    assert all(slot["reused"] for slot in reused.arm_slots.values())
    assert {path: path.read_bytes() for path in snapshot} == snapshot


def test_progress_end_failure_retains_already_returned_native_payload(
    tmp_path, monkeypatch
):
    module = api()
    declaration, value, root = pair("A"), protocol(), tmp_path / "attempt"
    reader = setup_progress(module, root, value, declaration)
    install_native_stub(module, monkeypatch)

    def fail_after_return():
        for event in reader.poll():
            if event["event"] == "native_returned":
                raise OSError("manufactured parent progress publication failure")

    result = module._worker_execute_pair(
        root,
        declaration,
        SimpleNamespace(dataset=SimpleNamespace(symbols=("A",))),
        (),
        value,
        progress_enabled=True,
        on_progress=fail_after_return,
    )
    assert (
        result.failure["message"] == "manufactured parent progress publication failure"
    )
    slot = result.arm_slots["nonrl"]
    assert slot["state"] == "COMPLETED"
    assert module.load_gzip(root / slot["receipt"]["path"]) == {
        "scope": "MANUFACTURED_RETURNED_FACTS_ONLY"
    }
    assert "cash" not in result.arm_slots


def test_failed_journal_entry_does_not_claim_native_execution_started(
    tmp_path, monkeypatch
):
    module = api()
    declaration, value, root = pair("A"), protocol(), tmp_path / "attempt"
    setup_progress(module, root, value, declaration)
    calls = install_native_stub(module, monkeypatch)

    def fail_before_native():
        raise OSError("manufactured journal start failure")

    result = module._worker_execute_pair(
        root,
        declaration,
        SimpleNamespace(dataset=None),
        (),
        value,
        progress_enabled=True,
        on_progress=fail_before_native,
    )
    assert result.failure["message"] == "manufactured journal start failure"
    assert calls == []
    assert result.arm_slots["nonrl"]["execution_started"] is False
    assert "cash" not in result.arm_slots


def test_rejected_readback_still_retains_the_actual_returned_payload(
    tmp_path, monkeypatch
):
    module = api()
    install_native_stub(module, monkeypatch)
    raw = {"scope": "MANUFACTURED_REJECTED_NATIVE_RETURN"}
    monkeypatch.setattr(
        module,
        "run_terminal_allocation_execution",
        lambda *args, **kwargs: SimpleNamespace(payload=raw),
    )

    def reject(*args, **kwargs):
        raise ValueError("manufactured readback rejection")

    monkeypatch.setattr(
        module,
        "TerminalAllocationExecutionReceipt",
        SimpleNamespace(from_payload=reject),
    )
    result = module._worker_execute_pair(
        tmp_path, pair("A"), SimpleNamespace(dataset=None), (), protocol()
    )
    assert result.failure["message"] == "manufactured readback rejection"
    slot = result.arm_slots["nonrl"]
    assert slot["state"] == "FAILED"
    assert slot["receipt"] is not None, "returned native facts were discarded"
    assert module.load_gzip(tmp_path / slot["receipt"]["path"]) == raw


@pytest.mark.parametrize("fail_sink", [False, True])
def test_parent_publishes_parallel_native_progress_before_futures_finish(
    tmp_path, monkeypatch, fail_sink
):
    module = api()
    io = importlib.import_module("tools.tier_d_nonrl.progress_io")
    progress = importlib.import_module("tools.tier_d_nonrl.progress")
    value, root = protocol(), tmp_path / "attempt"
    value["accounts"] = ["A", "B", "C"] if fail_sink else ["A", "B"]
    value["concurrency"]["max_workers"] = 2
    pairs = tuple(pair(account) for account in value["accounts"])
    state = module.initialize_attempt(root, value)
    specs = tuple(
        progress.ArmProgressSpec(
            "fixture", p.account, "BASEL0", role, plan.digest, 1, 2, 3
        )
        for p in pairs
        for role, plan in zip(("nonrl", "cash"), p.controls.plans, strict=True)
    )
    io.publish_roster(root, value, specs)
    observed = threading.Event()
    original_log = module.append_log

    def publish(root, event, **values):
        original_log(root, event, **values)
        if event == "native_advanced":
            observed.set()
            if fail_sink:
                raise OSError("manufactured progress sink failure")

    def worker(root, declaration, *args, progress_enabled=False):
        assert progress_enabled
        spec = next(
            s for s in specs if s.account == declaration.account and s.role == "nonrl"
        )
        with io.ProgressJournal(root, spec) as journal:
            journal.advance("actor", 1, 2)
            assert observed.wait(timeout=5), "parent blocked until future completion"
        return module.PairExecutionResult(
            "fixture",
            declaration.account,
            "BASEL0",
            0,
            {},
            {},
            {role: {"state": "COMPLETED"} for role in ("nonrl", "cash")},
        )

    monkeypatch.setattr(module, "append_log", publish)
    monkeypatch.setattr(module, "_worker_execute_pair", worker)
    monkeypatch.setattr(module, "symbol_estimates", lambda *args: ())
    monkeypatch.setattr(
        module.concurrent.futures,
        "ProcessPoolExecutor",
        concurrent.futures.ThreadPoolExecutor,
    )
    inputs = SimpleNamespace(dataset=SimpleNamespace(symbols=("ARTIFICIAL",)))
    if fail_sink:
        with pytest.raises(RuntimeError, match="manufactured progress sink failure"):
            module.execute_pairs(root, pairs, {"fixture": inputs}, {}, value, state)
        assert {
            slot["state"] for slot in state["arms"] if slot["account"] in {"A", "B"}
        } == {"COMPLETED"}
        assert {slot["state"] for slot in state["arms"] if slot["account"] == "C"} == {
            "UNATTEMPTED"
        }
        assert len(state["publication_failures"]) == 1
    else:
        groups, _ = module.execute_pairs(
            root, pairs, {"fixture": inputs}, {}, value, state
        )
        assert len(groups) == 2
    assert observed.is_set()
    rows = [
        json.loads(line) for line in (root / "progress.jsonl").read_text().splitlines()
    ]
    assert len([row for row in rows if row["event"] == "native_advanced"]) == (
        1 if fail_sink else 2
    )


@pytest.mark.parametrize("slow_stage", ["payload", "readback", "compression"])
def test_early_native_return_ends_deadline_before_postprocessing(
    tmp_path, monkeypatch, slow_stage
):
    from tools.tier_d_nonrl.progress_io import ProgressReader

    module = api()
    declaration, value, root = pair("A"), protocol(), tmp_path / "attempt"
    setup_progress(module, root, value, declaration)
    install_native_stub(module, monkeypatch)
    clock, observations = [0.0], []
    reader = ProgressReader(root, expected_protocol=value, clock=lambda: clock[0])

    def slow(stage):
        if stage == slow_stage:
            clock[0] += value["progress_policy"]["no_progress_seconds"] + 1
            observations.append(reader.stalled())

    class Returned:
        @property
        def payload(self):
            slow("payload")
            return {"scope": "MANUFACTURED_EARLY_NATIVE_RETURN"}

    def native(*args, on_advance):
        on_advance("actor", 1, 2)  # Return before the declared reserve advance.
        return Returned()

    def readback(*args, **kwargs):
        slow("readback")
        return SimpleNamespace(digest="c" * 64)

    original_save = module.save_gzip

    def save(*args, **kwargs):
        slow("compression")
        return original_save(*args, **kwargs)

    monkeypatch.setattr(module, "run_terminal_allocation_execution", native)
    monkeypatch.setattr(
        module,
        "TerminalAllocationExecutionReceipt",
        SimpleNamespace(from_payload=readback),
    )
    monkeypatch.setattr(module, "save_gzip", save)
    result = module._worker_execute_pair(
        root,
        declaration,
        SimpleNamespace(dataset=SimpleNamespace(symbols=("A",))),
        (),
        value,
        progress_enabled=True,
        on_progress=lambda: list(reader.poll()),
    )
    assert result.failure is None
    assert len(observations) == 2 and not any(observations)
    assert all(
        state["event"] == "native_returned"
        and state["count"] == 1
        and state["target"] == 2
        for state in reader.states.values()
    )


def test_deferred_progress_failure_keeps_readback_primary_and_available_raw(
    tmp_path, monkeypatch
):
    module = api()
    declaration, value, root = pair("A"), protocol(), tmp_path / "attempt"
    reader = setup_progress(module, root, value, declaration)
    install_native_stub(module, monkeypatch)
    raw = {"scope": "MANUFACTURED_RETURNED_FACTS_ONLY"}

    def publish():
        for event in reader.poll():
            if event["event"] == "native_returned":
                raise OSError("manufactured deferred progress sink failure")

    def reject(*args, **kwargs):
        raise ValueError("manufactured primary readback failure")

    monkeypatch.setattr(
        module,
        "TerminalAllocationExecutionReceipt",
        SimpleNamespace(from_payload=reject),
    )
    result = module._worker_execute_pair(
        root,
        declaration,
        SimpleNamespace(dataset=None),
        (),
        value,
        progress_enabled=True,
        on_progress=publish,
    )
    assert result.failure["type"] == "ValueError"
    assert result.failure["message"] == "manufactured primary readback failure"
    slot = result.arm_slots["nonrl"]
    assert slot["state"] == "FAILED" and "cash" not in result.arm_slots
    assert slot["progress_publication_failure"]["message"] == (
        "manufactured deferred progress sink failure"
    )
    assert module.load_gzip(root / slot["receipt"]["path"]) == raw


def test_deferred_progress_failure_keeps_save_primary_and_partial_raw(
    tmp_path, monkeypatch
):
    module = api()
    declaration, value, root = pair("A"), protocol(), tmp_path / "attempt"
    reader = setup_progress(module, root, value, declaration)
    install_native_stub(module, monkeypatch)
    original_save = module.save_gzip

    def publish():
        for event in reader.poll():
            if event["event"] == "native_returned":
                raise OSError("manufactured deferred progress sink failure")

    def save(root, relative, raw):
        if not relative.endswith(".partial.json.gz"):
            raise PermissionError("manufactured primary receipt save failure")
        return original_save(root, relative, raw)

    monkeypatch.setattr(module, "save_gzip", save)
    result = module._worker_execute_pair(
        root,
        declaration,
        SimpleNamespace(dataset=None),
        (),
        value,
        progress_enabled=True,
        on_progress=publish,
    )
    assert result.failure["type"] == "PermissionError"
    assert result.failure["message"] == "manufactured primary receipt save failure"
    slot = result.arm_slots["nonrl"]
    assert slot["state"] == "FAILED" and "cash" not in result.arm_slots
    assert "progress sink failure" in slot["progress_publication_failure"]["message"]
    assert module.load_gzip(root / slot["receipt"]["path"]) == {
        "scope": "MANUFACTURED_RETURNED_FACTS_ONLY"
    }


def test_native_primary_and_partial_survive_failed_end_notification(
    tmp_path, monkeypatch
):
    module = api()
    declaration, value, root = pair("A"), protocol(), tmp_path / "attempt"
    reader = setup_progress(module, root, value, declaration)
    install_native_stub(module, monkeypatch)
    error = LookupError("manufactured primary native failure")
    error.terminal_execution_receipt = {"scope": "MANUFACTURED_NATIVE_PARTIAL"}

    def native(*args, on_advance):
        on_advance("actor", 1, 2)
        raise error

    def publish():
        for event in reader.poll():
            if event["event"] == "native_failed":
                raise OSError("manufactured failed-end progress sink failure")

    monkeypatch.setattr(module, "run_terminal_allocation_execution", native)
    result = module._worker_execute_pair(
        root,
        declaration,
        SimpleNamespace(dataset=None),
        (),
        value,
        progress_enabled=True,
        on_progress=publish,
    )
    assert result.failure["type"] == "LookupError"
    assert result.failure["message"] == str(error)
    slot = result.arm_slots["nonrl"]
    assert "cash" not in result.arm_slots
    assert (
        "failed-end progress sink failure"
        in slot["progress_publication_failure"]["message"]
    )
    assert (
        module.load_gzip(root / slot["receipt"]["path"])
        == error.terminal_execution_receipt
    )


@pytest.mark.parametrize("invalid_roster", ["missing", "misbound"])
def test_declared_progress_requires_valid_roster_before_any_submission(
    tmp_path, monkeypatch, invalid_roster
):
    module = api()
    declaration, value, root = pair("A"), protocol(), tmp_path / "attempt"
    setup_progress(module, root, value, declaration)
    state = json.loads((root / "attempt.json").read_text())
    roster = root / "native-progress-roster.json"
    if invalid_roster == "missing":
        roster.unlink()
    else:
        raw = json.loads(roster.read_text())
        raw["protocol_digest"] = "0" * 64
        roster.write_text(json.dumps(raw))
    monkeypatch.setattr(
        module,
        "_worker_execute_pair",
        lambda *args, **kwargs: pytest.fail("submitted native work"),
    )
    with pytest.raises(ValueError, match="roster"):
        module.execute_pairs(root, (declaration,), {}, {}, value, state)
    assert {slot["state"] for slot in state["arms"]} == {"UNATTEMPTED"}
    logs = [
        json.loads(line) for line in (root / "progress.jsonl").read_text().splitlines()
    ]
    assert not any(row["event"] == "pair_submitted" for row in logs)
