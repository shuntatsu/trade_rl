"""Manufactured orchestration outcomes; no Dataset or native execution."""

import concurrent.futures
import importlib
import json
import pickle
from types import SimpleNamespace

import pytest


def api():
    return importlib.import_module("tools.tier_d_nonrl.tier_d_runner")


def pair(account):
    return SimpleNamespace(
        account=account,
        scenario="BASEL0",
        configuration="fixture",
        symbol_index=0,
        controls=SimpleNamespace(
            plans=(SimpleNamespace(digest="a" * 64), SimpleNamespace(digest="b" * 64))
        ),
    )


def test_second_role_failure_retains_first_role_and_is_picklable(tmp_path, monkeypatch):
    module = api()
    monkeypatch.setattr(module, "execution_folds", lambda *args: ())
    monkeypatch.setattr(
        module, "carrier", lambda *args, **kwargs: SimpleNamespace(execution_cost=None)
    )
    calls = []

    def manufactured(*args):
        calls.append(None)
        if len(calls) == 2:
            raise ValueError("manufactured second-role failure")
        return SimpleNamespace(payload={"scope": "ARTIFICIAL_ORCHESTRATION_ONLY"})

    monkeypatch.setattr(module, "run_terminal_allocation_execution", manufactured)
    monkeypatch.setattr(
        module,
        "TerminalAllocationExecutionReceipt",
        SimpleNamespace(
            from_payload=lambda *args, **kwargs: SimpleNamespace(digest="c" * 64)
        ),
    )
    monkeypatch.setattr(module, "native_closure", lambda *args: {})
    monkeypatch.setattr(module, "ledger_decomposition", lambda *args: {})
    result = module._worker_execute_pair(
        tmp_path,
        pair("A"),
        SimpleNamespace(dataset=None),
        (),
        {
            "actor": {"vintage_starts": [1], "stop": 2},
            "configurations": [{"id": "fixture", "horizon_hours": 24}],
        },
    )
    restored = pickle.loads(pickle.dumps(result))
    assert restored.arm_slots["nonrl"]["state"] == "COMPLETED"
    assert restored.arm_slots["cash"]["state"] == "FAILED"
    assert restored.failure["message"] == "manufactured second-role failure"
    assert (tmp_path / "receipts/A_BASEL0_nonrl.json.gz").is_file()
    assert not (tmp_path / "receipts/A_BASEL0_cash.json.gz").exists()


def test_first_failed_future_drains_completed_and_cancelled_outcomes(
    tmp_path, monkeypatch
):
    module = api()
    pairs = tuple(pair(name) for name in ("A", "B", "C"))
    protocol = {
        "scope": "ARTIFICIAL_ORCHESTRATION_ONLY",
        "source_head": "a" * 40,
        "accounts": ["A", "B", "C"],
        "scenarios": ["BASEL0"],
        "roles": ["nonrl", "cash"],
        "configurations": [{"id": "fixture", "horizon_hours": 24}],
        "concurrency": {"max_workers": 3},
    }
    inputs = SimpleNamespace(dataset=SimpleNamespace(symbols=("ARTIFICIAL",)))
    monkeypatch.setattr(module, "build_candidate_run_provenance", lambda: {})
    monkeypatch.setattr(
        module, "prepare_inputs", lambda *args, **kwargs: ({"fixture": inputs}, {})
    )
    monkeypatch.setattr(module, "declare_matrix", lambda *args: pairs)
    monkeypatch.setattr(
        module,
        "freeze_inputs",
        lambda *args: {"scope": "ARTIFICIAL_ORCHESTRATION_ONLY"},
    )
    monkeypatch.setattr(module, "symbol_estimates", lambda *args: ())
    monkeypatch.setattr(
        module,
        "SettledNonRLComparisonPlan",
        lambda *args: SimpleNamespace(
            digest="d" * 64, payload={"scope": "ARTIFICIAL_ORCHESTRATION_ONLY"}
        ),
    )
    futures = []

    class Future:
        def __init__(self, account):
            self.account, self.was_cancelled = account, False

        def cancel(self):
            self.was_cancelled = self.account == "C"
            return self.was_cancelled

        def cancelled(self):
            return self.was_cancelled

        def result(self):
            if self.was_cancelled:
                raise concurrent.futures.CancelledError()
            if self.account == "A":
                raise ValueError("manufactured first future failure")
            return module.PairExecutionResult(
                "fixture",
                "B",
                "BASEL0",
                0,
                {"scope": "ARTIFICIAL_ORCHESTRATION_ONLY"},
                {},
                {
                    role: {
                        "state": "COMPLETED",
                        "receipt_digest": "e" * 64,
                        "plan_digest": "f" * 64,
                        "receipt": {"path": "fixture", "bytes": 1},
                    }
                    for role in ("nonrl", "cash")
                },
            )

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, root, declaration, *args):
            future = Future(declaration.account)
            futures.append(future)
            return future

    monkeypatch.setattr(module.concurrent.futures, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(
        module.concurrent.futures, "as_completed", lambda roster: iter(tuple(roster))
    )
    root = tmp_path / "attempt"
    with pytest.raises(Exception):
        module.execute(root, tmp_path / "absent-dataset", object(), protocol)
    state = json.loads((root / "attempt.json").read_text())
    assert state["status"] == "FAILED"
    assert {row["state"] for row in state["arms"] if row["account"] == "B"} == {
        "COMPLETED"
    }
    assert {row["state"] for row in state["arms"] if row["account"] == "C"} == {
        "CANCELLED_BEFORE_START"
    }
    assert futures[-1].was_cancelled


def test_second_retained_corruption_keeps_the_completed_prefix(tmp_path, monkeypatch):
    module = api()
    module.save_gzip(
        tmp_path,
        "receipts/A_BASEL0_nonrl.json.gz",
        {"scope": "ARTIFICIAL_ORCHESTRATION_ONLY"},
    )
    corrupt = tmp_path / "receipts/A_BASEL0_cash.json.gz"
    corrupt.write_bytes(b"retained corrupt gzip")
    before = {path.name: path.read_bytes() for path in corrupt.parent.iterdir()}
    carrier_calls = []
    monkeypatch.setattr(module, "execution_folds", lambda *args: ())
    monkeypatch.setattr(
        module,
        "carrier",
        lambda *args, **kwargs: (
            carrier_calls.append(kwargs["role"]) or SimpleNamespace(execution_cost=None)
        ),
    )
    monkeypatch.setattr(
        module,
        "TerminalAllocationExecutionReceipt",
        SimpleNamespace(
            from_payload=lambda *args, **kwargs: SimpleNamespace(digest="c" * 64)
        ),
    )
    monkeypatch.setattr(module, "native_closure", lambda *args: {})
    monkeypatch.setattr(module, "ledger_decomposition", lambda *args: {})
    result = module._worker_execute_pair(
        tmp_path,
        pair("A"),
        SimpleNamespace(dataset=None),
        (),
        {
            "actor": {"vintage_starts": [1], "stop": 2},
            "configurations": [{"id": "fixture", "horizon_hours": 24}],
        },
    )
    assert result.arm_slots["nonrl"]["state"] == "COMPLETED"
    assert result.arm_slots["cash"]["state"] == "FAILED"
    assert result.failure["type"] == "BadGzipFile"
    assert carrier_calls == ["nonrl"]
    assert {path.name: path.read_bytes() for path in corrupt.parent.iterdir()} == before


def test_foreign_worker_identity_cannot_mutate_another_declared_pair(
    tmp_path, monkeypatch
):
    module = api()
    protocol = {
        "scope": "ARTIFICIAL_ORCHESTRATION_ONLY",
        "accounts": ["A", "B"],
        "scenarios": ["BASEL0"],
        "roles": ["nonrl", "cash"],
        "configurations": [{"id": "fixture", "horizon_hours": 24}],
        "concurrency": {"max_workers": 1},
    }
    root = tmp_path / "attempt"
    state = module.initialize_attempt(root, protocol)
    foreign = module.PairExecutionResult(
        "fixture",
        "B",
        "BASEL0",
        0,
        {},
        {},
        {
            role: {
                "state": "COMPLETED",
                "receipt_digest": "e" * 64,
                "plan_digest": "f" * 64,
                "receipt": {"path": "fixture", "bytes": 1},
            }
            for role in ("nonrl", "cash")
        },
    )
    monkeypatch.setattr(module, "_worker_execute_pair", lambda *args: foreign)
    monkeypatch.setattr(module, "symbol_estimates", lambda *args: ())
    inputs = SimpleNamespace(dataset=SimpleNamespace(symbols=("ARTIFICIAL",)))
    with pytest.raises(RuntimeError, match="identity"):
        module.execute_pairs(
            root, (pair("A"), pair("B")), {"fixture": inputs}, {}, protocol, state
        )
    assert {slot["state"] for slot in state["arms"] if slot["account"] == "B"} == {
        "UNATTEMPTED"
    }


def test_out_of_order_futures_publish_the_frozen_scenario_order(tmp_path, monkeypatch):
    module = api()
    scenarios = ["BASEL0", "COST2L0", "BASEL2"]
    pairs = tuple(pair("A") for _ in scenarios)
    for declaration, scenario in zip(pairs, scenarios):
        declaration.scenario = scenario
    protocol = {
        "scope": "ARTIFICIAL_ORDER_ONLY",
        "accounts": ["A"],
        "scenarios": scenarios,
        "roles": ["nonrl", "cash"],
        "configurations": [{"id": "fixture", "horizon_hours": 24}],
        "concurrency": {"max_workers": 3},
    }
    root = tmp_path / "attempt"
    state = module.initialize_attempt(root, protocol)
    inputs = SimpleNamespace(dataset=SimpleNamespace(symbols=("ARTIFICIAL",)))
    monkeypatch.setattr(module, "symbol_estimates", lambda *args: ())

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, root, declaration, *args):
            future = concurrent.futures.Future()
            future.set_result(
                module.PairExecutionResult(
                    "fixture",
                    "A",
                    declaration.scenario,
                    0,
                    {"scenario": declaration.scenario},
                    {},
                    {role: {"state": "COMPLETED"} for role in ("nonrl", "cash")},
                )
            )
            return future

    monkeypatch.setattr(module.concurrent.futures, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(
        module.concurrent.futures,
        "as_completed",
        lambda roster: iter(tuple(roster)[::-1]),
    )
    groups, _ = module.execute_pairs(
        root, pairs, {"fixture": inputs}, {}, protocol, state
    )
    assert [row["scenario"] for row in groups] == scenarios
    assert [
        row["scenario"] for row in json.loads((root / "pairs.json").read_text())
    ] == scenarios
