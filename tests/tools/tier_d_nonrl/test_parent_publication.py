"""A parent write failure cannot discard already submitted worker observations."""

import concurrent.futures
from types import SimpleNamespace

import pytest

from tests.tools.tier_d_nonrl.test_outcomes import api, pair


def test_one_parent_publication_failure_still_drains_started_results(
    tmp_path, monkeypatch
):
    module = api()
    pairs = tuple(pair(name) for name in ("A", "B", "C"))
    protocol = {
        "scope": "ARTIFICIAL_PUBLICATION_DRAIN_ONLY",
        "source_head": "a" * 40,
        "accounts": ["A", "B", "C"],
        "scenarios": ["BASEL0"],
        "roles": ["nonrl", "cash"],
        "configurations": [{"id": "fixture", "horizon_hours": 24}],
        "concurrency": {"max_workers": 2},
    }
    inputs = SimpleNamespace(dataset=SimpleNamespace(symbols=("ARTIFICIAL",)))
    monkeypatch.setattr(module, "build_candidate_run_provenance", lambda: {})
    monkeypatch.setattr(
        module, "prepare_inputs", lambda *args: ({"fixture": inputs}, {})
    )
    monkeypatch.setattr(module, "declare_matrix", lambda *args: pairs)
    monkeypatch.setattr(
        module, "freeze_inputs", lambda *args: {"scope": protocol["scope"]}
    )
    monkeypatch.setattr(module, "symbol_estimates", lambda *args: ())
    monkeypatch.setattr(
        module,
        "SettledNonRLComparisonPlan",
        lambda *args: SimpleNamespace(digest="a" * 64),
    )
    submitted, retained = [], {}

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, root, declaration, *args):
            submitted.append(declaration.account)
            slots = {}
            for role in ("nonrl", "cash"):
                relative = f"receipts/{declaration.account}_{role}.json.gz"
                metadata = module.save_gzip(
                    root, relative, {"scope": protocol["scope"]}
                )
                retained[relative] = (root / relative).read_bytes()
                slots[role] = {
                    "state": "COMPLETED",
                    "receipt": metadata,
                    "receipt_digest": "b" * 64,
                    "plan_digest": "c" * 64,
                }
            future = concurrent.futures.Future()
            future.set_result(
                module.PairExecutionResult(
                    "fixture",
                    declaration.account,
                    "BASEL0",
                    0,
                    {"scope": protocol["scope"]},
                    {},
                    slots,
                )
            )
            return future

    monkeypatch.setattr(module.concurrent.futures, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(
        module.concurrent.futures, "as_completed", lambda roster: iter(tuple(roster))
    )
    original, failed = module.write_json, False

    def write(path, payload, **kwargs):
        nonlocal failed
        if path.name == "pairs.json" and not failed:
            failed = True
            raise OSError("manufactured one-shot parent publication failure")
        return original(path, payload, **kwargs)

    monkeypatch.setattr(module, "write_json", write)
    root = tmp_path / "attempt"
    with pytest.raises((RuntimeError, OSError), match="publication"):
        module.execute(root, tmp_path / "absent-input", object(), protocol)
    state = module.read_json(root / "attempt.json")
    assert state["status"] == "FAILED"
    assert submitted == ["A", "B"]
    assert {s["state"] for s in state["arms"] if s["account"] in {"A", "B"}} == {
        "COMPLETED"
    }
    assert {s["state"] for s in state["arms"] if s["account"] == "C"} == {"UNATTEMPTED"}
    assert all((root / name).read_bytes() == value for name, value in retained.items())
