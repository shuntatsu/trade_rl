"""Post-worker failures retain observed outcomes; no Dataset/native execution."""

import importlib
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "stage", ["aggregation", "result", "completion_log", "manifest"]
)
def test_final_publication_failure_marks_failed_and_retains_completed_slots(
    tmp_path, monkeypatch, stage
):
    module = importlib.import_module("tools.tier_d_nonrl.tier_d_runner")
    root = tmp_path / "attempt"
    protocol = {
        "scope": "ARTIFICIAL_FINALIZATION_ONLY",
        "source_head": "a" * 40,
        "accounts": ["A"],
        "scenarios": ["BASEL0"],
        "roles": ["nonrl", "cash"],
        "configurations": [{"id": "fixture"}],
        "budget": {},
        "decision": {},
    }
    dataset = SimpleNamespace(n_bars=0, n_symbols=0, n_features=0)
    inputs = SimpleNamespace(stream=SimpleNamespace(vintages=()))
    monkeypatch.setattr(module, "build_candidate_run_provenance", lambda: {})
    monkeypatch.setattr(
        module, "prepare_inputs", lambda *args: ({"fixture": inputs}, {})
    )
    monkeypatch.setattr(
        module, "declare_matrix", lambda *args: (SimpleNamespace(controls=None),)
    )
    monkeypatch.setattr(
        module,
        "SettledNonRLComparisonPlan",
        lambda *args: SimpleNamespace(digest="b" * 64),
    )
    monkeypatch.setattr(
        module, "freeze_inputs", lambda *args: {"scope": protocol["scope"]}
    )
    retained = {}

    def completed(root, pairs, prepared, tables, protocol, state):
        for slot in state["arms"]:
            relative = f"receipts/{slot['role']}.json.gz"
            metadata = module.save_gzip(root, relative, {"scope": protocol["scope"]})
            retained[relative] = (root / relative).read_bytes()
            slot.update(state="COMPLETED", receipt=metadata, execution_started=False)
        return [{"comparison_complete": True}], {}

    monkeypatch.setattr(module, "execute_pairs", completed)
    if stage == "aggregation":

        def aggregate(*args):
            raise ValueError("manufactured aggregation failure")

        monkeypatch.setattr(module, "combined_rows", aggregate)
    else:
        monkeypatch.setattr(module, "combined_rows", lambda *args: [])
    original_write, original_log = module.write_json, module.append_log
    failed = False

    def write(path, payload, **kwargs):
        nonlocal failed
        if not failed and (
            (stage == "result" and path.name == "result.json")
            or (stage == "manifest" and path.name == "output-manifest.json")
        ):
            failed = True
            raise OSError(f"manufactured {stage} failure")
        return original_write(path, payload, **kwargs)

    def log(root, event, **kwargs):
        if stage == "completion_log" and event == "attempt_completed":
            raise OSError("manufactured completion_log failure")
        return original_log(root, event, **kwargs)

    monkeypatch.setattr(module, "write_json", write)
    monkeypatch.setattr(module, "append_log", log)
    with pytest.raises((ValueError, OSError), match="manufactured"):
        module.execute(root, tmp_path / "absent-input", dataset, protocol)
    state = module.read_json(root / "attempt.json")
    assert state["status"] == "FAILED"
    assert all(slot["state"] == "COMPLETED" for slot in state["arms"])
    assert stage in state["failure"]["message"]
    assert all(
        (root / relative).read_bytes() == raw for relative, raw in retained.items()
    )
    assert "attempt_failed" in (root / "progress.jsonl").read_text()
