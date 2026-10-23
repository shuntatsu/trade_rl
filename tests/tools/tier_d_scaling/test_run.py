"""Prefix facts survive validation/publication failures without completion claims."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.tier_d_nonrl.tier_d_records import load_gzip, read_json


def _module(monkeypatch):
    module = importlib.import_module("tools.tier_d_scaling.run")
    protocol = {
        "scope": "ARTIFICIAL_SOFTWARE_PROFILE_ONLY",
        "accounts": ["A"],
        "scenarios": ["BASEL0"],
        "configurations": [{"id": "H72_RIDGE_A10"}],
        "roles": ["nonrl", "cash"],
        "progress_policy": {
            "schema": "issue810_native_progress_v1",
            "sample_seconds": 1,
            "no_progress_seconds": 600,
        },
        "actor": {"start": 2903, "stop": 9430, "vintage_starts": [2903, 5087, 7295]},
        "terminal": {"stop_inclusive": 9502},
        "measurement": {"cases": [{"name": "n100", "target": 100, "profile": False}]},
    }
    plan = SimpleNamespace(digest="a" * 64)
    pair = SimpleNamespace(
        configuration="H72_RIDGE_A10",
        account="A",
        scenario="BASEL0",
        symbol_index=0,
        controls=SimpleNamespace(nonrl_plan=plan, cash_plan=plan),
    )
    inputs = SimpleNamespace(costs_by_symbol=((), ()))
    context = SimpleNamespace(
        protocol=protocol, envelope={"native": {}}, inputs=inputs, pair=pair
    )
    monkeypatch.setattr(module, "load_prepared", lambda root: context)
    monkeypatch.setattr(module, "validate_profile_source", lambda protocol: None)
    monkeypatch.setattr(module, "carrier", lambda *a, **k: object())
    monkeypatch.setattr(module, "execution_folds", lambda *a: ())

    def native(folds, env, plan, *, on_advance):
        try:
            for index in range(2903, 3003):
                on_advance("actor", index, index + 1)
        except Exception as error:
            error.terminal_execution_receipt = {
                "status": "integrity_failure",
                "frontier": index + 1,
            }
            raise

    monkeypatch.setattr(module, "run_terminal_allocation_execution", native)
    monkeypatch.setattr(
        module.TerminalAllocationExecutionReceipt,
        "from_payload",
        lambda raw, **k: SimpleNamespace(digest="b" * 64),
    )
    return module


def test_expected_prefix_keeps_failed_raw_and_cash_unattempted(
    tmp_path: Path, monkeypatch
):
    module = _module(monkeypatch)
    root = tmp_path / "case"
    report = module.measure(root, tmp_path / "prepared", "n100")
    assert report["status"] == "PREFIX_OBSERVED"
    assert report["count"] == 100 and report["processing_index"] == 3003
    assert report["financial_authority"] is False
    assert load_gzip(root / "receipt.partial.json.gz") == {
        "status": "integrity_failure",
        "frontier": 3003,
    }
    state = read_json(root / "attempt.json")
    assert [arm["state"] for arm in state["arms"]] == ["FAILED", "UNATTEMPTED"]
    assert state["status"] == "FAILED"
    assert read_json(root / "output-manifest.json")


def test_reader_failure_does_not_lose_raw_or_observed_count(
    tmp_path: Path, monkeypatch
):
    module = _module(monkeypatch)

    def invalid(*a, **k):
        raise ValueError("strict evidence rejected")

    monkeypatch.setattr(
        module.TerminalAllocationExecutionReceipt, "from_payload", invalid
    )
    report = module.measure(tmp_path / "case", tmp_path / "prepared", "n100")
    assert report["status"] == "FAILED"
    assert report["count"] == 100
    assert report["readback_failure"]["message"] == "strict evidence rejected"
    assert load_gzip(tmp_path / "case/receipt.partial.json.gz")["frontier"] == 3003


def test_case_cannot_write_into_prepared_inputs(tmp_path: Path, monkeypatch):
    module = _module(monkeypatch)
    root = tmp_path / "prepared"
    root.mkdir()
    with pytest.raises(ValueError, match="disjoint"):
        module.measure(root / "child", root, "n100")
    assert list(root.iterdir()) == []


def test_source_postflight_failure_preserves_partial_facts(tmp_path: Path, monkeypatch):
    module = _module(monkeypatch)

    def drift(protocol):
        raise ValueError("source changed during native prefix")

    monkeypatch.setattr(module, "validate_profile_source", drift)
    report = module.measure(tmp_path / "case", tmp_path / "prepared", "n100")
    assert report["status"] == "FAILED"
    assert report["count"] == 100
    assert report["source_postflight_failure"]["type"] == "ValueError"
    assert load_gzip(tmp_path / "case/receipt.partial.json.gz")["frontier"] == 3003


def test_returned_native_facts_survive_journal_exit_failure(
    tmp_path: Path, monkeypatch
):
    module = _module(monkeypatch)
    returned = SimpleNamespace(payload={"status": "economic_stop", "frontier": 2903})
    monkeypatch.setattr(
        module, "run_terminal_allocation_execution", lambda *a, **k: returned
    )
    original = module.ProgressJournal.__exit__

    def failed_exit(self, *args):
        original(self, *args)
        raise OSError("journal end unavailable")

    monkeypatch.setattr(module.ProgressJournal, "__exit__", failed_exit)
    root = tmp_path / "case"
    report = module.measure(root, tmp_path / "prepared", "n100")
    assert report["status"] == "FAILED"
    assert report["count"] == 0
    assert load_gzip(root / "receipt.partial.json.gz") == returned.payload
    assert (
        report["progress_publication_failure"]["message"] == "journal end unavailable"
    )
