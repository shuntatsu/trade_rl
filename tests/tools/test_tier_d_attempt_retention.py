"""Attempt retention boundaries using only temporary JSON/gzip bytes."""

from __future__ import annotations

import gzip
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def driver(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "tools/tier_d_nonrl"
    module = importlib.import_module("tools.tier_d_nonrl.tier_d_runner")
    assert Path(module.__file__).resolve().parent == directory

    def forbidden(*args, **kwargs):
        raise AssertionError("retention test must never load data, fit, or execute")

    for name in (
        "load_market_dataset_artifact",
        "prepare_allocation_market_inputs",
        "run_terminal_allocation_execution",
        "build_candidate_run_provenance",
    ):
        monkeypatch.setattr(module, name, forbidden)
    return module


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_resume_cannot_reopen_an_existing_attempt_in_place(driver, tmp_path):
    protocol = {
        "scope": "ARTIFICIAL_FILE_RETENTION_ONLY",
        "accounts": ["A"],
        "scenarios": ["BASEL0"],
        "roles": ["nonrl", "cash"],
    }
    root = tmp_path / "attempt"
    driver.initialize_attempt(root, protocol)
    (root / "provenance.json").write_text(
        json.dumps({"implementation": "original"}), encoding="utf-8"
    )
    before = snapshot(root)

    with pytest.raises(FileExistsError):
        driver.initialize_attempt(root, protocol, resume=True)

    assert snapshot(root) == before


def test_receipt_publication_is_exclusive_and_preserves_prior_bytes(driver, tmp_path):
    relative = "receipts/A_BASEL0_nonrl.json.gz"
    driver.save_gzip(tmp_path, relative, {"original": True})
    before = snapshot(tmp_path)

    with pytest.raises(FileExistsError):
        driver.save_gzip(tmp_path, relative, {"replacement": True})

    assert snapshot(tmp_path) == before


def test_corrupt_retained_receipt_is_rejected_before_environment_work(
    driver, tmp_path, monkeypatch
):
    receipt = tmp_path / "receipts/A_BASEL0_nonrl.json.gz"
    receipt.parent.mkdir()
    receipt.write_bytes(b"not a gzip archive; preserved failed-attempt evidence")
    before = snapshot(tmp_path)
    monkeypatch.setattr(driver, "execution_folds", lambda *args: ())

    def forbidden_environment(*args, **kwargs):
        raise AssertionError(
            "corrupt retained receipt reached environment construction"
        )

    monkeypatch.setattr(driver, "carrier", forbidden_environment)
    pair = SimpleNamespace(
        account="A",
        scenario="BASEL0",
        configuration="H24_A1",
        symbol_index=0,
        controls=SimpleNamespace(plans=(object(), object())),
    )
    protocol = {
        "actor": {"vintage_starts": [1], "stop": 2},
        "configurations": [{"id": "H24_A1", "horizon_hours": 24}],
    }

    with pytest.raises(gzip.BadGzipFile):
        driver._worker_execute_pair(tmp_path, pair, object(), (), protocol)

    assert snapshot(tmp_path) == before


def test_existing_destination_without_attempt_index_is_never_overwritten(
    driver, tmp_path
):
    protocol = {
        "scope": "ARTIFICIAL_FILE_RETENTION_ONLY",
        "accounts": ["A"],
        "scenarios": ["BASEL0"],
        "roles": ["nonrl", "cash"],
    }
    (tmp_path / "protocol.json").write_text('{"original":true}', encoding="utf-8")
    before = snapshot(tmp_path)
    with pytest.raises(FileExistsError):
        driver.initialize_attempt(tmp_path, protocol)
    assert snapshot(tmp_path) == before


def test_failed_index_serialization_preserves_the_previous_json(driver, tmp_path):
    path = tmp_path / "attempt.json"
    driver.write_json(path, {"status": "FAILED", "retained": True})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        driver.write_json(path, {"nonfinite": float("nan")}, replace=True)
    assert path.read_bytes() == before
