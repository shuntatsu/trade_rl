from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from tests.evaluation.test_candidate_run_observation_contract import (
    _provenance,
    _result,
)
from trade_rl.evaluation.runs.artifact import (
    LoadedCandidateRun,
    inspect_candidate_run_artifact,
    load_candidate_run_artifact,
    publish_candidate_run,
)


@pytest.mark.parametrize(
    "mutation",
    [
        "empty_symbols",
        "missing_row",
        "empty_strategies",
        "wrong_symbol",
        "wrong_index",
        "duplicate_strategy",
        "different_strategy_rosters",
    ],
)
@pytest.mark.parametrize("version", [6, 8, 10])
def test_candidate_rejects_incomplete_coverage_roster(
    tmp_path: Path, mutation: str, version: int, monkeypatch
) -> None:
    artifact = publish_candidate_run(
        tmp_path / "run",
        _result(),
        _provenance(),
    )
    summary = json.loads(artifact.summary_path.read_text(encoding="utf-8"))
    summary["schema_version"] = f"lean_candidate_result_v{version}"
    if version < 10:
        summary["candidate_config"].pop("forecast_switch_cost")
    if version < 8:
        summary["candidate_config"].pop("supervised_forecast_horizon_hours")

    with np.load(artifact.returns_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    artifact.summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert load_candidate_run_artifact(
        artifact.root
    ).has_verified_full_evaluation_coverage

    row = summary["by_symbol"][0]
    strategy = row["strategies"][0]
    if mutation == "empty_symbols":
        summary["symbols"] = []
        summary["by_symbol"] = []
        arrays = {}
    elif mutation == "missing_row":
        summary["by_symbol"] = []
        arrays = {}
    elif mutation == "empty_strategies":
        row["strategies"] = []
        arrays = {}
    elif mutation == "wrong_symbol":
        row["symbol"] = "ETHUSDT"
    elif mutation == "wrong_index":
        row["symbol_index"] = True
    elif mutation == "duplicate_strategy":
        duplicate = deepcopy(strategy)
        duplicate["return_key"] = "duplicate"
        row["strategies"].append(duplicate)
        arrays["duplicate"] = arrays[strategy["return_key"]].copy()
    elif mutation == "different_strategy_rosters":
        summary["symbols"].append("ETHUSDT")
        other = deepcopy(row)
        other["symbol"] = "ETHUSDT"
        other["symbol_index"] = 1
        other["strategies"][0]["name"] = "different"
        other["strategies"][0]["return_key"] = "other"
        strategy["final_quantities"].append(0.0)
        other["strategies"][0]["final_quantities"].append(0.0)
        summary["by_symbol"].append(other)
        arrays["other"] = arrays[strategy["return_key"]].copy()

    artifact.summary_path.write_text(json.dumps(summary), encoding="utf-8")
    np.savez_compressed(artifact.returns_path, **arrays)
    run = LoadedCandidateRun(
        root=artifact.root,
        summary=summary,
        returns=arrays,
        provenance={},
    )
    assert not run.has_verified_full_evaluation_coverage

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("malformed roster reached return array decoding")

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", forbidden_decode)
    for reader in (load_candidate_run_artifact, inspect_candidate_run_artifact):
        with pytest.raises(ValueError, match="candidate coverage"):
            reader(artifact.root)


def test_candidate_coverage_requires_exact_return_keys(tmp_path: Path) -> None:
    artifact = publish_candidate_run(tmp_path / "run", _result(), _provenance())
    loaded = load_candidate_run_artifact(artifact.root)
    assert loaded.has_verified_full_evaluation_coverage
    for arrays in ({}, {**loaded.returns, "extra": np.zeros(1)}):
        run = LoadedCandidateRun(
            root=loaded.root,
            summary=loaded.summary,
            returns=arrays,
            provenance=loaded.provenance,
        )
        assert not run.has_verified_full_evaluation_coverage


def test_candidate_writer_rejects_missing_symbol_before_publication(
    tmp_path: Path,
) -> None:
    result = _result()
    result.comparison.by_symbol = ()
    destination = tmp_path / "run"
    with pytest.raises(ValueError, match="candidate coverage symbol roster"):
        publish_candidate_run(destination, result, _provenance())
    assert not destination.exists()


@pytest.mark.parametrize("schema", [[], {}])
def test_candidate_rejects_unhashable_schema(tmp_path: Path, schema) -> None:
    artifact = publish_candidate_run(tmp_path / "run", _result(), _provenance())
    summary = json.loads(artifact.summary_path.read_text(encoding="utf-8"))
    summary["schema_version"] = schema
    artifact.summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported candidate result schema"):
        load_candidate_run_artifact(artifact.root)
    run = LoadedCandidateRun(
        root=artifact.root, summary=summary, returns={}, provenance={}
    )
    assert not run.has_verified_full_evaluation_coverage
