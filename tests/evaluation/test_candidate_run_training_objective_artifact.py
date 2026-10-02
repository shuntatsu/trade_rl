from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.evaluation.test_candidate_run_observation_contract import (
    _provenance,
    _result,
)
from trade_rl.evaluation.runs.artifact import (
    load_candidate_run_artifact,
    publish_candidate_run,
)
from trade_rl.strategies.rl.ppo_training import (
    ppo_training_objective_contract_payload,
)


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def test_new_candidate_artifact_binds_ppo_training_objective(tmp_path: Path) -> None:
    result = _result()
    result.spec.config.ppo_gamma = 0.9975
    result.spec.lean_config.ppo_gamma = 0.9975

    published = publish_candidate_run(
        tmp_path / "run",
        result,  # type: ignore[arg-type]
        _provenance(),
    )

    summary = json.loads(published.summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "lean_candidate_result_v8"
    assert summary["candidate_config"]["ppo_gamma"] == pytest.approx(0.9975)
    assert summary["ppo_training_objective"] == (
        ppo_training_objective_contract_payload(gamma=0.9975)
    )
    loaded = load_candidate_run_artifact(published.root)
    assert loaded.summary["ppo_training_objective"] == (
        ppo_training_objective_contract_payload(gamma=0.9975)
    )


def test_candidate_artifact_rejects_tampered_ppo_training_objective(
    tmp_path: Path,
) -> None:
    result = _result()
    result.spec.config.ppo_gamma = 0.9975
    result.spec.lean_config.ppo_gamma = 0.9975
    published = publish_candidate_run(
        tmp_path / "run",
        result,  # type: ignore[arg-type]
        _provenance(),
    )
    summary = json.loads(published.summary_path.read_text(encoding="utf-8"))
    objective = dict(summary["ppo_training_objective"])
    objective["gamma"] = 0.99
    summary["ppo_training_objective"] = objective
    _write_summary(published.summary_path, summary)

    with pytest.raises(ValueError, match="training objective"):
        load_candidate_run_artifact(published.root)


def test_legacy_candidate_schema_rejects_backfilled_training_objective(
    tmp_path: Path,
) -> None:
    result = _result()
    published = publish_candidate_run(
        tmp_path / "run",
        result,  # type: ignore[arg-type]
        _provenance(),
    )
    summary = json.loads(published.summary_path.read_text(encoding="utf-8"))
    summary["schema_version"] = "lean_candidate_result_v6"
    _write_summary(published.summary_path, summary)

    with pytest.raises(ValueError, match="legacy.*training objective"):
        load_candidate_run_artifact(published.root)
