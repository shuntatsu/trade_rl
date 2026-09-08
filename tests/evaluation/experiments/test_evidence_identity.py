from __future__ import annotations

from pathlib import Path

import pytest

from tests.evaluation.experiments.test_evidence import (
    _config,
    _dataset,
    _fake_execute,
    _plan,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import publish_market_dataset_artifact
from trade_rl.evaluation.experiments.evidence import execute_evidence_set
from trade_rl.evaluation.experiments.store import StudyStore


def test_evidence_set_identity_binds_seed_normalized_semantic_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.experiments import evidence as evidence_module

    dataset = _dataset()
    dataset_root = tmp_path / "dataset"
    artifact = publish_market_dataset_artifact(dataset_root, dataset)
    plan = _plan(dataset, artifact)
    monkeypatch.setattr(evidence_module, "execute_candidate_run", _fake_execute())
    context = content_digest({"kind": "baseline", "study": plan.digest})

    result = execute_evidence_set(
        store=StudyStore(tmp_path / "study"),
        target=Path("evidence/baseline"),
        dataset_root=dataset_root,
        plan=plan,
        config=_config(),
        research_context_digest=context,
    )

    assert result.semantic_config_digest == plan.baseline_config.digest
