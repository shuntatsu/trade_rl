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
from trade_rl.evaluation.experiments.evidence import (
    execute_evidence_set,
    load_evidence_set,
)
from trade_rl.evaluation.experiments.store import StudyStore


def test_evidence_set_identity_binds_seedless_semantic_config(
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

    store = StudyStore(tmp_path / "study")
    result = execute_evidence_set(
        store=store,
        target=Path("evidence/baseline"),
        dataset_root=dataset_root,
        plan=plan,
        config=_config(),
        research_context_digest=context,
    )
    loaded = load_evidence_set(store.root / "evidence" / "baseline")

    assert "ppo_seed" not in loaded.semantic_config
    assert result.semantic_config_digest == content_digest(loaded.semantic_config)
    assert [
        loaded.runs[seed].summary["candidate_config"]["ppo_seed"]
        for seed in plan.ppo_seeds
    ] == list(plan.ppo_seeds)
