from __future__ import annotations

import json
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
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.evidence import (
    execute_evidence_set,
    load_evidence_set,
)
from trade_rl.evaluation.experiments.store import StudyStore


def _published_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
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
    return store, plan, result


def test_evidence_set_identity_binds_seedless_semantic_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, plan, result = _published_evidence(tmp_path, monkeypatch)
    loaded = load_evidence_set(store.root / "evidence" / "baseline")

    assert "ppo_seed" not in loaded.semantic_config
    assert result.semantic_config_digest == content_digest(loaded.semantic_config)
    assert [
        loaded.runs[seed].summary["candidate_config"]["ppo_seed"]
        for seed in plan.ppo_seeds
    ] == list(plan.ppo_seeds)


def test_evidence_loader_rejects_self_consistent_manifest_that_reintroduces_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, plan, _ = _published_evidence(tmp_path, monkeypatch)
    evidence_root = store.root / "evidence" / "baseline"
    manifest_path = evidence_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    semantic_config = dict(manifest["semantic_config"])
    semantic_config["ppo_seed"] = plan.ppo_seeds[0]
    semantic_digest = content_digest(semantic_config)
    manifest["semantic_config"] = semantic_config
    manifest["semantic_config_digest"] = semantic_digest
    manifest["fingerprint"] = content_digest(
        {
            "schema_version": manifest["schema_version"],
            "semantic_config_digest": semantic_digest,
            "ppo_seeds": manifest["ppo_seeds"],
            "run_digests": manifest["run_digests"],
            "research_context_digest": manifest["research_context_digest"],
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactIntegrityError, match="ppo_seed|semantic config"):
        load_evidence_set(evidence_root)
