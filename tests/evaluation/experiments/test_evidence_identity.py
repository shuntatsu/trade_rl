from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from tests.evaluation.experiments.test_evidence import (
    _config,
    _dataset,
    _fake_execute,
    _plan,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import load_market_dataset_artifact, publish_market_dataset_artifact
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.evidence import (
    execute_evidence_set,
    load_evidence_set,
)
from trade_rl.evaluation.experiments.store import StudyStore
from trade_rl.evaluation.runs import (
    build_candidate_run_provenance,
    inspect_candidate_run_artifact,
    load_candidate_run_artifact,
    publish_candidate_run,
    resolve_candidate_run_spec,
)


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


def test_evidence_loader_rejects_valid_member_run_with_different_resolved_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.experiments import evidence as evidence_module

    store, plan, _ = _published_evidence(tmp_path, monkeypatch)
    evidence_root = store.root / "evidence" / "baseline"
    dataset = load_market_dataset_artifact(tmp_path / "dataset")
    context = content_digest({"kind": "baseline", "study": plan.digest})
    changed_config = replace(
        _config(),
        ppo_seed=plan.ppo_seeds[0],
        forecast_entry_threshold=0.02,
    )
    spec = resolve_candidate_run_spec(
        dataset,
        dataset_artifact_schema=plan.dataset_artifact_schema,
        dataset_artifact_digest=plan.dataset_artifact_digest,
        config=changed_config,
        execution_overlay=plan.baseline_config.execution_overlay,
    )
    wrong_run_root = tmp_path / "self-consistent-wrong-config-run"
    publish_candidate_run(
        wrong_run_root,
        evidence_module.execute_candidate_run(dataset, spec),
        build_candidate_run_provenance(research_context_digest=context),
    )
    wrong_run = load_candidate_run_artifact(wrong_run_root)
    assert wrong_run.summary["candidate_config"]["forecast_entry_threshold"] == 0.02

    member_root = evidence_root / "runs" / f"seed-{plan.ppo_seeds[0]}"
    shutil.rmtree(member_root)
    shutil.copytree(wrong_run_root, member_root)

    manifest_path = evidence_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = inspect_candidate_run_artifact(member_root)
    for run_digest in manifest["run_digests"]:
        if run_digest["ppo_seed"] == plan.ppo_seeds[0]:
            run_digest["artifact_digest"] = identity.artifact_digest
    manifest["fingerprint"] = evidence_module._evidence_fingerprint(
        semantic_config_digest=manifest["semantic_config_digest"],
        ppo_seeds=tuple(manifest["ppo_seeds"]),
        run_digests=tuple(
            (entry["ppo_seed"], entry["artifact_digest"])
            for entry in manifest["run_digests"]
        ),
        research_context_digest=manifest["research_context_digest"],
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactIntegrityError, match="config"):
        load_evidence_set(evidence_root)


def test_evidence_loader_binds_each_run_dataset_to_study_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, plan, _ = _published_evidence(tmp_path, monkeypatch)
    evidence_root = store.root / "evidence" / "baseline"
    manifest_path = evidence_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_digests = {entry["ppo_seed"]: entry for entry in manifest["run_digests"]}

    for seed in plan.ppo_seeds:
        member_root = evidence_root / "runs" / f"seed-{seed}"
        summary_path = member_root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["dataset_id"] = "f" * 64
        summary["dataset_artifact"]["artifact_digest"] = "e" * 64
        summary_path.write_text(
            json.dumps(summary, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        run_digests[seed]["artifact_digest"] = inspect_candidate_run_artifact(
            member_root
        ).artifact_digest

    from trade_rl.evaluation.experiments import evidence as evidence_module

    manifest["fingerprint"] = evidence_module._evidence_fingerprint(
        semantic_config_digest=manifest["semantic_config_digest"],
        ppo_seeds=tuple(manifest["ppo_seeds"]),
        run_digests=tuple(
            (entry["ppo_seed"], entry["artifact_digest"])
            for entry in manifest["run_digests"]
        ),
        research_context_digest=manifest["research_context_digest"],
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactIntegrityError, match="dataset identity"):
        load_evidence_set(evidence_root, expected_plan=plan)
