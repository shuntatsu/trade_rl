"""Frozen plan identity must survive independently re-sealed member artifacts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.evaluation.experiments.test_evidence_identity import (
    _published_evidence,
    _published_study_baseline,
    _rebind_evidence_manifest,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments import inspect_study
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.evidence import load_evidence_set


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _alter_provenance(member: Path, kind: str) -> None:
    path = member / "provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    field = "implementation" if kind == "implementation" else "runtime_environment"
    payload[field]["synthetic_foreign_identity"] = True
    payload[field + "_digest"] = content_digest(payload[field])
    _write(path, payload)


@pytest.mark.parametrize("kind", ["implementation", "runtime"])
@pytest.mark.parametrize("all_members", [False, True])
def test_frozen_plan_rejects_resealed_foreign_provenance(
    tmp_path, monkeypatch, kind, all_members
):
    store, plan, _ = _published_evidence(tmp_path, monkeypatch)
    root = store.root / "evidence" / "baseline"
    seeds = plan.ppo_seeds if all_members else plan.ppo_seeds[:1]
    for seed in seeds:
        _alter_provenance(root / "runs" / f"seed-{seed}", kind)
    _rebind_evidence_manifest(root)
    with pytest.raises(ArtifactIntegrityError, match=f"{kind} provenance"):
        load_evidence_set(root, expected_plan=plan)


@pytest.mark.parametrize("kind", ["implementation", "runtime"])
def test_unpinned_evidence_still_requires_cross_seed_provenance(
    tmp_path, monkeypatch, kind
):
    store, plan, _ = _published_evidence(tmp_path, monkeypatch)
    root = store.root / "evidence" / "baseline"
    _alter_provenance(root / "runs" / f"seed-{plan.ppo_seeds[-1]}", kind)
    _rebind_evidence_manifest(root)
    with pytest.raises(ArtifactIntegrityError, match=f"{kind} provenance"):
        load_evidence_set(root)


def test_frozen_plan_rejects_reordered_seed_roster(tmp_path, monkeypatch):
    store, plan, _ = _published_evidence(tmp_path, monkeypatch)
    root = store.root / "evidence" / "baseline"
    path = root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["ppo_seeds"].reverse()
    payload["run_digests"].reverse()
    _write(path, payload)
    _rebind_evidence_manifest(root)
    # Without a frozen plan, a coherent explicitly ordered roster is legal.
    assert load_evidence_set(root).evidence.ppo_seeds == plan.ppo_seeds[::-1]
    with pytest.raises(ArtifactIntegrityError, match="seed roster"):
        load_evidence_set(root, expected_plan=plan)


def test_frozen_plan_rejects_wrong_symbol_roster(tmp_path, monkeypatch):
    store, plan, _ = _published_evidence(tmp_path, monkeypatch)
    root = store.root / "evidence" / "baseline"
    # The separately frozen plan, not a claim in the first member, owns
    # the expected roster. All other pins are intentionally unchanged.
    wrong = replace(plan, symbols=("FOREIGN",))
    with pytest.raises(ArtifactIntegrityError, match="symbol roster"):
        load_evidence_set(root, expected_plan=wrong)


def test_study_inspection_rejects_rebound_baseline_provenance(tmp_path, monkeypatch):
    from trade_rl.evaluation.experiments.codec import _analysis_binding

    root, _ = _published_study_baseline(tmp_path, monkeypatch)
    snapshot = inspect_study(root)
    assert snapshot.baseline is not None
    evidence = root / "baseline" / "evidence"
    for seed in snapshot.plan.ppo_seeds:
        _alter_provenance(evidence / "runs" / f"seed-{seed}", "implementation")
    _rebind_evidence_manifest(evidence)
    manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    analysis_path = root / "baseline" / "analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))["analysis"]
    _write(
        analysis_path,
        _analysis_binding(
            evidence_fingerprint=manifest["fingerprint"], analysis=analysis
        ),
    )
    with pytest.raises(ArtifactIntegrityError, match="baseline EvidenceSet"):
        inspect_study(root)


def test_publication_rejects_resealed_provenance_before_final_evidence(
    tmp_path, monkeypatch
):
    from tests.evaluation.experiments.test_evidence import (
        _config,
        _dataset,
        _fake_execute,
        _plan,
    )
    from trade_rl.data import publish_market_dataset_artifact
    from trade_rl.evaluation.experiments import evidence as owner
    from trade_rl.evaluation.experiments.store import StudyStore

    dataset = _dataset()
    dataset_root = tmp_path / "dataset"
    artifact = publish_market_dataset_artifact(dataset_root, dataset)
    plan = _plan(dataset, artifact)
    store = StudyStore(tmp_path / "study")
    original = owner.publish_candidate_run

    def changed_publish(root, result, provenance):
        published = original(root, result, provenance)
        _alter_provenance(Path(root), "implementation")
        return published

    monkeypatch.setattr(owner, "execute_candidate_run", _fake_execute())
    monkeypatch.setattr(owner, "publish_candidate_run", changed_publish)
    target = Path("evidence/baseline")
    with pytest.raises(ArtifactIntegrityError, match="implementation provenance"):
        owner.execute_evidence_set(
            store=store,
            target=target,
            dataset_root=dataset_root,
            plan=plan,
            config=_config(),
            research_context_digest=content_digest({"study": plan.digest}),
        )
    assert not (store.root / target).exists()
    assert not any(".staging-" in p.name for p in store.root.rglob("*"))


def test_archived_consistent_provenance_is_checked_against_its_frozen_plan(
    tmp_path, monkeypatch
):
    store, plan, _ = _published_evidence(tmp_path, monkeypatch)
    root = store.root / "evidence" / "baseline"
    for seed in plan.ppo_seeds:
        member = root / "runs" / f"seed-{seed}"
        _alter_provenance(member, "implementation")
        _alter_provenance(member, "runtime")
    _rebind_evidence_manifest(root)
    manifest = json.loads(
        (root / "runs" / f"seed-{plan.ppo_seeds[0]}" / "provenance.json").read_text(
            encoding="utf-8"
        )
    )
    historical_plan = replace(
        plan,
        implementation_digest=manifest["implementation_digest"],
        runtime_environment_digest=manifest["runtime_environment_digest"],
    )
    assert historical_plan.implementation_digest != plan.implementation_digest
    assert historical_plan.runtime_environment_digest != plan.runtime_environment_digest
    assert load_evidence_set(root, expected_plan=historical_plan).evidence == (
        load_evidence_set(root).evidence
    )
