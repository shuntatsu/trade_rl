"""Synthetic adversarial readback changes at the EvidenceSet identity boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.evaluation.experiments.test_evidence import (
    _config,
    _dataset,
    _fake_execute,
    _plan,
)
from tests.evaluation.experiments.test_evidence_identity import _published_evidence
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import (
    inspect_published_market_dataset_artifact,
    load_published_market_dataset_artifact,
    publish_market_dataset_artifact,
)
from trade_rl.evaluation.experiments import evidence as evidence_module
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.store import StudyStore
from trade_rl.evaluation.runs import (
    inspect_candidate_run_artifact,
    load_candidate_run_artifact,
)


def test_evidence_cannot_bind_loaded_run_to_later_restored_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, plan, published = _published_evidence(tmp_path, monkeypatch)
    root = store.root / "evidence" / "baseline"
    member = root / "runs" / f"seed-{plan.ppo_seeds[0]}"
    original = load_candidate_run_artifact(member)
    key = next(
        entry["return_key"]
        for entry in original.summary["by_symbol"][0]["strategies"]
        if entry["name"] == "ppo"
    )
    path = member / "returns.npz"
    good_bytes = path.read_bytes()
    # The alternate snapshot remains a structurally valid Candidate Run.
    # Modify only PPO's raw series, so the deterministic seed check cannot
    # accidentally be the rejection oracle.
    values = {name: data.copy() for name, data in original.returns.items()}
    values[key][0] += 0.001
    with path.open("wb") as handle:
        np.savez_compressed(handle, **values)
    altered = load_candidate_run_artifact(member)
    assert altered.returns[key][0] != original.returns[key][0]
    calls = []

    def restore_after_load(candidate_root):
        loaded = load_candidate_run_artifact(candidate_root)
        if Path(candidate_root) == member:
            calls.append(True)
            assert loaded.returns[key][0] == altered.returns[key][0]
            path.write_bytes(good_bytes)
        return loaded

    monkeypatch.setattr(
        evidence_module, "load_candidate_run_artifact", restore_after_load
    )
    with pytest.raises(ArtifactIntegrityError, match="Run artifact digest mismatch"):
        evidence_module.load_evidence_set(root, expected_plan=plan)
    assert calls == [True]
    assert path.read_bytes() == good_bytes
    monkeypatch.undo()
    assert (
        evidence_module.load_evidence_set(root, expected_plan=plan).evidence
        == published
    )


def test_plan_inputs_do_not_mix_dataset_manifest_and_array_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset()
    root = tmp_path / "dataset"
    artifact = publish_market_dataset_artifact(root, dataset)
    plan = _plan(dataset, artifact)
    manifest_path = root / "manifest.json"
    arrays_path = root / "arrays.npz"
    good_manifest = manifest_path.read_bytes()
    good_arrays = arrays_path.read_bytes()
    # Repack identical arrays to produce a different artifact identity, while
    # retaining the same valid Dataset content identity.
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    with arrays_path.open("wb") as handle:
        np.savez(handle, **arrays)
    other_arrays = arrays_path.read_bytes()
    assert other_arrays != good_arrays
    payload = json.loads(good_manifest)
    payload["arrays_digest"] = hashlib.sha256(other_arrays).hexdigest()
    payload.pop("artifact_digest")
    payload["artifact_digest"] = content_digest(payload)
    other_manifest = json.dumps(payload).encode()
    manifest_path.write_bytes(other_manifest)
    other_artifact, other_dataset = load_published_market_dataset_artifact(root)
    assert other_dataset.dataset_id == dataset.dataset_id
    assert other_artifact.artifact_digest != artifact.artifact_digest

    def earlier_inspection(candidate_root):
        # Emulate the earlier valid inspection, followed by replacement
        # before the separate Dataset content read.
        assert Path(candidate_root) == root
        manifest_path.write_bytes(good_manifest)
        arrays_path.write_bytes(good_arrays)
        try:
            return inspect_published_market_dataset_artifact(root)
        finally:
            arrays_path.write_bytes(other_arrays)
            manifest_path.write_bytes(other_manifest)

    monkeypatch.setattr(
        evidence_module,
        "inspect_published_market_dataset_artifact",
        earlier_inspection,
        raising=False,
    )
    with pytest.raises(
        ArtifactIntegrityError, match="Study dataset artifact digest mismatch"
    ):
        evidence_module._verify_plan_inputs(
            dataset_root=root,
            plan=plan,
            research_context_digest="a" * 64,
        )


def test_publication_pins_the_loaded_run_used_for_seed_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset()
    dataset_root = tmp_path / "dataset"
    artifact = publish_market_dataset_artifact(dataset_root, dataset)
    plan = _plan(dataset, artifact)
    store = StudyStore(tmp_path / "study")
    context = content_digest({"kind": "baseline", "study": plan.digest})
    monkeypatch.setattr(evidence_module, "execute_candidate_run", _fake_execute())
    from trade_rl.evaluation.runs import artifact as run_artifact

    original_load = run_artifact._load_with_evidence
    snapshots = []

    def record(root):
        loaded, raw = original_load(root)
        snapshots.append(loaded)
        return loaded, raw

    monkeypatch.setattr(run_artifact, "_load_with_evidence", record)
    published = evidence_module.execute_evidence_set(
        store=store,
        target=Path("evidence/baseline"),
        dataset_root=dataset_root,
        plan=plan,
        config=_config(),
        research_context_digest=context,
    )
    assert len(snapshots) == len(plan.ppo_seeds)
    assert published.run_digests == tuple(
        (seed, loaded.artifact_digest)
        for seed, loaded in zip(plan.ppo_seeds, snapshots, strict=True)
    )
    # Independently read the actual published files and check the existing
    # artifact identity, rather than merely trusting the builder's digest.
    root = store.root / "evidence" / "baseline"
    for seed, digest in published.run_digests:
        assert (
            inspect_candidate_run_artifact(
                root / "runs" / f"seed-{seed}"
            ).artifact_digest
            == digest
        )
    assert (
        evidence_module.load_evidence_set(root, expected_plan=plan).evidence
        == published
    )
