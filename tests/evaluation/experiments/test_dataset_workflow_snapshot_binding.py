"""Pin only the Dataset snapshot actually used by each maintained workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.evaluation.experiments.bootstrap.test_workflow import (
    _config_path,
    _install_fakes,
)
from tests.evaluation.experiments.test_evidence import (
    _config,
    _dataset,
    _fake_execute,
    _plan,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import (
    inspect_published_market_dataset_artifact,
    load_published_market_dataset_artifact,
    publish_market_dataset_artifact,
)
from trade_rl.evaluation.experiments import ControlledFactor, create_study
from trade_rl.evaluation.experiments import workflow as study_workflow
from trade_rl.evaluation.experiments.bootstrap import workflow as bootstrap_workflow
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.runs import candidate, load_candidate_run_artifact


def _replace_after_inspection(root: Path, module, monkeypatch):
    """Start with a different physical archive of the same valid Dataset."""
    first, dataset = load_published_market_dataset_artifact(root)
    manifest = root / "manifest.json"
    arrays_path = root / "arrays.npz"
    original_manifest = manifest.read_bytes()
    original_arrays = arrays_path.read_bytes()
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    with arrays_path.open("wb") as handle:
        np.savez(handle, **arrays)
    alternate_arrays = arrays_path.read_bytes()
    assert alternate_arrays != original_arrays
    payload = json.loads(original_manifest)
    payload["arrays_digest"] = hashlib.sha256(alternate_arrays).hexdigest()
    payload.pop("artifact_digest")
    payload["artifact_digest"] = content_digest(payload)
    alternate_manifest = json.dumps(payload).encode()
    manifest.write_bytes(alternate_manifest)
    alternate, reloaded = load_published_market_dataset_artifact(root)
    assert reloaded.dataset_id == dataset.dataset_id
    assert alternate.artifact_digest != first.artifact_digest

    def restore_for_inspect(path):
        assert Path(path) == root
        manifest.write_bytes(original_manifest)
        arrays_path.write_bytes(original_arrays)
        try:
            return inspect_published_market_dataset_artifact(path)
        finally:
            arrays_path.write_bytes(alternate_arrays)
            manifest.write_bytes(alternate_manifest)

    monkeypatch.setattr(
        module,
        "inspect_published_market_dataset_artifact",
        restore_for_inspect,
        raising=False,
    )
    return first, alternate


def test_candidate_publication_pins_actual_loaded_dataset(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    publish_market_dataset_artifact(root, _dataset())
    first, actual = _replace_after_inspection(root, candidate, monkeypatch)
    monkeypatch.setattr(candidate, "execute_candidate_run", _fake_execute())
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(_config().to_json_payload()), encoding="utf-8")
    output = candidate.run_candidate_artifact(
        dataset_root=root, config_path=config_path, output_root=tmp_path / "run"
    )
    loaded = load_candidate_run_artifact(output.root)
    assert (
        loaded.summary["dataset_artifact"]["artifact_digest"] == actual.artifact_digest
    )
    assert actual.artifact_digest != first.artifact_digest


def test_study_creation_pins_actual_loaded_dataset(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    publish_market_dataset_artifact(root, _dataset())
    first, actual = _replace_after_inspection(root, study_workflow, monkeypatch)
    snapshot = create_study(
        tmp_path / "study",
        dataset_root=root,
        research_question="Bind the actual loaded synthetic fixture.",
        baseline_config=_config(),
        ppo_seeds=(2, 5),
        allowed_factors=(ControlledFactor.PPO_TRAINING_BUDGET,),
        max_experiments=1,
        n_bootstrap=16,
        bootstrap_seed=3,
    )
    assert snapshot.plan.dataset_artifact_digest == actual.artifact_digest
    assert actual.artifact_digest != first.artifact_digest


def test_study_preflight_rejects_mismatched_archive_pin(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    dataset = _dataset()
    published = publish_market_dataset_artifact(root, dataset)
    plan = _plan(dataset, published)
    _replace_after_inspection(root, study_workflow, monkeypatch)
    with pytest.raises(
        ArtifactIntegrityError, match="artifact differs from frozen plan"
    ):
        study_workflow._validate_dataset_root(root, plan)


def test_bootstrap_inspection_does_not_bind_different_archive(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)
    root = tmp_path / "bootstrap"
    bootstrap_workflow.bootstrap_canonical_m2_study(_config_path(tmp_path), root)
    _replace_after_inspection(root / "dataset", bootstrap_workflow, monkeypatch)
    with pytest.raises(ValueError, match="dataset artifact digest differs"):
        bootstrap_workflow.inspect_canonical_m2_bootstrap(root)


def test_bootstrap_reuses_each_verified_pair_and_rechecks_final_root(
    tmp_path, monkeypatch
):
    _install_fakes(monkeypatch)
    observed = []

    def record(path):
        artifact, dataset = load_published_market_dataset_artifact(path)
        observed.append((artifact, dataset))
        return artifact, dataset

    monkeypatch.setattr(
        bootstrap_workflow, "load_published_market_dataset_artifact", record
    )
    root = tmp_path / "bootstrap"
    published = bootstrap_workflow.bootstrap_canonical_m2_study(
        _config_path(tmp_path), root
    )
    # Once after Dataset publication and once for final whole-root inspection.
    # Study creation is a separate workflow that verifies its own input.
    assert len(observed) == 2
    assert all(
        artifact.artifact_digest == published.dataset_artifact_digest
        and dataset.dataset_id == published.dataset_id
        for artifact, dataset in observed
    )
    observed.clear()
    assert bootstrap_workflow.inspect_canonical_m2_bootstrap(root) == published
    assert len(observed) == 1
