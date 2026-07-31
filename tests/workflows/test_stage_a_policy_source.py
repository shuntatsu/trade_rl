from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from tests.stage_a_helpers import stage_a_test_manifest, stage_a_test_manifest_for_plan
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    StageAZeroShotEvaluationPlan,
    build_stage_a_zero_shot_evaluation_plan,
)
from trade_rl.rl.checkpointing import (
    CHECKPOINT_MANIFEST_NAME,
    CHECKPOINT_POLICY_NAME,
    CheckpointManifest,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.stage_a_policy_source import (
    StageAPolicySourceBinding,
    StageAPolicySourceStore,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidate_config_digest() -> str:
    return _digest("candidate-a:config")


def _execution_identity() -> str:
    return ExecutionCostConfig(path_mode="conservative").execution_policy_digest


def _checkpoint_payload(seed: int) -> dict[str, object]:
    policy_digest = hashlib.sha256(f"policy-{seed}".encode("utf-8")).hexdigest()
    return {
        "algorithm": "ppo",
        "environment_digest": _execution_identity(),
        "observed_timestep": 128,
        "policy_digest": policy_digest,
        "policy_file": CHECKPOINT_POLICY_NAME,
        "requested_timestep": 128,
        "schema_version": "policy_checkpoint_v1",
        "seed": seed,
        "training_config_digest": _candidate_config_digest(),
    }


def _checkpoint_digest(seed: int) -> str:
    return content_digest(_checkpoint_payload(seed))


def _manifest():
    return stage_a_test_manifest(
        symbol_disjoint_manifest_digest=_digest("symbol-manifest"),
        symbol_disjoint_triplet_manifest_digest=_digest("triplet-manifest"),
        feature_identity=_digest("features"),
        validation_triplet_ids=(_digest("validation-triplet"),),
        test_triplet_ids=(_digest("test-triplet"),),
        folds=(0, 1),
    )


def _plan() -> StageAZeroShotEvaluationPlan:
    manifest = _manifest()
    candidate = StageACandidate.create(
        candidate_id="candidate-a",
        candidate_config_digest=_candidate_config_digest(),
        final_training_completion_digest=_digest("candidate-a:complete"),
        policy_identity=_digest("candidate-a:policy"),
        checkpoint_digests=(
            (0, _checkpoint_digest(0)),
            (1, _checkpoint_digest(1)),
        ),
    )
    return build_stage_a_zero_shot_evaluation_plan(
        symbol_disjoint_manifest_digest=manifest.symbol_disjoint_manifest_digest,
        symbol_disjoint_triplet_manifest_digest=manifest.symbol_disjoint_triplet_manifest_digest,
        evaluation_dataset_manifest_digest=manifest.digest,
        feature_identity=manifest.feature_identity,
        execution_identity=_execution_identity(),
        evaluation_identity=_digest("evaluation"),
        candidates=(candidate,),
        seeds=(0, 1),
        folds=(0, 1),
        validation_triplet_ids=(_digest("validation-triplet"),),
        test_triplet_ids=(_digest("test-triplet"),),
        bootstrap_confidence_level=0.95,
        bootstrap_resamples=1_000,
        bootstrap_seed=17,
        minimum_validation_lower_bound=0.0,
        minimum_test_lower_bound=0.0,
        minimum_validation_worst_triplet_excess=0.0,
        minimum_test_worst_triplet_excess=0.0,
        minimum_validation_worst_seed_excess=0.0,
        minimum_test_worst_seed_excess=0.0,
        minimum_validation_triplet_pass_fraction=1.0,
        minimum_test_triplet_pass_fraction=1.0,
    )


def _request(
    plan: StageAZeroShotEvaluationPlan,
    *,
    seed: int = 0,
) -> StageAEvaluationCellRequest:
    manifest = stage_a_test_manifest_for_plan(plan)
    candidate = plan.candidate("candidate-a")
    return StageAEvaluationCellRequest(
        plan_digest=plan.digest,
        split="validation",
        triplet_id=plan.validation_triplet_ids[0],
        fold=0,
        seed=seed,
        candidate_id=candidate.candidate_id,
        checkpoint_digest=candidate.checkpoint_digest(seed),
        evaluation_dataset_manifest_digest=manifest.digest,
        dataset_id=manifest.dataset_id_for("validation", plan.validation_triplet_ids[0]),
        evaluation_range=manifest.range_for("validation", 0),
        feature_identity=plan.feature_identity,
        execution_identity=plan.execution_identity,
        evaluation_identity=plan.evaluation_identity,
    )


def _baseline_request(
    plan: StageAZeroShotEvaluationPlan,
) -> StageAEvaluationCellRequest:
    manifest = stage_a_test_manifest_for_plan(plan)
    return StageAEvaluationCellRequest(
        plan_digest=plan.digest,
        split="validation",
        triplet_id=plan.validation_triplet_ids[0],
        fold=0,
        seed=0,
        candidate_id=None,
        checkpoint_digest=None,
        evaluation_dataset_manifest_digest=manifest.digest,
        dataset_id=manifest.dataset_id_for("validation", plan.validation_triplet_ids[0]),
        evaluation_range=manifest.range_for("validation", 0),
        feature_identity=plan.feature_identity,
        execution_identity=plan.execution_identity,
        evaluation_identity=plan.evaluation_identity,
    )


def _write_checkpoint(
    root: Path,
    *,
    plan: StageAZeroShotEvaluationPlan,
    seed: int = 0,
) -> tuple[Path, CheckpointManifest]:
    candidate = plan.candidate("candidate-a")
    destination = root / "checkpoints" / candidate.candidate_id / f"seed-{seed}"
    destination.mkdir(parents=True)
    policy_path = destination / CHECKPOINT_POLICY_NAME
    policy_path.write_bytes(f"policy-{seed}".encode("utf-8"))
    payload = _checkpoint_payload(seed)
    manifest = CheckpointManifest(
        digest=content_digest(payload),
        algorithm="ppo",
        seed=seed,
        requested_timestep=128,
        observed_timestep=128,
        environment_digest=plan.execution_identity,
        training_config_digest=candidate.candidate_config_digest,
        policy_digest=str(payload["policy_digest"]),
        policy_path=policy_path,
    )
    manifest_path = destination / CHECKPOINT_MANIFEST_NAME
    manifest_path.write_bytes(
        canonical_json_bytes(
            {
                "algorithm": manifest.algorithm,
                "digest": manifest.digest,
                "environment_digest": manifest.environment_digest,
                "observed_timestep": manifest.observed_timestep,
                "policy_digest": manifest.policy_digest,
                "policy_path": CHECKPOINT_POLICY_NAME,
                "requested_timestep": manifest.requested_timestep,
                "schema_version": manifest.schema_version,
                "seed": manifest.seed,
                "training_config_digest": manifest.training_config_digest,
            }
        )
    )
    return manifest_path, manifest


def _publish(
    tmp_path: Path,
) -> tuple[
    StageAZeroShotEvaluationPlan,
    StageAEvaluationCellRequest,
    Path,
    Path,
    StageAPolicySourceStore,
    StageAPolicySourceBinding,
]:
    plan = _plan()
    request = _request(plan)
    root = tmp_path / "artifacts"
    manifest_path, _ = _write_checkpoint(root, plan=plan)
    store = StageAPolicySourceStore(root)
    binding = store.publish(
        plan=plan,
        manifest=stage_a_test_manifest_for_plan(plan),
        request=request,
        checkpoint_manifest_path=manifest_path,
    )
    return plan, request, root, manifest_path, store, binding


def _binding_path(root: Path, request_digest: str) -> Path:
    index_path = root / "by-request" / f"{request_digest}.json"
    raw = json.loads(index_path.read_bytes())
    assert isinstance(raw, dict)
    relative = raw["binding_path"]
    assert isinstance(relative, str)
    return root / relative


def test_publish_and_load_checkpoint_binding(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan)
    root = tmp_path / "artifacts"
    manifest_path, manifest = _write_checkpoint(root, plan=plan)
    store = StageAPolicySourceStore(root)

    binding = store.publish(
        plan=plan,
        manifest=stage_a_test_manifest_for_plan(plan),
        request=request,
        checkpoint_manifest_path=manifest_path,
    )
    loaded = store.load(request.digest)
    validated = loaded.validate(
        root=root,
        plan=plan,
        manifest=stage_a_test_manifest_for_plan(plan),
        request=request,
    )

    assert loaded == binding
    assert binding.checkpoint_digest == request.checkpoint_digest
    assert binding.seed == request.seed
    assert (
        binding.candidate_config_digest
        == plan.candidate("candidate-a").candidate_config_digest
    )
    assert binding.checkpoint_policy_digest == manifest.policy_digest
    assert validated.digest == manifest.digest


def test_publish_rejects_baseline_request(tmp_path: Path) -> None:
    plan = _plan()
    root = tmp_path / "artifacts"
    manifest_path, _ = _write_checkpoint(root, plan=plan)

    with pytest.raises(ValueError, match="baseline"):
        StageAPolicySourceStore(root).publish(
            plan=plan,
            manifest=stage_a_test_manifest_for_plan(plan),
            request=_baseline_request(plan),
            checkpoint_manifest_path=manifest_path,
        )


def test_publish_rejects_checkpoint_for_different_seed(tmp_path: Path) -> None:
    plan = _plan()
    root = tmp_path / "artifacts"
    manifest_path, _ = _write_checkpoint(root, plan=plan, seed=1)

    with pytest.raises(ValueError, match="seed"):
        StageAPolicySourceStore(root).publish(
            plan=plan,
            manifest=stage_a_test_manifest_for_plan(plan),
            request=_request(plan, seed=0),
            checkpoint_manifest_path=manifest_path,
        )


def test_load_rejects_checkpoint_policy_tampering(tmp_path: Path) -> None:
    _, _, _, manifest_path, store, binding = _publish(tmp_path)
    (manifest_path.parent / CHECKPOINT_POLICY_NAME).write_bytes(b"substituted")

    with pytest.raises(ValueError, match="policy digest mismatch"):
        store.load(binding.request_digest)


def test_publish_accepts_only_identical_retry(tmp_path: Path) -> None:
    plan, request, _, manifest_path, store, first = _publish(tmp_path)

    second = store.publish(
        plan=plan,
        manifest=stage_a_test_manifest_for_plan(plan),
        request=request,
        checkpoint_manifest_path=manifest_path,
    )

    assert second == first


def test_publish_rejects_request_rebinding(tmp_path: Path) -> None:
    plan, request, root, manifest_path, store, _ = _publish(tmp_path)
    index_path = root / "by-request" / f"{request.digest}.json"
    index_path.write_bytes(b"{}\n")

    with pytest.raises(ValueError, match="already bound"):
        store.publish(
            plan=plan,
            manifest=stage_a_test_manifest_for_plan(plan),
            request=request,
            checkpoint_manifest_path=manifest_path,
        )


def test_validate_rejects_every_binding_identity_substitution(tmp_path: Path) -> None:
    plan, request, root, _, _, binding = _publish(tmp_path)
    substitutions = (
        (
            replace(binding, plan_digest=_digest("other-plan"), digest=""),
            "plan mismatch",
        ),
        (
            replace(binding, request_digest=_digest("other-request"), digest=""),
            "request mismatch",
        ),
        (
            replace(binding, candidate_id="candidate-b", digest=""),
            "candidate mismatch",
        ),
        (replace(binding, seed=1, digest=""), "seed mismatch"),
        (
            replace(binding, checkpoint_digest=_digest("other-checkpoint"), digest=""),
            "checkpoint mismatch",
        ),
        (
            replace(
                binding,
                candidate_config_digest=_digest("other-config"),
                digest="",
            ),
            "config mismatch",
        ),
        (
            replace(
                binding,
                checkpoint_policy_digest=_digest("other-policy"),
                digest="",
            ),
            "policy digest mismatch",
        ),
    )

    for substituted, message in substitutions:
        with pytest.raises(ValueError, match=message):
            substituted.validate(
                    root=root,
                    plan=plan,
                    manifest=stage_a_test_manifest_for_plan(plan),
                    request=request,
                )


def test_load_rejects_request_index_digest_tampering(tmp_path: Path) -> None:
    _, request, root, _, store, _ = _publish(tmp_path)
    index_path = root / "by-request" / f"{request.digest}.json"
    raw = json.loads(index_path.read_bytes())
    assert isinstance(raw, dict)
    raw["digest"] = _digest("tampered-index")
    index_path.write_bytes(canonical_json_bytes(raw) + b"\n")

    with pytest.raises(ValueError, match="index digest mismatch"):
        store.load(request.digest)


def test_load_rejects_request_index_field_injection(tmp_path: Path) -> None:
    _, request, root, _, store, _ = _publish(tmp_path)
    index_path = root / "by-request" / f"{request.digest}.json"
    raw = json.loads(index_path.read_bytes())
    assert isinstance(raw, dict)
    raw["undeclared"] = True
    index_path.write_bytes(canonical_json_bytes(raw) + b"\n")

    with pytest.raises(ValueError, match="field closure mismatch"):
        store.load(request.digest)


def test_load_rejects_binding_identity_tampering(tmp_path: Path) -> None:
    _, request, root, _, store, _ = _publish(tmp_path)
    binding_path = _binding_path(root, request.digest)
    raw = json.loads(binding_path.read_bytes())
    assert isinstance(raw, dict)
    raw["candidate_id"] = "candidate-b"
    binding_path.write_bytes(canonical_json_bytes(raw) + b"\n")

    with pytest.raises(ValueError, match="binding digest mismatch"):
        store.load(request.digest)


def test_load_rejects_noncanonical_binding_encoding(tmp_path: Path) -> None:
    _, request, root, _, store, _ = _publish(tmp_path)
    binding_path = _binding_path(root, request.digest)
    binding_path.write_bytes(b" " + binding_path.read_bytes())

    with pytest.raises(ValueError, match="canonical encoding"):
        store.load(request.digest)


def test_load_rejects_checkpoint_manifest_tampering(tmp_path: Path) -> None:
    _, request, _, manifest_path, store, _ = _publish(tmp_path)
    raw = json.loads(manifest_path.read_bytes())
    assert isinstance(raw, dict)
    raw["observed_timestep"] = 129
    manifest_path.write_bytes(canonical_json_bytes(raw))

    with pytest.raises(ValueError, match="manifest digest mismatch"):
        store.load(request.digest)


def test_binding_rejects_noncanonical_checkpoint_paths(tmp_path: Path) -> None:
    _, _, _, _, _, binding = _publish(tmp_path)

    for path in (
        "/tmp/checkpoint.json",
        "../checkpoint.json",
        "checkpoints/./checkpoint.json",
    ):
        with pytest.raises(ValueError, match="canonical relative path"):
            replace(binding, checkpoint_manifest_path=path, digest="")


@pytest.mark.skipif(os.name == "nt", reason="symlink semantics require POSIX")
def test_publish_rejects_symlinked_checkpoint_source(tmp_path: Path) -> None:
    plan = _plan()
    real_root = tmp_path / "real-artifacts"
    real_manifest, _ = _write_checkpoint(real_root, plan=plan)
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "checkpoints").symlink_to(
        real_root / "checkpoints",
        target_is_directory=True,
    )
    manifest_path = root / real_manifest.relative_to(real_root)

    with pytest.raises(ValueError, match="symlink"):
        StageAPolicySourceStore(root).publish(
            plan=plan,
            manifest=stage_a_test_manifest_for_plan(plan),
            request=_request(plan),
            checkpoint_manifest_path=manifest_path,
        )


@pytest.mark.skipif(os.name == "nt", reason="symlink semantics require POSIX")
def test_publish_rejects_symlinked_request_index_directory(tmp_path: Path) -> None:
    plan = _plan()
    root = tmp_path / "artifacts"
    manifest_path, _ = _write_checkpoint(root, plan=plan)
    external_index = tmp_path / "external-index"
    external_index.mkdir()
    (root / "by-request").symlink_to(external_index, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        StageAPolicySourceStore(root).publish(
            plan=plan,
            manifest=stage_a_test_manifest_for_plan(plan),
            request=_request(plan),
            checkpoint_manifest_path=manifest_path,
        )


@pytest.mark.skipif(os.name == "nt", reason="symlink semantics require POSIX")
def test_load_rejects_symlinked_request_index_directory(tmp_path: Path) -> None:
    _, request, root, _, store, _ = _publish(tmp_path)
    index_directory = root / "by-request"
    external_index = tmp_path / "external-index"
    index_directory.rename(external_index)
    index_directory.symlink_to(external_index, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        store.load(request.digest)
