from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

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
from trade_rl.workflows.stage_a_policy_source import StageAPolicySourceStore
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


def _plan() -> StageAZeroShotEvaluationPlan:
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
        symbol_disjoint_manifest_digest=_digest("symbol-manifest"),
        symbol_disjoint_triplet_manifest_digest=_digest("triplet-manifest"),
        dataset_identity=_digest("dataset"),
        feature_identity=_digest("features"),
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
    candidate = plan.candidate("candidate-a")
    return StageAEvaluationCellRequest(
        plan_digest=plan.digest,
        split="validation",
        triplet_id=plan.validation_triplet_ids[0],
        fold=0,
        seed=seed,
        candidate_id=candidate.candidate_id,
        checkpoint_digest=candidate.checkpoint_digest(seed),
        dataset_identity=plan.dataset_identity,
        feature_identity=plan.feature_identity,
        execution_identity=plan.execution_identity,
        evaluation_identity=plan.evaluation_identity,
    )


def _baseline_request(plan: StageAZeroShotEvaluationPlan) -> StageAEvaluationCellRequest:
    return StageAEvaluationCellRequest(
        plan_digest=plan.digest,
        split="validation",
        triplet_id=plan.validation_triplet_ids[0],
        fold=0,
        seed=0,
        candidate_id=None,
        checkpoint_digest=None,
        dataset_identity=plan.dataset_identity,
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


def test_publish_and_load_checkpoint_binding(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan)
    root = tmp_path / "artifacts"
    manifest_path, manifest = _write_checkpoint(root, plan=plan)
    store = StageAPolicySourceStore(root)

    binding = store.publish(
        plan=plan,
        request=request,
        checkpoint_manifest_path=manifest_path,
    )
    loaded = store.load(request.digest)
    validated = loaded.validate(root=root, plan=plan, request=request)

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
            request=_request(plan, seed=0),
            checkpoint_manifest_path=manifest_path,
        )


def test_load_rejects_checkpoint_policy_tampering(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan)
    root = tmp_path / "artifacts"
    manifest_path, _ = _write_checkpoint(root, plan=plan)
    store = StageAPolicySourceStore(root)
    binding = store.publish(
        plan=plan,
        request=request,
        checkpoint_manifest_path=manifest_path,
    )
    (manifest_path.parent / CHECKPOINT_POLICY_NAME).write_bytes(b"substituted")

    with pytest.raises(ValueError, match="policy digest mismatch"):
        store.load(binding.request_digest)
