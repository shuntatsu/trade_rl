from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.selection import PolicyMode
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
from trade_rl.serving.bundle import (
    ServingBundle,
    ServingBundleManifest,
    write_serving_bundle_manifest,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.stage_a_policy_source import (
    CanonicalServingBundleStageAPolicyLoader,
    StageAPolicyRuntimeHandle,
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


def _plan() -> StageAZeroShotEvaluationPlan:
    candidate = StageACandidate.create(
        candidate_id="candidate-a",
        candidate_config_digest=_candidate_config_digest(),
        final_training_completion_digest=_digest("candidate-a:complete"),
        policy_identity=_digest("candidate-a:policy"),
        checkpoint_digests=((0, _checkpoint_digest(0)),),
    )
    return build_stage_a_zero_shot_evaluation_plan(
        symbol_disjoint_manifest_digest=_digest("symbol-manifest"),
        symbol_disjoint_triplet_manifest_digest=_digest("triplet-manifest"),
        dataset_identity=_digest("dataset"),
        feature_identity=_digest("features"),
        execution_identity=_execution_identity(),
        evaluation_identity=_digest("evaluation"),
        candidates=(candidate,),
        seeds=(0,),
        folds=(0,),
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


def _request(plan: StageAZeroShotEvaluationPlan) -> StageAEvaluationCellRequest:
    candidate = plan.candidate("candidate-a")
    return StageAEvaluationCellRequest(
        plan_digest=plan.digest,
        split="validation",
        triplet_id=plan.validation_triplet_ids[0],
        fold=0,
        seed=0,
        candidate_id=candidate.candidate_id,
        checkpoint_digest=candidate.checkpoint_digest(0),
        dataset_identity=plan.dataset_identity,
        feature_identity=plan.feature_identity,
        execution_identity=plan.execution_identity,
        evaluation_identity=plan.evaluation_identity,
    )


def _write_checkpoint(
    root: Path,
    *,
    plan: StageAZeroShotEvaluationPlan,
) -> tuple[Path, CheckpointManifest]:
    candidate = plan.candidate("candidate-a")
    destination = root / "checkpoints" / candidate.candidate_id / "seed-0"
    destination.mkdir(parents=True)
    policy_path = destination / CHECKPOINT_POLICY_NAME
    policy_path.write_bytes(b"policy-0")
    payload = _checkpoint_payload(0)
    manifest = CheckpointManifest(
        digest=content_digest(payload),
        algorithm="ppo",
        seed=0,
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


def _write_bundle(
    root: Path,
    *,
    policy_digest: str,
    environment_digest: str,
) -> tuple[Path, ServingBundleManifest]:
    bundle_root = root / "bundles" / _digest(
        f"{policy_digest}:{environment_digest}:bundle"
    )
    bundle_root.mkdir(parents=True)
    (bundle_root / "policy.bin").write_bytes(b"flat-policy-artifact")
    manifest = ServingBundleManifest.build(
        root=bundle_root,
        dataset_id=_digest("serving-dataset"),
        action_schema="target_weight_v1",
        observation_schema="flat_observation_v1",
        observation_size=4,
        environment_digest=environment_digest,
        initial_capital=10_000.0,
        policy_mode=PolicyMode.RESIDUAL_POLICY,
        policy_digest=policy_digest,
        signal_digest=_digest("signal"),
        selection_digest=_digest("selection"),
        artifact_paths=("policy.bin",),
        created_at=datetime(2026, 7, 31, tzinfo=timezone.utc),
        action_size=2,
        action_names=("asset-a", "asset-b"),
        action_spec_digest=_digest("action-spec"),
        training_run_digest=_digest("training-run"),
        selection_proposal_digest=_digest("selection-proposal"),
        selection_authorization_digest=_digest("selection-authorization"),
        walk_forward_run_digest=_digest("walk-forward"),
        gate_evidence_digest=_digest("gate-evidence"),
        confirmation_evidence_digest=_digest("confirmation-evidence"),
    )
    write_serving_bundle_manifest(bundle_root, manifest)
    return bundle_root, manifest


class _FlatFallback:
    def __init__(self) -> None:
        self.policy = object()
        self.loaded_bundle: ServingBundle | None = None

    def load(self, bundle: ServingBundle) -> object:
        self.loaded_bundle = bundle
        return self.policy


def _publish_with_bundle(
    tmp_path: Path,
) -> tuple[
    StageAZeroShotEvaluationPlan,
    StageAEvaluationCellRequest,
    Path,
    CheckpointManifest,
    Path,
    ServingBundleManifest,
    StageAPolicySourceStore,
]:
    plan = _plan()
    request = _request(plan)
    root = tmp_path / "artifacts"
    checkpoint_path, checkpoint = _write_checkpoint(root, plan=plan)
    bundle_root, bundle = _write_bundle(
        root,
        policy_digest=checkpoint.policy_digest,
        environment_digest=checkpoint.environment_digest,
    )
    store = StageAPolicySourceStore(root)
    store.publish(
        plan=plan,
        request=request,
        checkpoint_manifest_path=checkpoint_path,
        serving_bundle_path=bundle_root,
    )
    return plan, request, root, checkpoint, bundle_root, bundle, store


def test_publish_and_load_checkpoint_plus_serving_bundle(tmp_path: Path) -> None:
    plan, request, root, checkpoint, bundle_root, bundle, store = _publish_with_bundle(
        tmp_path
    )

    binding = store.load(request.digest)
    validated = binding.validate(root=root, plan=plan, request=request)

    assert validated.digest == checkpoint.digest
    assert binding.serving_bundle_digest == bundle.bundle_digest
    assert binding.serving_bundle_path == bundle_root.relative_to(root).as_posix()


def test_runtime_loader_returns_policy_with_complete_source_identity(
    tmp_path: Path,
) -> None:
    plan, request, root, checkpoint, _, bundle, store = _publish_with_bundle(tmp_path)
    binding = store.load(request.digest)
    fallback = _FlatFallback()
    loader = CanonicalServingBundleStageAPolicyLoader(root, fallback=fallback)

    handle = loader.load(plan=plan, request=request, binding=binding)

    assert isinstance(handle, StageAPolicyRuntimeHandle)
    assert handle.policy is fallback.policy
    assert fallback.loaded_bundle is not None
    assert handle.plan_digest == plan.digest
    assert handle.request_digest == request.digest
    assert handle.candidate_id == "candidate-a"
    assert handle.seed == 0
    assert handle.checkpoint_digest == checkpoint.digest
    assert handle.candidate_config_digest == checkpoint.training_config_digest
    assert handle.checkpoint_policy_digest == checkpoint.policy_digest
    assert handle.serving_bundle_digest == bundle.bundle_digest
    assert handle.architecture_digest is None


def test_runtime_loader_rejects_flat_bundle_without_explicit_fallback(
    tmp_path: Path,
) -> None:
    plan, request, root, _, _, _, store = _publish_with_bundle(tmp_path)
    binding = store.load(request.digest)

    with pytest.raises(RuntimeError, match="explicit policy loader"):
        CanonicalServingBundleStageAPolicyLoader(root).load(
            plan=plan,
            request=request,
            binding=binding,
        )


def test_publish_rejects_serving_policy_substitution(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan)
    root = tmp_path / "artifacts"
    checkpoint_path, checkpoint = _write_checkpoint(root, plan=plan)
    bundle_root, _ = _write_bundle(
        root,
        policy_digest=_digest("substituted-policy"),
        environment_digest=checkpoint.environment_digest,
    )

    with pytest.raises(ValueError, match="serving policy digest mismatch"):
        StageAPolicySourceStore(root).publish(
            plan=plan,
            request=request,
            checkpoint_manifest_path=checkpoint_path,
            serving_bundle_path=bundle_root,
        )


def test_publish_rejects_serving_environment_substitution(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan)
    root = tmp_path / "artifacts"
    checkpoint_path, checkpoint = _write_checkpoint(root, plan=plan)
    bundle_root, _ = _write_bundle(
        root,
        policy_digest=checkpoint.policy_digest,
        environment_digest=_digest("substituted-environment"),
    )

    with pytest.raises(ValueError, match="serving environment digest mismatch"):
        StageAPolicySourceStore(root).publish(
            plan=plan,
            request=request,
            checkpoint_manifest_path=checkpoint_path,
            serving_bundle_path=bundle_root,
        )


def test_validate_rejects_serving_bundle_digest_substitution(tmp_path: Path) -> None:
    plan, request, root, _, _, _, store = _publish_with_bundle(tmp_path)
    binding = store.load(request.digest)
    substituted = replace(
        binding,
        serving_bundle_digest=_digest("substituted-bundle"),
        digest="",
    )

    with pytest.raises(ValueError, match="serving bundle digest mismatch"):
        substituted.validate(root=root, plan=plan, request=request)


def test_load_rejects_serving_artifact_tampering(tmp_path: Path) -> None:
    _, request, _, _, bundle_root, _, store = _publish_with_bundle(tmp_path)
    (bundle_root / "policy.bin").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="artifact size mismatch|artifact digest mismatch"):
        store.load(request.digest)


def test_load_rejects_undeclared_serving_file(tmp_path: Path) -> None:
    _, request, _, _, bundle_root, _, store = _publish_with_bundle(tmp_path)
    (bundle_root / "undeclared.bin").write_bytes(b"undeclared")

    with pytest.raises(ValueError, match="undeclared files"):
        store.load(request.digest)


def test_checkpoint_only_binding_cannot_be_upgraded_in_place(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan)
    root = tmp_path / "artifacts"
    checkpoint_path, checkpoint = _write_checkpoint(root, plan=plan)
    bundle_root, _ = _write_bundle(
        root,
        policy_digest=checkpoint.policy_digest,
        environment_digest=checkpoint.environment_digest,
    )
    store = StageAPolicySourceStore(root)
    store.publish(
        plan=plan,
        request=request,
        checkpoint_manifest_path=checkpoint_path,
    )

    with pytest.raises(ValueError, match="already bound"):
        store.publish(
            plan=plan,
            request=request,
            checkpoint_manifest_path=checkpoint_path,
            serving_bundle_path=bundle_root,
        )


@pytest.mark.skipif(os.name == "nt", reason="symlink semantics require POSIX")
def test_publish_rejects_symlinked_serving_bundle(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan)
    root = tmp_path / "artifacts"
    checkpoint_path, checkpoint = _write_checkpoint(root, plan=plan)
    real_bundle, _ = _write_bundle(
        root,
        policy_digest=checkpoint.policy_digest,
        environment_digest=checkpoint.environment_digest,
    )
    linked_bundle = root / "linked-bundle"
    linked_bundle.symlink_to(real_bundle, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        StageAPolicySourceStore(root).publish(
            plan=plan,
            request=request,
            checkpoint_manifest_path=checkpoint_path,
            serving_bundle_path=linked_bundle,
        )


def test_load_rejects_serving_manifest_identity_tampering(tmp_path: Path) -> None:
    _, request, _, _, bundle_root, _, store = _publish_with_bundle(tmp_path)
    manifest_path = bundle_root / "bundle.json"
    raw = json.loads(manifest_path.read_bytes())
    assert isinstance(raw, dict)
    raw["policy_digest"] = _digest("tampered-policy")
    manifest_path.write_bytes(canonical_json_bytes(raw))

    with pytest.raises(ValueError, match="bundle digest does not match"):
        store.load(request.digest)
