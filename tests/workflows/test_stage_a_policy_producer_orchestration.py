from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
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
from trade_rl.serving.bundle import ServingBundleManifest, write_serving_bundle_manifest
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders import OrderBookState, OrderEvent, OrderStatus
from trade_rl.workflows.stage_a_execution_producer import (
    StageAEvaluationEpisodeResult,
    StageAExecutionArtifactProducer,
)
from trade_rl.workflows.stage_a_execution_store import StageAExecutionPromotionStore
from trade_rl.workflows.stage_a_policy_source import (
    StageAPolicyRuntimeHandle,
    StageAPolicySourceBinding,
    StageAPolicySourceStore,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_COST = ExecutionCostConfig(path_mode="conservative")
_BASELINE_CONFIG_DIGEST = _digest("baseline-config")


def _candidate_config_digest() -> str:
    return _digest("candidate-a:config")


def _checkpoint_payload() -> dict[str, object]:
    return {
        "algorithm": "ppo",
        "environment_digest": _COST.execution_policy_digest,
        "observed_timestep": 128,
        "policy_digest": hashlib.sha256(b"policy-0").hexdigest(),
        "policy_file": CHECKPOINT_POLICY_NAME,
        "requested_timestep": 128,
        "schema_version": "policy_checkpoint_v1",
        "seed": 0,
        "training_config_digest": _candidate_config_digest(),
    }


def _plan() -> StageAZeroShotEvaluationPlan:
    candidate = StageACandidate.create(
        candidate_id="candidate-a",
        candidate_config_digest=_candidate_config_digest(),
        final_training_completion_digest=_digest("candidate-a:complete"),
        policy_identity=_digest("candidate-a:policy"),
        checkpoint_digests=(
            (0, content_digest(_checkpoint_payload())),
            (1, _digest("candidate-a:checkpoint:1")),
        ),
    )
    return build_stage_a_zero_shot_evaluation_plan(
        symbol_disjoint_manifest_digest=_digest("symbol-manifest"),
        symbol_disjoint_triplet_manifest_digest=_digest("triplet-manifest"),
        dataset_identity=_digest("dataset"),
        feature_identity=_digest("features"),
        execution_identity=_COST.execution_policy_digest,
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
    plan: StageAZeroShotEvaluationPlan, *, policy: bool
) -> StageAEvaluationCellRequest:
    candidate_id = "candidate-a" if policy else None
    checkpoint_digest = (
        plan.candidate("candidate-a").checkpoint_digest(0) if policy else None
    )
    return StageAEvaluationCellRequest(
        plan_digest=plan.digest,
        split="validation",
        triplet_id=plan.validation_triplet_ids[0],
        fold=0,
        seed=0,
        candidate_id=candidate_id,
        checkpoint_digest=checkpoint_digest,
        dataset_identity=plan.dataset_identity,
        feature_identity=plan.feature_identity,
        execution_identity=plan.execution_identity,
        evaluation_identity=plan.evaluation_identity,
    )


def _write_policy_source(
    root: Path,
    *,
    plan: StageAZeroShotEvaluationPlan,
    request: StageAEvaluationCellRequest,
) -> tuple[StageAPolicySourceStore, StageAPolicySourceBinding]:
    candidate = plan.candidate("candidate-a")
    checkpoint_root = root / "checkpoints" / candidate.candidate_id / "seed-0"
    checkpoint_root.mkdir(parents=True)
    policy_path = checkpoint_root / CHECKPOINT_POLICY_NAME
    policy_path.write_bytes(b"policy-0")
    payload = _checkpoint_payload()
    checkpoint = CheckpointManifest(
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
    checkpoint_path = checkpoint_root / CHECKPOINT_MANIFEST_NAME
    checkpoint_path.write_bytes(
        canonical_json_bytes(
            {
                "algorithm": checkpoint.algorithm,
                "digest": checkpoint.digest,
                "environment_digest": checkpoint.environment_digest,
                "observed_timestep": checkpoint.observed_timestep,
                "policy_digest": checkpoint.policy_digest,
                "policy_path": CHECKPOINT_POLICY_NAME,
                "requested_timestep": checkpoint.requested_timestep,
                "schema_version": checkpoint.schema_version,
                "seed": checkpoint.seed,
                "training_config_digest": checkpoint.training_config_digest,
            }
        )
    )

    bundle_root = root / "bundles" / _digest("candidate-a:bundle")
    bundle_root.mkdir(parents=True)
    (bundle_root / "policy.bin").write_bytes(b"flat-policy")
    bundle = ServingBundleManifest.build(
        root=bundle_root,
        dataset_id=_digest("serving-dataset"),
        action_schema="target_weight_v1",
        observation_schema="flat_observation_v1",
        observation_size=1,
        environment_digest=checkpoint.environment_digest,
        initial_capital=1_000.0,
        policy_mode=PolicyMode.RESIDUAL_POLICY,
        policy_digest=checkpoint.policy_digest,
        signal_digest=_digest("signal"),
        selection_digest=_digest("selection"),
        artifact_paths=("policy.bin",),
        created_at=datetime(2026, 7, 31, tzinfo=timezone.utc),
        action_size=1,
        action_names=("asset-a",),
        action_spec_digest=_digest("action-spec"),
        training_run_digest=_digest("training-run"),
        selection_proposal_digest=_digest("selection-proposal"),
        selection_authorization_digest=_digest("selection-authorization"),
        walk_forward_run_digest=_digest("walk-forward"),
        gate_evidence_digest=_digest("gate-evidence"),
        confirmation_evidence_digest=_digest("confirmation-evidence"),
    )
    write_serving_bundle_manifest(bundle_root, bundle)

    store = StageAPolicySourceStore(root)
    binding = store.publish(
        plan=plan,
        request=request,
        checkpoint_manifest_path=checkpoint_path,
        serving_bundle_path=bundle_root,
    )
    return store, binding


def _event(request: StageAEvaluationCellRequest, *, price: float = 100.0) -> OrderEvent:
    return OrderEvent(
        schema_version="order_event_v1",
        sequence=0,
        order_id="a" * 64,
        replaced_order_id=None,
        dataset_id=request.dataset_identity,
        execution_policy_digest=request.execution_identity,
        symbol_index=0,
        event_type="filled",
        processing_index=1,
        timestamp_ns=1,
        previous_status=OrderStatus.ELIGIBLE,
        new_status=OrderStatus.FILLED,
        requested_quantity=1.0,
        remaining_quantity=0.0,
        filled_quantity=1.0,
        execution_price=price,
        filled_notional=price,
        capacity_before=10.0,
        capacity_after=9.0,
        participation_rate=0.1,
        trigger_segment=None,
        available_volume_fraction=1.0,
        reason=None,
        path_mode="conservative",
        path_points=(price, price + 1.0, price - 1.0, price + 0.5),
    )


def _episode_result(
    request: StageAEvaluationCellRequest,
    *,
    policy_source_digest: str | None,
    candidate_config_digest: str,
    action: float = 0.4,
) -> StageAEvaluationEpisodeResult:
    return StageAEvaluationEpisodeResult(
        request_digest=request.digest,
        policy_source_digest=policy_source_digest,
        candidate_config_digest=candidate_config_digest,
        actions=((action,),),
        observation_digests=(_digest("observation-0"), _digest("observation-1")),
        equity_curve=(1_000.0, 1_100.0),
        order_events=(_event(request),),
        terminal_book=BookState(
            quantities=np.array((1.0,), dtype=np.float64),
            cash=1_000.0,
            mark_prices=np.array((100.0,), dtype=np.float64),
            peak_value=1_100.0,
        ),
        terminal_order_book=OrderBookState.empty(),
    )


class _RuntimeLoader:
    def __init__(self, handle: StageAPolicyRuntimeHandle) -> None:
        self.handle = handle
        self.calls = 0

    def load(
        self,
        *,
        plan: StageAZeroShotEvaluationPlan,
        request: StageAEvaluationCellRequest,
        binding: StageAPolicySourceBinding,
    ) -> StageAPolicyRuntimeHandle:
        del plan, request, binding
        self.calls += 1
        return self.handle


class _Executor:
    def __init__(
        self,
        transform: Callable[
            [StageAEvaluationEpisodeResult], StageAEvaluationEpisodeResult
        ] = lambda result: result,
    ) -> None:
        self.transform = transform
        self.calls: list[tuple[object | None, str | None, str]] = []
        self.action = 0.4

    def execute(
        self,
        request: StageAEvaluationCellRequest,
        *,
        policy: object | None,
        policy_source_digest: str | None,
        candidate_config_digest: str,
    ) -> StageAEvaluationEpisodeResult:
        self.calls.append((policy, policy_source_digest, candidate_config_digest))
        return self.transform(
            _episode_result(
                request,
                policy_source_digest=policy_source_digest,
                candidate_config_digest=candidate_config_digest,
                action=self.action,
            )
        )


class _CostResolver:
    def __init__(self, cost: ExecutionCostConfig = _COST) -> None:
        self.cost = cost

    def resolve(self, request: StageAEvaluationCellRequest) -> ExecutionCostConfig:
        del request
        return self.cost


class _BombSourceStore:
    root = Path(".")

    def load(self, request_digest: str) -> StageAPolicySourceBinding:
        del request_digest
        raise AssertionError("baseline must not load a policy source")


class _BombRuntimeLoader:
    def load(
        self,
        *,
        plan: StageAZeroShotEvaluationPlan,
        request: StageAEvaluationCellRequest,
        binding: StageAPolicySourceBinding,
    ) -> StageAPolicyRuntimeHandle:
        del plan, request, binding
        raise AssertionError("baseline must not load a policy")


def _matching_handle(binding: StageAPolicySourceBinding) -> StageAPolicyRuntimeHandle:
    assert binding.serving_bundle_digest is not None
    return StageAPolicyRuntimeHandle(
        policy=object(),
        binding_digest=binding.digest,
        plan_digest=binding.plan_digest,
        request_digest=binding.request_digest,
        candidate_id=binding.candidate_id,
        seed=binding.seed,
        checkpoint_digest=binding.checkpoint_digest,
        candidate_config_digest=binding.candidate_config_digest,
        checkpoint_policy_digest=binding.checkpoint_policy_digest,
        serving_bundle_digest=binding.serving_bundle_digest,
        architecture_digest=None,
    )


def _policy_producer(
    tmp_path: Path,
    *,
    handle_transform: Callable[
        [StageAPolicyRuntimeHandle], StageAPolicyRuntimeHandle
    ] = lambda handle: handle,
    result_transform: Callable[
        [StageAEvaluationEpisodeResult], StageAEvaluationEpisodeResult
    ] = lambda result: result,
) -> tuple[
    StageAExecutionArtifactProducer,
    StageAEvaluationCellRequest,
    _RuntimeLoader,
    _Executor,
]:
    plan = _plan()
    request = _request(plan, policy=True)
    source_store, binding = _write_policy_source(
        tmp_path / "sources",
        plan=plan,
        request=request,
    )
    runtime = _RuntimeLoader(handle_transform(_matching_handle(binding)))
    executor = _Executor(result_transform)
    producer = StageAExecutionArtifactProducer(
        plan=plan,
        policy_source_store=source_store,
        policy_runtime_loader=runtime,
        episode_executor=executor,
        execution_store=StageAExecutionPromotionStore(tmp_path / "executions"),
        execution_cost_resolver=_CostResolver(),
        baseline_config_digest=_BASELINE_CONFIG_DIGEST,
    )
    return producer, request, runtime, executor


def test_produce_publishes_policy_execution_with_complete_identity(
    tmp_path: Path,
) -> None:
    producer, request, runtime, executor = _policy_producer(tmp_path)

    stored = producer.produce(request)

    assert stored.artifact.cell_identity.request_digest == request.digest
    assert stored.artifact.cell_identity.candidate_id == "candidate-a"
    assert (
        stored.artifact.cell_identity.candidate_config_digest
        == _candidate_config_digest()
    )
    assert runtime.calls == 1
    assert len(executor.calls) == 1
    assert executor.calls[0][0] is not None
    assert executor.calls[0][1] is not None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("binding_digest", _digest("other-binding"), "binding digest mismatch"),
        ("request_digest", _digest("other-request"), "request digest mismatch"),
        ("checkpoint_digest", _digest("other-checkpoint"), "checkpoint mismatch"),
        (
            "checkpoint_policy_digest",
            _digest("other-policy"),
            "policy digest mismatch",
        ),
        (
            "candidate_config_digest",
            _digest("other-config"),
            "config digest mismatch",
        ),
        (
            "serving_bundle_digest",
            _digest("other-bundle"),
            "serving bundle mismatch",
        ),
    ),
)
def test_produce_rejects_runtime_handle_identity_substitution(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    producer, request, _, executor = _policy_producer(
        tmp_path,
        handle_transform=lambda handle: replace(handle, **{field: value}),
    )

    with pytest.raises(ValueError, match=message):
        producer.produce(request)
    assert executor.calls == []


@pytest.mark.parametrize(
    ("transform", "message"),
    (
        (
            lambda result: replace(result, request_digest=_digest("other-request")),
            "request digest mismatch",
        ),
        (
            lambda result: replace(
                result,
                policy_source_digest=_digest("other-source"),
            ),
            "policy source digest mismatch",
        ),
        (
            lambda result: replace(
                result,
                candidate_config_digest=_digest("other-config"),
            ),
            "candidate config digest mismatch",
        ),
    ),
)
def test_produce_rejects_executor_identity_substitution(
    tmp_path: Path,
    transform: Callable[[StageAEvaluationEpisodeResult], StageAEvaluationEpisodeResult],
    message: str,
) -> None:
    producer, request, _, _ = _policy_producer(
        tmp_path,
        result_transform=transform,
    )

    with pytest.raises(ValueError, match=message):
        producer.produce(request)


def test_policy_request_fails_when_source_is_missing(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan, policy=True)
    producer = StageAExecutionArtifactProducer(
        plan=plan,
        policy_source_store=StageAPolicySourceStore(tmp_path / "missing-sources"),
        policy_runtime_loader=_BombRuntimeLoader(),
        episode_executor=_Executor(),
        execution_store=StageAExecutionPromotionStore(tmp_path / "executions"),
        execution_cost_resolver=_CostResolver(),
        baseline_config_digest=_BASELINE_CONFIG_DIGEST,
    )

    with pytest.raises(ValueError, match="request index is missing"):
        producer.produce(request)


def test_baseline_production_bypasses_policy_source_and_runtime(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan, policy=False)
    executor = _Executor()
    producer = StageAExecutionArtifactProducer(
        plan=plan,
        policy_source_store=_BombSourceStore(),
        policy_runtime_loader=_BombRuntimeLoader(),
        episode_executor=executor,
        execution_store=StageAExecutionPromotionStore(tmp_path / "executions"),
        execution_cost_resolver=_CostResolver(),
        baseline_config_digest=_BASELINE_CONFIG_DIGEST,
    )

    stored = producer.produce(request)

    assert stored.artifact.cell_identity.candidate_id is None
    assert stored.artifact.cell_identity.checkpoint_digest is None
    assert (
        stored.artifact.cell_identity.candidate_config_digest == _BASELINE_CONFIG_DIGEST
    )
    assert executor.calls == [(None, None, _BASELINE_CONFIG_DIGEST)]


def test_produce_rejects_execution_cost_identity_substitution(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan, policy=False)
    producer = StageAExecutionArtifactProducer(
        plan=plan,
        policy_source_store=_BombSourceStore(),
        policy_runtime_loader=_BombRuntimeLoader(),
        episode_executor=_Executor(),
        execution_store=StageAExecutionPromotionStore(tmp_path / "executions"),
        execution_cost_resolver=_CostResolver(
            ExecutionCostConfig(path_mode="conservative", maker_fee_rate=0.001)
        ),
        baseline_config_digest=_BASELINE_CONFIG_DIGEST,
    )

    with pytest.raises(ValueError, match="execution cost identity mismatch"):
        producer.produce(request)


def test_request_rebinding_with_different_execution_bytes_is_rejected(
    tmp_path: Path,
) -> None:
    producer, request, _, executor = _policy_producer(tmp_path)
    producer.produce(request)
    executor.action = 0.8

    with pytest.raises(ValueError, match="already bound"):
        producer.produce(request)
