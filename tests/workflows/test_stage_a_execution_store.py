from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from tests.evaluation.replay_support import execution_episode
from tests.stage_a_helpers import stage_a_test_manifest, stage_a_test_manifest_for_plan
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    StageAZeroShotEvaluationPlan,
    build_stage_a_zero_shot_evaluation_plan,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.execution_promotion import execution_evidence_from_cost
from trade_rl.simulation.execution_replay import (
    build_execution_event_artifact,
    write_execution_event_artifact,
)
from trade_rl.workflows.stage_a_execution_store import StageAExecutionPromotionStore
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
    cost = ExecutionCostConfig(path_mode="conservative")
    candidate = StageACandidate.create(
        candidate_id="candidate-a",
        candidate_config_digest=_digest("candidate-a:config"),
        final_training_completion_digest=_digest("candidate-a:complete"),
        policy_identity=_digest("candidate-a:policy"),
        checkpoint_digests=(
            (0, _digest("candidate-a:checkpoint:0")),
            (1, _digest("candidate-a:checkpoint:1")),
        ),
    )
    return build_stage_a_zero_shot_evaluation_plan(
        symbol_disjoint_manifest_digest=manifest.symbol_disjoint_manifest_digest,
        symbol_disjoint_triplet_manifest_digest=manifest.symbol_disjoint_triplet_manifest_digest,
        evaluation_dataset_manifest_digest=manifest.digest,
        feature_identity=manifest.feature_identity,
        execution_identity=cost.execution_policy_digest,
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


def _request(plan: StageAZeroShotEvaluationPlan) -> StageAEvaluationCellRequest:
    manifest = stage_a_test_manifest_for_plan(plan)
    return StageAEvaluationCellRequest(
        plan_digest=plan.digest,
        split="validation",
        triplet_id=plan.validation_triplet_ids[0],
        fold=0,
        seed=0,
        candidate_id="candidate-a",
        checkpoint_digest=plan.candidate("candidate-a").checkpoint_digest(0),
        evaluation_dataset_manifest_digest=manifest.digest,
        dataset_id=manifest.dataset_id_for("validation", plan.validation_triplet_ids[0]),
        evaluation_range=manifest.range_for("validation", 0),
        feature_identity=plan.feature_identity,
        execution_identity=plan.execution_identity,
        evaluation_identity=plan.evaluation_identity,
    )


def _promotion_paths(
    root: Path,
    request: StageAEvaluationCellRequest,
    *,
    candidate_config_digest: str,
    terminal_equity: float,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    actions = ((0.4,),)
    observations = (_digest("observation-0"), _digest("observation-1"))
    equity = (1_000.0, terminal_equity)
    events, terminal_book, terminal_order_book = execution_episode(
        dataset_id=request.dataset_id,
        execution_policy_digest=request.execution_identity,
        cash=terminal_equity - 100.0,
    )
    event_artifact = build_execution_event_artifact(
        candidate_config_digest=candidate_config_digest,
        evaluation_run_digest=request.digest,
        fold=request.fold,
        seed=request.seed,
        dataset_id=request.dataset_id,
        execution_policy_digest=request.execution_identity,
        actions=actions,
        observation_digests=observations,
        equity_curve=equity,
        order_events=events,
        terminal_book=terminal_book,
        terminal_order_book=terminal_order_book,
    )
    event_path = write_execution_event_artifact(
        root / "order-events.json", event_artifact
    )
    evidence = execution_evidence_from_cost(
        dataset_id=request.dataset_id,
        cost=ExecutionCostConfig(path_mode="conservative"),
        sensitivity_path_modes=("conservative",),
        order_event_artifact_path=event_path,
    )
    evidence_path = root / "execution-evidence.json"
    evidence_path.write_bytes(canonical_json_bytes(evidence.to_mapping()) + b"\n")
    return event_path, evidence_path


def _publish(
    store: StageAExecutionPromotionStore,
    *,
    plan: StageAZeroShotEvaluationPlan,
    request: StageAEvaluationCellRequest,
    source_root: Path,
    final_equity: float = 1_100.0,
):
    candidate_config_digest = plan.candidate("candidate-a").candidate_config_digest
    event_path, evidence_path = _promotion_paths(
        source_root,
        request,
        candidate_config_digest=candidate_config_digest,
        terminal_equity=final_equity,
    )
    return store.publish(
        request=request,
        candidate_config_digest=candidate_config_digest,
        actions=((0.4,),),
        observation_digests=(_digest("observation-0"), _digest("observation-1")),
        equity_curve=(1_000.0, final_equity),
        event_artifact_path=event_path,
        execution_evidence_path=evidence_path,
    )


def test_publish_and_load_round_trip_by_request_digest(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan)
    store = StageAExecutionPromotionStore(tmp_path / "store")

    published = _publish(
        store,
        plan=plan,
        request=request,
        source_root=tmp_path / "source",
    )
    loaded = store.load(request.digest)

    assert loaded.artifact == published.artifact
    assert loaded.artifact.cell_identity.request_digest == request.digest
    assert loaded.event_path.name.startswith(loaded.artifact.event_artifact_digest)
    assert loaded.evidence_path.name.startswith(
        loaded.artifact.execution_evidence_digest
    )


def test_request_index_cannot_be_rebound(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan)
    store = StageAExecutionPromotionStore(tmp_path / "store")
    _publish(
        store,
        plan=plan,
        request=request,
        source_root=tmp_path / "source-a",
    )

    with pytest.raises(ValueError, match="already bound"):
        _publish(
            store,
            plan=plan,
            request=request,
            source_root=tmp_path / "source-b",
            final_equity=1_200.0,
        )


def test_load_rejects_tampered_event_bytes(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan)
    store = StageAExecutionPromotionStore(tmp_path / "store")
    published = _publish(
        store,
        plan=plan,
        request=request,
        source_root=tmp_path / "source",
    )
    published.event_path.write_bytes(b"{}\n")

    with pytest.raises(ValueError, match="event artifact digest mismatch"):
        store.load(request.digest)


@pytest.mark.skipif(os.name == "nt", reason="symlink semantics differ on Windows")
def test_load_rejects_symlink_request_index(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan)
    store = StageAExecutionPromotionStore(tmp_path / "store")
    published = _publish(
        store,
        plan=plan,
        request=request,
        source_root=tmp_path / "source",
    )
    index_path = published.index_path
    original = tmp_path / "original-index.json"
    index_path.replace(original)
    index_path.symlink_to(original)

    with pytest.raises(ValueError, match="must not be a symlink"):
        store.load(request.digest)


def test_identical_retry_is_idempotent(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan)
    store = StageAExecutionPromotionStore(tmp_path / "store")
    first = _publish(
        store,
        plan=plan,
        request=request,
        source_root=tmp_path / "source-a",
    )
    second = _publish(
        store,
        plan=plan,
        request=request,
        source_root=tmp_path / "source-b",
    )

    assert second == first


def test_load_rejects_tampered_evidence_bytes(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan)
    store = StageAExecutionPromotionStore(tmp_path / "store")
    published = _publish(
        store,
        plan=plan,
        request=request,
        source_root=tmp_path / "source",
    )
    published.evidence_path.write_bytes(b"{}\n")

    with pytest.raises(ValueError, match="evidence artifact digest mismatch"):
        store.load(request.digest)


def test_load_rejects_tampered_cell_bytes(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan)
    store = StageAExecutionPromotionStore(tmp_path / "store")
    published = _publish(
        store,
        plan=plan,
        request=request,
        source_root=tmp_path / "source",
    )
    published.artifact_path.write_bytes(b"{}\n")

    with pytest.raises(ValueError, match="field closure mismatch"):
        store.load(request.digest)


def test_load_rejects_request_index_path_traversal(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan)
    store = StageAExecutionPromotionStore(tmp_path / "store")
    published = _publish(
        store,
        plan=plan,
        request=request,
        source_root=tmp_path / "source",
    )
    raw = json.loads(published.index_path.read_bytes())
    raw["artifact_path"] = "../../outside.json"
    payload = {key: value for key, value in raw.items() if key != "digest"}
    raw["digest"] = content_digest(payload)
    published.index_path.write_bytes(canonical_json_bytes(raw) + b"\n")

    with pytest.raises(ValueError, match="canonical relative path"):
        store.load(request.digest)


def test_load_rejects_noncanonical_request_index(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan)
    store = StageAExecutionPromotionStore(tmp_path / "store")
    published = _publish(
        store,
        plan=plan,
        request=request,
        source_root=tmp_path / "source",
    )
    canonical = published.index_path.read_bytes()
    published.index_path.write_bytes(canonical.rstrip() + b"  \n")

    with pytest.raises(ValueError, match="canonical encoding"):
        store.load(request.digest)


@pytest.mark.skipif(os.name == "nt", reason="symlink semantics differ on Windows")
def test_load_rejects_symlink_event_artifact(tmp_path: Path) -> None:
    plan = _plan()
    request = _request(plan)
    store = StageAExecutionPromotionStore(tmp_path / "store")
    published = _publish(
        store,
        plan=plan,
        request=request,
        source_root=tmp_path / "source",
    )
    original = tmp_path / "original-event.json"
    published.event_path.replace(original)
    published.event_path.symlink_to(original)

    with pytest.raises(ValueError, match="must not be a symlink"):
        store.load(request.digest)
