from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from trade_rl.artifacts.run_manifest import (
    TrainingRunManifest,
    write_training_run_manifest,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.execution_promotion import (
    execution_evidence_from_cost,
    write_execution_evidence,
)
from trade_rl.simulation.execution_replay import (
    EXECUTION_EVENT_ARTIFACT_FILE_NAME,
    build_execution_event_artifact,
    write_execution_event_artifact,
)
from tests.evaluation.replay_support import execution_episode

_TARGET_MODULES = {
    "tests.serving.test_package",
    "tests.serving.test_package_critical_branches",
}


def _rebuild_v3_training_run(
    original: Callable[..., TrainingRunManifest],
    *args: Any,
    **kwargs: Any,
) -> TrainingRunManifest:
    manifest = original(*args, **kwargs)
    raw_root = args[0] if args else kwargs.get("root")
    if not isinstance(raw_root, Path):
        raise TypeError("serving training fixture root must be a Path")
    root = raw_root
    execution_cost = ExecutionCostConfig(path_mode="conservative")
    events, terminal_book, terminal_order_book = execution_episode(
        dataset_id=manifest.dataset_id,
        execution_policy_digest=execution_cost.execution_policy_digest,
    )
    event_artifact = build_execution_event_artifact(
        candidate_config_digest=manifest.training_config_digest,
        evaluation_run_digest=manifest.walk_forward_run_digest or "3" * 64,
        fold=0,
        seed=0,
        dataset_id=manifest.dataset_id,
        execution_policy_digest=execution_cost.execution_policy_digest,
        actions=((0.4,),),
        observation_digests=("1" * 64, "2" * 64),
        equity_curve=(1_000.0, 1_000.0),
        order_events=events,
        terminal_book=terminal_book,
        terminal_order_book=terminal_order_book,
    )
    event_path = write_execution_event_artifact(
        root / EXECUTION_EVENT_ARTIFACT_FILE_NAME,
        event_artifact,
    )
    evidence_path = root / "execution-evidence.json"
    evidence_path.unlink()
    evidence = execution_evidence_from_cost(
        dataset_id=manifest.dataset_id,
        cost=execution_cost,
        order_event_artifact_path=event_path,
        sensitivity_path_modes=("conservative",),
    )
    execution_path_mode = kwargs.get("execution_path_mode", "conservative")
    if execution_path_mode != "conservative":
        evidence = replace(evidence, path_mode=execution_path_mode)
    execution_policy_digest = kwargs.get("execution_policy_digest")
    if execution_policy_digest is not None:
        evidence = replace(
            evidence,
            execution_policy_digest=execution_policy_digest,
        )
    write_execution_evidence(evidence_path, evidence)

    run_path = root / "run.json"
    run_path.unlink()
    artifact_paths = [item.path for item in manifest.files]
    artifact_paths.append(EXECUTION_EVENT_ARTIFACT_FILE_NAME)
    rebuilt = TrainingRunManifest.build(
        root=root,
        run_id=manifest.run_id,
        dataset_id=manifest.dataset_id,
        environment_digest=manifest.environment_digest,
        ensemble_digest=manifest.ensemble_digest,
        training_config_digest=manifest.training_config_digest,
        provenance_digest=manifest.provenance_digest,
        artifact_paths=tuple(artifact_paths),
        created_at=manifest.created_at,
        completed_at=manifest.completed_at,
        run_kind=manifest.run_kind,
        selection_proposal_digest=manifest.selection_proposal_digest,
        selection_authorization_digest=manifest.selection_authorization_digest,
        walk_forward_run_digest=manifest.walk_forward_run_digest,
        gate_evidence_digest=manifest.gate_evidence_digest,
    )
    write_training_run_manifest(root, rebuilt)
    return rebuilt


@pytest.fixture(autouse=True)
def _bind_execution_event_fixture(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = request.module
    if module.__name__ not in _TARGET_MODULES or not hasattr(module, "_training_run"):
        return
    original = module._training_run

    def wrapped(*args: Any, **kwargs: Any) -> TrainingRunManifest:
        return _rebuild_v3_training_run(original, *args, **kwargs)

    monkeypatch.setattr(module, "_training_run", wrapped)
