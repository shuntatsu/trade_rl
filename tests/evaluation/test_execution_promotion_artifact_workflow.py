from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.evaluation.execution_promotion_artifacts import (
    load_execution_promotion_artifacts,
    write_execution_promotion_artifacts,
)
from trade_rl.simulation.execution_promotion import validate_execution_promotion
from tests.evaluation.replay_support import (
    CANDIDATE_CONFIG_DIGEST,
    COST,
    DATASET_ID,
    EVALUATION_RUN_DIGEST,
    FOLD,
    POLICY_DIGEST,
    SEED,
    execution_episode,
)


def _write(root: Path):
    events, book, order_book = execution_episode()
    return write_execution_promotion_artifacts(
        root=root,
        candidate_config_digest=CANDIDATE_CONFIG_DIGEST,
        evaluation_run_digest=EVALUATION_RUN_DIGEST,
        fold=FOLD,
        seed=SEED,
        dataset_id=DATASET_ID,
        cost=COST,
        actions=((0.4,),),
        observation_digests=("1" * 64, "2" * 64),
        equity_curve=(1_000.0, 1_000.0),
        order_events=events,
        terminal_book=book,
        terminal_order_book=order_book,
        sensitivity_path_modes=("conservative",),
    )


def test_workflow_emits_and_reloads_one_verified_promotion_root(tmp_path: Path) -> None:
    artifacts = _write(tmp_path)

    assert artifacts.replay_path.name == (
        f"{artifacts.replay_digest}.execution-replay.json"
    )
    assert artifacts.evidence_path.name == (
        f"{artifacts.evidence_digest}.execution-evidence.json"
    )
    assert artifacts.manifest_path.name == f"{artifacts.replay_digest}.json"

    loaded = load_execution_promotion_artifacts(
        root=tmp_path,
        replay_digest=artifacts.replay_digest,
    )
    assert loaded == artifacts
    validate_execution_promotion(
        loaded.evidence,
        expected_policy_digest=POLICY_DIGEST,
        event_artifact_path=loaded.replay_path,
        expected_candidate_config_digest=CANDIDATE_CONFIG_DIGEST,
        expected_evaluation_run_digest=EVALUATION_RUN_DIGEST,
        expected_fold=FOLD,
        expected_seed=SEED,
    )


def test_workflow_is_idempotent_only_for_the_same_verified_bytes(tmp_path: Path) -> None:
    first = _write(tmp_path)
    second = _write(tmp_path)
    assert second == first

    manifest = first.manifest_path
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        load_execution_promotion_artifacts(
            root=tmp_path,
            replay_digest=first.replay_digest,
        )
