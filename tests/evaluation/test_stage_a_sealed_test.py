from __future__ import annotations

import hashlib

import pytest

from trade_rl.evaluation.stage_a_sealed_test import (
    StageASealedTestCellSpec,
    StageASealedTestLedger,
    build_stage_a_sealed_test_authorization_batch,
)
from trade_rl.evaluation.walk_forward.folds import IndexRange


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _batch():
    return build_stage_a_sealed_test_authorization_batch(
        experiment_plan_digest=_digest("plan"),
        evaluation_dataset_manifest_digest=_digest("manifest"),
        evaluation_identity=_digest("evaluation"),
        selected_configuration="candidate-a",
        selected_policy_digest=_digest("candidate-a:policy"),
        cells=(
            StageASealedTestCellSpec(
                triplet_id=_digest("triplet-b"),
                dataset_id=_digest("dataset-b"),
                fold_index=1,
                test_range=IndexRange(40, 60),
            ),
            StageASealedTestCellSpec(
                triplet_id=_digest("triplet-a"),
                dataset_id=_digest("dataset-a"),
                fold_index=1,
                test_range=IndexRange(40, 60),
            ),
            StageASealedTestCellSpec(
                triplet_id=_digest("triplet-a"),
                dataset_id=_digest("dataset-a"),
                fold_index=0,
                test_range=IndexRange(20, 40),
            ),
        ),
    )


def test_batch_sorts_cells_and_binds_generic_access_records() -> None:
    batch = _batch()
    expected_keys = tuple(
        sorted(
            (
                (_digest("triplet-b"), 1),
                (_digest("triplet-a"), 1),
                (_digest("triplet-a"), 0),
            )
        )
    )

    assert tuple(
        (cell.triplet_id, cell.fold_index) for cell in batch.cells
    ) == expected_keys
    assert batch.cell_count == 3
    assert batch.batch_digest == batch.digest
    assert len(batch.records) == 3
    for cell, record in zip(batch.cells, batch.records, strict=True):
        assert cell.access_record == record
        assert record.experiment_plan_digest == batch.experiment_plan_digest
        assert record.dataset_id == cell.dataset_id
        assert record.fold_index == cell.fold_index
        assert record.test_range == cell.test_range
        assert record.selected_configuration == batch.selected_configuration
        assert record.selected_policy_digest == batch.selected_policy_digest


def test_batch_digest_is_deterministic_across_caller_cell_order() -> None:
    first = _batch()
    second = build_stage_a_sealed_test_authorization_batch(
        experiment_plan_digest=first.experiment_plan_digest,
        evaluation_dataset_manifest_digest=(
            first.evaluation_dataset_manifest_digest
        ),
        evaluation_identity=first.evaluation_identity,
        selected_configuration=first.selected_configuration,
        selected_policy_digest=first.selected_policy_digest,
        cells=tuple(
            StageASealedTestCellSpec(
                triplet_id=cell.triplet_id,
                dataset_id=cell.dataset_id,
                fold_index=cell.fold_index,
                test_range=cell.test_range,
            )
            for cell in reversed(first.cells)
        ),
    )

    assert second == first
    assert second.batch_digest == first.batch_digest


def test_batch_rejects_duplicate_triplet_fold_cell() -> None:
    duplicate = StageASealedTestCellSpec(
        triplet_id=_digest("triplet-a"),
        dataset_id=_digest("dataset-a"),
        fold_index=0,
        test_range=IndexRange(20, 40),
    )

    with pytest.raises(ValueError, match="unique"):
        build_stage_a_sealed_test_authorization_batch(
            experiment_plan_digest=_digest("plan"),
            evaluation_dataset_manifest_digest=_digest("manifest"),
            evaluation_identity=_digest("evaluation"),
            selected_configuration="candidate-a",
            selected_policy_digest=_digest("candidate-a:policy"),
            cells=(duplicate, duplicate),
        )


def test_in_memory_batch_ledger_authorizes_plan_once() -> None:
    ledger = StageASealedTestLedger()
    batch = _batch()

    assert ledger.authorize_once(batch) == batch
    assert ledger.records == (batch,)

    with pytest.raises(ValueError, match="already opened"):
        ledger.authorize_once(batch)
    assert ledger.records == (batch,)


def test_in_memory_batch_ledger_rejects_rebinding_same_plan() -> None:
    ledger = StageASealedTestLedger()
    first = _batch()
    ledger.authorize_once(first)
    rebound = build_stage_a_sealed_test_authorization_batch(
        experiment_plan_digest=first.experiment_plan_digest,
        evaluation_dataset_manifest_digest=(
            first.evaluation_dataset_manifest_digest
        ),
        evaluation_identity=first.evaluation_identity,
        selected_configuration="candidate-b",
        selected_policy_digest=_digest("candidate-b:policy"),
        cells=tuple(
            StageASealedTestCellSpec(
                triplet_id=cell.triplet_id,
                dataset_id=cell.dataset_id,
                fold_index=cell.fold_index,
                test_range=cell.test_range,
            )
            for cell in first.cells
        ),
    )

    with pytest.raises(ValueError, match="already opened"):
        ledger.authorize_once(rebound)
