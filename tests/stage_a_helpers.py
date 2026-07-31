from __future__ import annotations

from datetime import UTC, datetime, timedelta

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.walk_forward.folds import IndexRange
from trade_rl.workflows.stage_a_evaluation_dataset_manifest import (
    StageAEvaluationDatasetFold,
    StageAEvaluationDatasetManifest,
    StageAEvaluationDatasetTriplet,
)


def stage_a_test_manifest(
    *,
    symbol_disjoint_manifest_digest: str,
    symbol_disjoint_triplet_manifest_digest: str,
    feature_identity: str,
    validation_triplet_ids: tuple[str, ...],
    test_triplet_ids: tuple[str, ...],
    folds: tuple[int, ...],
    dataset_ids_by_triplet: dict[str, str] | None = None,
) -> StageAEvaluationDatasetManifest:
    """Build a deterministic manifest fixture with disjoint symbols and ranges."""

    triplets: list[StageAEvaluationDatasetTriplet] = []
    for split, ids, prefix in (
        ("validation", validation_triplet_ids, "VAL"),
        ("test", test_triplet_ids, "TEST"),
    ):
        for index, triplet_id in enumerate(ids):
            symbols = tuple(f"{prefix}{index}_{slot}USDT" for slot in ("A", "B", "C"))
            triplets.append(
                StageAEvaluationDatasetTriplet(
                    split=split,
                    triplet_id=triplet_id,
                    symbols=symbols,
                    dataset_id=(
                        dataset_ids_by_triplet[triplet_id]
                        if dataset_ids_by_triplet is not None
                        and triplet_id in dataset_ids_by_triplet
                        else content_digest(
                            {
                                "fixture": "stage_a_dataset",
                                "split": split,
                                "triplet_id": triplet_id,
                                "symbols": symbols,
                            }
                        )
                    ),
                )
            )

    fold_entries = tuple(
        StageAEvaluationDatasetFold(
            fold=fold,
            configuration_selection=IndexRange(20 + ordinal * 100, 40 + ordinal * 100),
            test=IndexRange(60 + ordinal * 100, 80 + ordinal * 100),
        )
        for ordinal, fold in enumerate(sorted(folds))
    )
    required_bars = max(item.test.stop for item in fold_entries) + 20
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=15 * required_bars)
    return StageAEvaluationDatasetManifest(
        symbol_disjoint_manifest_digest=symbol_disjoint_manifest_digest,
        symbol_disjoint_triplet_manifest_digest=symbol_disjoint_triplet_manifest_digest,
        source_closure_digest=content_digest({"fixture": "stage_a_source"}),
        source_metadata_evidence_digest=content_digest(
            {"fixture": "stage_a_metadata"}
        ),
        indicator_cache_id="stage-a-test-cache",
        feature_identity=feature_identity,
        timeline_start_time=start,
        timeline_end_time=end,
        triplets=tuple(triplets),
        folds=fold_entries,
    )


def stage_a_test_manifest_for_plan(plan) -> StageAEvaluationDatasetManifest:
    """Reconstruct the deterministic manifest fixture bound by a test plan."""

    manifest = stage_a_test_manifest(
        symbol_disjoint_manifest_digest=plan.symbol_disjoint_manifest_digest,
        symbol_disjoint_triplet_manifest_digest=(
            plan.symbol_disjoint_triplet_manifest_digest
        ),
        feature_identity=plan.feature_identity,
        validation_triplet_ids=plan.validation_triplet_ids,
        test_triplet_ids=plan.test_triplet_ids,
        folds=plan.folds,
    )
    if manifest.digest != plan.evaluation_dataset_manifest_digest:
        raise AssertionError("test plan is not bound to the deterministic manifest")
    return manifest
