from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from trade_rl.evaluation.walk_forward.folds import IndexRange
from trade_rl.workflows.stage_a_evaluation_dataset_manifest import (
    STAGE_A_EVALUATION_DATASET_MANIFEST_SCHEMA,
    StageAEvaluationDatasetFold,
    StageAEvaluationDatasetManifest,
    StageAEvaluationDatasetTriplet,
    load_stage_a_evaluation_dataset_manifest,
    write_stage_a_evaluation_dataset_manifest,
)


def _digest(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


def _manifest() -> StageAEvaluationDatasetManifest:
    return StageAEvaluationDatasetManifest(
        symbol_disjoint_manifest_digest=_digest("symbol-manifest"),
        symbol_disjoint_triplet_manifest_digest=_digest("triplet-manifest"),
        source_closure_digest=_digest("source-closure"),
        source_metadata_evidence_digest=_digest("metadata"),
        indicator_cache_id="cache-2026-07",
        feature_identity=_digest("features"),
        timeline_start_time=datetime(2024, 1, 1, tzinfo=UTC),
        timeline_end_time=datetime(2024, 1, 2, tzinfo=UTC),
        triplets=(
            StageAEvaluationDatasetTriplet(
                split="validation",
                triplet_id=_digest("validation-triplet"),
                symbols=("ETHUSDT", "BNBUSDT", "SOLUSDT"),
                dataset_id=_digest("validation-dataset"),
            ),
            StageAEvaluationDatasetTriplet(
                split="test",
                triplet_id=_digest("test-triplet"),
                symbols=("XRPUSDT", "ADAUSDT", "DOGEUSDT"),
                dataset_id=_digest("test-dataset"),
            ),
        ),
        folds=(
            StageAEvaluationDatasetFold(
                fold=0,
                configuration_selection=IndexRange(20, 30),
                test=IndexRange(35, 45),
            ),
            StageAEvaluationDatasetFold(
                fold=1,
                configuration_selection=IndexRange(45, 55),
                test=IndexRange(60, 70),
            ),
        ),
    )


def test_manifest_resolves_exact_triplet_dataset_and_split_range() -> None:
    manifest = _manifest()
    validation = manifest.triplet_for("validation", _digest("validation-triplet"))

    assert manifest.schema_version == STAGE_A_EVALUATION_DATASET_MANIFEST_SCHEMA
    assert validation.symbols == ("ETHUSDT", "BNBUSDT", "SOLUSDT")
    assert manifest.dataset_id_for("validation", validation.triplet_id) == _digest(
        "validation-dataset"
    )
    assert manifest.range_for("validation", 1) == IndexRange(45, 55)
    assert manifest.range_for("test", 1) == IndexRange(60, 70)
    assert manifest.folds_declared == (0, 1)
    assert manifest.triplet_ids_for("validation") == (validation.triplet_id,)


def test_manifest_json_round_trip_is_strict(tmp_path) -> None:
    manifest = _manifest()
    path = write_stage_a_evaluation_dataset_manifest(
        tmp_path / "manifest.json", manifest
    )

    assert load_stage_a_evaluation_dataset_manifest(path) == manifest

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="field closure mismatch"):
        load_stage_a_evaluation_dataset_manifest(path)


def test_manifest_rejects_legacy_schema(tmp_path) -> None:
    manifest = _manifest()
    payload = manifest.to_json_dict()
    payload["schema_version"] = "stage_a_evaluation_dataset_manifest_v0"
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match="unsupported Stage A evaluation dataset manifest"
    ):
        load_stage_a_evaluation_dataset_manifest(path)


def test_manifest_rejects_cross_split_symbol_reuse() -> None:
    manifest = _manifest()
    reused = StageAEvaluationDatasetTriplet(
        split="test",
        triplet_id=_digest("other-test-triplet"),
        symbols=("ETHUSDT", "ADAUSDT", "DOGEUSDT"),
        dataset_id=_digest("other-test-dataset"),
    )

    with pytest.raises(ValueError, match="split symbols must be disjoint"):
        StageAEvaluationDatasetManifest(
            **{
                **manifest.constructor_payload(),
                "triplets": (manifest.triplets[0], reused),
            }
        )


def test_manifest_rejects_duplicate_triplet_and_dataset_bindings() -> None:
    manifest = _manifest()
    duplicate = StageAEvaluationDatasetTriplet(
        split="test",
        triplet_id=manifest.triplets[0].triplet_id,
        symbols=("XRPUSDT", "ADAUSDT", "DOGEUSDT"),
        dataset_id=manifest.triplets[0].dataset_id,
    )

    with pytest.raises(ValueError, match="triplet IDs must be unique"):
        StageAEvaluationDatasetManifest(
            **{
                **manifest.constructor_payload(),
                "triplets": (manifest.triplets[0], duplicate),
            }
        )


def test_manifest_rejects_fold_range_outside_timeline() -> None:
    manifest = _manifest()
    invalid_fold = StageAEvaluationDatasetFold(
        fold=1,
        configuration_selection=IndexRange(45, 55),
        test=IndexRange(60, 100),
    )

    with pytest.raises(ValueError, match="range exceeds the common timeline"):
        StageAEvaluationDatasetManifest(
            **{
                **manifest.constructor_payload(),
                "folds": (manifest.folds[0], invalid_fold),
            }
        )


def test_manifest_rejects_naive_or_unaligned_timeline() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="timezone"):
        StageAEvaluationDatasetManifest(
            **{
                **manifest.constructor_payload(),
                "timeline_start_time": datetime(2024, 1, 1),
            }
        )

    with pytest.raises(ValueError, match="15-minute clock"):
        StageAEvaluationDatasetManifest(
            **{
                **manifest.constructor_payload(),
                "timeline_end_time": datetime(2024, 1, 2, tzinfo=UTC)
                + timedelta(minutes=1),
            }
        )


def test_triplet_requires_exactly_three_unique_symbols() -> None:
    with pytest.raises(ValueError, match="exactly three unique symbols"):
        StageAEvaluationDatasetTriplet(
            split="validation",
            triplet_id=_digest("triplet"),
            symbols=("ETHUSDT", "ETHUSDT", "SOLUSDT"),
            dataset_id=_digest("dataset"),
        )
