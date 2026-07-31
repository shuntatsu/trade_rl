"""Immutable dataset and fold bindings for Stage A evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, cast

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_non_empty, require_sha256
from trade_rl.evaluation.walk_forward.folds import IndexRange

STAGE_A_EVALUATION_DATASET_MANIFEST_SCHEMA: Final = (
    "stage_a_evaluation_dataset_manifest_v1"
)
_STAGE_A_BAR_SECONDS: Final = 15 * 60
StageAEvaluationDatasetSplit = Literal["validation", "test"]
_SPLITS: Final = frozenset({"validation", "test"})


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return value.astimezone(UTC)


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _sequence(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON list")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _require_fields(
    value: dict[str, object], expected: set[str], *, field: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{field} field closure mismatch")


def _parse_datetime(value: object, *, field: str) -> datetime:
    raw = _string(value, field=field)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    return _aware_utc(parsed, field=field)


def _range_payload(value: IndexRange) -> tuple[int, int]:
    return value.start, value.stop


def _load_range(value: object, *, field: str) -> IndexRange:
    raw = _sequence(value, field=field)
    if len(raw) != 2:
        raise ValueError(f"{field} must contain exactly two integers")
    return IndexRange(
        _non_negative_int(raw[0], field=f"{field}.start"),
        _non_negative_int(raw[1], field=f"{field}.stop"),
    )


@dataclass(frozen=True, slots=True)
class StageAEvaluationDatasetTriplet:
    """One declared evaluation triplet bound to one real dataset identity."""

    split: StageAEvaluationDatasetSplit
    triplet_id: str
    symbols: tuple[str, str, str]
    dataset_id: str

    def __post_init__(self) -> None:
        if self.split not in _SPLITS:
            raise ValueError("Stage A evaluation dataset split is invalid")
        require_sha256(
            self.triplet_id, field="stage_a_evaluation_dataset_triplet.triplet_id"
        )
        require_sha256(
            self.dataset_id, field="stage_a_evaluation_dataset_triplet.dataset_id"
        )
        symbols = tuple(
            require_non_empty(
                symbol, field="stage_a_evaluation_dataset_triplet.symbols"
            )
            for symbol in self.symbols
        )
        if len(symbols) != 3 or len(set(symbols)) != 3:
            raise ValueError(
                "Stage A evaluation dataset triplet requires exactly three unique symbols"
            )
        object.__setattr__(self, "symbols", cast(tuple[str, str, str], symbols))

    def to_json_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "split": self.split,
            "symbols": self.symbols,
            "triplet_id": self.triplet_id,
        }


@dataclass(frozen=True, slots=True)
class StageAEvaluationDatasetFold:
    """Scored validation and sealed-test ranges for one maintained fold."""

    fold: int
    configuration_selection: IndexRange
    test: IndexRange

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fold",
            _non_negative_int(
                self.fold, field="stage_a_evaluation_dataset_fold.fold"
            ),
        )
        if not isinstance(self.configuration_selection, IndexRange) or not isinstance(
            self.test, IndexRange
        ):
            raise ValueError("Stage A evaluation dataset fold ranges must be IndexRange")
        if self.configuration_selection.stop > self.test.start:
            raise ValueError(
                "Stage A evaluation dataset fold selection must precede test range"
            )

    def range_for(self, split: StageAEvaluationDatasetSplit) -> IndexRange:
        if split == "validation":
            return self.configuration_selection
        if split == "test":
            return self.test
        raise ValueError("Stage A evaluation dataset split is invalid")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "configuration_selection": _range_payload(self.configuration_selection),
            "fold": self.fold,
            "test": _range_payload(self.test),
        }


@dataclass(frozen=True, slots=True)
class StageAEvaluationDatasetManifest:
    """Content-addressed source, dataset, triplet, and fold closure for Stage A."""

    symbol_disjoint_manifest_digest: str
    symbol_disjoint_triplet_manifest_digest: str
    source_closure_digest: str
    source_metadata_evidence_digest: str
    indicator_cache_id: str
    feature_identity: str
    timeline_start_time: datetime
    timeline_end_time: datetime
    triplets: tuple[StageAEvaluationDatasetTriplet, ...]
    folds: tuple[StageAEvaluationDatasetFold, ...]
    schema_version: str = STAGE_A_EVALUATION_DATASET_MANIFEST_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_A_EVALUATION_DATASET_MANIFEST_SCHEMA:
            raise ValueError("unsupported Stage A evaluation dataset manifest schema")
        for field_name, value in (
            ("symbol_disjoint_manifest_digest", self.symbol_disjoint_manifest_digest),
            (
                "symbol_disjoint_triplet_manifest_digest",
                self.symbol_disjoint_triplet_manifest_digest,
            ),
            ("source_closure_digest", self.source_closure_digest),
            ("source_metadata_evidence_digest", self.source_metadata_evidence_digest),
            ("feature_identity", self.feature_identity),
        ):
            require_sha256(value, field=f"stage_a_evaluation_dataset.{field_name}")
        cache_id = require_non_empty(
            self.indicator_cache_id,
            field="stage_a_evaluation_dataset.indicator_cache_id",
        )
        start = _aware_utc(
            self.timeline_start_time,
            field="stage_a_evaluation_dataset.timeline_start_time",
        )
        end = _aware_utc(
            self.timeline_end_time,
            field="stage_a_evaluation_dataset.timeline_end_time",
        )
        if end <= start:
            raise ValueError("Stage A evaluation dataset timeline must be increasing")
        start_seconds = int(start.timestamp())
        end_seconds = int(end.timestamp())
        if start_seconds % _STAGE_A_BAR_SECONDS or end_seconds % _STAGE_A_BAR_SECONDS:
            raise ValueError(
                "Stage A evaluation dataset timeline must align to the 15-minute clock"
            )
        total_seconds = end_seconds - start_seconds
        bars, remainder = divmod(total_seconds, _STAGE_A_BAR_SECONDS)
        if remainder or bars < 3:
            raise ValueError(
                "Stage A evaluation dataset timeline must contain aligned bars"
            )

        if not self.triplets:
            raise ValueError("Stage A evaluation dataset manifest requires triplets")
        split_order = {"validation": 0, "test": 1}
        triplets = tuple(
            sorted(self.triplets, key=lambda item: (split_order[item.split], item.triplet_id))
        )
        triplet_ids = tuple(item.triplet_id for item in triplets)
        if len(set(triplet_ids)) != len(triplet_ids):
            raise ValueError("Stage A evaluation dataset triplet IDs must be unique")
        dataset_ids = tuple(item.dataset_id for item in triplets)
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError("Stage A evaluation dataset IDs must be unique")
        split_groups = {
            split: tuple(item for item in triplets if item.split == split)
            for split in cast(tuple[StageAEvaluationDatasetSplit, ...], ("validation", "test"))
        }
        if any(not group for group in split_groups.values()):
            raise ValueError(
                "Stage A evaluation dataset manifest requires validation and test triplets"
            )
        validation_symbols = {
            symbol for item in split_groups["validation"] for symbol in item.symbols
        }
        test_symbols = {symbol for item in split_groups["test"] for symbol in item.symbols}
        if not validation_symbols.isdisjoint(test_symbols):
            raise ValueError("Stage A evaluation dataset split symbols must be disjoint")

        if len(self.folds) < 2:
            raise ValueError("Stage A evaluation dataset manifest requires at least two folds")
        folds = tuple(sorted(self.folds, key=lambda item: item.fold))
        fold_ids = tuple(item.fold for item in folds)
        if len(set(fold_ids)) != len(fold_ids):
            raise ValueError("Stage A evaluation dataset folds must be unique")
        for item in folds:
            if max(item.configuration_selection.stop, item.test.stop) > bars:
                raise ValueError(
                    "Stage A evaluation dataset fold range exceeds the common timeline"
                )
        for previous, current in zip(folds, folds[1:], strict=False):
            if current.test.start < previous.test.stop:
                raise ValueError(
                    "Stage A evaluation dataset test ranges must not overlap"
                )

        object.__setattr__(self, "indicator_cache_id", cache_id)
        object.__setattr__(self, "timeline_start_time", start)
        object.__setattr__(self, "timeline_end_time", end)
        object.__setattr__(self, "triplets", triplets)
        object.__setattr__(self, "folds", folds)
        expected = content_digest(self.digest_payload())
        if self.digest and self.digest != expected:
            raise ValueError("Stage A evaluation dataset manifest digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def n_bars(self) -> int:
        return int(
            (self.timeline_end_time - self.timeline_start_time).total_seconds()
            // _STAGE_A_BAR_SECONDS
        )

    @property
    def folds_declared(self) -> tuple[int, ...]:
        return tuple(item.fold for item in self.folds)

    def triplet_ids_for(
        self, split: StageAEvaluationDatasetSplit
    ) -> tuple[str, ...]:
        if split not in _SPLITS:
            raise ValueError("Stage A evaluation dataset split is invalid")
        return tuple(item.triplet_id for item in self.triplets if item.split == split)

    def triplet_for(
        self, split: StageAEvaluationDatasetSplit, triplet_id: str
    ) -> StageAEvaluationDatasetTriplet:
        if split not in _SPLITS:
            raise ValueError("Stage A evaluation dataset split is invalid")
        require_sha256(triplet_id, field="stage_a_evaluation_dataset.triplet_id")
        for item in self.triplets:
            if item.split == split and item.triplet_id == triplet_id:
                return item
        raise ValueError("Stage A evaluation dataset triplet is not declared")

    def dataset_id_for(
        self, split: StageAEvaluationDatasetSplit, triplet_id: str
    ) -> str:
        return self.triplet_for(split, triplet_id).dataset_id

    def fold_for(self, fold: int) -> StageAEvaluationDatasetFold:
        resolved = _non_negative_int(fold, field="stage_a_evaluation_dataset.fold")
        for item in self.folds:
            if item.fold == resolved:
                return item
        raise ValueError("Stage A evaluation dataset fold is not declared")

    def range_for(
        self, split: StageAEvaluationDatasetSplit, fold: int
    ) -> IndexRange:
        return self.fold_for(fold).range_for(split)

    def constructor_payload(self) -> dict[str, object]:
        return {
            "feature_identity": self.feature_identity,
            "folds": self.folds,
            "indicator_cache_id": self.indicator_cache_id,
            "schema_version": self.schema_version,
            "source_closure_digest": self.source_closure_digest,
            "source_metadata_evidence_digest": self.source_metadata_evidence_digest,
            "symbol_disjoint_manifest_digest": self.symbol_disjoint_manifest_digest,
            "symbol_disjoint_triplet_manifest_digest": (
                self.symbol_disjoint_triplet_manifest_digest
            ),
            "timeline_end_time": self.timeline_end_time,
            "timeline_start_time": self.timeline_start_time,
            "triplets": self.triplets,
        }

    def digest_payload(self) -> dict[str, object]:
        return {
            "feature_identity": self.feature_identity,
            "folds": tuple(item.to_json_dict() for item in self.folds),
            "indicator_cache_id": self.indicator_cache_id,
            "schema_version": self.schema_version,
            "source_closure_digest": self.source_closure_digest,
            "source_metadata_evidence_digest": self.source_metadata_evidence_digest,
            "symbol_disjoint_manifest_digest": self.symbol_disjoint_manifest_digest,
            "symbol_disjoint_triplet_manifest_digest": (
                self.symbol_disjoint_triplet_manifest_digest
            ),
            "timeline_end_time": self.timeline_end_time.isoformat(),
            "timeline_start_time": self.timeline_start_time.isoformat(),
            "triplets": tuple(item.to_json_dict() for item in self.triplets),
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}


def write_stage_a_evaluation_dataset_manifest(
    path: str | Path, manifest: StageAEvaluationDatasetManifest
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_bytes(manifest.to_json_dict()))
    return target


def load_stage_a_evaluation_dataset_manifest(
    path: str | Path,
) -> StageAEvaluationDatasetManifest:
    payload = _mapping(
        json.loads(Path(path).read_text(encoding="utf-8")),
        field="stage_a_evaluation_dataset_manifest",
    )
    _require_fields(
        payload,
        {
            "digest",
            "feature_identity",
            "folds",
            "indicator_cache_id",
            "schema_version",
            "source_closure_digest",
            "source_metadata_evidence_digest",
            "symbol_disjoint_manifest_digest",
            "symbol_disjoint_triplet_manifest_digest",
            "timeline_end_time",
            "timeline_start_time",
            "triplets",
        },
        field="stage_a_evaluation_dataset_manifest",
    )
    schema = _string(payload["schema_version"], field="schema_version")
    if schema != STAGE_A_EVALUATION_DATASET_MANIFEST_SCHEMA:
        raise ValueError("unsupported Stage A evaluation dataset manifest schema")
    triplets: list[StageAEvaluationDatasetTriplet] = []
    for index, raw in enumerate(_sequence(payload["triplets"], field="triplets")):
        item = _mapping(raw, field=f"triplets[{index}]")
        _require_fields(
            item,
            {"dataset_id", "split", "symbols", "triplet_id"},
            field=f"triplets[{index}]",
        )
        symbol_values = _sequence(item["symbols"], field=f"triplets[{index}].symbols")
        if len(symbol_values) != 3:
            raise ValueError(
                f"triplets[{index}].symbols must contain exactly three values"
            )
        split = _string(item["split"], field=f"triplets[{index}].split")
        triplets.append(
            StageAEvaluationDatasetTriplet(
                split=cast(StageAEvaluationDatasetSplit, split),
                triplet_id=_string(
                    item["triplet_id"], field=f"triplets[{index}].triplet_id"
                ),
                symbols=cast(
                    tuple[str, str, str],
                    tuple(
                        _string(value, field=f"triplets[{index}].symbols")
                        for value in symbol_values
                    ),
                ),
                dataset_id=_string(
                    item["dataset_id"], field=f"triplets[{index}].dataset_id"
                ),
            )
        )
    folds: list[StageAEvaluationDatasetFold] = []
    for index, raw in enumerate(_sequence(payload["folds"], field="folds")):
        item = _mapping(raw, field=f"folds[{index}]")
        _require_fields(
            item,
            {"configuration_selection", "fold", "test"},
            field=f"folds[{index}]",
        )
        folds.append(
            StageAEvaluationDatasetFold(
                fold=_non_negative_int(item["fold"], field=f"folds[{index}].fold"),
                configuration_selection=_load_range(
                    item["configuration_selection"],
                    field=f"folds[{index}].configuration_selection",
                ),
                test=_load_range(item["test"], field=f"folds[{index}].test"),
            )
        )
    return StageAEvaluationDatasetManifest(
        symbol_disjoint_manifest_digest=_string(
            payload["symbol_disjoint_manifest_digest"],
            field="symbol_disjoint_manifest_digest",
        ),
        symbol_disjoint_triplet_manifest_digest=_string(
            payload["symbol_disjoint_triplet_manifest_digest"],
            field="symbol_disjoint_triplet_manifest_digest",
        ),
        source_closure_digest=_string(
            payload["source_closure_digest"], field="source_closure_digest"
        ),
        source_metadata_evidence_digest=_string(
            payload["source_metadata_evidence_digest"],
            field="source_metadata_evidence_digest",
        ),
        indicator_cache_id=_string(
            payload["indicator_cache_id"], field="indicator_cache_id"
        ),
        feature_identity=_string(
            payload["feature_identity"], field="feature_identity"
        ),
        timeline_start_time=_parse_datetime(
            payload["timeline_start_time"], field="timeline_start_time"
        ),
        timeline_end_time=_parse_datetime(
            payload["timeline_end_time"], field="timeline_end_time"
        ),
        triplets=tuple(triplets),
        folds=tuple(folds),
        schema_version=schema,
        digest=_string(payload["digest"], field="digest"),
    )


__all__ = [
    "STAGE_A_EVALUATION_DATASET_MANIFEST_SCHEMA",
    "StageAEvaluationDatasetFold",
    "StageAEvaluationDatasetManifest",
    "StageAEvaluationDatasetSplit",
    "StageAEvaluationDatasetTriplet",
    "load_stage_a_evaluation_dataset_manifest",
    "write_stage_a_evaluation_dataset_manifest",
]
