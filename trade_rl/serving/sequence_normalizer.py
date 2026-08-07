"""Canonical structured-sequence normalizer sidecars for serving."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.rl.sequence_normalization import SequenceFeatureNormalizer

SEQUENCE_NORMALIZER_ARTIFACT_NAME = "sequence-normalizer.json"


def write_sequence_feature_normalizer(
    root: Path,
    normalizer: SequenceFeatureNormalizer,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    sample_count = normalizer.sample_count
    if sample_count is None:
        raise RuntimeError("sequence normalizer sample counts are unavailable")
    path = root / SEQUENCE_NORMALIZER_ARTIFACT_NAME
    temporary = path.with_name(f".{path.name}.tmp")
    payload = {
        "center": {
            key: tuple(float(value) for value in normalizer.center[key])
            for key in normalizer.feature_names
        },
        "clip": normalizer.clip,
        "dataset_id": normalizer.dataset_id,
        "digest": normalizer.digest,
        "epsilon": normalizer.epsilon,
        "feature_names": dict(normalizer.feature_names),
        "minimum_samples_per_channel": normalizer.minimum_samples_per_channel,
        "sample_count": {
            key: tuple(int(value) for value in sample_count[key])
            for key in normalizer.feature_names
        },
        "scale": {
            key: tuple(float(value) for value in normalizer.scale[key])
            for key in normalizer.feature_names
        },
        "schema_version": normalizer.schema_version,
        "sequence_schema_digest": normalizer.sequence_schema_digest,
        "source_dataset_id": normalizer.source_dataset_id,
        "train_range": [normalizer.train_start, normalizer.train_end],
    }
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(path)
    return path


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    return float(value)


def load_sequence_feature_normalizer(root: Path) -> SequenceFeatureNormalizer:
    path = Path(root) / SEQUENCE_NORMALIZER_ARTIFACT_NAME
    if not path.is_file():
        raise ValueError("serving sequence normalizer sidecar is missing")
    raw = _mapping(
        json.loads(path.read_text(encoding="utf-8")), field="sequence normalizer"
    )
    raw_names = _mapping(raw.get("feature_names"), field="feature_names")
    raw_center = _mapping(raw.get("center"), field="center")
    raw_scale = _mapping(raw.get("scale"), field="scale")
    raw_counts = _mapping(raw.get("sample_count"), field="sample_count")
    raw_range = raw.get("train_range")
    if not isinstance(raw_range, list) or len(raw_range) != 2:
        raise ValueError("sequence normalizer train_range must contain two integers")
    clocks = ("15m", "1h", "4h", "1d")
    try:
        normalizer = SequenceFeatureNormalizer(
            feature_names={
                key: tuple(str(value) for value in cast(list[object], raw_names[key]))
                for key in clocks
            },
            center={
                key: np.asarray(cast(list[float], raw_center[key]), dtype=np.float64)
                for key in clocks
            },
            scale={
                key: np.asarray(cast(list[float], raw_scale[key]), dtype=np.float64)
                for key in clocks
            },
            sample_count={
                key: np.asarray(cast(list[int], raw_counts[key]), dtype=np.int64)
                for key in clocks
            },
            train_start=_integer(raw_range[0], field="train_range[0]"),
            train_end=_integer(raw_range[1], field="train_range[1]"),
            dataset_id=str(raw["dataset_id"]),
            source_dataset_id=str(raw["source_dataset_id"]),
            sequence_schema_digest=str(raw["sequence_schema_digest"]),
            minimum_samples_per_channel=_integer(
                raw["minimum_samples_per_channel"],
                field="minimum_samples_per_channel",
            ),
            clip=_number(raw["clip"], field="clip"),
            epsilon=_number(raw.get("epsilon", 1e-8), field="epsilon"),
            schema_version=str(raw["schema_version"]),
            digest=str(raw["digest"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"serving sequence normalizer sidecar is invalid: {error}"
        ) from error
    return normalizer


__all__ = [
    "SEQUENCE_NORMALIZER_ARTIFACT_NAME",
    "load_sequence_feature_normalizer",
    "write_sequence_feature_normalizer",
]
