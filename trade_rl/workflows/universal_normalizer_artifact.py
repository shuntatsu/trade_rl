"""Immutable canonical artifact for a Universal shared feature normalizer."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.rl.universal_normalization import SymbolBalancedStandardNormalizer

_FILENAME = "universal-normalizer.json"
_ARTIFACT_SCHEMA = "universal_shared_normalizer_artifact_v1"
_NORMALIZER_VERSION = "symbol_balanced_standard_normalizer_v1"


def _digest_payload(
    *,
    version: str,
    catalog_digest: str,
    split_manifest_digest: str,
    train_symbols: tuple[str, ...],
    fold_train_range: tuple[int, int],
    feature_schema_digest: str,
    sample_count_per_feature: tuple[int, ...],
    mean: np.ndarray,
    std: np.ndarray,
    constant_mask: np.ndarray,
    availability_aware: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "catalog_digest": catalog_digest,
        "constant_mask": constant_mask.tolist(),
        "feature_schema_digest": feature_schema_digest,
        "fold_train_range": fold_train_range,
        "mean": mean.tolist(),
        "sample_count_per_feature": sample_count_per_feature,
        "split_manifest_digest": split_manifest_digest,
        "std": std.tolist(),
        "train_symbols": train_symbols,
        "version": version,
    }
    if availability_aware:
        payload["availability_aware"] = True
    return payload


def _json_payload(
    normalizer: SymbolBalancedStandardNormalizer,
) -> dict[str, object]:
    return {
        "artifact_schema_version": _ARTIFACT_SCHEMA,
        "catalog_digest": normalizer.catalog_digest,
        "clip_value": normalizer.clip_value,
        "constant_mask": normalizer.constant_mask.tolist(),
        "feature_schema_digest": normalizer.feature_schema_digest,
        "fold_train_range": list(normalizer.fold_train_range),
        "mean": normalizer.mean.tolist(),
        "sample_count_per_feature": list(normalizer.sample_count_per_feature),
        "split_manifest_digest": normalizer.split_manifest_digest,
        "statistics_digest": normalizer.statistics_digest,
        "std": normalizer.std.tolist(),
        "train_symbols": list(normalizer.train_symbols),
        "version": normalizer.version,
    }


def write_universal_shared_normalizer(
    root: str | Path, normalizer: SymbolBalancedStandardNormalizer
) -> Path:
    output = Path(root)
    path = output / _FILENAME
    payload = canonical_json_bytes(_json_payload(normalizer)) + b"\n"
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return output
        raise FileExistsError("shared normalizer already exists with different content")
    output.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return output


def load_universal_shared_normalizer(
    root: str | Path,
) -> SymbolBalancedStandardNormalizer:
    path = Path(root) / _FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("shared normalizer JSON is invalid") from error
    expected = {
        "artifact_schema_version",
        "catalog_digest",
        "clip_value",
        "constant_mask",
        "feature_schema_digest",
        "fold_train_range",
        "mean",
        "sample_count_per_feature",
        "split_manifest_digest",
        "statistics_digest",
        "std",
        "train_symbols",
        "version",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("shared normalizer has unknown or missing fields")
    if payload["artifact_schema_version"] != _ARTIFACT_SCHEMA:
        raise ValueError("shared normalizer artifact schema mismatch")
    if payload["version"] != _NORMALIZER_VERSION:
        raise ValueError("shared normalizer version mismatch")
    train_symbols = tuple(payload["train_symbols"])
    fold_range = tuple(payload["fold_train_range"])
    sample_counts = tuple(payload["sample_count_per_feature"])
    if (
        not train_symbols
        or any(not isinstance(item, str) or not item for item in train_symbols)
        or len(set(train_symbols)) != len(train_symbols)
    ):
        raise ValueError("shared normalizer train symbols are invalid")
    if (
        len(fold_range) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in fold_range)
        or fold_range[0] < 0
        or fold_range[1] <= fold_range[0]
    ):
        raise ValueError("shared normalizer fold range is invalid")
    mean = np.asarray(payload["mean"], dtype=np.float64)
    std = np.asarray(payload["std"], dtype=np.float64)
    constant_mask = np.asarray(payload["constant_mask"], dtype=np.bool_)
    if (
        mean.ndim != 1
        or not mean.size
        or std.shape != mean.shape
        or constant_mask.shape != mean.shape
        or len(sample_counts) != len(mean)
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in sample_counts)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0.0)
    ):
        raise ValueError("shared normalizer statistics are invalid")
    catalog_digest = str(payload["catalog_digest"])
    split_digest = str(payload["split_manifest_digest"])
    feature_digest = str(payload["feature_schema_digest"])
    statistics_digest = str(payload["statistics_digest"])
    for field, digest in (
        ("catalog_digest", catalog_digest),
        ("split_manifest_digest", split_digest),
        ("feature_schema_digest", feature_digest),
        ("statistics_digest", statistics_digest),
    ):
        require_sha256(digest, field=f"shared normalizer {field}")
    candidates = {
        content_digest(
            _digest_payload(
                version=str(payload["version"]),
                catalog_digest=catalog_digest,
                split_manifest_digest=split_digest,
                train_symbols=train_symbols,
                fold_train_range=(int(fold_range[0]), int(fold_range[1])),
                feature_schema_digest=feature_digest,
                sample_count_per_feature=tuple(int(item) for item in sample_counts),
                mean=mean,
                std=std,
                constant_mask=constant_mask,
                availability_aware=availability_aware,
            )
        )
        for availability_aware in (False, True)
    }
    if statistics_digest not in candidates:
        raise ValueError("shared normalizer statistics digest mismatch")
    clip_value = payload["clip_value"]
    if (
        isinstance(clip_value, bool)
        or not isinstance(clip_value, int | float)
        or not math.isfinite(float(clip_value))
        or float(clip_value) <= 0.0
    ):
        raise ValueError("shared normalizer clip value is invalid")
    for value in (mean, std, constant_mask):
        value.setflags(write=False)
    return SymbolBalancedStandardNormalizer(
        mean=mean,
        std=std,
        constant_mask=constant_mask,
        train_symbols=train_symbols,
        feature_schema_digest=feature_digest,
        catalog_digest=catalog_digest,
        split_manifest_digest=split_digest,
        fold_train_range=(int(fold_range[0]), int(fold_range[1])),
        sample_count_per_feature=tuple(int(item) for item in sample_counts),
        statistics_digest=statistics_digest,
        version=str(payload["version"]),
        clip_value=float(clip_value),
    )


__all__ = [
    "load_universal_shared_normalizer",
    "write_universal_shared_normalizer",
]
