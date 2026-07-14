"""Canonical observation-normalizer sidecars for serving bundles."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.rl.normalization import ObservationNormalizer

NORMALIZER_ARTIFACT_NAME = "normalizer.json"


def write_observation_normalizer(root: Path, normalizer: ObservationNormalizer) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / NORMALIZER_ARTIFACT_NAME
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(
        canonical_json_bytes(
            {"digest": normalizer.digest, **normalizer.digest_payload()}
        )
    )
    temporary.replace(path)
    return path


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return value


def _optional_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer or null")
    return value


def load_observation_normalizer(root: Path) -> ObservationNormalizer:
    path = Path(root) / NORMALIZER_ARTIFACT_NAME
    if not path.is_file():
        raise ValueError("serving bundle normalizer sidecar is missing")
    raw = _mapping(json.loads(path.read_text(encoding="utf-8")), field="normalizer")
    try:
        mean = np.asarray(cast(list[float], raw["mean"]), dtype=np.float64)
        scale = np.asarray(cast(list[float], raw["scale"]), dtype=np.float64)
        passthrough = tuple(
            int(value) for value in cast(list[int], raw["passthrough_indices"])
        )
        normalizer = ObservationNormalizer(
            mean=mean,
            scale=scale,
            train_start=cast(int, raw["train_start"]),
            train_end=cast(int, raw["train_end"]),
            clip=float(cast(int | float, raw["clip"])),
            epsilon=float(cast(int | float, raw["epsilon"])),
            passthrough_indices=passthrough,
            dataset_id=_optional_string(raw.get("dataset_id"), field="dataset_id"),
            source_dataset_id=_optional_string(
                raw.get("source_dataset_id"), field="source_dataset_id"
            ),
            source_dataset_artifact_digest=_optional_string(
                raw.get("source_dataset_artifact_digest"),
                field="source_dataset_artifact_digest",
            ),
            absolute_train_start=_optional_int(
                raw.get("absolute_train_start"), field="absolute_train_start"
            ),
            absolute_train_end=_optional_int(
                raw.get("absolute_train_end"), field="absolute_train_end"
            ),
            observation_schema=str(raw["observation_schema"]),
            observation_schema_digest=_optional_string(
                raw.get("observation_schema_digest"),
                field="observation_schema_digest",
            ),
            action_spec_digest=_optional_string(
                raw.get("action_spec_digest"), field="action_spec_digest"
            ),
            alpha_artifact_digest=_optional_string(
                raw.get("alpha_artifact_digest"), field="alpha_artifact_digest"
            ),
            factor_artifact_digest=_optional_string(
                raw.get("factor_artifact_digest"), field="factor_artifact_digest"
            ),
            candidate_config_digest=_optional_string(
                raw.get("candidate_config_digest"), field="candidate_config_digest"
            ),
            schema_version=str(raw["schema_version"]),
            digest=str(raw["digest"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("serving normalizer sidecar is invalid") from error
    return normalizer


__all__ = [
    "NORMALIZER_ARTIFACT_NAME",
    "load_observation_normalizer",
    "write_observation_normalizer",
]
