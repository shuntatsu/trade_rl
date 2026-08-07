"""Deterministic persistence for frozen causal scenario libraries."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.causal_scenario.conditions import (
    CausalConditionConfig,
    CausalConditionLayout,
    TrainRobustConditionNormalizer,
)
from trade_rl.workflows.causal_scenario.library import (
    CausalScenarioLibraryConfig,
    FrozenCausalScenarioLibrary,
)

CAUSAL_SCENARIO_LIBRARY_ARTIFACT_SCHEMA: Final = "causal_scenario_library_artifact_v1"
CAUSAL_SCENARIO_LIBRARY_MANIFEST_NAME: Final = "manifest.json"
CAUSAL_SCENARIO_LIBRARY_ARRAYS_NAME: Final = "arrays.npz"
_FIXED_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
_ALLOWED_FILES: Final = frozenset(
    {CAUSAL_SCENARIO_LIBRARY_MANIFEST_NAME, CAUSAL_SCENARIO_LIBRARY_ARRAYS_NAME}
)
_BASE_ARRAYS: Final = (
    "anchor_indices",
    "raw_conditions",
    "normalized_conditions",
    "layout_continuous_mask",
    "normalizer_median",
    "normalizer_scale",
)


def _expected_array_names() -> tuple[str, ...]:
    return tuple(sorted(_BASE_ARRAYS))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, np.asarray(array), allow_pickle=False)
    return output.getvalue()


def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _npy_bytes(arrays[name]))
    return output.getvalue()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _verify_exact_files(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"causal scenario library artifact is missing: {root}")
    actual: set[str] = set()
    for entry in root.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(
                "causal scenario library artifact contains invalid file entry"
            )
        actual.add(entry.name)
    if actual != _ALLOWED_FILES:
        raise ValueError("causal scenario library artifact file closure mismatch")


def _prepare_root(root: Path) -> None:
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise ValueError("causal scenario library artifact root is invalid")
        for entry in root.iterdir():
            if (
                entry.is_symlink()
                or not entry.is_file()
                or entry.name not in _ALLOWED_FILES
            ):
                raise ValueError(
                    "causal scenario library artifact root contains invalid entries"
                )
    else:
        root.mkdir(parents=True)


def _array_metadata(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "dtype": array.dtype.str,
            "shape": tuple(int(size) for size in array.shape),
        }
        for name, array in sorted(arrays.items())
    }


def _library_arrays(library: FrozenCausalScenarioLibrary) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "anchor_indices": library.anchor_indices,
        "raw_conditions": library.raw_conditions,
        "normalized_conditions": library.normalized_conditions,
        "layout_continuous_mask": library.layout.continuous_mask,
        "normalizer_median": library.normalizer.median,
        "normalizer_scale": library.normalizer.scale,
    }
    if set(arrays) != set(_expected_array_names()):
        raise RuntimeError("causal scenario library array closure is invalid")
    return arrays


def _condition_payload(config: CausalConditionConfig) -> dict[str, object]:
    return {
        "correlation_hours": config.correlation_hours,
        "liquidity_floor": config.liquidity_floor,
        "scale_epsilon": config.scale_epsilon,
        "schema_version": config.schema_version,
        "volatility_hours": config.volatility_hours,
    }


def _config_payload(config: CausalScenarioLibraryConfig) -> dict[str, object]:
    return {
        "condition": _condition_payload(config.condition),
        "horizon_decisions": config.horizon_decisions,
        "relative_floor": config.relative_floor,
        "scenario_count": config.scenario_count,
        "schema_version": config.schema_version,
    }


def _manifest_base(
    library: FrozenCausalScenarioLibrary,
    *,
    arrays_digest: str,
    metadata: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        "anchor_count": library.anchor_count,
        "array_metadata": {
            name: dict(value) for name, value in sorted(metadata.items())
        },
        "arrays_digest": arrays_digest,
        "arrays_file": CAUSAL_SCENARIO_LIBRARY_ARRAYS_NAME,
        "config_payload": _config_payload(library.config),
        "dataset_id": library.dataset_id,
        "feature_names": library.feature_names,
        "global_feature_names": library.global_feature_names,
        "layout_feature_names": library.layout.feature_names,
        "layout_schema_version": library.layout.schema_version,
        "library_digest": library.library_digest,
        "normalizer_schema_version": library.normalizer.schema_version,
        "schema_version": CAUSAL_SCENARIO_LIBRARY_ARTIFACT_SCHEMA,
        "symbols": library.symbols,
        "train_start": library.train_start,
        "train_stop": library.train_stop,
        "train_view_digest": library.train_view_digest,
        "trend_config_payload": dict(library.trend_config_payload),
    }


def write_causal_scenario_library_artifact(
    root: str | Path,
    library: FrozenCausalScenarioLibrary,
) -> str:
    target = Path(root)
    _prepare_root(target)
    arrays = _library_arrays(library)
    arrays_payload = _deterministic_npz(arrays)
    arrays_digest = _sha256_bytes(arrays_payload)
    base = _manifest_base(
        library,
        arrays_digest=arrays_digest,
        metadata=_array_metadata(arrays),
    )
    artifact_digest = content_digest(base)
    manifest = dict(base)
    manifest["artifact_digest"] = artifact_digest
    _atomic_write(target / CAUSAL_SCENARIO_LIBRARY_ARRAYS_NAME, arrays_payload)
    _atomic_write(
        target / CAUSAL_SCENARIO_LIBRARY_MANIFEST_NAME,
        canonical_json_bytes(manifest),
    )
    return artifact_digest


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise ValueError(f"{field} must be a sequence")


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _strings(value: object, *, field: str) -> tuple[str, ...]:
    return tuple(
        _string(item, field=f"{field}[{index}]")
        for index, item in enumerate(_sequence(value, field=field))
    )


def _condition_from_payload(payload: Mapping[str, object]) -> CausalConditionConfig:
    expected = {
        "correlation_hours",
        "liquidity_floor",
        "scale_epsilon",
        "schema_version",
        "volatility_hours",
    }
    if set(payload) != expected:
        raise ValueError("condition config field closure mismatch")
    return CausalConditionConfig(
        volatility_hours=_float(payload["volatility_hours"], field="volatility_hours"),
        correlation_hours=_float(
            payload["correlation_hours"], field="correlation_hours"
        ),
        scale_epsilon=_float(payload["scale_epsilon"], field="scale_epsilon"),
        liquidity_floor=_float(payload["liquidity_floor"], field="liquidity_floor"),
        schema_version=_string(
            payload["schema_version"], field="condition.schema_version"
        ),
    )


def _config_from_payload(payload: Mapping[str, object]) -> CausalScenarioLibraryConfig:
    expected = {
        "condition",
        "horizon_decisions",
        "relative_floor",
        "scenario_count",
        "schema_version",
    }
    if set(payload) != expected:
        raise ValueError("library config field closure mismatch")
    return CausalScenarioLibraryConfig(
        horizon_decisions=_integer(
            payload["horizon_decisions"], field="horizon_decisions"
        ),
        scenario_count=_integer(payload["scenario_count"], field="scenario_count"),
        relative_floor=_float(payload["relative_floor"], field="relative_floor"),
        condition=_condition_from_payload(
            _mapping(payload["condition"], field="condition")
        ),
        schema_version=_string(
            payload["schema_version"], field="config.schema_version"
        ),
    )


def _load_manifest(root: Path) -> tuple[dict[str, object], bytes]:
    _verify_exact_files(root)
    raw = json.loads(
        (root / CAUSAL_SCENARIO_LIBRARY_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    manifest = dict(_mapping(raw, field="manifest"))
    artifact_digest = _string(
        manifest.pop("artifact_digest", None), field="artifact_digest"
    )
    if content_digest(manifest) != artifact_digest:
        raise ValueError("causal scenario library manifest digest mismatch")
    expected_fields = {
        "anchor_count",
        "array_metadata",
        "arrays_digest",
        "arrays_file",
        "config_payload",
        "dataset_id",
        "feature_names",
        "global_feature_names",
        "layout_feature_names",
        "layout_schema_version",
        "library_digest",
        "normalizer_schema_version",
        "schema_version",
        "symbols",
        "train_start",
        "train_stop",
        "train_view_digest",
        "trend_config_payload",
    }
    if set(manifest) != expected_fields:
        raise ValueError("causal scenario library manifest field closure mismatch")
    if manifest["arrays_file"] != CAUSAL_SCENARIO_LIBRARY_ARRAYS_NAME:
        raise ValueError("causal scenario library arrays file identity is invalid")
    if manifest["schema_version"] != CAUSAL_SCENARIO_LIBRARY_ARTIFACT_SCHEMA:
        raise ValueError("unsupported causal scenario library artifact schema")
    payload = (root / CAUSAL_SCENARIO_LIBRARY_ARRAYS_NAME).read_bytes()
    if _sha256_bytes(payload) != manifest["arrays_digest"]:
        raise ValueError("causal scenario library arrays digest mismatch")
    manifest["artifact_digest"] = artifact_digest
    return manifest, payload


def _load_arrays(
    payload: bytes,
    metadata: Mapping[str, object],
) -> dict[str, np.ndarray]:
    expected_names = set(_expected_array_names())
    if set(metadata) != expected_names:
        raise ValueError("causal scenario library array metadata closure mismatch")
    arrays: dict[str, np.ndarray] = {}
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != expected_names:
            raise ValueError("causal scenario library array name closure mismatch")
        for name in sorted(expected_names):
            item = _mapping(metadata[name], field=f"array_metadata.{name}")
            if set(item) != {"dtype", "shape"}:
                raise ValueError(
                    f"causal scenario library array metadata invalid: {name}"
                )
            shape = tuple(
                _integer(value, field=f"array_metadata.{name}.shape")
                for value in _sequence(
                    item["shape"], field=f"array_metadata.{name}.shape"
                )
            )
            array = np.asarray(archive[name])
            if array.shape != shape:
                raise ValueError(
                    f"causal scenario library array shape mismatch: {name}"
                )
            if array.dtype.str != item["dtype"]:
                raise ValueError(
                    f"causal scenario library array dtype mismatch: {name}"
                )
            arrays[name] = array
    return arrays


def load_causal_scenario_library_artifact(
    root: str | Path,
) -> FrozenCausalScenarioLibrary:
    manifest, payload = _load_manifest(Path(root))
    arrays = _load_arrays(
        payload,
        _mapping(manifest["array_metadata"], field="array_metadata"),
    )
    config = _config_from_payload(
        _mapping(manifest["config_payload"], field="config_payload")
    )
    symbols = _strings(manifest["symbols"], field="symbols")
    layout = CausalConditionLayout(
        symbols=symbols,
        feature_names=_strings(
            manifest["layout_feature_names"], field="layout_feature_names"
        ),
        continuous_mask=arrays["layout_continuous_mask"],
        schema_version=_string(
            manifest["layout_schema_version"], field="layout_schema_version"
        ),
    )
    train_view_digest = _string(
        manifest["train_view_digest"], field="train_view_digest"
    )
    normalizer = TrainRobustConditionNormalizer(
        feature_names=layout.feature_names,
        continuous_mask=layout.continuous_mask,
        median=arrays["normalizer_median"],
        scale=arrays["normalizer_scale"],
        train_view_digest=train_view_digest,
        schema_version=_string(
            manifest["normalizer_schema_version"], field="normalizer_schema_version"
        ),
    )
    anchor_count = _integer(manifest["anchor_count"], field="anchor_count")
    if arrays["anchor_indices"].shape != (anchor_count,):
        raise ValueError("anchor count does not match anchor index array")
    width = len(layout.feature_names)
    expected_condition_shape = (anchor_count, width)
    if arrays["raw_conditions"].shape != expected_condition_shape:
        raise ValueError("raw condition array shape does not match anchor count")
    if arrays["normalized_conditions"].shape != expected_condition_shape:
        raise ValueError("normalized condition array shape does not match anchor count")
    trend_payload = dict(
        _mapping(manifest["trend_config_payload"], field="trend_config_payload")
    )
    return FrozenCausalScenarioLibrary(
        dataset_id=_string(manifest["dataset_id"], field="dataset_id"),
        train_view_digest=train_view_digest,
        train_start=_integer(manifest["train_start"], field="train_start"),
        train_stop=_integer(manifest["train_stop"], field="train_stop"),
        symbols=symbols,
        feature_names=_strings(manifest["feature_names"], field="feature_names"),
        global_feature_names=_strings(
            manifest["global_feature_names"], field="global_feature_names"
        ),
        config=config,
        trend_config_payload=trend_payload,
        layout=layout,
        normalizer=normalizer,
        anchor_indices=arrays["anchor_indices"],
        raw_conditions=arrays["raw_conditions"],
        normalized_conditions=arrays["normalized_conditions"],
        library_digest=_string(manifest["library_digest"], field="library_digest"),
    )


__all__ = [
    "CAUSAL_SCENARIO_LIBRARY_ARRAYS_NAME",
    "CAUSAL_SCENARIO_LIBRARY_ARTIFACT_SCHEMA",
    "CAUSAL_SCENARIO_LIBRARY_MANIFEST_NAME",
    "load_causal_scenario_library_artifact",
    "write_causal_scenario_library_artifact",
]
