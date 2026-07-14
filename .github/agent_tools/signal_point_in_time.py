from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "tests/artifacts/test_signal_point_in_time.py"


def write_tests() -> None:
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PATH.write_text(
        '''from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.signals import write_signal_artifact
from trade_rl.integrations.signal_artifacts import load_alpha_artifact

DATASET_ID = "d" * 64


def test_rowwise_knowledge_cutoff_allows_first_fold_oof_signal(tmp_path: Path) -> None:
    values = np.arange(20, dtype=np.float64)[:, None]
    valid = np.zeros(20, dtype=np.bool_)
    valid[4:] = True
    cutoff = np.full(20, -1, dtype=np.int64)
    cutoff[4:] = np.arange(3, 19, dtype=np.int64)
    write_signal_artifact(
        tmp_path,
        kind="alpha",
        dataset_id=DATASET_ID,
        fit_start=0,
        fit_stop=12,
        names=("BTC",),
        values=values,
        valid_mask=valid,
        knowledge_cutoff=cutoff,
        generator_digest="a" * 64,
    )

    loaded = load_alpha_artifact(
        tmp_path,
        dataset_id=DATASET_ID,
        evaluation_start=0,
        expected_symbols=("BTC",),
    )
    assert loaded.minimum_index == 4
    dataset = type("Dataset", (), {"dataset_id": DATASET_ID, "n_bars": 20})()
    np.testing.assert_array_equal(loaded.predict_at(dataset, 4), values[4])


def test_signal_artifact_rejects_future_knowledge_cutoff(tmp_path: Path) -> None:
    values = np.zeros((8, 1), dtype=np.float64)
    valid = np.ones(8, dtype=np.bool_)
    cutoff = np.arange(8, dtype=np.int64)
    with pytest.raises(ValueError, match="knowledge cutoff"):
        write_signal_artifact(
            tmp_path,
            kind="alpha",
            dataset_id=DATASET_ID,
            fit_start=0,
            fit_stop=4,
            names=("BTC",),
            values=values,
            valid_mask=valid,
            knowledge_cutoff=cutoff,
            generator_digest="a" * 64,
        )


def test_invalid_signal_rows_cannot_be_consumed(tmp_path: Path) -> None:
    values = np.zeros((8, 1), dtype=np.float64)
    valid = np.zeros(8, dtype=np.bool_)
    valid[3:] = True
    cutoff = np.full(8, -1, dtype=np.int64)
    cutoff[3:] = np.arange(2, 7, dtype=np.int64)
    write_signal_artifact(
        tmp_path,
        kind="alpha",
        dataset_id=DATASET_ID,
        fit_start=0,
        fit_stop=4,
        names=("BTC",),
        values=values,
        valid_mask=valid,
        knowledge_cutoff=cutoff,
        generator_digest="a" * 64,
    )
    loaded = load_alpha_artifact(tmp_path, dataset_id=DATASET_ID)
    dataset = type("Dataset", (), {"dataset_id": DATASET_ID, "n_bars": 8})()
    with pytest.raises(ValueError, match="unavailable"):
        loaded.predict_at(dataset, 2)
''',
        encoding="utf-8",
    )


def apply_implementation() -> None:
    signals = ROOT / "trade_rl/artifacts/signals.py"
    signals.write_text(
        '''"""Canonical point-in-time alpha and factor array artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

SIGNAL_MANIFEST_NAME: Final = "manifest.json"
SIGNAL_ARRAYS_NAME: Final = "arrays.npz"
SIGNAL_ARTIFACT_SCHEMA: Final = "causal_signal_array_artifact_v2"
_FIXED_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
SignalKind = Literal["alpha", "factor"]


@dataclass(frozen=True, slots=True)
class SignalArrayManifest:
    artifact_digest: str
    arrays_digest: str
    dataset_id: str
    fit_start: int
    fit_stop: int
    kind: SignalKind
    names: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    generator_digest: str
    schema_version: str = SIGNAL_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        require_sha256(self.artifact_digest, field="artifact_digest")
        require_sha256(self.arrays_digest, field="arrays_digest")
        require_sha256(self.dataset_id, field="dataset_id")
        require_sha256(self.generator_digest, field="generator_digest")
        if self.kind not in {"alpha", "factor"}:
            raise ValueError("signal artifact kind is unsupported")
        if self.fit_start < 0 or self.fit_stop <= self.fit_start:
            raise ValueError("signal fit range must be non-empty and half-open")
        if not self.names or any(not name for name in self.names):
            raise ValueError("signal names must be non-empty")
        if len(set(self.names)) != len(self.names):
            raise ValueError("signal names must be unique")
        if not self.shape or any(size <= 0 for size in self.shape):
            raise ValueError("signal shape must be positive")
        if self.schema_version != SIGNAL_ARTIFACT_SCHEMA:
            raise ValueError("unsupported signal artifact schema")

    def digest_payload(self) -> dict[str, object]:
        return {
            "arrays_digest": self.arrays_digest,
            "arrays_file": SIGNAL_ARRAYS_NAME,
            "dataset_id": self.dataset_id,
            "dtype": self.dtype,
            "fit_start": self.fit_start,
            "fit_stop": self.fit_stop,
            "generator_digest": self.generator_digest,
            "kind": self.kind,
            "names": self.names,
            "schema_version": self.schema_version,
            "shape": self.shape,
        }


@dataclass(frozen=True, slots=True)
class SignalArrayPayload:
    values: np.ndarray
    valid_mask: np.ndarray
    knowledge_cutoff: np.ndarray


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            npy = io.BytesIO()
            np.lib.format.write_array(npy, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, npy.getvalue())
    return output.getvalue()


def _availability(
    n_bars: int,
    *,
    fit_stop: int,
    valid_mask: np.ndarray | None,
    knowledge_cutoff: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if valid_mask is None:
        valid = np.arange(n_bars, dtype=np.int64) >= fit_stop
    else:
        valid = np.asarray(valid_mask, dtype=np.bool_).reshape(-1)
    if valid.shape != (n_bars,):
        raise ValueError("signal valid_mask must match the bar count")
    if knowledge_cutoff is None:
        cutoff = np.full(n_bars, -1, dtype=np.int64)
        cutoff[valid] = fit_stop - 1
    else:
        cutoff = np.asarray(knowledge_cutoff, dtype=np.int64).reshape(-1)
    if cutoff.shape != (n_bars,):
        raise ValueError("signal knowledge_cutoff must match the bar count")
    indices = np.arange(n_bars, dtype=np.int64)
    if np.any(cutoff[~valid] != -1):
        raise ValueError("invalid signal rows must use knowledge cutoff -1")
    if np.any(cutoff[valid] < 0) or np.any(cutoff[valid] >= indices[valid]):
        raise ValueError("signal knowledge cutoff must strictly precede each prediction")
    return valid, cutoff


def write_signal_artifact(
    root: str | Path,
    *,
    kind: SignalKind,
    dataset_id: str,
    fit_start: int,
    fit_stop: int,
    names: tuple[str, ...],
    values: np.ndarray,
    valid_mask: np.ndarray | None = None,
    knowledge_cutoff: np.ndarray | None = None,
    generator_digest: str | None = None,
) -> str:
    array = np.asarray(values)
    if array.ndim not in {2, 3} or not np.issubdtype(array.dtype, np.number):
        raise ValueError("signal values must be a numeric rank-2 or rank-3 array")
    if not np.isfinite(array).all():
        raise ValueError("signal values must be finite")
    if kind == "alpha" and (array.ndim != 2 or array.shape[1] != len(names)):
        raise ValueError("alpha values must have shape (bars, symbols)")
    if kind == "factor" and (array.ndim != 3 or array.shape[1] != len(names)):
        raise ValueError("factor values must have shape (bars, factors, symbols)")
    if fit_start < 0 or fit_stop <= fit_start or fit_stop > array.shape[0]:
        raise ValueError("signal fit range is invalid")
    valid, cutoff = _availability(
        array.shape[0],
        fit_stop=fit_stop,
        valid_mask=valid_mask,
        knowledge_cutoff=knowledge_cutoff,
    )
    resolved_generator = generator_digest or content_digest(
        {
            "kind": kind,
            "schema_version": "default_signal_generator_contract_v1",
        }
    )
    require_sha256(resolved_generator, field="generator_digest")
    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    payload = _deterministic_npz(
        {
            "knowledge_cutoff": cutoff.astype(np.int64),
            "valid_mask": valid.astype(np.bool_),
            "values": array,
        }
    )
    arrays_digest = _sha256(payload)
    base = {
        "arrays_digest": arrays_digest,
        "arrays_file": SIGNAL_ARRAYS_NAME,
        "dataset_id": dataset_id,
        "dtype": array.dtype.str,
        "fit_start": fit_start,
        "fit_stop": fit_stop,
        "generator_digest": resolved_generator,
        "kind": kind,
        "names": names,
        "schema_version": SIGNAL_ARTIFACT_SCHEMA,
        "shape": tuple(int(size) for size in array.shape),
    }
    manifest = SignalArrayManifest(
        artifact_digest=content_digest(base),
        arrays_digest=arrays_digest,
        dataset_id=dataset_id,
        fit_start=fit_start,
        fit_stop=fit_stop,
        kind=kind,
        names=names,
        shape=tuple(int(size) for size in array.shape),
        dtype=array.dtype.str,
        generator_digest=resolved_generator,
    )
    _atomic_write(output / SIGNAL_ARRAYS_NAME, payload)
    _atomic_write(output / SIGNAL_MANIFEST_NAME, canonical_json_bytes(manifest))
    return manifest.artifact_digest


def _load_signal_payload(
    root: str | Path,
    *,
    expected_kind: SignalKind | None = None,
) -> tuple[SignalArrayManifest, SignalArrayPayload]:
    path = Path(root)
    manifest_path = path / SIGNAL_MANIFEST_NAME
    arrays_path = path / SIGNAL_ARRAYS_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"signal manifest is missing: {manifest_path}")
    if not arrays_path.is_file():
        raise FileNotFoundError(f"signal arrays are missing: {arrays_path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("signal manifest must be a mapping")
    try:
        manifest = SignalArrayManifest(
            artifact_digest=str(raw["artifact_digest"]),
            arrays_digest=str(raw["arrays_digest"]),
            dataset_id=str(raw["dataset_id"]),
            fit_start=int(raw["fit_start"]),
            fit_stop=int(raw["fit_stop"]),
            kind=str(raw["kind"]),  # type: ignore[arg-type]
            names=tuple(str(value) for value in raw["names"]),
            shape=tuple(int(value) for value in raw["shape"]),
            dtype=str(raw["dtype"]),
            generator_digest=str(raw["generator_digest"]),
            schema_version=str(raw["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("signal manifest is invalid") from error
    if content_digest(manifest.digest_payload()) != manifest.artifact_digest:
        raise ValueError("signal manifest digest mismatch")
    if expected_kind is not None and manifest.kind != expected_kind:
        raise ValueError("signal artifact kind mismatch")
    payload = arrays_path.read_bytes()
    if _sha256(payload) != manifest.arrays_digest:
        raise ValueError("signal arrays digest mismatch")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != {"values", "valid_mask", "knowledge_cutoff"}:
                raise ValueError("signal array allow-list mismatch")
            values = np.asarray(archive["values"])
            valid = np.asarray(archive["valid_mask"], dtype=np.bool_)
            cutoff = np.asarray(archive["knowledge_cutoff"], dtype=np.int64)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("signal arrays are invalid") from error
    if values.shape != manifest.shape or values.dtype.str != manifest.dtype:
        raise ValueError("signal array shape or dtype mismatch")
    if not np.isfinite(values).all():
        raise ValueError("signal values must be finite")
    valid, cutoff = _availability(
        values.shape[0],
        fit_stop=manifest.fit_stop,
        valid_mask=valid,
        knowledge_cutoff=cutoff,
    )
    return manifest, SignalArrayPayload(values, valid, cutoff)


def load_signal_artifact(
    root: str | Path,
    *,
    expected_kind: SignalKind | None = None,
) -> tuple[SignalArrayManifest, np.ndarray]:
    manifest, payload = _load_signal_payload(root, expected_kind=expected_kind)
    return manifest, payload.values


def load_signal_artifact_payload(
    root: str | Path,
    *,
    expected_kind: SignalKind | None = None,
) -> tuple[SignalArrayManifest, SignalArrayPayload]:
    return _load_signal_payload(root, expected_kind=expected_kind)


__all__ = [
    "SIGNAL_ARRAYS_NAME",
    "SIGNAL_ARTIFACT_SCHEMA",
    "SIGNAL_MANIFEST_NAME",
    "SignalArrayManifest",
    "SignalArrayPayload",
    "load_signal_artifact",
    "load_signal_artifact_payload",
    "write_signal_artifact",
]
''',
        encoding="utf-8",
    )

    integrations = ROOT / "trade_rl/integrations/signal_artifacts.py"
    integrations.write_text(
        '''"""Validated point-in-time alpha and factor artifact providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trade_rl.artifacts.signals import (
    SignalArrayManifest,
    load_signal_artifact_payload,
)
from trade_rl.data.market import MarketDataset


@dataclass(frozen=True, slots=True)
class LoadedAlphaArtifact:
    manifest: SignalArrayManifest
    values: np.ndarray
    valid_mask: np.ndarray
    knowledge_cutoff: np.ndarray
    bound_dataset_id: str | None = None
    offset: int = 0
    bound_bars: int | None = None

    @property
    def artifact_digest(self) -> str:
        return self.manifest.artifact_digest

    @property
    def minimum_index(self) -> int:
        return _minimum_index(
            self.valid_mask,
            offset=self.offset,
            bound_bars=self.bound_bars,
        )

    @property
    def dataset_id(self) -> str:
        return self.bound_dataset_id or self.manifest.dataset_id

    def for_view(
        self, *, start: int, stop: int, dataset_id: str
    ) -> LoadedAlphaArtifact:
        _validate_view(self.values, start=start, stop=stop)
        return LoadedAlphaArtifact(
            manifest=self.manifest,
            values=self.values,
            valid_mask=self.valid_mask,
            knowledge_cutoff=self.knowledge_cutoff,
            bound_dataset_id=dataset_id,
            offset=start,
            bound_bars=stop - start,
        )

    def predict_at(self, dataset: MarketDataset, index: int) -> np.ndarray:
        source_index = _validate_dataset_and_index(
            self.manifest,
            self.values,
            self.valid_mask,
            self.knowledge_cutoff,
            dataset,
            index,
            dataset_id=self.dataset_id,
            offset=self.offset,
            bound_bars=self.bound_bars,
        )
        return np.asarray(self.values[source_index], dtype=np.float64).copy()


@dataclass(frozen=True, slots=True)
class LoadedFactorArtifact:
    manifest: SignalArrayManifest
    values: np.ndarray
    valid_mask: np.ndarray
    knowledge_cutoff: np.ndarray
    bound_dataset_id: str | None = None
    offset: int = 0
    bound_bars: int | None = None

    @property
    def artifact_digest(self) -> str:
        return self.manifest.artifact_digest

    @property
    def minimum_index(self) -> int:
        return _minimum_index(
            self.valid_mask,
            offset=self.offset,
            bound_bars=self.bound_bars,
        )

    @property
    def dataset_id(self) -> str:
        return self.bound_dataset_id or self.manifest.dataset_id

    @property
    def n_factors(self) -> int:
        return len(self.manifest.names)

    def for_view(
        self, *, start: int, stop: int, dataset_id: str
    ) -> LoadedFactorArtifact:
        _validate_view(self.values, start=start, stop=stop)
        return LoadedFactorArtifact(
            manifest=self.manifest,
            values=self.values,
            valid_mask=self.valid_mask,
            knowledge_cutoff=self.knowledge_cutoff,
            bound_dataset_id=dataset_id,
            offset=start,
            bound_bars=stop - start,
        )

    def basis_at(self, dataset: MarketDataset, index: int) -> np.ndarray:
        source_index = _validate_dataset_and_index(
            self.manifest,
            self.values,
            self.valid_mask,
            self.knowledge_cutoff,
            dataset,
            index,
            dataset_id=self.dataset_id,
            offset=self.offset,
            bound_bars=self.bound_bars,
        )
        return np.asarray(self.values[source_index], dtype=np.float64).copy()


def _validate_view(values: np.ndarray, *, start: int, stop: int) -> None:
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or not 0 <= start < stop <= values.shape[0]
    ):
        raise ValueError("signal artifact view is outside the source values")


def _minimum_index(
    valid_mask: np.ndarray,
    *,
    offset: int,
    bound_bars: int | None,
) -> int:
    stop = valid_mask.shape[0] if bound_bars is None else offset + bound_bars
    available = np.flatnonzero(valid_mask[offset:stop])
    if available.size == 0:
        raise ValueError("signal artifact has no valid predictions in the bound range")
    return int(available[0])


def _validate_dataset_and_index(
    manifest: SignalArrayManifest,
    values: np.ndarray,
    valid_mask: np.ndarray,
    knowledge_cutoff: np.ndarray,
    dataset: MarketDataset,
    index: int,
    *,
    dataset_id: str,
    offset: int,
    bound_bars: int | None,
) -> int:
    if dataset.dataset_id != dataset_id:
        raise ValueError("signal artifact dataset identity mismatch")
    expected_bars = values.shape[0] if bound_bars is None else bound_bars
    if dataset.n_bars != expected_bars:
        raise ValueError("signal artifact bar count does not match dataset")
    source_index = index + offset
    if not 0 <= index < expected_bars or not valid_mask[source_index]:
        raise ValueError("signal artifact is unavailable at the requested index")
    if knowledge_cutoff[source_index] >= source_index:
        raise ValueError("signal artifact violates its point-in-time knowledge cutoff")
    return source_index


def _validate_common(
    manifest: SignalArrayManifest,
    valid_mask: np.ndarray,
    *,
    dataset_id: str,
    evaluation_start: int | None,
) -> None:
    if manifest.dataset_id != dataset_id:
        raise ValueError("signal artifact dataset identity mismatch")
    if evaluation_start is not None:
        if evaluation_start < 0 or evaluation_start >= valid_mask.shape[0]:
            raise ValueError("signal evaluation start is outside the artifact")
        if not np.any(valid_mask[evaluation_start:]):
            raise ValueError("signal artifact has no valid predictions for evaluation")


def load_alpha_artifact(
    root: str | Path,
    *,
    dataset_id: str,
    evaluation_start: int | None = None,
    expected_symbols: tuple[str, ...] | None = None,
) -> LoadedAlphaArtifact:
    manifest, payload = load_signal_artifact_payload(root, expected_kind="alpha")
    _validate_common(
        manifest,
        payload.valid_mask,
        dataset_id=dataset_id,
        evaluation_start=evaluation_start,
    )
    if expected_symbols is not None and manifest.names != expected_symbols:
        raise ValueError("alpha symbol names do not match the dataset")
    return LoadedAlphaArtifact(
        manifest=manifest,
        values=payload.values,
        valid_mask=payload.valid_mask,
        knowledge_cutoff=payload.knowledge_cutoff,
    )


def load_factor_artifact(
    root: str | Path,
    *,
    dataset_id: str,
    evaluation_start: int | None = None,
    expected_names: tuple[str, ...] | None = None,
    expected_symbols: int | None = None,
) -> LoadedFactorArtifact:
    manifest, payload = load_signal_artifact_payload(root, expected_kind="factor")
    _validate_common(
        manifest,
        payload.valid_mask,
        dataset_id=dataset_id,
        evaluation_start=evaluation_start,
    )
    if expected_names is not None and manifest.names != expected_names:
        raise ValueError("factor names do not match the action specification")
    if expected_symbols is not None and payload.values.shape[2] != expected_symbols:
        raise ValueError("factor artifact symbol count does not match the dataset")
    return LoadedFactorArtifact(
        manifest=manifest,
        values=payload.values,
        valid_mask=payload.valid_mask,
        knowledge_cutoff=payload.knowledge_cutoff,
    )


__all__ = [
    "LoadedAlphaArtifact",
    "LoadedFactorArtifact",
    "load_alpha_artifact",
    "load_factor_artifact",
]
''',
        encoding="utf-8",
    )

    tests = ROOT / "tests/artifacts/test_signal_artifacts.py"
    text = tests.read_text(encoding="utf-8")
    old = '''def test_alpha_artifact_rejects_fit_range_touching_evaluation(tmp_path: Path) -> None:
    write_signal_artifact(
        tmp_path,
        kind="alpha",
        dataset_id=DATASET_ID,
        fit_start=0,
        fit_stop=100,
        names=("BTC", "ETH"),
        values=np.zeros((200, 2), dtype=np.float64),
    )

    with pytest.raises(ValueError, match="strictly before"):
        load_alpha_artifact(
            tmp_path,
            dataset_id=DATASET_ID,
            evaluation_start=99,
            expected_symbols=("BTC", "ETH"),
        )
'''
    new = '''def test_alpha_artifact_uses_first_valid_prediction_after_fit(tmp_path: Path) -> None:
    write_signal_artifact(
        tmp_path,
        kind="alpha",
        dataset_id=DATASET_ID,
        fit_start=0,
        fit_stop=100,
        names=("BTC", "ETH"),
        values=np.zeros((200, 2), dtype=np.float64),
    )

    loaded = load_alpha_artifact(
        tmp_path,
        dataset_id=DATASET_ID,
        evaluation_start=0,
        expected_symbols=("BTC", "ETH"),
    )
    assert loaded.minimum_index == 100
'''
    if text.count(old) != 1:
        raise RuntimeError("legacy alpha evaluation test changed")
    tests.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: signal_point_in_time.py tests|implementation")
    if sys.argv[1] == "tests":
        write_tests()
    else:
        apply_implementation()


if __name__ == "__main__":
    main()
