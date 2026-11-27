from __future__ import annotations

import io
import json
import warnings
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.runs.artifact import (
    inspect_candidate_run_artifact,
    load_candidate_run_artifact,
)


def _summary() -> dict[str, object]:
    return {
        "schema_version": "lean_candidate_result_v1",
        "dataset_id": "b" * 64,
        "dataset_artifact": {
            "schema_version": "market_dataset_artifact_v3",
            "artifact_digest": "d" * 64,
        },
        "symbols": ["BTCUSDT"],
        "candidate_config": {},
        "evaluation": {},
        "by_symbol": [
            {
                "symbol_index": 0,
                "symbol": "BTCUSDT",
                "strategies": [
                    {
                        "name": "cash",
                        "return_key": "symbol_0_strategy_0",
                        "metrics": {},
                        "diagnostics": {},
                        "final_portfolio_value": 1000.0,
                        "fill_count": 0,
                    }
                ],
            }
        ],
    }


def _provenance() -> dict[str, object]:
    implementation: dict[str, object] = {
        "schema_version": "candidate_run_implementation_v1",
        "files": [],
    }
    runtime: dict[str, object] = {"schema_version": "candidate_run_runtime_v1"}
    return {
        "schema_version": "candidate_run_provenance_v1",
        "implementation": implementation,
        "implementation_digest": content_digest(implementation),
        "runtime_environment": runtime,
        "runtime_environment_digest": content_digest(runtime),
        "research_context_digest": None,
    }


def _write_root(
    root: Path,
    *,
    compressed: bool,
    array: np.ndarray | None = None,
) -> None:
    root.mkdir()
    (root / "summary.json").write_text(
        json.dumps(_summary(), sort_keys=True, indent=2), encoding="utf-8"
    )
    (root / "provenance.json").write_text(
        json.dumps(_provenance(), sort_keys=True, indent=2), encoding="utf-8"
    )
    value = np.asarray([0.0, 0.01, -0.02]) if array is None else array
    writer = np.savez_compressed if compressed else np.savez
    writer(root / "returns.npz", symbol_0_strategy_0=value)


def test_candidate_artifact_semantic_digest_survives_npz_repacking(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write_root(first_root, compressed=True)
    _write_root(second_root, compressed=False)

    first = inspect_candidate_run_artifact(first_root)
    second = inspect_candidate_run_artifact(second_root)

    assert first.returns_file_sha256 != second.returns_file_sha256
    assert first.artifact_digest == second.artifact_digest
    assert first.summary_file_size > 0
    assert first.returns_file_size > 0
    assert first.provenance_file_size > 0


@pytest.mark.parametrize("extra_name", ["extra.json", "notes.txt"])
def test_candidate_artifact_rejects_extra_files(
    tmp_path: Path, extra_name: str
) -> None:
    root = tmp_path / "artifact"
    _write_root(root, compressed=True)
    (root / extra_name).write_text("extra", encoding="utf-8")

    with pytest.raises(
        ValueError, match="exactly summary.json, returns.npz, provenance.json"
    ):
        inspect_candidate_run_artifact(root)


def test_candidate_artifact_rejects_missing_provenance(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write_root(root, compressed=True)
    (root / "provenance.json").unlink()

    with pytest.raises(
        ValueError, match="exactly summary.json, returns.npz, provenance.json"
    ):
        inspect_candidate_run_artifact(root)


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.asarray([[0.0, 0.1]]), "one-dimensional"),
        (np.asarray([0.0, np.nan]), "finite"),
        (np.asarray([0.0, np.inf]), "finite"),
        (np.asarray(["a", "b"], dtype="U1"), "numeric"),
    ],
)
def test_candidate_artifact_rejects_invalid_return_arrays(
    tmp_path: Path,
    array: np.ndarray,
    message: str,
) -> None:
    root = tmp_path / "artifact"
    _write_root(root, compressed=True, array=array)

    with pytest.raises(ValueError, match=message):
        inspect_candidate_run_artifact(root)


def test_candidate_artifact_rejects_object_array_without_pickle_loading(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _write_root(root, compressed=True, array=np.asarray([object()], dtype=object))

    with pytest.raises(ValueError, match="pickle|object|numeric"):
        inspect_candidate_run_artifact(root)


def test_candidate_artifact_rejects_return_key_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "summary.json").write_text(json.dumps(_summary()), encoding="utf-8")
    (root / "provenance.json").write_text(json.dumps(_provenance()), encoding="utf-8")
    np.savez(root / "returns.npz", unexpected=np.asarray([0.0]))

    with pytest.raises(ValueError, match="return keys"):
        inspect_candidate_run_artifact(root)


def test_candidate_artifact_rejects_unsupported_summary_schema(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write_root(root, compressed=True)
    summary = _summary()
    summary["schema_version"] = "unknown"
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate result schema"):
        inspect_candidate_run_artifact(root)


def test_candidate_artifact_rejects_unsupported_provenance_schema(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _write_root(root, compressed=True)
    provenance = _provenance()
    provenance["schema_version"] = "unknown"
    (root / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate provenance schema"):
        inspect_candidate_run_artifact(root)


def test_candidate_artifact_rejects_inconsistent_provenance_digest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _write_root(root, compressed=True)
    provenance = _provenance()
    provenance["implementation_digest"] = "0" * 64
    (root / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="implementation digest"):
        inspect_candidate_run_artifact(root)


def test_candidate_artifact_rejects_symlinked_required_file(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write_root(root, compressed=True)
    real = root / "summary.real"
    (root / "summary.json").rename(real)
    try:
        (root / "summary.json").symlink_to(real.name)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(ValueError, match="regular file|symlink"):
        inspect_candidate_run_artifact(root)


@pytest.mark.parametrize(
    "reader", [load_candidate_run_artifact, inspect_candidate_run_artifact]
)
@pytest.mark.parametrize(
    "duplicate_kind", ["identical", "different", "extensionless-alias"]
)
def test_candidate_reader_rejects_ambiguous_physical_return_members(
    tmp_path: Path, reader: Callable[[str | Path], object], duplicate_kind: str
) -> None:
    root = tmp_path / "artifact"
    _write_root(root, compressed=True)
    path = root / "returns.npz"
    key = "symbol_0_strategy_0.npy"
    with zipfile.ZipFile(path) as archive:
        original = archive.read(key)
    different = io.BytesIO()
    np.save(different, np.asarray([0.0, 0.5, 0.4]), allow_pickle=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, mode="a") as archive:
            archive.writestr(
                key.removesuffix(".npy")
                if duplicate_kind == "extensionless-alias"
                else key,
                original if duplicate_kind == "identical" else different.getvalue(),
            )
    before = path.read_bytes()
    with pytest.raises(ValueError, match="return keys|duplicate|member"):
        reader(root)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "reader", [load_candidate_run_artifact, inspect_candidate_run_artifact]
)
@pytest.mark.parametrize(
    "array",
    [
        np.asarray([0.0 + 0.0j, 0.01 + 0.0j, -0.02 + 0.0j]),
        np.asarray([0.0 + 0.5j, 0.01 - 0.7j, -0.02 + 0.0j]),
        np.asarray([0, 1, 2], dtype="timedelta64[ns]"),
    ],
)
def test_candidate_reader_refuses_nonreal_return_arrays(
    tmp_path: Path, reader: Callable[[str | Path], object], array: np.ndarray
) -> None:
    root = tmp_path / "artifact"
    _write_root(root, compressed=True, array=array)
    with pytest.raises(ValueError, match="real numeric"):
        reader(root)


@pytest.mark.parametrize(
    "reader", [load_candidate_run_artifact, inspect_candidate_run_artifact]
)
def test_candidate_reader_refuses_npy_instead_of_npz(tmp_path: Path, reader) -> None:
    root = tmp_path / "artifact"
    _write_root(root, compressed=True)
    with (root / "returns.npz").open("wb") as handle:
        np.save(handle, np.asarray([0.0, 0.01]), allow_pickle=False)
    with pytest.raises(ValueError, match="NPZ|archive"):
        reader(root)


@pytest.mark.parametrize("dtype", ["int32", "uint64", "float32", "float64"])
def test_candidate_reader_preserves_valid_real_array_dtype_and_bytes(
    tmp_path: Path, dtype: str
) -> None:
    root = tmp_path / "artifact"
    value = np.asarray([0, 1, 2], dtype=dtype)
    _write_root(root, compressed=True, array=value)
    actual = load_candidate_run_artifact(root).returns["symbol_0_strategy_0"]
    assert actual.dtype == value.dtype
    assert actual.tobytes() == value.tobytes()
    assert not actual.flags.writeable


def test_candidate_reader_resolves_exact_physical_members_not_suffix_aliases(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _write_root(root, compressed=True)
    summary = _summary()
    first = summary["by_symbol"][0]["strategies"][0]  # type: ignore[index]
    first["return_key"] = "cell"
    other = {**first, "name": "trend", "return_key": "cell.npy"}
    summary["by_symbol"][0]["strategies"].append(other)  # type: ignore[index]
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    np.savez(
        root / "returns.npz",
        **{"cell": np.asarray([0.1]), "cell.npy": np.asarray([0.2])},
    )
    loaded = load_candidate_run_artifact(root)
    np.testing.assert_array_equal(loaded.returns["cell"], [0.1])
    np.testing.assert_array_equal(loaded.returns["cell.npy"], [0.2])


@pytest.mark.parametrize(
    "reader", [load_candidate_run_artifact, inspect_candidate_run_artifact]
)
def test_candidate_public_reader_refuses_trailing_npy_bytes(
    tmp_path: Path, reader
) -> None:
    root = tmp_path / "artifact"
    _write_root(root, compressed=True)
    path = root / "returns.npz"
    key = "symbol_0_strategy_0.npy"
    with zipfile.ZipFile(path) as archive:
        member = archive.read(key)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(key, member + b"not part of the return array")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="byte extent"):
        reader(root)
    assert path.read_bytes() == original


def test_loaded_semantic_digest_binds_original_snapshot_not_replaced_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _write_root(root, compressed=True)
    loaded = load_candidate_run_artifact(root)
    expected = inspect_candidate_run_artifact(root)
    assert loaded.artifact_digest == expected.artifact_digest
    # Preserve the historical digest encoding, independently reconstructing
    # its payload from the actual primitive fixture.
    from hashlib import sha256

    values = np.asarray([0.0, 0.01, -0.02], dtype=np.float64)
    assert loaded.artifact_digest == content_digest(
        {
            "schema_version": "candidate_run_artifact_identity_v1",
            "summary": _summary(),
            "returns": [
                {
                    "key": "symbol_0_strategy_0",
                    "dtype": values.dtype.str,
                    "shape": [3],
                    "sha256": sha256(values.tobytes()).hexdigest(),
                }
            ],
            "provenance": _provenance(),
        }
    )
    np.savez(root / "returns.npz", symbol_0_strategy_0=values + 0.1)
    assert loaded.artifact_digest == expected.artifact_digest
    assert (
        inspect_candidate_run_artifact(root).artifact_digest != expected.artifact_digest
    )
