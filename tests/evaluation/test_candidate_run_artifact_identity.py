from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.runs.artifact import inspect_candidate_run_artifact


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
