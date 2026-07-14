from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.signals import load_signal_artifact, write_signal_artifact
from trade_rl.integrations.signal_artifacts import load_alpha_artifact

DATASET_ID = "d" * 64
GENERATOR_CONFIG = "c" * 64
GENERATOR_CODE = "e" * 64


def _write(root: Path, **overrides: object) -> None:
    values = np.arange(12, dtype=np.float64).reshape(6, 2)
    kwargs: dict[str, object] = {
        "kind": "alpha",
        "dataset_id": DATASET_ID,
        "fit_start": 0,
        "fit_stop": 2,
        "prediction_start": 2,
        "prediction_stop": 6,
        "generator_config_digest": GENERATOR_CONFIG,
        "generator_code_digest": GENERATOR_CODE,
        "names": ("BTC", "ETH"),
        "values": values,
        "valid": np.ones_like(values, dtype=np.bool_),
        "available_at": np.arange(6, dtype=np.int64),
    }
    kwargs.update(overrides)
    write_signal_artifact(root, **kwargs)  # type: ignore[arg-type]


def test_signal_v2_records_prediction_and_generator_lineage(tmp_path: Path) -> None:
    _write(tmp_path)
    manifest, arrays = load_signal_artifact(tmp_path)

    assert manifest.prediction_start == 2
    assert manifest.prediction_stop == 6
    assert manifest.generator_config_digest == GENERATOR_CONFIG
    assert manifest.generator_code_digest == GENERATOR_CODE
    assert arrays.valid.shape == (6, 2)
    assert arrays.available_at.shape == (6,)


def test_signal_v2_rejects_prediction_before_fit_stop(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prediction range must start"):
        _write(tmp_path, prediction_start=1)


def test_signal_provider_rejects_delayed_or_invalid_prediction(tmp_path: Path) -> None:
    valid = np.ones((6, 2), dtype=np.bool_)
    valid[3, 0] = False
    available_at = np.arange(6, dtype=np.int64)
    available_at[4] = 5
    _write(tmp_path, valid=valid, available_at=available_at)
    loaded = load_alpha_artifact(
        tmp_path,
        dataset_id=DATASET_ID,
        expected_symbols=("BTC", "ETH"),
    )
    dataset = type(
        "Dataset",
        (),
        {
            "dataset_id": DATASET_ID,
            "n_bars": 6,
            "timestamps": np.arange(6, dtype=np.int64),
        },
    )()

    with pytest.raises(ValueError, match="invalid"):
        loaded.predict_at(dataset, 3)
    with pytest.raises(ValueError, match="not available"):
        loaded.predict_at(dataset, 4)


def test_signal_artifact_rejects_undeclared_file_and_symlink(tmp_path: Path) -> None:
    _write(tmp_path)
    (tmp_path / "extra.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="file closure"):
        load_signal_artifact(tmp_path)

    (tmp_path / "extra.txt").unlink()
    (tmp_path / "link").symlink_to(tmp_path / "arrays.npz")
    with pytest.raises(ValueError, match="file closure"):
        load_signal_artifact(tmp_path)
