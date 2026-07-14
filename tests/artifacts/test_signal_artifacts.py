from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.signals import load_signal_artifact, write_signal_artifact
from trade_rl.integrations.signal_artifacts import (
    load_alpha_artifact,
    load_factor_artifact,
)

DATASET_ID = "d" * 64


def test_alpha_artifact_rejects_fit_range_touching_evaluation(tmp_path: Path) -> None:
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


def test_factor_artifact_requires_exact_factor_names(tmp_path: Path) -> None:
    write_signal_artifact(
        tmp_path,
        kind="factor",
        dataset_id=DATASET_ID,
        fit_start=0,
        fit_stop=20,
        names=("value", "carry"),
        values=np.zeros((200, 2, 3), dtype=np.float64),
    )

    with pytest.raises(ValueError, match="factor names"):
        load_factor_artifact(
            tmp_path,
            dataset_id=DATASET_ID,
            evaluation_start=20,
            expected_names=("carry", "value"),
            expected_symbols=3,
        )


def test_signal_artifact_detects_array_tampering(tmp_path: Path) -> None:
    write_signal_artifact(
        tmp_path,
        kind="alpha",
        dataset_id=DATASET_ID,
        fit_start=0,
        fit_stop=10,
        names=("BTC",),
        values=np.zeros((20, 1), dtype=np.float64),
    )
    (tmp_path / "arrays.npz").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="digest"):
        load_signal_artifact(tmp_path)


def test_loaded_signal_artifacts_can_bind_to_a_dataset_view(tmp_path: Path) -> None:
    from trade_rl.integrations.signal_artifacts import load_alpha_artifact

    values = np.arange(20, dtype=np.float64).reshape(10, 2)
    write_signal_artifact(
        tmp_path,
        kind="alpha",
        dataset_id="a" * 64,
        fit_start=0,
        fit_stop=3,
        names=("BTC", "ETH"),
        values=values,
    )
    loaded = load_alpha_artifact(
        tmp_path, dataset_id="a" * 64, expected_symbols=("BTC", "ETH")
    )
    view = loaded.for_view(start=2, stop=8, dataset_id="b" * 64)

    assert view.minimum_index == 1
    dataset = type("Dataset", (), {"dataset_id": "b" * 64, "n_bars": 6})()
    np.testing.assert_array_equal(view.predict_at(dataset, 1), values[3])
