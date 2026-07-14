from __future__ import annotations

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
