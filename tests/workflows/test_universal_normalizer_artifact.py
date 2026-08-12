from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trade_rl.rl.universal_normalization import SymbolBalancedStandardNormalizer
from trade_rl.workflows.universal_normalizer_artifact import (
    load_universal_shared_normalizer,
    write_universal_shared_normalizer,
)


def shared_normalizer_fixture() -> SymbolBalancedStandardNormalizer:
    values = {
        "BTCUSDT": np.asarray([[1.0, 10.0], [2.0, 10.0], [3.0, 10.0]]),
        "ETHUSDT": np.asarray([[2.0, 10.0], [4.0, 10.0], [6.0, 10.0]]),
    }
    available = {
        symbol: np.ones(matrix.shape, dtype=np.bool_)
        for symbol, matrix in values.items()
    }
    return SymbolBalancedStandardNormalizer.fit(
        values,
        train_symbols=("BTCUSDT", "ETHUSDT"),
        feature_schema_digest="1" * 64,
        catalog_digest="2" * 64,
        split_manifest_digest="3" * 64,
        fold_train_range=(0, 3),
        symbol_available=available,
    )


def test_shared_normalizer_round_trip_is_exact_and_immutable(tmp_path: Path) -> None:
    normalizer = shared_normalizer_fixture()

    root = write_universal_shared_normalizer(tmp_path, normalizer)
    second = write_universal_shared_normalizer(tmp_path, normalizer)
    loaded = load_universal_shared_normalizer(root)

    assert second == root
    assert loaded.statistics_digest == normalizer.statistics_digest
    np.testing.assert_array_equal(loaded.mean, normalizer.mean)
    np.testing.assert_array_equal(loaded.std, normalizer.std)
    np.testing.assert_array_equal(loaded.constant_mask, normalizer.constant_mask)


def test_shared_normalizer_artifact_rejects_statistics_drift(tmp_path: Path) -> None:
    root = write_universal_shared_normalizer(tmp_path, shared_normalizer_fixture())
    path = root / "universal-normalizer.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["mean"][0] += 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="statistics digest"):
        load_universal_shared_normalizer(root)
