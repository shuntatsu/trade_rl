from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data import load_market_dataset_artifact
from trade_rl.data.market import MarketDataset


def _write_full_artificial_dataset(root: Path) -> MarketDataset:
    module_name = "tools.tier_d_scaling.fixture"
    assert importlib.util.find_spec(module_name) is not None, (
        "the artificial scaling fixture module must be present"
    )
    from tools.tier_d_scaling.fixture import write_full_artificial_dataset

    return write_full_artificial_dataset(root)


def test_full_artificial_dataset_builds_deterministic_causal_artifact(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    built = _write_full_artificial_dataset(first_root)
    repeated = _write_full_artificial_dataset(second_root)
    restored = load_market_dataset_artifact(first_root)

    assert built.identity_verified
    assert restored.identity_verified
    assert built.dataset_id == repeated.dataset_id == restored.dataset_id
    assert built.n_bars == restored.n_bars == 9_504
    assert built.n_symbols == restored.n_symbols == 2
    assert built.n_features == restored.n_features == 6
    assert restored.symbols == ("ARTIFICIAL_A", "ARTIFICIAL_B")
    assert restored.feature_names == (
        "artificial_return_1",
        "artificial_return_24",
        "artificial_body",
        "artificial_range",
        "artificial_volatility",
        "artificial_volume_z",
    )

    expected_times = np.datetime64("2000-01-01T00:00:00", "ns") + np.arange(
        9_504
    ) * np.timedelta64(1, "h")
    np.testing.assert_array_equal(restored.timestamps, expected_times)
    assert restored.timestamps.dtype == np.dtype("datetime64[ns]")
    assert restored.features.shape == (9_504, 2, 6)

    assert np.isfinite(restored.close).all()
    assert (restored.close > 0.0).all()
    np.testing.assert_array_equal(restored.close[0], np.array([100.0, 125.0]))
    np.testing.assert_array_equal(
        restored.close[2_903:],
        np.broadcast_to(restored.close[2_903], restored.close[2_903:].shape),
    )
    np.testing.assert_array_equal(restored.volume, np.full((9_504, 2), 1e7))

    funding_event_count = restored.resolved_array("funding_event_count")
    funding_price_rate = restored.resolved_array("funding_price_rate")
    np.testing.assert_array_equal(
        funding_event_count[::8], np.ones((9_504 // 8, 2), dtype=np.int32)
    )
    np.testing.assert_array_equal(
        restored.funding_rate[::8], np.full((9_504 // 8, 2), 0.0001)
    )
    np.testing.assert_array_equal(
        funding_price_rate[::8],
        restored.close[::8] * 0.0001,
    )

    information_available = restored.resolved_array("information_available")
    assert np.all(restored.available_at <= restored.timestamps[:, None])
    assert information_available.all()
    assert not restored.feature_available[0, :, 0].any()
    assert restored.feature_available[1:, :, 0].all()
    assert not restored.feature_available[:24, :, 1].any()
    assert restored.feature_available[24:, :, 1].all()
    assert restored.feature_available[:, :, 2].all()
    assert restored.feature_available[:, :, 3].all()

    assert (restored.resolved_array("fee_rate") == 0.0005).all()
    assert (restored.resolved_array("spread_rate") == 0.0002).all()
    assert (restored.resolved_array("max_participation_rate") == 0.05).all()
    assert restored.resolved_array("borrow_available").all()
    assert (restored.resolved_array("borrow_rate") == 0.0).all()

    for filename in ("manifest.json", "arrays.npz"):
        assert (first_root / filename).read_bytes() == (
            second_root / filename
        ).read_bytes()


def test_full_artificial_dataset_refuses_existing_root_without_changes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "existing"
    root.mkdir()
    marker = root / "keep.bin"
    marker.write_bytes(b"preserve existing destination")
    before = {path.name: path.read_bytes() for path in root.iterdir()}

    with pytest.raises(FileExistsError, match="fresh"):
        _write_full_artificial_dataset(root)

    after = {path.name: path.read_bytes() for path in root.iterdir()}
    assert after == before
