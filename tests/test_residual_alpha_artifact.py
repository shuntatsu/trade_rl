from pathlib import Path

import numpy as np
import pytest

from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.trading.residual_alpha import FrozenResidualAlpha


def _feature_set(n_bars: int = 180, names: list[str] | None = None) -> FeatureSet:
    names = names or ["signal", "noise"]
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    signal = np.sin(np.arange(n_bars) / 10.0)
    features = np.zeros((n_bars, 3, 2), dtype=np.float32)
    features[:, 0, 0] = signal
    features[:, 1, 0] = -signal
    features[:, 2, 0] = signal * 0.5
    close = np.ones((n_bars, 3), dtype=np.float64)
    close[1:, 0] = np.cumprod(1.0 + 0.002 * signal[:-1])
    close[1:, 1] = np.cumprod(1.0 - 0.002 * signal[:-1])
    close[1:, 2] = np.cumprod(1.0 + 0.001 * signal[:-1])
    return FeatureSet(
        symbols=["A", "B", "C"],
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        close=close,
        open_next=close.copy(),
        funding_rate=np.zeros_like(close),
        feature_names=names,
        global_feature_names=["global"],
    )


def test_fit_cutoff_excludes_unrealized_horizon() -> None:
    fs = _feature_set()
    artifact = FrozenResidualAlpha.fit(
        fs,
        horizon=4,
        model="ridge",
        gate_result={"passed": True},
    )

    assert artifact.fit_cutoff_index == fs.n_bars - 4
    assert artifact.feature_names == tuple(fs.feature_names)


def test_prediction_is_market_neutral_and_gross_bounded() -> None:
    fs = _feature_set()
    artifact = FrozenResidualAlpha.fit(
        fs,
        horizon=4,
        model="ridge",
        gate_result={"passed": True},
    )

    weights = artifact.predict_at(fs, 150)

    assert weights.sum() == pytest.approx(0.0, abs=1e-12)
    assert np.abs(weights).sum() <= 1.0 + 1e-12


def test_disabled_gate_returns_zero_weights() -> None:
    fs = _feature_set()
    artifact = FrozenResidualAlpha.fit(
        fs,
        horizon=4,
        model="ridge",
        gate_result={"passed": False},
    )

    np.testing.assert_array_equal(artifact.predict_at(fs, 150), np.zeros(3))


def test_save_load_does_not_refit_and_rejects_feature_order(tmp_path: Path) -> None:
    fs = _feature_set()
    artifact = FrozenResidualAlpha.fit(
        fs,
        horizon=4,
        model="ridge",
        gate_result={"passed": True},
    )
    path = tmp_path / "alpha.json"
    artifact.save(path)
    loaded = FrozenResidualAlpha.load(path)

    np.testing.assert_allclose(loaded.predict_at(fs, 150), artifact.predict_at(fs, 150))
    mismatched = _feature_set(names=["noise", "signal"])
    with pytest.raises(ValueError, match="feature order"):
        loaded.predict_at(mismatched, 150)
