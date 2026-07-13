"""Validated in-memory market dataset used by research and simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.domain.common import require_sha256, require_unique_non_empty


def _readonly_array(
    value: np.ndarray,
    *,
    dtype: np.dtype[np.generic] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy(order="C")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class MarketDataset:
    """Shape-checked market arrays bound to one content identity."""

    dataset_id: str
    symbols: tuple[str, ...]
    timestamps: np.ndarray
    features: np.ndarray
    global_features: np.ndarray
    close: np.ndarray
    funding_rate: np.ndarray
    feature_names: tuple[str, ...]
    global_feature_names: tuple[str, ...]
    periods_per_year: int

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        symbols = require_unique_non_empty(self.symbols, field="symbols")
        feature_names = require_unique_non_empty(
            self.feature_names,
            field="feature_names",
        )
        global_names = require_unique_non_empty(
            self.global_feature_names,
            field="global_feature_names",
        )
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")

        timestamps = _readonly_array(self.timestamps)
        features = _readonly_array(self.features, dtype=np.dtype(np.float32))
        global_features = _readonly_array(
            self.global_features,
            dtype=np.dtype(np.float32),
        )
        close = _readonly_array(self.close, dtype=np.dtype(np.float64))
        funding = _readonly_array(self.funding_rate, dtype=np.dtype(np.float64))

        if timestamps.ndim != 1:
            raise ValueError("timestamps must be one-dimensional")
        if not np.issubdtype(timestamps.dtype, np.datetime64):
            raise ValueError("timestamps must use a datetime64 dtype")
        n_bars = timestamps.shape[0]
        n_symbols = len(symbols)
        if n_bars < 3:
            raise ValueError("market dataset requires at least three bars")
        if np.any(np.diff(timestamps.astype("datetime64[ns]").astype(np.int64)) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        if features.shape != (n_bars, n_symbols, len(feature_names)):
            raise ValueError("features shape does not match bars, symbols, and names")
        if global_features.shape != (n_bars, len(global_names)):
            raise ValueError("global_features shape does not match bars and names")
        if close.shape != (n_bars, n_symbols):
            raise ValueError("close shape does not match bars and symbols")
        if funding.shape != (n_bars, n_symbols):
            raise ValueError("funding_rate shape does not match bars and symbols")
        for field, array in (
            ("features", features),
            ("global_features", global_features),
            ("close", close),
            ("funding_rate", funding),
        ):
            if not np.isfinite(array).all():
                raise ValueError(f"{field} must contain only finite values")
        if np.any(close <= 0.0):
            raise ValueError("close prices must be strictly positive")

        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "global_feature_names", global_names)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "global_features", global_features)
        object.__setattr__(self, "close", close)
        object.__setattr__(self, "funding_rate", funding)

    @property
    def n_bars(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)
