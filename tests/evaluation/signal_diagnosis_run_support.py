"""Synthetic panels and fakes for the SD2 runner tests (no market value is read).

The panel covers the whole frozen protocol window on the hourly grid with the real
roster and the 167 protocol feature names, but every number is drawn from a seeded
generator. A *planted* signal is one column of a family that carries the standardised
idiosyncratic 24-bar forward return plus noise, so the oracle is known by construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from trade_rl.evaluation.signal_diagnosis.protocol import (
    SignalDiagnosisProtocol,
    default_signal_diagnosis_protocol,
)
from trade_rl.evaluation.signal_diagnosis_run.fitting import RunPlan
from trade_rl.evaluation.signal_diagnosis_run.panel import (
    DatasetIdentity,
    SignalPanel,
    parse_instant_ns,
    protocol_feature_names,
)
from trade_rl.evaluation.signal_diagnosis_run.runner import CodeProvenance
from trade_rl.evaluation.signal_measurement import market_mean

HOUR = 3_600_000_000_000
CLEAN = CodeProvenance("c" * 40, "d" * 40, False, "e" * 64, 1)
DIRTY = CodeProvenance("c" * 40, "d" * 40, True, "e" * 64, 1)
SYNTHETIC_IDENTITY = DatasetIdentity("1" * 64, "2" * 64, "3" * 64, "4" * 64)
# One refit every 800 days keeps a full-window synthetic run affordable; a real run must use
# the protocol's 30 days (the runner refuses any other plan outside synthetic scope).
TEST_PLAN = RunPlan(refit_interval_days=800, reported_layers=False)


def protocol() -> SignalDiagnosisProtocol:
    return default_signal_diagnosis_protocol()


def synthetic_panel(
    *,
    seed: int,
    planted_families: tuple[str, ...] = (),
    signal: float = 0.0,
    hourly_vol: float = 0.004,
    leak_copy_of_label: bool = False,
    identity: DatasetIdentity = SYNTHETIC_IDENTITY,
    end_extra_hours: int = 0,
) -> SignalPanel:
    spec = protocol()
    start = parse_instant_ns(spec.data.evaluation_start) + HOUR
    end = parse_instant_ns(spec.data.evaluation_end) + end_extra_hours * HOUR
    stamps = np.arange(start, end + HOUR, HOUR, dtype=np.int64)
    rows, symbols = stamps.size, len(spec.data.symbols)
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((rows, 1))
    returns = hourly_vol * (0.6 * common + rng.standard_normal((rows, symbols)))
    log_close = np.log(np.array([100.0, 50.0, 20.0, 5.0, 1.0])) + np.cumsum(
        returns, axis=0
    )
    close = np.exp(log_close)
    horizon = spec.horizons.primary_bars
    forward = np.full((rows, symbols), np.nan)
    forward[:-horizon] = log_close[horizon:] - log_close[:-horizon]
    demeaned = forward - market_mean(forward)[:, None]
    scale = float(np.nanstd(demeaned))
    names = protocol_feature_names(spec)
    features = np.zeros((rows, symbols, len(names)), dtype=np.float32)
    for family in spec.families:
        columns = [names.index(name) for name in family.feature_names[:3]]
        for column in columns:
            features[:, :, column] = rng.standard_normal((rows, symbols))
        if family.family_id in planted_families:
            planted = signal * np.nan_to_num(demeaned / scale)
            features[:, :, columns[0]] = planted + rng.standard_normal((rows, symbols))
    if leak_copy_of_label:
        copy = 2.0 * np.nan_to_num(forward) + 1.0
        features[:, :, names.index(spec.families[0].feature_names[3])] = copy
    return SignalPanel(
        identity=identity,
        symbols=spec.data.symbols,
        feature_names=names,
        timestamps_ns=stamps,
        close=close,
        features=features,
        feature_available=np.ones(features.shape, dtype=np.bool_),
        available_at_ns=np.repeat(stamps[:, None], symbols, axis=1),
    )


class WeightedLinearRegressor:
    """Deterministic stand-in for ``lightgbm.LGBMRegressor`` (weighted least squares).

    Used where LightGBM is not installed (the Core CI job); the real regressor is
    exercised by the tests that import it.
    """

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.coef: np.ndarray | None = None

    def fit(
        self, features: np.ndarray, labels: np.ndarray, *, sample_weight: np.ndarray
    ) -> WeightedLinearRegressor:
        design = np.column_stack((np.ones(features.shape[0]), features))
        root = np.sqrt(sample_weight)
        self.coef = np.linalg.lstsq(design * root[:, None], labels * root, rcond=None)[
            0
        ]
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert self.coef is not None
        return np.column_stack((np.ones(features.shape[0]), features)) @ self.coef
