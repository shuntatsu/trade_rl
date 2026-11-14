"""Leakage screens of the signal diagnosis (SD2; validity flag ``leakage_detected``).

Two independent screens, both on data the run is allowed to see:

* the maintained feature calculation is rebuilt from seeded **synthetic** bars after
  perturbing every input after a cutoff (the SD1 ``future_perturbation_probe``); a
  feature row that moves is a future dependence. This probes the code, not the Dataset,
  and the 4h/1d aggregation and alignment are not probed;
* every Dataset feature is tested against the matured label for an affine or monotone
  copy (the crudest leak), pooled over the rows the diagnosis uses.

Neither screen proves causality when it finds nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from trade_rl.data.features.core import calculate_feature_events
from trade_rl.data.features.cross_asset import (
    CROSS_ASSET_FEATURE_KINDS,
    calculate_cross_asset_feature_events,
)
from trade_rl.evaluation.signal_diagnosis.protocol import SignalDiagnosisProtocol
from trade_rl.evaluation.signal_diagnosis_run.fitting import Labels
from trade_rl.evaluation.signal_diagnosis_run.measure import grouped_ic
from trade_rl.evaluation.signal_diagnosis_run.panel import SignalPanel
from trade_rl.evaluation.signal_measurement import future_perturbation_probe
from trade_rl.integrations.binance.dataset import binance_multitimeframe_feature_specs

BuilderProbe = Callable[[SignalDiagnosisProtocol], dict[str, Any]]
_PROBE_BARS = 320
_CUTOFFS = (120, 180, 240)
_RELATIVE_SCALE = 0.05
_COPY_TOLERANCE = 1e-9


def synthetic_bars(seed: int, rows: int = _PROBE_BARS) -> np.ndarray:
    """Seeded positive OHLCV-like bars (open, high, low, close, volume, funding)."""

    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(0.01 * rng.standard_normal(rows)))
    open_ = close * np.exp(0.002 * rng.standard_normal(rows))
    high = np.maximum(open_, close) * (1.0 + 0.003 * rng.random(rows))
    low = np.minimum(open_, close) * (1.0 - 0.003 * rng.random(rows))
    volume = 1_000.0 * (1.0 + rng.random(rows))
    funding = 1e-4 * (1.0 + rng.standard_normal(rows))
    return np.stack((open_, high, low, close, volume, funding), axis=1)


def maintained_builder_probe(protocol: SignalDiagnosisProtocol) -> dict[str, Any]:
    """Future-perturbation probe of every maintained feature calculation."""

    specs = binance_multitimeframe_feature_specs(
        base_timeframe=protocol.data.base_interval,
        feature_timeframes=protocol.data.feature_timeframes,
    )
    seed = protocol.bootstrap.seed
    inputs = synthetic_bars(seed)
    rows = inputs.shape[0]
    always = np.ones(rows, dtype=np.bool_)
    leaking: list[str] = []
    cross_inputs = np.random.default_rng(seed + 1).standard_normal(
        (rows, len(protocol.data.symbols))
    )
    symbols = protocol.data.symbols
    reference = symbols[0]
    for spec in specs:
        if spec.kind in CROSS_ASSET_FEATURE_KINDS:
            panel_inputs = cross_inputs

            def builder(values: np.ndarray, spec: Any = spec) -> np.ndarray:
                events = calculate_cross_asset_feature_events(
                    spec,
                    aligned_returns=values,
                    return_available=np.ones(values.shape, dtype=np.bool_),
                    return_age_hours=np.zeros(values.shape),
                    symbols=symbols,
                    reference_symbol=reference,
                )
                return np.where(events.valid, events.values, np.nan)

        else:
            panel_inputs = inputs

            def builder(values: np.ndarray, spec: Any = spec) -> np.ndarray:
                out, valid, _ = calculate_feature_events(
                    spec,
                    open_price=values[:, 0],
                    high=values[:, 1],
                    low=values[:, 2],
                    close=values[:, 3],
                    volume=values[:, 4],
                    funding_rate=values[:, 5],
                    funding_available=always,
                    row_present=always,
                    active=always,
                )
                return np.where(valid, out, np.nan)

        result = future_perturbation_probe(
            builder,
            panel_inputs,
            cutoffs=_CUTOFFS,
            seed=seed,
            relative_scale=_RELATIVE_SCALE,
        )
        if result.leaks:
            leaking.append(spec.name)
    return {
        "screen": "future_perturbation_probe_on_synthetic_bars",
        "seed": seed,
        "features_probed": len(specs),
        "cutoffs": list(_CUTOFFS),
        "leaks": bool(leaking),
        "leaking_features": leaking,
        "limits": "code probe on synthetic bars; 4h/1d aggregation and alignment "
        "are not probed; no change found is not proof of causality",
    }


def label_copy_screen(
    panel: SignalPanel, labels_by_horizon: dict[int, Labels]
) -> dict[str, Any]:
    """Features that are an affine or monotone copy of a matured label (pooled rows)."""

    flagged: list[dict[str, Any]] = []
    available_now = panel.available_at_ns <= panel.timestamps_ns[:, None]
    for index, name in enumerate(panel.feature_names):
        x = panel.features[:, :, index].astype(np.float64)
        ok = panel.feature_available[:, :, index] & np.isfinite(x) & available_now
        for horizon, labels in labels_by_horizon.items():
            keep = ok & labels.valid
            if int(keep.sum()) < 3:
                continue
            score, target = x[keep], labels.raw[keep]
            one_day = np.zeros(score.size, dtype=np.int64)
            for method in ("linear", "rank"):
                _, values = grouped_ic(one_day, score, target, method=method)
                if values.size and abs(float(values[0])) >= 1.0 - _COPY_TOLERANCE:
                    flagged.append(
                        {"feature": name, "horizon_bars": horizon, "method": method}
                    )
    return {
        "screen": "affine_or_monotone_label_copy",
        "tolerance": _COPY_TOLERANCE,
        "leaks": bool(flagged),
        "flagged": flagged,
    }


__all__ = [
    "BuilderProbe",
    "label_copy_screen",
    "maintained_builder_probe",
    "synthetic_bars",
]
