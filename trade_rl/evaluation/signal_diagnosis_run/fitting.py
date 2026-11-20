"""Causal prequential fits of the decision cells (SD2; layer D3).

One decision cell is a feature family fitted by one model at the primary horizon and
transform. Vintages follow the shared ``PrequentialSchedule``: the vintage cut off at
``c`` trains on rows whose label closed strictly before ``c`` (the rule of the shared
``build_causal_forecast_training_set``, reproduced here with array operations because
that builder loops over every bar in Python) and serves the decision rows from ``c``
until the next cutoff. The Ridge penalty is chosen from the frozen grid by a
forward-chaining, label-purged inner CV on the training window only; the final Ridge is
the shared ``fit_ridge_forecast_from_training_set`` and the LightGBM is the shared
``fit_lightgbm_forecast_from_training_set``. Features enter standardised with the
train-window mean and standard deviation of each (symbol, feature) column and clipped
(the protocol's primary transform). The drift baseline is the expanding per-symbol mean
of matured labels at each refit instant.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from trade_rl.evaluation.signal_diagnosis.protocol import (
    FeatureFamily,
    SignalDiagnosisProtocol,
    SignalDiagnosisProtocolError,
)
from trade_rl.evaluation.signal_diagnosis_run.panel import (
    NS_PER_DAY,
    NS_PER_HOUR,
    SignalPanel,
    parse_instant_ns,
)
from trade_rl.evaluation.signal_measurement.targets import (
    cross_sectional_demean,
    forward_log_returns,
)
from trade_rl.strategies.forecasts.lightgbm import (
    fit_lightgbm_forecast_from_training_set,
)
from trade_rl.strategies.forecasts.prequential import PrequentialSchedule
from trade_rl.strategies.forecasts.ridge import fit_ridge_forecast_from_training_set
from trade_rl.strategies.forecasts.supervised import CausalForecastTrainingSet

_SCALE_FLOOR = 1e-12
RIDGE_ID = "ridge_inner_cv"
LIGHTGBM_ID = "lightgbm_fixed"


@dataclass(frozen=True, slots=True)
class RunPlan:
    """Compute-shaping choices; the default is read from the frozen protocol.

    A synthetic run may widen ``refit_interval_days`` or skip the reported-only layers
    (D0 profile, D1 beta, D2) to keep a test affordable; a real run must use the
    protocol value and run every layer (the runner enforces both).
    """

    refit_interval_days: int
    reported_layers: bool = True

    @staticmethod
    def from_protocol(protocol: SignalDiagnosisProtocol) -> RunPlan:
        return RunPlan(refit_interval_days=protocol.prequential.refit_interval_days)


@dataclass(frozen=True, slots=True)
class Labels:
    """Forward log returns at one horizon; ``valid`` marks usable matured labels."""

    horizon: int
    raw: np.ndarray
    demeaned: np.ndarray
    valid: np.ndarray


def build_labels(panel: SignalPanel, horizon: int) -> Labels:
    """Matured forward log returns; a label whose either bar was not yet available
    at its own timestamp is dropped, never imputed."""

    raw = forward_log_returns(panel.close, horizon)
    available = panel.available_at_ns <= panel.timestamps_ns[:, None]
    valid = np.zeros(raw.shape, dtype=np.bool_)
    valid[:-horizon] = available[:-horizon] & available[horizon:]
    valid &= np.isfinite(raw)
    raw = np.where(valid, raw, np.nan)
    return Labels(horizon, raw, cross_sectional_demean(raw), valid)


@dataclass(frozen=True, slots=True)
class FamilyData:
    family_id: str
    X: np.ndarray
    ok: np.ndarray


def family_data(panel: SignalPanel, family: FeatureFamily) -> FamilyData:
    """The family's columns and the rows where every one of them is usable."""

    index = [panel.feature_names.index(name) for name in family.feature_names]
    columns = panel.features[:, :, index]
    ok = (
        panel.feature_available[:, :, index].all(axis=2)
        & np.isfinite(columns).all(axis=2)
        & (panel.available_at_ns <= panel.timestamps_ns[:, None])
    )
    return FamilyData(family.family_id, columns, ok)


@dataclass(frozen=True, slots=True)
class Vintage:
    """One scheduled refit: its cutoff and the row range of decisions it serves."""

    cutoff_ns: int
    cut_idx: int
    first_row: int
    stop_row: int


def build_vintages(
    panel: SignalPanel, protocol: SignalDiagnosisProtocol, plan: RunPlan
) -> tuple[Vintage, ...]:
    schedule = PrequentialSchedule(
        first_fit_time=np.datetime64(
            parse_instant_ns(protocol.prequential.first_fit_time), "ns"
        ),
        refit_interval_days=plan.refit_interval_days,
        activation_delay_hours=protocol.prequential.activation_delay_hours,
    )
    stamps = panel.timestamps_ns
    cutoffs = [
        int(c.astype(np.int64))
        for c in schedule.cutoffs(np.datetime64(int(stamps[-1]), "ns"))
    ]
    starts = [int(np.searchsorted(stamps, c, side="left")) for c in cutoffs]
    stops = [*starts[1:], int(stamps.size)]
    return tuple(
        Vintage(c, a, a, b)
        for c, a, b in zip(cutoffs, starts, stops, strict=True)
        if a < b
    )


def drift_forecast(
    panel: SignalPanel, labels: Labels, vintages: tuple[Vintage, ...]
) -> np.ndarray:
    """Per symbol, the mean of labels that matured at or before each refit instant."""

    stamps = panel.timestamps_ns
    h = labels.horizon
    values = np.where(labels.valid, labels.raw, 0.0)
    total = np.cumsum(values, axis=0)
    count = np.cumsum(labels.valid, axis=0)
    out = np.full(labels.raw.shape, np.nan)
    for vintage in vintages:
        last = int(np.searchsorted(stamps, vintage.cutoff_ns, side="right")) - 1
        limit = last - h  # the row whose label closes at or before the cutoff
        if limit < 0:
            continue
        seen = count[limit]
        mean = np.divide(
            total[limit], seen, out=np.full(seen.shape, np.nan), where=seen > 0
        )
        out[vintage.first_row : vintage.stop_row] = mean
    return out


@dataclass(frozen=True, slots=True)
class Training:
    t_idx: np.ndarray
    s_idx: np.ndarray
    X: np.ndarray
    y: np.ndarray
    w: np.ndarray


def build_training(
    data: FamilyData, labels: Labels, n_symbols: int, cut_idx: int
) -> Training | None:
    """Rows whose label closes strictly before the cutoff, symbol-major, balanced weights."""

    limit = max(cut_idx - labels.horizon, 0)
    mask = data.ok[:limit] & labels.valid[:limit]
    t_parts = [np.flatnonzero(mask[:, s]) for s in range(n_symbols)]
    active = [s for s in range(n_symbols) if t_parts[s].size]
    total = sum(int(t_parts[s].size) for s in active)
    if total < 2:
        return None
    t_idx = np.concatenate([t_parts[s] for s in active])
    s_idx = np.concatenate([np.full(t_parts[s].size, s) for s in active])
    weights = np.concatenate(
        [
            np.full(t_parts[s].size, (total / len(active)) / t_parts[s].size)
            for s in active
        ]
    )
    X = data.X[t_idx, s_idx].astype(np.float64)
    return Training(t_idx, s_idx, X, labels.raw[t_idx, s_idx], weights)


@dataclass(frozen=True, slots=True)
class Standardiser:
    mean: np.ndarray
    scale: np.ndarray
    clip: float

    def apply(self, X: np.ndarray, s_idx: np.ndarray) -> np.ndarray:
        z = (X - self.mean[s_idx]) / self.scale[s_idx]
        return np.clip(z, -self.clip, self.clip)


def fit_standardiser(
    X: np.ndarray, s_idx: np.ndarray, n_symbols: int, clip: float
) -> Standardiser:
    """Per (symbol, feature) train-window mean and population standard deviation."""

    mean = np.zeros((n_symbols, X.shape[1]))
    scale = np.ones((n_symbols, X.shape[1]))
    for s in range(n_symbols):
        rows = X[s_idx == s]
        if rows.shape[0]:
            mean[s] = rows.mean(axis=0)
            std = rows.std(axis=0)
            scale[s] = np.where(std > _SCALE_FLOOR, std, 1.0)
    return Standardiser(mean, scale, clip)


# ------------------------------------------------------------- Ridge inner CV


@dataclass(frozen=True, slots=True)
class _Stats:
    w: float
    sx: np.ndarray
    sxx: np.ndarray
    sy: float
    sxy: np.ndarray

    def __add__(self, other: _Stats) -> _Stats:
        return _Stats(
            self.w + other.w,
            self.sx + other.sx,
            self.sxx + other.sxx,
            self.sy + other.sy,
            self.sxy + other.sxy,
        )

    def __sub__(self, other: _Stats) -> _Stats:
        return _Stats(
            self.w - other.w,
            self.sx - other.sx,
            self.sxx - other.sxx,
            self.sy - other.sy,
            self.sxy - other.sxy,
        )


def _stats(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> _Stats:
    weighted = X * w[:, None]
    return _Stats(
        float(w.sum()),
        weighted.sum(axis=0),
        weighted.T @ X,
        float(w @ y),
        weighted.T @ y,
    )


def ridge_from_stats(
    stats: _Stats, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(mean, scale, coefficients, intercept)`` of the shared Ridge, from moments.

    Algebraically the shared ``fit_ridge_forecast_from_training_set`` (weighted
    standardisation with the scale floor, intercept the weighted label mean, penalty
    ``alpha`` on the standardised Gram matrix); a test pins the equality.
    """

    mean = stats.sx / stats.w
    variance = np.maximum(np.diag(stats.sxx) / stats.w - mean**2, 0.0)
    raw_scale = np.sqrt(variance)
    scale = np.where(raw_scale > _SCALE_FLOOR, raw_scale, 1.0)
    centred = stats.sxx - stats.w * np.outer(mean, mean)
    gram = centred / np.outer(scale, scale)
    rhs = (stats.sxy - mean * stats.sy) / scale
    coefficients = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)
    return mean, scale, coefficients, stats.sy / stats.w


def select_ridge_penalty(
    Xz: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    t_idx: np.ndarray,
    *,
    horizon: int,
    grid: tuple[float, ...],
    folds: int,
) -> tuple[float, tuple[float, ...]]:
    """Forward-chaining CV over time; the penalty with the least mean validation MSE.

    The time-ordered rows are cut into ``folds + 1`` blocks at row-count quantiles. Fold
    ``j`` validates on block ``j`` and trains on rows whose label closed before the block
    started: decision times earlier than the block start minus the label horizon (the
    purge). Ties go to the larger penalty. Uses the training window only.
    """

    order = np.argsort(t_idx, kind="stable")
    times = t_idx[order]
    n = times.size
    bounds = [int(times[(i * n) // (folds + 1)]) for i in range(1, folds + 1)]
    edges = [*bounds, int(times[-1]) + 1]
    scores = np.zeros(len(grid))
    used = 0
    for j in range(folds):
        train = times < bounds[j] - horizon
        valid = (times >= edges[j]) & (times < edges[j + 1])
        if int(train.sum()) < 2 or not valid.any():
            raise SignalDiagnosisProtocolError(
                "inner CV fold has no training or validation rows"
            )
        rows_t, rows_v = order[train], order[valid]
        stats = _stats(Xz[rows_t], y[rows_t], w[rows_t])
        for index, alpha in enumerate(grid):
            mean, scale, coefficients, intercept = ridge_from_stats(stats, alpha)
            prediction = intercept + ((Xz[rows_v] - mean) / scale) @ coefficients
            scores[index] += float(
                w[rows_v] @ (y[rows_v] - prediction) ** 2 / w[rows_v].sum()
            )
        used += 1
    scores /= used
    best = 0
    for index in range(len(grid)):
        if scores[index] <= scores[best]:
            best = index
    return grid[best], tuple(float(s) for s in scores)


# -------------------------------------------------------------- family driver


@dataclass(frozen=True, slots=True)
class VintageRecord:
    cutoff_ns: int
    n_train: int
    n_decided: int
    ridge_penalty: float | None
    ridge_cv_scores: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class CellForecast:
    family_id: str
    model_id: str
    forecast: np.ndarray
    vintages: tuple[VintageRecord, ...]


def _causal_set(
    train: Training,
    Xz: np.ndarray,
    ts: np.ndarray,
    horizon: int,
    cutoff_ns: int,
    cut_idx: int,
) -> CausalForecastTrainingSet:
    return CausalForecastTrainingSet(
        feature_indices=tuple(range(Xz.shape[1])),
        features=Xz,
        labels=train.y,
        label_end_times=ts[train.t_idx + horizon].astype("datetime64[ns]"),
        sample_weights=train.w,
        fit_cutoff=np.datetime64(cutoff_ns, "ns"),
        horizon_hours=horizon,
    )


def forecast_family(
    panel: SignalPanel,
    protocol: SignalDiagnosisProtocol,
    family: FeatureFamily,
    labels: Labels,
    vintages: tuple[Vintage, ...],
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, CellForecast]:
    """Both models of one family over every vintage (primary horizon, primary transform)."""

    models = protocol.models
    transform = next(
        t
        for t in protocol.transforms
        if t.transform_id == protocol.decision_rules.primary_transform_id
    )
    data = family_data(panel, family)
    n_symbols = len(panel.symbols)
    ts = panel.timestamps_ns
    start_ns = parse_instant_ns(protocol.data.evaluation_start)
    forecasts = {m: np.full(labels.raw.shape, np.nan) for m in models.model_ids}
    records: dict[str, list[VintageRecord]] = {m: [] for m in models.model_ids}
    for vintage in vintages:
        train = build_training(data, labels, n_symbols, vintage.cut_idx)
        rows = slice(vintage.first_row, vintage.stop_row)
        usable = data.ok[rows] & labels.valid[rows]
        usable &= (ts[rows] > start_ns)[:, None]
        t_local, s_idx = np.nonzero(usable)
        if train is None or t_local.size == 0:
            for m in models.model_ids:
                records[m].append(VintageRecord(vintage.cutoff_ns, 0, 0, None, None))
            continue
        standardiser = fit_standardiser(
            train.X, train.s_idx, n_symbols, transform.clip_abs
        )
        Xz = standardiser.apply(train.X, train.s_idx)
        t_idx = t_local + vintage.first_row
        Xd = standardiser.apply(data.X[t_idx, s_idx].astype(np.float64), s_idx)
        causal = _causal_set(
            train, Xz, ts, labels.horizon, vintage.cutoff_ns, vintage.cut_idx
        )
        for model_id in models.model_ids:
            penalty: float | None = None
            cv: tuple[float, ...] | None = None
            if model_id == RIDGE_ID:
                penalty, cv = select_ridge_penalty(
                    Xz,
                    train.y,
                    train.w,
                    train.t_idx,
                    horizon=labels.horizon,
                    grid=models.ridge.penalty_grid,
                    folds=models.ridge.inner_cv_folds,
                )
                ridge = fit_ridge_forecast_from_training_set(causal, alpha=penalty)
                prediction = (
                    ridge.intercept
                    + ((Xd - ridge.feature_mean) / ridge.feature_scale)
                    @ ridge.coefficients
                )
            elif model_id == LIGHTGBM_ID:
                gbm = fit_lightgbm_forecast_from_training_set(
                    causal, random_state=models.lightgbm.random_state
                )
                prediction = np.asarray(gbm.predictor.predict(Xd), dtype=np.float64)
            else:
                raise SignalDiagnosisProtocolError(f"unknown model {model_id!r}")
            forecasts[model_id][t_idx, s_idx] = prediction
            records[model_id].append(
                VintageRecord(
                    vintage.cutoff_ns, int(train.y.size), int(t_idx.size), penalty, cv
                )
            )
        if progress is not None:
            progress(f"{family.family_id}: vintage {vintage.cutoff_ns} done")
    return {
        m: CellForecast(family.family_id, m, forecasts[m], tuple(records[m]))
        for m in models.model_ids
    }


__all__ = [
    "LIGHTGBM_ID",
    "NS_PER_DAY",
    "NS_PER_HOUR",
    "RIDGE_ID",
    "CellForecast",
    "FamilyData",
    "Labels",
    "RunPlan",
    "Standardiser",
    "Training",
    "Vintage",
    "VintageRecord",
    "build_labels",
    "build_training",
    "build_vintages",
    "drift_forecast",
    "family_data",
    "fit_standardiser",
    "forecast_family",
    "ridge_from_stats",
    "select_ridge_penalty",
]
