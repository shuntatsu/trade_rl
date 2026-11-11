"""Feature non-stationarity profile (Issue #1146, card SD1; plan layer D0).

For each feature column and each period, the values of that period are standardised
by the mean and standard deviation of **all rows of strictly earlier periods** (the
expanding train window, the TD8 rule) and the distribution of those z-scores is
recorded. A feature that has drifted, such as a funding rate whose level moved, shows
up as a mean z far from zero or a large share of ``|z| > clip``.

The first period has no earlier rows and is not profiled. The function reports
numbers only; it picks no threshold and flags nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.evaluation.signal_measurement.ic import SignalMeasurementError


@dataclass(frozen=True, slots=True)
class PeriodProfile:
    """z-score distribution of one feature in one period against its past."""

    feature: str
    period: int
    n_train: int
    n_eval: int
    train_mean: float
    train_std: float
    z_mean: float
    z_std: float
    z_p05: float
    z_p50: float
    z_p95: float
    share_abs_z_above_clip: float


@dataclass(frozen=True, slots=True)
class SkippedProfile:
    """A (feature, period) that could not be profiled, with the reason."""

    feature: str
    period: int
    reason: str


@dataclass(frozen=True, slots=True)
class NonstationarityProfile:
    profiles: tuple[PeriodProfile, ...]
    skipped: tuple[SkippedProfile, ...]


def nonstationarity_profile(
    values: object,
    period_ids: object,
    feature_names: tuple[str, ...],
    *,
    min_train_rows: int,
    clip_abs: float = 3.0,
) -> NonstationarityProfile:
    """Profile every feature column of ``values`` (rows x features) per period.

    ``period_ids`` labels each row (for example the calendar year). Non-finite values
    are ignored. A (feature, period) needs at least ``min_train_rows`` finite earlier
    values, at least one finite value in the period and a non-zero train standard
    deviation; otherwise it is listed in ``skipped`` rather than guessed.
    """

    matrix = np.asarray(values)
    periods = np.asarray(period_ids)
    if matrix.ndim != 2 or matrix.dtype.kind not in "fiu":
        raise SignalMeasurementError("values must be a 2-D numeric array")
    if periods.ndim != 1 or periods.dtype.kind not in "iu":
        raise SignalMeasurementError("period_ids must be a 1-D integer array")
    if periods.shape[0] != matrix.shape[0]:
        raise SignalMeasurementError("period_ids must align with the rows")
    if len(feature_names) != matrix.shape[1] or len(set(feature_names)) != len(
        feature_names
    ):
        raise SignalMeasurementError("feature_names must be unique, one per column")
    if type(min_train_rows) is not int or min_train_rows < 2:
        raise SignalMeasurementError("min_train_rows must be an integer >= 2")
    if not np.isfinite(clip_abs) or clip_abs <= 0.0:
        raise SignalMeasurementError("clip_abs must be positive and finite")
    data = matrix.astype(np.float64, copy=False)

    profiles: list[PeriodProfile] = []
    skipped: list[SkippedProfile] = []
    ordered = sorted(int(p) for p in np.unique(periods))
    for name_index, name in enumerate(feature_names):
        column = data[:, name_index]
        for period in ordered[1:]:
            train = column[periods < period]
            train = train[np.isfinite(train)]
            current = column[periods == period]
            current = current[np.isfinite(current)]
            if train.shape[0] < min_train_rows:
                skipped.append(SkippedProfile(name, period, "too few train rows"))
                continue
            if current.shape[0] == 0:
                skipped.append(SkippedProfile(name, period, "no finite values"))
                continue
            mean = float(train.mean())
            std = float(train.std())
            if std == 0.0 or not np.isfinite(std):
                skipped.append(SkippedProfile(name, period, "zero train deviation"))
                continue
            z = (current - mean) / std
            p05, p50, p95 = (float(q) for q in np.quantile(z, [0.05, 0.5, 0.95]))
            profiles.append(
                PeriodProfile(
                    feature=name,
                    period=period,
                    n_train=int(train.shape[0]),
                    n_eval=int(current.shape[0]),
                    train_mean=mean,
                    train_std=std,
                    z_mean=float(z.mean()),
                    z_std=float(z.std()),
                    z_p05=p05,
                    z_p50=p50,
                    z_p95=p95,
                    share_abs_z_above_clip=float(np.mean(np.abs(z) > clip_abs)),
                )
            )
    return NonstationarityProfile(tuple(profiles), tuple(skipped))


__all__ = [
    "NonstationarityProfile",
    "PeriodProfile",
    "SkippedProfile",
    "nonstationarity_profile",
]
