"""Forecast targets of the signal diagnosis (Issue #1146, card SD1).

All arrays are ``(time, symbol)`` matrices of bar-close prices or returns. A missing
label is ``NaN`` and stays ``NaN``; nothing is imputed.

Two different "market removal" targets live here on purpose, because they answer
different questions and the second one is **not** the first:

* :func:`cross_sectional_demean` subtracts the same-time mean over symbols. It removes
  the common *level* only. It estimates no coefficient, so a market move that loads
  unequally on the symbols leaves ``(beta_i - mean(beta)) * market`` behind.
* :func:`causal_beta_residual` subtracts ``beta_i * market`` with ``beta_i`` estimated
  by least squares from past, already matured rows only. It removes heterogeneous
  beta as far as the estimate is good, and it is available at the decision time.
"""

from __future__ import annotations

import numpy as np

from trade_rl.evaluation.signal_measurement.ic import SignalMeasurementError


def _matrix(values: object, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.dtype.kind not in "fiu":
        raise SignalMeasurementError(f"{name} must be a 2-D numeric array")
    return array.astype(np.float64, copy=False)


def _horizon(value: object) -> int:
    if type(value) is not int or value < 1:
        raise SignalMeasurementError("horizon_bars must be a positive integer")
    return value


def forward_log_returns(close: object, horizon_bars: int) -> np.ndarray:
    """``ln(close[t+h]) - ln(close[t])``; last ``h`` rows are not mature."""

    prices = _matrix(close, "close")
    horizon = _horizon(horizon_bars)
    if not (np.isfinite(prices) & (prices > 0.0)).all():
        raise SignalMeasurementError("close must be finite and positive")
    out = np.full(prices.shape, np.nan, dtype=np.float64)
    if prices.shape[0] > horizon:
        # The intermediate price ratio can overflow even when both logs are finite.
        out[:-horizon] = np.log(prices[horizon:]) - np.log(prices[:-horizon])
    return out


def market_mean(returns: object) -> np.ndarray:
    """Per-row mean over the symbols that have a finite return (``NaN`` if none)."""

    values = _matrix(returns, "returns")
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    total = np.where(finite, values, 0.0).sum(axis=1)
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    np.divide(total, count, out=out, where=count > 0)
    return out


def cross_sectional_demean(returns: object) -> np.ndarray:
    """``r(i,t) - mean_j r(j,t)`` over the symbols with a matured label at ``t``.

    A row with fewer than two matured labels has no cross-section, so every entry of
    that row is ``NaN`` (a lone symbol would otherwise demean to exactly zero).
    """

    values = _matrix(returns, "returns")
    mean = market_mean(values)
    out = values - mean[:, None]
    too_few = np.isfinite(values).sum(axis=1) < 2
    out[too_few, :] = np.nan
    return out


def causal_beta_residual(
    returns: object, horizon_bars: int, *, min_observations: int
) -> np.ndarray:
    """``r(i,t) - beta_i(t) * m(t)`` with ``beta_i(t)`` from matured past rows only.

    ``m`` is :func:`market_mean`. ``beta_i(t)`` is the least-squares slope (with an
    intercept) of ``r(i, s)`` on ``m(s)`` over the rows ``s <= t - horizon_bars``,
    the rows whose label closed at or before the decision time ``t``, where both are
    finite. With fewer than ``min_observations`` such rows, or a degenerate market
    variance, there is no estimate and the residual is ``NaN`` (reported missing,
    never filled). The current row's label never enters its own coefficient.
    """

    values = _matrix(returns, "returns")
    horizon = _horizon(horizon_bars)
    if type(min_observations) is not int or min_observations < 2:
        raise SignalMeasurementError("min_observations must be an integer >= 2")
    market = market_mean(values)
    rows, symbols = values.shape
    out = np.full(values.shape, np.nan, dtype=np.float64)
    market_ok = np.isfinite(market)
    for column in range(symbols):
        y = values[:, column]
        valid = np.isfinite(y) & market_ok
        x = np.where(valid, market, 0.0)
        z = np.where(valid, y, 0.0)
        count = np.cumsum(valid.astype(np.float64))
        sum_x = np.cumsum(x)
        sum_y = np.cumsum(z)
        sum_xx = np.cumsum(x * x)
        sum_xy = np.cumsum(x * z)
        for t in range(horizon, rows):
            s = t - horizon
            n = count[s]
            if n < min_observations or not valid[t]:
                continue
            variance = n * sum_xx[s] - sum_x[s] ** 2
            if variance <= 1e-18 * max(1.0, n * sum_xx[s]):
                continue
            beta = (n * sum_xy[s] - sum_x[s] * sum_y[s]) / variance
            out[t, column] = y[t] - beta * market[t]
    return out


__all__ = [
    "causal_beta_residual",
    "cross_sectional_demean",
    "forward_log_returns",
    "market_mean",
]
