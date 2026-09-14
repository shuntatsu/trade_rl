"""Deterministic primitives for the Issue #553 training-only flow diagnostic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_HOUR_NS = 3_600_000_000_000


@dataclass(frozen=True, slots=True)
class HourlyFlowPoint:
    """One completed UTC hour of signed aggressive taker notional."""

    hour_end_ns: int
    buy_taker_notional: float
    sell_taker_notional: float
    flow_imbalance: float


@dataclass(frozen=True, slots=True)
class FlowReturnSample:
    """One causally aligned flow predictor and next executable-hour label."""

    decision_timestamp_ns: int
    flow_imbalance: float
    execution_open: float
    label_end_open: float
    log_return: float


@dataclass(frozen=True, slots=True)
class CenteredOlsStats:
    """Deterministic scalar OLS-with-intercept summary."""

    count: int
    mean_x: float
    mean_y: float
    centered_numerator: float
    centered_denominator: float
    centered_y_denominator: float
    beta: float
    correlation: float


def _one_dimensional(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def aggregate_completed_hour_flows(
    *,
    timestamps: np.ndarray,
    prices: np.ndarray,
    quantities: np.ndarray,
    buyer_is_maker: np.ndarray,
) -> tuple[HourlyFlowPoint, ...]:
    """Aggregate exact taker-side notionals for each completed UTC hour."""

    time_array = _one_dimensional(timestamps, name="timestamps")
    price_array = _one_dimensional(prices, name="prices").astype(np.float64, copy=False)
    quantity_array = _one_dimensional(quantities, name="quantities").astype(
        np.float64,
        copy=False,
    )
    maker_array = _one_dimensional(buyer_is_maker, name="buyer_is_maker").astype(
        np.bool_,
        copy=False,
    )
    size = time_array.size
    if size <= 0:
        raise ValueError("aggTrades arrays must be non-empty")
    if any(array.size != size for array in (price_array, quantity_array, maker_array)):
        raise ValueError("aggTrades arrays must have equal length")
    if not np.issubdtype(time_array.dtype, np.datetime64):
        raise ValueError("timestamps must use a datetime64 dtype")
    timestamp_ns = time_array.astype("datetime64[ns]").astype(np.int64)
    if np.any(timestamp_ns == np.iinfo(np.int64).min):
        raise ValueError("timestamps must not contain NaT")
    if size > 1 and np.any(timestamp_ns[1:] < timestamp_ns[:-1]):
        raise ValueError("timestamps must be nondecreasing")
    if not np.isfinite(price_array).all() or np.any(price_array <= 0.0):
        raise ValueError("prices must be finite and strictly positive")
    if not np.isfinite(quantity_array).all() or np.any(quantity_array <= 0.0):
        raise ValueError("quantities must be finite and strictly positive")

    notionals = price_array * quantity_array
    if not np.isfinite(notionals).all() or np.any(notionals <= 0.0):
        raise ValueError("trade notionals must be finite and strictly positive")

    hour_keys = timestamp_ns // _HOUR_NS
    result: list[HourlyFlowPoint] = []
    start = 0
    while start < size:
        key = int(hour_keys[start])
        stop = start + 1
        while stop < size and int(hour_keys[stop]) == key:
            stop += 1
        buy = math.fsum(
            float(notionals[index])
            for index in range(start, stop)
            if not bool(maker_array[index])
        )
        sell = math.fsum(
            float(notionals[index])
            for index in range(start, stop)
            if bool(maker_array[index])
        )
        total = buy + sell
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("hourly total taker notional must be finite and positive")
        imbalance = (buy - sell) / total
        if not math.isfinite(imbalance) or abs(imbalance) > 1.0 + 1e-15:
            raise ValueError("flow imbalance must be finite and within [-1, 1]")
        result.append(
            HourlyFlowPoint(
                hour_end_ns=(key + 1) * _HOUR_NS,
                buy_taker_notional=buy,
                sell_taker_notional=sell,
                flow_imbalance=imbalance,
            )
        )
        start = stop
    return tuple(result)


def align_next_executable_hour_returns(
    *,
    flow_points: tuple[HourlyFlowPoint, ...],
    timestamps_ns: np.ndarray,
    open_prices: np.ndarray,
    active: np.ndarray,
    tradable: np.ndarray,
    cutoff_ns: int,
) -> tuple[FlowReturnSample, ...]:
    """Align completed-hour flow to exactly open[t+1] -> open[t+2]."""

    times = _one_dimensional(timestamps_ns, name="timestamps_ns").astype(
        np.int64,
        copy=False,
    )
    opens = _one_dimensional(open_prices, name="open_prices").astype(
        np.float64,
        copy=False,
    )
    active_mask = _one_dimensional(active, name="active").astype(np.bool_, copy=False)
    tradable_mask = _one_dimensional(tradable, name="tradable").astype(
        np.bool_,
        copy=False,
    )
    size = times.size
    if size <= 0 or any(array.size != size for array in (opens, active_mask, tradable_mask)):
        raise ValueError("market arrays must be non-empty and equal length")
    if size > 1 and np.any(times[1:] <= times[:-1]):
        raise ValueError("timestamps_ns must be strictly increasing")
    if not isinstance(cutoff_ns, int):
        raise TypeError("cutoff_ns must be an integer")

    result: list[FlowReturnSample] = []
    for point in flow_points:
        if not isinstance(point, HourlyFlowPoint):
            raise TypeError("flow_points must contain HourlyFlowPoint values")
        index = int(np.searchsorted(times, point.hour_end_ns, side="left"))
        if index >= size or int(times[index]) != point.hour_end_ns:
            continue
        if index + 2 >= size:
            continue
        if (
            int(times[index + 1]) - int(times[index]) != _HOUR_NS
            or int(times[index + 2]) - int(times[index + 1]) != _HOUR_NS
        ):
            continue
        # open[t+1] occurs when row t closes; open[t+2] occurs when row t+1 closes.
        if int(times[index]) >= cutoff_ns or int(times[index + 1]) >= cutoff_ns:
            continue
        if not (
            bool(active_mask[index])
            and bool(tradable_mask[index])
            and bool(active_mask[index + 1])
            and bool(tradable_mask[index + 1])
        ):
            continue
        execution_open = float(opens[index + 1])
        label_end_open = float(opens[index + 2])
        if (
            not math.isfinite(execution_open)
            or execution_open <= 0.0
            or not math.isfinite(label_end_open)
            or label_end_open <= 0.0
        ):
            continue
        log_return = math.log(label_end_open / execution_open)
        if not math.isfinite(log_return):
            raise ValueError("aligned log return must be finite")
        result.append(
            FlowReturnSample(
                decision_timestamp_ns=point.hour_end_ns,
                flow_imbalance=point.flow_imbalance,
                execution_open=execution_open,
                label_end_open=label_end_open,
                log_return=log_return,
            )
        )
    return tuple(result)


def centered_ols_stats(
    *,
    x: tuple[float, ...],
    y: tuple[float, ...],
) -> CenteredOlsStats:
    """Compute OLS slope with intercept using deterministic ``math.fsum`` reductions."""

    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x and y must have equal length of at least two")
    x_values = tuple(float(value) for value in x)
    y_values = tuple(float(value) for value in y)
    if any(not math.isfinite(value) for value in (*x_values, *y_values)):
        raise ValueError("x and y must contain only finite values")
    count = len(x_values)
    mean_x = math.fsum(x_values) / count
    mean_y = math.fsum(y_values) / count
    dx = tuple(value - mean_x for value in x_values)
    dy = tuple(value - mean_y for value in y_values)
    denominator = math.fsum(value * value for value in dx)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("predictor variance must be finite and positive")
    y_denominator = math.fsum(value * value for value in dy)
    if not math.isfinite(y_denominator) or y_denominator <= 0.0:
        raise ValueError("label variance must be finite and positive")
    numerator = math.fsum(left * right for left, right in zip(dx, dy, strict=True))
    beta = numerator / denominator
    correlation = numerator / math.sqrt(denominator * y_denominator)
    if not math.isfinite(beta) or not math.isfinite(correlation):
        raise ValueError("centered OLS outputs must be finite")
    return CenteredOlsStats(
        count=count,
        mean_x=mean_x,
        mean_y=mean_y,
        centered_numerator=numerator,
        centered_denominator=denominator,
        centered_y_denominator=y_denominator,
        beta=beta,
        correlation=correlation,
    )


__all__ = [
    "CenteredOlsStats",
    "FlowReturnSample",
    "HourlyFlowPoint",
    "aggregate_completed_hour_flows",
    "align_next_executable_hour_returns",
    "centered_ols_stats",
]
