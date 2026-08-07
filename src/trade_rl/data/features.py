"""Causal technical-indicator feature calculations shared by all clocks."""

from __future__ import annotations

import math

import numpy as np

from trade_rl.data.contracts import FeatureKind, FeatureSpec, NormalizationMode

_EPSILON = 1e-12
_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIGNAL = 9
_STOCHASTIC_SMOOTH = 3
_ICHIMOKU_TENKAN = 9
_ICHIMOKU_KIJUN = 26
_ICHIMOKU_SENKOU_B = 52


def _contiguous_segments(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    padded = np.concatenate((np.array([False]), mask, np.array([False])))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return tuple(
        (int(edges[index]), int(edges[index + 1])) for index in range(0, len(edges), 2)
    )


def _window_is_valid(mask: np.ndarray, start: int, stop: int) -> bool:
    return start >= 0 and bool(np.all(mask[start:stop]))


def _ema(values: np.ndarray, start: int, stop: int, period: int) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    alpha = 2.0 / (period + 1.0)
    result[start] = values[start]
    for index in range(start + 1, stop):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def _rolling_normalize(
    values: np.ndarray,
    valid: np.ndarray,
    source_start: np.ndarray,
    *,
    window: int,
    min_periods: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = np.zeros_like(values, dtype=np.float64)
    normalized_valid = np.zeros_like(valid, dtype=np.bool_)
    normalized_start = np.full_like(source_start, -1)
    for index in range(len(values)):
        if not valid[index]:
            continue
        start = max(0, index - window + 1)
        sample_mask = valid[start : index + 1]
        sample = values[start : index + 1][sample_mask]
        if sample.size < min_periods:
            continue
        std = float(np.std(sample))
        normalized[index] = (
            0.0 if std <= _EPSILON else (values[index] - float(np.mean(sample))) / std
        )
        normalized_valid[index] = True
        normalized_start[index] = int(
            np.min(source_start[start : index + 1][sample_mask])
        )
    return normalized, normalized_valid, normalized_start


def _true_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    result = np.zeros_like(close, dtype=np.float64)
    result[start] = high[start] - low[start]
    for index in range(start + 1, stop):
        result[index] = max(
            high[index] - low[index],
            abs(high[index] - close[index - 1]),
            abs(low[index] - close[index - 1]),
        )
    return result


def _wilder_average(
    values: np.ndarray, start: int, stop: int, period: int
) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    first = start + period - 1
    if first >= stop:
        return result
    result[first] = float(np.mean(values[start : first + 1]))
    for index in range(first + 1, stop):
        result[index] = ((period - 1) * result[index - 1] + values[index]) / period
    return result


def _rsi(
    close: np.ndarray,
    usable: np.ndarray,
    period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros_like(close, dtype=np.float64)
    valid = np.zeros_like(usable, dtype=np.bool_)
    source_start = np.full(len(close), -1, dtype=np.int64)
    for start, stop in _contiguous_segments(usable):
        if stop - start <= period:
            continue
        changes = np.zeros_like(close, dtype=np.float64)
        changes[start + 1 : stop] = np.diff(close[start:stop])
        gains = np.maximum(changes, 0.0)
        losses = np.maximum(-changes, 0.0)
        first = start + period
        avg_gain = float(np.mean(gains[start + 1 : first + 1]))
        avg_loss = float(np.mean(losses[start + 1 : first + 1]))
        for index in range(first, stop):
            if index > first:
                avg_gain = ((period - 1) * avg_gain + gains[index]) / period
                avg_loss = ((period - 1) * avg_loss + losses[index]) / period
            if avg_loss <= _EPSILON:
                rsi = 100.0 if avg_gain > _EPSILON else 50.0
            else:
                relative_strength = avg_gain / avg_loss
                rsi = 100.0 - 100.0 / (1.0 + relative_strength)
            values[index] = (rsi - 50.0) / 50.0
            valid[index] = True
            source_start[index] = start
    return values, valid, source_start


def _macd(
    close: np.ndarray,
    usable: np.ndarray,
    kind: FeatureKind,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros_like(close, dtype=np.float64)
    valid = np.zeros_like(usable, dtype=np.bool_)
    source_start = np.full(len(close), -1, dtype=np.int64)
    for start, stop in _contiguous_segments(usable):
        if stop - start < _MACD_SLOW:
            continue
        fast = _ema(close, start, stop, _MACD_FAST)
        slow = _ema(close, start, stop, _MACD_SLOW)
        line = fast - slow
        line_first = start + _MACD_SLOW - 1
        if kind is FeatureKind.MACD_LINE:
            indices = range(line_first, stop)
            selected = line
        else:
            signal = np.zeros_like(close, dtype=np.float64)
            signal[line_first] = line[line_first]
            alpha = 2.0 / (_MACD_SIGNAL + 1.0)
            for index in range(line_first + 1, stop):
                signal[index] = alpha * line[index] + (1.0 - alpha) * signal[index - 1]
            signal_first = line_first + _MACD_SIGNAL - 1
            indices = range(signal_first, stop)
            selected = signal if kind is FeatureKind.MACD_SIGNAL else line - signal
        for index in indices:
            values[index] = selected[index] / close[index]
            valid[index] = True
            source_start[index] = start
    return values, valid, source_start


def _atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    usable: np.ndarray,
    period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros_like(close, dtype=np.float64)
    valid = np.zeros_like(usable, dtype=np.bool_)
    source_start = np.full(len(close), -1, dtype=np.int64)
    for start, stop in _contiguous_segments(usable):
        tr = _true_range(high, low, close, start, stop)
        average = _wilder_average(tr, start, stop, period)
        first = start + period - 1
        for index in range(first, stop):
            values[index] = average[index] / close[index]
            valid[index] = True
            source_start[index] = start
    return values, valid, source_start


def _adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    usable: np.ndarray,
    period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros_like(close, dtype=np.float64)
    valid = np.zeros_like(usable, dtype=np.bool_)
    source_start = np.full(len(close), -1, dtype=np.int64)
    for start, stop in _contiguous_segments(usable):
        if stop - start < 2 * period:
            continue
        tr = _true_range(high, low, close, start, stop)
        plus_dm = np.zeros_like(close, dtype=np.float64)
        minus_dm = np.zeros_like(close, dtype=np.float64)
        for index in range(start + 1, stop):
            up = high[index] - high[index - 1]
            down = low[index - 1] - low[index]
            plus_dm[index] = up if up > down and up > 0.0 else 0.0
            minus_dm[index] = down if down > up and down > 0.0 else 0.0
        atr = _wilder_average(tr, start, stop, period)
        plus_avg = _wilder_average(plus_dm, start, stop, period)
        minus_avg = _wilder_average(minus_dm, start, stop, period)
        dx = np.zeros_like(close, dtype=np.float64)
        dx_valid = np.zeros_like(usable, dtype=np.bool_)
        di_first = start + period - 1
        for index in range(di_first, stop):
            if atr[index] <= _EPSILON:
                continue
            plus_di = 100.0 * plus_avg[index] / atr[index]
            minus_di = 100.0 * minus_avg[index] / atr[index]
            denominator = plus_di + minus_di
            dx[index] = (
                0.0
                if denominator <= _EPSILON
                else 100.0 * abs(plus_di - minus_di) / denominator
            )
            dx_valid[index] = True
        first = di_first + period - 1
        if first >= stop or not np.all(dx_valid[di_first : first + 1]):
            continue
        adx = float(np.mean(dx[di_first : first + 1]))
        values[first] = adx / 100.0
        valid[first] = True
        source_start[first] = start
        for index in range(first + 1, stop):
            if not dx_valid[index]:
                break
            adx = ((period - 1) * adx + dx[index]) / period
            values[index] = adx / 100.0
            valid[index] = True
            source_start[index] = start
    return values, valid, source_start


def _directional_indices(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    usable: np.ndarray,
    period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    plus_values = np.zeros_like(close, dtype=np.float64)
    minus_values = np.zeros_like(close, dtype=np.float64)
    valid = np.zeros_like(usable, dtype=np.bool_)
    source_start = np.full(len(close), -1, dtype=np.int64)
    for start, stop in _contiguous_segments(usable):
        if stop - start < period:
            continue
        tr = _true_range(high, low, close, start, stop)
        plus_dm = np.zeros_like(close, dtype=np.float64)
        minus_dm = np.zeros_like(close, dtype=np.float64)
        for index in range(start + 1, stop):
            up = high[index] - high[index - 1]
            down = low[index - 1] - low[index]
            plus_dm[index] = up if up > down and up > 0.0 else 0.0
            minus_dm[index] = down if down > up and down > 0.0 else 0.0
        atr = _wilder_average(tr, start, stop, period)
        plus_avg = _wilder_average(plus_dm, start, stop, period)
        minus_avg = _wilder_average(minus_dm, start, stop, period)
        first = start + period - 1
        for index in range(first, stop):
            if atr[index] <= _EPSILON:
                continue
            plus_values[index] = float(np.clip(plus_avg[index] / atr[index], 0.0, 1.0))
            minus_values[index] = float(
                np.clip(minus_avg[index] / atr[index], 0.0, 1.0)
            )
            valid[index] = True
            source_start[index] = start
    return plus_values, minus_values, valid, source_start


def _ema_feature(
    close: np.ndarray,
    usable: np.ndarray,
    period: int,
    *,
    slope: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros_like(close, dtype=np.float64)
    valid = np.zeros_like(usable, dtype=np.bool_)
    source_start = np.full(len(close), -1, dtype=np.int64)
    for start, stop in _contiguous_segments(usable):
        if stop - start < period:
            continue
        slow = _ema(close, start, stop, period)
        first = start + period - 1
        if slope:
            for index in range(max(first, start + 1), stop):
                values[index] = (slow[index] - slow[index - 1]) / close[index]
                valid[index] = True
                source_start[index] = start
        else:
            fast_period = max(2, period // 2)
            fast = _ema(close, start, stop, fast_period)
            for index in range(first, stop):
                values[index] = (fast[index] - slow[index]) / close[index]
                valid[index] = True
                source_start[index] = start
    return values, valid, source_start


def _atr_change(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    usable: np.ndarray,
    period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    atr_values, atr_valid, atr_start = _atr(high, low, close, usable, period)
    values = np.zeros_like(close, dtype=np.float64)
    valid = np.zeros_like(usable, dtype=np.bool_)
    source_start = np.full(len(close), -1, dtype=np.int64)
    for index in range(1, len(close)):
        if not atr_valid[index] or not atr_valid[index - 1]:
            continue
        previous = atr_values[index - 1]
        current = atr_values[index]
        values[index] = (
            0.0 if previous <= _EPSILON else math.log(max(current, _EPSILON) / previous)
        )
        valid[index] = True
        source_start[index] = min(int(atr_start[index - 1]), int(atr_start[index]))
    return values, valid, source_start


def _funding_feature(
    funding_rate: np.ndarray,
    funding_available: np.ndarray,
    usable: np.ndarray,
    lookback: int,
    *,
    zscore: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros_like(funding_rate, dtype=np.float64)
    valid = np.zeros_like(usable, dtype=np.bool_)
    source_start = np.full(len(funding_rate), -1, dtype=np.int64)
    history: list[int] = []
    for index in range(len(funding_rate)):
        if not usable[index]:
            history.clear()
            continue
        if not funding_available[index]:
            continue
        history.append(index)
        sample_indices = history[-lookback:]
        if zscore:
            if len(sample_indices) < min(4, lookback):
                continue
            sample = funding_rate[sample_indices]
            std = float(np.std(sample))
            values[index] = (
                0.0
                if std <= _EPSILON
                else (funding_rate[index] - float(np.mean(sample))) / std
            )
            source_start[index] = sample_indices[0]
        else:
            if len(history) < 2:
                continue
            previous = history[-2]
            values[index] = (funding_rate[index] - funding_rate[previous]) * 10_000.0
            source_start[index] = previous
        valid[index] = True
    return values, valid, source_start


def _ichimoku_line(high: np.ndarray, low: np.ndarray, start: int, stop: int) -> float:
    return 0.5 * (float(np.max(high[start:stop])) + float(np.min(low[start:stop])))


def calculate_feature_events(
    spec: FeatureSpec,
    *,
    open_price: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    funding_rate: np.ndarray,
    funding_available: np.ndarray,
    row_present: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return feature events, validity and earliest source index for each event."""

    arrays = (
        open_price,
        high,
        low,
        close,
        volume,
        funding_rate,
        funding_available,
        row_present,
        active,
    )
    n_bars = len(close)
    if any(np.asarray(value).shape != (n_bars,) for value in arrays):
        raise ValueError(
            "feature input arrays must have identical one-dimensional shape"
        )
    usable = np.asarray(row_present, dtype=np.bool_) & np.asarray(
        active, dtype=np.bool_
    )
    values = np.zeros(n_bars, dtype=np.float64)
    valid = np.zeros(n_bars, dtype=np.bool_)
    source_start = np.full(n_bars, -1, dtype=np.int64)
    kind = FeatureKind(spec.kind)

    if kind is FeatureKind.FUNDING_BPS:
        valid = np.asarray(funding_available, dtype=np.bool_) & usable
        values[valid] = np.asarray(funding_rate, dtype=np.float64)[valid] * 10_000.0
        source_start[valid] = np.flatnonzero(valid)
    elif kind in {FeatureKind.FUNDING_CHANGE, FeatureKind.FUNDING_ZSCORE}:
        values, valid, source_start = _funding_feature(
            funding_rate,
            np.asarray(funding_available, dtype=np.bool_),
            usable,
            spec.lookback,
            zscore=kind is FeatureKind.FUNDING_ZSCORE,
        )
    elif kind is FeatureKind.RSI:
        values, valid, source_start = _rsi(close, usable, spec.lookback)
    elif kind in {
        FeatureKind.MACD_LINE,
        FeatureKind.MACD_SIGNAL,
        FeatureKind.MACD_HISTOGRAM,
    }:
        values, valid, source_start = _macd(close, usable, kind)
    elif kind is FeatureKind.ATR_PCT:
        values, valid, source_start = _atr(high, low, close, usable, spec.lookback)
    elif kind is FeatureKind.ATR_CHANGE:
        values, valid, source_start = _atr_change(
            high, low, close, usable, spec.lookback
        )
    elif kind is FeatureKind.ADX:
        values, valid, source_start = _adx(high, low, close, usable, spec.lookback)
    elif kind in {FeatureKind.PLUS_DI, FeatureKind.MINUS_DI, FeatureKind.DI_SPREAD}:
        plus_di, minus_di, valid, source_start = _directional_indices(
            high, low, close, usable, spec.lookback
        )
        if kind is FeatureKind.PLUS_DI:
            values = plus_di
        elif kind is FeatureKind.MINUS_DI:
            values = minus_di
        else:
            values = plus_di - minus_di
    elif kind in {FeatureKind.EMA_DISTANCE, FeatureKind.EMA_SLOPE}:
        values, valid, source_start = _ema_feature(
            close,
            usable,
            spec.lookback,
            slope=kind is FeatureKind.EMA_SLOPE,
        )
    else:
        obv = np.zeros(n_bars, dtype=np.float64)
        for segment_start, segment_stop in _contiguous_segments(usable):
            for index in range(segment_start + 1, segment_stop):
                direction = float(np.sign(close[index] - close[index - 1]))
                obv[index] = obv[index - 1] + direction * volume[index]
        for index in range(n_bars):
            if not usable[index]:
                continue
            if kind is FeatureKind.BODY_RETURN:
                values[index] = math.log(close[index] / open_price[index])
                valid[index] = True
                source_start[index] = index
            elif kind is FeatureKind.HIGH_LOW_RANGE:
                values[index] = (high[index] - low[index]) / close[index]
                valid[index] = True
                source_start[index] = index
            elif kind in {
                FeatureKind.UPPER_WICK_RATIO,
                FeatureKind.LOWER_WICK_RATIO,
                FeatureKind.CLOSE_LOCATION_VALUE,
            }:
                spread = high[index] - low[index]
                if spread <= _EPSILON:
                    values[index] = 0.0
                elif kind is FeatureKind.UPPER_WICK_RATIO:
                    values[index] = (
                        high[index] - max(open_price[index], close[index])
                    ) / spread
                elif kind is FeatureKind.LOWER_WICK_RATIO:
                    values[index] = (
                        min(open_price[index], close[index]) - low[index]
                    ) / spread
                else:
                    values[index] = (
                        2.0 * close[index] - high[index] - low[index]
                    ) / spread
                valid[index] = True
                source_start[index] = index
            elif kind in {FeatureKind.GAP_RETURN, FeatureKind.VOLUME_LOG_CHANGE}:
                start = index - 1
                if _window_is_valid(usable, start, index + 1):
                    if kind is FeatureKind.GAP_RETURN:
                        values[index] = math.log(open_price[index] / close[start])
                    else:
                        previous = volume[start]
                        current = volume[index]
                        values[index] = (
                            0.0
                            if previous <= _EPSILON or current <= _EPSILON
                            else math.log(current / previous)
                        )
                    valid[index] = True
                    source_start[index] = start
            elif kind in {
                FeatureKind.PARKINSON_VOLATILITY,
                FeatureKind.GARMAN_KLASS_VOLATILITY,
            }:
                start = index - spec.lookback + 1
                if _window_is_valid(usable, start, index + 1):
                    log_range = np.log(high[start : index + 1] / low[start : index + 1])
                    if kind is FeatureKind.PARKINSON_VOLATILITY:
                        variance = float(np.mean(np.square(log_range))) / (
                            4.0 * math.log(2.0)
                        )
                    else:
                        log_body = np.log(
                            close[start : index + 1] / open_price[start : index + 1]
                        )
                        terms = 0.5 * np.square(log_range) - (
                            2.0 * math.log(2.0) - 1.0
                        ) * np.square(log_body)
                        variance = max(float(np.mean(terms)), 0.0)
                    values[index] = math.sqrt(max(variance, 0.0))
                    valid[index] = True
                    source_start[index] = start
            elif kind in {
                FeatureKind.DOWNSIDE_VOLATILITY,
                FeatureKind.UPSIDE_VOLATILITY,
                FeatureKind.VOLATILITY_OF_VOLATILITY,
            }:
                start = index - spec.lookback
                if _window_is_valid(usable, start, index + 1):
                    returns = np.diff(np.log(close[start : index + 1]))
                    if kind is FeatureKind.DOWNSIDE_VOLATILITY:
                        sample = returns[returns < 0.0]
                        values[index] = (
                            0.0
                            if sample.size == 0
                            else float(np.sqrt(np.mean(np.square(sample))))
                        )
                    elif kind is FeatureKind.UPSIDE_VOLATILITY:
                        sample = returns[returns > 0.0]
                        values[index] = (
                            0.0
                            if sample.size == 0
                            else float(np.sqrt(np.mean(np.square(sample))))
                        )
                    else:
                        values[index] = float(np.std(np.abs(returns)))
                    valid[index] = True
                    source_start[index] = start
            elif kind is FeatureKind.RANGE_EXPANSION:
                start = index - spec.lookback
                if _window_is_valid(usable, start, index + 1):
                    prior = (high[start:index] - low[start:index]) / close[start:index]
                    current = (high[index] - low[index]) / close[index]
                    mean = float(np.mean(prior))
                    values[index] = 0.0 if mean <= _EPSILON else current / mean - 1.0
                    valid[index] = True
                    source_start[index] = start
            elif kind in {
                FeatureKind.LINEAR_REGRESSION_SLOPE,
                FeatureKind.TREND_R2,
            }:
                start = index - spec.lookback + 1
                if _window_is_valid(usable, start, index + 1):
                    sample = np.log(close[start : index + 1])
                    x = np.arange(sample.size, dtype=np.float64)
                    x_centered = x - float(np.mean(x))
                    y_centered = sample - float(np.mean(sample))
                    denominator = float(np.dot(x_centered, x_centered))
                    slope = (
                        0.0
                        if denominator <= _EPSILON
                        else float(np.dot(x_centered, y_centered) / denominator)
                    )
                    if kind is FeatureKind.LINEAR_REGRESSION_SLOPE:
                        values[index] = slope
                    else:
                        fitted = float(np.mean(sample)) + slope * x_centered
                        total = float(np.dot(y_centered, y_centered))
                        residual = float(np.sum(np.square(sample - fitted)))
                        values[index] = (
                            0.0
                            if total <= _EPSILON
                            else float(np.clip(1.0 - residual / total, 0.0, 1.0))
                        )
                    valid[index] = True
                    source_start[index] = start
            elif kind is FeatureKind.MFI:
                start = index - spec.lookback
                if _window_is_valid(usable, start, index + 1):
                    typical = (
                        high[start : index + 1]
                        + low[start : index + 1]
                        + close[start : index + 1]
                    ) / 3.0
                    flow = typical * volume[start : index + 1]
                    changes = np.diff(typical)
                    positive = float(np.sum(flow[1:][changes > 0.0]))
                    negative = float(np.sum(flow[1:][changes < 0.0]))
                    if negative <= _EPSILON:
                        mfi = 100.0 if positive > _EPSILON else 50.0
                    else:
                        ratio = positive / negative
                        mfi = 100.0 - 100.0 / (1.0 + ratio)
                    values[index] = (mfi - 50.0) / 50.0
                    valid[index] = True
                    source_start[index] = start
            elif kind is FeatureKind.CMF:
                start = index - spec.lookback + 1
                if _window_is_valid(usable, start, index + 1):
                    spread = high[start : index + 1] - low[start : index + 1]
                    multiplier = np.divide(
                        2.0 * close[start : index + 1]
                        - high[start : index + 1]
                        - low[start : index + 1],
                        spread,
                        out=np.zeros_like(spread),
                        where=spread > _EPSILON,
                    )
                    denominator = float(np.sum(volume[start : index + 1]))
                    values[index] = (
                        0.0
                        if denominator <= _EPSILON
                        else float(
                            np.sum(multiplier * volume[start : index + 1]) / denominator
                        )
                    )
                    valid[index] = True
                    source_start[index] = start
            elif kind is FeatureKind.VWAP_DISTANCE:
                start = index - spec.lookback + 1
                if _window_is_valid(usable, start, index + 1):
                    typical = (
                        high[start : index + 1]
                        + low[start : index + 1]
                        + close[start : index + 1]
                    ) / 3.0
                    denominator = float(np.sum(volume[start : index + 1]))
                    vwap = (
                        close[index]
                        if denominator <= _EPSILON
                        else float(
                            np.sum(typical * volume[start : index + 1]) / denominator
                        )
                    )
                    values[index] = (close[index] - vwap) / close[index]
                    valid[index] = True
                    source_start[index] = start
            elif kind is FeatureKind.PRICE_VOLUME_CORRELATION:
                start = index - spec.lookback
                if _window_is_valid(usable, start, index + 1):
                    price_returns = np.diff(np.log(close[start : index + 1]))
                    safe_volume = np.maximum(volume[start : index + 1], _EPSILON)
                    volume_changes = np.diff(np.log(safe_volume))
                    if (
                        np.std(price_returns) <= _EPSILON
                        or np.std(volume_changes) <= _EPSILON
                    ):
                        values[index] = 0.0
                    else:
                        values[index] = float(
                            np.clip(
                                np.corrcoef(price_returns, volume_changes)[0, 1],
                                -1.0,
                                1.0,
                            )
                        )
                    valid[index] = True
                    source_start[index] = start
            elif kind in {FeatureKind.OBV_CHANGE, FeatureKind.OBV_ACCELERATION}:
                if kind is FeatureKind.OBV_CHANGE:
                    start = index - spec.lookback
                    middle = start
                else:
                    start = index - 2 * spec.lookback
                    middle = index - spec.lookback
                if _window_is_valid(usable, start, index + 1):
                    denominator = float(np.sum(volume[start + 1 : index + 1]))
                    if denominator <= _EPSILON:
                        values[index] = 0.0
                    elif kind is FeatureKind.OBV_CHANGE:
                        values[index] = float(
                            np.clip((obv[index] - obv[start]) / denominator, -1.0, 1.0)
                        )
                    else:
                        recent = obv[index] - obv[middle]
                        previous = obv[middle] - obv[start]
                        values[index] = float(
                            np.clip((recent - previous) / denominator, -1.0, 1.0)
                        )
                    valid[index] = True
                    source_start[index] = start
            elif kind is FeatureKind.RELATIVE_VOLUME:
                start = index - spec.lookback + 1
                if _window_is_valid(usable, start, index + 1):
                    mean = float(np.mean(volume[start : index + 1]))
                    relative = 0.0 if mean <= _EPSILON else volume[index] / mean - 1.0
                    values[index] = math.tanh(relative)
                    valid[index] = True
                    source_start[index] = start
            elif kind is FeatureKind.LOG_RETURN:
                start = index - spec.lookback
                if _window_is_valid(usable, start, index + 1):
                    values[index] = math.log(close[index] / close[start])
                    valid[index] = True
                    source_start[index] = start
            elif kind is FeatureKind.REALIZED_VOLATILITY:
                start = index - spec.lookback
                if _window_is_valid(usable, start, index + 1):
                    returns = np.diff(np.log(close[start : index + 1]))
                    values[index] = float(np.sqrt(np.mean(np.square(returns))))
                    valid[index] = True
                    source_start[index] = start
            elif kind is FeatureKind.VOLUME_ZSCORE:
                start = max(0, index - spec.lookback + 1)
                mask = usable[start : index + 1]
                sample = volume[start : index + 1][mask]
                if sample.size >= spec.min_periods:
                    std = float(np.std(sample))
                    values[index] = (
                        0.0
                        if std <= _EPSILON
                        else (volume[index] - float(np.mean(sample))) / std
                    )
                    valid[index] = True
                    source_start[index] = start
            elif kind in {
                FeatureKind.BOLLINGER_POSITION,
                FeatureKind.BOLLINGER_BANDWIDTH,
            }:
                start = index - spec.lookback + 1
                if _window_is_valid(usable, start, index + 1):
                    sample = close[start : index + 1]
                    mean = float(np.mean(sample))
                    std = float(np.std(sample))
                    if kind is FeatureKind.BOLLINGER_POSITION:
                        values[index] = (
                            0.0
                            if std <= _EPSILON
                            else (close[index] - mean) / (2.0 * std)
                        )
                    else:
                        values[index] = (
                            0.0 if abs(mean) <= _EPSILON else 4.0 * std / mean
                        )
                    valid[index] = True
                    source_start[index] = start
            elif kind in {FeatureKind.STOCHASTIC_K, FeatureKind.WILLIAMS_R}:
                start = index - spec.lookback + 1
                if _window_is_valid(usable, start, index + 1):
                    highest = float(np.max(high[start : index + 1]))
                    lowest = float(np.min(low[start : index + 1]))
                    spread = highest - lowest
                    ratio = (
                        0.5 if spread <= _EPSILON else (close[index] - lowest) / spread
                    )
                    values[index] = (
                        2.0 * ratio - 1.0
                        if kind is FeatureKind.STOCHASTIC_K
                        else ratio - 1.0
                    )
                    valid[index] = True
                    source_start[index] = start
            elif kind is FeatureKind.STOCHASTIC_D:
                base_start = index - spec.lookback - _STOCHASTIC_SMOOTH + 2
                if base_start < 0:
                    continue
                k_values: list[float] = []
                for k_index in range(index - _STOCHASTIC_SMOOTH + 1, index + 1):
                    start = k_index - spec.lookback + 1
                    if not _window_is_valid(usable, start, k_index + 1):
                        break
                    highest = float(np.max(high[start : k_index + 1]))
                    lowest = float(np.min(low[start : k_index + 1]))
                    spread = highest - lowest
                    ratio = (
                        0.5
                        if spread <= _EPSILON
                        else (close[k_index] - lowest) / spread
                    )
                    k_values.append(2.0 * ratio - 1.0)
                if len(k_values) == _STOCHASTIC_SMOOTH:
                    values[index] = float(np.mean(k_values))
                    valid[index] = True
                    source_start[index] = base_start
            elif kind is FeatureKind.CCI:
                start = index - spec.lookback + 1
                if _window_is_valid(usable, start, index + 1):
                    typical = (
                        high[start : index + 1]
                        + low[start : index + 1]
                        + close[start : index + 1]
                    ) / 3.0
                    mean = float(np.mean(typical))
                    deviation = float(np.mean(np.abs(typical - mean)))
                    raw = (
                        0.0
                        if deviation <= _EPSILON
                        else (typical[-1] - mean) / (0.015 * deviation)
                    )
                    values[index] = math.tanh(raw / 100.0)
                    valid[index] = True
                    source_start[index] = start
            elif kind is FeatureKind.OBV_SLOPE:
                start = index - spec.lookback
                if _window_is_valid(usable, start, index + 1):
                    denominator = float(np.sum(volume[start + 1 : index + 1]))
                    values[index] = (
                        0.0
                        if denominator <= _EPSILON
                        else float(
                            np.clip((obv[index] - obv[start]) / denominator, -1.0, 1.0)
                        )
                    )
                    valid[index] = True
                    source_start[index] = start
            elif kind in {
                FeatureKind.ICHIMOKU_TENKAN_DISTANCE,
                FeatureKind.ICHIMOKU_KIJUN_DISTANCE,
                FeatureKind.ICHIMOKU_CLOUD_POSITION,
                FeatureKind.ICHIMOKU_CLOUD_THICKNESS,
            }:
                required = {
                    FeatureKind.ICHIMOKU_TENKAN_DISTANCE: _ICHIMOKU_TENKAN,
                    FeatureKind.ICHIMOKU_KIJUN_DISTANCE: _ICHIMOKU_KIJUN,
                    FeatureKind.ICHIMOKU_CLOUD_POSITION: _ICHIMOKU_SENKOU_B,
                    FeatureKind.ICHIMOKU_CLOUD_THICKNESS: _ICHIMOKU_SENKOU_B,
                }[kind]
                start = index - required + 1
                if not _window_is_valid(usable, start, index + 1):
                    continue
                tenkan = _ichimoku_line(
                    high, low, index - _ICHIMOKU_TENKAN + 1, index + 1
                )
                if kind is FeatureKind.ICHIMOKU_TENKAN_DISTANCE:
                    values[index] = (close[index] - tenkan) / close[index]
                else:
                    kijun = _ichimoku_line(
                        high, low, index - _ICHIMOKU_KIJUN + 1, index + 1
                    )
                    if kind is FeatureKind.ICHIMOKU_KIJUN_DISTANCE:
                        values[index] = (close[index] - kijun) / close[index]
                    else:
                        span_a = 0.5 * (tenkan + kijun)
                        span_b = _ichimoku_line(
                            high, low, index - _ICHIMOKU_SENKOU_B + 1, index + 1
                        )
                        if kind is FeatureKind.ICHIMOKU_CLOUD_POSITION:
                            values[index] = (
                                close[index] - 0.5 * (span_a + span_b)
                            ) / close[index]
                        else:
                            values[index] = abs(span_a - span_b) / close[index]
                valid[index] = True
                source_start[index] = start
            else:
                raise ValueError(f"unsupported feature kind: {kind}")

    if spec.normalization is NormalizationMode.ROLLING_ZSCORE:
        return _rolling_normalize(
            values,
            valid,
            source_start,
            window=spec.normalization_window,
            min_periods=spec.min_periods,
        )
    return values, valid, source_start


__all__ = ["calculate_feature_events"]
