"""
テクニカル指標計算モジュール

Pandas/NumPyのみを使用して軽量に実装。
"""

from typing import Tuple

import numpy as np
import pandas as pd


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    # Wilder's Smoothing (Optional, using simple rolling for speed/stability initially)
    # To match standard TA exactly:
    # avg_gain = avg_gain.ewm(com=period-1, adjust=False).mean()
    # avg_loss = avg_loss.ewm(com=period-1, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Bollinger Bands
    Returns: (%B, BandWidth)
    """
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()

    upper = sma + (std * num_std)
    lower = sma - (std * num_std)

    # %B: (Price - Lower) / (Upper - Lower)
    percent_b = (series - lower) / (upper - lower).replace(0, 1e-9)

    # BandWidth: (Upper - Lower) / Middle
    bandwidth = (upper - lower) / sma.replace(0, 1e-9)

    return percent_b, bandwidth


def calc_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average Directional Index (ADX)"""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join="outer").max(axis=1)

    atr = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period).mean() / atr)
    minus_di = 100 * (minus_dm.abs().ewm(alpha=1 / period).mean() / atr)

    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1 / period).mean()

    return adx


def calc_cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """Commodity Channel Index (CCI)"""
    tp = (high + low + close) / 3.0
    sma = tp.rolling(window=period, min_periods=period).mean()
    mad = tp.rolling(window=period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return (tp - sma) / (0.015 * mad.replace(0, 1e-9))


def calc_rci(series: pd.Series, period: int = 9) -> pd.Series:
    """
    Rank Correlation Index (RCI)
    時間順位と価格順位の相関。
    """

    def rci_func(x):
        n = len(x)
        time_rank = np.arange(1, n + 1)
        price_rank = np.argsort(np.argsort(x)) + 1
        d = time_rank - price_rank
        d_sq = np.sum(d**2)
        rci_val = (1 - (6 * d_sq) / (n * (n**2 - 1))) * 100
        return rci_val

    # Rolling apply is slow for loop, but simplest for pure pandas
    # Optimized vectorization for RCI is complex.
    rci = series.rolling(window=period).apply(rci_func, raw=True)
    return rci


def fetch_fear_and_greed() -> pd.DataFrame:
    """
    Fetch Fear & Greed Index from Alternative.me
    Returns DataFrame with 'timestamp' and 'fng_value'
    """
    import requests

    try:
        url = "https://api.alternative.me/fng/"
        params = {"limit": 0}  # 0 means all data
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()["data"]

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df["fng_value"] = df["value"].astype(float)

        # Keep only necessary columns
        return df[["timestamp", "fng_value"]].sort_values("timestamp")
    except Exception as e:
        print(f"Warning: Failed to fetch Fear & Greed: {e}")
        return pd.DataFrame()
