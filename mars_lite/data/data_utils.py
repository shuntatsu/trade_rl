"""
データユーティリティ

タイムフレーム変換とOHLCVリサンプリング。
"""

import pandas as pd

TF_TO_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
}


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    1分足等のOHLCV DataFrame（timestamp列あり）を上位足へリサンプル

    バーのタイムスタンプは「バー開始時刻」（取引所慣行と一致）。
    欠損区間のバーは落とす。
    """
    if timeframe not in TF_TO_MINUTES:
        raise ValueError(f"unknown timeframe: {timeframe}")
    freq = f"{TF_TO_MINUTES[timeframe]}min"
    out = (
        df.set_index("timestamp")
        .resample(freq, label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "close"])
        .reset_index()
    )
    return out
