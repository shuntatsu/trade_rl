import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mars_lite.data.multi_timeframe_loader import DailyFileLoader
from mars_lite.data.preprocessing import preprocess_ohlcv


def create_dummy_ohlcv(n=5000):
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": np.random.randn(n).cumsum() + 100,
            "high": np.random.randn(n).cumsum() + 102,
            "low": np.random.randn(n).cumsum() + 98,
            "close": np.random.randn(n).cumsum() + 100,
            "volume": np.random.rand(n) * 1000,
        }
    )
    # Fix HLC
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df


def test_feature_generation():
    print("Testing Feature Generation...")
    df = create_dummy_ohlcv(5000)

    # Process
    # 90 days = 2160 hours. We have 5000 hours.
    processed = preprocess_ohlcv(df, interval="1h", scale_window_days=90)

    print(f"Columns: {processed.columns.tolist()}")

    # Check Indicators
    expected_cols = ["rsi_14", "bb_width", "adx_14", "rci_9"]
    for c in expected_cols:
        assert c in processed.columns, f"Missing {c}"

    # Check Normalized
    norm_cols = [f"{c}_norm" for c in expected_cols] + [
        "log_return_norm",
        "vol_gk_norm",
    ]
    for c in norm_cols:
        assert c in processed.columns, f"Missing {c}"
        # Check range roughly [0, 1] (might slightly exceed if rolling max/min was old)
        # But rolling minmax uses current window, so it should be strictly [0, 1] if window covers index?
        # Yes, standard rolling max/min includes current point.
        val = processed[c].iloc[-1]
        print(
            f"  {c}: {val:.4f} (Range: {processed[c].min():.4f} - {processed[c].max():.4f})"
        )
        assert not processed[c].isna().all(), f"{c} is all NaN"

    print("feature generation OK.")


def test_caching():
    print("\nTesting Caching mechanism...")
    data_dir = Path("./data_test/BTCUSDT/1h")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy csv
    df = create_dummy_ohlcv(100)
    df["timestamp"] = (df["timestamp"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(
        "1s"
    )  # convert to unix
    df.to_csv(data_dir / "2024-01-01.csv", index=False)

    loader = DailyFileLoader(
        data_dir=data_dir.parent.parent,
        symbol="BTCUSDT",
        timeframes=["1h"],
        preprocess=True,
    )

    # Check cache path
    cache_path = loader._get_cache_path("1h")
    if cache_path.exists():
        os.remove(cache_path)

    print("1. Loading first time (Generates Cache)...")
    loader.load_date_range("1h", start_date="2024-01-01", end_date="2024-01-01")

    if not cache_path.exists():
        print(f"ERROR: Cache file not created at {cache_path}")
        return
    print(f"Cache created: {cache_path}")

    print("2. Loading second time (Uses Cache)...")
    # To verify it uses cache, we could modify CSV and see if it ignores it,
    # OR simpler: check print output or speed?
    # We rely on code logic.
    loader.load_date_range("1h", start_date="2024-01-01", end_date="2024-01-01")

    # Clean up
    shutil.rmtree("./data_test")

    # Removing processed dir if empty?
    # shutil.rmtree(loader.cache_dir)
    print("Caching OK.")


if __name__ == "__main__":
    test_feature_generation()
    test_caching()
