import numpy as np
import pandas as pd

from mars_lite.data.data_utils import resample_ohlcv

# Dummy 1m data
df = pd.DataFrame(
    {
        "timestamp": pd.date_range("2024-01-01", periods=60 * 24, freq="1min"),
        "open": np.random.randn(1440).cumsum() + 10000,
        "high": np.random.randn(1440).cumsum() + 10000,
        "low": np.random.randn(1440).cumsum() + 10000,
        "close": np.random.randn(1440).cumsum() + 10000,
        "volume": np.random.rand(1440) * 100,
    }
)

print("Testing resample '15m'...")
try:
    res = resample_ohlcv(df, "15m")
    print(f"Success! {len(res)} rows")
    print(res.head())
except Exception as e:
    print(f"FAILED: {e}")
    import traceback

    traceback.print_exc()

print("Testing resample '1h'...")
try:
    res = resample_ohlcv(df, "1h")
    print(f"Success! {len(res)} rows")
except Exception as e:
    print(f"FAILED: {e}")
