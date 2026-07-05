"""
HyperliquidSource のオフラインテスト（ネットワーク非依存）

symbol→coin 正規化とキャッシュ読み込みを検証する。実API取得は
ネットワーク依存のためここでは行わない（fetch_hyperliquid.py で確認）。
"""

import numpy as np
import pandas as pd
import pytest

from mars_lite.data.sources import HyperliquidSource, create_source


def test_coin_normalization():
    src = HyperliquidSource(["BTCUSDT", "ETHUSDC", "SOL", "DOGEPERP"],
                            days=10, cache_dir="/tmp/hl_test_norm")
    assert src._coin("BTCUSDT") == "BTC"
    assert src._coin("ETHUSDC") == "ETH"
    assert src._coin("SOL") == "SOL"
    assert src._coin("DOGEPERP") == "DOGE"


def test_symbol_map_override(tmp_path):
    src = HyperliquidSource(["kBONK"], days=10, cache_dir=str(tmp_path),
                            symbol_map={"kBONK": "kBONK"})
    assert src._coin("kBONK") == "kBONK"


def test_reads_cache_without_network(tmp_path):
    """キャッシュCSVがあればネットワークに触れず読める"""
    coin, interval = "BTC", "1h"
    end = pd.Timestamp.now().floor("h")   # 実データ同様 tz-naive
    idx = pd.date_range(end - pd.Timedelta(days=10), end, freq="1h")
    df = pd.DataFrame({
        "timestamp": idx,
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.5, "volume": 10.0,
    })
    (tmp_path / f"{coin}_{interval}.csv").write_text(df.to_csv(index=False))

    src = HyperliquidSource(["BTCUSDT"], days=8, cache_dir=str(tmp_path))
    # _fetch_candles を呼んだら失敗させ、キャッシュ経路のみを検証
    src._fetch_candles = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("network should not be called"))
    out = src.load_klines("BTCUSDT", "1h")
    assert len(out) == len(idx)
    assert list(out.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_factory_registers_hyperliquid(tmp_path):
    src = create_source("hyperliquid", ["BTCUSDT"], days=5, cache_dir=str(tmp_path))
    assert isinstance(src, HyperliquidSource)
