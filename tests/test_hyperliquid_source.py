"""
HyperliquidSource のオフラインテスト（ネットワーク非依存）

symbol→coin 正規化とキャッシュ読み込みを検証する。実API取得は
ネットワーク依存のためここでは行わない（fetch_hyperliquid.py で確認）。
"""

import pandas as pd

from mars_lite.data.sources import HyperliquidSource, create_source


def test_coin_normalization():
    src = HyperliquidSource(
        ["BTCUSDT", "ETHUSDC", "SOL", "DOGEPERP"],
        days=10,
        cache_dir="/tmp/hl_test_norm",
    )
    assert src._coin("BTCUSDT") == "BTC"
    assert src._coin("ETHUSDC") == "ETH"
    assert src._coin("SOL") == "SOL"
    assert src._coin("DOGEPERP") == "DOGE"


def test_symbol_map_override(tmp_path):
    src = HyperliquidSource(
        ["kBONK"], days=10, cache_dir=str(tmp_path), symbol_map={"kBONK": "kBONK"}
    )
    assert src._coin("kBONK") == "kBONK"


def test_reads_cache_without_network(tmp_path):
    """キャッシュCSVがあればネットワークに触れず読める"""
    coin, interval = "BTC", "1h"
    end = pd.Timestamp.now().floor("h")  # 実データ同様 tz-naive
    idx = pd.date_range(end - pd.Timedelta(days=10), end, freq="1h")
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
        }
    )
    (tmp_path / f"{coin}_{interval}.csv").write_text(df.to_csv(index=False))

    src = HyperliquidSource(["BTCUSDT"], days=8, cache_dir=str(tmp_path))
    # _fetch_candles を呼んだら失敗させ、キャッシュ経路のみを検証
    src._fetch_candles = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("network should not be called")
    )
    out = src.load_klines("BTCUSDT", "1h")
    assert len(out) == len(idx)
    assert list(out.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_factory_registers_hyperliquid(tmp_path):
    src = create_source("hyperliquid", ["BTCUSDT"], days=5, cache_dir=str(tmp_path))
    assert isinstance(src, HyperliquidSource)


def test_load_orderflow_reads_cache(tmp_path):
    """fetch_hl_derivatives.py が書くレイアウトのオーダーフローキャッシュを読める"""
    of = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1min"),
            "buy_volume": [1.0] * 5,
            "sell_volume": [1.0] * 5,
            "trade_count": [10] * 5,
            "avg_trade_size": [0.2] * 5,
            "volume_imbalance": [0.0] * 5,
        }
    )
    of.to_csv(tmp_path / "BTC_orderflow_1m.csv", index=False)

    src = HyperliquidSource(["BTCUSDT"], days=1, cache_dir=str(tmp_path))
    out = src.load_orderflow("BTCUSDT")
    assert len(out) == 5
    assert "volume_imbalance" in out.columns


def test_load_derivatives_reads_binance_proxy_cache(tmp_path):
    """fetch_hl_derivatives.py が書くBinance代理キャッシュを読める（ネイティブ無し）"""
    deriv = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1h"),
            "open_interest": [100.0, 101, 102, 103, 104],
            "ls_ratio": [1.0] * 5,
            "liq_notional": [0.0] * 5,
        }
    )
    deriv.to_csv(tmp_path / "BTC_derivatives.csv", index=False)

    src = HyperliquidSource(["BTCUSDT"], days=1, cache_dir=str(tmp_path))
    out = src.load_derivatives("BTCUSDT")
    assert len(out) == 5
    assert list(out["open_interest"]) == [100.0, 101, 102, 103, 104]


def test_native_snapshot_overrides_overlapping_open_interest(tmp_path):
    """collect_hl_snapshots.py のネイティブOIが重複期間でBinance代理を上書きする"""
    deriv = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1h"),
            "open_interest": [100.0, 101, 102, 103, 104],
            "ls_ratio": [1.0] * 5,
            "liq_notional": [0.0] * 5,
        }
    )
    deriv.to_csv(tmp_path / "BTC_derivatives.csv", index=False)

    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    snap = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01 03:00", periods=2, freq="1h"),
            "open_interest": [999.0, 998.0],
        }
    )
    snap.to_csv(snap_dir / "BTC_ctx.csv", index=False)

    src = HyperliquidSource(["BTCUSDT"], days=1, cache_dir=str(tmp_path))
    out = src.load_derivatives("BTCUSDT")
    row3 = out[out["timestamp"] == pd.Timestamp("2024-01-01 03:00:00")]
    row0 = out[out["timestamp"] == pd.Timestamp("2024-01-01 00:00:00")]
    assert row3["open_interest"].iloc[0] == 999.0
    assert row0["open_interest"].iloc[0] == 100.0
