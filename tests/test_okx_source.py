"""
OKXSource のオフラインテスト（ネットワーク非依存）

instId正規化・キャッシュ読み込み・ファクトリ登録を検証する。実API取得は
ネットワーク依存のためここでは行わない。
"""

import pandas as pd

from mars_lite.data.sources import OKXSource, create_source


def test_inst_id_normalization():
    src = OKXSource(["BTCUSDT", "ETHUSDC", "SOL"], days=10, cache_dir="/tmp/okx_test_norm")
    assert src._inst_id("BTCUSDT") == "BTC-USDT-SWAP"
    assert src._inst_id("ETHUSDC") == "ETH-USDC-SWAP"
    assert src._inst_id("SOL") == "SOL-USDT-SWAP"


def test_reads_cache_without_network(tmp_path):
    """キャッシュCSVがあればネットワークに触れず読める"""
    end = pd.Timestamp.now().floor("h")
    idx = pd.date_range(end - pd.Timedelta(days=10), end, freq="1h")
    df = pd.DataFrame({
        "timestamp": idx,
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.5, "volume": 10.0,
    })
    (tmp_path / "BTC-USDT-SWAP_1h.csv").write_text(df.to_csv(index=False))

    src = OKXSource(["BTCUSDT"], days=8, cache_dir=str(tmp_path))
    src._fetch_candles = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("network should not be called"))
    out = src.load_klines("BTCUSDT", "1h")
    assert len(out) == len(idx)
    assert list(out.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_factory_registers_okx(tmp_path):
    src = create_source("okx", ["BTCUSDT"], days=5, cache_dir=str(tmp_path))
    assert isinstance(src, OKXSource)


def test_load_funding_reads_cache(tmp_path):
    fund = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="8h"),
        "funding_rate": [0.0001] * 5,
    })
    fund.to_csv(tmp_path / "BTC-USDT-SWAP_funding.csv", index=False)

    src = OKXSource(["BTCUSDT"], days=1, cache_dir=str(tmp_path))
    out = src.load_funding("BTCUSDT")
    assert len(out) == 5
    assert "funding_rate" in out.columns


def test_load_orderflow_and_derivatives_default_empty(tmp_path):
    """OKXはオーダーフロー/デリバティブ非対応のため既定の空フレームを返す"""
    src = OKXSource(["BTCUSDT"], days=1, cache_dir=str(tmp_path))
    assert src.load_orderflow("BTCUSDT").empty
    assert src.load_derivatives("BTCUSDT").empty


def test_fetch_candles_paginates_backward_until_empty(monkeypatch, tmp_path):
    """afterカーソルで後方ページングし、空応答が来たら停止する"""
    import mars_lite.data.sources as sources_mod

    calls = []

    def fake_get(path, params, max_retries=6):
        calls.append(params)
        after = int(params["after"])
        if len(calls) <= 2:
            t = after - 3_600_000
            return [[str(t), "100", "101", "99", "100.5", "1.0"]]
        return []

    monkeypatch.setattr(sources_mod, "_okx_get", fake_get)

    src = OKXSource(["BTCUSDT"], days=200, cache_dir=str(tmp_path))
    df = src._fetch_candles("BTC-USDT-SWAP", "1H")
    assert len(calls) >= 2
    assert not df.empty
