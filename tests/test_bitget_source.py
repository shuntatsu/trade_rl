"""
BitgetSource のオフラインテスト（ネットワーク非依存）

キャッシュ読み込みとファクトリ登録を検証する。実API取得はネットワーク
依存のためここでは行わない。
"""

import pandas as pd

from mars_lite.data.sources import BitgetSource, create_source


def test_reads_cache_without_network(tmp_path):
    """キャッシュCSVがあればネットワークに触れず読める"""
    end = pd.Timestamp.now().floor("h")
    idx = pd.date_range(end - pd.Timedelta(days=10), end, freq="1h")
    df = pd.DataFrame({
        "timestamp": idx,
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.5, "volume": 10.0,
    })
    (tmp_path / "BTCUSDT_1h.csv").write_text(df.to_csv(index=False))

    src = BitgetSource(["BTCUSDT"], days=8, cache_dir=str(tmp_path))
    src._fetch_candles = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("network should not be called"))
    out = src.load_klines("BTCUSDT", "1h")
    assert len(out) == len(idx)
    assert list(out.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_factory_registers_bitget(tmp_path):
    src = create_source("bitget", ["BTCUSDT"], days=5, cache_dir=str(tmp_path))
    assert isinstance(src, BitgetSource)


def test_load_funding_reads_cache(tmp_path):
    fund = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="8h"),
        "funding_rate": [0.0001] * 5,
    })
    fund.to_csv(tmp_path / "BTCUSDT_funding.csv", index=False)

    src = BitgetSource(["BTCUSDT"], days=1, cache_dir=str(tmp_path))
    out = src.load_funding("BTCUSDT")
    assert len(out) == 5
    assert "funding_rate" in out.columns


def test_load_orderflow_and_derivatives_default_empty(tmp_path):
    """Bitgetはオーダーフロー/デリバティブ非対応のため既定の空フレームを返す"""
    src = BitgetSource(["BTCUSDT"], days=1, cache_dir=str(tmp_path))
    assert src.load_orderflow("BTCUSDT").empty
    assert src.load_derivatives("BTCUSDT").empty


def test_fetch_candles_paginates_until_empty_response(monkeypatch, tmp_path):
    """90日/1000本の壁を超える範囲は複数回叩き、空応答が来たら停止する"""
    import mars_lite.data.sources as sources_mod

    calls = []

    def fake_get(path, params, max_retries=6):
        calls.append(params)
        end_ms = int(params["endTime"])
        # 2回目までは1件返し、3回目以降は「これ以上遡れない」として空を返す
        if len(calls) <= 2:
            t = end_ms
            return [[str(t), "100", "101", "99", "100.5", "1.0", "100.0"]]
        return []

    monkeypatch.setattr(sources_mod, "_bitget_get", fake_get)

    src = BitgetSource(["BTCUSDT"], days=200, cache_dir=str(tmp_path))
    df = src._fetch_candles("BTCUSDT", "1H", step_ms=3_600_000)
    assert len(calls) >= 2
    assert not df.empty
