"""binance_vision metrics 取得のテスト"""

import io
import zipfile
from datetime import datetime, timezone

import pandas as pd
import pytest

from mars_lite.data.binance_vision import (
    download_metrics_day,
    fetch_metrics_range,
    metrics_zip_url,
    normalize_metrics_df,
)

SAMPLE_CSV = """create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio
2024-06-01 00:05:00,BTCUSDT,76537.751,5180097949.0,2.15,1.58,2.17,1.66
2024-06-01 00:10:00,BTCUSDT,76513.982,5175528164.0,2.15,1.58,2.17,0.89
"""


def test_metrics_zip_url():
    day = datetime(2024, 6, 1, tzinfo=timezone.utc)
    url = metrics_zip_url("BTCUSDT", day)
    assert url.endswith("BTCUSDT-metrics-2024-06-01.zip")


def test_normalize_metrics_df():
    raw = pd.read_csv(io.StringIO(SAMPLE_CSV))
    out = normalize_metrics_df(raw)
    assert list(out.columns) == [
        "timestamp",
        "open_interest",
        "ls_ratio",
        "liq_notional",
        "funding_predicted",
    ]
    assert len(out) == 2
    assert out["open_interest"].iloc[0] == 76537.751
    assert out["ls_ratio"].iloc[0] == 2.17
    assert out["liq_notional"].iloc[0] == pytest.approx(0.66)


def test_download_metrics_day_parses_zip(monkeypatch):
    """ZIPバイト列をモックしてパース経路を検証"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("BTCUSDT-metrics-2024-06-01.csv", SAMPLE_CSV)
    payload = buf.getvalue()

    class FakeResp:
        status_code = 200
        content = payload

        def raise_for_status(self):
            pass

    class FakeSession:
        def get(self, url, timeout=60):
            return FakeResp()

    day = datetime(2024, 6, 1, tzinfo=timezone.utc)
    out = download_metrics_day("BTCUSDT", day, session=FakeSession())
    assert len(out) == 2


def test_fetch_metrics_range_integration():
    """vision から固定日付3日分を実取得（ネットワーク依存）"""
    import requests

    start_ms = int(datetime(2024, 6, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime(2024, 6, 4, tzinfo=timezone.utc).timestamp() * 1000)

    try:
        out = fetch_metrics_range("BTCUSDT", start_ms, end_ms, pause_sec=0.02)
    except requests.RequestException:
        return  # オフラインCIではスキップ

    assert len(out) > 100  # 3日×288本/日 程度
    assert out["timestamp"].is_monotonic_increasing
