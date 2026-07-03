"""
上場日検出のテスト（Binance APIへの疎通が必要）

APIに到達できない環境（オフライン・地域制限など）ではskipする。
"""

import sys
import os

import pytest
import requests

sys.path.append(os.getcwd())

from scripts.fetch_binance import detect_listing_date


def _binance_reachable() -> bool:
    try:
        resp = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


@pytest.mark.skipif(not _binance_reachable(), reason="Binance API is not reachable")
def test_detect_listing_date():
    symbol = "WLDUSDT"
    date = detect_listing_date(symbol)
    assert date is not None
    # WLDUSDTは2023年7月上場
    assert str(date) >= "2023-01-01"
