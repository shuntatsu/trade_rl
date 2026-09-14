from __future__ import annotations

from pathlib import Path

path = Path("tests/integrations/test_binance_index_price_source.py")
text = path.read_text(encoding="utf-8")
marker = "def test_transport_rejects_row_outside_archive_calendar_month"
if marker in text:
    raise RuntimeError("month-boundary test already exists")
text += '''\n\ndef test_transport_rejects_row_outside_archive_calendar_month(\n    monkeypatch: pytest.MonkeyPatch,\n) -> None:\n    url = vision_monthly_index_price_kline_url(\n        market=BinanceMarket.USDS_M,\n        symbol="BTCUSDT",\n        interval="1h",\n        month="2022-01",\n    )\n    rows = [\n        _row(datetime(2022, 1, 31, 23, tzinfo=UTC), close=100.0),\n        _row(datetime(2022, 2, 1, 0, tzinfo=UTC), close=101.0),\n    ]\n    payload = _zip_bytes("BTCUSDT-1h-2022-01.csv", _csv_bytes(rows))\n    digest = hashlib.sha256(payload).hexdigest()\n    checksum = f"{digest}  BTCUSDT-1h-2022-01.zip\\n".encode()\n    transport = BinancePublicTransport(max_attempts=1)\n\n    def request_bytes(request_url: str) -> bytes:\n        if request_url == url:\n            return payload\n        if request_url == url + ".CHECKSUM":\n            return checksum\n        raise AssertionError(request_url)\n\n    monkeypatch.setattr(transport, "_request_bytes", request_bytes)\n    with pytest.raises(BinanceTransportError, match="month|boundary|archive"):\n        transport.load_index_price_klines(\n            market=BinanceMarket.USDS_M,\n            symbol="BTCUSDT",\n            interval="1h",\n            start_ms=_ms(datetime(2022, 1, 1, tzinfo=UTC)),\n            end_ms=_ms(datetime(2022, 2, 1, tzinfo=UTC)),\n            mode=BinanceTransportMode.VISION,\n            expected_archive_sha256={url: digest},\n        )\n'''
path.write_text(text, encoding="utf-8")
