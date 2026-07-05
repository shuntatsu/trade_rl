"""
PostgresSource のテスト（DB到達可能時のみ実行、無ければスキップ）

CI/オフライン環境ではdocker-composeのPostgresが立っていないため、
接続失敗時は全テストをスキップする（binance_visionのオンライン
テストと同じ方針）。
"""

import pandas as pd
import pytest

from mars_lite.data.sources import PostgresSource, create_source


def _db_available(dsn: str) -> bool:
    try:
        import psycopg
        with psycopg.connect(dsn, connect_timeout=3):
            return True
    except Exception:
        return False


DSN = "postgresql://trade_rl:trade_rl@localhost:5433/trade_rl"
pytestmark = pytest.mark.skipif(not _db_available(DSN), reason="Postgres not reachable")


class TestPostgresSource:

    def test_factory_registers_postgres(self):
        src = create_source("postgres", ["BTCUSDT"], dsn=DSN, source="hyperliquid")
        assert isinstance(src, PostgresSource)

    def test_load_klines_native_timeframe(self):
        """rl_klinesがtimeframe列そのものを保持している場合はそのまま返す"""
        src = PostgresSource(["BTCUSDT"], dsn=DSN, source="hyperliquid")
        df = src.load_klines("BTCUSDT", "1h")
        if df.empty:
            pytest.skip("no hyperliquid klines seeded in this DB")
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert df["timestamp"].is_monotonic_increasing

    def test_cross_source_derivatives(self):
        """klines/fundingとderivatives/orderflowで異なるsourceラベルを使える"""
        src = PostgresSource(
            ["BTCUSDT"], dsn=DSN, source="hyperliquid",
            derivatives_source="binance", orderflow_source="binance",
        )
        assert src.derivatives_source == "binance"
        assert src.orderflow_source == "binance"
        assert src.source == "hyperliquid"
        # 存在有無に関わらずクラッシュしないこと
        deriv = src.load_derivatives("BTCUSDT")
        assert list(deriv.columns) == ["timestamp", "open_interest", "ls_ratio", "liq_notional"]

    def test_missing_symbol_returns_empty_not_error(self):
        src = PostgresSource(["NOSUCHSYM"], dsn=DSN, source="hyperliquid")
        of = src.load_orderflow("NOSUCHSYM")
        assert of.empty
        fund = src.load_funding("NOSUCHSYM")
        assert fund.empty
