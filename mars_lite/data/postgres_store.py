"""
PostgreSQL への rl_ テーブル UPSERT（fetch スクリプト共通）
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def ensure_schema(cur) -> None:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rl_klines (
            source TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume DOUBLE PRECISION,
            PRIMARY KEY (source, symbol, timeframe, timestamp)
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rl_derivatives (
            source TEXT NOT NULL DEFAULT 'binance',
            symbol TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            open_interest DOUBLE PRECISION,
            ls_ratio DOUBLE PRECISION,
            liq_notional DOUBLE PRECISION,
            funding_predicted DOUBLE PRECISION,
            PRIMARY KEY (source, symbol, timestamp)
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rl_funding_rate (
            source TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            funding_rate DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (source, symbol, timestamp)
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rl_orderflow_1m (
            source TEXT NOT NULL DEFAULT 'binance',
            symbol TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            buy_volume DOUBLE PRECISION,
            sell_volume DOUBLE PRECISION,
            trade_count INTEGER,
            avg_trade_size DOUBLE PRECISION,
            volume_imbalance DOUBLE PRECISION,
            PRIMARY KEY (source, symbol, timestamp)
        )""")


def _ts_ms(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series.astype("int64"), unit="ms", utc=True)
    return pd.to_datetime(series, utc=True)


def upsert_klines(
    dsn: str,
    source: str,
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    batch_size: int = 5000,
) -> int:
    if df is None or df.empty:
        return 0
    import psycopg

    out = df.copy()
    out["timestamp"] = _ts_ms(out["timestamp"])
    rows = [
        (
            source, symbol, timeframe, r["timestamp"].to_pydatetime(),
            float(r["open"]), float(r["high"]), float(r["low"]),
            float(r["close"]), float(r["volume"]),
        )
        for _, r in out.iterrows()
    ]
    sql = """
        INSERT INTO rl_klines VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (source, symbol, timeframe, timestamp) DO UPDATE SET
          open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
          close=EXCLUDED.close, volume=EXCLUDED.volume
    """
    total = 0
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        ensure_schema(cur)
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            cur.executemany(sql, batch)
            total += len(batch)
        conn.commit()
    return total


def upsert_funding(dsn: str, source: str, symbol: str, funding: pd.DataFrame) -> int:
    if funding is None or funding.empty:
        return 0
    import psycopg

    rows = [
        (source, symbol, _ts_ms(pd.Series([r["timestamp"]]))[0].to_pydatetime(),
         float(r["funding_rate"]))
        for _, r in funding.iterrows()
    ]
    sql = """
        INSERT INTO rl_funding_rate VALUES (%s,%s,%s,%s)
        ON CONFLICT (source, symbol, timestamp) DO UPDATE SET
          funding_rate = EXCLUDED.funding_rate
    """
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        ensure_schema(cur)
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


def upsert_orderflow(dsn: str, source: str, symbol: str, orderflow: pd.DataFrame) -> int:
    if orderflow is None or orderflow.empty:
        return 0
    import psycopg

    rows = [
        (
            source, symbol,
            _ts_ms(pd.Series([r["timestamp"]]))[0].to_pydatetime(),
            float(r["buy_volume"]), float(r["sell_volume"]), int(r["trade_count"]),
            float(r["avg_trade_size"]), float(r["volume_imbalance"]),
        )
        for _, r in orderflow.iterrows()
    ]
    sql = """
        INSERT INTO rl_orderflow_1m VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (source, symbol, timestamp) DO UPDATE SET
          buy_volume=EXCLUDED.buy_volume, sell_volume=EXCLUDED.sell_volume,
          trade_count=EXCLUDED.trade_count, avg_trade_size=EXCLUDED.avg_trade_size,
          volume_imbalance=EXCLUDED.volume_imbalance
    """
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        ensure_schema(cur)
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


def upsert_derivatives(
    dsn: str, source: str, symbol: str, derivatives: pd.DataFrame,
    batch_size: int = 5000,
) -> int:
    if derivatives is None or derivatives.empty:
        return 0
    import psycopg

    rows = [
        (
            source, symbol,
            _ts_ms(pd.Series([r["timestamp"]]))[0].to_pydatetime(),
            float(r["open_interest"]), float(r["ls_ratio"]),
            float(r["liq_notional"]), float(r.get("funding_predicted", 0.0)),
        )
        for _, r in derivatives.iterrows()
    ]
    sql = """
        INSERT INTO rl_derivatives VALUES (%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (source, symbol, timestamp) DO UPDATE SET
          open_interest=EXCLUDED.open_interest, ls_ratio=EXCLUDED.ls_ratio,
          liq_notional=EXCLUDED.liq_notional,
          funding_predicted=EXCLUDED.funding_predicted
    """
    total = 0
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        ensure_schema(cur)
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            cur.executemany(sql, batch)
            total += len(batch)
        conn.commit()
    return total


def upsert_binance_bundle(
    dsn: str,
    symbol: str,
    klines_1m: pd.DataFrame,
    funding: pd.DataFrame,
    orderflow: pd.DataFrame,
    derivatives: Optional[pd.DataFrame] = None,
) -> dict:
    """Binance fetch_futures 用の一括 UPSERT"""
    return {
        "klines": upsert_klines(dsn, "binance", symbol, "1m", klines_1m),
        "funding": upsert_funding(dsn, "binance", symbol, funding),
        "orderflow": upsert_orderflow(dsn, "binance", symbol, orderflow),
        "derivatives": upsert_derivatives(
            dsn, "binance", symbol,
            derivatives if derivatives is not None else pd.DataFrame(),
        ),
    }


def get_existing_kline_days(dsn: str, source: str, symbol: str, timeframe: str) -> set:
    """DBからすでに取得済みの日付（day単位）セットを取得。再ダウンロード防止用。"""
    if not dsn:
        return set()
    try:
        import psycopg
        from datetime import timezone
        sql = "SELECT DISTINCT date_trunc('day', timestamp) FROM rl_klines WHERE source=%s AND symbol=%s AND timeframe=%s"
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(sql, (source, symbol, timeframe))
            # 取得した日付をUTCのdatetimeとしてセットで返す
            return {row[0].replace(tzinfo=timezone.utc) for row in cur.fetchall()}
    except Exception:
        return set()


def get_existing_derivative_days(dsn: str, source: str, symbol: str) -> set:
    if not dsn:
        return set()
    try:
        import psycopg
        from datetime import timezone
        sql = "SELECT DISTINCT date_trunc('day', timestamp) FROM rl_derivatives WHERE source=%s AND symbol=%s"
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(sql, (source, symbol))
            return {row[0].replace(tzinfo=timezone.utc) for row in cur.fetchall()}
    except Exception as e:
        print(f"Warning: failed to fetch existing derivative dates: {e}")
        return set()

def get_existing_orderflow_days(dsn: str, source: str, symbol: str) -> set:
    if not dsn:
        return set()
    try:
        import psycopg
        from datetime import timezone
        sql = "SELECT DISTINCT date_trunc('day', timestamp) FROM rl_orderflow_1m WHERE source=%s AND symbol=%s"
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(sql, (source, symbol))
            return {row[0].replace(tzinfo=timezone.utc) for row in cur.fetchall()}
    except Exception as e:
        print(f"Warning: failed to fetch existing orderflow dates: {e}")
        return set()
