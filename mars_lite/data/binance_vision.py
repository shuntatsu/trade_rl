"""
Binance Data Collection (data.binance.vision) から UM先物 metrics を取得

日次ZIPに5分足の OI / L-S比率 / taker量比が入る（RESTデリバAPIの30日制限を回避）。
https://data.binance.vision/data/futures/um/daily/metrics/{SYMBOL}/
"""

from __future__ import annotations

import io
import time
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

VISION_BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
KLINES_DAILY_BASE = "https://data.binance.vision/data/futures/um/daily/klines"
AGGTRADES_DAILY_BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"
FUNDING_MONTHLY_BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate"

FUNDING_COLUMNS = ["timestamp", "funding_rate"]

DERIV_COLUMNS = [
    "timestamp", "open_interest", "ls_ratio", "liq_notional", "funding_predicted",
]

KLINES_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def metrics_zip_url(symbol: str, day: datetime) -> str:
    d = day.strftime("%Y-%m-%d")
    return f"{VISION_BASE}/{symbol}/{symbol}-metrics-{d}.zip"


def normalize_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    """vision CSV → fetch_futures / CsvSource 互換形式（timestamp=epoch ms）"""
    if df is None or df.empty:
        return pd.DataFrame(columns=DERIV_COLUMNS)

    ts = pd.to_datetime(df["create_time"], utc=True)
    ts_ms = (ts.astype("int64") // 1000).astype("int64")
    taker = pd.to_numeric(df["sum_taker_long_short_vol_ratio"], errors="coerce").fillna(1.0)
    return pd.DataFrame({
        "timestamp": ts_ms,
        "open_interest": pd.to_numeric(df["sum_open_interest"], errors="coerce").fillna(0.0),
        "ls_ratio": pd.to_numeric(df["count_long_short_ratio"], errors="coerce").fillna(1.0),
        "liq_notional": (taker - 1.0).abs(),
        "funding_predicted": 0.0001,
    })


def download_metrics_day(
    symbol: str,
    day: datetime,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """1日分の metrics ZIP を取得。無い日は空DataFrame。"""
    url = metrics_zip_url(symbol, day)
    sess = session or requests
    try:
        resp = sess.get(url, timeout=60)
        if resp.status_code == 404:
            return pd.DataFrame(columns=DERIV_COLUMNS)
        resp.raise_for_status()
    except requests.RequestException:
        return pd.DataFrame(columns=DERIV_COLUMNS)

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        raw = pd.read_csv(zf.open(zf.namelist()[0]))
    return normalize_metrics_df(raw)


def fetch_metrics_range(
    symbol: str,
    start_ms: int,
    end_ms: int,
    pause_sec: float = 0.05,
    exclude_days: Optional[set] = None,
    save_cb=None,
) -> pd.DataFrame:
    """
    期間内の metrics を日次ZIPから結合（5分足、timestamp=epoch ms）。

    当日分は翌日朝以降に公開されるため、直近1日は欠けることがある。
    """
    start_day = datetime.fromtimestamp(
        start_ms / 1000, tz=timezone.utc,
    ).replace(hour=0, minute=0, second=0, microsecond=0)
    end_day = datetime.fromtimestamp(
        end_ms / 1000, tz=timezone.utc,
    ).replace(hour=0, minute=0, second=0, microsecond=0)

    parts = []
    session = requests.Session()
    day = start_day
    while day <= end_day:
        if exclude_days and day in exclude_days:
            day += timedelta(days=1)
            continue
        chunk = download_metrics_day(symbol, day, session)
        if not chunk.empty:
            if save_cb:
                save_cb(chunk)
            else:
                parts.append(chunk)
        day += timedelta(days=1)
        if pause_sec > 0:
            time.sleep(pause_sec)

    if save_cb:
        return pd.DataFrame(columns=DERIV_COLUMNS)

    if not parts:
        return pd.DataFrame(columns=DERIV_COLUMNS)

    out = pd.concat(parts, ignore_index=True)
    out = out[(out["timestamp"] >= start_ms) & (out["timestamp"] <= end_ms)]
    return (
        out.drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def klines_daily_url(symbol: str, interval: str, day: datetime) -> str:
    d = day.strftime("%Y-%m-%d")
    return f"{KLINES_DAILY_BASE}/{symbol}/{interval}/{symbol}-{interval}-{d}.zip"


def normalize_klines_df(df: pd.DataFrame) -> pd.DataFrame:
    """vision kline CSV → 標準OHLCV（timestamp=epoch ms）"""
    if df is None or df.empty:
        return pd.DataFrame(columns=KLINES_COLUMNS)
    return pd.DataFrame({
        "timestamp": df["open_time"].astype("int64"),
        "open": pd.to_numeric(df["open"], errors="coerce"),
        "high": pd.to_numeric(df["high"], errors="coerce"),
        "low": pd.to_numeric(df["low"], errors="coerce"),
        "close": pd.to_numeric(df["close"], errors="coerce"),
        "volume": pd.to_numeric(df["volume"], errors="coerce"),
    }).dropna(subset=["open", "close"])


def download_klines_day(
    symbol: str,
    interval: str,
    day: datetime,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """1日分の kline ZIP を取得。無い日は空。"""
    url = klines_daily_url(symbol, interval, day)
    sess = session or requests
    try:
        resp = sess.get(url, timeout=90)
        if resp.status_code == 404:
            return pd.DataFrame(columns=KLINES_COLUMNS)
        resp.raise_for_status()
    except requests.RequestException:
        return pd.DataFrame(columns=KLINES_COLUMNS)

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        raw = pd.read_csv(zf.open(zf.namelist()[0]))
    return normalize_klines_df(raw)


def fetch_klines_range(
    symbol: str,
    start_ms: int,
    end_ms: int,
    interval: str = "1m",
    pause_sec: float = 0.02,
    progress_cb=None,
    exclude_days: Optional[set] = None,
    save_cb=None,
) -> pd.DataFrame:
    """期間内の1m（等）足を日次ZIPから結合。"""
    start_day = datetime.fromtimestamp(
        start_ms / 1000, tz=timezone.utc,
    ).replace(hour=0, minute=0, second=0, microsecond=0)
    end_day = datetime.fromtimestamp(
        end_ms / 1000, tz=timezone.utc,
    ).replace(hour=0, minute=0, second=0, microsecond=0)

    parts = []
    session = requests.Session()
    day = start_day
    n_days = (end_day - start_day).days + 1
    idx = 0
    while day <= end_day:
        if exclude_days and day in exclude_days:
            idx += 1
            if progress_cb:
                progress_cb(idx, n_days, day, -1)
            day += timedelta(days=1)
            continue

        chunk = download_klines_day(symbol, interval, day, session)
        if not chunk.empty:
            if save_cb:
                save_cb(chunk)
            else:
                parts.append(chunk)
        idx += 1
        if progress_cb:
            progress_cb(idx, n_days, day, len(chunk))
        day += timedelta(days=1)
        if pause_sec > 0:
            time.sleep(pause_sec)

    if save_cb:
        return pd.DataFrame(columns=KLINES_COLUMNS)

    if not parts:
        return pd.DataFrame(columns=KLINES_COLUMNS)

    out = pd.concat(parts, ignore_index=True)
    out = out[(out["timestamp"] >= start_ms) & (out["timestamp"] < end_ms + 86_400_000)]
    return (
        out.drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

def funding_monthly_url(symbol: str, month_start: datetime) -> str:
    m = month_start.strftime("%Y-%m")
    return f"{FUNDING_MONTHLY_BASE}/{symbol}/{symbol}-fundingRate-{m}.zip"


def normalize_funding_df(df: pd.DataFrame) -> pd.DataFrame:
    """vision fundingRate CSV(calc_time, funding_interval_hours, last_funding_rate)
    → CsvSource互換（timestamp=epoch ms, funding_rate）"""
    if df is None or df.empty:
        return pd.DataFrame(columns=FUNDING_COLUMNS)
    return pd.DataFrame({
        "timestamp": pd.to_numeric(df["calc_time"], errors="coerce").astype("int64"),
        "funding_rate": pd.to_numeric(df["last_funding_rate"], errors="coerce"),
    }).dropna(subset=["funding_rate"])


def download_funding_month(
    symbol: str,
    month_start: datetime,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """1ヶ月分の fundingRate ZIP を取得。無い月は空。"""
    url = funding_monthly_url(symbol, month_start)
    sess = session or requests
    try:
        resp = sess.get(url, timeout=60)
        if resp.status_code == 404:
            return pd.DataFrame(columns=FUNDING_COLUMNS)
        resp.raise_for_status()
    except requests.RequestException:
        return pd.DataFrame(columns=FUNDING_COLUMNS)

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        raw = pd.read_csv(zf.open(zf.namelist()[0]))
    return normalize_funding_df(raw)


def fetch_funding_range(
    symbol: str,
    start_ms: int,
    end_ms: int,
    pause_sec: float = 0.1,
) -> pd.DataFrame:
    """
    期間内のfunding rateを月次ZIPから結合（8時間毎、timestamp=epoch ms）。

    REST /fapi/v1/fundingRate と異なり地域ブロックの影響を受けない
    （data.binance.vision は静的ファイル配信）。当月分は未公開のことがある。
    """
    start_month = datetime.fromtimestamp(
        start_ms / 1000, tz=timezone.utc,
    ).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_month = datetime.fromtimestamp(
        end_ms / 1000, tz=timezone.utc,
    ).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    parts = []
    session = requests.Session()
    month = start_month
    while month <= end_month:
        chunk = download_funding_month(symbol, month, session)
        if not chunk.empty:
            parts.append(chunk)
        if month.month == 12:
            month = month.replace(year=month.year + 1, month=1)
        else:
            month = month.replace(month=month.month + 1)
        if pause_sec > 0:
            time.sleep(pause_sec)

    if not parts:
        return pd.DataFrame(columns=FUNDING_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    out = out[(out["timestamp"] >= start_ms) & (out["timestamp"] <= end_ms)]
    return (
        out.drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def aggtrades_daily_url(symbol: str, day: datetime) -> str:
    d = day.strftime("%Y-%m-%d")
    return f"{AGGTRADES_DAILY_BASE}/{symbol}/{symbol}-aggTrades-{d}.zip"

def normalize_aggtrades_to_orderflow(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "buy_volume", "sell_volume", "trade_count", "avg_trade_size", "volume_imbalance"])
    
    if "transact_time" not in df.columns:
        df.columns = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"]

    df["transact_time"] = pd.to_numeric(df["transact_time"], errors="coerce")
    df["minute"] = (df["transact_time"] // 60000) * 60000
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["is_sell"] = df["is_buyer_maker"].astype(bool)

    gb = df.groupby("minute")
    buy_vol = df[~df["is_sell"]].groupby("minute")["quantity"].sum()
    sell_vol = df[df["is_sell"]].groupby("minute")["quantity"].sum()
    trade_count = gb.size()

    out = pd.DataFrame({
        "timestamp": list(gb.groups.keys()),
    })
    out["buy_volume"] = out["timestamp"].map(buy_vol).fillna(0.0)
    out["sell_volume"] = out["timestamp"].map(sell_vol).fillna(0.0)
    out["trade_count"] = out["timestamp"].map(trade_count).fillna(0)

    out["avg_trade_size"] = (out["buy_volume"] + out["sell_volume"]) / out["trade_count"].replace(0, 1)
    
    vol_sum = out["buy_volume"] + out["sell_volume"]
    out["volume_imbalance"] = (out["buy_volume"] - out["sell_volume"]) / vol_sum.replace(0, 1)
    
    return out.sort_values("timestamp").reset_index(drop=True)

def download_orderflow_day(
    symbol: str,
    day: datetime,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    url = aggtrades_daily_url(symbol, day)
    sess = session or requests
    try:
        resp = sess.get(url, timeout=120)
        if resp.status_code == 404:
            return pd.DataFrame()
        resp.raise_for_status()
    except requests.RequestException:
        return pd.DataFrame()

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            raw = pd.read_csv(zf.open(zf.namelist()[0]))
        return normalize_aggtrades_to_orderflow(raw)
    except Exception:
        return pd.DataFrame()

def fetch_orderflow_vision(
    symbol: str,
    start_ms: int,
    end_ms: int,
    pause_sec: float = 0.05,
    progress_cb=None,
    exclude_days: Optional[set] = None,
    save_cb=None,
) -> pd.DataFrame:
    start_day = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    end_day = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    parts = []
    session = requests.Session()
    day = start_day
    n_days = (end_day - start_day).days + 1
    idx = 0

    while day <= end_day:
        if exclude_days and day in exclude_days:
            idx += 1
            if progress_cb: progress_cb(idx, n_days, day, -1)
            day += timedelta(days=1)
            continue

        chunk = download_orderflow_day(symbol, day, session)
        if not chunk.empty:
            if save_cb:
                save_cb(chunk)
            else:
                parts.append(chunk)
        
        idx += 1
        if progress_cb: progress_cb(idx, n_days, day, len(chunk))
        day += timedelta(days=1)
        if pause_sec > 0:
            time.sleep(pause_sec)

    if save_cb:
        return pd.DataFrame()
    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out = out[(out["timestamp"] >= start_ms) & (out["timestamp"] < end_ms + 86_400_000)]
    return out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
