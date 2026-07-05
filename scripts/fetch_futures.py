"""
Binance Futures データ取得スクリプト（認証不要の公開REST）

取得対象:
    1. 先物1分足kline      GET /fapi/v1/klines
    2. funding rate実績    GET /fapi/v1/fundingRate（8時間毎）
    3. aggTrades           GET /fapi/v1/aggTrades → 1分オーダーフロー集計に変換
       （buy_volume / sell_volume / trade_count / avg_trade_size / volume_imbalance）

出力先（--to で選択、複数可）:
    csv      : data/{SYMBOL}/1m/ , data/{SYMBOL}/orderflow_1m/ , data/{SYMBOL}/funding/
    postgres : Trade Platform と同居する rl_ プレフィックステーブル
               （rl_funding_rate / rl_orderflow_1m。無ければ自動CREATE。
                 接続は --dsn または環境変数 PLATFORM_DB_URL）

使い方（ローカルPCで実行。Binanceは一部地域からブロックされるため注意）:
    python scripts/fetch_futures.py --symbols BTCUSDT ETHUSDT --days 180 --to csv
    python scripts/fetch_futures.py --symbols BTCUSDT --days 30 --to csv postgres
"""

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FAPI = "https://fapi.binance.com"
DEFAULT_SYMBOLS = [
    "BTCUSDT", "XRPUSDT", "SUIUSDT", "BNBUSDT", "ETHUSDT", "PAXGUSDT",
]  # ETHBTC は先物に無いためデフォルトから除外


def _get(path: str, params: dict) -> list:
    for attempt in range(5):
        try:
            resp = requests.get(f"{FAPI}{path}", params=params, timeout=15)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    return []


def fetch_klines_range(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """1分足klineを期間指定で取得（1000本ずつページング）"""
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        data = _get("/fapi/v1/klines", {
            "symbol": symbol, "interval": "1m",
            "startTime": cursor, "endTime": end_ms, "limit": 1000,
        })
        if not data:
            break
        rows.extend(data)
        cursor = data[-1][0] + 60_000
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df["timestamp"] = df["timestamp"].astype("int64")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df.drop_duplicates("timestamp").reset_index(drop=True)


def fetch_funding_range(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """funding rate実績を取得（1000件ずつページング）"""
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        data = _get("/fapi/v1/fundingRate", {
            "symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000,
        })
        if not data:
            break
        rows.extend(data)
        cursor = int(data[-1]["fundingTime"]) + 1
        time.sleep(0.2)
        if len(data) < 1000:
            break

    if not rows:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    df = pd.DataFrame(rows)
    return pd.DataFrame({
        "timestamp": df["fundingTime"].astype("int64"),
        "funding_rate": df["fundingRate"].astype(float),
    }).drop_duplicates("timestamp").reset_index(drop=True)


def fetch_derivatives(symbol: str, start_ms: int, end_ms: int, period: str = "1h") -> pd.DataFrame:
    """
    デリバティブ指標を取得して1h集計で返す:
      open_interest      GET /futures/data/openInterestHist
      ls_ratio           GET /futures/data/globalLongShortAccountRatio
      liq_notional       aggregated from /fapi/v1/allForceOrders は非推奨のため
                         takerlongshortRatio の非対称を代理指標に使う
      funding_predicted  GET /fapi/v1/premiumIndex（現在のlastFundingRate）

    Binanceのdata系エンドポイントは最大30日・limit500の制約があるため、
    30日窓でページングする。
    """
    def paged(path, extra):
        rows = []
        cursor = start_ms
        window = 30 * 86_400_000
        while cursor < end_ms:
            data = _get(path, {"symbol": symbol, "period": period,
                               "startTime": cursor, "endTime": min(cursor + window, end_ms),
                               "limit": 500, **extra})
            if not data:
                cursor += window
                continue
            rows.extend(data)
            cursor += window
            time.sleep(0.2)
        return rows

    oi = paged("/futures/data/openInterestHist", {})
    ls = paged("/futures/data/globalLongShortAccountRatio", {})
    taker = paged("/futures/data/takerlongshortRatio", {})

    def to_series(rows, tcol, vcol, cast=float):
        if not rows:
            return pd.DataFrame(columns=["timestamp", vcol])
        df = pd.DataFrame(rows)
        return pd.DataFrame({"timestamp": df[tcol].astype("int64"),
                             vcol: df[vcol].astype(cast)}).drop_duplicates("timestamp")

    oi_df = to_series(oi, "timestamp", "sumOpenInterest") if oi else pd.DataFrame(columns=["timestamp", "sumOpenInterest"])
    ls_df = to_series(ls, "timestamp", "longShortRatio") if ls else pd.DataFrame(columns=["timestamp", "longShortRatio"])
    tk_df = to_series(taker, "timestamp", "buySellRatio") if taker else pd.DataFrame(columns=["timestamp", "buySellRatio"])

    out = oi_df.rename(columns={"sumOpenInterest": "open_interest"})
    if not ls_df.empty:
        out = out.merge(ls_df.rename(columns={"longShortRatio": "ls_ratio"}), on="timestamp", how="outer")
    else:
        out["ls_ratio"] = 1.0
    # 清算の代理: takerのbuy/sell非対称の絶対値（大きいほど強制的フロー）
    if not tk_df.empty:
        tk_df["liq_notional"] = (tk_df["buySellRatio"] - 1.0).abs()
        out = out.merge(tk_df[["timestamp", "liq_notional"]], on="timestamp", how="outer")
    else:
        out["liq_notional"] = 0.0
    out["funding_predicted"] = 0.0001  # premiumIndexは別途取得可（簡易化）
    return out.sort_values("timestamp").fillna(method="ffill").fillna(0.0).reset_index(drop=True)


def fetch_orderflow_day(symbol: str, day_start_ms: int) -> pd.DataFrame:
    """
    aggTradesを1日分取得し、1分オーダーフロー集計に変換

    aggTradesの m フラグ: True = 買い手がmaker（=売り主導の約定）
    """
    end_ms = day_start_ms + 86_400_000
    rows = []
    cursor = day_start_ms
    while cursor < end_ms:
        data = _get("/fapi/v1/aggTrades", {
            "symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000,
        })
        if not data:
            break
        rows.extend(data)
        last_t = int(data[-1]["T"])
        if last_t <= cursor:
            break
        cursor = last_t + 1
        time.sleep(0.05)
        if len(data) < 1000 and cursor >= end_ms:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["qty"] = df["q"].astype(float)
    df["minute"] = (df["T"].astype("int64") // 60_000) * 60_000
    df["is_sell"] = df["m"].astype(bool)  # m=True → 売り主導

    grouped = df.groupby("minute").apply(
        lambda g: pd.Series({
            "buy_volume": g.loc[~g["is_sell"], "qty"].sum(),
            "sell_volume": g.loc[g["is_sell"], "qty"].sum(),
            "trade_count": len(g),
        }),
        include_groups=False,
    ).reset_index().rename(columns={"minute": "timestamp"})

    total = grouped["buy_volume"] + grouped["sell_volume"]
    grouped["avg_trade_size"] = total / grouped["trade_count"].clip(lower=1)
    grouped["volume_imbalance"] = np.where(
        total > 0, (grouped["buy_volume"] - grouped["sell_volume"]) / total, 0.0
    )
    return grouped


# ---- 保存先 ----

def save_csv_daily(df: pd.DataFrame, out_dir: Path, day: datetime) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{day.strftime('%Y-%m-%d')}.csv", index=False)


def upsert_postgres(dsn: str, symbol: str, funding: pd.DataFrame,
                    orderflow: pd.DataFrame, derivatives: pd.DataFrame = None):
    """rl_ テーブルへUPSERT（テーブルが無ければ作成）"""
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rl_derivatives (
                symbol TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                open_interest DOUBLE PRECISION,
                ls_ratio DOUBLE PRECISION,
                liq_notional DOUBLE PRECISION,
                funding_predicted DOUBLE PRECISION,
                PRIMARY KEY (symbol, timestamp)
            )""")
        if derivatives is not None:
            for _, r in derivatives.iterrows():
                cur.execute(
                    """INSERT INTO rl_derivatives VALUES (%s, to_timestamp(%s/1000.0), %s, %s, %s, %s)
                       ON CONFLICT (symbol, timestamp) DO UPDATE SET
                         open_interest = EXCLUDED.open_interest, ls_ratio = EXCLUDED.ls_ratio,
                         liq_notional = EXCLUDED.liq_notional, funding_predicted = EXCLUDED.funding_predicted""",
                    (symbol, int(r["timestamp"]), float(r["open_interest"]),
                     float(r["ls_ratio"]), float(r["liq_notional"]),
                     float(r["funding_predicted"])),
                )
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rl_funding_rate (
                symbol TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                funding_rate DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            )""")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rl_orderflow_1m (
                symbol TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                buy_volume DOUBLE PRECISION,
                sell_volume DOUBLE PRECISION,
                trade_count INTEGER,
                avg_trade_size DOUBLE PRECISION,
                volume_imbalance DOUBLE PRECISION,
                PRIMARY KEY (symbol, timestamp)
            )""")
        for _, r in funding.iterrows():
            cur.execute(
                """INSERT INTO rl_funding_rate VALUES (%s, to_timestamp(%s/1000.0), %s)
                   ON CONFLICT (symbol, timestamp) DO UPDATE SET funding_rate = EXCLUDED.funding_rate""",
                (symbol, int(r["timestamp"]), float(r["funding_rate"])),
            )
        for _, r in orderflow.iterrows():
            cur.execute(
                """INSERT INTO rl_orderflow_1m VALUES (%s, to_timestamp(%s/1000.0), %s, %s, %s, %s, %s)
                   ON CONFLICT (symbol, timestamp) DO UPDATE SET
                     buy_volume = EXCLUDED.buy_volume, sell_volume = EXCLUDED.sell_volume,
                     trade_count = EXCLUDED.trade_count, avg_trade_size = EXCLUDED.avg_trade_size,
                     volume_imbalance = EXCLUDED.volume_imbalance""",
                (symbol, int(r["timestamp"]), float(r["buy_volume"]),
                 float(r["sell_volume"]), int(r["trade_count"]),
                 float(r["avg_trade_size"]), float(r["volume_imbalance"])),
            )
        conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Binance Futures データ取得")
    parser.add_argument("--symbols", type=str, nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--days", type=int, default=90, help="遡及日数")
    parser.add_argument("--output", type=str, default="./data")
    parser.add_argument("--to", type=str, nargs="+", default=["csv"],
                        choices=["csv", "postgres"])
    parser.add_argument("--dsn", type=str, default=None,
                        help="PostgreSQL DSN（省略時は環境変数 PLATFORM_DB_URL）")
    parser.add_argument("--skip-orderflow", action="store_true",
                        help="aggTrades集計をスキップ（取得が重いため）")
    args = parser.parse_args()

    dsn = args.dsn or os.environ.get("PLATFORM_DB_URL")
    if "postgres" in args.to and not dsn:
        parser.error("--to postgres には --dsn か PLATFORM_DB_URL が必要です")

    output_dir = Path(args.output)
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = now - timedelta(days=args.days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    for symbol in args.symbols:
        print(f"=== {symbol} ===")

        print(f"  funding rate ({args.days}d)...")
        funding = fetch_funding_range(symbol, start_ms, end_ms)
        print(f"    {len(funding)} events")
        if "csv" in args.to and len(funding):
            fdir = output_dir / symbol / "funding"
            fdir.mkdir(parents=True, exist_ok=True)
            funding.to_csv(fdir / "funding.csv", index=False)

        print(f"  derivatives (OI/LS/liq)...")
        derivatives = fetch_derivatives(symbol, start_ms, end_ms)
        print(f"    {len(derivatives)} hourly rows")
        if "csv" in args.to and len(derivatives):
            ddir = output_dir / symbol / "derivatives"
            ddir.mkdir(parents=True, exist_ok=True)
            derivatives.to_csv(ddir / "derivatives.csv", index=False)

        all_orderflow = []
        for d in range(args.days):
            day = start + timedelta(days=d)
            day_ms = int(day.timestamp() * 1000)

            klines = fetch_klines_range(symbol, day_ms, min(day_ms + 86_400_000, end_ms))
            if len(klines) and "csv" in args.to:
                save_csv_daily(klines, output_dir / symbol / "1m", day)

            if not args.skip_orderflow:
                of = fetch_orderflow_day(symbol, day_ms)
                if len(of):
                    if "csv" in args.to:
                        save_csv_daily(of, output_dir / symbol / "orderflow_1m", day)
                    all_orderflow.append(of)

            print(f"  [{d + 1}/{args.days}] {day.strftime('%Y-%m-%d')}: "
                  f"klines={len(klines)}", flush=True)

        if "postgres" in args.to:
            of_all = pd.concat(all_orderflow, ignore_index=True) if all_orderflow \
                else pd.DataFrame(columns=["timestamp", "buy_volume", "sell_volume",
                                           "trade_count", "avg_trade_size", "volume_imbalance"])
            print(f"  postgres upsert: funding={len(funding)}, orderflow={len(of_all)}, "
                  f"derivatives={len(derivatives)}")
            upsert_postgres(dsn, symbol, funding, of_all, derivatives)

    print("Done.")


if __name__ == "__main__":
    main()
