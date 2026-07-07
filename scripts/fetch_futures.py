"""
Binance Futures データ取得スクリプト（認証不要の公開REST）

取得対象:
    1. 先物1分足kline      GET /fapi/v1/klines
    2. funding rate実績    GET /fapi/v1/fundingRate（8時間毎）
    3. デリバティブ指標   data.binance.vision 日次 metrics ZIP（5分足・長期履歴）
       ＋直近ギャップは REST /futures/data/* で補完（30日制限あり）
    4. aggTrades           GET /fapi/v1/aggTrades → 1分オーダーフロー集計に変換
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
# Phase D: HL上場でBinance先物がある流動性上位15銘柄
# 既存7（BTC/ETH/SOL/XRP/BNB/SUI/DOGE）+ ADA/AVAX/LINK/LTC/BCH/APT/ARB/OP
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "SUIUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT",
    "LTCUSDT", "BCHUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
]


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


def _fetch_derivatives_rest(symbol: str, start_ms: int, end_ms: int, period: str = "1h") -> pd.DataFrame:
    """
    RESTデリバAPI（直近30日のみ）。visionで埋まらないギャップ補完用。
    """
    max_hist_ms = 30 * 86_400_000
    eff_start = max(start_ms, end_ms - max_hist_ms)

    def paged(path, extra):
        rows = []
        cursor = eff_start
        window = 30 * 86_400_000
        while cursor < end_ms:
            try:
                data = _get(path, {"symbol": symbol, "period": period,
                                   "startTime": cursor, "endTime": min(cursor + window, end_ms),
                                   "limit": 500, **extra})
            except requests.HTTPError:
                break
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
    return out.sort_values("timestamp").ffill().fillna(0.0).reset_index(drop=True)


def fetch_derivatives(symbol: str, start_ms: int, end_ms: int, period: str = "1h", exclude_days: set = None, save_cb=None) -> pd.DataFrame:
    """
    OI / L-S / taker量比を取得（vision優先、RESTで直近ギャップ補完）。

    主データ: data.binance.vision 日次 metrics（5分足・数年分）
    補完: REST openInterestHist 等（直近30日、当日分など vision 未公開分）
    """
    from mars_lite.data.binance_vision import fetch_metrics_range

    vision = fetch_metrics_range(symbol, start_ms, end_ms, exclude_days=exclude_days, save_cb=save_cb)
    rest = _fetch_derivatives_rest(symbol, start_ms, end_ms, period)

    if vision.empty:
        if save_cb and not rest.empty:
            save_cb(rest)
            return pd.DataFrame()
        return rest
    if rest.empty:
        return vision

    have = set(vision["timestamp"].tolist())
    extra = rest[~rest["timestamp"].isin(have)]
    if save_cb and not extra.empty:
        save_cb(extra)
        return pd.DataFrame()
    if extra.empty:
        return vision
    return (
        pd.concat([vision, extra], ignore_index=True)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


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
    parser.add_argument("--klines-source", choices=["vision", "rest"], default="vision",
                        help="1m足: vision=data.binance.vision日次ZIP（長期向け）, rest=REST API")
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
        
        existing_kline_days = set()
        existing_deriv_days = set()
        existing_orderflow_days = set()
        if "postgres" in args.to and dsn:
            from mars_lite.data.postgres_store import get_existing_kline_days, get_existing_derivative_days, get_existing_orderflow_days
            existing_kline_days = get_existing_kline_days(dsn, "binance", symbol, "1m")
            existing_deriv_days = get_existing_derivative_days(dsn, "binance", symbol)
            existing_orderflow_days = get_existing_orderflow_days(dsn, "binance", symbol)

        # Phase D: 上場日をログ表示（品質ゲート選定の判断材料）
        try:
            info = _get("/fapi/v1/exchangeInfo", {})
            sym_info = next((s for s in info.get("symbols", []) if s["symbol"] == symbol), None)
            if sym_info:
                listing_ms = sym_info.get("onboardDate") or sym_info.get("deliveryDate")
                if listing_ms:
                    listing_dt = datetime.fromtimestamp(listing_ms / 1000, tz=timezone.utc)
                    days_listed = (datetime.now(timezone.utc) - listing_dt).days
                    print(f"  listing: {listing_dt.strftime('%Y-%m-%d')} ({days_listed}d ago)")
        except Exception:
            pass  # 取得失敗は無視して続行

        print(f"  funding rate ({args.days}d)...")
        funding = fetch_funding_range(symbol, start_ms, end_ms)
        print(f"    {len(funding)} events")
        if "postgres" in args.to and dsn and not funding.empty:
            from mars_lite.data.postgres_store import upsert_funding
            upsert_funding(dsn, "binance", symbol, funding)
        if "csv" in args.to and len(funding):
            fdir = output_dir / symbol / "funding"
            fdir.mkdir(parents=True, exist_ok=True)
            funding.to_csv(fdir / "funding.csv", index=False)

        deriv_save_cb = None
        if "postgres" in args.to and dsn:
            from mars_lite.data.postgres_store import upsert_derivatives
            def _deriv_save_cb(df):
                if not df.empty:
                    upsert_derivatives(dsn, "binance", symbol, df)
            deriv_save_cb = _deriv_save_cb

        print(f"  derivatives (OI/LS/liq, vision+REST)...")
        derivatives = fetch_derivatives(symbol, start_ms, end_ms, exclude_days=existing_deriv_days, save_cb=deriv_save_cb)
        print(f"    {len(derivatives)} rows (5m native)")
        if "csv" in args.to and len(derivatives):
            ddir = output_dir / symbol / "derivatives"
            ddir.mkdir(parents=True, exist_ok=True)
            derivatives.to_csv(ddir / "derivatives.csv", index=False)

        all_klines = []
        all_orderflow = []

        if args.klines_source == "vision":
            from mars_lite.data.binance_vision import fetch_klines_range as fetch_klines_vision

            def _kl_progress(i, n, day, nrows):
                if i % 30 == 0 or i == n:
                    r_str = "SKIPPED" if nrows == -1 else str(nrows)
                    print(f"  [klines {i}/{n}] {day.strftime('%Y-%m-%d')}: "
                          f"rows={r_str}", flush=True)

            kline_save_cb = None
            if "postgres" in args.to and dsn:
                from mars_lite.data.postgres_store import upsert_klines
                def _kline_save_cb(df):
                    if not df.empty:
                        upsert_klines(dsn, "binance", symbol, "1m", df)
                kline_save_cb = _kline_save_cb

            print(f"  klines 1m ({args.days}d, vision)...")
            klines_all = fetch_klines_vision(
                symbol, start_ms, end_ms, interval="1m",
                pause_sec=0.02, progress_cb=_kl_progress,
                exclude_days=existing_kline_days,
                save_cb=kline_save_cb,
            )
            print(f"    {len(klines_all)} bars total")
            if len(klines_all) and "csv" in args.to:
                for d in range(args.days):
                    day = start + timedelta(days=d)
                    day_ms = int(day.timestamp() * 1000)
                    day_end_ms = day_ms + 86_400_000
                    sl = klines_all[
                        (klines_all["timestamp"] >= day_ms)
                        & (klines_all["timestamp"] < day_end_ms)
                    ]
                    if len(sl):
                        save_csv_daily(
                            sl.reset_index(drop=True),
                            output_dir / symbol / "1m", day,
                        )
            if "postgres" in args.to and len(klines_all):
                all_klines = [klines_all]
        else:
            for d in range(args.days):
                day = start + timedelta(days=d)
                day_ms = int(day.timestamp() * 1000)

                klines = fetch_klines_range(symbol, day_ms, min(day_ms + 86_400_000, end_ms))
                if len(klines):
                    if "csv" in args.to:
                        save_csv_daily(klines, output_dir / symbol / "1m", day)
                    if "postgres" in args.to:
                        all_klines.append(klines)

                print(f"  [klines {d + 1}/{args.days}] {day.strftime('%Y-%m-%d')}: "
                      f"bars={len(klines)}", flush=True)

        if not args.skip_orderflow:
            def _of_progress(i, n, day, nrows):
                if i % 30 == 0 or i == n:
                    r_str = "SKIPPED" if nrows == -1 else str(nrows)
                    print(f"  [orderflow {i}/{n}] {day.strftime('%Y-%m-%d')}: "
                          f"rows={r_str}", flush=True)

            of_save_cb = None
            if "postgres" in args.to and dsn:
                from mars_lite.data.postgres_store import upsert_orderflow
                def _of_save_cb(df):
                    if not df.empty:
                        upsert_orderflow(dsn, "binance", symbol, df)
                of_save_cb = _of_save_cb

            print(f"  orderflow 1m ({args.days}d, vision)...")
            from mars_lite.data.binance_vision import fetch_orderflow_vision
            of_all = fetch_orderflow_vision(
                symbol, start_ms, end_ms,
                pause_sec=0.02, progress_cb=_of_progress,
                exclude_days=existing_orderflow_days,
                save_cb=of_save_cb,
            )
            
            if len(of_all) and "csv" in args.to:
                for d in range(args.days):
                    day = start + timedelta(days=d)
                    day_ms = int(day.timestamp() * 1000)
                    day_end_ms = day_ms + 86_400_000
                    sl = of_all[
                        (of_all["timestamp"] >= day_ms)
                        & (of_all["timestamp"] < day_end_ms)
                    ]
                    if len(sl):
                        save_csv_daily(
                            sl.reset_index(drop=True),
                            output_dir / symbol / "orderflow_1m", day,
                        )



    print("Done.")


if __name__ == "__main__":
    main()
