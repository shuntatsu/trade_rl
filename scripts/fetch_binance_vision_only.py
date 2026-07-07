"""
Binance Futures 実データ取得スクリプト（data.binance.vision のみ・地域制限回避）

fetch_futures.py は funding rate・デリバREST補完で fapi.binance.com に
直接アクセスするため、Binanceが地域ブロックされる環境（HTTP 451）では
クラッシュする。本スクリプトは data.binance.vision（静的ファイル配信、
地域ブロックの対象外）のみを使い、REST呼び出しを一切行わない:

    kline (1m)      : data.binance.vision daily klines ZIP
    orderflow (1m)  : data.binance.vision daily aggTrades ZIP
    derivatives     : data.binance.vision daily metrics ZIP（OI/L-S/liq代理）
    funding rate    : data.binance.vision monthly fundingRate ZIP

出力は fetch_futures.py --to csv と同じ CsvSource 互換レイアウト。

使い方:
    python scripts/fetch_binance_vision_only.py --symbols BTCUSDT ETHUSDT --days 365
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mars_lite.data.binance_vision import (
    fetch_funding_range,
    fetch_klines_range,
    fetch_metrics_range,
    fetch_orderflow_vision,
)

DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "BNBUSDT",
    "SUIUSDT",
    "DOGEUSDT",
]


def save_csv_daily(df, out_dir: Path, day: datetime) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{day.strftime('%Y-%m-%d')}.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description="Binance Futures 実データ取得(vision専用)")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--output", default="./data")
    ap.add_argument(
        "--skip-orderflow",
        action="store_true",
        help="aggTrades集計をスキップ（取得が重いため）",
    )
    args = ap.parse_args()

    output_dir = Path(args.output)
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = now - timedelta(days=args.days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    for symbol in args.symbols:
        print(f"=== {symbol} ===", flush=True)

        print("  funding rate (monthly vision)...", flush=True)
        funding = fetch_funding_range(symbol, start_ms, end_ms)
        print(f"    {len(funding)} events", flush=True)
        if len(funding):
            fdir = output_dir / symbol / "funding"
            fdir.mkdir(parents=True, exist_ok=True)
            funding.to_csv(fdir / "funding.csv", index=False)

        print("  derivatives (OI/LS/liq, vision metrics)...", flush=True)
        derivatives = fetch_metrics_range(symbol, start_ms, end_ms)
        print(f"    {len(derivatives)} rows (5m native)", flush=True)
        if len(derivatives):
            ddir = output_dir / symbol / "derivatives"
            ddir.mkdir(parents=True, exist_ok=True)
            derivatives.to_csv(ddir / "derivatives.csv", index=False)

        print(f"  klines 1m ({args.days}d, vision)...", flush=True)

        def _kl_progress(i, n, day, nrows):
            if i % 30 == 0 or i == n:
                print(
                    f"    [klines {i}/{n}] {day.strftime('%Y-%m-%d')}: rows={nrows}",
                    flush=True,
                )

        klines_all = fetch_klines_range(
            symbol,
            start_ms,
            end_ms,
            interval="1m",
            pause_sec=0.02,
            progress_cb=_kl_progress,
        )
        print(f"    {len(klines_all)} bars total", flush=True)
        if len(klines_all):
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
                        sl.reset_index(drop=True), output_dir / symbol / "1m", day
                    )

        if not args.skip_orderflow:
            print(f"  orderflow 1m ({args.days}d, vision aggTrades)...", flush=True)

            def _of_progress(i, n, day, nrows):
                if i % 30 == 0 or i == n:
                    print(
                        f"    [orderflow {i}/{n}] {day.strftime('%Y-%m-%d')}: rows={nrows}",
                        flush=True,
                    )

            of_all = fetch_orderflow_vision(
                symbol,
                start_ms,
                end_ms,
                pause_sec=0.02,
                progress_cb=_of_progress,
            )
            print(f"    {len(of_all)} rows total", flush=True)
            if len(of_all):
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
                            output_dir / symbol / "orderflow_1m",
                            day,
                        )

    print("Done.")


if __name__ == "__main__":
    main()
