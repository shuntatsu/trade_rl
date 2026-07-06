"""
オーダーフロー(aggTrades)のみを data.binance.vision から追加取得するスクリプト

fetch_binance_vision_only.py --skip-orderflow で先に取得済みの kline/funding/
derivatives に、重いため後回しにしていたオーダーフローを追加する。
出力レイアウトは CsvSource 互換（{data_dir}/{SYMBOL}/orderflow_1m/YYYY-MM-DD.csv）。
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mars_lite.data.binance_vision import fetch_orderflow_vision

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "SUIUSDT", "DOGEUSDT",
]


def save_csv_daily(df, out_dir: Path, day: datetime) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{day.strftime('%Y-%m-%d')}.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description="オーダーフロー追加取得(vision専用)")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--days", type=int, default=465)
    ap.add_argument("--output", default="./data")
    args = ap.parse_args()

    output_dir = Path(args.output)
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = now - timedelta(days=args.days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    for symbol in args.symbols:
        print(f"=== {symbol} ===", flush=True)

        def _of_progress(i, n, day, nrows):
            if i % 30 == 0 or i == n:
                print(f"    [orderflow {i}/{n}] {day.strftime('%Y-%m-%d')}: rows={nrows}", flush=True)

        of_all = fetch_orderflow_vision(
            symbol, start_ms, end_ms, pause_sec=0.02, progress_cb=_of_progress,
        )
        print(f"    {len(of_all)} rows total", flush=True)
        if len(of_all):
            for d in range(args.days):
                day = start + timedelta(days=d)
                day_ms = int(day.timestamp() * 1000)
                day_end_ms = day_ms + 86_400_000
                sl = of_all[
                    (of_all["timestamp"] >= day_ms) & (of_all["timestamp"] < day_end_ms)
                ]
                if len(sl):
                    save_csv_daily(sl.reset_index(drop=True), output_dir / symbol / "orderflow_1m", day)

    print("Done.")


if __name__ == "__main__":
    main()
