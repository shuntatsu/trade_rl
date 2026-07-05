"""
Hyperliquid 実データ取得スクリプト（公開API・認証不要）

上位足（15m/1h/4h/1d）と funding を Hyperliquid 情報APIから取得し、
data/hyperliquid/{COIN}_{interval}.csv にキャッシュする。
--to postgres で rl_klines / rl_funding_rate にも投入できる。

使い方:
    python scripts/fetch_hyperliquid.py --symbols BTCUSDT ETHUSDT --days 180
    python scripts/fetch_hyperliquid.py --symbols BTCUSDT ETHUSDT --days 180 --to postgres
"""

import argparse
import os
import time

from mars_lite.data.sources import HyperliquidSource

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "SUIUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT",
    "LTCUSDT", "BCHUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
]
TIMEFRAMES = ["15m", "1h", "4h", "1d"]


def main():
    ap = argparse.ArgumentParser(description="Hyperliquid 実データ取得")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--cache-dir", default="./data/hyperliquid")
    ap.add_argument("--end", default=None, help="終了時刻（既定=現在）")
    ap.add_argument("--to", nargs="+", default=["csv"], choices=["csv", "postgres"])
    ap.add_argument("--dsn", default=None, help="PostgreSQL DSN（--to postgres 時）")
    args = ap.parse_args()

    dsn = args.dsn or os.environ.get("PLATFORM_DB_URL")
    if "postgres" in args.to and not dsn:
        ap.error("--to postgres には --dsn か PLATFORM_DB_URL が必要です")

    src = HyperliquidSource(args.symbols, days=args.days,
                            cache_dir=args.cache_dir, end=args.end)
    print(f"Fetching {len(args.symbols)} symbols x {len(TIMEFRAMES)} TF "
          f"({args.days}日) -> {args.cache_dir}")
    if "postgres" in args.to:
        from mars_lite.data.postgres_store import upsert_funding, upsert_klines

    for sym in args.symbols:
        t = time.time()
        rows = {}
        for tf in TIMEFRAMES:
            df = src.load_klines(sym, tf)
            rows[tf] = len(df)
            if "postgres" in args.to and len(df):
                upsert_klines(dsn, "hyperliquid", sym, tf, df)
        time.sleep(1.0)
        f = src.load_funding(sym)
        if "postgres" in args.to and len(f):
            upsert_funding(dsn, "hyperliquid", sym, f)
        ok = "OK" if rows["1h"] > 0 else "EMPTY(コイン未対応?)"
        print(f"  {sym:<10} {src._coin(sym):<6} "
              f"{'/'.join(str(rows[tf]) for tf in TIMEFRAMES)} bars, "
              f"funding={len(f)}  [{ok}, {time.time()-t:.1f}s]")
    dest = ", ".join(args.to)
    print(f"完了 ({dest})。")


if __name__ == "__main__":
    main()
