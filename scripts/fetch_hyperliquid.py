"""
Hyperliquid 実データ取得スクリプト（公開API・認証不要）

上位足（15m/1h/4h/1d）と funding を Hyperliquid 情報APIから取得し、
data/hyperliquid/{COIN}_{interval}.csv にキャッシュする。以後は
`--source hyperliquid` でそのまま学習・検証に使える。

使い方:
    python scripts/fetch_hyperliquid.py --symbols BTCUSDT ETHUSDT SOLUSDT --days 180
    python scripts/train_portfolio.py --phase train --source hyperliquid \
        --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT SUIUSDT DOGEUSDT --days 180
"""

import argparse
import time

from mars_lite.data.sources import HyperliquidSource

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "SUIUSDT", "DOGEUSDT",
]
TIMEFRAMES = ["15m", "1h", "4h", "1d"]


def main():
    ap = argparse.ArgumentParser(description="Hyperliquid 実データ取得")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--cache-dir", default="./data/hyperliquid")
    ap.add_argument("--end", default=None, help="終了時刻（既定=現在）")
    args = ap.parse_args()

    src = HyperliquidSource(args.symbols, days=args.days,
                            cache_dir=args.cache_dir, end=args.end)
    print(f"Fetching {len(args.symbols)} symbols x {len(TIMEFRAMES)} TF "
          f"({args.days}日) -> {args.cache_dir}")
    for sym in args.symbols:
        t = time.time()
        rows = {}
        for tf in TIMEFRAMES:
            df = src.load_klines(sym, tf)
            rows[tf] = len(df)
        f = src.load_funding(sym)
        ok = "OK" if rows["1h"] > 0 else "EMPTY(コイン未対応?)"
        print(f"  {sym:<10} {src._coin(sym):<6} "
              f"{'/'.join(str(rows[tf]) for tf in TIMEFRAMES)} bars, "
              f"funding={len(f)}  [{ok}, {time.time()-t:.1f}s]")
    print("完了。--source hyperliquid で学習・検証に使えます。")


if __name__ == "__main__":
    main()
