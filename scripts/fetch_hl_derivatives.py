"""
Hyperliquid銘柄向けデリバティブ/オーダーフロー取得スクリプト（Binance代理）

Hyperliquidの公開APIはOI・L/S比率・清算の**履歴**を提供しない（現在値の
スナップショットのみ）。同一コインのBinance USDT-M先物 metrics を
data.binance.vision から取得し（5分足・長期履歴）、HyperliquidSourceの
キャッシュ規約に保存する:

    data/hyperliquid/{COIN}_derivatives.csv   (timestamp, open_interest, ls_ratio, liq_notional)
    data/hyperliquid/{COIN}_orderflow_1m.csv  (timestamp, buy_volume, sell_volume,
                                               trade_count, avg_trade_size, volume_imbalance)

これにより oi_z/oi_change/ls_ratio_z/liq_z/of_imbalance/of_count_z/of_size_z
特徴が実データで埋まる（HyperliquidSource.load_derivatives/load_orderflow が
このキャッシュを読む）。

前向き（今後の期間）のHLネイティブ建玉残高・funding・プレミアムは
scripts/collect_hl_snapshots.py で別途蓄積する。

使い方:
    python scripts/fetch_hl_derivatives.py --symbols BTCUSDT ETHUSDT SOLUSDT --days 180
    python scripts/fetch_hl_derivatives.py --skip-orderflow  # OI/LS/liqのみ（速い）
"""

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_futures import fetch_derivatives, fetch_orderflow_day  # noqa: E402

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "SUIUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT",
    "LTCUSDT", "BCHUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
]


def _coin(symbol: str) -> str:
    s = symbol.upper()
    for suf in ("USDT", "USDC", "PERP"):
        if s.endswith(suf) and len(s) > len(suf):
            return s[: -len(suf)]
    return s


def main():
    ap = argparse.ArgumentParser(description="Hyperliquid銘柄向けBinance代理デリバティブ取得")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                    help="Binance先物のシンボル表記（例: BTCUSDT）")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--cache-dir", default="./data/hyperliquid")
    ap.add_argument("--skip-orderflow", action="store_true",
                    help="aggTrades集計をスキップ（OI/LS/liqのみ、高速）")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = now - timedelta(days=args.days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    for symbol in args.symbols:
        coin = _coin(symbol)
        print(f"=== {symbol} (coin={coin}) ===")

        print("  derivatives (OI/LS/liq, Binance vision)...")
        deriv = fetch_derivatives(symbol, start_ms, end_ms)
        deriv = deriv.rename(columns={"timestamp": "ts_ms"})
        import pandas as pd
        deriv["timestamp"] = pd.to_datetime(deriv["ts_ms"], unit="ms")
        deriv = deriv[["timestamp", "open_interest", "ls_ratio", "liq_notional"]]
        deriv.to_csv(cache_dir / f"{coin}_derivatives.csv", index=False)
        print(f"    {len(deriv)} rows -> {coin}_derivatives.csv")

        if not args.skip_orderflow:
            all_of = []
            for d in range(args.days):
                day = start + timedelta(days=d)
                day_ms = int(day.timestamp() * 1000)
                of = fetch_orderflow_day(symbol, day_ms)
                if len(of):
                    all_of.append(of)
                if (d + 1) % 30 == 0 or d == args.days - 1:
                    print(f"    orderflow [{d + 1}/{args.days}] days processed", flush=True)
            if all_of:
                of_all = pd.concat(all_of, ignore_index=True)
                of_all["timestamp"] = pd.to_datetime(of_all["timestamp"], unit="ms")
                of_all.to_csv(cache_dir / f"{coin}_orderflow_1m.csv", index=False)
                print(f"    {len(of_all)} rows -> {coin}_orderflow_1m.csv")
            else:
                print("    orderflow: no data")

        time.sleep(0.2)

    print("完了。--source hyperliquid で oi_z/ls_ratio_z/liq_z/of_* が実データで埋まります。")


if __name__ == "__main__":
    main()
