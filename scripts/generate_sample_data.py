"""
サンプルデータ生成スクリプト

Binance APIに接続できない環境（オフライン・地域制限）でも
学習パイプライン全体を検証できるよう、fetch_binance.pyと同じ
ディレクトリ構成（data/{SYMBOL}/{interval}/YYYY-MM-DD.csv）で
合成OHLCVデータを生成する。

使い方:
    python scripts/generate_sample_data.py --symbols BTCUSDT ETHUSDT --days 14 --output ./data
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


MINUTES_PER_DAY = 1440


def generate_day(
    rng: np.random.Generator,
    date: datetime,
    start_price: float,
    base_volume: float,
) -> pd.DataFrame:
    """1日分の1分足OHLCVを生成（ランダムウォーク + U字型出来高）"""
    n = MINUTES_PER_DAY
    returns = rng.normal(0, 0.0008, n)
    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[start_price], close[:-1]])
    spread = np.abs(rng.normal(0, 0.0005, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)

    tod = np.arange(n)
    u_shape = 1 + 0.6 * np.cos(2 * np.pi * tod / n)
    volume = base_volume * u_shape * rng.exponential(1, n)

    start_ms = int(date.timestamp() * 1000)
    timestamps = start_ms + np.arange(n) * 60_000

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def main():
    parser = argparse.ArgumentParser(description="合成OHLCVデータ生成")
    parser.add_argument("--symbols", type=str, nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--days", type=int, default=14, help="生成日数")
    parser.add_argument("--output", type=str, default="./data", help="出力ディレクトリ")
    parser.add_argument("--start-date", type=str, default=None, help="開始日 YYYY-MM-DD")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    start_prices = {"BTCUSDT": 40000.0, "ETHUSDT": 2500.0}

    if args.start_date:
        start = datetime.fromisoformat(args.start_date)
    else:
        start = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=args.days)

    rng = np.random.default_rng(args.seed)

    for symbol in args.symbols:
        price = start_prices.get(symbol, float(rng.uniform(1, 1000)))
        base_volume = float(rng.uniform(200, 2000))
        sym_dir = output_dir / symbol / "1m"
        sym_dir.mkdir(parents=True, exist_ok=True)

        for d in range(args.days):
            date = start + timedelta(days=d)
            df = generate_day(rng, date, price, base_volume)
            price = float(df["close"].iloc[-1])
            df.to_csv(sym_dir / f"{date.strftime('%Y-%m-%d')}.csv", index=False)

        print(f"{symbol}: {args.days} days -> {sym_dir}")

    metadata = {
        "symbols": args.symbols,
        "intervals": ["1m"],
        "generated": True,
        "days": args.days,
        "start_date": start.strftime("%Y-%m-%d"),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"metadata.json -> {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
