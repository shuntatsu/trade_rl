"""
サンプルデータ生成スクリプト

Binance APIに接続できない環境でも学習パイプライン全体を検証できるよう、
fetch_binance.py / fetch_futures.py と同じディレクトリ構成で合成データを生成する。

生成物:
    data/{SYMBOL}/1m/YYYY-MM-DD.csv            OHLCV 1分足
    data/{SYMBOL}/orderflow_1m/YYYY-MM-DD.csv  オーダーフロー1分集計
    data/{SYMBOL}/funding/funding.csv          8時間毎funding rate
    data/metadata.json

アルファ注入（--alpha）:
    none    : 純ランダムウォーク。取引しないことが最適（健全性試験の陰性対照）
    cross   : 銘柄ごとの持続的ドリフト（AR(1)潜在状態）。過去リターンの
              クロスセクショナル相対強弱が将来リターンを予測する（陽性対照）
    meanrev : 24時間移動平均からの乖離が反転する平均回帰アルファ

使い方:
    python scripts/generate_sample_data.py --days 60 --alpha cross --output ./data
    python scripts/generate_sample_data.py --days 60 --alpha none --output ./data_noise
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


MINUTES_PER_DAY = 1440
DEFAULT_SYMBOLS = [
    "BTCUSDT", "XRPUSDT", "SUIUSDT", "BNBUSDT", "ETHUSDT", "PAXGUSDT", "ETHBTC",
]
START_PRICES = {
    "BTCUSDT": 40000.0, "XRPUSDT": 0.6, "SUIUSDT": 1.5, "BNBUSDT": 300.0,
    "ETHUSDT": 2500.0, "PAXGUSDT": 2000.0, "ETHBTC": 0.06,
}

from mars_lite.data.synthetic import (  # noqa: E402
    generate_market, build_ohlcv, build_orderflow, build_funding,
)


def main():
    parser = argparse.ArgumentParser(description="合成マーケットデータ生成（アルファ注入対応）")
    parser.add_argument("--symbols", type=str, nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--days", type=int, default=60, help="生成日数")
    parser.add_argument("--output", type=str, default="./data", help="出力ディレクトリ")
    parser.add_argument("--start-date", type=str, default=None, help="開始日 YYYY-MM-DD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alpha", type=str, default="none", choices=["none", "cross", "meanrev", "multi"],
        help="注入するアルファの種類"
    )
    parser.add_argument(
        "--alpha-strength", type=float, default=0.002,
        help="アルファ強度（1時間あたりの予測可能ドリフトの標準偏差）"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    if args.start_date:
        start = datetime.fromisoformat(args.start_date)
    else:
        start = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=args.days)

    rng = np.random.default_rng(args.seed)
    n_minutes = args.days * MINUTES_PER_DAY
    n_symbols = len(args.symbols)

    print(f"Generating {args.days} days x {n_symbols} symbols "
          f"(alpha={args.alpha}, strength={args.alpha_strength})...")

    returns, latent = generate_market(
        rng, n_symbols, n_minutes, args.alpha, args.alpha_strength
    )

    for i, symbol in enumerate(args.symbols):
        price = START_PRICES.get(symbol, float(rng.uniform(1, 1000)))
        base_volume = float(rng.uniform(200, 2000))

        kline_df = build_ohlcv(rng, returns[:, i], price, base_volume, start)
        of_df = build_orderflow(rng, kline_df, latent[:, i], args.alpha)
        funding_df = build_funding(rng, latent[:, i], start, args.days, args.alpha)

        # 日別ファイルに分割保存（fetch系スクリプトと同一レイアウト）
        for name, df in [("1m", kline_df), ("orderflow_1m", of_df)]:
            sym_dir = output_dir / symbol / name
            sym_dir.mkdir(parents=True, exist_ok=True)
            for d in range(args.days):
                day_slice = df.iloc[d * MINUTES_PER_DAY:(d + 1) * MINUTES_PER_DAY]
                date_str = (start + timedelta(days=d)).strftime("%Y-%m-%d")
                day_slice.to_csv(sym_dir / f"{date_str}.csv", index=False)

        funding_dir = output_dir / symbol / "funding"
        funding_dir.mkdir(parents=True, exist_ok=True)
        funding_df.to_csv(funding_dir / "funding.csv", index=False)

        print(f"  {symbol}: klines + orderflow + funding -> {output_dir / symbol}")

    metadata = {
        "symbols": args.symbols,
        "intervals": ["1m"],
        "generated": True,
        "days": args.days,
        "start_date": start.strftime("%Y-%m-%d"),
        "alpha": args.alpha,
        "alpha_strength": args.alpha_strength,
        "seed": args.seed,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"metadata.json -> {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
