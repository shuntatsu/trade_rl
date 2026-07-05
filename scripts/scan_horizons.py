"""
ホライズンスキャンCLI

指定ソースのデータで、複数ホライズン（バー数）ごとのウォークフォワード
OOS ICを計測し、最良ホライズンを提示する。train/test混在を避けるため、
本スクリプトは既定でデータの先頭80%（学習スライス相当）のみを走査する。

使い方:
    python scripts/scan_horizons.py --source synthetic --alpha cross --days 90
    python scripts/scan_horizons.py --source hyperliquid --days 180 \
        --symbols BTCUSDT ETHUSDT SOLUSDT --horizons 1 2 4 8 24 48 72
"""

import argparse
import json
from pathlib import Path

from mars_lite.data.sources import create_source, SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.features.horizon_scan import run_horizon_scan, DEFAULT_HORIZONS

DEFAULT_SYMBOLS = [
    "BTCUSDT", "XRPUSDT", "SUIUSDT", "BNBUSDT", "ETHUSDT", "PAXGUSDT", "ETHBTC",
]


def main():
    ap = argparse.ArgumentParser(description="ホライズンスキャン")
    ap.add_argument("--source", choices=["synthetic", "csv", "hyperliquid"], default="synthetic")
    ap.add_argument("--data", type=str, default="./data")
    ap.add_argument("--symbols", type=str, nargs="+", default=None)
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--alpha", default="cross", choices=["none", "cross", "meanrev", "multi", "bull"])
    ap.add_argument("--alpha-strength", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--train-fraction", type=float, default=0.8,
                    help="testリーク防止のため先頭この割合のみ走査")
    ap.add_argument("--output", type=str, default="./output/horizon_scan.json")
    args = ap.parse_args()

    if args.source == "synthetic":
        source = SyntheticSource(n_days=args.days, alpha=args.alpha,
                                 alpha_strength=args.alpha_strength, seed=args.seed)
        symbols = source.symbols
    else:
        symbols = args.symbols or DEFAULT_SYMBOLS
        kwargs = {"data_dir": args.data} if args.source == "csv" else {"days": args.days}
        source = create_source(args.source, symbols, **kwargs)

    fs = FeaturePipeline(symbols).build(source)
    print(f"FeatureSet: {fs.n_bars} bars x {fs.n_symbols} symbols x {fs.n_features} features")

    split = int(fs.n_bars * args.train_fraction)
    train_fs = fs.slice(0, split)
    print(f"Scanning on train slice only: {train_fs.n_bars}/{fs.n_bars} bars "
          f"(test-leak防止のため残り{100 - args.train_fraction * 100:.0f}%は使わない)")

    report = run_horizon_scan(train_fs, horizons=tuple(args.horizons), n_folds=args.n_folds)
    print(report.summary())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"Report -> {out_path}")


if __name__ == "__main__":
    main()
