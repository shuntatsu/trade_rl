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
from mars_lite.features.horizon_scan import (
    run_horizon_scan, compute_breakeven_ic, DEFAULT_HORIZONS,
)

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
    ap.add_argument("--breakeven", action="store_true",
                    help="各ホライズンについて、意思決定頻度"
                         "（既定 max(1,horizon//2)）込みで黒字化する"
                         "最小の目標ICをノイズオラクルで推定して併記する")
    ap.add_argument("--cs-demean", action="store_true",
                    help="raw（絶対リターン）に加えcs_demean（市場中立の"
                         "相対アルファ）でもスキャンし併記する。狭い"
                         "ユニバースでは市場全体の方向がICを汚染しうるため、"
                         "両者を比較すると相対アルファの実力が見える")
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

    report = run_horizon_scan(train_fs, horizons=tuple(args.horizons), n_folds=args.n_folds,
                              target="raw")
    print(report.summary())

    payload = report.to_dict()
    payload["target"] = "raw"

    if args.cs_demean:
        cs_report = run_horizon_scan(train_fs, horizons=tuple(args.horizons),
                                     n_folds=args.n_folds, target="cs_demean")
        print("\n[Horizon Scan: cs_demean（市場中立の相対アルファ）]")
        print(cs_report.summary())
        payload["cs_demean"] = cs_report.to_dict()

    if args.breakeven:
        print("\n[Breakeven IC] (decision_every = max(1, horizon//2) 込みでコスト後に黒字化する最小の目標IC)")
        breakeven = {}
        for r in report.results:
            de = max(1, r.horizon // 2)
            be_ic = compute_breakeven_ic(train_fs, r.horizon, decision_every=de)
            breakeven[r.horizon] = be_ic
            be_str = f"{be_ic:.2f}" if be_ic is not None else "黒字化なし（試した範囲では割に合わない）"
            print(f"  horizon={r.horizon:<4} decision_every={de:<3} breakeven_ic={be_str}")
        payload["breakeven_ic_by_horizon"] = breakeven

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Report -> {out_path}")


if __name__ == "__main__":
    main()
