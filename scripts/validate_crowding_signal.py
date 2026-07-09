"""
crowding シグナル（建玉クラウディング）の独立検証: OIデータ品質 + csz_oi IC再計測

目的（実データ検証の潔白性の担保）: crowding戦略は Postgres/Binance の
2024-11〜2026-07 期間で dev検証に合格した（cost2x median +3.89%, Sharpe 1.86,
csz_oi の h24 cs_demean IC ≈ -0.069）が、DSR 0.229 と多重検定補正後の有意性は
弱く、「期間固有のまぐれ」の疑いが残っていた。

このスクリプトは **別の未接触期間・別のユニバース**（例: data.binance.vision の
2023-08〜2025-03、7銘柄）で以下を機械的に検証する:

  1. OIデータ品質: 欠損率・open_interestの同値連続ラン（前方補完アーティ
     ファクト検出）・timestamp単調性/ギャップ。ライブ運用前に必須。
  2. csz_oi IC再計測: 建玉のCSズ(csz_oi)の複数ホライズンでのクロスセクショナル
     rank IC。**戦略評価ではなく特徴量の予測力測定なので holdout を消費しない。**
     2023-24年でも h24 cs_demean IC が -0.05 前後の負を再現すれば、crowding は
     実在ファクターとほぼ確定。出なければ棄却（DSR警告どおり）。

使い方:
    uv run python scripts/validate_crowding_signal.py \
        --data ./data_past --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT \
        BNBUSDT SUIUSDT DOGEUSDT --warmup-days 100 --output ./output/crowding_validation
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mars_lite.data.sources import create_source
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.features.signal_check import (
    _forward_returns,
    _rank_ic,
    _transform_target,
)


def oi_quality(source, symbols) -> dict:
    """各銘柄のOI(open_interest)データ品質を機械チェックする。"""
    report = {}
    for sym in symbols:
        d = source.load_derivatives(sym)
        if d.empty or "open_interest" not in d.columns:
            report[sym] = {"status": "MISSING", "rows": 0}
            continue
        oi = d["open_interest"].to_numpy(dtype=np.float64)
        ts = d["timestamp"]
        # 同値連続ラン（前方補完アーティファクト）の最大長
        max_run = 1
        cur = 1
        for i in range(1, len(oi)):
            if oi[i] == oi[i - 1]:
                cur += 1
                max_run = max(max_run, cur)
            else:
                cur = 1
        # timestampギャップ（中央値間隔の3倍超を欠損とみなす）
        deltas = ts.diff().dropna().dt.total_seconds().to_numpy()
        med = float(np.median(deltas)) if len(deltas) else 0.0
        n_gaps = int(np.sum(deltas > 3 * med)) if med > 0 else 0
        report[sym] = {
            "status": "OK",
            "rows": int(len(oi)),
            "nan_frac": float(np.mean(~np.isfinite(oi))),
            "zero_frac": float(np.mean(oi <= 0)),
            "max_consecutive_dup_run": int(max_run),
            "dup_run_frac": float(max_run / len(oi)),
            "median_interval_sec": med,
            "n_timestamp_gaps": n_gaps,
            "monotonic_timestamp": bool(ts.is_monotonic_increasing),
        }
    return report


def pooled_feature_ic(fs, feat_name, horizon, target) -> float:
    if feat_name not in fs.feature_names:
        return float("nan")
    fi = fs.feature_names.index(feat_name)
    n_bars = fs.n_bars
    fwd = _transform_target(_forward_returns(fs, horizon), target, fs)
    valid = n_bars - horizon
    x = fs.features[:valid, :, fi].reshape(-1)
    y = fwd[:valid].reshape(-1)
    m = np.isfinite(y) & np.isfinite(x)
    return float(_rank_ic(x[m], y[m]))


def main() -> int:
    ap = argparse.ArgumentParser(description="crowding シグナル独立検証")
    ap.add_argument("--data", default="./data_past")
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--base-timeframe", default="1h")
    ap.add_argument("--warmup-days", type=float, default=100)
    ap.add_argument(
        "--features",
        nargs="+",
        default=["csz_oi", "csz_funding", "csz_ls", "oi_z", "oi_change"],
    )
    ap.add_argument("--horizons", nargs="+", type=int, default=[2, 4, 8, 24])
    ap.add_argument("--output", default="./output/crowding_validation")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    source = create_source("csv", args.symbols, data_dir=args.data)

    print("=== OI データ品質 ===", flush=True)
    q = oi_quality(source, args.symbols)
    for sym, r in q.items():
        if r["status"] != "OK":
            print(f"  {sym:10s} {r['status']}")
            continue
        print(
            f"  {sym:10s} rows={r['rows']:6d} nan={r['nan_frac']:.1%} "
            f"zero={r['zero_frac']:.1%} max_dup_run={r['max_consecutive_dup_run']} "
            f"({r['dup_run_frac']:.1%}) gaps={r['n_timestamp_gaps']} "
            f"mono={r['monotonic_timestamp']}"
        )

    print("\n=== FeatureSet 構築 ===", flush=True)
    fs = FeaturePipeline(args.symbols, base_timeframe=args.base_timeframe).build(source)
    if args.warmup_days > 0:
        from mars_lite.data.data_utils import TF_TO_MINUTES

        wb = int(args.warmup_days * 24 * 60 / TF_TO_MINUTES[args.base_timeframe])
        if wb < fs.n_bars:
            fs = fs.slice(wb, fs.n_bars)
    print(f"  {fs.n_bars} bars x {fs.n_symbols} symbols", flush=True)

    print("\n=== 特徴量 個別クロスセクショナル rank IC ===", flush=True)
    print(f"{'feature':14s} " + " ".join(f"h{h:<2d}(raw/csz)" for h in args.horizons))
    ic_table = {}
    for name in args.features:
        row = f"{name:14s} "
        ic_table[name] = {}
        for h in args.horizons:
            ic_raw = pooled_feature_ic(fs, name, h, "raw")
            ic_cs = pooled_feature_ic(fs, name, h, "cs_demean")
            ic_table[name][h] = {"raw": ic_raw, "cs_demean": ic_cs}
            row += f"{ic_raw:+.3f}/{ic_cs:+.3f} "
        print(row)

    # crowding の主シグナル csz_oi の h24 cs_demean IC を明示判定
    key = ic_table.get("csz_oi", {}).get(24, {}).get("cs_demean", float("nan"))
    verdict = (
        "REPRODUCED (実在ファクターの傍証)"
        if np.isfinite(key) and key <= -0.03
        else "NOT REPRODUCED (期間固有の疑い)"
    )
    print(f"\n[判定] csz_oi h24 cs_demean IC = {key:+.4f} -> {verdict}")
    print("  参考: Postgres(2024-11〜2026-07)では -0.069")

    with open(out / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data": args.data,
                "symbols": args.symbols,
                "n_bars": fs.n_bars,
                "oi_quality": q,
                "feature_ic": ic_table,
                "csz_oi_h24_cs_demean": key,
                "verdict": verdict,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nReport -> {out / 'validation_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
