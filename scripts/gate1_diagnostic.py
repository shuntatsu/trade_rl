"""
ゲート1 診断マトリクス（horizon×target、ネストウォークフォワード）

目的: gate1(IC>=0.02)不合格の原因が「パラメータ選択(horizon/target/正則化)」
なのか「特徴量セット/意思決定粒度そのものの限界」なのかを切り分ける。

守っていること:
- holdout（末尾--holdout-frac）はこのスクリプトのどの選択（horizon/target/
  lambda/feature-mask）にも一切使わない。dev区間だけで完結させる。
- lambda選択はネストウォークフォワード（外側テストに触れない）。
- 標準化は各foldの学習区間の平均・標準偏差だけを使う。
- feature-maskは主結果（horizon×targetマトリクス）には適用しない。
  別出しの4パターン比較実験としてのみ使う。
- 固定閾値0.02は緩和しない。compute_breakeven_icは参考値として併記するのみ。

使い方:
    uv run python scripts/gate1_diagnostic.py --source postgres
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mars_lite.data.quality import run_quality_gate
from mars_lite.data.sources import create_source
from mars_lite.eval.gate1_diagnostics import nested_walk_forward_ic, run_matrix
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.features.horizon_scan import compute_breakeven_ic
from mars_lite.features.signal_check import compute_feature_mask
from mars_lite.pipeline.dataset_builder import DEFAULT_SYMBOLS

HORIZONS = (1, 2, 4, 8, 12, 24, 48, 72)
TARGETS = ("raw", "cs_demean", "vol_norm")


def build_fs(
    source_name: str,
    base_timeframe: str,
    args,
    cached_source=None,
    cached_symbols=None,
):
    """
    cached_source/cached_symbols を渡すと品質ゲートの再実行とDB再クエリを
    省略できる（1h診断で確定した通過銘柄・ソースを4h診断でも使い回す）。
    """
    if cached_source is not None and cached_symbols is not None:
        return (
            FeaturePipeline(cached_symbols, base_timeframe=base_timeframe).build(
                cached_source
            ),
            cached_source,
            cached_symbols,
        )

    symbols = args.symbols or DEFAULT_SYMBOLS
    kwargs = {}
    if source_name == "postgres":
        kwargs = {
            "dsn": args.pg_dsn,
            "source": args.pg_source,
            "derivatives_source": args.pg_derivatives_source,
            "orderflow_source": args.pg_derivatives_source,
        }
    elif source_name == "hyperliquid":
        kwargs = {"days": args.days}
    elif source_name == "csv":
        kwargs = {"data_dir": args.data}
    source = create_source(source_name, symbols, **kwargs)
    qrep = run_quality_gate(source, symbols, base_timeframe="1h")
    print(qrep.summary())
    symbols = qrep.passing_symbols
    if len(symbols) < 2:
        raise ValueError("品質ゲート通過銘柄が2未満です")
    fs = FeaturePipeline(symbols, base_timeframe=base_timeframe).build(source)
    return fs, source, symbols


def carve_holdout(fs, holdout_frac: float, purge: int):
    holdout_start = int(fs.n_bars * (1.0 - holdout_frac))
    dev = fs.slice(0, holdout_start)
    holdout = fs.slice(holdout_start + purge, fs.n_bars)
    return dev, holdout


def print_matrix(results, threshold=0.02):
    print("\n=== horizon x target マトリクス（devのみ、ネストウォークフォワード） ===")
    for r in results:
        print(r.summary_line())
    passing = [r for r in results if r.passed(threshold)]
    return passing


def run_feature_set_comparison(dev_fs, horizon, target):
    print(f"\n=== 特徴量セット比較（h={horizon}, target={target}） ===")
    names = dev_fs.feature_names
    n = len(names)

    all_mask = np.ones(n, dtype=bool)
    res_all = nested_walk_forward_ic(
        dev_fs, horizon, target=target, feature_mask=all_mask
    )
    print(f"[全特徴Ridge]              {res_all.summary_line()}")

    train_fs = dev_fs.slice(0, int(dev_fs.n_bars * 0.8))
    fm = compute_feature_mask(train_fs, horizon=horizon, target=target)
    res_mask = nested_walk_forward_ic(
        dev_fs, horizon, target=target, feature_mask=np.asarray(fm["mask"])
    )
    print(
        f"[現在のfeature-mask]        {res_mask.summary_line()}  (kept={sum(fm['mask'])}/{n})"
    )

    def is_divergence(name: str) -> bool:
        return name.endswith("rsi_divergence") or name.endswith("cci_divergence")

    ichi_no_div = np.array([not is_divergence(nm) for nm in names])
    res_ichi_no_div = nested_walk_forward_ic(
        dev_fs, horizon, target=target, feature_mask=ichi_no_div
    )
    print(
        f"[一目あり・divergenceなし]  {res_ichi_no_div.summary_line()}  (kept={ichi_no_div.sum()}/{n})"
    )

    ichi_1h_div = np.array(
        [
            (not is_divergence(nm)) or nm in ("1h_rsi_divergence", "1h_cci_divergence")
            for nm in names
        ]
    )
    res_ichi_1h_div = nested_walk_forward_ic(
        dev_fs, horizon, target=target, feature_mask=ichi_1h_div
    )
    print(
        f"[一目+1h divergenceのみ]    {res_ichi_1h_div.summary_line()}  (kept={ichi_1h_div.sum()}/{n})"
    )

    return {
        "all_features": res_all,
        "current_feature_mask": res_mask,
        "ichimoku_no_divergence": res_ichi_no_div,
        "ichimoku_1h_divergence_only": res_ichi_1h_div,
    }


def run_breakeven(dev_fs, horizons):
    print("\n=== コスト後黒字化に必要なIC（compute_breakeven_ic、参考値） ===")
    out = {}
    for h in horizons:
        be = compute_breakeven_ic(dev_fs, horizon=h)
        out[h] = be
        be_str = f"{be:.3f}" if be is not None else "黒字化不可(候補IC全滅)"
        print(f"  horizon={h:<4} breakeven_ic={be_str}  (固定閾値0.02は緩和しない)")
    return out


def full_diagnostic(
    source_name, base_timeframe, args, cached_source=None, cached_symbols=None
):
    fs, source, symbols = build_fs(
        source_name, base_timeframe, args, cached_source, cached_symbols
    )
    purge = max(24, max(HORIZONS))
    dev_fs, holdout_fs = carve_holdout(fs, args.holdout_frac, purge)
    print(
        f"\n[base_timeframe={base_timeframe}] dev={dev_fs.n_bars}本 / "
        f"holdout={holdout_fs.n_bars}本(未使用のまま) / n_features={dev_fs.n_features}"
    )

    t0 = time.time()
    results = run_matrix(dev_fs, horizons=HORIZONS, targets=TARGETS)
    print(f"(matrix time: {time.time() - t0:.1f}s)")
    passing = print_matrix(results)

    breakeven = run_breakeven(dev_fs, HORIZONS)

    ref_h, ref_t = 4, "raw"
    if passing:
        best = max(passing, key=lambda r: r.mean_oos_ic)
        ref_h, ref_t = best.horizon, best.target
    comparison = run_feature_set_comparison(dev_fs, ref_h, ref_t)

    return (
        {
            "base_timeframe": base_timeframe,
            "dev_bars": dev_fs.n_bars,
            "holdout_bars": holdout_fs.n_bars,
            "results": results,
            "passing": passing,
            "breakeven_ic": breakeven,
            "comparison_horizon": ref_h,
            "comparison_target": ref_t,
            "comparison": comparison,
        },
        source,
        symbols,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="postgres", choices=["postgres", "csv", "hyperliquid"]
    )
    parser.add_argument("--data", default="./data")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--days", type=int, default=600)
    parser.add_argument(
        "--pg-dsn", default="postgresql://trade_rl:trade_rl@localhost:5433/trade_rl"
    )
    parser.add_argument("--pg-source", default="binance")
    parser.add_argument("--pg-derivatives-source", default=None)
    parser.add_argument("--holdout-frac", type=float, default=0.15)
    parser.add_argument("--output", default="./output/gate1_diagnostic")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("#" * 70)
    print("# STAGE 1: base_timeframe=1h")
    print("#" * 70)
    diag_1h, source, symbols = full_diagnostic(args.source, "1h", args)
    # 1h観測での24-72h先ホライズンが合格するか判定
    long_horizon_pass = [
        r for r in diag_1h["results"] if r.horizon >= 24 and r.passed()
    ]

    diag_4h = None
    if not diag_1h["passing"]:
        print("\n" + "#" * 70)
        print("# 1hベースで全horizon×target不合格 -> STAGE 2: base_timeframe=4h")
        print("#" * 70)
        diag_4h, _, _ = full_diagnostic(
            args.source, "4h", args, cached_source=source, cached_symbols=symbols
        )

    print("\n" + "=" * 70)
    print("判断")
    print("=" * 70)
    if long_horizon_pass:
        best_long = max(long_horizon_pass, key=lambda r: r.mean_oos_ic)
        print(
            f"[判定] 1hベースでhorizon={best_long.horizon}(target={best_long.target})が合格"
            f"(IC={best_long.mean_oos_ic:+.4f})。"
        )
        print(
            f"  -> 提案: 1h観測を維持し、decision_everyを4〜{best_long.horizon // 4 or 4}程度に下げて"
            f"低頻度化する（--decision-every {max(4, best_long.horizon // 6)}）"
        )
    elif diag_4h is not None and diag_4h["passing"]:
        best4 = max(diag_4h["passing"], key=lambda r: r.mean_oos_ic)
        print(
            f"[判定] 1hベースは全滅だが、4hベースでhorizon={best4.horizon}"
            f"(target={best4.target})が合格(IC={best4.mean_oos_ic:+.4f})。"
        )
        print("  -> 提案: base_timeframe=4hに切り替えて学習パイプラインを回す")
    elif diag_4h is not None:
        print("[判定] 1hベース・4hベースともに全horizon×targetで不合格。")
        print(
            "  -> 提案: ここで初めて新規特徴量追加の検討へ進む（PROFIT_DESIGN.md Phase A再試行）"
        )
    else:
        print(
            "[判定] 1hベースで合格ケースあり（24h以上ではないが特定horizon/targetで合格）。"
        )
        print(
            "  -> 詳細は上記マトリクスを参照し、合格した(horizon, target)で運用を検討"
        )

    def _serialize(diag):
        if diag is None:
            return None
        return {
            "base_timeframe": diag["base_timeframe"],
            "dev_bars": diag["dev_bars"],
            "holdout_bars": diag["holdout_bars"],
            "results": [
                {
                    "horizon": int(r.horizon),
                    "target": r.target,
                    "mean_oos_ic": float(r.mean_oos_ic),
                    "positive_fold_ratio": float(r.positive_fold_ratio),
                    "t_stat": float(r.t_stat),
                    "stability_passed": bool(r.stability_passed),
                    "passed": bool(r.passed()),
                    "group_ic": {k: float(v) for k, v in r.group_ic.items()},
                }
                for r in diag["results"]
            ],
            "breakeven_ic": {
                str(k): (float(v) if v is not None else None)
                for k, v in diag["breakeven_ic"].items()
            },
            "comparison_horizon": int(diag["comparison_horizon"]),
            "comparison_target": diag["comparison_target"],
            "comparison": {
                k: {
                    "mean_oos_ic": float(v.mean_oos_ic),
                    "positive_fold_ratio": float(v.positive_fold_ratio),
                    "t_stat": float(v.t_stat),
                    "passed": bool(v.passed()),
                }
                for k, v in diag["comparison"].items()
            },
        }

    with open(output_dir / "gate1_diagnostic_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {"stage_1h": _serialize(diag_1h), "stage_4h": _serialize(diag_4h)},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nレポート -> {output_dir / 'gate1_diagnostic_report.json'}")


if __name__ == "__main__":
    main()
