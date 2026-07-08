"""
レジーム・ハイブリッドルーターの検証オーケストレータ。

単一戦略（trend_following）は局面によって明確に勝敗が分かれることが実データ
検証で判明した。本スクリプトは6分類レジームごとに {trend_following / flat /
RL専門家} を割り当てるルーターを、段階的に・過学習に注意しながら検証する。

--stage phase1 : ルールのみ（trend_following/flat）。学習ゼロ・数分で仮説判定
--stage phase2 : trend_down_early にRL専門家を追加（phase1合格時のみ意味がある）
--stage phase3 : 未接触ホールドアウトで一度だけ最終判定

過学習対策:
  - 割当表(RouterTable)の導出は常に「あるfoldのテスト区間より前」のデータ
    のみを使う（walk-forward化）。表がテスト区間を覗き見ることはない。
  - 合否基準はコード側(mars_lite.learning.regime_router.ROUTER_GATE_CRITERIA)
    に事前登録してあり、結果を見てから基準を変えない。
  - ホールドアウトはphase3で一度だけ触れる。実行済みマーカーで再実行を記録する。

使い方:
    uv run python scripts/run_router.py --stage phase1 --source csv --data ./data \\
        --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT SUIUSDT DOGEUSDT \\
        --warmup-days 100 --holdout-frac 0.15 --folds 4 --horizon 8 \\
        --output ./output/router
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.features.regime_taxonomy import FINE_REGIMES, label_fine_regimes
from mars_lite.learning.baselines import (
    flat_strategy,
    simulate_strategy,
    trend_following_strategy,
)
from mars_lite.learning.regime_router import (
    DEFAULT_LABELER_PARAMS,
    ROUTER_GATE_CRITERIA,
    RouterTable,
    derive_router_table,
    make_router_weight_fn,
)
from mars_lite.pipeline.cli import build_parser
from mars_lite.pipeline.dataset_builder import build_feature_set


def _print_step(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _fold_edges(n_bars: int, n_folds: int) -> np.ndarray:
    """walk_forward.pyと同じ流儀: [0.4n, n]をn_folds+1点で等分"""
    return np.linspace(int(n_bars * 0.4), n_bars, n_folds + 1).astype(int)


def run_phase1_walk_forward(
    fs: FeatureSet,
    n_folds: int = 4,
    purge: int = 24,
    min_regime_bars: int = 300,
) -> List[Dict]:
    """
    dev区間をwalk-forward化し、fold毎に「foldのテスト区間より前」のデータ
    だけから割当表を導出し、テスト区間で router/tf/flat を比較する。
    """
    labels = label_fine_regimes(fs, **DEFAULT_LABELER_PARAMS)
    edges = _fold_edges(fs.n_bars, n_folds)
    # simulate_strategy は内部で fs.close[t+1] を参照するため end_idx の上限は
    # fs.n_bars-1（walk_forward.pyはスライス済みfsを使うため同じ問題を踏まない
    # が、ここでは元のfsをstart_idx/end_idxで区切って使うため明示的にクランプする）。
    max_end = fs.n_bars - 1

    results = []
    for k in range(n_folds):
        train_end = int(edges[k])
        test_start = train_end + purge
        test_end = min(int(edges[k + 1]), max_end)
        if test_end - test_start < 50 or train_end < 100:
            continue

        table = derive_router_table(
            fs,
            labels,
            end=train_end,
            labeler_params=dict(DEFAULT_LABELER_PARAMS),
            min_regime_bars=min_regime_bars,
        )
        router_fn = make_router_weight_fn(fs, table)

        router_res = simulate_strategy(
            fs, router_fn, name="router", start_idx=test_start, end_idx=test_end
        )
        tf_res = simulate_strategy(
            fs,
            trend_following_strategy,
            name="trend_following",
            start_idx=test_start,
            end_idx=test_end,
        )
        flat_res = simulate_strategy(
            fs, flat_strategy, name="flat", start_idx=test_start, end_idx=test_end
        )

        results.append(
            {
                "fold": k,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "table": table.to_dict(),
                "router_return": router_res.total_return,
                "router_sharpe": router_res.sharpe,
                "tf_return": tf_res.total_return,
                "tf_sharpe": tf_res.sharpe,
                "flat_return": flat_res.total_return,
                "uplift_pt": (router_res.total_return - tf_res.total_return) * 100.0,
                "beat_tf": bool(router_res.total_return > tf_res.total_return),
            }
        )
        print(
            f"[fold {k}] test={test_end - test_start}本  "
            f"router={router_res.total_return:+.2%}  "
            f"tf={tf_res.total_return:+.2%}  "
            f"flat={flat_res.total_return:+.2%}  "
            f"uplift={results[-1]['uplift_pt']:+.2f}pt  "
            f"{'BEAT' if results[-1]['beat_tf'] else 'LOST'}"
        )
    return results


def judge_phase1(results: List[Dict]) -> Dict:
    """ROUTER_GATE_CRITERIA['phase1']に基づく合否判定（事前登録済み基準）"""
    crit = ROUTER_GATE_CRITERIA["phase1"]
    n_folds = len(results)
    n_beat = sum(1 for r in results if r["beat_tf"])
    uplifts = [r["uplift_pt"] for r in results]
    router_returns = [r["router_return"] for r in results]
    median_uplift = float(np.median(uplifts)) if uplifts else 0.0
    median_return = float(np.median(router_returns)) if router_returns else 0.0

    passed = (
        n_folds >= crit["min_folds_total"]
        and n_beat >= crit["min_folds_beat_tf"]
        and median_uplift >= crit["min_median_uplift_pt"]
        and (not crit["require_positive_median_return"] or median_return > 0)
    )
    return {
        "passed": bool(passed),
        "n_folds": n_folds,
        "n_beat_tf": n_beat,
        "median_uplift_pt": median_uplift,
        "median_router_return": median_return,
        "criteria": crit,
    }


def build_dev_holdout_split(args, output_dir: Path):
    fs_full = build_feature_set(args, output_dir=output_dir)
    purge = max(24, args.horizon)
    holdout_start = int(fs_full.n_bars * (1.0 - args.holdout_frac))
    min_bars = 50
    if (fs_full.n_bars - holdout_start - purge) < min_bars or holdout_start < min_bars:
        raise ValueError(
            f"データが短すぎてホールドアウト分離できません(n_bars={fs_full.n_bars})"
        )
    fs_dev = fs_full.slice(0, holdout_start)
    fs_holdout = fs_full.slice(holdout_start + purge, fs_full.n_bars)
    print(
        f"[holdout] dev={fs_dev.n_bars}本 (phase1/2が使用) / "
        f"holdout={fs_holdout.n_bars}本 (phase3でのみ・1回だけ使用)"
    )
    return fs_dev, fs_holdout


def main() -> int:
    parser = build_parser()
    parser.add_argument(
        "--stage", choices=["phase1", "phase2", "phase3"], default="phase1"
    )
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.15,
        help="phase1/2が一切触れない最終ホールドアウト区間の割合（末尾から）",
    )
    parser.add_argument(
        "--min-regime-bars",
        type=int,
        default=300,
        help="このバー数未満のレジームは割当表で判断せず既定'tf'にする",
    )
    parser.add_argument(
        "--router-config",
        type=str,
        default=None,
        help="phase2/3で読み書きする router_config.json のパス（既定: <output>/router_config.json）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    router_config_path = Path(args.router_config or (output_dir / "router_config.json"))

    fs_dev, fs_holdout = build_dev_holdout_split(args, output_dir)

    if args.stage == "phase1":
        _print_step(
            "Phase 1: ルールのみルーター（trend_following/flat）walk-forward検証"
        )
        results = run_phase1_walk_forward(
            fs_dev,
            n_folds=args.folds,
            purge=max(24, args.horizon),
            min_regime_bars=args.min_regime_bars,
        )
        verdict = judge_phase1(results)
        print(f"\n[Phase1 判定] {'PASS' if verdict['passed'] else 'FAIL'}")
        print(json.dumps(verdict, indent=2, ensure_ascii=False))

        report_path = output_dir / "phase1_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {"folds": results, "verdict": verdict}, f, indent=2, ensure_ascii=False
            )
        print(f"\nReport -> {report_path}")

        if not verdict["passed"]:
            print(
                "\n[STOP] Phase1不合格。単一戦略ルーターでも trend_following を"
                "上回れないため、仮説（レジーム限定によるTF劣位局面の是正）を"
                "この形では支持できない。ここで撤退するのが正しい。"
            )
            return 1

        # 合格した場合、フル devデータで導出した最終表を保存（phase2の出発点）
        final_labels = label_fine_regimes(fs_dev, **DEFAULT_LABELER_PARAMS)
        final_table = derive_router_table(
            fs_dev,
            final_labels,
            end=fs_dev.n_bars,
            labeler_params=dict(DEFAULT_LABELER_PARAMS),
            min_regime_bars=args.min_regime_bars,
        )
        final_table.save(router_config_path)
        print(f"Router config -> {router_config_path}")
        return 0

    print(f"[STOP] --stage {args.stage} は未実装です。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
