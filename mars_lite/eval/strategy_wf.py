"""
ルールベース戦略候補の多期間walk-forward審判ハーネス

背景: 単一holdout期間だけでtrend_followingとの優劣を判断すると、その期間が
たまたま強トレンド期（今回の実データ検証がまさにそう）だったのか、一般的な
傾向なのかを区別できない。「最強のトレードシステム」を名乗るには、複数の
非重複test区間（fold）・コスト2倍・DSR（多重検定補正）・bootstrap有意性検定
を通過することを事前登録済みの基準（STRATEGY_GATE_CRITERIA）で機械判定し、
合格者だけが最後に未接触holdoutを一度だけ見る、という規律を守る。

fold分割は mars_lite.eval.walk_forward / scripts/run_router.py と同じ流儀
（[0.4n, n] を等分、非重複）。DSRのtrial_sharpesにはこのモジュールで実際に
試した全候補×全fold×全グリッドのSharpeを累積して渡すこと（呼び出し側の責務）。
"""

import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from mars_lite.eval.bootstrap_eval import bootstrap_sharpe_difference
from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.learning.baselines import StrategyResult, WeightFn, simulate_strategy
from mars_lite.trading.post_processor import BARS_PER_YEAR_1H
from mars_lite.utils.metrics import deflated_sharpe_ratio

BASELINE_NAME = "trend_following"

# 事前登録合否基準（実行前にコミット。実行後の後出し変更は禁止）。
STRATEGY_GATE_CRITERIA = {
    "trend_v2": {
        "min_folds_beat_tf": 4,
        "min_folds_total": 6,
        "min_median_uplift_pt": 1.0,
        "require_positive_cost2x_median": True,
        "min_dsr": 0.90,
        "min_bootstrap_ci_lower": -0.25,
    },
    "carry": {
        "require_positive_cost2x_median": True,
        "min_median_sharpe": 0.5,
        "max_tf_correlation": 0.3,
        "min_dsr": 0.90,
    },
    "combo": {
        "min_median_uplift_vs_tf_pt": 0.0,
        "min_median_uplift_vs_trend_v2_pt": 0.0,
        "require_maxdd_improvement": True,
        "require_positive_cost2x_median": True,
    },
    # crowding（建玉クラウディング）の事前登録基準。市場中立・低回転の
    # クロスセクショナル戦略として、コスト2倍でも頑健に正であることを要求する。
    "crowding": {
        "min_folds_beat_tf": 5,
        "min_folds_total": 6,
        "require_positive_cost2x_median": True,
        "min_median_sharpe_cost2x": 0.5,
        "min_bootstrap_ci_lower": 0.0,
    },
}


def fold_edges(n_bars: int, n_folds: int) -> np.ndarray:
    """[0.4n, n] を n_folds+1 点で等分（walk_forward.py/run_router.pyと同じ流儀）"""
    return np.linspace(int(n_bars * 0.4), n_bars, n_folds + 1).astype(int)


def _bar_returns(equity_curve: np.ndarray) -> np.ndarray:
    eq = np.asarray(equity_curve, dtype=np.float64)
    if len(eq) < 2:
        return np.zeros(1)
    return np.diff(eq) / eq[:-1]


def run_strategy_walk_forward(
    fs: FeatureSet,
    weight_fns: Dict[str, WeightFn],
    n_folds: int = 6,
    purge: int = 24,
    cost_multipliers: Tuple[float, ...] = (1.0, 2.0),
    fee_rate: float = 0.0005,
    spread_rate: float = 0.0002,
    impact_rate: float = 0.0001,
    min_trade_delta: float = 0.02,
    baseline_name: str = BASELINE_NAME,
) -> dict:
    """weight_fns を n_folds 個の非重複test区間 × cost_multipliers で評価する。

    baseline_name が weight_fns に無ければ mars_lite.learning.baselines.BASELINES
    から自動追加する（trend_followingを常時併記するため）。

    Returns:
        {
          "config": {...},
          "folds": [{"fold", "train_bars", "test_bars",
                      "by_cost": {cost_mult: {name: StrategyResult.to_dict()}}}, ...],
          "fold_returns": {cost_mult: {name: [np.ndarray, ...]}},  # DSR/bootstrap用
        }
    """
    if baseline_name not in weight_fns:
        from mars_lite.learning.baselines import BASELINES

        weight_fns = {**weight_fns, baseline_name: BASELINES[baseline_name]}

    edges = fold_edges(fs.n_bars, n_folds)
    folds_out: List[dict] = []
    fold_returns: Dict[float, Dict[str, List[np.ndarray]]] = {
        cm: {name: [] for name in weight_fns} for cm in cost_multipliers
    }

    for k in range(n_folds):
        train_end = int(edges[k])
        test_start = train_end + purge
        test_end = int(edges[k + 1])
        if test_end - test_start < 50 or train_end < 200:
            continue
        test_fs = fs.slice(test_start, test_end)

        by_cost: Dict[float, Dict[str, dict]] = {}
        for cm in cost_multipliers:
            by_name = {}
            for name, wfn in weight_fns.items():
                res = simulate_strategy(
                    test_fs,
                    wfn,
                    name=name,
                    fee_rate=fee_rate,
                    spread_rate=spread_rate,
                    impact_rate=impact_rate,
                    min_trade_delta=min_trade_delta,
                    cost_multiplier=cm,
                )
                by_name[name] = res.to_dict()
                fold_returns[cm][name].append(_bar_returns(res.equity_curve))
            by_cost[cm] = by_name
        folds_out.append(
            {
                "fold": k,
                "train_bars": train_end,
                "test_bars": test_fs.n_bars,
                "by_cost": by_cost,
            }
        )

    return {
        "config": {
            "n_folds": n_folds,
            "purge": purge,
            "cost_multipliers": list(cost_multipliers),
            "n_bars_total": fs.n_bars,
        },
        "folds": folds_out,
        "fold_returns": fold_returns,
    }


def summarize(report: dict, baseline_name: str = BASELINE_NAME) -> Dict[str, dict]:
    """fold結果から戦略ごとの中央値・trend_following対比・相関を集計する。"""
    cost_mults = report["config"]["cost_multipliers"]
    names = list(report["fold_returns"][cost_mults[0]].keys())
    out: Dict[str, dict] = {}
    for name in names:
        entry: Dict[float, dict] = {}
        for cm in cost_mults:
            cand = np.array(
                [f["by_cost"][cm][name]["total_return"] for f in report["folds"]]
            )
            base = np.array(
                [
                    f["by_cost"][cm][baseline_name]["total_return"]
                    for f in report["folds"]
                ]
            )
            sharpes = np.array(
                [f["by_cost"][cm][name]["sharpe"] for f in report["folds"]]
            )
            maxdds = np.array(
                [f["by_cost"][cm][name]["max_drawdown"] for f in report["folds"]]
            )
            uplift_pt = (cand - base) * 100.0
            corr = 0.0
            if len(cand) > 1 and cand.std() > 1e-12 and base.std() > 1e-12:
                corr = float(np.corrcoef(cand, base)[0, 1])
            entry[cm] = {
                "median_return": float(np.median(cand)) if len(cand) else 0.0,
                "median_sharpe": float(np.median(sharpes)) if len(sharpes) else 0.0,
                "median_maxdd": float(np.median(maxdds)) if len(maxdds) else 0.0,
                "n_folds_beat_baseline": int(np.sum(cand > base)),
                "n_folds_total": int(len(cand)),
                "median_uplift_pt": float(np.median(uplift_pt)) if len(cand) else 0.0,
                "correlation_with_baseline": corr,
            }
        out[name] = entry
    return out


def compute_dsr(
    report: dict, name: str, cost_multiplier: float, trial_sharpes: List[float]
) -> dict:
    """指定戦略の全fold代表リターンを連結した honest OOS 系列で DSR を計算する。"""
    rets_list = report["fold_returns"][cost_multiplier][name]
    oos = np.concatenate(rets_list) if rets_list else np.zeros(1)
    return deflated_sharpe_ratio(
        oos, trial_sharpes, annualization_factor=BARS_PER_YEAR_1H
    )


def compute_bootstrap_vs_baseline(
    report: dict,
    name: str,
    cost_multiplier: float,
    baseline_name: str = BASELINE_NAME,
    seed: Optional[int] = None,
) -> dict:
    """指定戦略とbaselineの全fold代表リターンを連結し、bootstrap Sharpe差を測る。"""
    cand = np.concatenate(report["fold_returns"][cost_multiplier][name])
    base = np.concatenate(report["fold_returns"][cost_multiplier][baseline_name])
    n = min(len(cand), len(base))
    return bootstrap_sharpe_difference(
        cand[:n], base[:n], seed=seed, annualization_factor=BARS_PER_YEAR_1H
    )


def judge_trend_v2(
    summary: Dict[str, dict], dsr: dict, bootstrap: dict, name: str = "trend_v2"
) -> dict:
    crit = STRATEGY_GATE_CRITERIA["trend_v2"]
    s2 = summary[name][2.0]
    checks = {
        "folds_beat_tf": bool(
            s2["n_folds_beat_baseline"] >= crit["min_folds_beat_tf"]
            and s2["n_folds_total"] >= crit["min_folds_total"]
        ),
        "median_uplift_pt": bool(
            s2["median_uplift_pt"] >= crit["min_median_uplift_pt"]
        ),
        "cost2x_median_positive": bool(s2["median_return"] > 0.0),
        "dsr": bool(dsr["dsr"] >= crit["min_dsr"]),
        "bootstrap_ci_lower": bool(
            bootstrap["lower_ci"] >= crit["min_bootstrap_ci_lower"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "criteria": crit}


def judge_carry(summary: Dict[str, dict], dsr: dict, name: str = "carry") -> dict:
    crit = STRATEGY_GATE_CRITERIA["carry"]
    s2 = summary[name][2.0]
    checks = {
        "cost2x_median_positive": bool(s2["median_return"] > 0.0),
        "median_sharpe": bool(s2["median_sharpe"] >= crit["min_median_sharpe"]),
        "low_tf_correlation": bool(
            abs(s2["correlation_with_baseline"]) < crit["max_tf_correlation"]
        ),
        "dsr": bool(dsr["dsr"] >= crit["min_dsr"]),
    }
    return {"passed": all(checks.values()), "checks": checks, "criteria": crit}


def judge_crowding(
    summary: Dict[str, dict], bootstrap: dict, name: str = "crowding"
) -> dict:
    crit = STRATEGY_GATE_CRITERIA["crowding"]
    s2 = summary[name][2.0]
    checks = {
        "folds_beat_tf": bool(
            s2["n_folds_beat_baseline"] >= crit["min_folds_beat_tf"]
            and s2["n_folds_total"] >= crit["min_folds_total"]
        ),
        "cost2x_median_positive": bool(s2["median_return"] > 0.0),
        "median_sharpe_cost2x": bool(
            s2["median_sharpe"] >= crit["min_median_sharpe_cost2x"]
        ),
        "bootstrap_ci_lower": bool(
            bootstrap["lower_ci"] >= crit["min_bootstrap_ci_lower"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "criteria": crit}


def judge_combo(
    summary: Dict[str, dict],
    name: str = "combo",
    trend_v2_name: str = "trend_v2",
    baseline_name: str = BASELINE_NAME,
) -> dict:
    crit = STRATEGY_GATE_CRITERIA["combo"]
    s2 = summary[name][2.0]
    base2 = summary[baseline_name][2.0]
    tv2 = summary[trend_v2_name][2.0]
    uplift_vs_tf_pt = (s2["median_return"] - base2["median_return"]) * 100.0
    uplift_vs_trend_v2_pt = (s2["median_return"] - tv2["median_return"]) * 100.0
    maxdd_improved = bool(
        s2["median_maxdd"] <= base2["median_maxdd"]
        and s2["median_maxdd"] <= tv2["median_maxdd"]
    )
    checks = {
        "uplift_vs_tf": bool(uplift_vs_tf_pt >= crit["min_median_uplift_vs_tf_pt"]),
        "uplift_vs_trend_v2": bool(
            uplift_vs_trend_v2_pt >= crit["min_median_uplift_vs_trend_v2_pt"]
        ),
        "maxdd_improved": maxdd_improved,
        "cost2x_median_positive": bool(s2["median_return"] > 0.0),
    }
    return {"passed": all(checks.values()), "checks": checks, "criteria": crit}


def save_report(report: dict, path: Path) -> None:
    """fold_returns（生配列、JSON非対応）を除いた集計結果を保存する。"""
    safe = {"config": report["config"], "folds": report["folds"]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2, ensure_ascii=False)


def run_holdout_once(
    fs_holdout: FeatureSet,
    weight_fn: WeightFn,
    output_dir: Path,
    name: str,
    trial_sharpes: List[float],
    baseline_name: str = BASELINE_NAME,
    cost_multiplier: float = 2.0,
    fee_rate: float = 0.0005,
    spread_rate: float = 0.0002,
    impact_rate: float = 0.0001,
    min_trade_delta: float = 0.02,
) -> dict:
    """未接触holdoutで候補とbaselineを1回だけ評価する。

    lockbox（mars_lite.pipeline.phases）と同じマーカー方式: 既にholdoutを
    見ていたら警告する（ブロックはしない。判断の記録として残す）。
    """
    from mars_lite.learning.baselines import BASELINES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "strategy_holdout_used.marker"
    if marker.exists():
        print(
            f"\n[Holdout] 警告: このholdoutは既に {marker.read_text().strip()} に"
            "一度使用済みです。繰り返し見て判断を調整するのは過学習の抜け道です。"
        )

    fee_kwargs = dict(
        fee_rate=fee_rate,
        spread_rate=spread_rate,
        impact_rate=impact_rate,
        min_trade_delta=min_trade_delta,
        cost_multiplier=cost_multiplier,
    )
    res = simulate_strategy(fs_holdout, weight_fn, name=name, **fee_kwargs)
    base_res = simulate_strategy(
        fs_holdout, BASELINES[baseline_name], name=baseline_name, **fee_kwargs
    )
    cand_rets = _bar_returns(res.equity_curve)
    base_rets = _bar_returns(base_res.equity_curve)
    n = min(len(cand_rets), len(base_rets))

    dsr = deflated_sharpe_ratio(
        cand_rets, trial_sharpes, annualization_factor=BARS_PER_YEAR_1H
    )
    bootstrap = bootstrap_sharpe_difference(
        cand_rets[:n], base_rets[:n], annualization_factor=BARS_PER_YEAR_1H
    )

    report = {
        "candidate": res.to_dict(),
        "baseline": base_res.to_dict(),
        "beat_baseline": bool(res.total_return > base_res.total_return),
        "dsr": dsr,
        "bootstrap_vs_baseline": bootstrap,
    }
    with open(output_dir / "holdout_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    marker.write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())
    return report
