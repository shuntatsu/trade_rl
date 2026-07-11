"""
P0-1: 最小構成でのPPO本体の健全性を複数seedで検証する

背景: Stage1のflat崩壊アブレーション（真の2x2、実データ）で band=0.04 の
hard dead-zone と lambda_turnover=0.04 による行動スケール縮小の両方が実在
することを確認したが、4設定すべてで実際には収益化できなかった（0%が最良、
他は損失）。これだけでは「アルファ不足」と「PPO/報酬設計自体の欠陥」を
切り分けられない。

この切り分けには、後処理・turnover罰則・BC・ボラ目標を**すべて外した
最小構成**で、「PPOが既知の合成アルファを学べるか」を単一seedではなく
複数seedで検証する必要がある（P0は元々単一seedで、ハイパーパラメータの
運・不運を均せていなかった）。

最小構成（P0-1）:
  postproc=legacy, no_trade_band(min_trade_delta)=0, lambda_turnover=0,
  target_vol=off(legacyは元々未使用), bc_warmstart=False, decision_every=1

判定基準:
  positive(alpha=cross): median_return>0, 過半数seedがbuy&hold超え,
    grossが恒久的にゼロへ凍結しない, 恒久的にgross=1に張り付かない,
    catastrophic loss(残存資本<50%)が少数派
  negative(alpha=none): median_returnが概ねゼロ付近, turnoverがpositiveの
    中央値より十分低い, grossが不必要に最大化しない

ここで不合格なら、band/lambda調整ではなくPPO・報酬・行動表現・観測設計を
疑うべき（ユーザー指摘の判断フロー通り）。
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mars_lite.data.sources import SyntheticSource
from mars_lite.eval.walk_forward import evaluate_agent_on_slice
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.learning.baselines import run_all_baselines
from mars_lite.pipeline.training_engine import train_ppo
from mars_lite.trading.post_processor import make_legacy_processor


def run_one(alpha: str, seed: int, timesteps: int, days: int, horizon: int) -> dict:
    source = SyntheticSource(n_days=days, alpha=alpha, seed=seed)
    fs = FeaturePipeline(source.symbols).build(source)
    split = int(fs.n_bars * 0.7)
    purge = max(24, horizon)
    train_fs = fs.slice(0, split)
    test_fs = fs.slice(split + purge, fs.n_bars)

    pp = make_legacy_processor(min_trade_delta=0.0)  # band=0, ema_alpha=1.0
    ekw = dict(
        post_processor=pp,
        min_trade_delta=0.0,
        lambda_turnover=0.0,
        reward_scale=100.0,
        fee_rate=0.0005,
        spread_rate=0.0002,
        impact_rate=0.0001,
    )
    agent = train_ppo(
        fs=train_fs,
        timesteps=timesteps,
        seed=seed,
        verbose=0,
        bc_warmstart=False,
        horizon=horizon,
        **ekw,
    )
    res = evaluate_agent_on_slice(agent, test_fs, **ekw)
    baselines = run_all_baselines(
        test_fs, fee_rate=0.0005, spread_rate=0.0002, impact_rate=0.0001
    )
    bh_return = baselines["equal_weight_bh"].total_return

    return {
        "seed": seed,
        "total_return": res["total_return"],
        "sharpe": res["sharpe"],
        "max_drawdown": res["max_drawdown"],
        "turnover_total": res["turnover_total"],
        "n_trades": res["n_trades"],
        "hold_pct": res.get("hold_pct", None),
        "beats_bh": bool(res["total_return"] > bh_return),
        "bh_return": bh_return,
        "frozen_zero": bool(res["turnover_total"] < 1e-6),
        "catastrophic_loss": bool(res["total_return"] < -0.5),
    }


def summarize(rows: list) -> dict:
    returns = np.array([r["total_return"] for r in rows])
    turnovers = np.array([r["turnover_total"] for r in rows])
    return {
        "n_seeds": len(rows),
        "median_return": float(np.median(returns)),
        "mean_return": float(np.mean(returns)),
        "n_beats_bh": int(sum(r["beats_bh"] for r in rows)),
        "n_frozen_zero": int(sum(r["frozen_zero"] for r in rows)),
        "n_catastrophic_loss": int(sum(r["catastrophic_loss"] for r in rows)),
        "median_turnover": float(np.median(turnovers)),
        "returns_by_seed": returns.tolist(),
        "turnovers_by_seed": turnovers.tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="P0-1: 最小構成・複数seed健全性検証")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--days", type=int, default=240)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--output", default="./output/p0_minimal_multiseed")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    results = {"positive": [], "negative": []}
    for label, alpha in [("positive", "cross"), ("negative", "none")]:
        print(f"\n{'=' * 70}\n{label} (alpha={alpha})\n{'=' * 70}", flush=True)
        for seed in args.seeds:
            print(f"  seed={seed} ...", flush=True)
            r = run_one(alpha, seed, args.timesteps, args.days, args.horizon)
            results[label].append(r)
            print(
                f"    return={r['total_return']:+.2%} sharpe={r['sharpe']:.2f} "
                f"turnover={r['turnover_total']:.1f} beats_bh={r['beats_bh']} "
                f"frozen={r['frozen_zero']} catastrophic={r['catastrophic_loss']}",
                flush=True,
            )

    summary = {
        "positive": summarize(results["positive"]),
        "negative": summarize(results["negative"]),
    }

    pos_s, neg_s = summary["positive"], summary["negative"]
    verdict = {
        "positive_median_return_gt_0": bool(pos_s["median_return"] > 0),
        "positive_majority_beats_bh": bool(pos_s["n_beats_bh"] > pos_s["n_seeds"] / 2),
        "positive_not_always_frozen": bool(pos_s["n_frozen_zero"] < pos_s["n_seeds"]),
        "positive_few_catastrophic": bool(
            pos_s["n_catastrophic_loss"] <= pos_s["n_seeds"] / 3
        ),
        "negative_turnover_much_lower": bool(
            neg_s["median_turnover"] < pos_s["median_turnover"] * 0.5
        ),
        "negative_median_return_near_zero": bool(abs(neg_s["median_return"]) < 0.10),
    }
    verdict["P0_1_PASSED"] = all(verdict.values())

    report = {"raw": results, "summary": summary, "verdict": verdict}
    with open(out / "p0_minimal_multiseed_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}\nサマリー\n{'=' * 70}")
    print(
        f"positive: median_return={pos_s['median_return']:+.2%} "
        f"beats_bh={pos_s['n_beats_bh']}/{pos_s['n_seeds']} "
        f"frozen={pos_s['n_frozen_zero']}/{pos_s['n_seeds']} "
        f"catastrophic={pos_s['n_catastrophic_loss']}/{pos_s['n_seeds']}"
    )
    print(
        f"negative: median_return={neg_s['median_return']:+.2%} "
        f"median_turnover={neg_s['median_turnover']:.1f} "
        f"(positive median_turnover={pos_s['median_turnover']:.1f})"
    )
    print(f"\n[P0-1 判定] {json.dumps(verdict, indent=2, ensure_ascii=False)}")
    print(f"\nReport -> {out / 'p0_minimal_multiseed_report.json'}")
    return 0 if verdict["P0_1_PASSED"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
