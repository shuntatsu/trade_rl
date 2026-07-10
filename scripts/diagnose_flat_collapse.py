"""
RL方策のflat崩壊 原因切り分けアブレーション: 真の2x2（no_trade_band x lambda_turnover）

背景: 実データ実行(output/rl_gbm_signal.log等)でPPOが繰り返しturnover=0の
完全flatに収束した。最初のA-E版アブレーションで、training_engine.py::
build_post_processor() の no_trade_band=0.04 がハードコードされ
--min-trade-delta と一切連動していないバグを発見した（本番修正済み、
build_post_processor参照）。旧C実験（min_trade_delta=0のつもり）は実際には
A実験と全く同じ環境（no_trade_band=0.04のまま）で学習しており、新しい情報を
含んでいなかった。

このスクリプトは修正済みの build_post_processor/build_env_kwargs を直接
呼び出し（本番と同一の経路、二重実装を避ける）、no_trade_band と
lambda_turnover を独立に振る真の2x2を行う:
  A       : no_trade_band=0.04(既定), lambda_turnover=0.04(既定)  現状再現
  B       : no_trade_band=0.04(既定), lambda_turnover=0            回転罰則のみ除去
  C_true  : no_trade_band=0,          lambda_turnover=0.04         バンドのみ除去
  F       : no_trade_band=0,          lambda_turnover=0            両方除去

ema_alpha/target_vol/postproc="full"は全実験で固定し、no_trade_bandと
lambda_turnoverだけを独立変数にする（legacy切替はema_alphaも同時に変えて
しまい単一変数比較にならないため、この2x2には含めない）。

env.step() 自体は変更せず(本番学習経路への影響を避ける)、学習済み方策を
検証スライスで決定的にロールアウトし、生行動・執行後ウェイト・報酬内訳
(gross_pnl/cost/funding/turnover罰則)を外側で再計算して記録する。
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mars_lite.pipeline.dataset_builder import build_feature_set
from mars_lite.pipeline.training_engine import (
    build_env_kwargs,
    build_post_processor,
    train_ppo,
)
from mars_lite.trading.execution import make_execution_model
from mars_lite.trading.pipeline import DecisionPipeline, MarketView, PortfolioState

EXPERIMENTS = {
    "A_current": dict(min_trade_delta=0.04, lambda_turnover=0.04, bc_warmstart=True),
    "B_no_turnover_penalty": dict(
        min_trade_delta=0.04, lambda_turnover=0.0, bc_warmstart=True
    ),
    "C_true_no_band": dict(
        min_trade_delta=0.0, lambda_turnover=0.04, bc_warmstart=True
    ),
    "F_no_band_no_penalty": dict(
        min_trade_delta=0.0, lambda_turnover=0.0, bc_warmstart=True
    ),
}


def build_pp_and_ekw(cfg: dict):
    """本番の build_post_processor/build_env_kwargs をそのまま使う
    （二重実装を避け、修正済みの単一経路で検証する）。"""
    ds_args = SimpleNamespace(
        postproc="full",
        min_trade_delta=cfg["min_trade_delta"],
        target_vol=0.5,
        beta_neutral=False,
        base_timeframe="1h",
        lambda_turnover=cfg["lambda_turnover"],
        reward_scale=100.0,
        fee_profile="taker",
        htf_gate=False,
        decision_every=1,
        scan_horizons=False,
    )
    pp = build_post_processor(ds_args)
    ekw = build_env_kwargs(ds_args, pp)
    return pp, ekw


def diagnose_rollout(agent, fs, ekw, vol_lookback: int) -> dict:
    """学習済み方策を決定的にロールアウトし、生行動・執行ウェイト・報酬内訳を記録する。

    env.step()を再実装せず、DecisionPipeline/execution modelを直接呼んで
    同一の計算を外側から行う(診断専用、本番経路には影響しない)。
    """
    from mars_lite.env.portfolio_env import PortfolioTradingEnv

    env = PortfolioTradingEnv(fs, episode_bars=fs.n_bars - 2, **ekw)
    obs, _ = env.reset(options={"start_idx": 0})

    pp = ekw.get("post_processor")
    exec_model = make_execution_model(
        fee_rate=ekw.get("fee_rate", 0.0005),
        spread_rate=ekw.get("spread_rate", 0.0002),
        impact_rate=ekw.get("impact_rate", 0.0001),
    )
    pipeline = DecisionPipeline(
        post_processor=pp, min_trade_delta=ekw.get("min_trade_delta", 0.04)
    )

    rows = []
    prev = np.zeros(fs.n_symbols)
    t = env.start_idx
    done = False
    collapse_bar = None
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        raw = np.asarray(action, dtype=np.float64).flatten()
        proj = env.project_weights(raw)
        state = PortfolioState(weights=prev, portfolio_value=env.portfolio_value)
        market = MarketView.from_feature_set(fs, t, vol_lookback=vol_lookback)
        target, _ = pipeline.target_weights(proj, state, market)

        delta = target - prev
        turnover = float(np.abs(delta).sum())
        cost = exec_model.cost_fraction(delta)
        r_vec = fs.close[t + 1] / fs.close[t] - 1.0
        funding = float(np.sum(target * fs.funding_rate[t + 1]))
        gross_pnl = float(np.dot(target, r_vec))
        net = gross_pnl - cost - funding
        turnover_penalty = ekw.get("lambda_turnover", 0.0) * turnover

        rows.append(
            {
                "t": t,
                "raw_action_absmean": float(np.abs(raw).mean()),
                "proj_gross": float(np.abs(proj).sum()),
                "executed_gross": float(np.abs(target).sum()),
                "turnover": turnover,
                "gross_pnl": gross_pnl,
                "cost": cost,
                "funding": funding,
                "net": net,
                "turnover_penalty": turnover_penalty,
            }
        )
        if collapse_bar is None and float(np.abs(target).sum()) < 1e-6:
            collapse_bar = t

        obs, _, term, trunc, _info = env.step(action)
        prev = target
        t += 1
        done = term or trunc

    executed_gross = np.array([r["executed_gross"] for r in rows])
    # 「そこから先ずっとgross<1e-6」の最初のバー(真の恒久崩壊点)
    permanent_collapse = None
    below = executed_gross < 1e-6
    for i in range(len(below)):
        if below[i:].all():
            permanent_collapse = rows[i]["t"]
            break

    return {
        "n_bars": len(rows),
        "raw_action_absmean_mean": float(
            np.mean([r["raw_action_absmean"] for r in rows])
        ),
        "proj_gross_mean": float(np.mean([r["proj_gross"] for r in rows])),
        "executed_gross_mean": float(np.mean(executed_gross)),
        "executed_gross_final_100_mean": float(np.mean(executed_gross[-100:])),
        "turnover_total": float(np.sum([r["turnover"] for r in rows])),
        "gross_pnl_total": float(np.sum([r["gross_pnl"] for r in rows])),
        "cost_total": float(np.sum([r["cost"] for r in rows])),
        "funding_total": float(np.sum([r["funding"] for r in rows])),
        "turnover_penalty_total": float(np.sum([r["turnover_penalty"] for r in rows])),
        "net_total_return_approx": float(np.sum([r["net"] for r in rows])),
        "first_zero_gross_bar": collapse_bar,
        "permanent_collapse_bar": permanent_collapse,
        "pct_bars_zero_gross": float(np.mean(executed_gross < 1e-6)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="flat崩壊アブレーション A-E")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--target", default="cs_demean")
    ap.add_argument("--warmup-days", type=float, default=100)
    ap.add_argument("--output", default="./output/flat_collapse_ablation")
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS.keys()))
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ds_args = SimpleNamespace(
        source="postgres",
        symbols=None,
        pg_dsn=None,
        pg_source="binance",
        pg_derivatives_source=None,
        base_timeframe="1h",
        warmup_days=args.warmup_days,
        days=240,
    )
    fs = build_feature_set(ds_args)
    holdout_start = int(fs.n_bars * 0.85)
    fs_dev = fs.slice(0, holdout_start)
    split = int(fs_dev.n_bars * 0.8)
    train_fs = fs_dev.slice(0, split)
    test_fs = fs_dev.slice(split + 24, fs_dev.n_bars)
    print(f"train={train_fs.n_bars} test={test_fs.n_bars}", flush=True)

    results = {}
    for name in args.experiments:
        cfg = EXPERIMENTS[name]
        print(f"\n{'=' * 70}\n実験 {name}: {cfg}\n{'=' * 70}", flush=True)
        pp, ekw = build_pp_and_ekw(cfg)
        agent = train_ppo(
            fs=train_fs,
            timesteps=args.timesteps,
            seed=args.seed,
            verbose=1,
            bc_warmstart=cfg["bc_warmstart"],
            horizon=args.horizon,
            signal_target=args.target,
            **ekw,
        )
        vol_lookback = pp.cfg.vol_lookback if hasattr(pp, "cfg") else 0
        diag = diagnose_rollout(agent, test_fs, ekw, vol_lookback)
        results[name] = diag
        print(f"\n[{name}] 診断結果:")
        for k, v in diag.items():
            print(f"  {k}: {v}")

    with open(out / "ablation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nReport -> {out / 'ablation_report.json'}")

    print(
        f"\n{'=' * 70}\nサマリー（実行gross=0となったバー割合・恒久崩壊点）\n{'=' * 70}"
    )
    for name, d in results.items():
        print(
            f"  {name:24s} pct_zero_gross={d['pct_bars_zero_gross']:.1%} "
            f"permanent_collapse_bar={d['permanent_collapse_bar']} "
            f"executed_gross_mean={d['executed_gross_mean']:.3f} "
            f"net_total_return_approx={d['net_total_return_approx']:+.2%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
