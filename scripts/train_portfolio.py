"""
ポートフォリオRL学習スクリプト

フェーズ:
    p0    : 健全性試験。アルファ注入合成データ（陽性対照）と純ノイズ（陰性対照）
            の両方で学習し、①陽性でベースライン超え ②陰性で低回転 を確認する
    train : 指定ソースで学習（P2。実データはローカルPCで --source csv/postgres）
    wf    : ウォークフォワード検証（P3）

使い方:
    python scripts/train_portfolio.py --phase p0 --timesteps 300000
    python scripts/train_portfolio.py --phase train --source csv --data ./data --timesteps 2000000
    python scripts/train_portfolio.py --phase wf --source csv --data ./data
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mars_lite.data.sources import create_source, SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline, FeatureSet
from mars_lite.features.signal_check import run_signal_check
from mars_lite.env.portfolio_env import PortfolioTradingEnv
from mars_lite.models.portfolio_extractor import PortfolioExtractor
from mars_lite.learning.baselines import run_all_baselines
from mars_lite.eval.walk_forward import (
    evaluate_agent_on_slice, run_walk_forward, plot_comparison,
)

DEFAULT_SYMBOLS = [
    "BTCUSDT", "XRPUSDT", "SUIUSDT", "BNBUSDT", "ETHUSDT", "PAXGUSDT", "ETHBTC",
]


def make_env_fns(fs: FeatureSet, n_envs: int, seed: int, **env_kwargs):
    from stable_baselines3.common.monitor import Monitor

    def make_one(rank: int):
        def _init():
            env = PortfolioTradingEnv(fs, **env_kwargs)
            env.reset(seed=seed + rank)
            return Monitor(env)
        return _init

    return [make_one(i) for i in range(n_envs)]


def train_ppo(
    fs: FeatureSet,
    timesteps: int = 300_000,
    seed: int = 0,
    n_envs: int = 8,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.002,
    gamma: float = 0.995,
    verbose: int = 0,
    callbacks=None,
    val_fs: FeatureSet = None,
    val_eval_freq: int = 20_000,
    **env_kwargs,
):
    """FeatureSetでPPOを学習して返す

    val_fs を渡すと検証スライスで定期評価し、最良時点のパラメータを
    最終モデルとして採用する（小データへの過学習対策）。
    val_fs 省略時は fs の末尾15%を自動で検証に割く。
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CallbackList
    from mars_lite.learning.val_selection import ValSelectionCallback

    if val_fs is None and fs.n_bars > 400:
        cut = int(fs.n_bars * 0.85)
        val_fs = fs.slice(cut, fs.n_bars)
        fs = fs.slice(0, cut)

    env = DummyVecEnv(make_env_fns(fs, n_envs, seed, **env_kwargs))
    probe = PortfolioTradingEnv(fs, **env_kwargs)

    policy_kwargs = {
        "features_extractor_class": PortfolioExtractor,
        "features_extractor_kwargs": probe.obs_layout,
        "net_arch": dict(pi=[64, 64], vf=[64, 64]),
    }

    def lr_schedule(progress_remaining: float) -> float:
        return learning_rate * progress_remaining

    agent = PPO(
        "MlpPolicy", env,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_schedule,
        n_steps=256, batch_size=256, n_epochs=6,
        gamma=gamma, gae_lambda=0.95,
        ent_coef=ent_coef, vf_coef=0.5, max_grad_norm=0.5,
        seed=seed, device="cpu", verbose=verbose,
    )
    val_cb = None
    if val_fs is not None:
        val_cb = ValSelectionCallback(
            val_fs, eval_freq=val_eval_freq, env_kwargs=env_kwargs,
            verbose=verbose,
        )
        callbacks = CallbackList(
            ([callbacks] if callbacks is not None else []) + [val_cb]
        )

    agent.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)

    if val_cb is not None:
        agent = val_cb.restore_best(agent)
        if verbose:
            print(f"[train_ppo] Restored best-val model (score={val_cb.best_score:+.4f})")
    return agent


def report_comparison(agent_res: dict, baselines: dict, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"{'strategy':<18} {'return':>9} {'sharpe':>8} {'maxDD':>7} {'turnover':>9}")
    print(f"{'RL Agent':<18} {agent_res['total_return']:>+8.2%} "
          f"{agent_res['sharpe']:>8.2f} {agent_res['max_drawdown']:>7.2%} "
          f"{agent_res['turnover_total']:>9.1f}")
    for name, r in baselines.items():
        d = r.to_dict() if hasattr(r, "to_dict") else r
        print(f"{d['name']:<18} {d['total_return']:>+8.2%} "
              f"{d['sharpe']:>8.2f} {d['max_drawdown']:>7.2%} "
              f"{d['turnover_total']:>9.1f}")


def build_feature_set(args) -> FeatureSet:
    if args.source == "synthetic":
        source = SyntheticSource(
            n_days=args.days, alpha=args.alpha,
            alpha_strength=args.alpha_strength, seed=args.seed,
        )
        symbols = source.symbols
    else:
        symbols = args.symbols or DEFAULT_SYMBOLS
        kwargs = {"data_dir": args.data} if args.source == "csv" else {}
        source = create_source(args.source, symbols, **kwargs)
    return FeaturePipeline(symbols).build(source)


def phase_p0(args, output_dir: Path) -> None:
    """P0健全性試験: 陽性対照（アルファ有）と陰性対照（ノイズ）"""
    results = {}

    for label, alpha in [("positive(alpha=cross)", "cross"), ("negative(alpha=none)", "none")]:
        print(f"\n{'=' * 60}\nP0 {label}\n{'=' * 60}")
        source = SyntheticSource(
            n_days=args.days, alpha=alpha,
            alpha_strength=args.alpha_strength, seed=args.seed,
        )
        fs = FeaturePipeline(source.symbols).build(source)

        # ICゲート（陽性はPASS、陰性はFAILが期待値）
        ic = run_signal_check(fs)
        print(ic.summary())

        # train/test 時系列分割（70/30、purge 24本）
        split = int(fs.n_bars * 0.7)
        train_fs = fs.slice(0, split)
        test_fs = fs.slice(split + 24, fs.n_bars)

        print(f"\nTraining PPO: {args.timesteps:,} steps "
              f"(train {train_fs.n_bars} bars, test {test_fs.n_bars} bars)...")
        agent = train_ppo(fs=train_fs, timesteps=args.timesteps, seed=args.seed,
                          verbose=args.verbose)

        agent_res = evaluate_agent_on_slice(agent, test_fs)
        baselines = run_all_baselines(test_fs)
        report_comparison(agent_res, baselines, f"OOS comparison: {label}")

        plot_comparison(
            agent_res, baselines,
            output_dir / f"p0_{alpha}_equity.png",
            title=f"P0 {label}: RL vs Baselines (OOS)",
        )

        agent_res_slim = {k: v for k, v in agent_res.items() if k != "equity_curve"}
        results[label] = {
            "signal_gate": ic.to_dict(),
            "agent": agent_res_slim,
            "baselines": {k: v.to_dict() for k, v in baselines.items()},
        }
        agent.save(str(output_dir / f"p0_{alpha}_model"))

    # ゲート判定
    pos = results["positive(alpha=cross)"]
    neg = results["negative(alpha=none)"]
    pos_beats_bh = pos["agent"]["total_return"] > pos["baselines"]["equal_weight_bh"]["total_return"]
    pos_beats_flat = pos["agent"]["total_return"] > 0
    neg_low_activity = neg["agent"]["turnover_total"] < pos["agent"]["turnover_total"] * 0.5 \
        or abs(neg["agent"]["total_return"]) < 0.05

    gate = {
        "positive_beats_buy_and_hold": bool(pos_beats_bh),
        "positive_beats_flat": bool(pos_beats_flat),
        "negative_stays_quiet": bool(neg_low_activity),
        "P0_PASSED": bool(pos_beats_bh and pos_beats_flat and neg_low_activity),
    }
    results["gate"] = gate

    with open(output_dir / "p0_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}\nP0 GATE: {'PASSED' if gate['P0_PASSED'] else 'FAILED'}")
    for k, v in gate.items():
        print(f"  {k}: {v}")
    print(f"Report: {output_dir / 'p0_report.json'}")


def phase_train(args, output_dir: Path) -> None:
    """P2: 指定ソースで学習・評価"""
    fs = build_feature_set(args)
    print(f"FeatureSet: {fs.n_bars} bars x {fs.n_symbols} symbols x {fs.n_features} features")

    ic = run_signal_check(fs)
    print(ic.summary())
    if not ic.passed and not args.skip_gate:
        print("\n[STOP] ゲート1不合格: 特徴量に予測力がありません。"
              "RL学習をスキップします（--skip-gate で強制続行可）。")
        return

    split = int(fs.n_bars * 0.8)
    train_fs = fs.slice(0, split)
    test_fs = fs.slice(split + 24, fs.n_bars)

    print(f"Training PPO: {args.timesteps:,} steps...")
    agent = train_ppo(fs=train_fs, timesteps=args.timesteps, seed=args.seed,
                      verbose=args.verbose)

    agent_res = evaluate_agent_on_slice(agent, test_fs)
    baselines = run_all_baselines(test_fs)
    report_comparison(agent_res, baselines, "OOS comparison")
    plot_comparison(agent_res, baselines, output_dir / "train_equity.png")

    agent.save(str(output_dir / "portfolio_model"))
    with open(output_dir / "train_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "signal_gate": ic.to_dict(),
            "agent": {k: v for k, v in agent_res.items() if k != "equity_curve"},
            "baselines": {k: v.to_dict() for k, v in baselines.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"Model & report -> {output_dir}")


def phase_wf(args, output_dir: Path) -> None:
    """P3: ウォークフォワード検証（複数シード、コスト感度）"""
    fs = build_feature_set(args)
    print(f"FeatureSet: {fs.n_bars} bars x {fs.n_symbols} symbols")

    def train_fn(train_fs: FeatureSet, seed: int):
        return train_ppo(fs=train_fs, timesteps=args.timesteps, seed=seed)

    for cost_mult in [1.0, 2.0]:
        print(f"\n--- Walk-forward (cost x{cost_mult}) ---")
        report = run_walk_forward(
            fs, train_fn,
            n_folds=args.folds,
            seeds=list(range(args.n_seeds)),
            cost_multiplier=cost_mult,
        )
        path = output_dir / f"walk_forward_cost{cost_mult:.0f}x.json"
        report.save(path)
        print(json.dumps(report.summary(), indent=2))
        print(f"Report -> {path}")


def main():
    parser = argparse.ArgumentParser(description="ポートフォリオRL学習")
    parser.add_argument("--phase", choices=["p0", "train", "wf"], default="p0")
    parser.add_argument("--source", choices=["synthetic", "csv", "postgres"],
                        default="synthetic")
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--symbols", type=str, nargs="+", default=None)
    parser.add_argument("--days", type=int, default=120, help="syntheticの生成日数")
    parser.add_argument("--alpha", default="cross", choices=["none", "cross", "meanrev"])
    parser.add_argument("--alpha-strength", type=float, default=0.002)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--output", type=str, default="./output/portfolio")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--skip-gate", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "p0":
        phase_p0(args, output_dir)
    elif args.phase == "train":
        phase_train(args, output_dir)
    else:
        phase_wf(args, output_dir)


if __name__ == "__main__":
    main()
