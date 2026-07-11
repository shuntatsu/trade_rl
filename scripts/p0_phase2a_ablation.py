#!/usr/bin/env python3
"""
Phase 2A: 単独効果の因果的アブレーション (Single-Element Ablations)

検証対象:
  - B0: 最小構成 (ema_alpha=1.0, max_weight=1.0, target_vol=None)
  - C:  集中上限 (ema_alpha=1.0, max_weight=0.4, target_vol=None)
  - E:  EMA平滑 (ema_alpha=0.5, max_weight=1.0, target_vol=None)
  - V:  ボラ目標 (ema_alpha=1.0, max_weight=1.0, target_vol=0.20)

中間ウェイト変換記録・追従率 (Tracking Ratio)・凍結率 (Freeze Rate)・キャップヒット率を算出。
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from mars_lite.data.sources import SyntheticSource
from mars_lite.env.portfolio_env import PortfolioTradingEnv
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.learning.val_selection import sortino_score
from mars_lite.trading.post_processor import PortfolioPostProcessor, PostProcessConfig


class NullPolicy:
    """常にウェイト0（無取引・全額現金保有）を出力する Null Policy"""

    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def predict(self, obs: Any, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        return np.zeros(self.action_dim, dtype=np.float32), None


def make_post_processor(config_id: str) -> PortfolioPostProcessor:
    if config_id == "B0":
        cfg = PostProcessConfig(
            ema_alpha=1.0,
            max_weight=1.0,
            target_vol=None,
            no_trade_band=0.0,
        )
    elif config_id == "C":
        cfg = PostProcessConfig(
            ema_alpha=1.0,
            max_weight=0.4,
            target_vol=None,
            no_trade_band=0.0,
        )
    elif config_id == "E":
        cfg = PostProcessConfig(
            ema_alpha=0.5,
            max_weight=1.0,
            target_vol=None,
            no_trade_band=0.0,
        )
    elif config_id == "V":
        cfg = PostProcessConfig(
            ema_alpha=1.0,
            max_weight=1.0,
            target_vol=0.20,
            no_trade_band=0.0,
        )
    else:
        raise ValueError(f"Unknown config_id: {config_id}")
    return PortfolioPostProcessor(cfg)


def evaluate_policy_detailed(
    agent: Any,
    fs: Any,
    post_processor: PortfolioPostProcessor,
) -> Dict[str, Any]:
    env = PortfolioTradingEnv(
        fs=fs,
        fee_rate=0.0005,
        spread_rate=0.0002,
        impact_rate=0.0001,
        decision_every=1,
        lambda_turnover=0.0,
        post_processor=post_processor,
    )

    obs, _ = env.reset()
    done = False
    tracking_ratios = []
    freezes = 0
    cap_hits = 0
    max_abs_weights = []
    total_steps = 0

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc

        pp_info = info.get("pp_info")
        if pp_info is not None:
            tracking_ratios.append(pp_info.tracking_ratio)
            if pp_info.is_freeze:
                freezes += 1
            if pp_info.is_cap_hit:
                cap_hits += 1
            max_abs_weights.append(pp_info.max_abs_weight)
        total_steps += 1

    ret = float(env.portfolio_value / env.initial_capital - 1.0)
    sortino = float(sortino_score(getattr(env, "_returns_history", [])))
    max_dd = float(env.max_dd)
    turnover = float(env.turnover_total)

    mean_tracking_ratio = float(np.mean(tracking_ratios)) if tracking_ratios else 1.0
    freeze_rate = float(freezes / max(total_steps, 1))
    cap_hit_rate = float(cap_hits / max(total_steps, 1))
    max_abs_weight = float(np.max(max_abs_weights)) if max_abs_weights else 0.0

    return {
        "return": ret,
        "sortino": sortino,
        "max_dd": max_dd,
        "turnover": turnover,
        "tracking_ratio": mean_tracking_ratio,
        "freeze_rate": freeze_rate,
        "cap_hit_rate": cap_hit_rate,
        "max_abs_weight": max_abs_weight,
    }


def run_single_trial(
    config_id: str,
    alpha: str,
    data_seed: int,
    model_seed: int,
    timesteps: int = 10000,
    days: int = 60,
    horizon: int = 24,
    eval_freq: int = 2000,
    min_improvement: float = 0.001,
    max_mdd: float = 0.12,
) -> Dict[str, Any]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    source = SyntheticSource(n_days=days, alpha=alpha, seed=data_seed)
    fs = FeaturePipeline(source.symbols).build(source)

    purge = max(24, horizon)
    b_50 = int(fs.n_bars * 0.50)
    b_68 = int(fs.n_bars * 0.68)
    b_84 = int(fs.n_bars * 0.84)

    train_fs = fs.slice(0, b_50)
    val_select_fs = fs.slice(b_50 + purge, b_68)
    val_confirm_fs = fs.slice(b_68 + purge, b_84)
    test_fs = fs.slice(b_84 + purge, fs.n_bars)

    post_proc = make_post_processor(config_id)

    class CheckpointCallback(BaseCallback):
        def __init__(self, eval_fs, pp, freq):
            super().__init__(verbose=0)
            self.eval_fs = eval_fs
            self.pp = pp
            self.freq = freq
            self.best_score = -999.0
            self.best_agent = None

        def _on_step(self) -> bool:
            if self.n_calls % self.freq == 0:
                res = evaluate_policy_detailed(self.model, self.eval_fs, self.pp)
                score = res["sortino"]
                if score > self.best_score + min_improvement:
                    self.best_score = score
                    import copy
                    self.best_agent = copy.deepcopy(self.model)
            return True

    def make_env():
        return PortfolioTradingEnv(
            fs=train_fs,
            fee_rate=0.0005,
            spread_rate=0.0002,
            impact_rate=0.0001,
            decision_every=1,
            lambda_turnover=0.0,
            post_processor=post_proc,
        )

    vec_env = DummyVecEnv([make_env])
    agent = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        seed=model_seed,
        verbose=0,
    )

    cb = CheckpointCallback(val_select_fs, post_proc, eval_freq)
    agent.learn(total_timesteps=timesteps, callback=cb)

    best_agent = cb.best_agent if cb.best_agent is not None else agent

    # val_confirm での安全性ゲート判定
    val_res_rl = evaluate_policy_detailed(best_agent, val_confirm_fs, post_proc)
    null_policy = NullPolicy(action_dim=fs.n_symbols)
    val_res_null = evaluate_policy_detailed(null_policy, val_confirm_fs, post_proc)

    gate_passed = (
        val_res_rl["return"] >= val_res_null["return"]
        and val_res_rl["sortino"] > 0.0
        and val_res_rl["max_dd"] <= max_mdd
    )

    selected_policy = best_agent if gate_passed else null_policy
    selected_label = "RL" if gate_passed else "Null"

    test_res = evaluate_policy_detailed(selected_policy, test_fs, post_proc)

    return {
        "config_id": config_id,
        "alpha": alpha,
        "data_seed": data_seed,
        "model_seed": model_seed,
        "selected": selected_label,
        "test_return": test_res["return"],
        "test_sortino": test_res["sortino"],
        "test_max_dd": test_res["max_dd"],
        "test_turnover": test_res["turnover"],
        "tracking_ratio": test_res["tracking_ratio"],
        "freeze_rate": test_res["freeze_rate"],
        "cap_hit_rate": test_res["cap_hit_rate"],
        "max_abs_weight": test_res["max_abs_weight"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["B0", "C", "E", "V"])
    parser.add_argument("--alphas", nargs="+", default=["cross", "none"])
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--output", default="output/phase2a_ablation_results.json")
    args = parser.parse_args()

    data_seeds = [42, 43, 44, 45, 46]
    model_seeds = [100, 101, 102, 103, 104]

    all_results = []
    for cfg in args.configs:
        for alpha in args.alphas:
            print(f"\nRunning Config={cfg}, Alpha={alpha} (25 cells)...")
            for ds in data_seeds:
                for ms in model_seeds:
                    res = run_single_trial(
                        config_id=cfg,
                        alpha=alpha,
                        data_seed=ds,
                        model_seed=ms,
                        timesteps=args.timesteps,
                    )
                    all_results.append(res)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n=== Phase 2A Summary Saved to {args.output} ===")
    # 集計サマリー表示
    for alpha in args.alphas:
        print(f"\n--- Alpha: {alpha} ---")
        print(f"{'Config':<8} {'RL_Select%':>10} {'Med_Ret%':>10} {'Med_Sortino':>12} {'Med_MaxDD%':>12} {'Tracking%':>10} {'Freeze%':>10} {'MaxWeight':>10}")
        for cfg in args.configs:
            subset = [r for r in all_results if r["config_id"] == cfg and r["alpha"] == alpha]
            if not subset:
                continue
            rl_pct = sum(1 for r in subset if r["selected"] == "RL") / len(subset) * 100.0
            med_ret = np.median([r["test_return"] for r in subset]) * 100.0
            med_sort = np.median([r["test_sortino"] for r in subset])
            med_dd = np.median([r["test_max_dd"] for r in subset]) * 100.0
            med_track = np.median([r["tracking_ratio"] for r in subset]) * 100.0
            med_freeze = np.median([r["freeze_rate"] for r in subset]) * 100.0
            max_w = np.max([r["max_abs_weight"] for r in subset])
            print(f"{cfg:<8} {rl_pct:>9.1f}% {med_ret:>9.2f}% {med_sort:>12.2f} {med_dd:>11.2f}% {med_track:>9.1f}% {med_freeze:>9.1f}% {max_w:>10.2f}")


if __name__ == "__main__":
    main()
