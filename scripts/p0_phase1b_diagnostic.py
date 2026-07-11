"""
P0-1B: 無信号時の暴走診断および Null Policy を含むモデル選択系の検証スクリプト

検証仕様 (開発試験: 構成A 5x5=25試行):
1. データの4分割と時間的分離:
   Train (0-50%) -> Purge -> val_select (50-68%) -> Purge -> val_confirm (68-84%) -> Purge -> Test (84-100%)
   - val_select で最良 RL チェックポイント (20k間隔) を選択
   - val_confirm (未見検証スライス) で Null Policy との複合ゲート判定を実施
   - Test は最終選定された1つの方策に対して一度だけ評価
2. 複合 Null ゲート条件 (val_confirm 上):
   - Sortino >= min_improvement (既定値 0.5)
   - net total_return >= +0.5% (0.005)
   - max_drawdown <= 5% (0.05)
3. 同一 data_seed での none / cross 価格ノイズ共有
4. 行動およびエクスポージャー統計の記録:
   signed_action_mean, mean_abs_action, action_rms, policy_std_mean, raw_gross, executed_gross, turnover
5. 5x5 マトリクスおよびデータ・モデル効果の分析レポート出力
"""

import argparse
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mars_lite.data.sources import SyntheticSource
from mars_lite.env.portfolio_env import PortfolioTradingEnv
from mars_lite.features.feature_pipeline import FeaturePipeline, FeatureSet
from mars_lite.learning.baselines import run_all_baselines
from mars_lite.learning.val_selection import sortino_score
from mars_lite.pipeline.training_engine import make_env_fns


class NullPolicy:
    """常にウェイト0（無取引・全額現金保有）を出力する Null Policy"""

    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def predict(self, obs: Any, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        return np.zeros(self.action_dim, dtype=np.float32), None


@dataclass
class CheckpointMetric:
    step: int
    val_select_return: float
    val_select_sortino: float
    val_select_turnover: float
    val_confirm_return: float
    val_confirm_sortino: float
    val_confirm_turnover: float


def evaluate_policy_detailed(
    agent: Any, fs: FeatureSet, record_std: bool = False, **env_kwargs
) -> Dict[str, Any]:
    """決定論的評価を行い、リターン・Turnover・詳細な行動統計量を返す"""
    env_kwargs = {
        k: v
        for k, v in env_kwargs.items()
        if k not in ("episode_bars", "regime_start_pool")
    }
    env = PortfolioTradingEnv(fs, episode_bars=fs.n_bars - 2, **env_kwargs)
    obs, _ = env.reset(options={"start_idx": 0})
    done = False

    actions = []
    stds = []
    equity = [env.portfolio_value]
    raw_gross_list = []
    exec_gross_list = []

    has_policy = hasattr(agent, "policy") and hasattr(agent.policy, "get_distribution")

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        actions.append(action)

        if record_std and has_policy:
            import torch

            with torch.no_grad():
                obs_t = agent.policy.obs_to_tensor(obs)[0]
                dist = agent.policy.get_distribution(obs_t)
                stds.append(dist.distribution.stddev.cpu().numpy().flatten())

        obs, _, term, trunc, info = env.step(action)
        equity.append(env.portfolio_value)
        raw_gross_list.append(float(np.sum(np.abs(action))))
        exec_gross_list.append(float(np.sum(np.abs(env.weights))))
        done = term or trunc

    actions_arr = np.array(actions)
    if len(actions_arr) > 0:
        signed_mean = float(np.mean(actions_arr))
        mean_abs = float(np.mean(np.abs(actions_arr)))
        rms = float(np.sqrt(np.mean(actions_arr**2)))
    else:
        signed_mean = mean_abs = rms = 0.0

    if stds and len(stds) > 0:
        stds_arr = np.array(stds)
        std_mean = float(np.mean(stds_arr))
    else:
        std_mean = 0.0

    raw_gross = float(np.mean(raw_gross_list)) if raw_gross_list else 0.0
    exec_gross = float(np.mean(exec_gross_list)) if exec_gross_list else 0.0

    ret = env.portfolio_value / env.initial_capital - 1.0
    sortino = sortino_score(getattr(env, "_returns_history", []))

    return {
        "total_return": float(ret),
        "sortino": float(sortino),
        "sharpe": float(info.get("sharpe", 0.0)),
        "max_drawdown": float(info.get("max_drawdown", 0.0)),
        "turnover_total": float(info.get("turnover_total", 0.0)),
        "n_trades": int(info.get("n_trades", 0)),
        "signed_action_mean": signed_mean,
        "mean_abs_action": mean_abs,
        "action_rms": rms,
        "policy_std_mean": std_mean,
        "raw_gross": raw_gross,
        "executed_gross": exec_gross,
    }


def run_single_trial(
    alpha: str,
    data_seed: int,
    model_seed: int,
    timesteps: int,
    days: int,
    horizon: int,
    eval_freq: int,
    min_improvement: float,
    max_mdd: float = 0.12,
) -> Dict[str, Any]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    from mars_lite.trading.post_processor import make_legacy_processor

    source = SyntheticSource(n_days=days, alpha=alpha, seed=data_seed)
    fs = FeaturePipeline(source.symbols).build(source)

    # 4分割: Train(0-50%), val_select(50-68%), val_confirm(68-84%), Test(84-100%)
    purge = max(24, horizon)
    b_50 = int(fs.n_bars * 0.50)
    b_68 = int(fs.n_bars * 0.68)
    b_84 = int(fs.n_bars * 0.84)

    train_fs = fs.slice(0, b_50)
    val_select_fs = fs.slice(b_50 + purge, b_68)
    val_confirm_fs = fs.slice(b_68 + purge, b_84)
    test_fs = fs.slice(b_84 + purge, fs.n_bars)

    pp = make_legacy_processor(min_trade_delta=0.0)
    ekw = dict(
        post_processor=pp,
        min_trade_delta=0.0,
        lambda_turnover=0.0,
        reward_scale=100.0,
        fee_rate=0.0005,
        spread_rate=0.0002,
        impact_rate=0.0001,
    )

    action_dim = len(source.symbols)
    null_policy = NullPolicy(action_dim=action_dim)

    # チェックポイント保存リスト: (step, ppo_bytes, select_metrics, confirm_metrics)
    checkpoints: List[Tuple[int, bytes, Dict[str, Any], Dict[str, Any]]] = []

    class DiagnosticCallback(BaseCallback):
        def __init__(self):
            super().__init__(0)

        def _evaluate_save(self):
            sel_m = evaluate_policy_detailed(self.model, val_select_fs, **ekw)
            conf_m = evaluate_policy_detailed(self.model, val_confirm_fs, **ekw)
            buf = io.BytesIO()
            self.model.save(buf)
            checkpoints.append((self.num_timesteps, buf.getvalue(), sel_m, conf_m))

        def _on_training_start(self):
            self._evaluate_save()

        def _on_step(self) -> bool:
            if self.num_timesteps % eval_freq < self.training_env.num_envs:
                self._evaluate_save()
            return True

        def _on_training_end(self):
            self._evaluate_save()

    env = DummyVecEnv(make_env_fns(train_fs, n_envs=8, seed=model_seed, **ekw))
    probe = PortfolioTradingEnv(train_fs, **ekw)

    from mars_lite.models.portfolio_extractor import PortfolioExtractor

    policy_kwargs = {
        "features_extractor_class": PortfolioExtractor,
        "features_extractor_kwargs": {**probe.obs_layout, "size": "small"},
        "net_arch": dict(pi=[64, 64], vf=[64, 64]),
    }

    def lr_schedule(progress_remaining: float) -> float:
        return 3e-4 * progress_remaining

    agent = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_schedule,
        n_steps=256,
        batch_size=256,
        n_epochs=6,
        gamma=0.5,
        gae_lambda=0.9,
        ent_coef=0.002,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=model_seed,
        device="cpu",
        verbose=0,
    )

    cb = DiagnosticCallback()
    agent.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)

    # 1. val_select 上で最良の RL チェックポイントを選定
    best_step, best_bytes, best_sel_m, best_conf_m = max(
        checkpoints, key=lambda x: x[2]["sortino"]
    )

    # 2. 独立した val_confirm 上で複合 Null ゲート判定
    passes_null_gate = (
        best_conf_m["sortino"] >= min_improvement
        and best_conf_m["total_return"] >= 0.005  # net return >= +0.5%
        and best_conf_m["max_drawdown"] <= max_mdd
    )

    selected_type: str
    if passes_null_gate:
        selected_type = f"rl_checkpoint_{best_step}"
        buf_sel = io.BytesIO(best_bytes)
        selected_agent = PPO.load(buf_sel, device=agent.device)
    else:
        selected_type = "null_policy"
        selected_agent = null_policy

    # 3. 最終 Test 評価を1度だけ実施
    test_eval_selected = evaluate_policy_detailed(
        selected_agent, test_fs, record_std=True, **ekw
    )

    # 比較用: best_rl (ゲートなし) および final_rl の Test 評価
    buf_best = io.BytesIO(best_bytes)
    best_rl_agent = PPO.load(buf_best, device=agent.device)
    test_eval_best_rl = evaluate_policy_detailed(
        best_rl_agent, test_fs, record_std=True, **ekw
    )
    test_eval_final_rl = evaluate_policy_detailed(
        agent, test_fs, record_std=True, **ekw
    )

    baselines = run_all_baselines(
        test_fs, fee_rate=0.0005, spread_rate=0.0002, impact_rate=0.0001
    )
    bh_return = float(baselines["equal_weight_bh"].total_return)

    trajectory = [
        CheckpointMetric(
            step=s,
            val_select_return=sm["total_return"],
            val_select_sortino=sm["sortino"],
            val_select_turnover=sm["turnover_total"],
            val_confirm_return=cm["total_return"],
            val_confirm_sortino=cm["sortino"],
            val_confirm_turnover=cm["turnover_total"],
        )
        for s, _, sm, cm in checkpoints
    ]

    return {
        "alpha": alpha,
        "data_seed": data_seed,
        "model_seed": model_seed,
        "selected_type": selected_type,
        "best_checkpoint_step": best_step,
        "passes_null_gate": passes_null_gate,
        "val_confirm_metrics": best_conf_m,
        "test_selected": test_eval_selected,
        "test_best_rl": test_eval_best_rl,
        "test_final_rl": test_eval_final_rl,
        "bh_return": bh_return,
        "trajectory": [asdict(t) for t in trajectory],
    }


def analyze_matrix_5x5(
    runs: List[Dict[str, Any]], data_seeds: List[int], model_seeds: List[int]
) -> Dict[str, Any]:
    matrix_selected = {}
    matrix_ret_best = {}
    matrix_turnover_best = {}
    matrix_mean_abs = {}
    matrix_raw_gross = {}

    for d in data_seeds:
        row_sel = {}
        row_ret = {}
        row_to = {}
        row_abs = {}
        row_gross = {}
        for m in model_seeds:
            r = next(
                (r for r in runs if r["data_seed"] == d and r["model_seed"] == m), None
            )
            if r:
                sel = (
                    "Null"
                    if r["selected_type"] == "null_policy"
                    else f"RL_{r['best_checkpoint_step'] // 1000}k"
                )
                row_sel[str(m)] = sel
                row_ret[str(m)] = r["test_best_rl"]["total_return"]
                row_to[str(m)] = r["test_best_rl"]["turnover_total"]
                row_abs[str(m)] = r["test_best_rl"]["mean_abs_action"]
                row_gross[str(m)] = r["test_best_rl"]["raw_gross"]
        matrix_selected[str(d)] = row_sel
        matrix_ret_best[str(d)] = row_ret
        matrix_turnover_best[str(d)] = row_to
        matrix_mean_abs[str(d)] = row_abs
        matrix_raw_gross[str(d)] = row_gross

    # 分散分析・効果分解
    rets_2d = np.array(
        [
            [matrix_ret_best[str(d)].get(str(m), 0.0) for m in model_seeds]
            for d in data_seeds
        ]
    )
    grand_mean = float(np.mean(rets_2d))
    row_means = np.mean(rets_2d, axis=1) - grand_mean
    col_means = np.mean(rets_2d, axis=0) - grand_mean

    ss_data = float(len(model_seeds) * np.sum(row_means**2))
    ss_model = float(len(data_seeds) * np.sum(col_means**2))
    ss_total = float(np.sum((rets_2d - grand_mean) ** 2))
    ss_inter = max(0.0, ss_total - ss_data - ss_model)

    return {
        "matrix_selected_type": matrix_selected,
        "matrix_best_rl_return": matrix_ret_best,
        "matrix_best_rl_turnover": matrix_turnover_best,
        "matrix_best_rl_mean_abs_action": matrix_mean_abs,
        "matrix_best_rl_raw_gross": matrix_raw_gross,
        "anova_variance_share": {
            "data_seed_effect_pct": float(ss_data / max(1e-12, ss_total)),
            "model_seed_effect_pct": float(ss_model / max(1e-12, ss_total)),
            "interaction_effect_pct": float(ss_inter / max(1e-12, ss_total)),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="P0-1B: 無信号暴走診断とNull Policy選択")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--days", type=int, default=240)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--eval-freq", type=int, default=20_000)
    ap.add_argument("--min-improvement", type=float, default=0.5)
    ap.add_argument("--max-mdd", type=float, default=0.12)
    ap.add_argument("--data-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--mode", choices=["paired", "grid"], default="grid")
    ap.add_argument("--output", default="./output/p0_phase1b")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    import subprocess

    try:
        commit_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        commit_sha = "unknown"

    pairs: List[Tuple[int, int]] = []
    if args.mode == "paired":
        for d, m in zip(args.data_seeds, args.model_seeds):
            pairs.append((d, m))
    else:
        for d in args.data_seeds:
            for m in args.model_seeds:
                pairs.append((d, m))

    results = {"negative": [], "positive": []}
    # 推奨順序: 陰性対照を先に回し、完了したら陽性対照
    for label, alpha in [("negative", "none"), ("positive", "cross")]:
        print(
            f"\n{'=' * 70}\n{label} (alpha={alpha}) - {len(pairs)} runs\n{'=' * 70}",
            flush=True,
        )
        for d_seed, m_seed in pairs:
            print(
                f"  data_seed={d_seed}, model_seed={m_seed} ...",
                end=" ",
                flush=True,
            )
            r = run_single_trial(
                alpha=alpha,
                data_seed=d_seed,
                model_seed=m_seed,
                timesteps=args.timesteps,
                days=args.days,
                horizon=args.horizon,
                eval_freq=args.eval_freq,
                min_improvement=args.min_improvement,
                max_mdd=args.max_mdd,
            )
            results[label].append(r)
            sel = r["selected_type"]
            t_sel = r["test_selected"]
            print(
                f"selected={sel:<18} "
                f"ret={t_sel['total_return']:+.2%} "
                f"turnover={t_sel['turnover_total']:5.1f} "
                f"(raw_rl ret={r['test_best_rl']['total_return']:+.2%} "
                f"to={r['test_best_rl']['turnover_total']:5.1f})",
                flush=True,
            )

    neg_runs = results["negative"]
    pos_runs = results["positive"]

    neg_rets = [r["test_selected"]["total_return"] for r in neg_runs]
    neg_null_cnt = sum(1 for r in neg_runs if r["selected_type"] == "null_policy")
    neg_catastrophic = sum(
        1 for r in neg_runs if r["test_selected"]["total_return"] < -0.05
    )

    pos_rets = [r["test_selected"]["total_return"] for r in pos_runs]
    pos_rl_cnt = sum(1 for r in pos_runs if r["selected_type"] != "null_policy")
    pos_beats_bh = sum(
        1 for r in pos_runs if r["test_selected"]["total_return"] > r["bh_return"]
    )

    verdict = {
        "A1_negative_null_select_pass": bool(neg_null_cnt >= len(neg_runs) * 0.95),
        "A1_negative_no_loss_lt_neg5pct": bool(neg_catastrophic == 0),
        "A2_positive_rl_select_pass": bool(pos_rl_cnt >= len(pos_runs) * 0.88),
        "A2_positive_beats_bh_pass": bool(pos_beats_bh >= len(pos_runs) * 0.88),
        "A2_positive_pos_ret_pass": bool(
            sum(1 for r in pos_runs if r["test_selected"]["total_return"] > 0)
            >= len(pos_runs) * 0.88
        ),
    }
    verdict["A1_NEGATIVE_PASSED"] = (
        verdict["A1_negative_null_select_pass"]
        and verdict["A1_negative_no_loss_lt_neg5pct"]
    )
    verdict["A2_POSITIVE_PASSED"] = (
        verdict["A2_positive_rl_select_pass"]
        and verdict["A2_positive_beats_bh_pass"]
        and verdict["A2_positive_pos_ret_pass"]
    )

    matrix_analysis = {
        "negative": analyze_matrix_5x5(neg_runs, args.data_seeds, args.model_seeds),
        "positive": analyze_matrix_5x5(pos_runs, args.data_seeds, args.model_seeds),
    }

    report = {
        "verdict": verdict,
        "config": {
            "git_commit_sha": commit_sha,
            "timesteps": args.timesteps,
            "min_improvement": args.min_improvement,
            "max_mdd": args.max_mdd,
            "data_seeds": args.data_seeds,
            "model_seeds": args.model_seeds,
            "mode": args.mode,
        },
        "matrix_analysis": matrix_analysis,
        "results": results,
    }

    with open(out / "phase1b_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}\nPhase 1B 構成A (開発試験 5x5) サマリー\n{'=' * 70}")
    print(
        f"A1 陰性対照 (alpha=none): {'PASSED' if verdict['A1_NEGATIVE_PASSED'] else 'FAILED'}"
    )
    print(
        f"  Null Policy 選択率: {neg_null_cnt}/{len(neg_runs)} ({neg_null_cnt / max(1, len(neg_runs)):.1%})"
    )
    print(f"  損失 < -5% の試行数: {neg_catastrophic}/{len(neg_runs)}")
    print(
        f"  ANOVA 効果割合: data_seed={matrix_analysis['negative']['anova_variance_share']['data_seed_effect_pct']:.1%}, model_seed={matrix_analysis['negative']['anova_variance_share']['model_seed_effect_pct']:.1%}"
    )
    print(
        f"\nA2 陽性対照 (alpha=cross): {'PASSED' if verdict['A2_POSITIVE_PASSED'] else 'FAILED'}"
    )
    print(
        f"  RL 選択率: {pos_rl_cnt}/{len(pos_runs)} ({pos_rl_cnt / max(1, len(pos_runs)):.1%})"
    )
    print(f"  B&H 勝率:   {pos_beats_bh}/{len(pos_runs)}")
    print(f"\n詳細レポート -> {out / 'phase1b_report.json'}")
    return 0 if (verdict["A1_NEGATIVE_PASSED"] and verdict["A2_POSITIVE_PASSED"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
