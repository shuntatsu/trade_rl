from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mars_lite.eval.relative_evaluation import (
    _moving_block_mean_test,
    evaluate_relative_agent,
)
from mars_lite.features.signal_check import run_leak_self_test, run_signal_check
from mars_lite.learning.baselines import run_all_baselines
from mars_lite.learning.manifest import generate_and_save_manifest
from mars_lite.learning.residual_ensemble import ResidualActionEnsemble
from mars_lite.pipeline.dataset_builder import build_feature_set
from mars_lite.pipeline.gates import (
    evaluate_baseline_only_gate,
    evaluate_residual_gate2,
)
from mars_lite.pipeline.training_engine import (
    build_env_kwargs,
    build_post_processor,
    train_ppo,
)
from mars_lite.trading.residual_alpha import FrozenResidualAlpha
from mars_lite.trading.trend_family import TrendFamily, TrendFamilyConfig


class IdentityResidualAgent:
    def predict(self, observation, deterministic: bool = True):
        observation_array = np.asarray(observation)
        shape = (2,) if observation_array.ndim == 1 else (observation_array.shape[0], 2)
        return np.zeros(shape, dtype=np.float32), None


def _slim_baselines(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: value.to_dict() if hasattr(value, "to_dict") else dict(value)
        for name, value in results.items()
    }


def _is_fallback(policy: object) -> bool:
    selection = getattr(policy, "validation_selection", None)
    score = getattr(selection, "best_score", None)
    return bool(getattr(score, "baseline_fallback", False))


def run_baseline_residual(args, output_dir: str | Path) -> dict[str, Any]:
    """Run a leak-separated research workflow for baseline-anchored residual PPO."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    args.action_mode = "baseline-residual"
    args.min_trade_delta = 0.0
    args.lambda_turnover = 0.0

    fs = build_feature_set(args, output_dir=output)
    leak = run_leak_self_test(fs, horizon=args.horizon)
    if not leak["healthy"]:
        raise RuntimeError("leak self-test failed; refusing residual training")

    n = fs.n_bars
    train_end = int(n * 0.65)
    val_start = train_end + max(24, args.horizon)
    val_end = int(n * 0.82)
    test_start = val_end + max(24, args.horizon)
    if train_end < 200 or val_end - val_start < 100 or n - test_start < 100:
        raise ValueError("insufficient bars for train/validation/test residual workflow")
    train_fs = fs.slice(0, train_end)
    val_fs = fs.slice(val_start, val_end)
    test_fs = fs.slice(test_start, n)

    signal_gate = run_signal_check(
        train_fs,
        horizon=args.horizon,
        target="cs_demean",
    )
    alpha = FrozenResidualAlpha.fit(
        train_fs,
        horizon=args.horizon,
        target="cs_demean",
        model=getattr(args, "signal_model", "gbm"),
        gate_result=signal_gate.to_dict(),
    )
    alpha.save(output / "residual_alpha.json")

    trend_family = TrendFamily(
        TrendFamilyConfig(base_timeframe=getattr(args, "base_timeframe", "1h"))
    )
    post_processor = build_post_processor(args, horizon=args.horizon)
    env_kwargs = build_env_kwargs(args, post_processor, horizon=args.horizon)
    ensemble_size = max(1, int(getattr(args, "ensemble", 3)))
    policies = []
    for member in range(ensemble_size):
        policies.append(
            train_ppo(
                fs=train_fs,
                val_fs=val_fs,
                timesteps=args.timesteps,
                seed=args.seed + member,
                gamma=args.gamma,
                ent_coef=getattr(args, "ent_coef", 0.002),
                learning_rate=getattr(args, "learning_rate", 3e-4),
                verbose=args.verbose,
                action_mode="baseline-residual",
                run_tier=getattr(args, "run_tier", "research"),
                n_seeds=ensemble_size,
                trend_family=trend_family,
                alpha_provider=alpha,
                alpha_enabled=alpha.enabled,
                bc_warmstart=False,
                **env_kwargs,
            )
        )
    agent = policies[0] if len(policies) == 1 else ResidualActionEnsemble(policies)
    if len(policies) == 1:
        policies[0].save(str(output / "portfolio_residual_model"))
    else:
        agent.save(output / "portfolio_residual_ensemble")

    evaluation_kwargs = {
        **env_kwargs,
        "trend_family": trend_family,
        "alpha_provider": alpha,
        "alpha_enabled": alpha.enabled,
    }
    relative = evaluate_relative_agent(
        agent,
        test_fs,
        env_kwargs=evaluation_kwargs,
        bootstrap_seed=args.seed,
    )
    baselines = run_all_baselines(
        test_fs,
        noisy_oracle_ic=(
            args.noisy_oracle_ic if getattr(args, "noisy_oracle_ic", 0.0) > 0 else None
        ),
        fee_rate=env_kwargs["fee_rate"],
        spread_rate=env_kwargs["spread_rate"],
        impact_rate=env_kwargs["impact_rate"],
    )
    baseline_payload = _slim_baselines(baselines)

    all_fallback = all(_is_fallback(policy) for policy in policies)
    if all_fallback:
        identity = IdentityResidualAgent()
        cost2x_kwargs = dict(evaluation_kwargs)
        cost2x_kwargs["cost_multiplier"] = 2.0
        cost2x = evaluate_relative_agent(
            identity,
            test_fs,
            env_kwargs=cost2x_kwargs,
            bootstrap_seed=args.seed,
        )
        shadow_returns = np.asarray(
            BaselineResidualReturnView(identity, test_fs, evaluation_kwargs).shadow_returns,
            dtype=np.float64,
        )
        positive = _moving_block_mean_test(shadow_returns, seed=args.seed)
        trend_dev_gate = {
            "passed": relative["shadow"]["total_return"] > 0.0,
            "source": "research_test_only",
        }
        gate = evaluate_baseline_only_gate(
            trend_development_gate=trend_dev_gate,
            holdout=relative["shadow"],
            cost2x_holdout=cost2x["shadow"],
            positive_return_p_value=float(positive["p_value"]),
            max_drawdown_limit=float(getattr(args, "baseline_max_drawdown", 0.30)),
        )
    else:
        gate = evaluate_residual_gate2(
            hybrid=relative["hybrid"],
            shadow=relative["shadow"],
            flat={"total_return": 0.0, "max_drawdown": 0.0},
            paired_p_value=float(relative["paired"]["p_value"]),
            diagnostic_results=baseline_payload,
        )

    report = {
        "mode": "baseline_only" if all_fallback else "ppo_residual_ensemble",
        "action_schema": "baseline_residual_v1",
        "split": {
            "train_bars": train_fs.n_bars,
            "validation_bars": val_fs.n_bars,
            "test_bars": test_fs.n_bars,
        },
        "leak_self_test": leak,
        "signal_gate": signal_gate.to_dict(),
        "alpha_enabled": alpha.enabled,
        "relative": relative,
        "baselines": baseline_payload,
        "gate": gate,
        "seed_fallbacks": [_is_fallback(policy) for policy in policies],
    }
    (output / "residual_train_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    generate_and_save_manifest(
        output_filepath=str(output / "residual_model_manifest.json"),
        fs=train_fs,
        hyperparams={
            "timesteps": args.timesteps,
            "gamma": args.gamma,
            "seed": args.seed,
            "ensemble": ensemble_size,
            "decision_every": args.decision_every,
            "action_mode": "baseline-residual",
            "run_tier": getattr(args, "run_tier", "research"),
        },
        seed=args.seed,
        additional_metadata={
            "action_schema": "baseline_residual_v1",
            "policy_mode": report["mode"],
            "alpha_dataset_identity": alpha.dataset_identity,
        },
    )
    return report


class BaselineResidualReturnView:
    """Small helper to expose base-bar returns from an identity rollout."""

    def __init__(self, agent, fs, env_kwargs: dict[str, Any]):
        from mars_lite.env.baseline_residual_env import BaselineResidualTradingEnv

        env = BaselineResidualTradingEnv(
            fs,
            episode_bars=fs.n_bars - 2,
            **env_kwargs,
        )
        obs, _ = env.reset(options={"start_idx": 0})
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        self.hybrid_returns = tuple(env.hybrid.returns_history)
        self.shadow_returns = tuple(env.shadow.returns_history)
