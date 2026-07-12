from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mars_lite.eval.context_window import with_history_context
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


class FixedResidualAgent:
    def __init__(self, action: tuple[float, float]):
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError("fixed residual action must be finite with shape (2,)")
        self.action = value

    def predict(self, observation, deterministic: bool = True):
        observation_array = np.asarray(observation)
        if observation_array.ndim == 1:
            return self.action.copy(), None
        return np.repeat(self.action[None, :], observation_array.shape[0], axis=0), None


class IdentityResidualAgent(FixedResidualAgent):
    def __init__(self):
        super().__init__((0.0, 0.0))


def _slim_baselines(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: value.to_dict() if hasattr(value, "to_dict") else dict(value)
        for name, value in results.items()
    }


def _is_fallback(policy: object) -> bool:
    selection = getattr(policy, "validation_selection", None)
    score = getattr(selection, "best_score", None)
    return bool(getattr(score, "baseline_fallback", False))


def _eligible_relative_result(result: dict[str, Any], drawdown_slack: float) -> bool:
    excess = float(result["paired"]["excess_log_return"])
    hybrid_dd = float(result["hybrid"]["max_drawdown"])
    shadow_dd = float(result["shadow"]["max_drawdown"])
    return excess > 0.0 and hybrid_dd <= shadow_dd + drawdown_slack


def select_residual_configuration(
    development_results: dict[str, dict[str, Any]],
    *,
    drawdown_slack: float = 0.05,
) -> dict[str, Any]:
    """Apply the preregistered A/B/C/D selection rule on development data only.

    A is pure base trend. B is PPO trend mixing. C is a fixed +15% alpha diagnostic.
    D is PPO trend mixing plus alpha. C is never itself release-selected; it establishes
    the hurdle that D must beat before the combined RL design is adopted.
    """

    if "A" not in development_results:
        raise ValueError("development matrix requires configuration A")
    scores = {
        name: float(result["paired"]["excess_log_return"])
        for name, result in development_results.items()
    }
    eligible = {
        name: _eligible_relative_result(result, drawdown_slack)
        for name, result in development_results.items()
    }
    selected = "A"
    reasons = ["A is the identity baseline"]

    if eligible.get("B", False):
        selected = "B"
        reasons.append("B adds positive development excess within drawdown slack")

    if eligible.get("D", False):
        hurdle = max(scores.get("B", 0.0), scores.get("C", 0.0))
        if scores["D"] > hurdle:
            selected = "D"
            reasons.append("D strictly beats both B and fixed-alpha diagnostic C")
        else:
            reasons.append("D did not beat the stronger of B and C")

    return {
        "selected": selected,
        "policy_mode": (
            "baseline_only" if selected == "A" else "ppo_residual_ensemble"
        ),
        "scores": scores,
        "eligible": eligible,
        "reasons": reasons,
        "drawdown_slack": drawdown_slack,
    }


def _train_residual_ensemble(
    *,
    label: str,
    args,
    train_fs,
    val_fs,
    trend_family: TrendFamily,
    alpha: FrozenResidualAlpha,
    alpha_enabled: bool,
    env_kwargs: dict[str, Any],
    output: Path,
) -> tuple[object, list[object], Path]:
    ensemble_size = max(1, int(getattr(args, "ensemble", 3)))
    policies: list[object] = []
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
                alpha_enabled=alpha_enabled,
                bc_warmstart=False,
                **env_kwargs,
            )
        )
    agent: object = (
        policies[0] if len(policies) == 1 else ResidualActionEnsemble(policies)
    )
    if len(policies) == 1:
        model_path = output / f"{label}_model.zip"
        policies[0].save(str(model_path))
    else:
        model_path = output / f"{label}_ensemble"
        agent.save(model_path)
    return agent, policies, model_path


def _evaluation_kwargs(
    env_kwargs: dict[str, Any],
    trend_family: TrendFamily,
    alpha: FrozenResidualAlpha,
    *,
    alpha_enabled: bool,
) -> dict[str, Any]:
    return {
        **env_kwargs,
        "trend_family": trend_family,
        "alpha_provider": alpha,
        "alpha_enabled": alpha_enabled,
    }


def run_baseline_residual(args, output_dir: str | Path) -> dict[str, Any]:
    """Run a leak-separated A/B/C/D baseline-anchored residual workflow."""

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
        raise ValueError(
            "insufficient bars for train/validation/test residual workflow"
        )
    trend_config = TrendFamilyConfig(
        base_timeframe=getattr(args, "base_timeframe", "1h")
    )
    trend_family = TrendFamily(trend_config)
    history_bars = (
        max(
            trend_config.fast_lookback,
            trend_config.base_lookback,
            trend_config.slow_lookback,
        )
        + trend_config.rebalance_every
    )
    train_fs = fs.slice(0, train_end)
    val_window = with_history_context(
        fs, start=val_start, end=val_end, history_bars=history_bars
    )
    test_window = with_history_context(
        fs, start=test_start, end=n, history_bars=history_bars
    )
    val_fs = val_window.feature_set
    test_fs = test_window.feature_set

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

    post_processor = build_post_processor(args, horizon=args.horizon)
    env_kwargs = build_env_kwargs(args, post_processor, horizon=args.horizon)

    identity = IdentityResidualAgent()
    fixed_alpha = FixedResidualAgent((0.0, 0.5))
    development_results: dict[str, dict[str, Any]] = {}
    development_results["A"] = evaluate_relative_agent(
        identity,
        val_fs,
        env_kwargs=_evaluation_kwargs(
            env_kwargs, trend_family, alpha, alpha_enabled=False
        ),
        bootstrap_seed=args.seed,
    )

    b_agent, b_policies, b_model_path = _train_residual_ensemble(
        label="B_trend_mix",
        args=args,
        train_fs=train_fs,
        val_fs=val_fs,
        trend_family=trend_family,
        alpha=alpha,
        alpha_enabled=False,
        env_kwargs=env_kwargs,
        output=output,
    )
    development_results["B"] = evaluate_relative_agent(
        b_agent,
        val_fs,
        env_kwargs=_evaluation_kwargs(
            env_kwargs, trend_family, alpha, alpha_enabled=False
        ),
        bootstrap_seed=args.seed,
    )

    if alpha.enabled:
        development_results["C"] = evaluate_relative_agent(
            fixed_alpha,
            val_fs,
            env_kwargs=_evaluation_kwargs(
                env_kwargs, trend_family, alpha, alpha_enabled=True
            ),
            bootstrap_seed=args.seed,
        )
        d_agent, d_policies, d_model_path = _train_residual_ensemble(
            label="D_combined",
            args=args,
            train_fs=train_fs,
            val_fs=val_fs,
            trend_family=trend_family,
            alpha=alpha,
            alpha_enabled=True,
            env_kwargs=env_kwargs,
            output=output,
        )
        development_results["D"] = evaluate_relative_agent(
            d_agent,
            val_fs,
            env_kwargs=_evaluation_kwargs(
                env_kwargs, trend_family, alpha, alpha_enabled=True
            ),
            bootstrap_seed=args.seed,
        )
    else:
        d_agent = None
        d_policies = []
        d_model_path = None

    selection = select_residual_configuration(development_results)
    selected = str(selection["selected"])
    if selected == "D":
        assert d_agent is not None and d_model_path is not None
        selected_agent = d_agent
        selected_policies = d_policies
        selected_model_path: Path | None = d_model_path
        selected_alpha_enabled = True
    elif selected == "B":
        selected_agent = b_agent
        selected_policies = b_policies
        selected_model_path = b_model_path
        selected_alpha_enabled = False
    else:
        selected_agent = identity
        selected_policies = []
        selected_model_path = None
        selected_alpha_enabled = False

    selected_env_kwargs = _evaluation_kwargs(
        env_kwargs,
        trend_family,
        alpha,
        alpha_enabled=selected_alpha_enabled,
    )
    relative = evaluate_relative_agent(
        selected_agent,
        test_fs,
        env_kwargs=selected_env_kwargs,
        bootstrap_seed=args.seed,
    )
    cost2x_env_kwargs = {**selected_env_kwargs, "cost_multiplier": 2.0}
    cost2x = evaluate_relative_agent(
        selected_agent,
        test_fs,
        env_kwargs=cost2x_env_kwargs,
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
        start_idx=test_window.start_idx,
    )
    baseline_payload = _slim_baselines(baselines)

    if selected == "A":
        shadow_returns = np.asarray(
            BaselineResidualReturnView(
                identity, test_fs, selected_env_kwargs
            ).shadow_returns,
            dtype=np.float64,
        )
        positive = _moving_block_mean_test(shadow_returns, seed=args.seed)
        development_a = development_results["A"]
        trend_dev_gate = {
            "passed": (
                float(development_a["shadow"]["total_return"]) > 0.0
                and float(development_a["shadow"]["max_drawdown"])
                <= float(getattr(args, "baseline_max_drawdown", 0.30))
            ),
            "source": "development_validation",
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
        "mode": str(selection["policy_mode"]),
        "action_schema": "baseline_residual_v1",
        "selected_configuration": selected,
        "selected_model_path": (
            str(selected_model_path) if selected_model_path is not None else None
        ),
        "split": {
            "train_bars": train_fs.n_bars,
            "validation_bars": val_window.scored_bars,
            "validation_context_bars": val_window.start_idx,
            "test_bars": test_window.scored_bars,
            "test_context_bars": test_window.start_idx,
        },
        "leak_self_test": leak,
        "signal_gate": signal_gate.to_dict(),
        "alpha_enabled": alpha.enabled,
        "development_matrix": development_results,
        "selection": selection,
        "relative": relative,
        "cost2x": cost2x,
        "baselines": baseline_payload,
        "gate": gate,
        "selected_seed_fallbacks": [
            _is_fallback(policy) for policy in selected_policies
        ],
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
            "ensemble": max(1, int(getattr(args, "ensemble", 3))),
            "decision_every": args.decision_every,
            "action_mode": "baseline-residual",
            "run_tier": getattr(args, "run_tier", "research"),
            "selected_configuration": selected,
        },
        seed=args.seed,
        additional_metadata={
            "action_schema": "baseline_residual_v1",
            "policy_mode": report["mode"],
            "alpha_dataset_identity": alpha.dataset_identity,
            "selection_frozen_before_test": True,
        },
    )
    return report


class BaselineResidualReturnView:
    """Small helper to expose base-bar returns from a residual rollout."""

    def __init__(self, agent, fs, env_kwargs: dict[str, Any]):
        from mars_lite.env.baseline_residual_env import BaselineResidualTradingEnv

        start_idx = int(getattr(fs, "_evaluation_start_idx", 0))
        env = BaselineResidualTradingEnv(
            fs,
            episode_bars=max(1, fs.n_bars - 2 - start_idx),
            **env_kwargs,
        )
        obs, _ = env.reset(options={"start_idx": start_idx})
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        self.hybrid_returns = tuple(env.hybrid.returns_history)
        self.shadow_returns = tuple(env.shadow.returns_history)
