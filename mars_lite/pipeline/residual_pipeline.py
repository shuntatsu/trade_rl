from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mars_lite.eval.context_window import with_history_context
from mars_lite.eval.gate1_diagnostics import walk_forward_ic
from mars_lite.eval.relative_evaluation import (
    _moving_block_mean_test,
    evaluate_relative_agent,
)
from mars_lite.features.signal_check import run_leak_self_test
from mars_lite.learning.baselines import run_all_baselines
from mars_lite.learning.manifest import generate_and_save_manifest
from mars_lite.pipeline.dataset_builder import build_feature_set
from mars_lite.pipeline.gates import (
    evaluate_baseline_only_gate,
    evaluate_residual_alpha_gate,
    evaluate_residual_gate2,
)
from mars_lite.pipeline.residual_candidates import (
    FixedResidualAgent,
    IdentityResidualAgent,
    ResidualCandidateSelection,
    _evaluation_kwargs,
    _is_fallback,
    _slim_baselines,
    _train_residual_ensemble,
    select_residual_configuration,
    train_select_residual_candidates,
)
from mars_lite.pipeline.training_engine import build_env_kwargs, build_post_processor
from mars_lite.trading.residual_alpha import FrozenResidualAlpha
from mars_lite.trading.trend_family import TrendFamily, TrendFamilyConfig

__all__ = [
    "FixedResidualAgent",
    "IdentityResidualAgent",
    "ResidualCandidateSelection",
    "select_residual_configuration",
    "run_baseline_residual",
]


def _history_bars(config: TrendFamilyConfig) -> int:
    return (
        max(config.fast_lookback, config.base_lookback, config.slow_lookback)
        + config.rebalance_every
    )


def run_baseline_residual(args, output_dir: str | Path) -> dict[str, Any]:
    """Run the leak-separated single-split residual research workflow."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    args.action_mode = "baseline-residual"
    args.min_trade_delta = 0.0
    args.lambda_turnover = 0.0

    fs = build_feature_set(args, output_dir=output)
    n_bars = fs.n_bars
    purge = max(24, int(args.horizon))
    policy_train_end = int(n_bars * 0.58)
    checkpoint_validation_start = policy_train_end + purge
    checkpoint_validation_end = int(n_bars * 0.70)
    configuration_selection_start = checkpoint_validation_end + purge
    configuration_selection_end = int(n_bars * 0.82)
    test_start = configuration_selection_end + purge
    segment_sizes = {
        "policy_train": policy_train_end,
        "checkpoint_validation": (
            checkpoint_validation_end - checkpoint_validation_start
        ),
        "configuration_selection": (
            configuration_selection_end - configuration_selection_start
        ),
        "test": n_bars - test_start,
    }
    minimums = {
        "policy_train": 200,
        "checkpoint_validation": 100,
        "configuration_selection": 100,
        "test": 100,
    }
    invalid = [name for name, size in segment_sizes.items() if size < minimums[name]]
    if invalid:
        raise ValueError(
            "insufficient bars for separated residual workflow: " + ", ".join(invalid)
        )

    train_fs = fs.slice(0, policy_train_end)
    leak = run_leak_self_test(train_fs, horizon=args.horizon)
    if not leak["healthy"]:
        raise RuntimeError("leak self-test failed; refusing residual training")

    trend_config = TrendFamilyConfig(
        base_timeframe=getattr(args, "base_timeframe", "1h")
    )
    trend_family = TrendFamily(trend_config)
    history_bars = _history_bars(trend_config)
    checkpoint_window = with_history_context(
        fs,
        start=checkpoint_validation_start,
        end=checkpoint_validation_end,
        history_bars=history_bars,
    )
    selection_window = with_history_context(
        fs,
        start=configuration_selection_start,
        end=configuration_selection_end,
        history_bars=history_bars,
    )
    test_window = with_history_context(
        fs,
        start=test_start,
        end=n_bars,
        history_bars=history_bars,
    )

    signal_model = str(getattr(args, "signal_model", "gbm"))
    model_gate_report = walk_forward_ic(
        train_fs,
        horizon=args.horizon,
        target="cs_demean",
        model=signal_model,
    )
    signal_gate = evaluate_residual_alpha_gate(model_gate_report)
    alpha = FrozenResidualAlpha.fit(
        train_fs,
        horizon=args.horizon,
        target="cs_demean",
        model=signal_model,
        gate_result=signal_gate,
    )
    alpha.save(output / "residual_alpha.json")

    post_processor = build_post_processor(args, horizon=args.horizon)
    env_kwargs = build_env_kwargs(args, post_processor, horizon=args.horizon)
    candidates = train_select_residual_candidates(
        args=args,
        train_fs=train_fs,
        checkpoint_val_fs=checkpoint_window.feature_set,
        selection_fs=selection_window.feature_set,
        trend_family=trend_family,
        alpha=alpha,
        env_kwargs=env_kwargs,
        output=output,
    )

    selected = candidates.selected_configuration
    selected_env_kwargs = _evaluation_kwargs(
        env_kwargs,
        trend_family,
        alpha,
        alpha_enabled=candidates.selected_alpha_enabled,
    )
    relative = evaluate_relative_agent(
        candidates.selected_agent,
        test_window.feature_set,
        env_kwargs=selected_env_kwargs,
        bootstrap_seed=args.seed,
    )
    cost2x = evaluate_relative_agent(
        candidates.selected_agent,
        test_window.feature_set,
        env_kwargs={**selected_env_kwargs, "cost_multiplier": 2.0},
        bootstrap_seed=args.seed,
    )

    baselines = run_all_baselines(
        test_window.feature_set,
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
        identity = IdentityResidualAgent()
        shadow_returns = np.asarray(
            BaselineResidualReturnView(
                identity,
                test_window.feature_set,
                selected_env_kwargs,
            ).shadow_returns,
            dtype=np.float64,
        )
        positive = _moving_block_mean_test(shadow_returns, seed=args.seed)
        development_a = candidates.development_results["A"]
        trend_dev_gate = {
            "passed": (
                float(development_a["shadow"]["total_return"]) > 0.0
                and float(development_a["shadow"]["max_drawdown"])
                <= float(getattr(args, "baseline_max_drawdown", 0.30))
            ),
            "source": "configuration_selection",
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
            cost2x_hybrid=cost2x["hybrid"],
            cost2x_shadow=cost2x["shadow"],
            paired_p_value=float(relative["paired"]["p_value"]),
            diagnostic_results=baseline_payload,
        )

    report: dict[str, Any] = {
        "mode": str(candidates.selection["policy_mode"]),
        "action_schema": "baseline_residual_v1",
        "selected_configuration": selected,
        "selected_model_path": (
            str(candidates.selected_model_path)
            if candidates.selected_model_path is not None
            else None
        ),
        "split": {
            "policy_train_bars": train_fs.n_bars,
            "checkpoint_validation_bars": checkpoint_window.scored_bars,
            "checkpoint_validation_context_bars": checkpoint_window.start_idx,
            "configuration_selection_bars": selection_window.scored_bars,
            "configuration_selection_context_bars": selection_window.start_idx,
            "validation_bars": selection_window.scored_bars,
            "validation_context_bars": selection_window.start_idx,
            "test_bars": test_window.scored_bars,
            "test_context_bars": test_window.start_idx,
            "purge_bars": purge,
        },
        "leak_self_test": leak,
        "signal_gate": signal_gate,
        "alpha_enabled": candidates.selected_alpha_enabled,
        "alpha_artifact_gate_passed": alpha.enabled,
        "development_matrix": candidates.development_results,
        "development_matrix_cost2x": candidates.development_cost2x_results,
        "selection": candidates.selection,
        "relative": relative,
        "cost2x": cost2x,
        "baselines": baseline_payload,
        "gate": gate,
        "selected_seed_fallbacks": [
            _is_fallback(policy) for policy in candidates.selected_policies
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
            "residual_alpha_enabled": candidates.selected_alpha_enabled,
            "alpha_dataset_identity": alpha.dataset_identity,
            "selection_frozen_before_test": True,
            "checkpoint_selection_separated": True,
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
