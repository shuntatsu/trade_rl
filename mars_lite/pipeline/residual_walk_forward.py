from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mars_lite.eval.context_window import with_history_context
from mars_lite.eval.gate1_diagnostics import walk_forward_ic
from mars_lite.eval.relative_evaluation import evaluate_relative_agent
from mars_lite.eval.residual_walk_forward import (
    ResidualFoldSpec,
    build_residual_fold_specs,
    save_residual_walk_forward_report,
    summarize_residual_folds,
)
from mars_lite.features.signal_check import run_leak_self_test
from mars_lite.learning.baselines import run_all_baselines
from mars_lite.pipeline.dataset_builder import build_feature_set
from mars_lite.pipeline.gates import evaluate_residual_alpha_gate
from mars_lite.pipeline.residual_candidates import train_select_residual_candidates
from mars_lite.pipeline.residual_pipeline import (
    _evaluation_kwargs,
    _is_fallback,
    _slim_baselines,
)
from mars_lite.pipeline.training_engine import build_env_kwargs, build_post_processor
from mars_lite.trading.execution import FEE_KWARG_KEYS
from mars_lite.trading.residual_alpha import FrozenResidualAlpha
from mars_lite.trading.trend_family import TrendFamily, TrendFamilyConfig


def _history_bars(config: TrendFamilyConfig) -> int:
    return (
        max(config.fast_lookback, config.base_lookback, config.slow_lookback)
        + config.rebalance_every
    )


def _baseline_kwargs(env_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: env_kwargs[key] for key in FEE_KWARG_KEYS if key in env_kwargs}


def run_residual_fold(
    *,
    fs,
    spec: ResidualFoldSpec,
    args,
    output_dir: str | Path,
) -> dict[str, object]:
    """Train/select on development data and score one frozen outer OOS fold."""

    fold_output = Path(output_dir) / "residual_wf" / f"fold_{spec.fold}"
    fold_output.mkdir(parents=True, exist_ok=True)
    fold_args = copy.copy(args)
    fold_args.seed = int(args.seed) + spec.fold * max(1, int(args.ensemble))

    train_fs = fs.slice(spec.policy_train_start, spec.policy_train_end)
    leak = run_leak_self_test(train_fs, horizon=args.horizon)
    if not leak["healthy"]:
        raise RuntimeError(f"fold {spec.fold} leak self-test failed")

    trend_config = TrendFamilyConfig(
        base_timeframe=getattr(args, "base_timeframe", "1h")
    )
    trend_family = TrendFamily(trend_config)
    history_bars = _history_bars(trend_config)
    checkpoint_window = with_history_context(
        fs,
        start=spec.checkpoint_validation_start,
        end=spec.checkpoint_validation_end,
        history_bars=history_bars,
    )
    selection_window = with_history_context(
        fs,
        start=spec.configuration_selection_start,
        end=spec.configuration_selection_end,
        history_bars=history_bars,
    )
    test_window = with_history_context(
        fs,
        start=spec.outer_test_start,
        end=spec.outer_test_end,
        history_bars=history_bars,
    )
    checkpoint_fs = checkpoint_window.feature_set
    selection_fs = selection_window.feature_set
    test_fs = test_window.feature_set

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
    alpha_path = fold_output / "residual_alpha.json"
    alpha.save(alpha_path)

    post_processor = build_post_processor(fold_args, horizon=args.horizon)
    env_kwargs = build_env_kwargs(fold_args, post_processor, horizon=args.horizon)
    candidates = train_select_residual_candidates(
        args=fold_args,
        train_fs=train_fs,
        val_fs=selection_fs,
        trend_family=trend_family,
        alpha=alpha,
        env_kwargs=env_kwargs,
        output=fold_output,
    )

    selected_env_kwargs = _evaluation_kwargs(
        env_kwargs,
        trend_family,
        alpha,
        alpha_enabled=candidates.selected_alpha_enabled,
    )
    relative_1x = evaluate_relative_agent(
        candidates.selected_agent,
        test_fs,
        env_kwargs=selected_env_kwargs,
        bootstrap_seed=fold_args.seed,
    )
    relative_2x = evaluate_relative_agent(
        candidates.selected_agent,
        test_fs,
        env_kwargs={**selected_env_kwargs, "cost_multiplier": 2.0},
        bootstrap_seed=fold_args.seed,
    )

    noisy_oracle = float(getattr(args, "noisy_oracle_ic", 0.0))
    baseline_kwargs = _baseline_kwargs(env_kwargs)
    baselines_1x = _slim_baselines(
        run_all_baselines(
            test_fs,
            noisy_oracle_ic=noisy_oracle if noisy_oracle > 0.0 else None,
            cost_multiplier=1.0,
            start_idx=test_window.start_idx,
            **baseline_kwargs,
        )
    )
    baselines_2x = _slim_baselines(
        run_all_baselines(
            test_fs,
            noisy_oracle_ic=noisy_oracle if noisy_oracle > 0.0 else None,
            cost_multiplier=2.0,
            start_idx=test_window.start_idx,
            **baseline_kwargs,
        )
    )

    selected_model_identity = (
        str(candidates.selected_model_path)
        if candidates.selected_model_path is not None
        else "identity:base_trend_v2"
    )
    report: dict[str, object] = {
        "fold": spec.fold,
        "split": {
            **asdict(spec),
            "inner_train_bars": train_fs.n_bars,
            "checkpoint_validation_scored_bars": checkpoint_window.scored_bars,
            "checkpoint_validation_context_bars": checkpoint_window.start_idx,
            "inner_validation_scored_bars": selection_window.scored_bars,
            "inner_validation_context_bars": selection_window.start_idx,
            "outer_test_scored_bars": test_window.scored_bars,
            "outer_test_context_bars": test_window.start_idx,
            "history_bars_required": history_bars,
        },
        "leak_self_test": leak,
        "signal_gate": signal_gate,
        "alpha_dataset_identity": alpha.dataset_identity,
        "alpha_artifact_path": str(alpha_path),
        "alpha_artifact_gate_passed": bool(alpha.enabled),
        "development_matrix": candidates.development_results,
        "development_matrix_cost2x": candidates.development_cost2x_results,
        "selection": candidates.selection,
        "selected_configuration": candidates.selected_configuration,
        "selected_policy_mode": str(candidates.selection["policy_mode"]),
        "selected_model_identity": selected_model_identity,
        "alpha_enabled": candidates.selected_alpha_enabled,
        "selected_seed_fallbacks": [
            _is_fallback(policy) for policy in candidates.selected_policies
        ],
        "outer_oos": {
            "relative_1x": relative_1x,
            "relative_2x": relative_2x,
            "baselines_1x": baselines_1x,
            "baselines_2x": baselines_2x,
            "same_selected_model_for_cost_scenarios": True,
        },
    }
    save_residual_walk_forward_report(fold_output / "fold_report.json", report)
    return report


def run_residual_walk_forward(
    args,
    output_dir: str | Path,
) -> dict[str, object]:
    """Run research-only nested expanding Walk-Forward for residual RL."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    args.action_mode = "baseline-residual"
    args.min_trade_delta = 0.0
    args.lambda_turnover = 0.0
    requested_seeds = max(1, int(getattr(args, "n_seeds", 1)))
    requested_ensemble = max(1, int(getattr(args, "ensemble", 1)))
    args.ensemble = max(requested_seeds, requested_ensemble)

    fs = build_feature_set(args, output_dir=output)
    requested_folds = int(getattr(args, "folds", 3))
    purge_bars = int(getattr(args, "purge_bars", 24))
    specs, skipped = build_residual_fold_specs(
        n_bars=fs.n_bars,
        n_folds=requested_folds,
        purge_bars=purge_bars,
        horizon=int(args.horizon),
    )

    folds: list[dict[str, object]] = []
    for spec in specs:
        folds.append(
            run_residual_fold(
                fs=fs,
                spec=spec,
                args=args,
                output_dir=output,
            )
        )

    report: dict[str, object] = {
        "mode": "baseline_residual_walk_forward_v1",
        "action_schema": "baseline_residual_v1",
        "config": {
            "requested_folds": requested_folds,
            "completed_folds": len(folds),
            "purge_bars": max(purge_bars, int(args.horizon), 24),
            "horizon": int(args.horizon),
            "decision_every": int(args.decision_every),
            "ensemble_size": int(args.ensemble),
            "requested_n_seeds": requested_seeds,
            "member_seeds_per_fold": int(args.ensemble),
            "run_tier": str(getattr(args, "run_tier", "research")),
            "n_bars_total": int(fs.n_bars),
            "outer_train_start_fraction": 0.4,
            "policy_train_fraction": 0.70,
            "checkpoint_validation_fraction": 0.15,
            "configuration_selection_fraction": 0.15,
        },
        "summary": summarize_residual_folds(
            folds,
            requested_folds=requested_folds,
            skipped_folds=skipped,
        ),
        "folds": folds,
        "skipped_folds": skipped,
        "release_eligible": False,
        "release_blocker": (
            "sealed residual release workflow remains incomplete; research output only"
        ),
    }
    save_residual_walk_forward_report(output / "residual_walk_forward.json", report)
    return report
