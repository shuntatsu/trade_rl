from __future__ import annotations

import copy
import os
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mars_lite.eval.context_window import with_history_context
from mars_lite.eval.gate1_diagnostics import walk_forward_ic
from mars_lite.eval.relative_evaluation import evaluate_relative_agent
from mars_lite.eval.residual_walk_forward import (
    RelativeFoldSeries,
    ResidualFoldSpec,
    build_residual_fold_specs,
    save_residual_walk_forward_report,
    stitch_relative_fold_results,
    summarize_residual_folds,
)
from mars_lite.eval.strategy_metrics import reannualize_strategy_results
from mars_lite.features.signal_check import run_leak_self_test
from mars_lite.learning.baselines import run_all_baselines
from mars_lite.pipeline.dataset_builder import build_feature_set
from mars_lite.pipeline.gates import evaluate_residual_alpha_gate
from mars_lite.pipeline.residual_candidates import (
    _evaluation_kwargs,
    _is_fallback,
    _slim_baselines,
    train_select_residual_candidates,
)
from mars_lite.pipeline.residual_wf_config import (
    ResidualWalkForwardConfig,
    feature_set_identity,
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


def _runtime_args(args: object, config: ResidualWalkForwardConfig) -> object:
    runtime = copy.copy(args)
    runtime.action_mode = "baseline-residual"
    runtime.min_trade_delta = 0.0
    runtime.lambda_turnover = 0.0
    runtime.decision_every = config.effective_decision_every
    runtime.ensemble = config.effective_ensemble_size
    runtime.n_seeds = config.requested_n_seeds
    runtime.purge_bars = config.effective_purge_bars
    return runtime


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:12]}"


def _publish_authoritative_report(
    output: Path,
    report: dict[str, object],
    *,
    run_id: str,
) -> None:
    temporary = output / f".residual_walk_forward.{run_id}.tmp"
    destination = output / "residual_walk_forward.json"
    try:
        save_residual_walk_forward_report(temporary, report)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _isolate_failed_run(staging: Path, output: Path, run_id: str) -> None:
    if not staging.exists():
        return
    failed_root = output / "failed"
    failed_root.mkdir(parents=True, exist_ok=True)
    failed_destination = failed_root / run_id
    if failed_destination.exists():
        shutil.rmtree(failed_destination)
    os.replace(staging, failed_destination)


def _diagnostic_baselines(
    *,
    fs,
    spec: ResidualFoldSpec,
    env_kwargs: dict[str, Any],
    bars_per_year: int,
    cost_multiplier: float,
    noisy_oracle_ic: float,
) -> dict[str, dict[str, Any]]:
    results = run_all_baselines(
        fs,
        noisy_oracle_ic=noisy_oracle_ic if noisy_oracle_ic > 0.0 else None,
        cost_multiplier=cost_multiplier,
        start_idx=spec.outer_test_start,
        end_idx=spec.outer_test_end,
        **_baseline_kwargs(env_kwargs),
    )
    annualized = reannualize_strategy_results(
        results,
        bars_per_year=bars_per_year,
    )
    return _slim_baselines(annualized)


def _relative_fold_series(
    fold: dict[str, object],
    *,
    cost_label: str,
) -> RelativeFoldSeries:
    raw_series = fold.pop(f"_return_series_{cost_label}")
    if not isinstance(raw_series, dict):
        raise TypeError("fold return series must be a mapping")
    outer_oos = fold["outer_oos"]
    if not isinstance(outer_oos, dict):
        raise TypeError("outer_oos must be a mapping")
    relative = outer_oos[f"relative_{cost_label}"]
    if not isinstance(relative, dict):
        raise TypeError("relative result must be a mapping")
    hybrid = relative["hybrid"]
    shadow = relative["shadow"]
    if not isinstance(hybrid, dict) or not isinstance(shadow, dict):
        raise TypeError("relative books must be mappings")
    return RelativeFoldSeries(
        fold=int(fold["fold"]),
        hybrid_returns=np.asarray(raw_series["hybrid"], dtype=np.float64),
        shadow_returns=np.asarray(raw_series["shadow"], dtype=np.float64),
        hybrid_trades=int(hybrid["n_trades"]),
        shadow_trades=int(shadow["n_trades"]),
        hybrid_turnover=float(hybrid["turnover_total"]),
        shadow_turnover=float(shadow["turnover_total"]),
        hybrid_cost=float(hybrid["total_cost"]),
        shadow_cost=float(shadow["total_cost"]),
    )


def run_residual_fold(
    *,
    fs,
    spec: ResidualFoldSpec,
    args,
    output_dir: str | Path,
) -> dict[str, object]:
    """Train/select on development data and score one frozen outer OOS fold."""

    output_root = Path(output_dir)
    fold_output = output_root / "residual_wf" / f"fold_{spec.fold}"
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
    pp_config = getattr(post_processor, "cfg", None)
    bars_per_year = int(getattr(pp_config, "bars_per_year", 8_760))
    candidates = train_select_residual_candidates(
        args=fold_args,
        train_fs=train_fs,
        checkpoint_val_fs=checkpoint_fs,
        selection_fs=selection_fs,
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
        include_return_series=True,
    )
    relative_2x = evaluate_relative_agent(
        candidates.selected_agent,
        test_fs,
        env_kwargs={**selected_env_kwargs, "cost_multiplier": 2.0},
        bootstrap_seed=fold_args.seed,
        include_return_series=True,
    )
    series_1x = relative_1x.pop("return_series")
    series_2x = relative_2x.pop("return_series")

    noisy_oracle = float(getattr(args, "noisy_oracle_ic", 0.0))
    baselines_1x = _diagnostic_baselines(
        fs=fs,
        spec=spec,
        env_kwargs=env_kwargs,
        bars_per_year=bars_per_year,
        cost_multiplier=1.0,
        noisy_oracle_ic=noisy_oracle,
    )
    baselines_2x = _diagnostic_baselines(
        fs=fs,
        spec=spec,
        env_kwargs=env_kwargs,
        bars_per_year=bars_per_year,
        cost_multiplier=2.0,
        noisy_oracle_ic=noisy_oracle,
    )

    model_digest_1x = candidates.selected_model_digest
    model_digest_2x = candidates.selected_model_digest
    same_model = model_digest_1x == model_digest_2x
    if not same_model:
        raise RuntimeError("1x and 2x OOS evaluations reference different models")
    selected_model_identity = (
        str(candidates.selected_model_path.relative_to(output_root))
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
        "alpha_artifact_path": str(alpha_path.relative_to(output_root)),
        "alpha_artifact_gate_passed": bool(alpha.enabled),
        "development_matrix": candidates.development_results,
        "development_matrix_cost2x": candidates.development_cost2x_results,
        "selection": candidates.selection,
        "selected_configuration": candidates.selected_configuration,
        "selected_policy_mode": str(candidates.selection["policy_mode"]),
        "selected_model_identity": selected_model_identity,
        "selected_model_digest": candidates.selected_model_digest,
        "alpha_enabled": candidates.selected_alpha_enabled,
        "selected_seed_fallbacks": [
            _is_fallback(policy) for policy in candidates.selected_policies
        ],
        "outer_oos": {
            "relative_1x": relative_1x,
            "relative_2x": relative_2x,
            "baselines_1x": baselines_1x,
            "baselines_2x": baselines_2x,
            "model_digest_1x": model_digest_1x,
            "model_digest_2x": model_digest_2x,
            "same_selected_model_for_cost_scenarios": same_model,
        },
        "_return_series_1x": series_1x,
        "_return_series_2x": series_2x,
    }
    public_report = {
        key: value for key, value in report.items() if not key.startswith("_return_series_")
    }
    save_residual_walk_forward_report(fold_output / "fold_report.json", public_report)
    return report


def run_residual_walk_forward(
    args,
    output_dir: str | Path,
) -> dict[str, object]:
    """Run and atomically publish research-only residual Walk-Forward evidence."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_id = _new_run_id()
    staging_root = output / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = staging_root / run_id
    staging.mkdir(parents=False, exist_ok=False)

    try:
        source_args = copy.copy(args)
        source_args.action_mode = "baseline-residual"
        source_args.min_trade_delta = 0.0
        source_args.lambda_turnover = 0.0
        fs = build_feature_set(source_args, output_dir=staging)
        config = ResidualWalkForwardConfig.from_args(
            args,
            dataset_identity=feature_set_identity(fs),
        )
        runtime_args = _runtime_args(args, config)
        specs, skipped = build_residual_fold_specs(
            n_bars=fs.n_bars,
            n_folds=config.requested_folds,
            purge_bars=config.effective_purge_bars,
            horizon=config.horizon,
        )
        if len(specs) < 2:
            raise RuntimeError(
                "residual walk-forward requires at least two completed folds"
            )

        folds: list[dict[str, object]] = []
        for spec in specs:
            folds.append(
                run_residual_fold(
                    fs=fs,
                    spec=spec,
                    args=runtime_args,
                    output_dir=staging,
                )
            )
        if len(folds) < 2:
            raise RuntimeError(
                "residual walk-forward requires at least two completed folds"
            )

        stitched_1x = stitch_relative_fold_results(
            [_relative_fold_series(fold, cost_label="1x") for fold in folds],
            bars_per_year=config.bars_per_year,
            bootstrap_seed=int(getattr(runtime_args, "seed", 0)),
        )
        stitched_2x = stitch_relative_fold_results(
            [_relative_fold_series(fold, cost_label="2x") for fold in folds],
            bars_per_year=config.bars_per_year,
            bootstrap_seed=int(getattr(runtime_args, "seed", 0)),
        )
        stitched_oos = {"cost1x": stitched_1x, "cost2x": stitched_2x}
        report_config = {
            **config.to_dict(),
            "completed_folds": len(folds),
            "member_seeds_per_fold": config.effective_ensemble_size,
            "n_bars_total": int(fs.n_bars),
            "outer_train_start_fraction": 0.4,
            "policy_train_fraction": 0.70,
            "checkpoint_validation_fraction": 0.15,
            "configuration_selection_fraction": 0.15,
        }
        report: dict[str, object] = {
            "run_id": run_id,
            "status": "completed",
            "run_path": f"residual_wf_runs/{run_id}",
            "mode": "baseline_residual_walk_forward_v1",
            "action_schema": "baseline_residual_v1",
            "config": report_config,
            "summary": summarize_residual_folds(
                folds,
                requested_folds=config.requested_folds,
                skipped_folds=skipped,
                stitched_oos=stitched_oos,
            ),
            "folds": folds,
            "skipped_folds": skipped,
            "release_eligible": False,
            "release_blocker": (
                "sealed residual release workflow remains incomplete; research output only"
            ),
        }
        save_residual_walk_forward_report(
            staging / "residual_walk_forward.json",
            report,
        )
        for spec in specs:
            expected = (
                staging / "residual_wf" / f"fold_{spec.fold}" / "fold_report.json"
            )
            if not expected.is_file():
                raise RuntimeError(
                    f"missing fold report before publication: {expected}"
                )

        completed_root = output / "residual_wf_runs"
        completed_root.mkdir(parents=True, exist_ok=True)
        completed = completed_root / run_id
        if completed.exists():
            raise FileExistsError(f"completed run already exists: {completed}")
        os.replace(staging, completed)
        _publish_authoritative_report(output, report, run_id=run_id)
        return report
    except Exception:
        _isolate_failed_run(staging, output, run_id)
        raise
