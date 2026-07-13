"""Single authoritative command-line interface for trade_rl."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import TextIO

from trade_rl import __version__
from trade_rl.data.market import MarketCalendarKind
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.rl.actions import ActionSpec, AlphaContract, AlphaSignalKind
from trade_rl.rl.configuration import EnvironmentExperimentManifest
from trade_rl.rl.environment import ResidualMarketEnvConfig
from trade_rl.rl.observations import OBSERVATION_SCHEMA
from trade_rl.rl.rewards import RewardConfig
from trade_rl.rl.training import ResidualTrainingConfig, gamma_from_half_life
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendMode
from trade_rl.workflows.walk_forward import WalkForwardWorkflowConfig


def _write_json(stdout: TextIO, payload: object) -> None:
    stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    stdout.write("\n")


def _status_handler(area: str) -> Callable[[argparse.Namespace, TextIO], int]:
    def handler(_: argparse.Namespace, stdout: TextIO) -> int:
        _write_json(
            stdout,
            {
                "area": area,
                "authoritative_package": "trade_rl",
                "production_status": "NO-GO",
                "schema": "component_status_v1",
            },
        )
        return 0

    return handler


def _train_config(args: argparse.Namespace, stdout: TextIO) -> int:
    decision_hours: float | None
    half_life_hours: float | None
    if args.gamma is not None:
        if args.discount_half_life_hours is not None:
            raise ValueError(
                "explicit --gamma cannot be combined with --discount-half-life-hours"
            )
        gamma = float(args.gamma)
        decision_hours = (
            None if args.decision_hours is None else float(args.decision_hours)
        )
        half_life_hours = None
    else:
        if args.decision_hours is None or args.discount_half_life_hours is None:
            raise ValueError(
                "provide --gamma or both --decision-hours and "
                "--discount-half-life-hours"
            )
        decision_hours = float(args.decision_hours)
        half_life_hours = float(args.discount_half_life_hours)
        gamma = gamma_from_half_life(
            decision_hours=decision_hours,
            half_life_hours=half_life_hours,
        )

    config = ResidualTrainingConfig(
        timesteps=args.timesteps,
        gamma=gamma,
        seeds=tuple(args.seed),
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        normalize_advantage=not args.no_normalize_advantage,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy=args.policy,
        device=args.device,
        decision_hours=decision_hours,
        discount_half_life_hours=half_life_hours,
        log_std_init=args.log_std_init,
        target_kl=args.target_kl,
        use_sde=args.use_sde,
        sde_sample_freq=args.sde_sample_freq,
        policy_net_arch=tuple(args.policy_net_arch),
        asset_set_encoder=not args.no_asset_set_encoder,
        asset_embedding_dim=args.asset_embedding_dim,
        global_embedding_dim=args.global_embedding_dim,
        algorithm=args.algorithm,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
    )
    _write_json(
        stdout,
        {
            "actual_timesteps": config.rounded_timesteps,
            "algorithm": config.algorithm,
            "asset_embedding_dim": config.asset_embedding_dim,
            "asset_set_encoder": config.asset_set_encoder,
            "batch_size": config.batch_size,
            "buffer_size": config.buffer_size,
            "clip_range": config.clip_range,
            "decision_hours": config.decision_hours,
            "device": config.device,
            "discount_half_life_hours": config.discount_half_life_hours,
            "ent_coef": config.ent_coef,
            "gae_lambda": config.gae_lambda,
            "gamma": config.gamma,
            "gradient_steps": config.gradient_steps,
            "global_embedding_dim": config.global_embedding_dim,
            "learning_rate": config.learning_rate,
            "learning_starts": config.learning_starts,
            "log_std_init": config.log_std_init,
            "max_grad_norm": config.max_grad_norm,
            "n_epochs": config.n_epochs,
            "n_steps": config.n_steps,
            "normalize_advantage": config.normalize_advantage,
            "observation_schema": OBSERVATION_SCHEMA,
            "policy": config.policy,
            "policy_net_arch": list(config.policy_net_arch),
            "requested_timesteps": config.timesteps,
            "schema": "residual_training_config_v4",
            "sde_sample_freq": config.sde_sample_freq,
            "seeds": list(config.seeds),
            "target_kl": config.target_kl,
            "train_freq": config.train_freq,
            "use_sde": config.use_sde,
            "vf_coef": config.vf_coef,
        },
    )
    return 0


def _environment_config(args: argparse.Namespace, stdout: TextIO) -> int:
    reward = RewardConfig(
        scale=args.reward_scale,
        absolute_growth_weight=args.absolute_growth_weight,
        excess_growth_weight=args.excess_growth_weight,
        incremental_drawdown_weight=args.incremental_drawdown_weight,
        drawdown_dead_zone=args.drawdown_dead_zone,
        baseline_underperformance_weight=args.baseline_underperformance_weight,
        baseline_window_hours=args.baseline_window_hours,
        baseline_window_steps=args.baseline_window_steps,
        baseline_tolerance=args.baseline_tolerance,
        baseline_progressive_power=args.baseline_progressive_power,
        projection_penalty_weight=args.projection_penalty_weight,
        terminal_equity_weight=args.terminal_equity_weight,
    )
    execution = ExecutionCostConfig(
        fee_rate=args.fee_rate,
        spread_rate=args.spread_rate,
        impact_rate=args.impact_rate,
        max_participation_rate=args.max_participation_rate,
        minimum_notional=args.minimum_notional,
        lot_size=args.lot_size,
        tick_size=args.tick_size,
        allow_short=not args.no_short,
        max_leverage=args.max_leverage,
        maintenance_margin_rate=args.maintenance_margin_rate,
        collateral_haircut=args.collateral_haircut,
        margin_mode=args.margin_mode,
        order_latency_bars=args.order_latency_bars,
        order_type=args.order_type,
        limit_offset_rate=args.limit_offset_rate,
    )
    environment = ResidualMarketEnvConfig(
        episode_hours=args.episode_hours,
        decision_hours=args.decision_hours,
        episode_hour_choices=tuple(args.episode_hour_choice or ()),
        reward_scale=args.reward_scale,
        initial_capital=args.initial_capital,
        minimum_equity_fraction=args.minimum_equity_fraction,
        reward_config=reward,
        liquidate_on_end=args.liquidate_on_end,
        finite_horizon_observation=args.finite_horizon_observation,
        initial_state_modes=tuple(args.initial_state_mode or ("cash",)),
        random_initial_gross=args.random_initial_gross,
        stress_drawdown_fraction=args.stress_drawdown_fraction,
        partial_fill_fraction=args.partial_fill_fraction,
        episode_sampling_mode=args.episode_sampling_mode,
        regime_feature_index=args.regime_feature_index,
        regime_bins=args.regime_bins,
        stress_quantile=args.stress_quantile,
        execution_cost=execution,
    )
    risk = PreTradeRiskConfig(
        max_gross=args.max_gross,
        max_abs_weight=args.max_abs_weight,
        max_turnover=args.max_turnover,
        drawdown_start=args.drawdown_start,
        drawdown_stop=args.drawdown_stop,
    )
    trend = TrendConfig(
        fast_hours=args.fast_hours,
        base_hours=args.base_hours,
        slow_hours=args.slow_hours,
        mode=TrendMode(args.trend_mode),
    )
    action = ActionSpec(
        alpha_enabled=args.alpha_artifact_digest is not None,
        n_factors=args.factor_count,
    )
    alpha_contract = AlphaContract(kind=AlphaSignalKind(args.alpha_signal_kind))
    manifest = EnvironmentExperimentManifest.build(
        calendar_kind=MarketCalendarKind(args.calendar_kind),
        action_spec=action,
        alpha_contract=alpha_contract,
        environment=environment,
        risk=risk,
        reward=reward,
        trend=trend,
        alpha_artifact_digest=args.alpha_artifact_digest,
        factor_artifact_digest=args.factor_artifact_digest,
        normalizer_digest=args.normalizer_digest,
    )
    _write_json(
        stdout,
        {
            "digest": manifest.digest,
            "schema": manifest.schema_version,
            "action_spec": asdict(manifest.action_spec),
            "alpha_contract": asdict(manifest.alpha_contract),
            "calendar_kind": MarketCalendarKind(manifest.calendar_kind).value,
            "environment": asdict(manifest.environment),
            "risk": asdict(manifest.risk),
            "reward": asdict(manifest.reward),
            "trend": asdict(manifest.trend),
            "alpha_artifact_digest": manifest.alpha_artifact_digest,
            "factor_artifact_digest": manifest.factor_artifact_digest,
            "normalizer_digest": manifest.normalizer_digest,
        },
    )
    return 0


def _range(value: object) -> list[int]:
    start = getattr(value, "start")
    stop = getattr(value, "stop")
    if not isinstance(start, int) or not isinstance(stop, int):
        raise TypeError("fold range must expose integer start and stop values")
    return [start, stop]


def _walk_forward_plan(args: argparse.Namespace, stdout: TextIO) -> int:
    config = WalkForwardWorkflowConfig(
        n_bars=args.bars,
        train_bars=args.train_bars,
        checkpoint_bars=args.checkpoint_bars,
        selection_bars=args.selection_bars,
        test_bars=args.test_bars,
        purge_bars=args.purge_bars,
        step_bars=args.step_bars,
        max_folds=args.max_folds,
        expanding_train=not args.rolling_train,
    )
    folds = config.build_folds()
    _write_json(
        stdout,
        {
            "config": asdict(config),
            "folds": [
                {
                    "checkpoint_validation": _range(fold.checkpoint_validation),
                    "configuration_selection": _range(fold.configuration_selection),
                    "fold_index": fold.fold_index,
                    "test": _range(fold.test),
                    "train": _range(fold.train),
                }
                for fold in folds
            ],
            "schema": "walk_forward_plan_v1",
        },
    )
    return 0


def _add_status_group(
    subparsers: argparse._SubParsersAction,
    *,
    name: str,
    help_text: str,
) -> None:
    group = subparsers.add_parser(name, help=help_text)
    commands = group.add_subparsers(dest=f"{name}_command", required=True)
    status = commands.add_parser("status", help=f"show {name} boundary status")
    status.set_defaults(handler=_status_handler(name))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trade-rl",
        description="Baseline-anchored residual RL research tooling.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="show the installed trade-rl version",
    )
    subparsers = parser.add_subparsers(dest="command")

    _add_status_group(subparsers, name="data", help_text="dataset validation")
    _add_status_group(subparsers, name="signal", help_text="signal artifacts and gates")

    environment = subparsers.add_parser(
        "environment",
        help="validate a complete market-environment manifest",
    )
    environment_commands = environment.add_subparsers(
        dest="environment_command",
        required=True,
    )
    environment_config = environment_commands.add_parser("config")
    environment_config.add_argument("--initial-capital", type=float, required=True)
    environment_config.add_argument(
        "--calendar-kind",
        choices=tuple(item.value for item in MarketCalendarKind),
        default=MarketCalendarKind.CONTINUOUS.value,
    )
    environment_config.add_argument("--episode-hours", type=float, default=720.0)
    environment_config.add_argument("--decision-hours", type=float, default=4.0)
    environment_config.add_argument(
        "--episode-hour-choice", type=float, action="append"
    )
    environment_config.add_argument(
        "--initial-state-mode",
        action="append",
        choices=("cash", "baseline", "random", "stress", "partial_fill"),
    )
    environment_config.add_argument(
        "--episode-sampling-mode",
        choices=("uniform", "regime_balanced", "stress_tail"),
        default="uniform",
    )
    environment_config.add_argument("--regime-feature-index", type=int, default=0)
    environment_config.add_argument("--regime-bins", type=int, default=4)
    environment_config.add_argument("--stress-quantile", type=float, default=0.9)
    environment_config.add_argument("--random-initial-gross", type=float, default=0.5)
    environment_config.add_argument(
        "--stress-drawdown-fraction", type=float, default=0.15
    )
    environment_config.add_argument("--partial-fill-fraction", type=float, default=0.5)
    environment_config.add_argument(
        "--minimum-equity-fraction", type=float, default=1e-6
    )
    environment_config.add_argument("--liquidate-on-end", action="store_true")
    environment_config.add_argument("--finite-horizon-observation", action="store_true")
    environment_config.add_argument("--fee-rate", type=float, default=0.0005)
    environment_config.add_argument("--spread-rate", type=float, default=0.0002)
    environment_config.add_argument("--impact-rate", type=float, default=0.0001)
    environment_config.add_argument(
        "--max-participation-rate", type=float, default=0.05
    )
    environment_config.add_argument("--minimum-notional", type=float, default=0.0)
    environment_config.add_argument("--lot-size", type=float, default=0.0)
    environment_config.add_argument("--tick-size", type=float, default=0.0)
    environment_config.add_argument("--no-short", action="store_true")
    environment_config.add_argument("--max-leverage", type=float, default=1.0)
    environment_config.add_argument(
        "--maintenance-margin-rate", type=float, default=0.25
    )
    environment_config.add_argument("--collateral-haircut", type=float, default=1.0)
    environment_config.add_argument(
        "--margin-mode", choices=("cross", "isolated"), default="cross"
    )
    environment_config.add_argument("--order-latency-bars", type=int, default=0)
    environment_config.add_argument(
        "--order-type", choices=("market", "limit"), default="market"
    )
    environment_config.add_argument("--limit-offset-rate", type=float, default=0.0005)
    environment_config.add_argument("--max-gross", type=float, default=1.0)
    environment_config.add_argument("--max-abs-weight", type=float, default=0.4)
    environment_config.add_argument("--max-turnover", type=float, default=1.0)
    environment_config.add_argument("--drawdown-start", type=float, default=0.1)
    environment_config.add_argument("--drawdown-stop", type=float, default=0.2)
    environment_config.add_argument("--reward-scale", type=float, default=100.0)
    environment_config.add_argument("--absolute-growth-weight", type=float, default=1.0)
    environment_config.add_argument("--excess-growth-weight", type=float, default=0.25)
    environment_config.add_argument(
        "--incremental-drawdown-weight", type=float, default=0.1
    )
    environment_config.add_argument("--drawdown-dead-zone", type=float, default=0.0025)
    environment_config.add_argument(
        "--baseline-underperformance-weight", type=float, default=0.15
    )
    environment_config.add_argument(
        "--baseline-window-hours", type=float, default=168.0
    )
    environment_config.add_argument("--baseline-window-steps", type=int)
    environment_config.add_argument("--baseline-tolerance", type=float, default=0.005)
    environment_config.add_argument(
        "--baseline-progressive-power", type=float, default=2.0
    )
    environment_config.add_argument(
        "--projection-penalty-weight", type=float, default=0.01
    )
    environment_config.add_argument("--terminal-equity-weight", type=float, default=1.0)
    environment_config.add_argument("--fast-hours", type=float, default=24.0)
    environment_config.add_argument("--base-hours", type=float, default=48.0)
    environment_config.add_argument("--slow-hours", type=float, default=96.0)
    environment_config.add_argument(
        "--trend-mode",
        choices=tuple(item.value for item in TrendMode),
        default=TrendMode.AUTO.value,
    )
    environment_config.add_argument(
        "--alpha-signal-kind",
        choices=tuple(item.value for item in AlphaSignalKind),
        default=AlphaSignalKind.TARGET_WEIGHT.value,
    )
    environment_config.add_argument("--alpha-artifact-digest")
    environment_config.add_argument("--factor-count", type=int, default=0)
    environment_config.add_argument("--factor-artifact-digest")
    environment_config.add_argument("--normalizer-digest")
    environment_config.set_defaults(handler=_environment_config)

    train = subparsers.add_parser("train", help="residual-policy training")
    train_commands = train.add_subparsers(dest="train_command", required=True)
    train_config = train_commands.add_parser(
        "config",
        help="validate and display a residual training configuration",
    )
    train_config.add_argument("--timesteps", type=int, required=True)
    train_config.add_argument(
        "--algorithm",
        choices=("ppo", "sac", "td3", "tqc"),
        default="ppo",
    )
    train_config.add_argument("--gamma", type=float)
    train_config.add_argument("--decision-hours", type=float)
    train_config.add_argument("--discount-half-life-hours", type=float)
    train_config.add_argument("--learning-rate", type=float, default=3e-4)
    train_config.add_argument("--n-steps", type=int, default=2_048)
    train_config.add_argument("--batch-size", type=int, default=64)
    train_config.add_argument("--buffer-size", type=int, default=100_000)
    train_config.add_argument("--learning-starts", type=int, default=10_000)
    train_config.add_argument("--train-freq", type=int, default=1)
    train_config.add_argument("--gradient-steps", type=int, default=1)
    train_config.add_argument("--n-epochs", type=int, default=10)
    train_config.add_argument("--gae-lambda", type=float, default=0.95)
    train_config.add_argument("--clip-range", type=float, default=0.2)
    train_config.add_argument("--ent-coef", type=float, default=0.0)
    train_config.add_argument("--vf-coef", type=float, default=0.5)
    train_config.add_argument("--max-grad-norm", type=float, default=0.5)
    train_config.add_argument("--log-std-init", type=float, default=-0.5)
    train_config.add_argument("--target-kl", type=float, default=0.02)
    train_config.add_argument("--use-sde", action="store_true")
    train_config.add_argument("--sde-sample-freq", type=int, default=-1)
    train_config.add_argument(
        "--policy-net-arch",
        type=int,
        action="append",
        default=None,
        help="repeat for each hidden layer; defaults to 128,128",
    )
    train_config.add_argument("--asset-embedding-dim", type=int, default=64)
    train_config.add_argument("--global-embedding-dim", type=int, default=64)
    train_config.add_argument("--no-asset-set-encoder", action="store_true")
    train_config.add_argument("--policy", default="MlpPolicy")
    train_config.add_argument("--device", default="auto")
    train_config.add_argument("--no-normalize-advantage", action="store_true")
    train_config.add_argument("--seed", type=int, action="append", required=True)
    train_config.set_defaults(handler=_train_config)

    walk_forward = subparsers.add_parser(
        "walk-forward",
        help="nested walk-forward planning and execution",
    )
    walk_forward_commands = walk_forward.add_subparsers(
        dest="walk_forward_command",
        required=True,
    )
    plan = walk_forward_commands.add_parser(
        "plan",
        help="construct and validate fold boundaries",
    )
    plan.add_argument("--bars", type=int, required=True)
    plan.add_argument("--train-bars", type=int, required=True)
    plan.add_argument("--checkpoint-bars", type=int, required=True)
    plan.add_argument("--selection-bars", type=int, required=True)
    plan.add_argument("--test-bars", type=int, required=True)
    plan.add_argument("--purge-bars", type=int, required=True)
    plan.add_argument("--step-bars", type=int)
    plan.add_argument("--max-folds", type=int)
    plan.add_argument("--rolling-train", action="store_true")
    plan.set_defaults(handler=_walk_forward_plan)

    _add_status_group(subparsers, name="evaluate", help_text="evaluation and gates")
    _add_status_group(subparsers, name="registry", help_text="artifact registry")
    _add_status_group(subparsers, name="serve", help_text="serving bundle runtime")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
) -> int:
    output = stdout or sys.stdout
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.version:
        output.write(f"trade-rl {__version__}\n")
        return 0
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help(file=output)
        return 2
    if getattr(args, "policy_net_arch", None) is None:
        args.policy_net_arch = [128, 128]
    return int(handler(args, output))


if __name__ == "__main__":
    raise SystemExit(main())
