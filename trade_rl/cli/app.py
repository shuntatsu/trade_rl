"""Single authoritative command-line interface for trade_rl."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import TextIO

from trade_rl import __version__
from trade_rl.rl.observations import OBSERVATION_SCHEMA
from trade_rl.rl.training import ResidualTrainingConfig, gamma_from_half_life
from trade_rl.workflows.walk_forward import WalkForwardWorkflowConfig


def _write_json(stdout: TextIO, payload: object) -> None:
    stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    stdout.write("\n")


def _status_handler(
    area: str,
) -> Callable[[argparse.Namespace, TextIO], int]:
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
    if args.gamma is not None:
        if args.decision_hours is not None or args.discount_half_life_hours is not None:
            raise ValueError(
                "explicit --gamma cannot be combined with discount half-life options"
            )
        gamma = float(args.gamma)
        timing: dict[str, float] = {}
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
        timing = {
            "decision_hours": decision_hours,
            "discount_half_life_hours": half_life_hours,
        }

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
    )
    _write_json(
        stdout,
        {
            "actual_timesteps": config.rounded_timesteps,
            "batch_size": config.batch_size,
            "clip_range": config.clip_range,
            "device": config.device,
            "ent_coef": config.ent_coef,
            "gae_lambda": config.gae_lambda,
            "gamma": config.gamma,
            "learning_rate": config.learning_rate,
            "max_grad_norm": config.max_grad_norm,
            "n_epochs": config.n_epochs,
            "n_steps": config.n_steps,
            "normalize_advantage": config.normalize_advantage,
            "observation_schema": OBSERVATION_SCHEMA,
            "policy": config.policy,
            "requested_timesteps": config.timesteps,
            "schema": "residual_training_config_v3",
            "seeds": list(config.seeds),
            "vf_coef": config.vf_coef,
            **timing,
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

    train = subparsers.add_parser("train", help="residual-policy training")
    train_commands = train.add_subparsers(dest="train_command", required=True)
    train_config = train_commands.add_parser(
        "config",
        help="validate and display a residual training configuration",
    )
    train_config.add_argument("--timesteps", type=int, required=True)
    train_config.add_argument("--gamma", type=float)
    train_config.add_argument("--decision-hours", type=float)
    train_config.add_argument("--discount-half-life-hours", type=float)
    train_config.add_argument("--learning-rate", type=float, default=3e-4)
    train_config.add_argument("--n-steps", type=int, default=2_048)
    train_config.add_argument("--batch-size", type=int, default=64)
    train_config.add_argument("--n-epochs", type=int, default=10)
    train_config.add_argument("--gae-lambda", type=float, default=0.95)
    train_config.add_argument("--clip-range", type=float, default=0.2)
    train_config.add_argument("--ent-coef", type=float, default=0.0)
    train_config.add_argument("--vf-coef", type=float, default=0.5)
    train_config.add_argument("--max-grad-norm", type=float, default=0.5)
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
    return int(handler(args, output))


if __name__ == "__main__":
    raise SystemExit(main())
