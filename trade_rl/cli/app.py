"""Single authoritative command-line interface for trade_rl."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import TextIO

from trade_rl import __version__
from trade_rl.rl.training import DEFAULT_RESIDUAL_GAMMA, ResidualTrainingConfig
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
    config = ResidualTrainingConfig(
        timesteps=args.timesteps,
        gamma=args.gamma,
        seeds=tuple(args.seed),
        allow_low_gamma=args.allow_low_gamma,
    )
    _write_json(
        stdout,
        {
            "allow_low_gamma": config.allow_low_gamma,
            "gamma": config.gamma,
            "schema": "residual_training_config_v2",
            "seeds": list(config.seeds),
            "timesteps": config.timesteps,
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
    train_config.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_RESIDUAL_GAMMA,
        help=(
            "PPO discount factor (default: 0.99). Values below 0.95 require "
            "--allow-low-gamma and are research-only ablations."
        ),
    )
    train_config.add_argument(
        "--allow-low-gamma",
        action="store_true",
        help="allow a residual gamma below 0.95 for an explicit research ablation",
    )
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
