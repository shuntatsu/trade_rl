from __future__ import annotations

import json
from io import StringIO

from trade_rl.cli.app import build_parser, main


def test_cli_exposes_one_authoritative_command_tree() -> None:
    help_text = build_parser().format_help()

    for command in (
        "data",
        "signal",
        "train",
        "walk-forward",
        "evaluate",
        "registry",
        "serve",
    ):
        assert command in help_text


def test_walk_forward_plan_outputs_typed_fold_plan() -> None:
    stdout = StringIO()

    exit_code = main(
        [
            "walk-forward",
            "plan",
            "--bars",
            "220",
            "--train-bars",
            "80",
            "--checkpoint-bars",
            "10",
            "--selection-bars",
            "10",
            "--test-bars",
            "20",
            "--purge-bars",
            "2",
            "--max-folds",
            "2",
        ],
        stdout=stdout,
    )

    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert payload["schema"] == "walk_forward_plan_v1"
    assert len(payload["folds"]) == 2
    assert payload["folds"][0]["train"] == [0, 80]
    assert payload["folds"][0]["test"] == [106, 126]


def test_train_config_outputs_validated_residual_configuration() -> None:
    stdout = StringIO()

    exit_code = main(
        [
            "train",
            "config",
            "--timesteps",
            "1024",
            "--gamma",
            "0.5",
            "--seed",
            "0",
            "--seed",
            "1",
        ],
        stdout=stdout,
    )

    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert payload == {
        "gamma": 0.5,
        "schema": "residual_training_config_v1",
        "seeds": [0, 1],
        "timesteps": 1024,
    }


def test_version_command_uses_trade_rl_package_name() -> None:
    stdout = StringIO()

    assert main(["--version"], stdout=stdout) == 0
    assert stdout.getvalue().startswith("trade-rl ")
