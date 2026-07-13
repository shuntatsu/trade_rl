from __future__ import annotations

import json
from io import StringIO

import pytest

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


def test_train_config_defaults_to_validated_long_horizon_gamma() -> None:
    stdout = StringIO()

    exit_code = main(
        [
            "train",
            "config",
            "--timesteps",
            "1024",
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
        "allow_low_gamma": False,
        "gamma": 0.99,
        "schema": "residual_training_config_v2",
        "seeds": [0, 1],
        "timesteps": 1024,
    }


def test_train_config_rejects_low_gamma_without_explicit_override() -> None:
    with pytest.raises(ValueError, match="allow_low_gamma"):
        main(
            [
                "train",
                "config",
                "--timesteps",
                "1024",
                "--gamma",
                "0.5",
                "--seed",
                "0",
            ],
            stdout=StringIO(),
        )


def test_train_config_accepts_explicit_low_gamma_research_ablation() -> None:
    stdout = StringIO()

    exit_code = main(
        [
            "train",
            "config",
            "--timesteps",
            "1024",
            "--gamma",
            "0.5",
            "--allow-low-gamma",
            "--seed",
            "0",
        ],
        stdout=stdout,
    )

    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert payload["gamma"] == 0.5
    assert payload["allow_low_gamma"] is True
    assert payload["schema"] == "residual_training_config_v2"


def test_version_command_uses_trade_rl_package_name() -> None:
    stdout = StringIO()

    assert main(["--version"], stdout=stdout) == 0
    assert stdout.getvalue().startswith("trade-rl ")
