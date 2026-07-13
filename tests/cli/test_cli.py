from __future__ import annotations

import json
from io import StringIO

import pytest

from trade_rl.cli.app import build_parser, main
from trade_rl.rl.observations import OBSERVATION_SCHEMA


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


def test_train_config_outputs_explicit_ppo_configuration() -> None:
    stdout = StringIO()

    exit_code = main(
        [
            "train",
            "config",
            "--timesteps",
            "1025",
            "--gamma",
            "0.5",
            "--n-steps",
            "512",
            "--batch-size",
            "64",
            "--learning-rate",
            "0.0001",
            "--device",
            "cuda",
            "--seed",
            "0",
            "--seed",
            "1",
        ],
        stdout=stdout,
    )

    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert payload["schema"] == "residual_training_config_v3"
    assert payload["gamma"] == 0.5
    assert payload["seeds"] == [0, 1]
    assert payload["requested_timesteps"] == 1025
    assert payload["actual_timesteps"] == 1536
    assert payload["n_steps"] == 512
    assert payload["batch_size"] == 64
    assert payload["learning_rate"] == pytest.approx(0.0001)
    assert payload["device"] == "cuda"
    assert payload["observation_schema"] == OBSERVATION_SCHEMA


def test_train_config_can_resolve_gamma_from_real_time_half_life() -> None:
    stdout = StringIO()

    exit_code = main(
        [
            "train",
            "config",
            "--timesteps",
            "1024",
            "--decision-hours",
            "4",
            "--discount-half-life-hours",
            "24",
            "--seed",
            "0",
        ],
        stdout=stdout,
    )

    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert payload["schema"] == "residual_training_config_v3"
    assert payload["decision_hours"] == pytest.approx(4.0)
    assert payload["discount_half_life_hours"] == pytest.approx(24.0)
    assert payload["gamma"] ** 6 == pytest.approx(0.5)


def test_version_command_uses_trade_rl_package_name() -> None:
    stdout = StringIO()

    assert main(["--version"], stdout=stdout) == 0
    assert stdout.getvalue().startswith("trade-rl ")
