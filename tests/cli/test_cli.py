from __future__ import annotations

import json
from io import StringIO

import pytest

from trade_rl.cli.app import main


def test_train_config_exposes_exploration_network_and_algorithm_contract() -> None:
    stdout = StringIO()
    assert main([
        "train", "config", "--timesteps", "1025", "--gamma", "0.5",
        "--n-steps", "512", "--batch-size", "64", "--log-std-init", "-1",
        "--target-kl", "0.01", "--policy-net-arch", "256", "--policy-net-arch", "128",
        "--seed", "0",
    ], stdout=stdout) == 0
    payload = json.loads(stdout.getvalue())
    assert payload["schema"] == "residual_training_config_v4"
    assert payload["actual_timesteps"] == 1536
    assert payload["log_std_init"] == pytest.approx(-1.0)
    assert payload["policy_net_arch"] == [256, 128]


def test_environment_manifest_covers_action_reward_execution_and_identity() -> None:
    stdout = StringIO()
    assert main([
        "environment", "config", "--initial-capital", "100000",
        "--calendar-kind", "continuous_24_7", "--factor-count", "2",
        "--factor-artifact-digest", "a" * 64, "--normalizer-digest", "b" * 64,
        "--episode-hour-choice", "168", "--episode-hour-choice", "720",
        "--initial-state-mode", "cash", "--initial-state-mode", "stress",
    ], stdout=stdout) == 0
    payload = json.loads(stdout.getvalue())
    assert len(payload["digest"]) == 64
    assert payload["action_spec"]["n_factors"] == 2
    assert payload["normalizer_digest"] == "b" * 64
