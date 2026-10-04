from __future__ import annotations

import numpy as np

from tests.evaluation.experiments.test_delta import _resolved
from trade_rl.evaluation.experiments.contracts import ControlledFactor
from trade_rl.evaluation.experiments.contracts.run import ResolvedRunConfig
from trade_rl.evaluation.experiments.delta import FACTOR_RULES
from trade_rl.evaluation.runs.candidate_suite import LeanCandidateConfig
from trade_rl.evaluation.runs.config import CandidateRunConfig
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.strategies.rl.intent import PPO_OBSERVATION_SCHEMA_V3
from trade_rl.strategies.rl.ppo_training import (
    PPO_DEFAULT_GAE_LAMBDA,
    PPO_DEFAULT_GAMMA,
    PPO_REWARD_SCHEMA,
    PPO_TRAINING_LAYOUT_INTERLEAVED,
    PPO_TRAINING_LAYOUT_SEQUENTIAL,
)


def test_ppo_discount_factor_changes_only_gamma() -> None:
    rule = FACTOR_RULES[ControlledFactor.PPO_DISCOUNT]

    assert rule.allowed_paths == frozenset({("ppo_gamma",)})
    assert rule.unaffected_strategies == frozenset(
        {
            "cash",
            "constant_long",
            "constant_short",
            "trend",
            "mean_reversion",
            "ridge24",
            "lightgbm24",
        }
    )


def test_resolved_run_v6_binds_fixed_reward_contract_and_gamma() -> None:
    config = _resolved(
        schema_version="resolved_run_config_v6",
        ppo_observation_schema="ppo_observation_v2",
        ppo_gamma=0.9975,
    )

    payload = config.to_payload()

    assert payload["schema_version"] == "resolved_run_config_v6"
    assert payload["ppo_reward_schema"] == PPO_REWARD_SCHEMA
    assert payload["ppo_gamma"] == 0.9975
    assert payload["ppo_gae_lambda"] == PPO_DEFAULT_GAE_LAMBDA


def test_ppo_objective_fields_preserve_prior_positional_config_parameters() -> None:
    def timestamp(day: int) -> np.datetime64:
        return np.datetime64(f"2026-01-{day:02}T00:00:00", "ns")

    candidate = CandidateRunConfig(
        "signal",
        ("signal",),
        ("BTCUSDT",),
        timestamp(1),
        timestamp(2),
        timestamp(3),
        0.1,
        0.02,
        0.01,
        0.002,
        128,
        7,
        0.5,
        1_000.0,
        PPO_TRAINING_LAYOUT_INTERLEAVED,
        64,
        2,
        PPO_OBSERVATION_SCHEMA_V3,
        True,
        PreTradeRiskConfig(),
    )
    lean = LeanCandidateConfig(
        0,
        (0,),
        (0,),
        timestamp(1),
        0.1,
        0.02,
        0.01,
        0.002,
        128,
        7,
        PPO_TRAINING_LAYOUT_INTERLEAVED,
        64,
        2,
        PPO_OBSERVATION_SCHEMA_V3,
        True,
    )
    resolved = ResolvedRunConfig(
        "signal",
        0,
        ("signal",),
        (0,),
        ("BTCUSDT",),
        (0,),
        "2026-01-01T00:00:00.000000000",
        0.1,
        0.02,
        0.01,
        0.002,
        128,
        7,
        "2026-01-02T00:00:00.000000000",
        "2026-01-03T00:00:00.000000000",
        0.5,
        1_000.0,
        "zero_overlay_dataset_fields_authoritative",
        None,
        (),
        "resolved_run_config_v1",
        PPO_TRAINING_LAYOUT_SEQUENTIAL,
        None,
        0,
        False,
        None,
    )

    assert candidate.ppo_training_layout == PPO_TRAINING_LAYOUT_INTERLEAVED
    assert candidate.ppo_rollout_steps_per_env == 64
    assert candidate.ppo_minimum_hold_bars == 2
    assert candidate.ppo_gamma == PPO_DEFAULT_GAMMA
    assert lean.ppo_training_layout == PPO_TRAINING_LAYOUT_INTERLEAVED
    assert lean.ppo_rollout_steps_per_env == 64
    assert lean.ppo_minimum_hold_bars == 2
    assert lean.ppo_gamma == PPO_DEFAULT_GAMMA
    assert resolved.schema_version == "resolved_run_config_v1"
    assert resolved.ppo_training_layout == PPO_TRAINING_LAYOUT_SEQUENTIAL
    assert resolved.ppo_minimum_hold_bars == 0
    assert resolved.ppo_gamma == PPO_DEFAULT_GAMMA
