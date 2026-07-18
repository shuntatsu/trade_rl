from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from gymnasium import spaces

from trade_rl.integrations.sb3_serving import _SB3StructuredSequenceEnsemblePolicy
from trade_rl.learning.oracle_teacher import (
    OracleTeacherConfig,
    project_portfolio_targets,
)
from trade_rl.risk.portfolio import PortfolioRiskConfig
from trade_rl.rl.policies import (
    SequenceAssetFeatureExtractor,
    SharedPerAssetActionHead,
    SharedPerAssetActorCriticPolicy,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples" / "binance-multitimeframe"


def _runner_namespace() -> dict[str, Any]:
    sys.path.insert(0, str(EXAMPLE_ROOT))
    return vars(importlib.import_module("full_research_pipeline"))


def test_training_policy_digest_uses_cli_policy_digest_and_fails_closed() -> None:
    resolve = _runner_namespace()["_training_policy_digest"]

    assert resolve({"policy_digest": "a" * 64}) == "a" * 64
    with pytest.raises(ValueError, match="policy_digest"):
        resolve({"artifact_digest": "b" * 64})
    with pytest.raises(ValueError, match="policy_digest"):
        resolve({"policy_digest": "A" * 64})


def test_confirmation_recheck_is_a_thin_finalize_launcher() -> None:
    content = (EXAMPLE_ROOT / "recheck_confirmation.py").read_text(encoding="utf-8")

    assert "runpy" not in content
    assert '"--phase", "finalize"' in content
    assert "from run_full_research_state import main" in content


def _sequence_policy() -> SharedPerAssetActorCriticPolicy:
    n_symbols = 2
    timeframes = ("15m", "1h", "4h", "1d")
    feature_counts = {timeframe: 2 for timeframe in timeframes}
    window_lengths = {timeframe: 3 for timeframe in timeframes}
    components: dict[str, spaces.Space] = {
        "current_snapshot": spaces.Box(
            -10.0, 10.0, shape=(n_symbols, 8), dtype=np.float32
        ),
        "asset_state": spaces.Box(-10.0, 10.0, shape=(n_symbols, 4), dtype=np.float32),
        "global_state": spaces.Box(-10.0, 10.0, shape=(3,), dtype=np.float32),
        "active": spaces.Box(0.0, 1.0, shape=(n_symbols,), dtype=np.float32),
    }
    for timeframe in timeframes:
        shape = (n_symbols, 3, 2)
        components[f"sequence_{timeframe}_values"] = spaces.Box(
            -10.0, 10.0, shape=shape, dtype=np.float16
        )
        components[f"sequence_{timeframe}_available"] = spaces.Box(
            0, 1, shape=shape, dtype=np.uint8
        )
        components[f"sequence_{timeframe}_staleness"] = spaces.Box(
            0.0, 100.0, shape=shape, dtype=np.float16
        )
    observation_space = spaces.Dict(components)
    return SharedPerAssetActorCriticPolicy(
        observation_space,
        spaces.Box(-1.0, 1.0, shape=(n_symbols,), dtype=np.float32),
        lambda _: 1e-3,
        net_arch={"pi": [11], "vf": [13]},
        features_extractor_class=SequenceAssetFeatureExtractor,
        features_extractor_kwargs={
            "feature_counts": feature_counts,
            "window_lengths": window_lengths,
            "snapshot_width": 8,
            "asset_state_width": 4,
            "global_width": 3,
            "n_symbols": n_symbols,
            "d_model": 16,
            "attention_heads": 4,
            "attention_layers": 1,
            "dropout": 0.0,
        },
        shared_actor_n_symbols=n_symbols,
        shared_actor_d_model=16,
        shared_actor_global_dim=128,
        shared_actor_net_arch=(11,),
        log_std_init=-0.5,
    )


def _policy_observations(
    policy: SharedPerAssetActorCriticPolicy,
) -> dict[str, torch.Tensor]:
    observations: dict[str, torch.Tensor] = {}
    for key, space in policy.observation_space.spaces.items():
        value = np.zeros((2, *space.shape), dtype=space.dtype)
        if key.endswith("_available"):
            value.fill(1)
        observations[key] = torch.as_tensor(value)
    observations["active"][:, 0] = 1.0
    observations["active"][:, 1] = 0.0
    return observations


def test_shared_actor_uses_explicit_activity_and_one_shared_exploration_scale() -> None:
    head = SharedPerAssetActionHead(
        n_symbols=2,
        token_dim=3,
        context_dim=8,
        hidden_dims=(5,),
    ).eval()
    contexts = torch.randn(1, 2, 8)
    contexts[:, :, -1] = torch.tensor([[1.0, 0.0]])
    with torch.no_grad():
        output = head(contexts.reshape(1, -1))
    assert output[0, 1].item() == 0.0

    policy = _sequence_policy()
    assert tuple(policy.log_std.shape) == (1,)
    observations = _policy_observations(policy)
    distribution = policy.get_distribution(observations)
    stochastic = distribution.get_actions(deterministic=False)
    deterministic = distribution.get_actions(deterministic=True)
    assert torch.count_nonzero(stochastic[:, 1]) == 0
    assert torch.count_nonzero(deterministic[:, 1]) == 0

    active_equivalent = torch.tensor([[0.2, 0.0], [-0.1, 0.0]])
    inactive_changed = torch.tensor([[0.2, 0.8], [-0.1, -0.7]])
    _, first_log_prob, _ = policy.evaluate_actions(observations, active_equivalent)
    _, second_log_prob, _ = policy.evaluate_actions(observations, inactive_changed)
    torch.testing.assert_close(first_log_prob, second_log_prob)


def test_oracle_portfolio_projection_matches_supported_runtime_limits() -> None:
    risk = PortfolioRiskConfig(
        max_abs_weight=0.4,
        max_net_exposure=0.5,
        max_position_to_market_notional=0.1,
    )
    targets = np.asarray([[[0.8, 0.8], [-0.8, 0.2]]], dtype=np.float64)
    projected = project_portfolio_targets(
        targets,
        portfolio_value=np.asarray([100.0]),
        market_notional=np.asarray([50.0, 20.0]),
        config=risk,
    )

    assert projected.shape == targets.shape
    assert np.max(np.abs(projected[..., 0])) <= 0.05 + 1e-12
    assert np.max(np.abs(projected[..., 1])) <= 0.02 + 1e-12
    assert np.max(np.abs(projected.sum(axis=-1))) <= 0.5 + 1e-12

    with pytest.raises(ValueError, match="oracle portfolio risk"):
        OracleTeacherConfig(portfolio_risk=PortfolioRiskConfig(volatility_target=0.1))


def test_structured_serving_rejects_feature_recipe_mismatch() -> None:
    policy = object.__new__(_SB3StructuredSequenceEnsemblePolicy)
    policy.dataset_reference = {
        "symbols": ["BTCUSDT"],
        "feature_names": ["return_1"],
        "global_feature_names": ["market_regime"],
        "bar_hours": 1.0,
        "feature_config_digest": "a" * 64,
    }
    policy.builder = SimpleNamespace(layout_digest=lambda _: "c" * 64)
    policy.sequence_normalizer = SimpleNamespace(sequence_schema_digest="c" * 64)
    dataset = SimpleNamespace(
        symbols=("BTCUSDT",),
        feature_names=("return_1",),
        global_feature_names=("market_regime",),
        bar_hours=1.0,
        feature_config_digest="b" * 64,
    )

    with pytest.raises(ValueError, match="feature recipe"):
        policy._validate_dataset(dataset)
