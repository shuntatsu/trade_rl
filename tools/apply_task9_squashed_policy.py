from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing Task 9 anchor in {path}: {old[:160]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, block: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if marker not in text:
        target.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def add_tests() -> None:
    append_once(
        "tests/rl/test_sequence_policy_core.py",
        "test_shared_sequence_policy_uses_squashed_target_weight_distribution",
        r'''
def test_shared_sequence_policy_uses_squashed_target_weight_distribution() -> None:
    import numpy as np
    from gymnasium import spaces
    from stable_baselines3.common.distributions import (
        SquashedDiagGaussianDistribution,
    )

    from trade_rl.rl.policies import (
        SequenceAssetFeatureExtractor,
        SharedPerAssetActorCriticPolicy,
    )

    n_symbols = 2
    timeframes = ("15m", "1h", "4h", "1d")
    feature_counts = {timeframe: 2 for timeframe in timeframes}
    window_lengths = {timeframe: 3 for timeframe in timeframes}
    components: dict[str, spaces.Space] = {
        "current_snapshot": spaces.Box(
            -10.0, 10.0, shape=(n_symbols, 8), dtype=np.float32
        ),
        "asset_state": spaces.Box(
            -10.0, 10.0, shape=(n_symbols, 4), dtype=np.float32
        ),
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
    policy = SharedPerAssetActorCriticPolicy(
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
    assert isinstance(policy.action_dist, SquashedDiagGaussianDistribution)
    assert policy.action_distribution_name == "squashed_diag_gaussian"

    batch = 4
    observations: dict[str, torch.Tensor] = {}
    for key, space in observation_space.spaces.items():
        array = np.zeros((batch, *space.shape), dtype=space.dtype)
        if key == "active" or key.endswith("_available"):
            array.fill(1)
        observations[key] = torch.as_tensor(array)

    distribution = policy.get_distribution(observations)
    stochastic = distribution.get_actions(deterministic=False)
    deterministic = distribution.get_actions(deterministic=True)
    assert torch.all(stochastic <= 1.0)
    assert torch.all(stochastic >= -1.0)
    assert torch.all(deterministic <= 1.0)
    assert torch.all(deterministic >= -1.0)

    boundary = torch.tensor(
        [[0.999, -0.999], [-0.999, 0.999], [0.5, -0.5], [0.0, 0.0]],
        dtype=torch.float32,
    )
    values, log_prob, entropy = policy.evaluate_actions(observations, boundary)
    assert torch.isfinite(values).all()
    assert torch.isfinite(log_prob).all()
    assert entropy is None
''',
    )

    replace_once(
        "tests/learning/test_behavior_cloning.py",
        '''class _Distribution:
    def __init__(self, mean: torch.Tensor) -> None:
        self.distribution = self
        self.mean = mean
''',
        '''class _Distribution:
    def __init__(self, mean: torch.Tensor) -> None:
        self.distribution = self
        self.mean = mean

    def get_actions(self, *, deterministic: bool = False) -> torch.Tensor:
        assert deterministic is True
        return self.mean
''',
    )
    append_once(
        "tests/learning/test_behavior_cloning.py",
        "test_behavior_cloning_uses_deterministic_action_space_output",
        r'''
class _SquashedDistribution:
    def __init__(self, raw_mean: torch.Tensor) -> None:
        self.distribution = type("Gaussian", (), {"mean": raw_mean})()
        self.action_mode = torch.tanh(raw_mean)

    def get_actions(self, *, deterministic: bool = False) -> torch.Tensor:
        assert deterministic is True
        return self.action_mode


class _SquashedPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw = torch.nn.Parameter(torch.tensor([[2.0]], dtype=torch.float32))
        self.device = torch.device("cpu")

    def get_distribution(self, observations: torch.Tensor) -> _SquashedDistribution:
        return _SquashedDistribution(self.raw.expand(len(observations), -1))


def test_behavior_cloning_uses_deterministic_action_space_output() -> None:
    from trade_rl.learning.behavior_cloning import _actor_mean

    observations = torch.zeros((3, 1), dtype=torch.float32)
    policy = _SquashedPolicy()
    action = _actor_mean(policy, observations)

    torch.testing.assert_close(action, torch.tanh(policy.raw).expand(3, -1))
    assert not torch.allclose(action, policy.raw.expand(3, -1))
''',
    )

    replace_once(
        "tests/integrations/test_sb3_training.py",
        '''from __future__ import annotations

from collections.abc import Callable
''',
        '''from __future__ import annotations

import json
from collections.abc import Callable
''',
    )
    replace_once(
        "tests/integrations/test_sb3_training.py",
        '''    class FakePolicy:
        def parameters(self) -> tuple[FakeParameter, ...]:
            return (FakeParameter(),)
''',
        '''    class FakePolicy:
        action_distribution_name = "squashed_diag_gaussian"

        def parameters(self) -> tuple[FakeParameter, ...]:
            return (FakeParameter(),)
''',
    )
    replace_once(
        "tests/integrations/test_sb3_training.py",
        '''    assert result.actual_timesteps == 2
    assert factory_calls == 1
''',
        '''    assert result.actual_timesteps == 2
    architecture = json.loads(
        (tmp_path / "model-architecture.json").read_text(encoding="utf-8")
    )
    assert architecture["architecture"].get("action_distribution") == (
        "squashed_diag_gaussian"
    )
    assert factory_calls == 1
''',
    )


def add_implementation() -> None:
    replace_once(
        "trade_rl/rl/policies.py",
        '''from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
''',
        '''from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
''',
    )
    replace_once(
        "trade_rl/rl/policies.py",
        '''class SharedPerAssetActorCriticPolicy(MultiInputActorCriticPolicy):
    """SB3 PPO policy with a truly shared actor head and portfolio-level critic."""

    def __init__(
''',
        '''class SharedPerAssetActorCriticPolicy(MultiInputActorCriticPolicy):
    """SB3 PPO policy with bounded shared target-weight actions."""

    action_distribution_name = "squashed_diag_gaussian"

    def __init__(
''',
    )
    replace_once(
        "trade_rl/rl/policies.py",
        '''    def _build(self, lr_schedule: Any) -> None:
        super()._build(lr_schedule)
        context_dim = 2 * self.shared_actor_d_model + self.shared_actor_global_dim
''',
        '''    def _build(self, lr_schedule: Any) -> None:
        self.action_dist = SquashedDiagGaussianDistribution(
            self.shared_actor_n_symbols
        )
        super()._build(lr_schedule)
        context_dim = 2 * self.shared_actor_d_model + self.shared_actor_global_dim
''',
    )

    replace_once(
        "trade_rl/learning/behavior_cloning.py",
        '''def _actor_mean(policy: Any, observations: Any) -> Any:
    distribution = policy.get_distribution(observations)
    mean = getattr(distribution.distribution, "mean", None)
    if mean is None:
        raise ValueError("policy distribution does not expose a continuous mean")
    return mean
''',
        '''def _actor_mean(policy: Any, observations: Any) -> Any:
    distribution = policy.get_distribution(observations)
    action = distribution.get_actions(deterministic=True)
    if action is None or not hasattr(action, "shape"):
        raise ValueError(
            "policy distribution does not expose deterministic action-space output"
        )
    return action
''',
    )

    replace_once(
        "trade_rl/integrations/sb3_training.py",
        '''            architecture_details: dict[str, object] = {
                "actor_net_arch": config.policy_net_arch,
                "critic_net_arch": config.value_net_arch,
                "sequence_encoder": config.sequence_encoder,
            }
''',
        '''            declared_distribution = getattr(
                model.policy, "action_distribution_name", None
            )
            action_distribution = (
                declared_distribution
                if isinstance(declared_distribution, str) and declared_distribution
                else type(getattr(model.policy, "action_dist", None)).__name__
            )
            architecture_details: dict[str, object] = {
                "action_distribution": action_distribution,
                "actor_net_arch": config.policy_net_arch,
                "critic_net_arch": config.value_net_arch,
                "sequence_encoder": config.sequence_encoder,
            }
''',
    )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: apply_task9_squashed_policy.py tests|implementation")
    if sys.argv[1] == "tests":
        add_tests()
    else:
        add_implementation()
