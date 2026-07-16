from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing Task 3 anchor in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_before(path: str, marker: str, block: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if block.splitlines()[0] in text:
        return
    if marker not in text:
        raise RuntimeError(f"missing append marker in {path}: {marker!r}")
    target.write_text(text.replace(marker, block.rstrip() + "\n\n" + marker, 1), encoding="utf-8")


def add_tests() -> None:
    append_before(
        "tests/rl/test_sequence_policy_core.py",
        "def test_partial_feature_availability_keeps_latest_timestep_usable()",
        '''def test_shared_per_asset_action_head_reuses_parameters_and_is_equivariant() -> None:
    from trade_rl.rl.policies import SharedPerAssetActionHead

    torch.manual_seed(29)
    single = SharedPerAssetActionHead(
        n_symbols=1,
        token_dim=4,
        context_dim=9,
        hidden_dims=(7, 5),
    ).eval()
    shared = SharedPerAssetActionHead(
        n_symbols=3,
        token_dim=4,
        context_dim=9,
        hidden_dims=(7, 5),
    ).eval()
    shared.shared_head.load_state_dict(single.shared_head.state_dict())

    assert sum(parameter.numel() for parameter in single.parameters()) == sum(
        parameter.numel() for parameter in shared.parameters()
    )
    contexts = torch.randn(2, 3, 9)
    permutation = torch.tensor([2, 0, 1])
    with torch.no_grad():
        original = shared(contexts.reshape(2, -1))
        permuted = shared(contexts[:, permutation].reshape(2, -1))

    torch.testing.assert_close(permuted, original[:, permutation])


def test_shared_actor_masks_inactive_zero_tokens() -> None:
    from trade_rl.rl.policies import SharedPerAssetActionHead

    head = SharedPerAssetActionHead(
        n_symbols=2,
        token_dim=3,
        context_dim=7,
        hidden_dims=(5,),
    ).eval()
    contexts = torch.randn(1, 2, 7)
    contexts[:, 1, :3] = 0.0

    with torch.no_grad():
        actions = head(contexts.reshape(1, -1))

    assert actions.shape == (1, 2)
    assert actions[0, 1].item() == 0.0


def test_shared_sequence_policy_installs_shared_actor_and_portfolio_critic() -> None:
    import numpy as np
    from gymnasium import spaces

    from trade_rl.rl.policies import (
        SequenceAssetFeatureExtractor,
        SharedPerAssetActionHead,
        SharedPerAssetActorCriticPolicy,
    )

    n_symbols = 2
    timeframes = ("15m", "1h", "4h", "1d")
    feature_counts = {timeframe: 2 for timeframe in timeframes}
    window_lengths = {timeframe: 3 for timeframe in timeframes}
    observation_spaces: dict[str, spaces.Space] = {
        "current_snapshot": spaces.Box(
            -1.0, 1.0, shape=(n_symbols, 8), dtype=np.float32
        ),
        "asset_state": spaces.Box(
            -1.0, 1.0, shape=(n_symbols, 4), dtype=np.float32
        ),
        "global_state": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
        "active": spaces.Box(0.0, 1.0, shape=(n_symbols,), dtype=np.float32),
    }
    for timeframe in timeframes:
        shape = (n_symbols, 3, 2)
        observation_spaces[f"sequence_{timeframe}_values"] = spaces.Box(
            -1.0, 1.0, shape=shape, dtype=np.float16
        )
        observation_spaces[f"sequence_{timeframe}_available"] = spaces.Box(
            0, 1, shape=shape, dtype=np.uint8
        )
        observation_spaces[f"sequence_{timeframe}_staleness"] = spaces.Box(
            0.0, 10.0, shape=shape, dtype=np.float16
        )
    observation_space = spaces.Dict(observation_spaces)
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
    )

    assert isinstance(policy.action_net, SharedPerAssetActionHead)
    assert policy.mlp_extractor.latent_dim_vf == 13
    assert policy.action_net.n_symbols == n_symbols
    constructor = policy._get_constructor_parameters()
    assert constructor["shared_actor_n_symbols"] == n_symbols
    assert constructor["shared_actor_net_arch"] == (11,)


''',
    )


def add_implementation() -> None:
    replace_once(
        "trade_rl/rl/policies.py",
        "from __future__ import annotations\n\nimport torch\n",
        "from __future__ import annotations\n\nfrom functools import partial\nfrom typing import Any\n\nimport torch\n",
    )
    replace_once(
        "trade_rl/rl/policies.py",
        "from stable_baselines3.common.torch_layers import BaseFeaturesExtractor\n",
        "from stable_baselines3.common.policies import MultiInputActorCriticPolicy\nfrom stable_baselines3.common.torch_layers import BaseFeaturesExtractor\n",
    )
    append_before(
        "trade_rl/rl/policies.py",
        '__all__ = ["AssetSetFeatureExtractor", "SequenceAssetFeatureExtractor"]',
        '''class SharedAssetActorCriticExtractor(nn.Module):
    """Split structured features into shared per-asset actor and portfolio critic latents."""

    def __init__(
        self,
        *,
        features_dim: int,
        n_symbols: int,
        token_dim: int,
        global_dim: int,
        critic_hidden_dims: tuple[int, ...],
        activation_fn: type[nn.Module],
    ) -> None:
        super().__init__()
        if n_symbols <= 0 or token_dim <= 0 or global_dim <= 0:
            raise ValueError("shared actor dimensions must be positive")
        expected = n_symbols * token_dim + token_dim + global_dim
        if features_dim != expected:
            raise ValueError("feature extractor output does not match shared actor layout")
        if not critic_hidden_dims or any(width <= 0 for width in critic_hidden_dims):
            raise ValueError("critic_hidden_dims must contain positive widths")
        self.n_symbols = n_symbols
        self.token_dim = token_dim
        self.global_dim = global_dim
        self.actor_context_dim = 2 * token_dim + global_dim
        self.latent_dim_pi = n_symbols * self.actor_context_dim
        layers: list[nn.Module] = []
        width = token_dim + global_dim
        for hidden in critic_hidden_dims:
            layers.extend((nn.Linear(width, hidden), activation_fn()))
            width = hidden
        self.critic_net = nn.Sequential(*layers)
        self.latent_dim_vf = width

    def _parts(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        asset_width = self.n_symbols * self.token_dim
        tokens = features[:, :asset_width].reshape(
            -1, self.n_symbols, self.token_dim
        )
        pooled = features[:, asset_width : asset_width + self.token_dim]
        globals_ = features[:, asset_width + self.token_dim :]
        return tokens, pooled, globals_

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        tokens, pooled, globals_ = self._parts(features)
        pooled_per_asset = pooled[:, None, :].expand(-1, self.n_symbols, -1)
        global_per_asset = globals_[:, None, :].expand(-1, self.n_symbols, -1)
        contexts = torch.cat((tokens, pooled_per_asset, global_per_asset), dim=-1)
        return contexts.reshape(features.shape[0], -1)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        _, pooled, globals_ = self._parts(features)
        return self.critic_net(torch.cat((pooled, globals_), dim=-1))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class SharedPerAssetActionHead(nn.Module):
    """Apply one actor MLP to every contextual asset token."""

    def __init__(
        self,
        *,
        n_symbols: int,
        token_dim: int,
        context_dim: int,
        hidden_dims: tuple[int, ...],
        activation_fn: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        if n_symbols <= 0 or token_dim <= 0 or context_dim < token_dim:
            raise ValueError("shared action-head dimensions are invalid")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        self.n_symbols = n_symbols
        self.token_dim = token_dim
        self.context_dim = context_dim
        layers: list[nn.Module] = []
        width = context_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(width, hidden), activation_fn()))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.shared_head = nn.Sequential(*layers)

    def forward(self, actor_latent: torch.Tensor) -> torch.Tensor:
        contexts = actor_latent.reshape(-1, self.n_symbols, self.context_dim)
        token = contexts[:, :, : self.token_dim]
        active = token.abs().sum(dim=-1) > 0.0
        means = self.shared_head(contexts).squeeze(-1)
        return means * active.to(dtype=means.dtype)


class SharedPerAssetActorCriticPolicy(MultiInputActorCriticPolicy):
    """SB3 PPO policy with a truly shared actor head and portfolio-level critic."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Any,
        *,
        shared_actor_n_symbols: int,
        shared_actor_d_model: int,
        shared_actor_global_dim: int = 128,
        shared_actor_net_arch: tuple[int, ...] = (128, 128),
        **kwargs: Any,
    ) -> None:
        if kwargs.get("use_sde", False):
            raise ValueError("shared per-asset actor does not support gSDE")
        if action_space.shape != (shared_actor_n_symbols,):
            raise ValueError("shared actor action space must contain one action per asset")
        self.shared_actor_n_symbols = shared_actor_n_symbols
        self.shared_actor_d_model = shared_actor_d_model
        self.shared_actor_global_dim = shared_actor_global_dim
        self.shared_actor_net_arch = tuple(shared_actor_net_arch)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _critic_architecture(self) -> tuple[int, ...]:
        architecture = self.net_arch
        if isinstance(architecture, dict):
            raw = architecture.get("vf", [])
        else:
            raw = architecture
        return tuple(int(width) for width in raw)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SharedAssetActorCriticExtractor(
            features_dim=self.features_dim,
            n_symbols=self.shared_actor_n_symbols,
            token_dim=self.shared_actor_d_model,
            global_dim=self.shared_actor_global_dim,
            critic_hidden_dims=self._critic_architecture(),
            activation_fn=self.activation_fn,
        ).to(self.device)

    def _build(self, lr_schedule: Any) -> None:
        super()._build(lr_schedule)
        context_dim = 2 * self.shared_actor_d_model + self.shared_actor_global_dim
        self.action_net = SharedPerAssetActionHead(
            n_symbols=self.shared_actor_n_symbols,
            token_dim=self.shared_actor_d_model,
            context_dim=context_dim,
            hidden_dims=self.shared_actor_net_arch,
            activation_fn=self.activation_fn,
        ).to(self.device)
        if self.ortho_init:
            self.action_net.apply(partial(self.init_weights, gain=0.01))
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            shared_actor_n_symbols=self.shared_actor_n_symbols,
            shared_actor_d_model=self.shared_actor_d_model,
            shared_actor_global_dim=self.shared_actor_global_dim,
            shared_actor_net_arch=self.shared_actor_net_arch,
        )
        return data


''',
    )
    replace_once(
        "trade_rl/rl/policies.py",
        '__all__ = ["AssetSetFeatureExtractor", "SequenceAssetFeatureExtractor"]',
        '''__all__ = [
    "AssetSetFeatureExtractor",
    "SequenceAssetFeatureExtractor",
    "SharedAssetActorCriticExtractor",
    "SharedPerAssetActionHead",
    "SharedPerAssetActorCriticPolicy",
]''',
    )
    replace_once(
        "trade_rl/integrations/sb3_training.py",
        "                from trade_rl.rl.policies import SequenceAssetFeatureExtractor\n",
        "                from trade_rl.rl.policies import (\n                    SequenceAssetFeatureExtractor,\n                    SharedPerAssetActorCriticPolicy,\n                )\n",
    )
    replace_once(
        "trade_rl/integrations/sb3_training.py",
        '''                    "features_extractor_kwargs": {
                        **sequence_metadata,
                        "d_model": config.sequence_d_model,
                        "attention_heads": config.sequence_attention_heads,
                        "attention_layers": config.sequence_attention_layers,
                        "dropout": config.sequence_dropout,
                    },
                }
''',
        '''                    "features_extractor_kwargs": {
                        **sequence_metadata,
                        "d_model": config.sequence_d_model,
                        "attention_heads": config.sequence_attention_heads,
                        "attention_layers": config.sequence_attention_layers,
                        "dropout": config.sequence_dropout,
                    },
                    "shared_actor_n_symbols": int(sequence_metadata["n_symbols"]),
                    "shared_actor_d_model": config.sequence_d_model,
                    "shared_actor_global_dim": 128,
                    "shared_actor_net_arch": tuple(config.policy_net_arch),
                }
''',
    )
    replace_once(
        "trade_rl/integrations/sb3_training.py",
        "            common: dict[str, Any] = {\n",
        "            policy_identifier: Any = (\n                SharedPerAssetActorCriticPolicy\n                if config.sequence_encoder\n                else config.policy\n            )\n            common: dict[str, Any] = {\n",
    )
    replace_once(
        "trade_rl/integrations/sb3_training.py",
        "                    config.policy,\n                    environment,\n",
        "                    policy_identifier,\n                    environment,\n",
    )
    replace_once(
        "trade_rl/integrations/sb3_training.py",
        '''                        "d_model": config.sequence_d_model,
                        "attention_heads": config.sequence_attention_heads,
''',
        '''                        "d_model": config.sequence_d_model,
                        "actor_head": "shared_per_asset_v1",
                        "actor_parameter_sharing": "one_head_all_assets",
                        "actor_symbol_order": tuple(identity["action_names"]),
                        "attention_heads": config.sequence_attention_heads,
''',
    )
    replace_once(
        "trade_rl/integrations/sb3_training.py",
        '                        "policy": config.policy,\n',
        '                        "policy": (\n                            policy_identifier.__name__\n                            if isinstance(policy_identifier, type)\n                            else policy_identifier\n                        ),\n',
    )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: apply_task3_shared_actor.py tests|implementation")
    if sys.argv[1] == "tests":
        add_tests()
    else:
        add_implementation()
