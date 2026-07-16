"""Permutation-invariant policy feature extractors for asset-set observations."""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from gymnasium import spaces
from stable_baselines3.common.distributions import (
    SquashedDiagGaussianDistribution,
    TanhBijector,
)
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class AssetSetFeatureExtractor(BaseFeaturesExtractor):
    """Encode each asset with shared weights and attention-pool active assets."""

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        n_symbols: int,
        per_symbol_width: int,
        global_width: int,
        active_column: int,
        asset_embedding_dim: int = 64,
        global_embedding_dim: int = 64,
    ) -> None:
        if n_symbols <= 0 or per_symbol_width <= 0 or global_width <= 0:
            raise ValueError("asset-set dimensions must be positive")
        if not 0 <= active_column < per_symbol_width:
            raise ValueError("active_column is outside the per-symbol observation")
        expected = n_symbols * per_symbol_width + global_width
        if observation_space.shape != (expected,):
            raise ValueError("observation space does not match asset-set dimensions")
        if asset_embedding_dim <= 0 or global_embedding_dim <= 0:
            raise ValueError("embedding dimensions must be positive")
        features_dim = asset_embedding_dim + global_embedding_dim
        super().__init__(observation_space, features_dim=features_dim)
        self.n_symbols = n_symbols
        self.per_symbol_width = per_symbol_width
        self.global_width = global_width
        self.active_column = active_column
        self.asset_encoder = nn.Sequential(
            nn.Linear(per_symbol_width, asset_embedding_dim),
            nn.LayerNorm(asset_embedding_dim),
            nn.Tanh(),
            nn.Linear(asset_embedding_dim, asset_embedding_dim),
            nn.Tanh(),
        )
        self.attention_score = nn.Linear(asset_embedding_dim, 1)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_width, global_embedding_dim),
            nn.LayerNorm(global_embedding_dim),
            nn.Tanh(),
            nn.Linear(global_embedding_dim, global_embedding_dim),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        asset_width = self.n_symbols * self.per_symbol_width
        assets = observations[:, :asset_width].reshape(
            -1,
            self.n_symbols,
            self.per_symbol_width,
        )
        globals_ = observations[:, asset_width:]
        encoded_assets = self.asset_encoder(assets)
        active = assets[:, :, self.active_column] > 0.5
        scores = self.attention_score(encoded_assets).squeeze(-1)
        scores = scores.masked_fill(~active, -1e9)
        has_active = active.any(dim=1, keepdim=True)
        safe_scores = torch.where(has_active, scores, torch.zeros_like(scores))
        weights = torch.softmax(safe_scores, dim=1)
        weights = torch.where(active, weights, torch.zeros_like(weights))
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        weights = weights / denominator
        pooled_assets = (encoded_assets * weights.unsqueeze(-1)).sum(dim=1)
        pooled_assets = torch.where(
            has_active,
            pooled_assets,
            torch.zeros_like(pooled_assets),
        )
        encoded_globals = self.global_encoder(globals_)
        return torch.cat((pooled_assets, encoded_globals), dim=1)


class SequenceAssetFeatureExtractor(BaseFeaturesExtractor):
    """Encode native-clock sequences and fuse current cross-asset state."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        feature_counts: dict[str, int],
        window_lengths: dict[str, int],
        snapshot_width: int,
        asset_state_width: int,
        global_width: int,
        n_symbols: int,
        d_model: int = 320,
        attention_heads: int = 8,
        attention_layers: int = 2,
        dropout: float = 0.05,
    ) -> None:
        from trade_rl.rl.sequence_policy import (
            MultiTimeframeAssetEncoder,
            SequencePolicyArchitecture,
        )

        timeframes = ("15m", "1h", "4h", "1d")
        if tuple(feature_counts) != timeframes or tuple(window_lengths) != timeframes:
            raise ValueError("sequence metadata must use ordered maintained clocks")
        expected_shapes: dict[str, tuple[int, ...]] = {
            "current_snapshot": (n_symbols, snapshot_width),
            "asset_state": (n_symbols, asset_state_width),
            "global_state": (global_width,),
            "active": (n_symbols,),
        }
        for timeframe in timeframes:
            sequence_shape = (
                n_symbols,
                window_lengths[timeframe],
                feature_counts[timeframe],
            )
            for suffix in ("values", "available", "staleness"):
                expected_shapes[f"sequence_{timeframe}_{suffix}"] = sequence_shape
        for key, expected_shape in expected_shapes.items():
            if key not in observation_space.spaces:
                raise ValueError(f"sequence observation space is missing {key}")
            if observation_space.spaces[key].shape != expected_shape:
                raise ValueError(f"sequence observation shape mismatch for {key}")
        super().__init__(
            observation_space,
            features_dim=n_symbols * d_model + d_model + 128 + n_symbols,
        )
        self.timeframes = timeframes
        architecture = SequencePolicyArchitecture(
            input_channels={
                timeframe: 3 * feature_counts[timeframe] for timeframe in timeframes
            },
            window_lengths=window_lengths,
            latent_dims={"15m": 192, "1h": 192, "4h": 160, "1d": 128},
            asset_state_width=asset_state_width,
            snapshot_width=snapshot_width,
            n_symbols=n_symbols,
            d_model=d_model,
            attention_heads=attention_heads,
            attention_layers=attention_layers,
            dropout=dropout,
        )
        self.asset_encoder = MultiTimeframeAssetEncoder(architecture)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_width, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        sequences: dict[str, torch.Tensor] = {}
        available: dict[str, torch.Tensor] = {}
        for timeframe in self.timeframes:
            values = observations[f"sequence_{timeframe}_values"].float()
            availability = observations[f"sequence_{timeframe}_available"].float()
            staleness = observations[f"sequence_{timeframe}_staleness"].float()
            sequences[timeframe] = torch.cat(
                (values, availability, torch.log1p(staleness.clamp_min(0.0))),
                dim=-1,
            )
            available[timeframe] = availability > 0.5
        asset_tokens, pooled_assets = self.asset_encoder(
            sequences=sequences,
            available=available,
            snapshot=observations["current_snapshot"].float(),
            asset_state=observations["asset_state"].float(),
            active=observations["active"].float(),
        )
        globals_ = self.global_encoder(observations["global_state"].float())
        ordered_assets = asset_tokens.reshape(asset_tokens.shape[0], -1)
        active = observations["active"].float()
        return torch.cat((ordered_assets, pooled_assets, globals_, active), dim=-1)


class SharedAssetActorCriticExtractor(nn.Module):
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
        expected = n_symbols * token_dim + token_dim + global_dim + n_symbols
        if features_dim != expected:
            raise ValueError(
                "feature extractor output does not match shared actor layout"
            )
        if not critic_hidden_dims or any(width <= 0 for width in critic_hidden_dims):
            raise ValueError("critic_hidden_dims must contain positive widths")
        self.n_symbols = n_symbols
        self.token_dim = token_dim
        self.global_dim = global_dim
        self.actor_context_dim = 2 * token_dim + global_dim + 1
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        asset_width = self.n_symbols * self.token_dim
        pooled_start = asset_width
        global_start = pooled_start + self.token_dim
        active_start = global_start + self.global_dim
        tokens = features[:, :asset_width].reshape(-1, self.n_symbols, self.token_dim)
        pooled = features[:, pooled_start:global_start]
        globals_ = features[:, global_start:active_start]
        active = features[:, active_start:]
        return tokens, pooled, globals_, active

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        tokens, pooled, globals_, active = self._parts(features)
        pooled_per_asset = pooled[:, None, :].expand(-1, self.n_symbols, -1)
        global_per_asset = globals_[:, None, :].expand(-1, self.n_symbols, -1)
        contexts = torch.cat(
            (tokens, pooled_per_asset, global_per_asset, active.unsqueeze(-1)),
            dim=-1,
        )
        return contexts.reshape(features.shape[0], -1)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        _, pooled, globals_, _ = self._parts(features)
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

    def active_mask(self, actor_latent: torch.Tensor) -> torch.Tensor:
        contexts = actor_latent.reshape(-1, self.n_symbols, self.context_dim)
        return contexts[:, :, -1] > 0.5

    def forward(self, actor_latent: torch.Tensor) -> torch.Tensor:
        contexts = actor_latent.reshape(-1, self.n_symbols, self.context_dim)
        active = self.active_mask(actor_latent)
        means = self.shared_head(contexts).squeeze(-1)
        return means * active.to(dtype=means.dtype)


class MaskedSharedSquashedDiagGaussianDistribution(SquashedDiagGaussianDistribution):
    """One shared exploration scale with inactive dimensions excluded."""

    def __init__(self, action_dim: int) -> None:
        super().__init__(action_dim)
        self.active_mask: torch.Tensor | None = None

    def set_active_mask(self, active_mask: torch.Tensor) -> None:
        mask = active_mask.to(dtype=torch.bool)
        if mask.ndim != 2 or mask.shape[1] != self.action_dim:
            raise ValueError("active action mask does not match action dimensions")
        self.active_mask = mask

    def _masked(self, actions: torch.Tensor) -> torch.Tensor:
        if self.active_mask is None:
            raise RuntimeError("active action mask is not configured")
        return actions * self.active_mask.to(dtype=actions.dtype)

    def sample(self) -> torch.Tensor:
        return self._masked(super().sample())

    def mode(self) -> torch.Tensor:
        return self._masked(super().mode())

    def log_prob(
        self,
        actions: torch.Tensor,
        gaussian_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.active_mask is None:
            raise RuntimeError("active action mask is not configured")
        if gaussian_actions is None:
            gaussian_actions = TanhBijector.inverse(actions)
        distribution = self.distribution
        if distribution is None:
            raise RuntimeError("masked action distribution is not initialized")
        per_dimension = distribution.log_prob(gaussian_actions)
        per_dimension -= torch.log(1 - actions**2 + self.epsilon)
        return (per_dimension * self.active_mask.to(dtype=per_dimension.dtype)).sum(
            dim=1
        )


class SharedPerAssetActorCriticPolicy(MultiInputActorCriticPolicy):
    """SB3 PPO policy with bounded shared target-weight actions."""

    action_distribution_name = "masked_shared_squashed_diag_gaussian"

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
            raise ValueError(
                "shared actor action space must contain one action per asset"
            )
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
        self.mlp_extractor = SharedAssetActorCriticExtractor(  # type: ignore[assignment]
            features_dim=self.features_dim,
            n_symbols=self.shared_actor_n_symbols,
            token_dim=self.shared_actor_d_model,
            global_dim=self.shared_actor_global_dim,
            critic_hidden_dims=self._critic_architecture(),
            activation_fn=self.activation_fn,
        ).to(self.device)

    def _build(self, lr_schedule: Any) -> None:
        self.action_dist = MaskedSharedSquashedDiagGaussianDistribution(
            self.shared_actor_n_symbols
        )
        super()._build(lr_schedule)
        context_dim = 2 * self.shared_actor_d_model + self.shared_actor_global_dim + 1
        self.action_net = SharedPerAssetActionHead(
            n_symbols=self.shared_actor_n_symbols,
            token_dim=self.shared_actor_d_model,
            context_dim=context_dim,
            hidden_dims=self.shared_actor_net_arch,
            activation_fn=self.activation_fn,
        ).to(self.device)
        if self.ortho_init:
            self.action_net.apply(partial(self.init_weights, gain=0.01))
        self.log_std = nn.Parameter(
            torch.full((1,), float(self.log_std_init), device=self.device)
        )
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Any:
        if not isinstance(
            self.action_dist, MaskedSharedSquashedDiagGaussianDistribution
        ):
            raise RuntimeError("shared policy action distribution is invalid")
        self.action_dist.set_active_mask(self.action_net.active_mask(latent_pi))
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            shared_actor_n_symbols=self.shared_actor_n_symbols,
            shared_actor_d_model=self.shared_actor_d_model,
            shared_actor_global_dim=self.shared_actor_global_dim,
            shared_actor_net_arch=self.shared_actor_net_arch,
        )
        return data


__all__ = [
    "AssetSetFeatureExtractor",
    "MaskedSharedSquashedDiagGaussianDistribution",
    "SequenceAssetFeatureExtractor",
    "SharedAssetActorCriticExtractor",
    "SharedPerAssetActionHead",
    "SharedPerAssetActorCriticPolicy",
]
