"""SB3 feature adapter for the frozen Universal Trade RL U1 observation surface."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Final

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.export_context import graph_export_active
from trade_rl.rl.sequence_policy import (
    MultiTimeframeAssetEncoder,
    SequencePolicyArchitecture,
    sequence_encoder_widths,
)
from trade_rl.rl.universal_trade_observation import (
    UNIVERSAL_TRADE_POLICY_STATE_FIELDS,
)

UNIVERSAL_TRADE_U1_SEQUENCE_ADAPTER_SCHEMA: Final = (
    "universal_trade_u1_sequence_adapter_v1"
)
_TIMEFRAMES: Final = ("15m", "1h", "4h", "1d")
_LATENT_DIMS: Final = {"15m": 192, "1h": 192, "4h": 160, "1d": 128}
_ACTIVE_FIELD: Final = "asset_active"
_CURRENT_WEIGHT_FIELD: Final = "current_weight"
_GLOBAL_STATE_FIELDS: Final = (
    "current_drawdown",
    "current_gross_exposure",
    "current_net_exposure",
    "cash_weight",
    "risk_scale",
    "margin_utilization",
)
_ASSET_STATE_FIELDS: Final = tuple(
    field
    for field in UNIVERSAL_TRADE_POLICY_STATE_FIELDS
    if field
    not in {
        _ACTIVE_FIELD,
        _CURRENT_WEIGHT_FIELD,
        *_GLOBAL_STATE_FIELDS,
    }
)
_SEQUENCE_CHANNEL_COMPOSITION: Final = "values_available_log1p_staleness_v1"
_SNAPSHOT_SOURCE: Final = "latest_15m_values_available_log1p_staleness_v1"


def _sequence_encoder_autocast(reference: torch.Tensor) -> Any:
    if reference.device.type != "cuda" or not torch.cuda.is_bf16_supported():
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _expected_observation_keys() -> frozenset[str]:
    keys = {"policy_state"}
    for timeframe in _TIMEFRAMES:
        keys.update(
            {
                f"sequence_{timeframe}_values",
                f"sequence_{timeframe}_available",
                f"sequence_{timeframe}_staleness",
            }
        )
    return frozenset(keys)


def _box(
    observation_space: spaces.Dict,
    key: str,
) -> spaces.Box:
    value = observation_space.spaces.get(key)
    if not isinstance(value, spaces.Box):
        raise ValueError(f"U1 sequence adapter requires Box observation {key}")
    return value


def _adapter_contract_payload(
    *,
    feature_counts: dict[str, int],
    window_lengths: dict[str, int],
) -> dict[str, object]:
    return {
        "schema_version": UNIVERSAL_TRADE_U1_SEQUENCE_ADAPTER_SCHEMA,
        "timeframes": _TIMEFRAMES,
        "feature_counts": feature_counts,
        "window_lengths": window_lengths,
        "n_symbols": 1,
        "policy_state_fields": UNIVERSAL_TRADE_POLICY_STATE_FIELDS,
        "asset_state_fields": _ASSET_STATE_FIELDS,
        "global_state_fields": _GLOBAL_STATE_FIELDS,
        "active_field": _ACTIVE_FIELD,
        "current_weight_field": _CURRENT_WEIGHT_FIELD,
        "sequence_channel_composition": _SEQUENCE_CHANNEL_COMPOSITION,
        "snapshot_source": _SNAPSHOT_SOURCE,
    }


def universal_trade_u1_sequence_adapter_metadata(
    observation_space: spaces.Space[Any],
) -> dict[str, object]:
    """Validate the frozen U1 Dict surface and return deterministic adapter metadata."""

    if not isinstance(observation_space, spaces.Dict):
        raise ValueError("U1 sequence adapter requires a Dict observation space")
    if frozenset(observation_space.spaces) != _expected_observation_keys():
        raise ValueError("U1 sequence adapter observation field closure mismatch")

    feature_counts: dict[str, int] = {}
    window_lengths: dict[str, int] = {}
    for timeframe in _TIMEFRAMES:
        values = _box(observation_space, f"sequence_{timeframe}_values")
        available = _box(observation_space, f"sequence_{timeframe}_available")
        staleness = _box(observation_space, f"sequence_{timeframe}_staleness")
        shape = values.shape
        if len(shape) != 3 or shape[0] != 1 or min(shape[1:]) <= 0:
            raise ValueError("U1 sequence adapter requires [1,time,feature] sequences")
        if available.shape != shape or staleness.shape != shape:
            raise ValueError("U1 sequence adapter sequence planes must share shape")
        if np.dtype(values.dtype) != np.dtype(np.float32):
            raise ValueError("U1 sequence values must remain float32")
        if np.dtype(available.dtype) != np.dtype(np.uint8):
            raise ValueError("U1 sequence availability must remain uint8")
        if np.dtype(staleness.dtype) != np.dtype(np.float32):
            raise ValueError("U1 sequence staleness must remain float32")
        window_lengths[timeframe] = int(shape[1])
        feature_counts[timeframe] = int(shape[2])

    policy_state = _box(observation_space, "policy_state")
    if policy_state.shape != (len(UNIVERSAL_TRADE_POLICY_STATE_FIELDS),):
        raise ValueError("U1 policy-state shape does not match the frozen field layout")
    if np.dtype(policy_state.dtype) != np.dtype(np.float32):
        raise ValueError("U1 policy-state dtype must remain float32")

    payload = _adapter_contract_payload(
        feature_counts=feature_counts,
        window_lengths=window_lengths,
    )
    return {
        **payload,
        "adapter_contract_digest": content_digest(payload),
    }


class UniversalTradeU1SequenceFeatureExtractor(BaseFeaturesExtractor):
    """Encode only U1 sequences/policy-state while preserving shared actor layout."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        feature_counts: dict[str, int],
        window_lengths: dict[str, int],
        n_symbols: int,
        policy_state_fields: tuple[str, ...],
        asset_state_fields: tuple[str, ...],
        global_state_fields: tuple[str, ...],
        active_field: str,
        current_weight_field: str,
        sequence_channel_composition: str,
        snapshot_source: str,
        schema_version: str,
        adapter_contract_digest: str,
        sequence_tcn_capacity: str = "standard",
        d_model: int = 320,
        timeframe_attention_heads: int = 8,
        timeframe_attention_layers: int = 2,
        timeframe_ffn_multiplier: int = 3,
        timeframe_gate_bias: float = -2.0,
        asset_attention_heads: int = 8,
        asset_attention_layers: int = 2,
        asset_ffn_multiplier: int = 3,
        asset_gate_bias: float = -2.0,
        dropout: float = 0.05,
    ) -> None:
        canonical = universal_trade_u1_sequence_adapter_metadata(observation_space)
        supplied = {
            "schema_version": schema_version,
            "timeframes": _TIMEFRAMES,
            "feature_counts": dict(feature_counts),
            "window_lengths": dict(window_lengths),
            "n_symbols": n_symbols,
            "policy_state_fields": tuple(policy_state_fields),
            "asset_state_fields": tuple(asset_state_fields),
            "global_state_fields": tuple(global_state_fields),
            "active_field": active_field,
            "current_weight_field": current_weight_field,
            "sequence_channel_composition": sequence_channel_composition,
            "snapshot_source": snapshot_source,
            "adapter_contract_digest": adapter_contract_digest,
        }
        if supplied != canonical:
            raise ValueError("U1 sequence adapter metadata does not match observation space")

        field_indices = {
            field: index for index, field in enumerate(UNIVERSAL_TRADE_POLICY_STATE_FIELDS)
        }
        asset_indices = tuple(field_indices[field] for field in _ASSET_STATE_FIELDS)
        global_indices = tuple(field_indices[field] for field in _GLOBAL_STATE_FIELDS)
        active_index = field_indices[_ACTIVE_FIELD]
        current_weight_index = field_indices[_CURRENT_WEIGHT_FIELD]
        snapshot_width = 3 * feature_counts["15m"]
        architecture = SequencePolicyArchitecture(
            input_channels={
                timeframe: 3 * feature_counts[timeframe] for timeframe in _TIMEFRAMES
            },
            window_lengths=dict(window_lengths),
            latent_dims=dict(_LATENT_DIMS),
            asset_state_width=len(asset_indices),
            snapshot_width=snapshot_width,
            n_symbols=1,
            d_model=d_model,
            timeframe_attention_heads=timeframe_attention_heads,
            timeframe_attention_layers=timeframe_attention_layers,
            timeframe_ffn_multiplier=timeframe_ffn_multiplier,
            timeframe_gate_bias=timeframe_gate_bias,
            asset_attention_heads=asset_attention_heads,
            asset_attention_layers=asset_attention_layers,
            asset_ffn_multiplier=asset_ffn_multiplier,
            asset_gate_bias=asset_gate_bias,
            dropout=dropout,
            encoder_widths=sequence_encoder_widths(sequence_tcn_capacity),
        )
        features_dim = 2 * d_model + 128 + 2
        super().__init__(observation_space, features_dim=features_dim)

        self.timeframes = _TIMEFRAMES
        self.asset_encoder = MultiTimeframeAssetEncoder(architecture)
        self.global_encoder = nn.Sequential(
            nn.Linear(len(global_indices), 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.register_buffer(
            "asset_state_indices",
            torch.tensor(asset_indices, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "global_state_indices",
            torch.tensor(global_indices, dtype=torch.long),
            persistent=True,
        )
        self.active_index = active_index
        self.current_weight_index = current_weight_index
        self.adapter_contract_digest = adapter_contract_digest

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        policy_state = observations["policy_state"].float()
        if not graph_export_active() and (
            policy_state.ndim != 2
            or policy_state.shape[1] != len(UNIVERSAL_TRADE_POLICY_STATE_FIELDS)
        ):
            raise ValueError("U1 policy-state tensor does not match frozen layout")

        sequences: dict[str, torch.Tensor] = {}
        available: dict[str, torch.Tensor] = {}
        staleness: dict[str, torch.Tensor] = {}
        values_by_timeframe: dict[str, torch.Tensor] = {}
        for timeframe in self.timeframes:
            values = observations[f"sequence_{timeframe}_values"].float()
            availability = observations[f"sequence_{timeframe}_available"].float()
            stale = observations[f"sequence_{timeframe}_staleness"].float()
            values_by_timeframe[timeframe] = values
            sequences[timeframe] = torch.cat(
                (values, availability, torch.log1p(stale.clamp_min(0.0))),
                dim=-1,
            )
            available[timeframe] = availability
            staleness[timeframe] = stale

        latest_values = values_by_timeframe["15m"][:, :, -1, :]
        latest_available = available["15m"][:, :, -1, :]
        latest_staleness = staleness["15m"][:, :, -1, :]
        snapshot = torch.cat(
            (
                latest_values,
                latest_available,
                torch.log1p(latest_staleness.clamp_min(0.0)),
            ),
            dim=-1,
        )
        asset_state = policy_state.index_select(1, self.asset_state_indices).unsqueeze(1)
        active = policy_state[:, self.active_index].reshape(-1, 1)
        current_weights = policy_state[:, self.current_weight_index].reshape(-1, 1)
        global_state = policy_state.index_select(1, self.global_state_indices)

        with _sequence_encoder_autocast(policy_state):
            asset_tokens, pooled_assets = self.asset_encoder(
                sequences=sequences,
                available=available,
                staleness=staleness,
                snapshot=snapshot,
                asset_state=asset_state,
                active=active,
            )
            encoded_globals = self.global_encoder(global_state)
            ordered_assets = asset_tokens.reshape(asset_tokens.shape[0], -1)
            encoded = torch.cat(
                (
                    ordered_assets,
                    pooled_assets,
                    encoded_globals,
                    active,
                    current_weights,
                ),
                dim=-1,
            )
        return encoded.float()


__all__ = [
    "UNIVERSAL_TRADE_U1_SEQUENCE_ADAPTER_SCHEMA",
    "UniversalTradeU1SequenceFeatureExtractor",
    "universal_trade_u1_sequence_adapter_metadata",
]
