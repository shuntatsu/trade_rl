"""Immutable identities for the maintained hierarchical sequence policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from trade_rl.artifacts.hashing import content_digest

if TYPE_CHECKING:
    from trade_rl.rl.sequence_policy import SequencePolicyArchitecture

_TIMEFRAMES = ("15m", "1h", "4h", "1d")
_ARCHITECTURE_SCHEMA = "hierarchical_sequence_policy_v4"
_ASSET_BINDING_SCHEMA = "sequence_asset_binding_v1"
_ASSET_IDENTITY_MODE = "identity_free_v1"


def _required_dilations(window_length: int) -> tuple[int, ...]:
    dilations: list[int] = []
    receptive_field = 1
    dilation = 1
    while receptive_field < window_length:
        dilations.append(dilation)
        receptive_field += 2 * dilation
        dilation *= 2
    return tuple(dilations)


def _string_tuple(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{field} must be a non-empty string tuple")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must be unique")
    return tuple(values)


@dataclass(frozen=True, slots=True)
class SequenceArchitectureIdentity:
    """Content-addressed model structure independent of concrete symbol names."""

    input_channels: tuple[int, ...]
    window_lengths: tuple[int, ...]
    latent_dims: tuple[int, ...]
    encoder_widths: tuple[tuple[int, ...], ...]
    dilations: tuple[tuple[int, ...], ...]
    asset_state_width: int
    snapshot_width: int
    n_symbols: int
    d_model: int
    timeframe_attention_heads: int
    timeframe_attention_layers: int
    timeframe_ffn_multiplier: int
    timeframe_gate_bias: float
    asset_attention_heads: int
    asset_attention_layers: int
    asset_ffn_multiplier: int
    asset_gate_bias: float
    dropout: float
    asset_identity_mode: str = _ASSET_IDENTITY_MODE
    timeframes: tuple[str, ...] = _TIMEFRAMES
    schema_version: str = _ARCHITECTURE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _ARCHITECTURE_SCHEMA:
            raise ValueError("unsupported sequence architecture identity schema")
        if self.asset_identity_mode != _ASSET_IDENTITY_MODE:
            raise ValueError("sequence architecture asset identity mode is invalid")
        if self.timeframes != _TIMEFRAMES:
            raise ValueError("sequence identity requires ordered 15m/1h/4h/1d clocks")
        width = len(self.timeframes)
        for field_name, values in (
            ("input_channels", self.input_channels),
            ("window_lengths", self.window_lengths),
            ("latent_dims", self.latent_dims),
            ("encoder_widths", self.encoder_widths),
            ("dilations", self.dilations),
        ):
            if len(values) != width:
                raise ValueError(f"{field_name} must match maintained timeframes")
        if isinstance(self.n_symbols, bool) or self.n_symbols <= 0:
            raise ValueError("n_symbols must be a positive integer")
        for window, dilations in zip(self.window_lengths, self.dilations, strict=True):
            if 1 + 2 * sum(dilations) < window:
                raise ValueError("dilation schedule does not cover declared window")

    def digest_payload(self) -> dict[str, object]:
        return {
            "asset_attention_heads": self.asset_attention_heads,
            "asset_attention_layers": self.asset_attention_layers,
            "asset_ffn_multiplier": self.asset_ffn_multiplier,
            "asset_gate_bias": self.asset_gate_bias,
            "asset_identity_mode": self.asset_identity_mode,
            "asset_state_width": self.asset_state_width,
            "d_model": self.d_model,
            "dilations": self.dilations,
            "dropout": self.dropout,
            "encoder_widths": self.encoder_widths,
            "input_channels": self.input_channels,
            "latent_dims": self.latent_dims,
            "n_symbols": self.n_symbols,
            "schema_version": self.schema_version,
            "snapshot_width": self.snapshot_width,
            "timeframe_attention_heads": self.timeframe_attention_heads,
            "timeframe_attention_layers": self.timeframe_attention_layers,
            "timeframe_ffn_multiplier": self.timeframe_ffn_multiplier,
            "timeframe_gate_bias": self.timeframe_gate_bias,
            "timeframes": self.timeframes,
            "window_lengths": self.window_lengths,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.digest_payload())


@dataclass(frozen=True, slots=True)
class SequenceAssetBindingIdentity:
    """Exact runtime mapping between asset slots, symbols, and actions."""

    n_symbols: int
    symbols: tuple[str, ...]
    action_names: tuple[str, ...]
    schema_version: str = _ASSET_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _ASSET_BINDING_SCHEMA:
            raise ValueError("unsupported sequence asset binding schema")
        if isinstance(self.n_symbols, bool) or self.n_symbols <= 0:
            raise ValueError("asset binding n_symbols must be a positive integer")
        symbols = _string_tuple(self.symbols, field="asset binding symbols")
        action_names = _string_tuple(
            self.action_names, field="asset binding action names"
        )
        if len(symbols) != self.n_symbols or len(action_names) != self.n_symbols:
            raise ValueError(
                "asset binding symbol and action counts must match n_symbols"
            )
        expected = tuple(f"target_weight:{symbol}" for symbol in symbols)
        if action_names != expected:
            raise ValueError(
                "asset binding requires ordered target-weight action names"
            )
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "action_names", action_names)

    def digest_payload(self) -> dict[str, object]:
        return {
            "action_names": self.action_names,
            "n_symbols": self.n_symbols,
            "schema_version": self.schema_version,
            "symbols": self.symbols,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.digest_payload())


def sequence_architecture_identity(
    architecture: SequencePolicyArchitecture,
) -> SequenceArchitectureIdentity:
    """Build an immutable symbol-agnostic identity from one architecture."""

    if tuple(architecture.input_channels) != _TIMEFRAMES:
        raise ValueError("architecture must use ordered 15m/1h/4h/1d clocks")
    widths = architecture.encoder_widths
    if widths is None:
        raise ValueError("architecture encoder widths were not resolved")
    return SequenceArchitectureIdentity(
        input_channels=tuple(architecture.input_channels[item] for item in _TIMEFRAMES),
        window_lengths=tuple(architecture.window_lengths[item] for item in _TIMEFRAMES),
        latent_dims=tuple(architecture.latent_dims[item] for item in _TIMEFRAMES),
        encoder_widths=tuple(tuple(widths[item]) for item in _TIMEFRAMES),
        dilations=tuple(
            _required_dilations(architecture.window_lengths[item])
            for item in _TIMEFRAMES
        ),
        asset_state_width=architecture.asset_state_width,
        snapshot_width=architecture.snapshot_width,
        n_symbols=architecture.n_symbols,
        d_model=architecture.d_model,
        timeframe_attention_heads=architecture.timeframe_attention_heads,
        timeframe_attention_layers=architecture.timeframe_attention_layers,
        timeframe_ffn_multiplier=architecture.timeframe_ffn_multiplier,
        timeframe_gate_bias=architecture.timeframe_gate_bias,
        asset_attention_heads=architecture.asset_attention_heads,
        asset_attention_layers=architecture.asset_attention_layers,
        asset_ffn_multiplier=architecture.asset_ffn_multiplier,
        asset_gate_bias=architecture.asset_gate_bias,
        dropout=architecture.dropout,
    )


def sequence_asset_binding_identity(
    *,
    n_symbols: int,
    symbols: tuple[str, ...],
    action_names: tuple[str, ...],
) -> SequenceAssetBindingIdentity:
    """Build the exact symbol/action binding kept outside model structure."""

    return SequenceAssetBindingIdentity(
        n_symbols=n_symbols,
        symbols=symbols,
        action_names=action_names,
    )


__all__ = [
    "SequenceArchitectureIdentity",
    "SequenceAssetBindingIdentity",
    "sequence_architecture_identity",
    "sequence_asset_binding_identity",
]
