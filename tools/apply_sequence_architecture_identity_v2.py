"""Split timeframe and asset attention in architecture construction and identity."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "trade_rl/rl/sequence_architecture.py"

IDENTITY_SOURCE = '''"""Immutable identity for the maintained hierarchical sequence policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from trade_rl.artifacts.hashing import content_digest

if TYPE_CHECKING:
    from trade_rl.rl.sequence_policy import SequencePolicyArchitecture

_TIMEFRAMES = ("15m", "1h", "4h", "1d")
_SCHEMA = "hierarchical_sequence_policy_v2"


def _required_dilations(window_length: int) -> tuple[int, ...]:
    dilations: list[int] = []
    receptive_field = 1
    dilation = 1
    while receptive_field < window_length:
        dilations.append(dilation)
        receptive_field += 2 * dilation
        dilation *= 2
    return tuple(dilations)


@dataclass(frozen=True, slots=True)
class SequenceArchitectureIdentity:
    """Content-addressed model semantics required for exact artifact loading."""

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
    symbols: tuple[str, ...]
    action_names: tuple[str, ...]
    timeframes: tuple[str, ...] = _TIMEFRAMES
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA:
            raise ValueError("unsupported sequence architecture identity schema")
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
        if len(self.symbols) != self.n_symbols or len(self.action_names) != self.n_symbols:
            raise ValueError("symbol and action counts must match n_symbols")
        if not self.symbols or any(not value for value in self.symbols):
            raise ValueError("symbols must be non-empty")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must be unique")
        if any(not value for value in self.action_names):
            raise ValueError("action names must be non-empty")
        for window, dilations in zip(self.window_lengths, self.dilations, strict=True):
            if 1 + 2 * sum(dilations) < window:
                raise ValueError("dilation schedule does not cover declared window")

    def digest_payload(self) -> dict[str, object]:
        return {
            "action_names": self.action_names,
            "asset_attention_heads": self.asset_attention_heads,
            "asset_attention_layers": self.asset_attention_layers,
            "asset_ffn_multiplier": self.asset_ffn_multiplier,
            "asset_gate_bias": self.asset_gate_bias,
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
            "symbols": self.symbols,
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


def sequence_architecture_identity(
    architecture: SequencePolicyArchitecture,
    *,
    symbols: tuple[str, ...],
    action_names: tuple[str, ...],
) -> SequenceArchitectureIdentity:
    """Build an exact immutable identity from one validated architecture."""

    if tuple(architecture.input_channels) != _TIMEFRAMES:
        raise ValueError("architecture must use ordered 15m/1h/4h/1d clocks")
    widths = architecture.encoder_widths
    if widths is None:
        raise ValueError("architecture encoder widths were not resolved")
    return SequenceArchitectureIdentity(
        input_channels=tuple(architecture.input_channels[item] for item in _TIMEFRAMES),
        window_lengths=tuple(
            architecture.window_lengths[item] for item in _TIMEFRAMES
        ),
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
        symbols=tuple(symbols),
        action_names=tuple(action_names),
    )


__all__ = ["SequenceArchitectureIdentity", "sequence_architecture_identity"]
'''


def _offsets(text: str) -> list[int]:
    result = [0]
    for index, character in enumerate(text):
        if character == "\n":
            result.append(index + 1)
    return result


def _absolute(offsets: list[int], line: int, column: int) -> int:
    return offsets[line - 1] + column


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _migrate_constructors(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "SequencePolicyArchitecture" not in text:
        return
    tree = ast.parse(text)
    offsets = _offsets(text)
    edits: list[tuple[int, int, str]] = []
    mapping = {
        "attention_heads": (
            "timeframe_attention_heads",
            "asset_attention_heads",
        ),
        "attention_layers": (
            "timeframe_attention_layers",
            "asset_attention_layers",
        ),
        "attention_ffn_multiplier": (
            "timeframe_ffn_multiplier",
            "asset_ffn_multiplier",
        ),
        "attention_gate_bias": (
            "timeframe_gate_bias",
            "asset_gate_bias",
        ),
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node.func) != "SequencePolicyArchitecture":
            continue
        for keyword in node.keywords:
            if keyword.arg not in mapping:
                continue
            start = _absolute(offsets, keyword.lineno, keyword.col_offset)
            end = _absolute(offsets, keyword.end_lineno, keyword.end_col_offset)
            value_start = _absolute(
                offsets, keyword.value.lineno, keyword.value.col_offset
            )
            value_end = _absolute(
                offsets, keyword.value.end_lineno, keyword.value.end_col_offset
            )
            value = text[value_start:value_end]
            first, second = mapping[keyword.arg]
            indent = " " * keyword.col_offset
            replacement = f"{first}={value},\n{indent}{second}={value}"
            edits.append((start, end, replacement))
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    text = text.replace(".attention_heads", ".timeframe_attention_heads")
    text = text.replace(".attention_layers", ".timeframe_attention_layers")
    text = text.replace(".attention_ffn_multiplier", ".timeframe_ffn_multiplier")
    text = text.replace(".attention_gate_bias", ".timeframe_gate_bias")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    TARGET.write_text(IDENTITY_SOURCE, encoding="utf-8")
    for root in (ROOT / "tests", ROOT / "trade_rl"):
        for path in root.rglob("*.py"):
            if path == TARGET:
                continue
            _migrate_constructors(path)


if __name__ == "__main__":
    main()
