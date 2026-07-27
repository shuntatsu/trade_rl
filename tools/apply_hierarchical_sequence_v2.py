"""Deterministically migrate the maintained policy/configuration contract to v2.

This tool is intentionally branch-local.  It edits the checked-out repository, runs
before verification, and is idempotent so a failed CI attempt can be reproduced.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMEFRAMES = ("15m", "1h", "4h", "1d")
SEQUENCE_ENCODER = "hierarchical_sequence_v2"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _write(relative: str, text: str) -> None:
    path = ROOT / relative
    path.write_text(text, encoding="utf-8")


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count == 0:
        if new in text:
            return text
        raise RuntimeError(f"migration anchor missing: {label}")
    if count != 1:
        raise RuntimeError(f"migration anchor is ambiguous ({count}): {label}")
    return text.replace(old, new, 1)


def _sub_once(text: str, pattern: str, replacement: str, *, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count == 0:
        if replacement.strip() in text:
            return text
        raise RuntimeError(f"migration pattern missing: {label}")
    return updated


def _migrate_training_config() -> None:
    path = "trade_rl/rl/training.py"
    text = _read(path)
    text = _sub_once(
        text,
        r"    sequence_encoder: bool = False\n"
        r"    sequence_capacity: str = \"standard\"\n"
        r"    sequence_d_model: int = 320\n"
        r"    sequence_attention_heads: int = 8\n"
        r"    sequence_attention_layers: int = 2\n"
        r"    sequence_dropout: float = 0\.05\n"
        r"    sequence_compile: bool = False\n"
        r"    sequence_compile_mode: str = \"reduce-overhead\"\n"
        r"    sequence_transfer_mode: str = \"synchronous\"\n"
        r"    max_policy_parameters: int = 12_000_000\n"
        r"    max_rollout_buffer_bytes: int = 805_306_368\n"
        r"    asset_set_encoder: bool = True\n"
        r"    asset_embedding_dim: int = 64\n"
        r"    global_embedding_dim: int = 64",
        "    observation_encoder: str = \"asset_set\"\n"
        "    sequence_tcn_capacity: str = \"standard\"\n"
        "    sequence_d_model: int = 320\n"
        "    sequence_timeframe_attention_heads: int = 8\n"
        "    sequence_timeframe_attention_layers: int = 2\n"
        "    sequence_timeframe_ffn_multiplier: int = 3\n"
        "    sequence_timeframe_gate_bias: float = -2.0\n"
        "    sequence_asset_attention_heads: int = 8\n"
        "    sequence_asset_attention_layers: int = 2\n"
        "    sequence_asset_ffn_multiplier: int = 3\n"
        "    sequence_asset_gate_bias: float = -2.0\n"
        "    sequence_dropout: float = 0.05\n"
        "    sequence_compile: bool = False\n"
        "    sequence_compile_mode: str = \"reduce-overhead\"\n"
        "    sequence_transfer_mode: str = \"synchronous\"\n"
        "    max_policy_parameters: int = 12_000_000\n"
        "    max_rollout_buffer_bytes: int = 805_306_368\n"
        "    asset_embedding_dim: int = 64\n"
        "    global_embedding_dim: int = 64",
        label="ResidualTrainingConfig encoder fields",
    )
    text = _sub_once(
        text,
        r"        if not isinstance\(self\.sequence_encoder, bool\):.*?"
        r"        for field_name, value in \(\n"
        r"            \(\"asset_embedding_dim\", self\.asset_embedding_dim\),\n"
        r"            \(\"global_embedding_dim\", self\.global_embedding_dim\),\n"
        r"        \):\n"
        r"            if isinstance\(value, bool\) or not isinstance\(value, int\) or value <= 0:\n"
        r"                raise ValueError\(f\"\{field_name\} must be a positive integer\"\)\n",
        "        encoder = self.observation_encoder.strip().lower()\n"
        "        allowed_encoders = {\n"
        "            \"flat_mlp\",\n"
        "            \"asset_set\",\n"
        "            \"hierarchical_sequence_v2\",\n"
        "        }\n"
        "        if encoder not in allowed_encoders:\n"
        "            raise ValueError(\n"
        "                \"observation_encoder must be flat_mlp, asset_set, or \"\n"
        "                \"hierarchical_sequence_v2\"\n"
        "            )\n"
        "        object.__setattr__(self, \"observation_encoder\", encoder)\n"
        "        sequence_active = encoder == \"hierarchical_sequence_v2\"\n"
        "        if not isinstance(self.sequence_compile, bool):\n"
        "            raise ValueError(\"sequence_compile must be a boolean\")\n"
        "        if self.sequence_compile_mode not in {\n"
        "            \"default\",\n"
        "            \"reduce-overhead\",\n"
        "            \"max-autotune\",\n"
        "        }:\n"
        "            raise ValueError(\n"
        "                \"sequence_compile_mode must be default, reduce-overhead, or \"\n"
        "                \"max-autotune\"\n"
        "            )\n"
        "        if self.sequence_transfer_mode not in {\n"
        "            \"synchronous\",\n"
        "            \"pinned_non_blocking\",\n"
        "        }:\n"
        "            raise ValueError(\n"
        "                \"sequence_transfer_mode must be synchronous or \"\n"
        "                \"pinned_non_blocking\"\n"
        "            )\n"
        "        if not self.sequence_compile and self.sequence_compile_mode != \"reduce-overhead\":\n"
        "            raise ValueError(\n"
        "                \"sequence_compile_mode is inactive when sequence_compile is false\"\n"
        "            )\n"
        "        if self.sequence_tcn_capacity not in {\"standard\", \"compact\"}:\n"
        "            raise ValueError(\n"
        "                \"sequence_tcn_capacity must be standard or compact\"\n"
        "            )\n"
        "        if sequence_active and self.policy != \"MultiInputPolicy\":\n"
        "            raise ValueError(\n"
        "                \"hierarchical_sequence_v2 requires MultiInputPolicy\"\n"
        "            )\n"
        "        if sequence_active and not ppo_like:\n"
        "            raise ValueError(\n"
        "                \"hierarchical_sequence_v2 currently requires a PPO-family algorithm\"\n"
        "            )\n"
        "        for field_name, value in (\n"
        "            (\"sequence_d_model\", self.sequence_d_model),\n"
        "            (\n"
        "                \"sequence_timeframe_attention_heads\",\n"
        "                self.sequence_timeframe_attention_heads,\n"
        "            ),\n"
        "            (\n"
        "                \"sequence_timeframe_attention_layers\",\n"
        "                self.sequence_timeframe_attention_layers,\n"
        "            ),\n"
        "            (\n"
        "                \"sequence_timeframe_ffn_multiplier\",\n"
        "                self.sequence_timeframe_ffn_multiplier,\n"
        "            ),\n"
        "            (\n"
        "                \"sequence_asset_attention_heads\",\n"
        "                self.sequence_asset_attention_heads,\n"
        "            ),\n"
        "            (\n"
        "                \"sequence_asset_attention_layers\",\n"
        "                self.sequence_asset_attention_layers,\n"
        "            ),\n"
        "            (\n"
        "                \"sequence_asset_ffn_multiplier\",\n"
        "                self.sequence_asset_ffn_multiplier,\n"
        "            ),\n"
        "            (\"max_policy_parameters\", self.max_policy_parameters),\n"
        "        ):\n"
        "            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:\n"
        "                raise ValueError(f\"{field_name} must be a positive integer\")\n"
        "        for field_name, value in (\n"
        "            (\n"
        "                \"sequence_timeframe_gate_bias\",\n"
        "                self.sequence_timeframe_gate_bias,\n"
        "            ),\n"
        "            (\"sequence_asset_gate_bias\", self.sequence_asset_gate_bias),\n"
        "        ):\n"
        "            if not math.isfinite(value):\n"
        "                raise ValueError(f\"{field_name} must be finite\")\n"
        "        for field_name, heads in (\n"
        "            (\n"
        "                \"sequence_timeframe_attention_heads\",\n"
        "                self.sequence_timeframe_attention_heads,\n"
        "            ),\n"
        "            (\n"
        "                \"sequence_asset_attention_heads\",\n"
        "                self.sequence_asset_attention_heads,\n"
        "            ),\n"
        "        ):\n"
        "            if self.sequence_d_model % heads != 0:\n"
        "                raise ValueError(\n"
        "                    f\"sequence_d_model must divide evenly across {field_name}\"\n"
        "                )\n"
        "        if (\n"
        "            not math.isfinite(self.sequence_dropout)\n"
        "            or not 0.0 <= self.sequence_dropout <= 0.05\n"
        "        ):\n"
        "            raise ValueError(\"sequence_dropout must be within [0, 0.05]\")\n"
        "        if (\n"
        "            isinstance(self.max_rollout_buffer_bytes, bool)\n"
        "            or not isinstance(self.max_rollout_buffer_bytes, int)\n"
        "            or self.max_rollout_buffer_bytes <= 0\n"
        "        ):\n"
        "            raise ValueError(\n"
        "                \"max_rollout_buffer_bytes must be a positive integer\"\n"
        "            )\n"
        "        for field_name, value in (\n"
        "            (\"asset_embedding_dim\", self.asset_embedding_dim),\n"
        "            (\"global_embedding_dim\", self.global_embedding_dim),\n"
        "        ):\n"
        "            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:\n"
        "                raise ValueError(f\"{field_name} must be a positive integer\")\n",
        label="encoder validation block",
    )
    text = _sub_once(
        text,
        r"        if not self\.sequence_encoder:\n.*?"
        r"                context=\"asset_set_encoder=False\",\n"
        r"            \)\n",
        "        if not sequence_active:\n"
        "            _require_inactive_defaults(\n"
        "                (\n"
        "                    (\"sequence_tcn_capacity\", self.sequence_tcn_capacity, \"standard\"),\n"
        "                    (\"sequence_d_model\", self.sequence_d_model, 320),\n"
        "                    (\n"
        "                        \"sequence_timeframe_attention_heads\",\n"
        "                        self.sequence_timeframe_attention_heads,\n"
        "                        8,\n"
        "                    ),\n"
        "                    (\n"
        "                        \"sequence_timeframe_attention_layers\",\n"
        "                        self.sequence_timeframe_attention_layers,\n"
        "                        2,\n"
        "                    ),\n"
        "                    (\n"
        "                        \"sequence_timeframe_ffn_multiplier\",\n"
        "                        self.sequence_timeframe_ffn_multiplier,\n"
        "                        3,\n"
        "                    ),\n"
        "                    (\n"
        "                        \"sequence_timeframe_gate_bias\",\n"
        "                        self.sequence_timeframe_gate_bias,\n"
        "                        -2.0,\n"
        "                    ),\n"
        "                    (\n"
        "                        \"sequence_asset_attention_heads\",\n"
        "                        self.sequence_asset_attention_heads,\n"
        "                        8,\n"
        "                    ),\n"
        "                    (\n"
        "                        \"sequence_asset_attention_layers\",\n"
        "                        self.sequence_asset_attention_layers,\n"
        "                        2,\n"
        "                    ),\n"
        "                    (\n"
        "                        \"sequence_asset_ffn_multiplier\",\n"
        "                        self.sequence_asset_ffn_multiplier,\n"
        "                        3,\n"
        "                    ),\n"
        "                    (\n"
        "                        \"sequence_asset_gate_bias\",\n"
        "                        self.sequence_asset_gate_bias,\n"
        "                        -2.0,\n"
        "                    ),\n"
        "                    (\"sequence_dropout\", self.sequence_dropout, 0.05),\n"
        "                    (\"sequence_compile\", self.sequence_compile, False),\n"
        "                    (\n"
        "                        \"sequence_compile_mode\",\n"
        "                        self.sequence_compile_mode,\n"
        "                        \"reduce-overhead\",\n"
        "                    ),\n"
        "                    (\n"
        "                        \"sequence_transfer_mode\",\n"
        "                        self.sequence_transfer_mode,\n"
        "                        \"synchronous\",\n"
        "                    ),\n"
        "                ),\n"
        "                context=f\"observation_encoder={encoder}\",\n"
        "            )\n"
        "        if encoder != \"asset_set\":\n"
        "            _require_inactive_defaults(\n"
        "                (\n"
        "                    (\"asset_embedding_dim\", self.asset_embedding_dim, 64),\n"
        "                    (\"global_embedding_dim\", self.global_embedding_dim, 64),\n"
        "                ),\n"
        "                context=f\"observation_encoder={encoder}\",\n"
        "            )\n",
        label="inactive encoder settings",
    )
    digest_replacements = {
        '            "asset_set_encoder": self.asset_set_encoder,\n': (
            '            "observation_encoder": self.observation_encoder,\n'
        ),
        '            "sequence_encoder": self.sequence_encoder,\n': "",
        '            "sequence_capacity": self.sequence_capacity,\n': (
            '            "sequence_tcn_capacity": self.sequence_tcn_capacity,\n'
        ),
        '            "sequence_attention_heads": self.sequence_attention_heads,\n': (
            '            "sequence_timeframe_attention_heads": '
            'self.sequence_timeframe_attention_heads,\n'
            '            "sequence_asset_attention_heads": '
            'self.sequence_asset_attention_heads,\n'
        ),
        '            "sequence_attention_layers": self.sequence_attention_layers,\n': (
            '            "sequence_timeframe_attention_layers": '
            'self.sequence_timeframe_attention_layers,\n'
            '            "sequence_timeframe_ffn_multiplier": '
            'self.sequence_timeframe_ffn_multiplier,\n'
            '            "sequence_timeframe_gate_bias": '
            'self.sequence_timeframe_gate_bias,\n'
            '            "sequence_asset_attention_layers": '
            'self.sequence_asset_attention_layers,\n'
            '            "sequence_asset_ffn_multiplier": '
            'self.sequence_asset_ffn_multiplier,\n'
            '            "sequence_asset_gate_bias": self.sequence_asset_gate_bias,\n'
        ),
    }
    for old, new in digest_replacements.items():
        text = _replace_once(text, old, new, label=f"digest key {old.strip()}")
    _write(path, text)


def _migrate_training_run_schema() -> None:
    path = "trade_rl/workflows/training_run.py"
    text = _read(path)
    text = _replace_once(
        text,
        '    schema_version: str = "training_run_config_v1"',
        '    schema_version: str = "training_run_config_v2"',
        label="run schema default",
    )
    text = _sub_once(
        text,
        r"        if self\.training\.sequence_encoder and \(\n"
        r"            self\.export_onnx or self\.export_torchscript\n"
        r"        \):\n"
        r"            raise ValueError\(\n"
        r"                \"structured sequence policies do not support flat ONNX/TorchScript export\"\n"
        r"            \)\n",
        "",
        label="obsolete flat export rejection",
    )
    text = _replace_once(
        text,
        '        if self.schema_version != "training_run_config_v1":\n'
        '            raise ValueError("unsupported training run configuration schema")',
        '        if self.schema_version == "training_run_config_v1":\n'
        '            raise ValueError(\n'
        '                "migrate training_run_config_v1 to training_run_config_v2"\n'
        '            )\n'
        '        if self.schema_version != "training_run_config_v2":\n'
        '            raise ValueError("unsupported training run configuration schema")',
        label="run schema validation",
    )
    text = _replace_once(
        text,
        '        schema_version = payload.get("schema_version", "training_run_config_v1")',
        '        schema_version = payload.get("schema_version", "training_run_config_v2")',
        label="run schema parser default",
    )
    _write(path, text)


def _migrate_sequence_policy() -> None:
    path = "trade_rl/rl/sequence_policy.py"
    text = _read(path)
    text = _replace_once(
        text,
        "    attention_heads: int = 8\n"
        "    attention_layers: int = 2\n"
        "    attention_ffn_multiplier: int = 3\n"
        "    attention_gate_bias: float = -2.0\n",
        "    timeframe_attention_heads: int = 8\n"
        "    timeframe_attention_layers: int = 2\n"
        "    timeframe_ffn_multiplier: int = 3\n"
        "    timeframe_gate_bias: float = -2.0\n"
        "    asset_attention_heads: int = 8\n"
        "    asset_attention_layers: int = 2\n"
        "    asset_ffn_multiplier: int = 3\n"
        "    asset_gate_bias: float = -2.0\n",
        label="split policy architecture fields",
    )
    text = _replace_once(
        text,
        "        if self.d_model % self.attention_heads != 0:\n"
        "            raise ValueError(\"d_model must be divisible by attention_heads\")\n"
        "        if self.attention_layers <= 0:\n"
        "            raise ValueError(\"attention_layers must be positive\")\n"
        "        if self.attention_ffn_multiplier <= 0:\n"
        "            raise ValueError(\"attention_ffn_multiplier must be positive\")\n"
        "        if not math.isfinite(self.attention_gate_bias):\n"
        "            raise ValueError(\"attention_gate_bias must be finite\")\n",
        "        for field_name, heads in (\n"
        "            (\"timeframe_attention_heads\", self.timeframe_attention_heads),\n"
        "            (\"asset_attention_heads\", self.asset_attention_heads),\n"
        "        ):\n"
        "            if self.d_model % heads != 0:\n"
        "                raise ValueError(f\"d_model must be divisible by {field_name}\")\n"
        "        for field_name, value in (\n"
        "            (\"timeframe_attention_layers\", self.timeframe_attention_layers),\n"
        "            (\"timeframe_ffn_multiplier\", self.timeframe_ffn_multiplier),\n"
        "            (\"asset_attention_layers\", self.asset_attention_layers),\n"
        "            (\"asset_ffn_multiplier\", self.asset_ffn_multiplier),\n"
        "        ):\n"
        "            if value <= 0:\n"
        "                raise ValueError(f\"{field_name} must be positive\")\n"
        "        for field_name, value in (\n"
        "            (\"timeframe_gate_bias\", self.timeframe_gate_bias),\n"
        "            (\"asset_gate_bias\", self.asset_gate_bias),\n"
        "        ):\n"
        "            if not math.isfinite(value):\n"
        "                raise ValueError(f\"{field_name} must be finite\")\n",
        label="split policy architecture validation",
    )
    replacements = {
        "            heads=architecture.attention_heads,\n"
        "            layers=architecture.attention_layers,\n"
        "            ffn_multiplier=architecture.attention_ffn_multiplier,\n"
        "            dropout=architecture.dropout,\n"
        "            gate_bias=architecture.attention_gate_bias,\n": (
            "            heads=architecture.timeframe_attention_heads,\n"
            "            layers=architecture.timeframe_attention_layers,\n"
            "            ffn_multiplier=architecture.timeframe_ffn_multiplier,\n"
            "            dropout=architecture.dropout,\n"
            "            gate_bias=architecture.timeframe_gate_bias,\n"
        ),
        "            heads=architecture.attention_heads,\n"
        "            layers=architecture.attention_layers,\n"
        "            ffn_multiplier=architecture.attention_ffn_multiplier,\n"
        "            dropout=architecture.dropout,\n"
        "            gate_bias=architecture.attention_gate_bias,\n": (
            "            heads=architecture.asset_attention_heads,\n"
            "            layers=architecture.asset_attention_layers,\n"
            "            ffn_multiplier=architecture.asset_ffn_multiplier,\n"
            "            dropout=architecture.dropout,\n"
            "            gate_bias=architecture.asset_gate_bias,\n"
        ),
    }
    first_old = next(iter(replacements))
    text = _replace_once(
        text,
        first_old,
        replacements[first_old],
        label="timeframe transformer config",
    )
    text = _replace_once(
        text,
        first_old,
        list(replacements.values())[1],
        label="asset transformer config",
    )
    text = _replace_once(
        text,
        "        available: Mapping[str, torch.Tensor],\n"
        "        snapshot: torch.Tensor,",
        "        available: Mapping[str, torch.Tensor],\n"
        "        staleness: Mapping[str, torch.Tensor],\n"
        "        snapshot: torch.Tensor,",
        label="explicit staleness forward argument",
    )
    text = _sub_once(
        text,
        r"            feature_count = self\.architecture\.input_channels\[timeframe\] // 3\n"
        r"            logged_staleness = sequence\[\.\.\., 2 \* feature_count :\]\n"
        r"            if availability\.ndim == 4:\n"
        r"                if availability\.shape\[-1\] != feature_count:\n"
        r"                    raise ValueError\(\"sequence availability channel count is invalid\"\)\n"
        r"                staleness = torch\.expm1\(logged_staleness\)\.clamp_min\(0\.0\)\n"
        r"                timestep_mask = availability\.any\(dim=-1\)\n"
        r"            else:\n"
        r"                staleness = torch\.expm1\(logged_staleness\)\.clamp_min\(0\.0\)\.mean\(dim=-1\)\n"
        r"                timestep_mask = availability\n"
        r"            flattened = sequence\.reshape\(\n"
        r"                batch \* assets, sequence\.shape\[2\], sequence\.shape\[3\]\n"
        r"            \)\n"
        r"            flattened_mask = timestep_mask\.reshape\(batch \* assets, sequence\.shape\[2\]\)\n"
        r"            encoded = self\.timeframe_encoders\[timeframe\]\(flattened, flattened_mask\)\n"
        r"            latents\[timeframe\] = encoded\.reshape\(batch, assets, -1\)\n"
        r"            quality_available\[timeframe\] = availability\n"
        r"            quality_staleness\[timeframe\] = staleness\n",
        "            raw_staleness = staleness[timeframe]\n"
        "            if raw_staleness.shape != availability.shape:\n"
        "                raise ValueError(\n"
        "                    \"sequence staleness must match sequence availability\"\n"
        "                )\n"
        "            timestep_mask = (\n"
        "                availability.any(dim=-1)\n"
        "                if availability.ndim == 4\n"
        "                else availability\n"
        "            )\n"
        "            flattened = sequence.reshape(\n"
        "                batch * assets, sequence.shape[2], sequence.shape[3]\n"
        "            )\n"
        "            flattened_mask = timestep_mask.reshape(\n"
        "                batch * assets, sequence.shape[2]\n"
        "            )\n"
        "            encoded = self.timeframe_encoders[timeframe](\n"
        "                flattened, flattened_mask\n"
        "            )\n"
        "            latents[timeframe] = encoded.reshape(batch, assets, -1)\n"
        "            quality_available[timeframe] = availability\n"
        "            quality_staleness[timeframe] = raw_staleness\n",
        label="explicit quality staleness path",
    )
    _write(path, text)


def _migrate_policies() -> None:
    path = "trade_rl/rl/policies.py"
    text = _read(path)
    text = _replace_once(
        text,
        "        sequence_capacity: str = \"standard\",\n"
        "        d_model: int = 320,\n"
        "        attention_heads: int = 8,\n"
        "        attention_layers: int = 2,\n"
        "        dropout: float = 0.05,",
        "        sequence_tcn_capacity: str = \"standard\",\n"
        "        d_model: int = 320,\n"
        "        timeframe_attention_heads: int = 8,\n"
        "        timeframe_attention_layers: int = 2,\n"
        "        timeframe_ffn_multiplier: int = 3,\n"
        "        timeframe_gate_bias: float = -2.0,\n"
        "        asset_attention_heads: int = 8,\n"
        "        asset_attention_layers: int = 2,\n"
        "        asset_ffn_multiplier: int = 3,\n"
        "        asset_gate_bias: float = -2.0,\n"
        "        dropout: float = 0.05,",
        label="feature extractor split config",
    )
    text = _replace_once(
        text,
        "            attention_heads=attention_heads,\n"
        "            attention_layers=attention_layers,\n"
        "            dropout=dropout,\n"
        "            encoder_widths=sequence_encoder_widths(sequence_capacity),",
        "            timeframe_attention_heads=timeframe_attention_heads,\n"
        "            timeframe_attention_layers=timeframe_attention_layers,\n"
        "            timeframe_ffn_multiplier=timeframe_ffn_multiplier,\n"
        "            timeframe_gate_bias=timeframe_gate_bias,\n"
        "            asset_attention_heads=asset_attention_heads,\n"
        "            asset_attention_layers=asset_attention_layers,\n"
        "            asset_ffn_multiplier=asset_ffn_multiplier,\n"
        "            asset_gate_bias=asset_gate_bias,\n"
        "            dropout=dropout,\n"
        "            encoder_widths=sequence_encoder_widths(sequence_tcn_capacity),",
        label="feature extractor architecture construction",
    )
    text = _replace_once(
        text,
        "            sequences: dict[str, torch.Tensor] = {}\n"
        "            available: dict[str, torch.Tensor] = {}",
        "            sequences: dict[str, torch.Tensor] = {}\n"
        "            available: dict[str, torch.Tensor] = {}\n"
        "            staleness_by_timeframe: dict[str, torch.Tensor] = {}",
        label="explicit staleness collection",
    )
    text = _replace_once(
        text,
        "                available[timeframe] = availability > 0.5\n"
        "            asset_tokens, pooled_assets = self.asset_encoder(\n"
        "                sequences=sequences,\n"
        "                available=available,",
        "                available[timeframe] = availability > 0.5\n"
        "                staleness_by_timeframe[timeframe] = staleness\n"
        "            asset_tokens, pooled_assets = self.asset_encoder(\n"
        "                sequences=sequences,\n"
        "                available=available,\n"
        "                staleness=staleness_by_timeframe,",
        label="pass explicit staleness",
    )
    _write(path, text)


def _migrate_model_assembly() -> None:
    path = "trade_rl/integrations/sb3_model_assembly.py"
    text = _read(path)
    text = text.replace(
        "if config.sequence_encoder", 
        'if config.observation_encoder == "hierarchical_sequence_v2"',
    )
    text = text.replace(
        "if config.asset_set_encoder", 
        'if config.observation_encoder == "asset_set"',
    )
    text = _replace_once(
        text,
        '            "sequence_capacity": config.sequence_capacity,\n'
        '            "d_model": config.sequence_d_model,\n'
        '            "attention_heads": config.sequence_attention_heads,\n'
        '            "attention_layers": config.sequence_attention_layers,\n'
        '            "dropout": config.sequence_dropout,',
        '            "sequence_tcn_capacity": config.sequence_tcn_capacity,\n'
        '            "d_model": config.sequence_d_model,\n'
        '            "timeframe_attention_heads": (\n'
        '                config.sequence_timeframe_attention_heads\n'
        '            ),\n'
        '            "timeframe_attention_layers": (\n'
        '                config.sequence_timeframe_attention_layers\n'
        '            ),\n'
        '            "timeframe_ffn_multiplier": (\n'
        '                config.sequence_timeframe_ffn_multiplier\n'
        '            ),\n'
        '            "timeframe_gate_bias": config.sequence_timeframe_gate_bias,\n'
        '            "asset_attention_heads": config.sequence_asset_attention_heads,\n'
        '            "asset_attention_layers": config.sequence_asset_attention_layers,\n'
        '            "asset_ffn_multiplier": config.sequence_asset_ffn_multiplier,\n'
        '            "asset_gate_bias": config.sequence_asset_gate_bias,\n'
        '            "dropout": config.sequence_dropout,',
        label="SB3 split architecture kwargs",
    )
    _write(path, text)


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for match in re.finditer("\n", text):
        offsets.append(match.end())
    return offsets


def _absolute(offsets: list[int], line: int, column: int) -> int:
    return offsets[line - 1] + column


def _extend_through_comma(text: str, end: int) -> int:
    cursor = end
    while cursor < len(text) and text[cursor] in " \t":
        cursor += 1
    if cursor < len(text) and text[cursor] == ",":
        cursor += 1
    while cursor < len(text) and text[cursor] in " \t":
        cursor += 1
    return cursor


def _migrate_python_call_keywords(text: str, *, path: Path) -> str:
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        raise RuntimeError(f"cannot parse {path}: {error}") from error
    offsets = _line_offsets(text)
    edits: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        old = [
            item
            for item in node.keywords
            if item.arg in {"sequence_encoder", "asset_set_encoder"}
        ]
        if old:
            values: dict[str, bool] = {}
            for item in old:
                if not isinstance(item.value, ast.Constant) or not isinstance(
                    item.value.value, bool
                ):
                    raise RuntimeError(
                        f"non-literal legacy encoder keyword in {path}:{item.lineno}"
                    )
                assert item.arg is not None
                values[item.arg] = item.value.value
            sequence = values.get("sequence_encoder", False)
            asset = values.get("asset_set_encoder", True)
            if sequence and asset:
                encoder = "invalid_legacy_combination"
            elif sequence:
                encoder = SEQUENCE_ENCODER
            elif asset:
                encoder = "asset_set"
            else:
                encoder = "flat_mlp"
            ordered = sorted(
                old,
                key=lambda item: (item.lineno, item.col_offset),
            )
            first = ordered[0]
            start = _absolute(offsets, first.lineno, first.col_offset)
            end = _absolute(offsets, first.end_lineno, first.end_col_offset)
            edits.append((start, end, f'observation_encoder="{encoder}"'))
            for item in ordered[1:]:
                start = _absolute(offsets, item.lineno, item.col_offset)
                end = _absolute(offsets, item.end_lineno, item.end_col_offset)
                edits.append((start, _extend_through_comma(text, end), ""))
        for item in node.keywords:
            if item.arg not in {
                "sequence_capacity",
                "sequence_attention_heads",
                "sequence_attention_layers",
            }:
                continue
            start = _absolute(offsets, item.lineno, item.col_offset)
            end = _absolute(offsets, item.end_lineno, item.end_col_offset)
            value_start = _absolute(
                offsets, item.value.lineno, item.value.col_offset
            )
            value_end = _absolute(
                offsets, item.value.end_lineno, item.value.end_col_offset
            )
            value = text[value_start:value_end]
            indent = " " * item.col_offset
            if item.arg == "sequence_capacity":
                replacement = f"sequence_tcn_capacity={value}"
            elif item.arg == "sequence_attention_heads":
                replacement = (
                    f"sequence_timeframe_attention_heads={value},\n"
                    f"{indent}sequence_asset_attention_heads={value}"
                )
            else:
                replacement = (
                    f"sequence_timeframe_attention_layers={value},\n"
                    f"{indent}sequence_asset_attention_layers={value}"
                )
            edits.append((start, end, replacement))
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


def _migrate_python_sources() -> None:
    excluded = {
        ROOT / "tools/apply_hierarchical_sequence_v2.py",
    }
    for path in sorted(ROOT.rglob("*.py")):
        if path in excluded or any(part in {".git", ".venv"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        original = text
        if any(
            name in text
            for name in (
                "sequence_encoder=",
                "asset_set_encoder=",
                "sequence_capacity=",
                "sequence_attention_heads=",
                "sequence_attention_layers=",
            )
        ):
            text = _migrate_python_call_keywords(text, path=path)
        text = re.sub(
            r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\.sequence_encoder\b",
            r'(\1.observation_encoder == "hierarchical_sequence_v2")',
            text,
        )
        text = re.sub(
            r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\.asset_set_encoder\b",
            r'(\1.observation_encoder == "asset_set")',
            text,
        )
        text = text.replace(".sequence_capacity", ".sequence_tcn_capacity")
        text = text.replace(
            ".sequence_attention_heads", ".sequence_timeframe_attention_heads"
        )
        text = text.replace(
            ".sequence_attention_layers", ".sequence_timeframe_attention_layers"
        )
        if text != original:
            path.write_text(text, encoding="utf-8")


def _migrate_json_configs() -> None:
    for path in sorted((ROOT / "examples/binance-multitimeframe").glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("training"), dict):
            continue
        training = raw["training"]
        sequence = training.pop("sequence_encoder", False)
        asset = training.pop("asset_set_encoder", True)
        if sequence and asset:
            raise RuntimeError(f"invalid legacy encoder combination in {path}")
        training["observation_encoder"] = (
            SEQUENCE_ENCODER if sequence else "asset_set" if asset else "flat_mlp"
        )
        if "sequence_capacity" in training:
            training["sequence_tcn_capacity"] = training.pop("sequence_capacity")
        heads = training.pop("sequence_attention_heads", 8)
        layers = training.pop("sequence_attention_layers", 2)
        training["sequence_timeframe_attention_heads"] = heads
        training["sequence_timeframe_attention_layers"] = layers
        training.setdefault("sequence_timeframe_ffn_multiplier", 3)
        training.setdefault("sequence_timeframe_gate_bias", -2.0)
        training["sequence_asset_attention_heads"] = heads
        training["sequence_asset_attention_layers"] = layers
        training.setdefault("sequence_asset_ffn_multiplier", 3)
        training.setdefault("sequence_asset_gate_bias", -2.0)
        raw["schema_version"] = "training_run_config_v2"
        path.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _add_contract_tests() -> None:
    path = ROOT / "tests/rl/test_observation_encoder_v2.py"
    path.write_text(
        '''from __future__ import annotations

import pytest

from trade_rl.rl.training import ResidualTrainingConfig


def _config(**overrides: object) -> ResidualTrainingConfig:
    values: dict[str, object] = {
        "timesteps": 128,
        "gamma": 0.99,
        "seeds": (0,),
        "n_steps": 128,
        "batch_size": 128,
    }
    values.update(overrides)
    return ResidualTrainingConfig(**values)  # type: ignore[arg-type]


def test_observation_encoder_is_one_closed_choice() -> None:
    with pytest.raises(ValueError, match="observation_encoder"):
        _config(observation_encoder="unknown")


def test_hierarchical_encoder_requires_multi_input_policy() -> None:
    with pytest.raises(ValueError, match="MultiInputPolicy"):
        _config(observation_encoder="hierarchical_sequence_v2")


def test_timeframe_and_asset_attention_are_independent_identity_fields() -> None:
    left = _config(
        observation_encoder="hierarchical_sequence_v2",
        policy="MultiInputPolicy",
        sequence_timeframe_attention_layers=1,
        sequence_asset_attention_layers=2,
    )
    right = _config(
        observation_encoder="hierarchical_sequence_v2",
        policy="MultiInputPolicy",
        sequence_timeframe_attention_layers=2,
        sequence_asset_attention_layers=1,
    )
    assert left.digest_payload() != right.digest_payload()


def test_sequence_fields_fail_closed_for_non_sequence_encoder() -> None:
    with pytest.raises(ValueError, match="inactive"):
        _config(
            observation_encoder="asset_set",
            sequence_timeframe_attention_layers=3,
        )
''',
        encoding="utf-8",
    )


def _assert_legacy_contract_removed() -> None:
    forbidden = (
        "sequence_encoder",
        "asset_set_encoder",
        "sequence_capacity",
        "sequence_attention_heads",
        "sequence_attention_layers",
        "training_run_config_v1",
    )
    roots = (ROOT / "trade_rl", ROOT / "tests", ROOT / "examples/binance-multitimeframe")
    violations: list[str] = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".json"}:
                continue
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    # The explicit v1 migration error is the only maintained mention.
                    if token == "training_run_config_v1" and "migrate training_run_config_v1" in text:
                        continue
                    violations.append(f"{path.relative_to(ROOT)}: {token}")
    if violations:
        raise RuntimeError("legacy configuration remains:\n" + "\n".join(violations))


def main() -> None:
    _migrate_training_config()
    _migrate_training_run_schema()
    _migrate_sequence_policy()
    _migrate_policies()
    _migrate_model_assembly()
    _migrate_python_sources()
    _migrate_json_configs()
    _add_contract_tests()
    _assert_legacy_contract_removed()


if __name__ == "__main__":
    main()
