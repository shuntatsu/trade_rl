"""Finish v2 migration for nested JSON candidates and local helper variables."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _migrate_mapping(value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            _migrate_mapping(item)
        return
    if not isinstance(value, dict):
        return
    if "sequence_encoder" in value or "asset_set_encoder" in value:
        sequence = value.pop("sequence_encoder", False)
        asset = value.pop("asset_set_encoder", True)
        if not isinstance(sequence, bool) or not isinstance(asset, bool):
            raise RuntimeError("JSON encoder flags must be booleans")
        if sequence and asset:
            encoder = "invalid_legacy_combination"
        elif sequence:
            encoder = "hierarchical_sequence_v2"
        elif asset:
            encoder = "asset_set"
        else:
            encoder = "flat_mlp"
        value["observation_encoder"] = encoder
    if "sequence_capacity" in value:
        value["sequence_tcn_capacity"] = value.pop("sequence_capacity")
    if "sequence_attention_heads" in value:
        heads = value.pop("sequence_attention_heads")
        value["sequence_timeframe_attention_heads"] = heads
        value["sequence_asset_attention_heads"] = heads
    if "sequence_attention_layers" in value:
        layers = value.pop("sequence_attention_layers")
        value["sequence_timeframe_attention_layers"] = layers
        value["sequence_asset_attention_layers"] = layers
    if value.get("observation_encoder") == "hierarchical_sequence_v2":
        value.setdefault("sequence_timeframe_ffn_multiplier", 3)
        value.setdefault("sequence_timeframe_gate_bias", -2.0)
        value.setdefault("sequence_asset_ffn_multiplier", 3)
        value.setdefault("sequence_asset_gate_bias", -2.0)
    for item in tuple(value.values()):
        _migrate_mapping(item)


def _migrate_json() -> None:
    root = ROOT / "examples" / "binance-multitimeframe"
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        _migrate_mapping(payload)
        if isinstance(payload, dict) and "schema_version" in payload:
            payload["schema_version"] = "training_run_config_v2"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _migrate_local_helper_names() -> None:
    path = ROOT / "tests" / "integrations" / "test_sb3_training.py"
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"\basset_set_encoder(?=\s*:\s*bool)",
        "asset_set_enabled",
        text,
    )
    text = re.sub(r"\basset_set_encoder\b", "asset_set_enabled", text)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    _migrate_json()
    _migrate_local_helper_names()


if __name__ == "__main__":
    main()
