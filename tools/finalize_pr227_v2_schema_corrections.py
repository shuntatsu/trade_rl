from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _correct_mapping(node: Any) -> int:
    changed = 0
    if isinstance(node, list):
        for item in node:
            changed += _correct_mapping(item)
        return changed
    if not isinstance(node, dict):
        return changed

    for value in tuple(node.values()):
        changed += _correct_mapping(value)

    if isinstance(node.get("candidates"), list) and isinstance(
        node.get("workflow"), Mapping
    ):
        if node.get("schema_version") != "market_walk_forward_config_v1":
            node["schema_version"] = "market_walk_forward_config_v1"
            changed += 1
    elif isinstance(node.get("training"), Mapping):
        if node.get("schema_version") != "training_run_config_v2":
            node["schema_version"] = "training_run_config_v2"
            changed += 1

    legacy_capacity = node.pop("sequence_capacity", None)
    if legacy_capacity is not None:
        existing = node.get("sequence_tcn_capacity")
        if existing is not None and existing != legacy_capacity:
            raise ValueError(
                "sequence_tcn_capacity disagrees with legacy sequence_capacity"
            )
        node["sequence_tcn_capacity"] = legacy_capacity
        changed += 1

    return changed


def _assert_contract(node: Any, *, path: Path) -> None:
    if isinstance(node, list):
        for item in node:
            _assert_contract(item, path=path)
        return
    if not isinstance(node, dict):
        return

    forbidden = {
        "asset_set_encoder",
        "sequence_encoder",
        "sequence_capacity",
        "sequence_attention_heads",
        "sequence_attention_layers",
    }
    overlap = forbidden.intersection(node)
    if overlap:
        raise RuntimeError(f"{path}: legacy v1 fields remain: {sorted(overlap)}")

    if isinstance(node.get("candidates"), list) and isinstance(
        node.get("workflow"), Mapping
    ):
        if node.get("schema_version") != "market_walk_forward_config_v1":
            raise RuntimeError(f"{path}: walk-forward schema was not preserved")
    elif isinstance(node.get("training"), Mapping):
        if node.get("schema_version") != "training_run_config_v2":
            raise RuntimeError(f"{path}: training run schema was not migrated")

    for value in node.values():
        _assert_contract(value, path=path)


def main() -> None:
    changed_files = 0
    for path in sorted((ROOT / "examples").rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        changed = _correct_mapping(payload)
        _assert_contract(payload, path=path)
        if changed:
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            changed_files += 1
    if changed_files <= 0:
        raise RuntimeError("expected PR227 v2 schema corrections to modify examples")


if __name__ == "__main__":
    main()
