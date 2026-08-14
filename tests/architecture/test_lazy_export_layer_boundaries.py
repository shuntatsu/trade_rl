from __future__ import annotations

import trade_rl.integrations as integrations
import trade_rl.rl as rl
from tests.architecture.import_linter_config import configured_layers


def _layer_indices() -> dict[str, int]:
    return {name: index for index, name in enumerate(configured_layers())}


def _owning_layer(module_name: str, indices: dict[str, int]) -> str:
    matches = tuple(
        layer
        for layer in indices
        if module_name == layer or module_name.startswith(f"{layer}.")
    )
    if len(matches) != 1:
        raise AssertionError(
            f"could not resolve one architecture layer for {module_name}"
        )
    return matches[0]


def test_rl_lazy_exports_do_not_target_an_upper_layer() -> None:
    indices = _layer_indices()
    source_layer = "trade_rl.rl"

    violations: list[str] = []
    for target_module in rl._MODULE_EXPORTS:
        if not target_module.startswith("trade_rl."):
            continue
        target_layer = _owning_layer(target_module, indices)
        if indices[target_layer] < indices[source_layer]:
            violations.append(f"{source_layer} -> {target_module}")

    assert violations == []


def test_sb3_backends_are_exported_only_from_integrations() -> None:
    for name in ("StableBaselines3Backend", "StableBaselines3PPOBackend"):
        assert name not in rl.__all__

    assert "StableBaselines3PolicyLoader" in integrations.__all__
