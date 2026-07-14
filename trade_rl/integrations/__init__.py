"""Framework-specific adapters composed around the core research runtime.

The package root intentionally avoids importing optional frameworks.  Consumers
should import adapters from their concrete modules; attribute access remains as
an explicit lazy compatibility shim.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "LoadedAlphaArtifact": (
        "trade_rl.integrations.signal_artifacts",
        "LoadedAlphaArtifact",
    ),
    "LoadedFactorArtifact": (
        "trade_rl.integrations.signal_artifacts",
        "LoadedFactorArtifact",
    ),
    "load_alpha_artifact": (
        "trade_rl.integrations.signal_artifacts",
        "load_alpha_artifact",
    ),
    "load_factor_artifact": (
        "trade_rl.integrations.signal_artifacts",
        "load_factor_artifact",
    ),
    "StableBaselines3CheckpointLoader": (
        "trade_rl.integrations.checkpoints",
        "StableBaselines3CheckpointLoader",
    ),
    "StableBaselines3PolicyLoader": (
        "trade_rl.integrations.sb3_serving",
        "StableBaselines3PolicyLoader",
    ),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    return getattr(import_module(module_name), attribute)


__all__ = tuple(_EXPORTS)
