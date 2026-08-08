"""Opt-in residual environment wrapper for execution dual-shadow evidence."""

from __future__ import annotations

from typing import Any

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_execution import (
    EnvironmentExecutionDualShadow,
    ExecutionDualShadowSnapshot,
)


class ExecutionDualShadowResidualMarketEnv(ResidualMarketEnv):
    """Residual environment that observes hybrid execution without changing authority."""

    def __init__(
        self,
        *args: Any,
        execution_dual_shadow: EnvironmentExecutionDualShadow,
        **kwargs: Any,
    ) -> None:
        if not execution_dual_shadow.identity_digest:
            raise ValueError("execution dual-shadow identity digest must be non-empty")
        super().__init__(*args, **kwargs)
        self._execution_dual_shadow = execution_dual_shadow
        self._execution_coordinator.dual_shadow = execution_dual_shadow
        self._dual_shadow_environment_digest = content_digest(
            {
                "base_environment_digest": super().environment_digest,
                "execution_dual_shadow_identity": execution_dual_shadow.identity_digest,
                "schema_version": "execution_dual_shadow_environment_v1",
            }
        )

    @property
    def environment_digest(self) -> str:
        return self._dual_shadow_environment_digest

    @property
    def latest_execution_dual_shadow(self) -> ExecutionDualShadowSnapshot | None:
        return self._execution_coordinator.latest_dual_shadow_snapshot

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        self._execution_coordinator.reset_dual_shadow(
            start_index=self.start_index,
            initial_quantities=tuple(float(value) for value in self.hybrid.quantities),
        )
        return observation, info


__all__ = ["ExecutionDualShadowResidualMarketEnv"]
