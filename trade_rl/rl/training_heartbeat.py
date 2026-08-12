"""Atomic, identity-bound progress heartbeat for long-running training."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes


def build_training_heartbeat_callback(
    path: Path,
    *,
    seed: int,
    algorithm: str,
    identity: Mapping[str, str],
) -> Any:
    """Build an SB3 callback that replaces one finite progress snapshot atomically."""

    from stable_baselines3.common.callbacks import BaseCallback

    output = Path(path)
    bound_identity = {str(key): str(value) for key, value in identity.items()}

    class TrainingHeartbeatCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)

        def _write(self, phase: str) -> None:
            logger = getattr(self.model, "logger", None)
            values = getattr(logger, "name_to_value", {})
            scalars = {
                str(key): float(value)
                for key, value in values.items()
                if isinstance(value, (int, float, np.number))
                and not isinstance(value, bool)
                and np.isfinite(value)
            }
            atomic_write_bytes(
                output,
                canonical_json_bytes(
                    {
                        "algorithm": algorithm,
                        "global_step": int(self.model.num_timesteps),
                        "phase": phase,
                        "scalars": scalars,
                        "schema_version": "training_heartbeat_v1",
                        "seed": seed,
                        "updated_at": datetime.now(UTC).isoformat(),
                        **bound_identity,
                    }
                )
                + b"\n",
            )

        def _on_training_start(self) -> None:
            self._write("training")

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            self._write("training")

        def _on_training_end(self) -> None:
            self._write("completed")

    return TrainingHeartbeatCallback()


__all__ = ["build_training_heartbeat_callback"]
