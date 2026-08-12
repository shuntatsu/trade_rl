"""Callback-list composition kept outside the SB3 training coordinator."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

from trade_rl.domain.common import require_sha256
from trade_rl.rl.training_heartbeat import build_training_heartbeat_callback


def assemble_training_callbacks(
    *,
    checkpoint_callback: object,
    metrics_callback: object | None,
    output_root: Path,
    seed: int,
    algorithm: str,
    environment_digest: str,
    training_config_digest: str,
) -> object:
    """Flatten maintained callbacks and append one identity-bound heartbeat."""

    checkpoint_callbacks = getattr(checkpoint_callback, "callbacks", None)
    callbacks: list[object] = (
        list(checkpoint_callbacks)
        if isinstance(checkpoint_callbacks, list)
        else [checkpoint_callback]
    )
    if metrics_callback is not None:
        callbacks.append(metrics_callback)
    identity = {
        "environment_digest": environment_digest,
        "training_config_digest": training_config_digest,
    }
    runtime_manifest_digest = os.environ.get(
        "TRADE_RL_RUNTIME_MANIFEST_DIGEST", ""
    ).strip()
    if runtime_manifest_digest:
        identity["runtime_manifest_digest"] = require_sha256(
            runtime_manifest_digest,
            field="TRADE_RL_RUNTIME_MANIFEST_DIGEST",
        )
    callbacks.append(
        build_training_heartbeat_callback(
            output_root / "training-heartbeat.json",
            seed=seed,
            algorithm=algorithm,
            identity=identity,
        )
    )
    if len(callbacks) == 1:
        return callbacks[0]

    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    # Runtime callback compatibility belongs to SB3 itself.  Keep test doubles and
    # custom maintained callbacks accepted here while narrowing only for Mypy's
    # CallbackList constructor contract.
    return CallbackList(cast(list[BaseCallback], callbacks))


__all__ = ["assemble_training_callbacks"]