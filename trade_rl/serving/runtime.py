"""Thread-safe serving runtime with validated fail-closed hot swaps."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Protocol

import numpy as np

from trade_rl.domain.selection import PolicyMode
from trade_rl.rl.actions import ACTION_SCHEMA
from trade_rl.serving.bundle import ServingBundle, load_serving_bundle


class LoadedPolicy(Protocol):
    """Minimal inference contract required by the serving runtime."""

    def predict(self, observation: np.ndarray) -> np.ndarray: ...


class PolicyLoader(Protocol):
    """Adapter that resolves one validated bundle into an inference policy."""

    def load(self, bundle: ServingBundle) -> LoadedPolicy: ...


class _BaselineIdentityPolicy:
    def predict(self, observation: np.ndarray) -> np.ndarray:
        del observation
        return np.zeros(2, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """Immutable serving identity exposed after a successful activation."""

    bundle_digest: str
    dataset_id: str
    action_schema: str
    policy_mode: PolicyMode
    policy_digest: str | None
    signal_digest: str
    selection_digest: str
    release_digest: str | None
    bundle_created_at: datetime


class ServingRuntime:
    """Validate and fully load a replacement before swapping live state."""

    def __init__(self, policy_loader: PolicyLoader | None = None) -> None:
        self.policy_loader = policy_loader
        self._lock = RLock()
        self._snapshot: RuntimeSnapshot | None = None
        self._policy: LoadedPolicy | None = None

    @staticmethod
    def _snapshot_for(bundle: ServingBundle) -> RuntimeSnapshot:
        manifest = bundle.manifest
        return RuntimeSnapshot(
            bundle_digest=manifest.bundle_digest,
            dataset_id=manifest.dataset_id,
            action_schema=manifest.action_schema,
            policy_mode=manifest.policy_mode,
            policy_digest=manifest.policy_digest,
            signal_digest=manifest.signal_digest,
            selection_digest=manifest.selection_digest,
            release_digest=manifest.release_digest,
            bundle_created_at=manifest.created_at,
        )

    def activate(self, root: Path) -> RuntimeSnapshot:
        """Activate only after bundle validation and policy loading both succeed."""

        bundle = load_serving_bundle(root)
        if bundle.manifest.action_schema != ACTION_SCHEMA:
            raise ValueError(
                "serving bundle action schema does not match runtime action schema"
            )

        if bundle.manifest.policy_mode is PolicyMode.BASELINE_ONLY:
            candidate_policy: LoadedPolicy = _BaselineIdentityPolicy()
        else:
            loader = self.policy_loader
            if loader is None:
                raise RuntimeError("residual policy bundle requires a policy loader")
            candidate_policy = loader.load(bundle)

        candidate_snapshot = self._snapshot_for(bundle)
        with self._lock:
            self._policy = candidate_policy
            self._snapshot = candidate_snapshot
        return candidate_snapshot

    def snapshot(self) -> RuntimeSnapshot:
        with self._lock:
            snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError("serving runtime has no active snapshot")
        return snapshot

    def predict(self, observation: np.ndarray) -> np.ndarray:
        vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("observation must be a non-empty finite vector")
        with self._lock:
            policy = self._policy
        if policy is None:
            raise RuntimeError("serving runtime has no active policy")
        raw_action = np.asarray(policy.predict(vector), dtype=np.float32).reshape(-1)
        if raw_action.shape != (2,) or not np.isfinite(raw_action).all():
            raise ValueError("policy output violates the residual action schema")
        if np.any(raw_action < -1.0) or np.any(raw_action > 1.0):
            raise ValueError("policy output violates the residual action schema bounds")
        return raw_action.copy()
