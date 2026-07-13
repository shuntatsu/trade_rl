"""Thread-safe serving runtime with validated fail-closed hot swaps."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Protocol

import numpy as np

from trade_rl.domain.selection import PolicyMode
from trade_rl.rl.actions import ACTION_SCHEMA
from trade_rl.rl.market_inputs import MarketInputResolver
from trade_rl.rl.observations import ObservationBuilder, ObservationInput
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
    observation_schema_digest: str
    observation_size: int
    market_inputs_digest: str
    policy_mode: PolicyMode
    policy_digest: str | None
    signal_digest: str
    selection_digest: str
    release_digest: str | None
    bundle_created_at: datetime


class ServingRuntime:
    """Validate and fully load a replacement before swapping live state."""

    def __init__(
        self,
        policy_loader: PolicyLoader | None = None,
        observation_builder: ObservationBuilder | None = None,
        market_input_resolver: MarketInputResolver | None = None,
    ) -> None:
        self.policy_loader = policy_loader
        self.observation_builder = observation_builder or ObservationBuilder()
        self.market_input_resolver = market_input_resolver or MarketInputResolver()
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
            observation_schema_digest=manifest.observation_schema_digest,
            observation_size=manifest.observation_size,
            market_inputs_digest=manifest.market_inputs_digest,
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

        if bundle.manifest.market_inputs_digest != self.market_input_resolver.digest:
            raise ValueError(
                "serving bundle market inputs do not match runtime market inputs"
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

    def _active(self) -> tuple[RuntimeSnapshot, LoadedPolicy]:
        with self._lock:
            snapshot = self._snapshot
            policy = self._policy
        if snapshot is None or policy is None:
            raise RuntimeError("serving runtime has no active policy")
        return snapshot, policy

    @staticmethod
    def _validated_vector(observation: np.ndarray, *, expected_size: int) -> np.ndarray:
        vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        if vector.size != expected_size:
            raise ValueError(
                "observation size does not match the active serving bundle"
            )
        if not np.isfinite(vector).all():
            raise ValueError("observation must contain only finite values")
        return vector

    @staticmethod
    def _validated_action(policy: LoadedPolicy, vector: np.ndarray) -> np.ndarray:
        raw_action = np.asarray(policy.predict(vector), dtype=np.float32).reshape(-1)
        if raw_action.shape != (2,) or not np.isfinite(raw_action).all():
            raise ValueError("policy output violates the residual action schema")
        if np.any(raw_action < -1.0) or np.any(raw_action > 1.0):
            raise ValueError("policy output violates the residual action schema bounds")
        return raw_action.copy()

    def build_observation(self, value: ObservationInput) -> np.ndarray:
        """Recompute untrusted Trend and Alpha through the causal runtime contract."""

        trends, alpha = self.market_input_resolver.resolve(
            value.dataset,
            value.index,
        )
        return self.observation_builder.build(
            replace(value, trends=trends, alpha=alpha)
        )

    @staticmethod
    def _validate_input_contract(
        snapshot: RuntimeSnapshot,
        *,
        dataset_id: str,
        observation_schema_digest: str,
        market_inputs_digest: str,
    ) -> None:
        if dataset_id != snapshot.dataset_id:
            raise ValueError(
                "serving dataset identity does not match the active bundle"
            )
        if observation_schema_digest != snapshot.observation_schema_digest:
            raise ValueError(
                "serving observation schema does not match the active bundle"
            )
        if market_inputs_digest != snapshot.market_inputs_digest:
            raise ValueError("serving market inputs do not match the active bundle")

    def predict_state(self, value: ObservationInput) -> np.ndarray:
        """Validate, build, and score one structured current-time market state."""

        snapshot, policy = self._active()
        dataset = value.dataset
        self._validate_input_contract(
            snapshot,
            dataset_id=dataset.dataset_id,
            observation_schema_digest=self.observation_builder.schema_digest(dataset),
            market_inputs_digest=self.market_input_resolver.digest,
        )
        vector = self._validated_vector(
            self.build_observation(value),
            expected_size=snapshot.observation_size,
        )
        with self._lock:
            if self._snapshot != snapshot or self._policy is not policy:
                raise RuntimeError("serving bundle changed during prediction")
            return self._validated_action(policy, vector)

    def predict(
        self,
        observation: np.ndarray,
        *,
        dataset_id: str,
        observation_schema_digest: str,
        market_inputs_digest: str,
    ) -> np.ndarray:
        snapshot, policy = self._active()
        self._validate_input_contract(
            snapshot,
            dataset_id=dataset_id,
            observation_schema_digest=observation_schema_digest,
            market_inputs_digest=market_inputs_digest,
        )
        vector = self._validated_vector(
            observation,
            expected_size=snapshot.observation_size,
        )
        with self._lock:
            if self._snapshot != snapshot or self._policy is not policy:
                raise RuntimeError("serving bundle changed during prediction")
            return self._validated_action(policy, vector)
