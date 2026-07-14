"""Thread-safe serving runtime with validated fail-closed hot swaps."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Protocol

import numpy as np

from trade_rl.domain.common import require_sha256
from trade_rl.domain.selection import PolicyMode
from trade_rl.rl.actions import ACTION_SCHEMA
from trade_rl.rl.observations import OBSERVATION_SCHEMA
from trade_rl.serving.bundle import ServingBundle, load_serving_bundle


class LoadedPolicy(Protocol):
    def predict(self, observation: np.ndarray) -> np.ndarray: ...


class PolicyLoader(Protocol):
    def load(self, bundle: ServingBundle) -> LoadedPolicy: ...


class _BaselineIdentityPolicy:
    def __init__(self, action_size: int) -> None:
        self.action_size = action_size

    def predict(self, observation: np.ndarray) -> np.ndarray:
        del observation
        return np.zeros(self.action_size, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class RuntimeIdentityContract:
    """Exact deployment identity required before a bundle can activate."""

    environment_digest: str
    action_names: tuple[str, ...]
    action_spec_digest: str
    normalizer_digest: str | None
    alpha_artifact_digest: str | None = None
    factor_artifact_digest: str | None = None

    def __post_init__(self) -> None:
        require_sha256(self.environment_digest, field="environment_digest")
        require_sha256(self.action_spec_digest, field="action_spec_digest")
        if not self.action_names or any(not name for name in self.action_names):
            raise ValueError("action_names must be non-empty")
        if len(set(self.action_names)) != len(self.action_names):
            raise ValueError("action_names must be unique")
        for field_name, value in (
            ("normalizer_digest", self.normalizer_digest),
            ("alpha_artifact_digest", self.alpha_artifact_digest),
            ("factor_artifact_digest", self.factor_artifact_digest),
        ):
            if value is not None:
                require_sha256(value, field=field_name)


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    bundle_digest: str
    dataset_id: str
    action_schema: str
    action_size: int
    action_names: tuple[str, ...]
    action_spec_digest: str | None
    observation_schema: str
    observation_size: int
    environment_digest: str
    initial_capital: float
    policy_mode: PolicyMode
    policy_digest: str | None
    signal_digest: str
    selection_digest: str
    release_digest: str | None
    alpha_artifact_digest: str | None
    factor_artifact_digest: str | None
    normalizer_digest: str | None
    bundle_created_at: datetime


class ServingRuntime:
    """Validate and fully load a replacement before swapping live state."""

    def __init__(
        self,
        policy_loader: PolicyLoader | None = None,
        *,
        allow_unreleased: bool = False,
        identity_contract: RuntimeIdentityContract | None = None,
        allow_unbound_identity: bool = False,
        expected_environment_digest: str | None = None,
        expected_action_names: tuple[str, ...] | None = None,
        expected_action_spec_digest: str | None = None,
        expected_normalizer_digest: str | None = None,
        expected_alpha_artifact_digest: str | None = None,
        expected_factor_artifact_digest: str | None = None,
    ) -> None:
        legacy_values = (
            expected_environment_digest,
            expected_action_names,
            expected_action_spec_digest,
            expected_normalizer_digest,
            expected_alpha_artifact_digest,
            expected_factor_artifact_digest,
        )
        if identity_contract is not None and any(
            value is not None for value in legacy_values
        ):
            raise ValueError(
                "identity_contract cannot be combined with legacy expected fields"
            )
        if identity_contract is None and any(
            value is not None for value in legacy_values
        ):
            if (
                expected_environment_digest is None
                or expected_action_names is None
                or expected_action_spec_digest is None
            ):
                raise ValueError(
                    "legacy serving identity requires environment, action names, "
                    "and action spec"
                )
            identity_contract = RuntimeIdentityContract(
                environment_digest=expected_environment_digest,
                action_names=expected_action_names,
                action_spec_digest=expected_action_spec_digest,
                normalizer_digest=expected_normalizer_digest,
                alpha_artifact_digest=expected_alpha_artifact_digest,
                factor_artifact_digest=expected_factor_artifact_digest,
            )
        if not isinstance(allow_unbound_identity, bool):
            raise ValueError("allow_unbound_identity must be a boolean")
        if identity_contract is None and not allow_unbound_identity:
            raise ValueError("serving runtime requires an explicit identity contract")
        self.policy_loader = policy_loader
        self.allow_unreleased = allow_unreleased
        self.identity_contract = identity_contract
        self.allow_unbound_identity = allow_unbound_identity
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
            action_size=manifest.action_size,
            action_names=manifest.action_names,
            action_spec_digest=manifest.action_spec_digest,
            observation_schema=manifest.observation_schema,
            observation_size=manifest.observation_size,
            environment_digest=manifest.environment_digest,
            initial_capital=manifest.initial_capital,
            policy_mode=manifest.policy_mode,
            policy_digest=manifest.policy_digest,
            signal_digest=manifest.signal_digest,
            selection_digest=manifest.selection_digest,
            release_digest=(None if bundle.release is None else bundle.release.digest),
            alpha_artifact_digest=manifest.alpha_artifact_digest,
            factor_artifact_digest=manifest.factor_artifact_digest,
            normalizer_digest=manifest.normalizer_digest,
            bundle_created_at=manifest.created_at,
        )

    @staticmethod
    def _validate_identity(
        manifest: object,
        contract: RuntimeIdentityContract,
    ) -> None:
        comparisons = (
            (
                getattr(manifest, "environment_digest"),
                contract.environment_digest,
                "environment identity",
            ),
            (
                getattr(manifest, "action_names"),
                contract.action_names,
                "action names",
            ),
            (
                getattr(manifest, "action_spec_digest"),
                contract.action_spec_digest,
                "action spec",
            ),
            (
                getattr(manifest, "normalizer_digest"),
                contract.normalizer_digest,
                "normalizer",
            ),
            (
                getattr(manifest, "alpha_artifact_digest"),
                contract.alpha_artifact_digest,
                "alpha artifact",
            ),
            (
                getattr(manifest, "factor_artifact_digest"),
                contract.factor_artifact_digest,
                "factor artifact",
            ),
        )
        for observed, expected, label in comparisons:
            if observed != expected:
                raise ValueError(f"serving bundle {label} does not match runtime")

    def activate(self, root: Path) -> RuntimeSnapshot:
        bundle = load_serving_bundle(root)
        manifest = bundle.manifest
        if bundle.release is None and not self.allow_unreleased:
            raise ValueError("serving bundle requires a verified release attestation")
        if manifest.action_schema != ACTION_SCHEMA:
            raise ValueError(
                "serving bundle action schema does not match runtime action schema"
            )
        if manifest.observation_schema != OBSERVATION_SCHEMA:
            raise ValueError(
                "serving bundle observation schema does not match runtime schema"
            )
        contract = self.identity_contract
        if contract is not None:
            self._validate_identity(manifest, contract)
        elif not self.allow_unbound_identity:
            raise RuntimeError("serving identity contract was not configured")

        if manifest.policy_mode is PolicyMode.BASELINE_ONLY:
            candidate_policy: LoadedPolicy = _BaselineIdentityPolicy(
                manifest.action_size
            )
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
            snapshot = self._snapshot
        if policy is None or snapshot is None:
            raise RuntimeError("serving runtime has no active policy")
        if vector.shape != (snapshot.observation_size,):
            raise ValueError("observation violates the active observation schema")
        raw_action = np.asarray(
            policy.predict(vector),
            dtype=np.float32,
        ).reshape(-1)
        if (
            raw_action.shape != (snapshot.action_size,)
            or not np.isfinite(raw_action).all()
        ):
            raise ValueError("policy output violates the residual action schema")
        if np.any(raw_action < -1.0) or np.any(raw_action > 1.0):
            raise ValueError("policy output violates the residual action schema bounds")
        return raw_action.copy()
