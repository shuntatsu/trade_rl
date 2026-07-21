"""Thread-safe serving runtime with validated fail-closed hot swaps."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Protocol

import numpy as np

from trade_rl.domain.common import require_sha256
from trade_rl.domain.selection import PolicyMode
from trade_rl.release.asymmetric import PublicVerificationKey
from trade_rl.release.attestation import ReleaseAttestation
from trade_rl.rl.actions import ACTION_SCHEMA
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import OBSERVATION_SCHEMA, PolicyObservationSnapshot
from trade_rl.rl.sequence_observations import SEQUENCE_OBSERVATION_SCHEMA
from trade_rl.serving.bundle import ServingBundle, load_serving_bundle
from trade_rl.serving.state import ServingStateGuard, ServingStateSnapshot

PolicyObservation = np.ndarray | Mapping[str, np.ndarray]


class LoadedPolicy(Protocol):
    def predict(self, observation: PolicyObservation) -> np.ndarray: ...


class PolicyLoader(Protocol):
    def load(self, bundle: ServingBundle) -> LoadedPolicy: ...


class _BaselineIdentityPolicy:
    def __init__(self, action_size: int) -> None:
        self.action_size = action_size

    def predict(self, observation: PolicyObservation) -> np.ndarray:
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
    training_run_digest: str | None
    run_kind: str
    selection_proposal_digest: str | None
    selection_authorization_digest: str | None
    walk_forward_run_digest: str | None
    gate_evidence_digest: str | None
    confirmation_evidence_digest: str | None
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
        trusted_attestation_keys: Mapping[str, PublicVerificationKey] | None = None,
        clock: Callable[[], datetime] | None = None,
        allow_unbound_state: bool = False,
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
        if not isinstance(allow_unbound_state, bool):
            raise ValueError("allow_unbound_state must be a boolean")
        if identity_contract is None and not allow_unbound_identity:
            raise ValueError("serving runtime requires an explicit identity contract")
        self.policy_loader = policy_loader
        self.allow_unreleased = allow_unreleased
        self.identity_contract = identity_contract
        self.allow_unbound_identity = allow_unbound_identity
        self.trusted_attestation_keys = dict(trusted_attestation_keys or {})
        self.clock = clock or (lambda: datetime.now(UTC))
        self.allow_unbound_state = allow_unbound_state or allow_unreleased
        self._state_guard = ServingStateGuard()
        self._lock = RLock()
        self._state_lock = RLock()
        self._snapshot: RuntimeSnapshot | None = None
        self._policy: LoadedPolicy | None = None
        self._normalizer: ObservationNormalizer | None = None

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
            training_run_digest=manifest.training_run_digest,
            run_kind=manifest.run_kind,
            selection_proposal_digest=manifest.selection_proposal_digest,
            selection_authorization_digest=manifest.selection_authorization_digest,
            walk_forward_run_digest=manifest.walk_forward_run_digest,
            gate_evidence_digest=manifest.gate_evidence_digest,
            confirmation_evidence_digest=manifest.confirmation_evidence_digest,
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

    @staticmethod
    def _predict_action(
        policy: LoadedPolicy,
        snapshot: RuntimeSnapshot,
        normalizer: ObservationNormalizer | None,
        observation: PolicyObservation,
    ) -> np.ndarray:
        if isinstance(observation, Mapping):
            if snapshot.observation_schema != SEQUENCE_OBSERVATION_SCHEMA:
                raise ValueError("structured observation violates the active schema")
            if not observation or any(
                np.asarray(value).size == 0 or not np.isfinite(np.asarray(value)).all()
                for value in observation.values()
            ):
                raise ValueError("structured observation violates the active schema")
            policy_input: Any = dict(observation)
        else:
            vector = np.asarray(observation, dtype=np.float32).reshape(-1)
            if (
                snapshot.observation_schema != OBSERVATION_SCHEMA
                or vector.shape != (snapshot.observation_size,)
                or not np.isfinite(vector).all()
            ):
                raise ValueError("observation violates the active observation schema")
            policy_input = (
                vector if normalizer is None else normalizer.transform(vector)
            )
        raw_action = np.asarray(
            policy.predict(policy_input),
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

    def activate(self, root: Path) -> RuntimeSnapshot:
        bundle = load_serving_bundle(root)
        manifest = bundle.manifest
        if self.allow_unreleased:
            pass
        elif bundle.release is None:
            raise ValueError("serving bundle requires a verified release attestation")
        elif isinstance(bundle.release, ReleaseAttestation):
            bundle.release.verify(
                self.trusted_attestation_keys,
                trusted_at=self.clock(),
            )
        else:
            raise TypeError("serving bundle release metadata has an unsupported type")
        if manifest.action_schema != ACTION_SCHEMA:
            raise ValueError(
                "serving bundle action schema does not match runtime action schema"
            )
        if manifest.observation_schema not in {
            OBSERVATION_SCHEMA,
            SEQUENCE_OBSERVATION_SCHEMA,
        }:
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
        candidate_normalizer = bundle.normalizer
        if candidate_normalizer is not None and not isinstance(
            candidate_normalizer, ObservationNormalizer
        ):
            raise ValueError("serving bundle normalizer type is invalid")
        smoke_factory = getattr(candidate_policy, "smoke_observation", None)
        smoke = (
            smoke_factory()
            if callable(smoke_factory)
            else np.zeros(candidate_snapshot.observation_size, dtype=np.float32)
        )
        self._predict_action(
            candidate_policy,
            candidate_snapshot,
            candidate_normalizer,
            smoke,
        )
        with self._lock:
            self._policy = candidate_policy
            self._snapshot = candidate_snapshot
            self._normalizer = candidate_normalizer
        return candidate_snapshot

    def snapshot(self) -> RuntimeSnapshot:
        with self._lock:
            snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError("serving runtime has no active snapshot")
        return snapshot

    def predict(self, observation: PolicyObservation) -> np.ndarray:
        if not isinstance(observation, Mapping):
            vector = np.asarray(observation, dtype=np.float32).reshape(-1)
            if vector.size == 0 or not np.isfinite(vector).all():
                raise ValueError("observation must be a non-empty finite vector")
            observation = vector
        with self._lock:
            policy = self._policy
            snapshot = self._snapshot
            normalizer = self._normalizer
        if policy is None or snapshot is None:
            raise RuntimeError("serving runtime has no active policy")
        return self._predict_action(policy, snapshot, normalizer, observation)

    def predict_from_dataset(
        self,
        dataset: Any,
        *,
        index: int,
        current_flat: np.ndarray,
        state_snapshot: ServingStateSnapshot | None = None,
        portfolio_state: np.ndarray | None = None,
        pending_target: np.ndarray | None = None,
    ) -> np.ndarray:
        with self._lock:
            policy = self._policy
            snapshot = self._snapshot
        if policy is None or snapshot is None:
            raise RuntimeError("serving runtime has no active policy")
        predictor = getattr(policy, "predict_from_dataset", None)
        if not callable(predictor):
            raise RuntimeError(
                "active policy does not support structured dataset serving"
            )
        if state_snapshot is None:
            if not self.allow_unbound_state:
                raise ValueError(
                    "structured serving requires an identity-bound state snapshot"
                )
            raw = np.asarray(
                predictor(dataset, index=index, current_flat=current_flat),
                dtype=np.float32,
            ).reshape(-1)
        else:
            if portfolio_state is None:
                raise ValueError(
                    "identity-bound structured serving requires the portfolio state"
                )
            if pending_target is None:
                raise ValueError(
                    "identity-bound structured serving requires the pending target"
                )
            dataset_id = getattr(dataset, "dataset_id", None)
            if not isinstance(dataset_id, str):
                raise ValueError("structured serving dataset lacks dataset_id")
            with self._state_lock:
                self._state_guard.require_matches(
                    state_snapshot,
                    dataset_id=dataset_id,
                    decision_index=index,
                    portfolio_state=portfolio_state,
                    pending_target=pending_target,
                    current_flat=current_flat,
                )
                raw = np.asarray(
                    predictor(dataset, index=index, current_flat=current_flat),
                    dtype=np.float32,
                ).reshape(-1)
                self._state_guard.accept(state_snapshot)
        if (
            raw.shape != (snapshot.action_size,)
            or not np.isfinite(raw).all()
            or np.any(raw < -1.0)
            or np.any(raw > 1.0)
        ):
            raise ValueError("policy output violates the residual action schema")
        return raw.copy()

    def predict_from_observation_snapshot(
        self,
        dataset: Any,
        observation_snapshot: PolicyObservationSnapshot,
    ) -> np.ndarray:
        """Predict from an immutable state exported by the training environment."""

        observation_snapshot.require_matches_dataset(dataset)
        with self._lock:
            policy = self._policy
            snapshot = self._snapshot
            normalizer = self._normalizer
        if policy is None or snapshot is None:
            raise RuntimeError("serving runtime has no active policy")
        raw_observation = observation_snapshot.raw_observation
        expected = (
            raw_observation.astype(np.float32, copy=True)
            if normalizer is None
            else normalizer.transform(raw_observation)
        )
        if not np.allclose(
            expected,
            observation_snapshot.normalized_observation,
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError("serving normalized observation parity mismatch")
        state = ServingStateSnapshot.create(
            dataset_id=observation_snapshot.dataset_id,
            decision_index=observation_snapshot.index,
            portfolio_state=observation_snapshot.hybrid_book_state,
            pending_target=observation_snapshot.pending_target,
            observation_digest=ServingStateSnapshot.observation_digest_for(
                raw_observation
            ),
        )
        with self._state_lock:
            self._state_guard.require_fresh(state)
            predictor = getattr(policy, "predict_from_dataset", None)
            if callable(predictor):
                action = np.asarray(
                    predictor(
                        dataset,
                        index=observation_snapshot.index,
                        current_flat=raw_observation,
                    ),
                    dtype=np.float32,
                ).reshape(-1)
            else:
                action = self._predict_action(
                    policy,
                    snapshot,
                    normalizer,
                    raw_observation,
                )
            self._state_guard.accept(state)
        if (
            action.shape != (snapshot.action_size,)
            or not np.isfinite(action).all()
            or np.any(action < -1.0)
            or np.any(action > 1.0)
        ):
            raise ValueError("policy output violates the residual action schema")
        return action.copy()
