from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.selection import PolicyMode
from trade_rl.release.asymmetric import PublicVerificationKey
from trade_rl.release.attestation import (
    default_attestation_path,
    write_release_attestation,
)
from trade_rl.release.offline_approval import create_release_attestation
from trade_rl.release.offline_signing import public_key_bytes
from trade_rl.rl.actions import ACTION_SCHEMA
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import OBSERVATION_SCHEMA
from trade_rl.serving.bundle import (
    ServingBundleManifest,
    write_serving_bundle_manifest,
)
from trade_rl.serving.normalizer import write_observation_normalizer
from trade_rl.serving.runtime import RuntimeIdentityContract

OBSERVATION_SIZE = 5
ACTION_NAMES = ("fast_tilt", "slow_tilt", "risk_tilt")
ACTION_SPEC_DIGEST = content_digest({"names": ACTION_NAMES})
INITIAL_CAPITAL = 250_000.0
TEST_ATTESTATION_KEY_ID = "test-attestation-key"
TEST_ATTESTATION_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x11" * 32)
_CREATED_AT = datetime(2026, 7, 13, tzinfo=UTC)
TEST_ATTESTATION_PUBLIC_KEY = PublicVerificationKey(
    key_id=TEST_ATTESTATION_KEY_ID,
    public_key=public_key_bytes(TEST_ATTESTATION_PRIVATE_KEY),
    purpose="release-verification",
    valid_from=datetime(2026, 1, 1, tzinfo=UTC),
    valid_until=datetime(2030, 1, 1, tzinfo=UTC),
)
TEST_TRUSTED_ATTESTATION_KEYS = {TEST_ATTESTATION_KEY_ID: TEST_ATTESTATION_PUBLIC_KEY}


def TEST_CLOCK() -> datetime:
    return datetime(2026, 7, 14, tzinfo=UTC)


TRAINING_RUN_DIGEST = "1" * 64
SELECTION_PROPOSAL_DIGEST = "2" * 64
SELECTION_AUTHORIZATION_DIGEST = "3" * 64
WALK_FORWARD_RUN_DIGEST = "4" * 64
GATE_EVIDENCE_DIGEST = "5" * 64
CONFIRMATION_EVIDENCE_DIGEST = "6" * 64


def _normalizer(
    *,
    observation_size: int = OBSERVATION_SIZE,
    mean: float = 0.0,
    scale: float = 1.0,
    action_spec_digest: str = ACTION_SPEC_DIGEST,
) -> ObservationNormalizer:
    return ObservationNormalizer(
        mean=np.full(observation_size, mean, dtype=np.float64),
        scale=np.full(observation_size, scale, dtype=np.float64),
        train_start=0,
        train_end=1,
        dataset_id="a" * 64,
        source_dataset_id="a" * 64,
        observation_schema=OBSERVATION_SCHEMA,
        action_spec_digest=action_spec_digest,
    )


NORMALIZER_DIGEST = _normalizer().digest


def _write_authenticated_attestation(
    root: Path, manifest: ServingBundleManifest
) -> None:
    attestation = create_release_attestation(
        bundle_digest=manifest.bundle_digest,
        dataset_id=manifest.dataset_id,
        training_run_digest=manifest.training_run_digest,
        run_kind=manifest.run_kind,
        selection_proposal_digest=manifest.selection_proposal_digest,
        selection_authorization_digest=manifest.selection_authorization_digest,
        walk_forward_run_digest=manifest.walk_forward_run_digest,
        gate_evidence_digest=manifest.gate_evidence_digest,
        confirmation_evidence_digest=manifest.confirmation_evidence_digest,
        selected_policy_digest=manifest.policy_digest,
        git_commit="3" * 40,
        dependency_digest="4" * 64,
        approver="serving-test",
        approved_at=_CREATED_AT,
        expires_at=_CREATED_AT + timedelta(days=365),
        key_id=TEST_ATTESTATION_KEY_ID,
        private_key=TEST_ATTESTATION_PRIVATE_KEY,
    )
    write_release_attestation(default_attestation_path(root), attestation)


def create_bundle(
    root: Path,
    *,
    policy_mode: PolicyMode = PolicyMode.BASELINE_ONLY,
    release_digest: str | None = "released",
    observation_size: int = OBSERVATION_SIZE,
    action_names: tuple[str, ...] = ACTION_NAMES,
    action_spec_digest: str = ACTION_SPEC_DIGEST,
    environment_digest: str = "d" * 64,
    normalizer_digest: str | None = NORMALIZER_DIGEST,
    normalizer_mean: float = 0.0,
    normalizer_scale: float = 1.0,
) -> Path:
    root.mkdir(parents=True)
    artifact_paths = ["dataset.json", "signal.json", "selection.json"]
    (root / "dataset.json").write_text('{"dataset":"a"}', encoding="utf-8")
    (root / "signal.json").write_text('{"signal":"rejected"}', encoding="utf-8")
    (root / "selection.json").write_text(
        f'{{"selection":"{policy_mode.value}"}}', encoding="utf-8"
    )
    if normalizer_digest is not None:
        resolved_normalizer = _normalizer(
            observation_size=observation_size,
            mean=normalizer_mean,
            scale=normalizer_scale,
            action_spec_digest=action_spec_digest,
        )
        normalizer_digest = resolved_normalizer.digest
        write_observation_normalizer(root, resolved_normalizer)
        artifact_paths.append("normalizer.json")
    policy_digest: str | None = None
    selected = policy_mode is PolicyMode.RESIDUAL_POLICY
    if selected:
        (root / "policy.zip").write_bytes(b"residual-policy")
        artifact_paths.append("policy.zip")
        policy_digest = "e" * 64
    manifest = ServingBundleManifest.build(
        root=root,
        dataset_id="a" * 64,
        action_schema=ACTION_SCHEMA,
        action_size=len(action_names),
        action_names=action_names,
        action_spec_digest=action_spec_digest,
        observation_schema=OBSERVATION_SCHEMA,
        observation_size=observation_size,
        environment_digest=environment_digest,
        initial_capital=INITIAL_CAPITAL,
        policy_mode=policy_mode,
        policy_digest=policy_digest,
        signal_digest="b" * 64,
        selection_digest="c" * 64,
        normalizer_digest=normalizer_digest,
        artifact_paths=tuple(artifact_paths),
        created_at=_CREATED_AT,
        training_run_digest=(TRAINING_RUN_DIGEST if selected else None),
        run_kind=("research_selected_final" if selected else "baseline_release"),
        selection_proposal_digest=(SELECTION_PROPOSAL_DIGEST if selected else None),
        selection_authorization_digest=(
            SELECTION_AUTHORIZATION_DIGEST if selected else None
        ),
        walk_forward_run_digest=(WALK_FORWARD_RUN_DIGEST if selected else None),
        gate_evidence_digest=(GATE_EVIDENCE_DIGEST if selected else None),
        confirmation_evidence_digest=(
            CONFIRMATION_EVIDENCE_DIGEST if selected else None
        ),
    )
    write_serving_bundle_manifest(root, manifest)
    if release_digest is not None:
        _write_authenticated_attestation(root, manifest)
    return root


def create_authenticated_bundle(root: Path, **kwargs: object) -> Path:
    if "release_digest" in kwargs:
        raise ValueError("authenticated bundle controls release externally")
    return create_bundle(root, release_digest="released", **kwargs)


def runtime_identity_contract(
    *,
    environment_digest: str = "d" * 64,
    action_names: tuple[str, ...] = ACTION_NAMES,
    action_spec_digest: str = ACTION_SPEC_DIGEST,
    normalizer_digest: str | None = NORMALIZER_DIGEST,
    normalizer_mean: float = 0.0,
    normalizer_scale: float = 1.0,
    alpha_artifact_digest: str | None = None,
    factor_artifact_digest: str | None = None,
) -> RuntimeIdentityContract:
    if normalizer_digest is not None:
        normalizer_digest = _normalizer(
            mean=normalizer_mean,
            scale=normalizer_scale,
            action_spec_digest=action_spec_digest,
        ).digest
    return RuntimeIdentityContract(
        environment_digest=environment_digest,
        action_names=action_names,
        action_spec_digest=action_spec_digest,
        normalizer_digest=normalizer_digest,
        alpha_artifact_digest=alpha_artifact_digest,
        factor_artifact_digest=factor_artifact_digest,
    )
