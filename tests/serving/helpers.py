from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.releases import ReleaseManifest
from trade_rl.domain.selection import PolicyMode
from trade_rl.rl.actions import ACTION_SCHEMA
from trade_rl.rl.observations import OBSERVATION_SCHEMA
from trade_rl.serving.bundle import ServingBundleManifest, write_serving_bundle_manifest
from trade_rl.serving.release import write_release_attestation
from trade_rl.serving.runtime import RuntimeIdentityContract

OBSERVATION_SIZE = 5
ACTION_NAMES = ("fast_tilt", "slow_tilt", "risk_tilt")
ACTION_SPEC_DIGEST = content_digest({"names": ACTION_NAMES})
INITIAL_CAPITAL = 250_000.0
_CREATED_AT = datetime(2026, 7, 13, tzinfo=UTC)


def create_bundle(
    root: Path,
    *,
    policy_mode: PolicyMode = PolicyMode.BASELINE_ONLY,
    release_digest: str | None = "released",
    observation_size: int = OBSERVATION_SIZE,
    action_names: tuple[str, ...] = ACTION_NAMES,
    action_spec_digest: str = ACTION_SPEC_DIGEST,
    environment_digest: str = "d" * 64,
    normalizer_digest: str | None = "9" * 64,
) -> Path:
    root.mkdir(parents=True)
    artifact_paths = ["dataset.json", "signal.json", "selection.json"]
    (root / "dataset.json").write_text('{"dataset":"a"}', encoding="utf-8")
    (root / "signal.json").write_text('{"signal":"rejected"}', encoding="utf-8")
    (root / "selection.json").write_text(
        f'{{"selection":"{policy_mode.value}"}}', encoding="utf-8"
    )
    policy_digest: str | None = None
    if policy_mode is PolicyMode.RESIDUAL_POLICY:
        (root / "policy.zip").write_bytes(b"residual-policy")
        artifact_paths.append("policy.zip")
        policy_digest = "e" * 64
    candidate = ServingBundleManifest.build(
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
        release_digest=None,
        normalizer_digest=normalizer_digest,
        artifact_paths=tuple(artifact_paths),
        created_at=_CREATED_AT,
    )
    manifest = candidate
    if release_digest is not None:
        release = ReleaseManifest(
            version="2026.07.13",
            git_commit="e" * 40,
            dataset_id=candidate.dataset_id,
            signal_digest=candidate.signal_digest,
            selection_digest=candidate.selection_digest,
            selection_evaluation_digest="1" * 64,
            gate_evaluation_digest="2" * 64,
            selected_policy_digest=candidate.policy_digest,
            bundle_digest=candidate.bundle_digest,
            created_at=_CREATED_AT,
        )
        manifest = candidate.with_release(release)
        write_release_attestation(root, release)
    write_serving_bundle_manifest(root, manifest)
    return root


def runtime_identity_contract(
    *,
    environment_digest: str = "d" * 64,
    action_names: tuple[str, ...] = ACTION_NAMES,
    action_spec_digest: str = ACTION_SPEC_DIGEST,
    normalizer_digest: str | None = "9" * 64,
    alpha_artifact_digest: str | None = None,
    factor_artifact_digest: str | None = None,
) -> RuntimeIdentityContract:
    return RuntimeIdentityContract(
        environment_digest=environment_digest,
        action_names=action_names,
        action_spec_digest=action_spec_digest,
        normalizer_digest=normalizer_digest,
        alpha_artifact_digest=alpha_artifact_digest,
        factor_artifact_digest=factor_artifact_digest,
    )
