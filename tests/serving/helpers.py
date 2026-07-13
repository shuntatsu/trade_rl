from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.selection import PolicyMode
from trade_rl.rl.actions import ACTION_SCHEMA
from trade_rl.rl.observations import OBSERVATION_SCHEMA
from trade_rl.serving.bundle import ServingBundleManifest, write_serving_bundle_manifest

OBSERVATION_SIZE = 5
ACTION_NAMES = ("fast_tilt", "slow_tilt", "risk_tilt")
ACTION_SPEC_DIGEST = content_digest({"names": ACTION_NAMES})
INITIAL_CAPITAL = 250_000.0


def create_bundle(
    root: Path,
    *,
    policy_mode: PolicyMode = PolicyMode.BASELINE_ONLY,
    release_digest: str | None = "f" * 64,
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
    (root / "selection.json").write_text(f'{{"selection":"{policy_mode.value}"}}', encoding="utf-8")
    if release_digest is not None:
        (root / "release.json").write_text('{"release":"approved"}', encoding="utf-8")
        artifact_paths.append("release.json")
    policy_digest: str | None = None
    if policy_mode is PolicyMode.RESIDUAL_POLICY:
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
        release_digest=release_digest,
        normalizer_digest=normalizer_digest,
        artifact_paths=tuple(artifact_paths),
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )
    write_serving_bundle_manifest(root, manifest)
    return root
