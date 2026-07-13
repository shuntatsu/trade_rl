from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from trade_rl.domain.selection import PolicyMode
from trade_rl.serving.bundle import (
    ServingBundleManifest,
    write_serving_bundle_manifest,
)


def create_bundle(
    root: Path,
    *,
    policy_mode: PolicyMode = PolicyMode.BASELINE_ONLY,
    dataset_id: str = "a" * 64,
    observation_schema_digest: str = "d" * 64,
    observation_size: int = 5,
) -> Path:
    root.mkdir(parents=True)
    artifact_paths = ["dataset.json", "signal.json", "selection.json"]
    (root / "dataset.json").write_text('{"dataset":"a"}', encoding="utf-8")
    (root / "signal.json").write_text('{"signal":"rejected"}', encoding="utf-8")
    (root / "selection.json").write_text(
        f'{{"selection":"{policy_mode.value}"}}',
        encoding="utf-8",
    )
    policy_digest: str | None = None
    if policy_mode is PolicyMode.RESIDUAL_POLICY:
        policy = root / "policy.zip"
        policy.write_bytes(b"residual-policy")
        artifact_paths.append("policy.zip")
        policy_digest = "e" * 64

    manifest = ServingBundleManifest.build(
        root=root,
        dataset_id=dataset_id,
        action_schema="baseline_residual_v1",
        observation_schema_digest=observation_schema_digest,
        observation_size=observation_size,
        policy_mode=policy_mode,
        policy_digest=policy_digest,
        signal_digest="b" * 64,
        selection_digest="c" * 64,
        release_digest=None,
        artifact_paths=tuple(artifact_paths),
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )
    write_serving_bundle_manifest(root, manifest)
    return root
