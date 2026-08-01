from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from trade_rl.domain.selection import PolicyMode
from trade_rl.rl.actions import ACTION_SCHEMA, ActionMode
from trade_rl.rl.observations import OBSERVATION_SCHEMA
from trade_rl.serving.bundle import (
    SERVING_BUNDLE_SCHEMA,
    ServingBundleManifest,
    load_serving_bundle,
    write_serving_bundle_manifest,
)
from trade_rl.serving.runtime import ServingRuntime


def _build_manifest(root: Path, *, action_mode: ActionMode) -> ServingBundleManifest:
    root.mkdir()
    (root / "policy.zip").write_bytes(b"policy")
    return ServingBundleManifest.build(
        root=root,
        dataset_id="a" * 64,
        action_schema=ACTION_SCHEMA,
        action_mode=action_mode,
        action_size=1,
        action_names=("target_weight:BTCUSDT",),
        action_spec_digest="b" * 64,
        observation_schema=OBSERVATION_SCHEMA,
        observation_size=5,
        environment_digest="c" * 64,
        initial_capital=100_000.0,
        policy_mode=PolicyMode.RESIDUAL_POLICY,
        policy_digest="d" * 64,
        signal_digest="e" * 64,
        selection_digest="f" * 64,
        artifact_paths=("policy.zip",),
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
        training_run_digest="1" * 64,
        run_kind="research_selected_final",
        selection_proposal_digest="2" * 64,
        selection_authorization_digest="3" * 64,
        walk_forward_run_digest="4" * 64,
        gate_evidence_digest="5" * 64,
        confirmation_evidence_digest="6" * 64,
    )


@pytest.mark.parametrize("action_mode", tuple(ActionMode))
def test_learned_policy_keeps_explicit_action_mode(
    tmp_path: Path,
    action_mode: ActionMode,
) -> None:
    root = tmp_path / action_mode.value
    manifest = _build_manifest(root, action_mode=action_mode)
    write_serving_bundle_manifest(root, manifest)

    bundle = load_serving_bundle(root)
    loaded = bundle.manifest
    snapshot = ServingRuntime._snapshot_for(bundle)

    assert SERVING_BUNDLE_SCHEMA == "serving_bundle_v6"
    assert loaded.policy_mode is PolicyMode.RESIDUAL_POLICY
    assert loaded.action_mode is action_mode
    assert loaded.digest_payload()["action_mode"] is action_mode
    assert snapshot.policy_mode is PolicyMode.RESIDUAL_POLICY
    assert snapshot.action_mode is action_mode
