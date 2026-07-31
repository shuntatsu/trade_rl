from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from tests.serving.test_package import (
    PUBLIC_KEY,
    _confirmation,
    _training_run,
)
from trade_rl.rl.actions import ActionMode, ActionSpec
from trade_rl.serving.bundle import load_serving_bundle
from trade_rl.workflows.release_packaging import package_selected_training_run


def test_release_packaging_preserves_target_weight_action_mode(
    tmp_path: Path,
) -> None:
    training_root = tmp_path / "training"
    training = _training_run(
        training_root,
        run_kind="research_selected_final",
        action_spec=ActionSpec(
            mode=ActionMode.TARGET_WEIGHT,
            risk_tilt_enabled=False,
            target_weight_count=1,
        ),
    )
    confirmation_path = tmp_path / "confirmation.json"
    _confirmation(confirmation_path, training)

    output_root = tmp_path / "bundle"
    manifest = package_selected_training_run(
        training_root=training_root,
        confirmation_path=confirmation_path,
        output_root=output_root,
        signal_digest="a" * 64,
        selection_digest="b" * 64,
        trusted_confirmation_keys={PUBLIC_KEY.key_id: PUBLIC_KEY},
        trusted_now=training.completed_at + timedelta(days=30),
    )

    loaded = load_serving_bundle(output_root).manifest
    assert manifest.action_mode is ActionMode.TARGET_WEIGHT
    assert loaded.action_mode is ActionMode.TARGET_WEIGHT
    assert loaded.policy_mode is manifest.policy_mode
