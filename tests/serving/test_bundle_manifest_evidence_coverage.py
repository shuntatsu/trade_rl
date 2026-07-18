from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.serving.helpers import create_bundle
from trade_rl.domain.selection import PolicyMode
from trade_rl.serving.bundle import load_serving_bundle


def test_baseline_manifest_rejects_selected_final_run_kind(tmp_path: Path) -> None:
    manifest = load_serving_bundle(
        create_bundle(tmp_path / "baseline-run-kind", release_digest=None)
    ).manifest

    with pytest.raises(ValueError, match="baseline_release run_kind"):
        replace(manifest, run_kind="research_selected_final")


def test_baseline_manifest_rejects_training_evidence(tmp_path: Path) -> None:
    manifest = load_serving_bundle(
        create_bundle(tmp_path / "baseline-evidence", release_digest=None)
    ).manifest

    with pytest.raises(ValueError, match="cannot contain training evidence"):
        replace(manifest, training_run_digest="f" * 64)


def test_residual_manifest_rejects_baseline_run_kind(tmp_path: Path) -> None:
    manifest = load_serving_bundle(
        create_bundle(
            tmp_path / "residual-run-kind",
            policy_mode=PolicyMode.RESIDUAL_POLICY,
            release_digest=None,
        )
    ).manifest

    with pytest.raises(ValueError, match="selected-final run_kind"):
        replace(manifest, run_kind="baseline_release")


def test_residual_manifest_requires_complete_authorization_chain(
    tmp_path: Path,
) -> None:
    manifest = load_serving_bundle(
        create_bundle(
            tmp_path / "residual-evidence",
            policy_mode=PolicyMode.RESIDUAL_POLICY,
            release_digest=None,
        )
    ).manifest

    with pytest.raises(ValueError, match="complete authorization chain"):
        replace(manifest, confirmation_evidence_digest=None)
