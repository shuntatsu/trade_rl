from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from trade_rl.workflows import release_packaging


def test_release_packaging_rejects_selected_final_without_authorization_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = SimpleNamespace(
        run_kind="research_selected_final",
        selection_proposal_digest=None,
        selection_authorization_digest="1" * 64,
        walk_forward_run_digest="2" * 64,
        gate_evidence_digest="3" * 64,
    )
    monkeypatch.setattr(
        release_packaging,
        "validate_training_run_directory",
        lambda _root: manifest,
    )

    with pytest.raises(ValueError, match="authorization chain"):
        release_packaging.package_selected_training_run(
            training_root=tmp_path / "training",
            confirmation_path=tmp_path / "confirmation.json",
            output_root=tmp_path / "bundle",
            signal_digest="a" * 64,
            selection_digest="b" * 64,
            trusted_confirmation_keys={},
            trusted_now=datetime(2026, 8, 1, tzinfo=UTC),
        )


def test_release_packaging_rechecks_runtime_promotion_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal_digest = "4" * 64
    promotion_digest = "5" * 64
    manifest = SimpleNamespace(selection_proposal_digest=proposal_digest)
    proposal = SimpleNamespace(
        digest=proposal_digest,
        runtime_promotion_report_digest=promotion_digest,
    )
    report = SimpleNamespace(requested_mode="nautilus_authoritative")
    monkeypatch.setattr(
        release_packaging,
        "load_selection_proposal",
        lambda _path: proposal,
    )
    monkeypatch.setattr(
        release_packaging,
        "load_execution_promotion_report",
        lambda _path: report,
    )

    def reject_mismatched_report(**kwargs: object) -> None:
        assert kwargs == {
            "proposal": proposal,
            "report": report,
            "required_mode": report.requested_mode,
        }
        raise ValueError("runtime promotion report digest mismatch")

    monkeypatch.setattr(
        release_packaging,
        "require_selection_execution_promotion",
        reject_mismatched_report,
    )

    with pytest.raises(ValueError, match="runtime promotion report digest mismatch"):
        release_packaging._require_runtime_promotion_binding(
            training_root=tmp_path,
            manifest=manifest,
        )


def test_release_packaging_rejects_unbound_runtime_promotion_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal_digest = "6" * 64
    manifest = SimpleNamespace(selection_proposal_digest=proposal_digest)
    proposal = SimpleNamespace(
        digest=proposal_digest,
        runtime_promotion_report_digest=None,
    )
    monkeypatch.setattr(
        release_packaging,
        "load_selection_proposal",
        lambda _path: proposal,
    )
    (tmp_path / "runtime-promotion-report.json").write_text("{}", encoding="utf-8")

    with pytest.raises(
        ValueError, match="does not authorize runtime promotion evidence"
    ):
        release_packaging._require_runtime_promotion_binding(
            training_root=tmp_path,
            manifest=manifest,
        )


def test_release_packaging_accepts_legacy_proposal_without_promotion_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal_digest = "7" * 64
    manifest = SimpleNamespace(selection_proposal_digest=proposal_digest)
    proposal = SimpleNamespace(
        digest=proposal_digest,
        runtime_promotion_report_digest=None,
    )
    monkeypatch.setattr(
        release_packaging,
        "load_selection_proposal",
        lambda _path: proposal,
    )

    release_packaging._require_runtime_promotion_binding(
        training_root=tmp_path,
        manifest=manifest,
    )


def test_release_packaging_rejects_proposal_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = SimpleNamespace(selection_proposal_digest="8" * 64)
    proposal = SimpleNamespace(
        digest="9" * 64,
        runtime_promotion_report_digest=None,
    )
    monkeypatch.setattr(
        release_packaging,
        "load_selection_proposal",
        lambda _path: proposal,
    )

    with pytest.raises(ValueError, match="selection proposal digest"):
        release_packaging._require_runtime_promotion_binding(
            training_root=tmp_path,
            manifest=manifest,
        )
