from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from trade_rl.workflows import release_packaging


def test_release_packaging_rejects_selected_final_without_authorization_chain(
    tmp_path, monkeypatch: pytest.MonkeyPatch
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
