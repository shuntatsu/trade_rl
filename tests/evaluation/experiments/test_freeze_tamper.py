from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.evaluation.experiments.test_workflow import _with_baseline
from trade_rl.evaluation.experiments import (
    ArtifactIntegrityError,
    StudyOutcome,
    freeze_study,
)


def test_freeze_rejects_tampered_seed_returns_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, snapshot = _with_baseline(tmp_path, monkeypatch)
    seed = snapshot.plan.ppo_seeds[0]
    returns_path = (
        root / "baseline" / "evidence" / "runs" / f"seed-{seed}" / "returns.npz"
    )
    payload = bytearray(returns_path.read_bytes())
    payload[0] ^= 0x01
    returns_path.write_bytes(payload)

    with pytest.raises(ArtifactIntegrityError):
        freeze_study(
            root,
            outcome=StudyOutcome.NO_WINNER,
            rationale="Tampered evidence cannot be frozen.",
            frozen_by="researcher",
            frozen_at=datetime(2026, 9, 9, tzinfo=UTC),
        )

    assert not (root / "freeze.json").exists()
