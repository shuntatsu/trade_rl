"""External hardlinks must not escape Study publication immutability checks."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.store import StudyStore


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX inode links")
def test_directory_publish_rejects_external_hardlink_alias(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "study")
    outside = tmp_path / "outside-mutable.json"
    outside.write_text('{"approved":true}', encoding="utf-8")

    def builder(stage: Path) -> None:
        os.link(outside, stage / "evidence.json")

    with pytest.raises(ArtifactIntegrityError, match="(hardlink|link count)"):
        store.publish_directory_once("evidence/synthetic", builder)
    assert outside.read_text(encoding="utf-8") == '{"approved":true}'
    assert not (store.root / "evidence" / "synthetic").exists()
