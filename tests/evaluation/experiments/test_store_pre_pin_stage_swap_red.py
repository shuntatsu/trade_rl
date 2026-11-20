"""RED: a substitute staging directory must not be admitted before its FD is pinned.

All paths contain only synthetic data inside pytest's temporary directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.store import StudyStore


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/fd").is_dir(),
    reason="requires POSIX stage descriptor pinning",
)
@pytest.mark.parametrize("forbid_bulk_listdir", (False, True))
def test_pre_pin_nonempty_staging_swap_is_rejected_without_deleting_foreign_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forbid_bulk_listdir: bool
) -> None:
    store = StudyStore(tmp_path / "study")
    saved_original_stage = tmp_path / "original-stage"
    true_mkdir = Path.mkdir
    swapped_stage: Path | None = None

    def inject_after_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal swapped_stage
        true_mkdir(path, *args, **kwargs)
        if path.name.startswith(".synthetic.staging-") and swapped_stage is None:
            swapped_stage = store.root / path.name
            path.rename(saved_original_stage)
            true_mkdir(path)
            (path / "forged.json").write_text(
                '{"is_validated":false}', encoding="utf-8"
            )

    monkeypatch.setattr(Path, "mkdir", inject_after_mkdir)
    if forbid_bulk_listdir:
        real_listdir = os.listdir

        def refuse_bulk_enumeration(path: int | str | bytes) -> list[str]:
            if isinstance(path, int):
                raise AssertionError("unbounded FD directory listing is forbidden")
            return real_listdir(path)

        monkeypatch.setattr(os, "listdir", refuse_bulk_enumeration)

    def build(stage: Path) -> None:
        (stage / "genuine.json").write_text('{"is_validated":true}', encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError):
        store.publish_directory_once("evidence/synthetic", build)

    assert swapped_stage is not None
    assert (saved_original_stage).is_dir()
    assert (swapped_stage / "forged.json").read_text(encoding="utf-8") == (
        '{"is_validated":false}'
    ), "a failed publication must preserve someone else's substituted data"
    assert not (store.root / "evidence" / "synthetic").exists()
    assert not (swapped_stage / "genuine.json").exists()
