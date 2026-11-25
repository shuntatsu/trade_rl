"""Isolated synthetic cache-stage cleanup and error-provenance regressions."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from trade_rl.integrations.binance.carry import CarryOperationalHalt
from trade_rl.integrations.market_store.archive_url import VISION_PREFIX
from trade_rl.integrations.market_store.carry_source import ArchiveDirFetcher
from trade_rl.integrations.market_store.ingest import FetchResult

URL = f"{VISION_PREFIX}synthetic/cached.zip"


@pytest.mark.parametrize("mode", ("published", "publication_failed", "raced_winner"))
def test_stage_cleanup_error_is_typed_and_preserves_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """A private-stage unlink failure may not escape as raw PermissionError."""
    target = ArchiveDirFetcher(tmp_path)._path(URL)
    original_unlink = Path.unlink
    orphan_stages: list[Path] = []

    def deny_stage_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path.parent == target.parent and path.name.startswith(
            f".{target.name}.part-"
        ):
            orphan_stages.append(path)
            raise PermissionError(errno.EACCES, "synthetic cleanup denied")
        original_unlink(path, *args, **kwargs)  # type: ignore[arg-type]

    class StubFetcher:
        def fetch(self, url: str) -> FetchResult:
            assert url == URL
            if mode == "raced_winner":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"rival winner bytes")
            return FetchResult(status=200, body=b"fresh synthetic bytes")

    def reject_hardlink(_source: Path, _destination: Path) -> None:
        raise OSError(errno.ENOTSUP, "synthetic unsupported hardlink")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "unlink", deny_stage_unlink)
        if mode == "publication_failed":
            scoped.setattr(os, "link", reject_hardlink)
        with pytest.raises(CarryOperationalHalt) as caught:
            ArchiveDirFetcher(tmp_path, StubFetcher()).fetch(URL)

    expected_kind = (
        "cache_publication_failed"
        if mode == "publication_failed"
        else "cache_cleanup_failed"
    )
    assert caught.value.evidence == {"kind": expected_kind, "url": URL}
    assert caught.value.kind == "operational"
    assert caught.value.resumable is True
    if mode == "publication_failed":
        assert isinstance(caught.value.__cause__, OSError)
        assert caught.value.__cause__.errno == errno.ENOTSUP
        assert "cleanup failed" in " ".join(getattr(caught.value, "__notes__", ()))
        assert not target.exists()
    else:
        assert isinstance(caught.value.__cause__, PermissionError)
        assert target.read_bytes() == (
            b"rival winner bytes"
            if mode == "raced_winner"
            else b"fresh synthetic bytes"
        )
    assert len(orphan_stages) == 1
    assert orphan_stages[0].is_file()
    for stage in orphan_stages:
        stage.unlink()
