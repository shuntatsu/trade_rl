"""Raw storage, manifest vintages, revision detection and fetch orchestration."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest

from trade_rl.integrations.cross_asset.catalog import CATALOG, series_by_key
from trade_rl.integrations.cross_asset.fetch import (
    TransportError,
    UrllibTransport,
    fetch_all,
)
from trade_rl.integrations.cross_asset.store import (
    CrossAssetStore,
    format_capture_utc,
    latest_records,
    manifest_digest,
)

T0 = datetime(2026, 11, 10, 5, 0, tzinfo=UTC)

CHALLENGE = b"<!DOCTYPE html><script>fetch('/__verify')</script>"


class FakeTransport:
    def __init__(self, responses: dict[str, bytes | Exception]) -> None:
        self.responses = responses
        self.requested: list[str] = []

    def get(self, url: str) -> bytes:
        self.requested.append(url)
        for fragment, response in self.responses.items():
            if fragment in url:
                if isinstance(response, Exception):
                    raise response
                return response
        raise TransportError("no fake response", status=404)


def _fred(series_id: str, rows: list[tuple[str, str]]) -> bytes:
    body = "".join(f"{day},{value}\n" for day, value in rows)
    return f"observation_date,{series_id}\n{body}".encode()


def _clock(start: datetime) -> Callable[[], datetime]:
    state = {"now": start}

    def now() -> datetime:
        current = state["now"]
        state["now"] = current + timedelta(seconds=1)
        return current

    return now


def _specs(*keys: str):  # type: ignore[no-untyped-def]
    return tuple(series_by_key(key) for key in keys)


def test_fetch_writes_raw_bytes_with_digest_url_and_fetch_time(tmp_path: Path) -> None:
    payload = _fred("DGS10", [("2026-11-04", "4.90"), ("2026-11-05", "4.84")])
    store = CrossAssetStore(tmp_path)
    records = fetch_all(
        store,
        _specs("DGS10"),
        FakeTransport({"id=DGS10": payload}),
        now=_clock(T0),
        sleep=lambda _seconds: None,
    )
    (record,) = records
    assert record["status"] == "ok"
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    assert record["bytes"] == len(payload)
    assert record["url"].startswith("https://fred.stlouisfed.org/graph/fredgraph.csv")
    assert record["fetched_at"] == "2026-11-10T05:00:00.000000Z"
    assert record["rows_total"] == 2 and record["first_obs"] == "2026-11-04"
    assert (tmp_path / record["path"]).read_bytes() == payload
    assert store.read_manifest() == [record]
    assert store.verify_raw(record)


def test_identical_refetch_is_recorded_as_unchanged_without_new_raw_file(
    tmp_path: Path,
) -> None:
    payload = _fred("DGS10", [("2026-11-04", "4.90")])
    store = CrossAssetStore(tmp_path)
    transport = FakeTransport({"id=DGS10": payload})
    first = fetch_all(
        store, _specs("DGS10"), transport, now=_clock(T0), sleep=lambda _s: None
    )[0]
    second = fetch_all(
        store,
        _specs("DGS10"),
        transport,
        now=_clock(T0 + timedelta(days=1)),
        sleep=lambda _s: None,
    )[0]
    assert second["status"] == "unchanged"
    assert second["path"] == first["path"]
    assert second["sha256"] == first["sha256"]
    assert len(list((tmp_path / "raw").rglob("*.csv"))) == 1


def test_revised_refetch_keeps_both_vintages_and_flags_the_revision(
    tmp_path: Path,
) -> None:
    store = CrossAssetStore(tmp_path)
    first_payload = _fred("DGS10", [("2026-11-04", "4.90"), ("2026-11-05", "4.84")])
    second_payload = _fred(
        "DGS10",
        [("2026-11-04", "4.91"), ("2026-11-05", "4.84"), ("2026-11-06", "4.86")],
    )
    fetch_all(
        store,
        _specs("DGS10"),
        FakeTransport({"id=DGS10": first_payload}),
        now=_clock(T0),
        sleep=lambda _s: None,
    )
    second = fetch_all(
        store,
        _specs("DGS10"),
        FakeTransport({"id=DGS10": second_payload}),
        now=_clock(T0 + timedelta(days=1)),
        sleep=lambda _s: None,
    )[0]
    assert second["status"] == "ok"
    assert second["revision"] == {
        "added": 1,
        "changed": 1,
        "dropped": 0,
        "first_changed": "2026-11-04",
    }
    assert len(list((tmp_path / "raw").rglob("*.csv"))) == 2
    assert len(store.read_manifest()) == 2


def test_blocked_source_is_recorded_once_and_remaining_series_are_skipped(
    tmp_path: Path,
) -> None:
    store = CrossAssetStore(tmp_path)
    transport = FakeTransport({"stooq.com": CHALLENGE})
    stooq = [spec for spec in CATALOG if spec.source == "stooq"][:3]
    records = fetch_all(
        store, tuple(stooq), transport, now=_clock(T0), sleep=lambda _s: None
    )
    assert [r["status"] for r in records] == ["unavailable"] * 3
    assert len(transport.requested) == 1
    assert "bot" in records[0]["reason"].lower()
    assert "skipped" in records[1]["reason"].lower()
    assert not (tmp_path / "raw").exists() or not list((tmp_path / "raw").rglob("*.*"))


def test_transport_and_parse_failures_are_isolated_per_series(tmp_path: Path) -> None:
    store = CrossAssetStore(tmp_path)
    good = _fred("DGS2", [("2026-11-04", "4.5")])
    transport = FakeTransport(
        {
            "id=DGS10": TransportError("boom", status=500),
            "id=T10Y2Y": b"garbage",
            "id=DGS2": good,
        }
    )
    records = fetch_all(
        store,
        _specs("DGS10", "T10Y2Y", "DGS2"),
        transport,
        now=_clock(T0),
        sleep=lambda _s: None,
    )
    assert [r["status"] for r in records] == ["error", "error", "ok"]
    assert records[0]["reason"].startswith("transport")
    assert records[1]["reason"].startswith("parse")


@pytest.mark.parametrize("linked_level", ("root", "above_root"))
def test_manifest_append_refuses_symlinked_store_ancestor(
    tmp_path: Path, linked_level: str
) -> None:
    """Even error/unavailable records must not redirect manifest writes."""
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "linked"
    if linked_level == "root":
        root = alias
        foreign = outside / "manifest.jsonl"
    else:
        root = alias / "store"
        (outside / "store").mkdir()
        foreign = outside / "store" / "manifest.jsonl"
    try:
        alias.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable")

    with pytest.raises(ValueError, match="symlink"):
        CrossAssetStore(root).append({"status": "unavailable"})
    assert alias.is_symlink()
    assert not foreign.exists()


def test_manifest_append_refuses_symlinked_leaf_without_mutating_foreign_file(
    tmp_path: Path,
) -> None:
    store = CrossAssetStore(tmp_path / "store")
    store.root.mkdir()
    outside = tmp_path / "outside-manifest"
    outside.write_bytes(b"foreign-file-must-remain-unchanged\\n")
    try:
        store.manifest_path.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks unavailable")

    with pytest.raises(ValueError, match="symlink"):
        store.append({"status": "unavailable"})
    assert store.manifest_path.is_symlink()
    assert outside.read_bytes() == b"foreign-file-must-remain-unchanged\\n"


def test_manifest_reader_refuses_symlinked_leaf_and_ancestor(tmp_path: Path) -> None:
    outside = tmp_path / "foreign"
    outside.mkdir()
    payload = (
        b'{"schema":"cross_asset_manifest_v1","key":"DGS10",'
        b'"fetched_at":"2026-11-10T05:00:00Z","status":"error"}\\n'
    )
    (outside / "manifest.jsonl").write_bytes(payload)
    root = tmp_path / "store"
    root.mkdir()
    try:
        (root / "manifest.jsonl").symlink_to(outside / "manifest.jsonl")
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks unavailable")
    with pytest.raises(ValueError, match="symlink"):
        CrossAssetStore(root).read_manifest()

    (root / "manifest.jsonl").unlink()
    root.rmdir()
    root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        CrossAssetStore(root).read_manifest()
    assert (outside / "manifest.jsonl").read_bytes() == payload


def test_manifest_append_refuses_root_swapped_after_mkdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The post-mkdir check must run before opening the append destination."""
    root = tmp_path / "store"
    outside = tmp_path / "external"
    outside.mkdir()
    parked = tmp_path / "parked-root"
    original_mkdir = Path.mkdir
    swapped = False

    def swap_on_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal swapped
        original_mkdir(path, *args, **kwargs)  # type: ignore[arg-type]
        if path == root and not swapped:
            root.rename(parked)
            try:
                root.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                pytest.skip("directory symlinks unavailable")
            swapped = True

    monkeypatch.setattr(Path, "mkdir", swap_on_mkdir)
    with pytest.raises(ValueError, match="symlink"):
        CrossAssetStore(root).append({"status": "unavailable"})
    assert swapped
    assert root.is_symlink()
    assert not (outside / "manifest.jsonl").exists()


def test_manifest_is_append_only_and_verified_on_read(tmp_path: Path) -> None:
    store = CrossAssetStore(tmp_path)
    payload = _fred("DGS10", [("2026-11-04", "4.90")])
    record = fetch_all(
        store,
        _specs("DGS10"),
        FakeTransport({"id=DGS10": payload}),
        now=_clock(T0),
        sleep=lambda _s: None,
    )[0]
    raw_path = tmp_path / record["path"]
    raw_path.write_bytes(payload + b"tamper\n")
    assert not store.verify_raw(record)
    (tmp_path / "manifest.jsonl").write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        store.read_manifest()


def test_latest_records_honours_the_as_of_vintage_cutoff(tmp_path: Path) -> None:
    store = CrossAssetStore(tmp_path)
    for index, value in enumerate(("4.90", "4.95")):
        fetch_all(
            store,
            _specs("DGS10"),
            FakeTransport({"id=DGS10": _fred("DGS10", [("2026-11-04", value)])}),
            now=_clock(T0 + timedelta(days=index)),
            sleep=lambda _s: None,
        )
    records = store.read_manifest()
    assert (
        latest_records(records)["DGS10"]["fetched_at"] == "2026-11-11T05:00:00.000000Z"
    )
    early = latest_records(records, as_of=T0 + timedelta(hours=1))
    assert early["DGS10"]["fetched_at"] == "2026-11-10T05:00:00.000000Z"
    assert latest_records(records, as_of=T0 - timedelta(days=1)) == {}
    assert manifest_digest(early) != manifest_digest(latest_records(records))


# Two actual, distinct SHA-256 digests with the same 32-bit filename prefix.
# The test verifies the collision instead of mocking or weakening SHA-256.
_COLLIDING_RAW = (
    b"synthetic cross-asset archive v1 #1223",
    b"synthetic cross-asset archive v1 #63106",
)


def _colliding_raw_target(store: CrossAssetStore) -> Path:
    first_digest = hashlib.sha256(_COLLIDING_RAW[0]).hexdigest()
    other_digest = hashlib.sha256(_COLLIDING_RAW[1]).hexdigest()
    assert first_digest != other_digest
    assert first_digest[:8] == other_digest[:8]
    stamp = format_capture_utc(T0).replace("-", "").replace(":", "")
    return store.root / f"raw/fred/DGS10/{stamp}_{first_digest[:8]}.csv"


def test_raw_publish_cannot_overwrite_a_late_colliding_rival(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-stamp SHA-prefix collision must not replace another vintage."""
    store = CrossAssetStore(tmp_path)
    target = _colliding_raw_target(store)
    real_link = os.link
    injected = False

    def inject_rival(staging: str | Path, destination: str | Path) -> None:
        nonlocal injected
        if Path(destination) == target:
            assert not injected
            injected = True
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_COLLIDING_RAW[0])
        real_link(staging, destination)

    monkeypatch.setattr(os, "link", inject_rival)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        store.write_raw("fred", "DGS10", T0, _COLLIDING_RAW[1], extension="csv")
    assert injected
    assert target.read_bytes() == _COLLIDING_RAW[0]
    assert not list(target.parent.glob(f".{target.name}.part-*"))


def test_two_colliding_raw_publishers_return_only_the_actual_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loser must fail, retaining exact winner bytes and no shared staging."""
    store = CrossAssetStore(tmp_path)
    target = _colliding_raw_target(store)
    real_link = os.link
    link_barrier = Barrier(2)
    stages: list[str | Path] = []

    def synchronized_link(staging: str | Path, destination: str | Path) -> None:
        assert Path(destination) == target
        stages.append(staging)
        link_barrier.wait(timeout=10)
        real_link(staging, destination)

    monkeypatch.setattr(os, "link", synchronized_link)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(store.write_raw, "fred", "DGS10", T0, raw, extension="csv")
            for raw in _COLLIDING_RAW
        ]
        outcomes: list[tuple[str, str]] = []
        failures: list[ValueError] = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=15))
            except ValueError as error:
                failures.append(error)
    assert len(stages) == len(set(map(str, stages))) == 2
    assert len(outcomes) == len(failures) == 1
    winner_payload = target.read_bytes()
    assert winner_payload in _COLLIDING_RAW
    assert outcomes[0][0] == target.relative_to(tmp_path).as_posix()
    assert outcomes[0][1] == hashlib.sha256(winner_payload).hexdigest()
    assert "refusing to overwrite" in str(failures[0])
    assert not list(target.parent.glob(f".{target.name}.part-*"))


def test_two_identical_raw_publishers_both_reuse_the_exclusive_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two same-byte writers must both succeed without a second publication."""

    store = CrossAssetStore(tmp_path)
    target = _colliding_raw_target(store)
    real_link = os.link
    barrier = Barrier(2)

    def concurrent_link(staging: str | Path, destination: str | Path) -> None:
        # Do not capture unrelated links from other test or runtime threads.
        if Path(destination) != target:
            return real_link(staging, destination)
        barrier.wait(timeout=10)
        real_link(staging, destination)

    monkeypatch.setattr(os, "link", concurrent_link)
    with ThreadPoolExecutor(max_workers=2) as pool:
        requests = [
            pool.submit(
                store.write_raw, "fred", "DGS10", T0, _COLLIDING_RAW[0], extension="csv"
            )
            for _ in range(2)
        ]
        results = [request.result(timeout=15) for request in requests]
    assert len(results) == 2
    assert results[0] == results[1]
    assert target.read_bytes() == _COLLIDING_RAW[0]
    assert results[0][1] == hashlib.sha256(_COLLIDING_RAW[0]).hexdigest()
    assert not list(target.parent.glob(f".{target.name}.part-*"))


@pytest.mark.parametrize(
    "unsupported_errno",
    sorted(
        {
            getattr(errno, name)
            for name in ("EINVAL", "ENOTSUP", "EOPNOTSUPP")
            if hasattr(errno, name)
        }
    ),
)
def test_raw_write_survives_unsupported_directory_fsync_without_claiming_durability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsupported_errno: int
) -> None:
    """Some POSIX filesystems reject directory fsync, not file fsync."""

    if os.name != "posix":
        pytest.skip("directory fsync specific to POSIX")
    store = CrossAssetStore(tmp_path)
    target = _colliding_raw_target(store)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_dir = target.parent.stat()
    real_fsync = os.fsync
    directory_attempts = 0

    def reject_directory_fsync(fd: int) -> None:
        nonlocal directory_attempts
        opened = os.fstat(fd)
        if stat.S_ISDIR(opened.st_mode) and (opened.st_dev, opened.st_ino) == (
            expected_dir.st_dev,
            expected_dir.st_ino,
        ):
            directory_attempts += 1
            raise OSError(unsupported_errno, "directory fsync unsupported")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", reject_directory_fsync)
    relative, digest = store.write_raw(
        "fred", "DGS10", T0, _COLLIDING_RAW[0], extension="csv"
    )
    assert directory_attempts == 1
    assert (tmp_path / relative).read_bytes() == _COLLIDING_RAW[0]
    assert digest == hashlib.sha256(_COLLIDING_RAW[0]).hexdigest()


def test_raw_write_propagates_unexpected_directory_fsync_eio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexpected I/O error is not silently treated as unsupported fsync."""

    if os.name != "posix":
        pytest.skip("directory fsync specific to POSIX")
    store = CrossAssetStore(tmp_path)
    target = _colliding_raw_target(store)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_dir = target.parent.stat()
    real_fsync = os.fsync
    seen = 0

    def fail_directory_fsync(fd: int) -> None:
        nonlocal seen
        opened = os.fstat(fd)
        if stat.S_ISDIR(opened.st_mode) and (opened.st_dev, opened.st_ino) == (
            expected_dir.st_dev,
            expected_dir.st_ino,
        ):
            seen += 1
            raise OSError(errno.EIO, "persistent directory I/O failure")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)
    with pytest.raises(OSError, match="persistent directory I/O failure") as failure:
        store.write_raw("fred", "DGS10", T0, _COLLIDING_RAW[0], extension="csv")
    assert failure.value.errno == errno.EIO
    assert seen == 1
    # Exclusive publication already succeeded; it must remain unmodified, but
    # the failed call must not report normal success or invent durability.
    assert target.read_bytes() == _COLLIDING_RAW[0]
    assert not list(target.parent.glob(f".{target.name}.part-*"))


def test_raw_write_rejects_preexisting_symlink_even_if_bytes_match(
    tmp_path: Path,
) -> None:
    store = CrossAssetStore(tmp_path)
    target = _colliding_raw_target(store)
    outside = tmp_path / "external-evidence.csv"
    outside.write_bytes(_COLLIDING_RAW[0])
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this filesystem")
    with pytest.raises(ValueError, match="raw"):
        store.write_raw("fred", "DGS10", T0, _COLLIDING_RAW[0], extension="csv")
    assert target.is_symlink()
    assert outside.read_bytes() == _COLLIDING_RAW[0]


@pytest.mark.parametrize("linked_level", ["raw", "root", "above_root"])
def test_raw_write_refuses_preexisting_symlinked_parent_before_publication(
    tmp_path: Path, linked_level: str
) -> None:
    """A static ancestor symlink must never redirect a new raw write outside root."""

    outside = tmp_path / "outside"
    outside.mkdir()
    if linked_level == "raw":
        root = tmp_path / "store"
        root.mkdir()
        alias = root / "raw"
        dest = outside / "fred" / "DGS10"
    elif linked_level == "root":
        root = tmp_path / "store"
        alias = root
        dest = outside / "raw" / "fred" / "DGS10"
    else:
        alias = tmp_path / "linked-parent"
        root = alias / "store"
        (outside / "store").mkdir()
        dest = outside / "store" / "raw" / "fred" / "DGS10"
    try:
        alias.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable")
    assert alias.is_symlink()
    assert not dest.exists()
    store = CrossAssetStore(root)
    with pytest.raises(ValueError, match="parent.*symlink|symlink.*parent"):
        store.write_raw("fred", "DGS10", T0, _COLLIDING_RAW[0], extension="csv")
    assert not dest.exists()
    assert not list(outside.rglob("*.part-*"))
    assert alias.is_symlink()


def test_raw_write_rechecks_a_directory_replaced_during_mkdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlink planted at the mkdir boundary is refused before staging."""

    root = tmp_path / "store"
    outside = tmp_path / "outside"
    outside.mkdir()
    target = _colliding_raw_target(CrossAssetStore(root))
    real_mkdir = Path.mkdir
    replaced = False

    def replace_after_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal replaced
        real_mkdir(path, *args, **kwargs)
        if path == target.parent and not replaced:
            replaced = True
            # The exact existing directory is moved away, and a static link
            # appears before write_raw may create its privately staged file.
            moved = tmp_path / "original-raw-tree"
            (root / "raw").rename(moved)
            (root / "raw").symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(Path, "mkdir", replace_after_mkdir)
    with pytest.raises(ValueError, match="parent.*symlink"):
        CrossAssetStore(root).write_raw(
            "fred", "DGS10", T0, _COLLIDING_RAW[0], extension="csv"
        )
    assert replaced
    assert (root / "raw").is_symlink()
    assert not list(outside.rglob("*"))
    assert not list((tmp_path / "original-raw-tree").rglob("*.part-*"))


def test_raw_write_fails_closed_when_exclusive_links_are_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CrossAssetStore(tmp_path)
    target = _colliding_raw_target(store)

    def links_unavailable(_source: str | Path, _target: str | Path) -> None:
        raise OSError(errno.ENOTSUP, "exclusive links unavailable")

    monkeypatch.setattr(os, "link", links_unavailable)
    with pytest.raises(OSError, match="exclusive links unavailable"):
        store.write_raw("fred", "DGS10", T0, _COLLIDING_RAW[0], extension="csv")
    assert not target.exists()
    assert not list(target.parent.glob(f".{target.name}.part-*"))


def test_exact_repeated_raw_write_reuses_verified_bytes_without_a_new_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equal full digests are idempotent, including the original raw path."""
    store = CrossAssetStore(tmp_path)
    first = store.write_raw("fred", "DGS10", T0, _COLLIDING_RAW[0], extension="csv")

    def unexpected_second_link(_source: str | Path, _target: str | Path) -> None:
        raise AssertionError("identical raw capture must not republish")

    monkeypatch.setattr(os, "link", unexpected_second_link)
    assert (
        store.write_raw("fred", "DGS10", T0, _COLLIDING_RAW[0], extension="csv")
        == first
    )
    assert (tmp_path / first[0]).read_bytes() == _COLLIDING_RAW[0]


def test_unsafe_series_keys_cannot_escape_the_raw_directory(tmp_path: Path) -> None:
    store = CrossAssetStore(tmp_path)
    with pytest.raises(ValueError, match="key"):
        store.write_raw("fred", "../evil", T0, b"x", extension="csv")


def test_urllib_transport_refuses_non_https_urls() -> None:
    with pytest.raises(TransportError, match="https"):
        UrllibTransport().get("http://example.com/x.csv")
    with pytest.raises(TransportError, match="https"):
        UrllibTransport().get("file:///etc/passwd")


def test_manifest_lines_are_canonical_json(tmp_path: Path) -> None:
    store = CrossAssetStore(tmp_path)
    fetch_all(
        store,
        _specs("DGS10"),
        FakeTransport({"id=DGS10": _fred("DGS10", [("2026-11-04", "4.90")])}),
        now=_clock(T0),
        sleep=lambda _s: None,
    )
    line = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert json.dumps(record, sort_keys=True, separators=(",", ":")) == line


def test_unchanged_recheck_keeps_the_original_capture_time(tmp_path: Path) -> None:
    store = CrossAssetStore(tmp_path)
    payload = _fred("DGS10", [("2026-11-04", "4.90")])
    transport = FakeTransport({"id=DGS10": payload})
    first = fetch_all(
        store, _specs("DGS10"), transport, now=_clock(T0), sleep=lambda _s: None
    )[0]
    later = T0 + timedelta(days=2)
    second = fetch_all(
        store, _specs("DGS10"), transport, now=_clock(later), sleep=lambda _s: None
    )[0]
    assert second["status"] == "unchanged"
    assert second["fetched_at"] == "2026-11-12T05:00:00.000000Z"  # the re-check itself
    assert (
        second["captured_at"] == first["captured_at"] == "2026-11-10T05:00:00.000000Z"
    )
    records = store.read_manifest()
    # The vintage was captured at T0: a cutoff between capture and re-check
    # still selects it, a cutoff before the capture does not.
    between = latest_records(records, as_of=T0 + timedelta(hours=1))
    assert between["DGS10"]["captured_at"] == "2026-11-10T05:00:00.000000Z"
    assert latest_records(records, as_of=T0 - timedelta(seconds=1)) == {}
    assert manifest_digest(between) == manifest_digest(latest_records(records))
