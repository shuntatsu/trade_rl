"""Exclusive file publication for the unchanged direct-simple stream."""

from __future__ import annotations

import errno
import importlib
import json
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest

from tests.strategies.test_simple_return_stream import fit, load_api, market
from trade_rl.artifacts import canonical_json_bytes, content_digest


def io_api():
    try:
        return importlib.import_module("trade_rl.strategies.forecasts.simple_stream_io")
    except ModuleNotFoundError as error:
        if error.name != "trade_rl.strategies.forecasts.simple_stream_io":
            raise
        pytest.fail(
            "Missing frozen simple-return stream filesystem boundary", pytrace=False
        )


@pytest.fixture
def stream():
    return fit(load_api())


def read(api, path, stream, **pins):
    return api.load_simple_return_stream_artifact(
        path,
        expected_digest=pins.get("expected_digest", stream.digest),
        expected_dataset_id=pins.get("expected_dataset_id", stream.dataset_id),
    )


def test_publication_retains_original_canonical_bytes_and_immutable_record(
    tmp_path, stream
):
    api = io_api()
    path = tmp_path / "nested" / "stream.json"
    record = api.publish_simple_return_stream_artifact(path, stream)
    expected = canonical_json_bytes(stream.payload())
    assert path.read_bytes() == expected
    assert sha256(expected).hexdigest() == stream.digest
    assert set(path.parent.iterdir()) == {path}
    assert record.path == path
    assert record.digest == stream.digest
    assert record.dataset_id == stream.dataset_id
    with pytest.raises(FrozenInstanceError):
        record.digest = "0" * 64
    restored = read(api, path, stream)
    assert restored.payload() == stream.payload()
    assert restored.digest == stream.digest
    assert restored.causal_scope_digest == stream.causal_scope_digest


@pytest.mark.parametrize("pin", ("expected_digest", "expected_dataset_id"))
def test_loading_requires_external_pins_and_rejects_drift(tmp_path, stream, pin):
    api = io_api()
    path = tmp_path / "stream.json"
    api.publish_simple_return_stream_artifact(path, stream)
    with pytest.raises(ValueError):
        read(api, path, stream, **{pin: "0" * 64})
    arguments = {
        "expected_digest": stream.digest,
        "expected_dataset_id": stream.dataset_id,
    }
    arguments.pop(pin)
    with pytest.raises(TypeError):
        api.load_simple_return_stream_artifact(path, **arguments)


def test_nonregular_input_is_rejected_before_opening(tmp_path, monkeypatch):
    api = io_api()
    path = tmp_path / "stationary-fifo"
    lstat = Path.lstat

    def fifo_metadata(self, *args, **kwargs):
        if self == path:
            return SimpleNamespace(st_mode=stat.S_IFIFO | 0o600)
        return lstat(self, *args, **kwargs)

    def forbidden_open(*_args, **_kwargs):
        pytest.fail("Nonregular input reached the potentially blocking opener")

    monkeypatch.setattr(Path, "lstat", fifo_metadata)
    monkeypatch.setattr(api, "open_regular_binary", forbidden_open)
    with pytest.raises(ValueError, match="regular file"):
        api.load_simple_return_stream_artifact(
            path, expected_digest="0" * 64, expected_dataset_id="1" * 64
        )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="Requires a POSIX FIFO")
def test_stationary_fifo_without_writer_is_rejected_without_blocking(tmp_path):
    path = tmp_path / "stationary-fifo"
    os.mkfifo(path)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from trade_rl.strategies.forecasts.simple_stream_io import "
            "load_simple_return_stream_artifact\n"
            "try:\n"
            "    load_simple_return_stream_artifact(sys.argv[1], "
            "expected_digest='0'*64, expected_dataset_id='1'*64)\n"
            "except ValueError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('FIFO was accepted')\n",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert child.returncode == 0, child.stderr


@pytest.mark.parametrize("target", ("file", "empty_directory", "occupied_directory"))
def test_existing_destination_is_never_replaced(tmp_path, stream, target):
    api = io_api()
    path = tmp_path / "stream.json"
    if target == "file":
        path.write_bytes(b"prior owner")
    else:
        path.mkdir()
        if target == "occupied_directory":
            (path / "retained.txt").write_bytes(b"prior owner")
    before = {
        p.relative_to(tmp_path).as_posix(): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    with pytest.raises(FileExistsError):
        api.publish_simple_return_stream_artifact(path, stream)
    assert {
        p.relative_to(tmp_path).as_posix(): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    } == before
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize("damage", ("truncate", "append"))
def test_loader_rejects_changed_file_bytes(tmp_path, stream, damage):
    api = io_api()
    path = tmp_path / "stream.json"
    api.publish_simple_return_stream_artifact(path, stream)
    original = path.read_bytes()
    path.write_bytes(original[:-1] if damage == "truncate" else original + b"\n")
    with pytest.raises(ValueError):
        read(api, path, stream)


def test_noncanonical_json_is_rejected_even_with_its_fresh_raw_hash(tmp_path, stream):
    api = io_api()
    path = tmp_path / "pretty.json"
    raw = json.dumps(stream.payload(), indent=2).encode("utf-8")
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        read(api, path, stream, expected_digest=sha256(raw).hexdigest())


def test_rehashed_packet_cannot_escape_existing_semantic_reader(tmp_path, stream):
    api = io_api()
    payload = stream.payload()
    packet = payload["packets"][0]
    packet["expected_simple_return"] += 0.1
    packet["digest"] = content_digest(
        {k: v for k, v in packet.items() if k != "digest"}
    )
    payload["causal_scope_digest"] = content_digest(
        {"vintages": payload["vintages"], "packets": payload["packets"]}
    )
    path = tmp_path / "stream.json"
    raw = canonical_json_bytes(payload)
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        read(api, path, stream, expected_digest=sha256(raw).hexdigest())


@pytest.mark.parametrize("stage", ("fsync", "link"))
def test_precommit_failure_removes_only_its_temporary_file(
    tmp_path, stream, monkeypatch, stage
):
    api = io_api()
    path = tmp_path / "stream.json"
    unrelated = tmp_path / ".other-owner.tmp"
    unrelated.write_bytes(b"keep")

    def fail(*_args, **_kwargs):
        raise OSError(errno.ENOTSUP, "injected unsupported filesystem")

    monkeypatch.setattr(api.os, stage, fail)
    with pytest.raises(OSError, match="injected unsupported filesystem"):
        api.publish_simple_return_stream_artifact(path, stream)
    assert not path.exists()
    assert set(tmp_path.iterdir()) == {unrelated}
    assert unrelated.read_bytes() == b"keep"


def test_partial_staging_write_failure_does_not_publish(tmp_path, stream, monkeypatch):
    api = io_api()
    path = tmp_path / "stream.json"
    unrelated = tmp_path / ".other-owner.tmp"
    unrelated.write_bytes(b"keep")
    fdopen = api.os.fdopen

    @contextmanager
    def partial_writer(descriptor, *args, **kwargs):
        with fdopen(descriptor, *args, **kwargs) as handle:

            class FailingWriter:
                def write(self, raw):
                    written = handle.write(raw[: len(raw) // 2])
                    handle.flush()
                    assert 0 < written < len(raw)
                    raise OSError("injected partial staging write failure")

            yield FailingWriter()

    monkeypatch.setattr(api.os, "fdopen", partial_writer)
    with pytest.raises(OSError, match="injected partial staging write failure"):
        api.publish_simple_return_stream_artifact(path, stream)
    assert not path.exists()
    assert set(tmp_path.iterdir()) == {unrelated}
    assert unrelated.read_bytes() == b"keep"


def test_staging_reader_failure_does_not_publish(tmp_path, stream, monkeypatch):
    api = io_api()
    path = tmp_path / "stream.json"
    unrelated = tmp_path / ".other-owner.tmp"
    unrelated.write_bytes(b"keep")
    loader = api.load_simple_return_stream_artifact

    def reject_staging(staging, **pins):
        assert loader(staging, **pins).digest == stream.digest
        raise ValueError("injected staging-reader validation failure")

    monkeypatch.setattr(api, "load_simple_return_stream_artifact", reject_staging)
    with pytest.raises(ValueError, match="injected staging-reader validation failure"):
        api.publish_simple_return_stream_artifact(path, stream)
    assert not path.exists()
    assert set(tmp_path.iterdir()) == {unrelated}
    assert unrelated.read_bytes() == b"keep"


def test_competing_publishers_commit_one_complete_stream(tmp_path, stream, monkeypatch):
    api = io_api()
    path = tmp_path / "stream.json"
    other = replace(stream, dataset_id="b" * 64)
    barrier = Barrier(2)
    link = api.os.link

    def race(source, destination, **kwargs):
        barrier.wait(timeout=10)
        return link(source, destination, **kwargs)

    monkeypatch.setattr(api.os, "link", race)
    outcomes = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [
            pool.submit(api.publish_simple_return_stream_artifact, path, s)
            for s in (stream, other)
        ]
        for job in jobs:
            try:
                outcomes.append(job.result(timeout=10))
            except FileExistsError:
                outcomes.append(None)
    winners = [record for record in outcomes if record is not None]
    assert len(winners) == 1
    winner = winners[0]
    assert set(tmp_path.iterdir()) == {path}
    assert sha256(path.read_bytes()).hexdigest() == winner.digest
    restored = api.load_simple_return_stream_artifact(
        path, expected_digest=winner.digest, expected_dataset_id=winner.dataset_id
    )
    assert restored.digest == winner.digest
    assert winner.digest in {stream.digest, other.digest}


def test_postcommit_cleanup_error_still_returns_successful_publication(
    tmp_path, stream, monkeypatch
):
    api = io_api()
    path = tmp_path / "stream.json"
    unlink = Path.unlink

    def fail_cleanup(self, *args, **kwargs):
        if self.parent == tmp_path and self != path:
            raise OSError("injected owned temporary cleanup failure")
        return unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    record = api.publish_simple_return_stream_artifact(path, stream)
    assert record.digest == stream.digest
    assert path.read_bytes() == canonical_json_bytes(stream.payload())
    assert read(api, path, stream).digest == stream.digest
    leftovers = set(tmp_path.iterdir()) - {path}
    assert len(leftovers) == 1
    assert next(iter(leftovers)).read_bytes() == path.read_bytes()


def test_unused_future_changes_keep_published_causal_identity(tmp_path):
    api = io_api()
    dataset = market()
    close = dataset.close.copy()
    close[14:] *= 11
    features = dataset.features.copy()
    features[14:] = -9999
    changed = replace(
        dataset,
        dataset_id="b" * 64,
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        features=features,
    )
    original, future = fit(load_api(), dataset), fit(load_api(), changed)
    original_path, future_path = tmp_path / "original.json", tmp_path / "future.json"
    api.publish_simple_return_stream_artifact(original_path, original)
    api.publish_simple_return_stream_artifact(future_path, future)
    a, b = read(api, original_path, original), read(api, future_path, future)
    assert a.causal_scope_digest == b.causal_scope_digest
    assert a.dataset_id != b.dataset_id
    assert a.digest != b.digest
    assert [p.digest for p in a.packets] == [p.digest for p in b.packets]
    assert a.packets[-1].horizon_end == np.datetime64("2026-02-01T14:00:00", "ns")


@pytest.mark.parametrize("target", ("valid_symlink", "broken_symlink"))
def test_final_symlinks_are_not_followed_or_replaced(tmp_path, stream, target):
    api = io_api()
    outside, alias = tmp_path / "outside.json", tmp_path / "alias.json"
    if target == "valid_symlink":
        api.publish_simple_return_stream_artifact(outside, stream)
    try:
        alias.symlink_to(outside)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows account does not have symlink permission")
        raise
    with pytest.raises(FileExistsError):
        api.publish_simple_return_stream_artifact(alias, stream)
    with pytest.raises(ValueError):
        read(api, alias, stream)
    assert alias.is_symlink()
    if target == "valid_symlink":
        assert outside.read_bytes() == canonical_json_bytes(stream.payload())
