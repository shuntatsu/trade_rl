"""Reject modified or foreign prepared inputs before Dataset deserialization."""

import importlib
from pathlib import Path

import pytest

from tools.tier_d_nonrl.tier_d_records import output_manifest, write_json


def _context_module(monkeypatch):
    module = importlib.import_module("tools.tier_d_scaling.context")
    monkeypatch.setattr(module, "validate_profile_source", lambda protocol: None)
    return module


def test_modified_preparation_is_rejected_before_dataset_loader(
    tmp_path: Path, monkeypatch
):
    module = _context_module(monkeypatch)
    root = tmp_path / "prepared"
    root.mkdir()
    (root / "data").write_bytes(b"original")
    write_json(root / "output-manifest.json", output_manifest(root))
    (root / "data").write_bytes(b"modified")

    def unexpected(*args, **kwargs):
        pytest.fail("must reject bytes before loading Dataset")

    monkeypatch.setattr(module, "load_market_dataset_artifact", unexpected)
    with pytest.raises(ValueError, match="manifest"):
        module.load_prepared(root)


@pytest.mark.parametrize("foreign", ["../outside", "C:/outside", "nested\\outside"])
def test_manifest_cannot_reference_external_paths(
    tmp_path: Path, monkeypatch, foreign: str
):
    module = _context_module(monkeypatch)
    write_json(
        tmp_path / "output-manifest.json", {foreign: {"bytes": 0, "sha256": "0" * 64}}
    )
    with pytest.raises(ValueError, match="manifest"):
        module.load_prepared(tmp_path)


def test_preparation_does_not_overwrite_any_existing_directory(
    tmp_path: Path, monkeypatch
):
    module = _context_module(monkeypatch)
    target = tmp_path / "already"
    target.mkdir()
    marker = target / "keep"
    marker.write_bytes(b"previous attempt")
    with pytest.raises(FileExistsError):
        module.prepare(target)
    assert marker.read_bytes() == b"previous attempt"


def test_resolved_directory_escape_is_rejected_before_content_hashes(
    tmp_path: Path, monkeypatch
):
    module = _context_module(monkeypatch)
    root = tmp_path / "prepared"
    root.mkdir()
    nested = root / "junction"
    nested.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    write_json(
        root / "output-manifest.json",
        {"junction/file": {"bytes": 0, "sha256": "a" * 64}},
    )
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        return outside if path == nested else original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)

    def unexpected(*args, **kwargs):
        pytest.fail("must check containment before hashing any content")

    monkeypatch.setattr(module, "output_manifest", unexpected)
    with pytest.raises(ValueError, match="containment"):
        module._check_manifest(root)
