from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.serving.helpers import create_bundle
from trade_rl.serving.registry import ServingRegistry


def test_registry_activates_released_validated_bundle_atomically(
    tmp_path: Path,
) -> None:
    source = create_bundle(tmp_path / "source")
    registry = ServingRegistry(tmp_path / "registry")

    active = registry.activate(source)

    assert (
        active.manifest.bundle_digest == registry.active_bundle().manifest.bundle_digest
    )
    pointer = json.loads(
        (tmp_path / "registry" / "active.json").read_text(encoding="utf-8")
    )
    assert pointer["bundle_digest"] == active.manifest.bundle_digest


def test_registry_rejects_unreleased_bundle_by_default(tmp_path: Path) -> None:
    source = create_bundle(tmp_path / "source", release_digest=None)
    registry = ServingRegistry(tmp_path / "registry")

    with pytest.raises(ValueError, match="release"):
        registry.activate(source)

    assert not (tmp_path / "registry" / "active.json").exists()


def test_research_registry_can_explicitly_allow_unreleased_bundle(
    tmp_path: Path,
) -> None:
    source = create_bundle(tmp_path / "source", release_digest=None)
    registry = ServingRegistry(tmp_path / "registry", allow_unreleased=True)

    active = registry.activate(source)

    assert active.manifest.release_digest is None


def test_failed_activation_preserves_previous_active_bundle(tmp_path: Path) -> None:
    valid = create_bundle(tmp_path / "valid")
    invalid = create_bundle(tmp_path / "invalid")
    registry = ServingRegistry(tmp_path / "registry")
    first = registry.activate(valid)
    (invalid / "dataset.json").write_text('{"tampered":true}', encoding="utf-8")

    with pytest.raises(ValueError, match="mismatch"):
        registry.activate(invalid)

    assert (
        registry.active_bundle().manifest.bundle_digest == first.manifest.bundle_digest
    )
