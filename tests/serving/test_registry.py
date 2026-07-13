from __future__ import annotations

import json
from pathlib import Path

import pytest

from trade_rl.serving.registry import ServingRegistry
from tests.serving.test_bundle import create_baseline_bundle


def test_registry_activates_validated_bundle_atomically(tmp_path: Path) -> None:
    source = create_baseline_bundle(tmp_path / "source")
    registry = ServingRegistry(tmp_path / "registry")

    active = registry.activate(source)

    assert active.manifest.bundle_digest == registry.active_bundle().manifest.bundle_digest
    pointer = json.loads(
        (tmp_path / "registry" / "active.json").read_text(encoding="utf-8")
    )
    assert pointer["bundle_digest"] == active.manifest.bundle_digest


def test_failed_activation_preserves_previous_active_bundle(tmp_path: Path) -> None:
    valid = create_baseline_bundle(tmp_path / "valid")
    invalid = create_baseline_bundle(tmp_path / "invalid")
    registry = ServingRegistry(tmp_path / "registry")
    first = registry.activate(valid)
    (invalid / "dataset.json").write_text('{"tampered":true}', encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        registry.activate(invalid)

    assert registry.active_bundle().manifest.bundle_digest == first.manifest.bundle_digest
