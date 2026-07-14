from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from tests.serving.helpers import create_bundle
import trade_rl.serving.bundle as bundle_module
from trade_rl.domain.selection import PolicyMode
from trade_rl.serving.bundle import BundleFile, ServingBundleManifest, load_serving_bundle

SHA = "a" * 64


@pytest.mark.parametrize("path", ["/absolute", "../escape", "bundle.json"])
def test_bundle_file_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValueError):
        BundleFile(path=path, digest=SHA, size_bytes=1)


def test_bundle_file_rejects_negative_size() -> None:
    with pytest.raises(ValueError, match="size_bytes"):
        BundleFile(path="artifact", digest=SHA, size_bytes=-1)


def test_bundle_manifest_rejects_invalid_identity_and_shape_contracts(
    tmp_path: Path,
) -> None:
    manifest = load_serving_bundle(create_bundle(tmp_path / "bundle")).manifest
    cases = (
        ({"observation_size": True}, "observation_size"),
        ({"observation_size": 0}, "observation_size"),
        ({"action_size": True}, "action_size"),
        ({"action_size": 0}, "action_size"),
        ({"action_names": ("only-one",)}, "action_names"),
        ({"action_names": ("x", "x", "z")}, "unique"),
        ({"action_names": ("", "y", "z")}, "non-empty"),
        ({"action_spec_digest": None}, "action_spec_digest"),
        ({"initial_capital": float("inf")}, "initial_capital"),
        ({"initial_capital": 0.0}, "initial_capital"),
        ({"policy_digest": "e" * 64}, "baseline_only"),
        (
            {"policy_mode": PolicyMode.RESIDUAL_POLICY, "policy_digest": None},
            "residual policy",
        ),
        ({"normalizer_digest": "bad"}, "normalizer_digest"),
        ({"files": ()}, "artifact files"),
        ({"files": (manifest.files[0], manifest.files[0])}, "unique"),
        ({"created_at": datetime(2026, 1, 1)}, "timezone-aware"),
        ({"schema_version": ""}, "schema_version"),
        ({"bundle_digest": "f" * 64}, "digest"),
    )
    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            replace(manifest, **changes)


def test_bundle_parsers_reject_malformed_values() -> None:
    with pytest.raises(ValueError, match="mapping"):
        bundle_module._mapping([], field="value")
    with pytest.raises(ValueError, match="string"):
        bundle_module._string(1, field="value")
    assert bundle_module._optional_string(None, field="value") is None
    with pytest.raises(ValueError, match="string"):
        bundle_module._optional_string(1, field="value")
    assert bundle_module._string_tuple(None, field="value") == ()
    with pytest.raises(ValueError, match="list of strings"):
        bundle_module._string_tuple(["ok", 1], field="value")
    with pytest.raises(ValueError, match="integer"):
        bundle_module._integer(True, field="value")
    with pytest.raises(ValueError, match="number"):
        bundle_module._number(True, field="value")
    with pytest.raises(ValueError, match="files must be a list"):
        bundle_module._parse_manifest({"files": {}})


def test_bundle_build_rejects_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ServingBundleManifest.build(
            root=tmp_path,
            dataset_id="a" * 64,
            action_schema="action",
            observation_schema="observation",
            observation_size=1,
            environment_digest="b" * 64,
            initial_capital=1.0,
            policy_mode=PolicyMode.BASELINE_ONLY,
            policy_digest=None,
            signal_digest="c" * 64,
            selection_digest="d" * 64,
            release_digest=None,
            artifact_paths=("missing",),
            created_at=datetime.now().astimezone(),
            action_size=1,
            action_names=("action",),
            action_spec_digest="e" * 64,
        )
