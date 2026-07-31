from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.sequence_observations import SEQUENCE_OBSERVATION_SCHEMA
from trade_rl.rl.structured_export import (
    STRUCTURED_EXPORT_MANIFEST_NAME,
    STRUCTURED_EXPORT_MODEL_NAME,
    StructuredExportManifest,
    StructuredInputSpec,
    canonical_structured_observation_keys,
    load_structured_export_manifest_bytes,
)
from trade_rl.serving.structured_policy import CanonicalStructuredPolicyLoader


def _manifest(*, action_size: int = 3) -> StructuredExportManifest:
    symbols = ("BTC", "ETH", "BNB")
    action_names = tuple(f"target_weight:{symbol}" for symbol in symbols)
    architecture = {
        "asset_identity_mode": "identity_free_v1",
        "d_model": 16,
        "n_symbols": 3,
        "schema_version": "hierarchical_sequence_policy_v4",
        "timeframes": ["15m", "1h", "4h", "1d"],
    }
    sequence_digest = content_digest(architecture)
    asset_binding = {
        "action_names": action_names,
        "n_symbols": 3,
        "schema_version": "sequence_asset_binding_v1",
        "symbols": symbols,
    }
    current_weight = {
        "bounds": (-1.0, 1.0),
        "dtype": "float32",
        "key": "current_weights",
        "observation_schema": SEQUENCE_OBSERVATION_SCHEMA,
        "shape": (3,),
        "source": "effective_book_weights",
    }
    exploration_contract = {
        "action_distribution": "masked_shared_squashed_diag_gaussian_v1",
        "change_intensity_coupling": "post_composition_gate_independent_v1",
        "log_std_parameterization": "shared_scalar_v1",
        "state_dependent_noise": False,
        "schema_version": "hierarchical_exploration_v1",
        "squashing": "tanh",
    }
    policy_architecture = {
        "actor_head": "hierarchical_gate_target_v1",
        "current_weight_observation": current_weight,
        "exploration_contract": exploration_contract,
        "gate_temperature": 1.0,
        "observation_encoder": "hierarchical_sequence_v2",
        "schema_version": "hierarchical_gate_target_policy_v3",
        "sequence_architecture_digest": sequence_digest,
    }
    identity = {
        "actor_head": "hierarchical_gate_target_v1",
        "asset_binding": asset_binding,
        "asset_binding_digest": content_digest(asset_binding),
        "current_weight_observation": current_weight,
        "exploration_contract": exploration_contract,
        "gate_temperature": 1.0,
        "observation_encoder": "hierarchical_sequence_v2",
        "policy_architecture_digest": content_digest(policy_architecture),
        "schema_version": "sb3_policy_identity_v4",
        "sequence_architecture": architecture,
        "sequence_architecture_digest": sequence_digest,
    }
    inputs = tuple(
        StructuredInputSpec(name=name, shape=(1,), dtype="float32")
        for name in canonical_structured_observation_keys()
    )
    payload = {
        "action_size": action_size,
        "architecture_digest": identity["policy_architecture_digest"],
        "inputs": inputs,
        "max_abs_error": 0.0,
        "model_digest": "a" * 64,
        "model_path": STRUCTURED_EXPORT_MODEL_NAME,
        "model_size_bytes": 1,
        "policy_identity": identity,
        "policy_identity_digest": content_digest(identity),
        "schema_version": "structured_policy_export_v2",
        "tolerance": 1e-5,
    }
    return StructuredExportManifest(digest=content_digest(payload), **payload)


def _bundle(root: Path, manifest_bytes: bytes) -> Any:
    model_path = root / STRUCTURED_EXPORT_MODEL_NAME
    model_path.write_bytes(b"x")
    return SimpleNamespace(
        root=root,
        manifest=SimpleNamespace(
            observation_schema=SEQUENCE_OBSERVATION_SCHEMA,
            files=(
                SimpleNamespace(
                    path=STRUCTURED_EXPORT_MANIFEST_NAME,
                    digest=hashlib.sha256(manifest_bytes).hexdigest(),
                    size_bytes=len(manifest_bytes),
                ),
                SimpleNamespace(
                    path=STRUCTURED_EXPORT_MODEL_NAME,
                    digest=hashlib.sha256(b"x").hexdigest(),
                    size_bytes=1,
                ),
            ),
        ),
    )


def test_structured_manifest_bytes_parser_requires_canonical_exact_bytes() -> None:
    manifest = _manifest()
    raw = canonical_json_bytes(manifest)

    assert load_structured_export_manifest_bytes(raw) == manifest
    with pytest.raises(ValueError, match="canonical"):
        load_structured_export_manifest_bytes(raw + b"\n")


def test_serving_loader_parses_the_exact_verified_manifest_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import trade_rl.serving.structured_policy as module
    from trade_rl.artifacts.verified_file import read_verified_bytes as real_read

    original = _manifest(action_size=3)
    replacement = _manifest(action_size=4)
    original_bytes = canonical_json_bytes(original)
    replacement_bytes = canonical_json_bytes(replacement)
    assert len(original_bytes) == len(replacement_bytes)
    manifest_path = tmp_path / STRUCTURED_EXPORT_MANIFEST_NAME
    manifest_path.write_bytes(original_bytes)
    bundle = _bundle(tmp_path, original_bytes)
    captured: list[StructuredExportManifest] = []

    def read_then_swap(path: Path, **kwargs: object) -> bytes:
        raw = real_read(path, **kwargs)
        if path == manifest_path:
            manifest_path.write_bytes(replacement_bytes)
        return raw

    class CapturePolicy:
        def __init__(self, *, root: Path, manifest: StructuredExportManifest) -> None:
            del root
            captured.append(manifest)

    monkeypatch.setattr(module, "read_verified_bytes", read_then_swap)
    monkeypatch.setattr(module, "StructuredTorchScriptPolicy", CapturePolicy)

    loader = CanonicalStructuredPolicyLoader(
        expected_architecture_digest=original.architecture_digest
    )
    loader.load(bundle)

    assert captured == [original]
