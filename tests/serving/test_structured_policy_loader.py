from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.selection import PolicyMode
from trade_rl.rl.actions import ACTION_SCHEMA
from trade_rl.rl.sequence_observations import SEQUENCE_OBSERVATION_SCHEMA
from trade_rl.rl.structured_export import (
    STRUCTURED_EXPORT_MANIFEST_NAME,
    STRUCTURED_EXPORT_MODEL_NAME,
    StructuredExportManifest,
    StructuredInputSpec,
    canonical_structured_observation_keys,
)
from trade_rl.serving.bundle import (
    ServingBundle,
    ServingBundleManifest,
    write_serving_bundle_manifest,
)
from trade_rl.serving.policy_loader import (
    STRUCTURED_POLICY_LOADER_NAME,
    StructuredTorchScriptEnsembleLoader,
    build_structured_policy_loader_payload,
    write_structured_policy_loader_manifest,
)
from trade_rl.serving.runtime import RuntimeIdentityContract, ServingRuntime


class _ConstantActor(nn.Module):
    def __init__(self, value: float, action_size: int) -> None:
        super().__init__()
        self.register_buffer("action", torch.full((action_size,), value))

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.action.unsqueeze(0).expand(inputs[0].shape[0], -1)


def _test_policy_identity(
    architecture: str, *, action_size: int = 2
) -> dict[str, object]:
    sequence_architecture = {
        "action_names": tuple(f"target_weight:{index}" for index in range(action_size)),
        "schema_version": "test_sequence_architecture_v1",
        "symbols": tuple(str(index) for index in range(action_size)),
        "test_architecture_marker": architecture,
    }
    sequence_digest = content_digest(sequence_architecture)
    current_weight = {
        "bounds": (-1.0, 1.0),
        "dtype": "float32",
        "key": "current_weights",
        "observation_schema": "native_timeframe_sequence_observation_v3",
        "shape": (action_size,),
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
        "schema_version": "hierarchical_gate_target_policy_v2",
        "sequence_architecture_digest": sequence_digest,
    }
    return {
        "actor_head": "hierarchical_gate_target_v1",
        "current_weight_observation": current_weight,
        "exploration_contract": exploration_contract,
        "gate_temperature": 1.0,
        "observation_encoder": "hierarchical_sequence_v2",
        "policy_architecture_digest": content_digest(policy_architecture),
        "schema_version": "sb3_policy_identity_v3",
        "sequence_architecture": sequence_architecture,
        "sequence_architecture_digest": sequence_digest,
    }


def _member(root: Path, index: int, value: float, architecture: str) -> None:
    directory = root / "members" / f"member-{index:03d}"
    directory.mkdir(parents=True)
    specs = tuple(
        StructuredInputSpec(name=name, shape=(1,), dtype="float32")
        for name in canonical_structured_observation_keys()
    )
    example = tuple(torch.zeros((1, 1), dtype=torch.float32) for _ in specs)
    model_path = directory / STRUCTURED_EXPORT_MODEL_NAME
    torch.jit.trace(_ConstantActor(value, 2), example, strict=False).save(
        str(model_path)
    )
    identity = _test_policy_identity(architecture, action_size=2)
    manifest = StructuredExportManifest.build(
        model_path=model_path,
        policy_identity=identity,
        inputs=specs,
        action_size=2,
        tolerance=1e-5,
        max_abs_error=0.0,
    )
    (directory / STRUCTURED_EXPORT_MANIFEST_NAME).write_bytes(
        canonical_json_bytes(asdict(manifest))
    )


def _bundle(root: Path, *, architecture: str) -> ServingBundle:
    action_names = ("target_weight:BTC", "target_weight:ETH")
    paths = tuple(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "bundle.json"
    )
    manifest = ServingBundleManifest.build(
        root=root,
        dataset_id="d" * 64,
        action_schema=ACTION_SCHEMA,
        observation_schema=SEQUENCE_OBSERVATION_SCHEMA,
        observation_size=1,
        environment_digest="e" * 64,
        initial_capital=1_000.0,
        policy_mode=PolicyMode.RESIDUAL_POLICY,
        policy_digest="b" * 64,
        signal_digest="c" * 64,
        selection_digest="d" * 64,
        artifact_paths=paths,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        action_size=2,
        action_names=action_names,
        action_spec_digest="a" * 64,
        architecture_digest=_test_policy_identity(architecture)[
            "policy_architecture_digest"
        ],
        training_run_digest="6" * 64,
        run_kind="research_selected_final",
        selection_proposal_digest="1" * 64,
        selection_authorization_digest="2" * 64,
        walk_forward_run_digest="3" * 64,
        gate_evidence_digest="4" * 64,
        confirmation_evidence_digest="5" * 64,
    )
    write_serving_bundle_manifest(root, manifest)
    return ServingBundle(root=root, manifest=manifest)


def _observation() -> dict[str, np.ndarray]:
    return {
        name: np.zeros((1,), dtype=np.float32)
        for name in canonical_structured_observation_keys()
    }


def test_structured_loader_builds_sb3_independent_mean_ensemble(
    tmp_path: Path,
) -> None:
    architecture = "f" * 64
    _member(tmp_path, 0, 0.2, architecture)
    _member(tmp_path, 1, 0.6, architecture)
    write_structured_policy_loader_manifest(tmp_path, expected_members=2)
    bundle = _bundle(tmp_path, architecture=architecture)

    policy = StructuredTorchScriptEnsembleLoader(
        expected_architecture_digest=_test_policy_identity(architecture)[
            "policy_architecture_digest"
        ]
    ).load(bundle)

    np.testing.assert_allclose(policy.predict(_observation()), [0.4, 0.4])


def test_structured_loader_builder_rejects_member_architecture_drift(
    tmp_path: Path,
) -> None:
    _member(tmp_path, 0, 0.2, "a" * 64)
    _member(tmp_path, 1, 0.6, "b" * 64)

    with pytest.raises(ValueError, match="inconsistent identity"):
        build_structured_policy_loader_payload(tmp_path, expected_members=2)


def test_runtime_constructs_canonical_structured_loader(tmp_path: Path) -> None:
    architecture = "f" * 64
    _member(tmp_path, 0, 0.25, architecture)
    write_structured_policy_loader_manifest(tmp_path, expected_members=1)
    _bundle(tmp_path, architecture=architecture)
    runtime = ServingRuntime(
        allow_unreleased=True,
        identity_contract=RuntimeIdentityContract(
            environment_digest="e" * 64,
            action_names=("target_weight:BTC", "target_weight:ETH"),
            action_spec_digest="a" * 64,
            normalizer_digest=None,
            architecture_digest=_test_policy_identity(architecture)[
                "policy_architecture_digest"
            ],
        ),
    )

    snapshot = runtime.activate(tmp_path)

    assert (
        snapshot.architecture_digest
        == _test_policy_identity(architecture)["policy_architecture_digest"]
    )
    np.testing.assert_allclose(runtime.predict(_observation()), [0.25, 0.25])


def test_loader_manifest_digest_binds_members(tmp_path: Path) -> None:
    architecture = "f" * 64
    _member(tmp_path, 0, 0.25, architecture)
    payload = build_structured_policy_loader_payload(tmp_path, expected_members=1)

    assert (
        payload["architecture_digest"]
        == _test_policy_identity(architecture)["policy_architecture_digest"]
    )
    assert payload["digest"] == content_digest(
        {
            "action_size": payload["action_size"],
            "architecture_digest": payload["architecture_digest"],
            "members": payload["members"],
            "schema_version": payload["schema_version"],
        }
    )
    assert STRUCTURED_POLICY_LOADER_NAME == "structured-policy-loader.json"
