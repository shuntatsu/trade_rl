from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import trade_rl.serving.registry as registry_module
import trade_rl.serving.runtime as runtime_module
from tests.serving.helpers import (
    ACTION_NAMES,
    OBSERVATION_SIZE,
    create_bundle,
    runtime_identity_contract,
)
from trade_rl.domain.selection import PolicyMode
from trade_rl.rl.actions import ACTION_SCHEMA
from trade_rl.rl.observations import OBSERVATION_SCHEMA
from trade_rl.serving.bundle import (
    ServingBundle,
    ServingBundleManifest,
    load_serving_bundle,
    write_serving_bundle_manifest,
)
from trade_rl.serving.registry import ServingRegistry
from trade_rl.serving.runtime import (
    LoadedPolicy,
    RuntimeIdentityContract,
    ServingRuntime,
)


class ConstantPolicy:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def predict(self, observation: np.ndarray) -> np.ndarray:
        del observation
        return self.value.copy()


class Loader:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def load(self, bundle: ServingBundle) -> LoadedPolicy:
        del bundle
        return ConstantPolicy(self.value)


def test_registry_platform_reuse_and_stale_stage_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_module.os, "name", "nt")
    registry_module._fsync_directory(tmp_path)
    monkeypatch.undo()

    source = create_bundle(tmp_path / "source")
    registry = ServingRegistry(tmp_path / "registry")
    first = registry.activate(source)
    second = registry.activate(source)
    assert first.manifest.bundle_digest == second.manifest.bundle_digest

    another = create_bundle(tmp_path / "another", environment_digest="e" * 64)
    digest = load_serving_bundle(another).manifest.bundle_digest
    stale = registry.staging_root / digest
    stale.mkdir()
    (stale / "partial").write_text("partial", encoding="utf-8")
    assert registry.activate(another).manifest.bundle_digest == digest
    assert not stale.exists()


def test_registry_detects_copy_and_installed_identity_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = create_bundle(tmp_path / "source")
    real = load_serving_bundle(source)
    fake = SimpleNamespace(
        manifest=SimpleNamespace(bundle_digest="f" * 64),
        release=real.release,
    )
    calls = iter((real, fake))
    monkeypatch.setattr(registry_module, "load_serving_bundle", lambda _: next(calls))
    registry = ServingRegistry(tmp_path / "registry")
    with pytest.raises(ValueError, match="changed during registry copy"):
        registry.activate(source)

    monkeypatch.setattr(registry_module, "load_serving_bundle", load_serving_bundle)
    other = create_bundle(tmp_path / "other")
    other_real = load_serving_bundle(other)
    registry = ServingRegistry(tmp_path / "registry-existing")
    (registry.versions_root / other_real.manifest.bundle_digest).mkdir()
    calls = iter((other_real, fake))
    monkeypatch.setattr(registry_module, "load_serving_bundle", lambda _: next(calls))
    with pytest.raises(ValueError, match="directory identity"):
        registry.activate(other)


def test_active_registry_pointer_rejects_malformed_and_mismatched_values(
    tmp_path: Path,
) -> None:
    registry = ServingRegistry(tmp_path / "registry", allow_unreleased=True)
    with pytest.raises(FileNotFoundError, match="no active"):
        registry.active_bundle()

    invalid_payloads = (
        ([], "mapping"),
        (
            {
                "bundle_digest": 1,
                "path": "versions/x",
                "schema": "serving_registry_pointer_v1",
            },
            "bundle_digest",
        ),
        (
            {
                "bundle_digest": "a" * 64,
                "path": 1,
                "schema": "serving_registry_pointer_v1",
            },
            "path",
        ),
        (
            {
                "bundle_digest": "a" * 64,
                "path": "versions/x",
                "schema": "bad",
            },
            "schema",
        ),
        (
            {
                "bundle_digest": "a" * 64,
                "path": "../outside",
                "schema": "serving_registry_pointer_v1",
            },
            "escapes",
        ),
    )
    for payload, message in invalid_payloads:
        registry.active_pointer.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            registry.active_bundle()

    source = create_bundle(tmp_path / "source")
    released = ServingRegistry(tmp_path / "released")
    released.activate(source)
    payload = json.loads(released.active_pointer.read_text(encoding="utf-8"))
    payload["bundle_digest"] = "f" * 64
    released.active_pointer.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        released.active_bundle()


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (
            {
                "environment_digest": "a" * 64,
                "action_names": (),
                "action_spec_digest": "b" * 64,
                "normalizer_digest": None,
            },
            "non-empty",
        ),
        (
            {
                "environment_digest": "a" * 64,
                "action_names": ("x", "x"),
                "action_spec_digest": "b" * 64,
                "normalizer_digest": None,
            },
            "unique",
        ),
        (
            {
                "environment_digest": "a" * 64,
                "action_names": ("x",),
                "action_spec_digest": "b" * 64,
                "normalizer_digest": "bad",
            },
            "normalizer_digest",
        ),
    ],
)
def test_runtime_identity_contract_rejects_invalid_fields(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RuntimeIdentityContract(**kwargs)  # type: ignore[arg-type]


def test_runtime_constructor_and_pre_activation_fail_closed() -> None:
    contract = runtime_identity_contract()
    with pytest.raises(ValueError, match="cannot be combined"):
        ServingRuntime(
            identity_contract=contract,
            expected_environment_digest="a" * 64,
        )
    with pytest.raises(ValueError, match="legacy serving identity"):
        ServingRuntime(expected_environment_digest="a" * 64)
    with pytest.raises(ValueError, match="boolean"):
        ServingRuntime(  # type: ignore[arg-type]
            identity_contract=contract,
            allow_unbound_identity=1,
        )

    legacy = ServingRuntime(
        expected_environment_digest=contract.environment_digest,
        expected_action_names=contract.action_names,
        expected_action_spec_digest=contract.action_spec_digest,
        expected_normalizer_digest=contract.normalizer_digest,
    )
    assert legacy.identity_contract == contract

    runtime = ServingRuntime(identity_contract=contract)
    with pytest.raises(RuntimeError, match="no active snapshot"):
        runtime.snapshot()
    with pytest.raises(ValueError, match="non-empty finite"):
        runtime.predict(np.array([]))
    with pytest.raises(ValueError, match="non-empty finite"):
        runtime.predict(np.array([np.nan]))
    with pytest.raises(RuntimeError, match="no active policy"):
        runtime.predict(np.zeros(OBSERVATION_SIZE))


def _schema_bundle(root: Path, action_schema: str, observation_schema: str) -> Path:
    create_bundle(root, release_digest=None)
    original = load_serving_bundle(root).manifest
    manifest = ServingBundleManifest.build(
        root=root,
        dataset_id=original.dataset_id,
        action_schema=action_schema,
        observation_schema=observation_schema,
        observation_size=original.observation_size,
        environment_digest=original.environment_digest,
        initial_capital=original.initial_capital,
        policy_mode=original.policy_mode,
        policy_digest=original.policy_digest,
        signal_digest=original.signal_digest,
        selection_digest=original.selection_digest,
        release_digest=None,
        artifact_paths=tuple(item.path for item in original.files),
        created_at=original.created_at,
        action_size=original.action_size,
        action_names=original.action_names,
        action_spec_digest=original.action_spec_digest,
        normalizer_digest=original.normalizer_digest,
    )
    write_serving_bundle_manifest(root, manifest)
    return root


def test_runtime_activation_rejects_release_schemas_loader_and_normalizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = runtime_identity_contract()
    with pytest.raises(ValueError, match="release"):
        ServingRuntime(identity_contract=contract).activate(
            create_bundle(tmp_path / "unreleased", release_digest=None)
        )

    action_root = _schema_bundle(
        tmp_path / "action-schema", "wrong-action", OBSERVATION_SCHEMA
    )
    with pytest.raises(ValueError, match="action schema"):
        ServingRuntime(identity_contract=contract, allow_unreleased=True).activate(
            action_root
        )

    observation_root = _schema_bundle(
        tmp_path / "observation-schema", ACTION_SCHEMA, "wrong-observation"
    )
    with pytest.raises(ValueError, match="observation schema"):
        ServingRuntime(identity_contract=contract, allow_unreleased=True).activate(
            observation_root
        )

    residual = create_bundle(
        tmp_path / "residual", policy_mode=PolicyMode.RESIDUAL_POLICY
    )
    with pytest.raises(RuntimeError, match="policy loader"):
        ServingRuntime(identity_contract=contract).activate(residual)

    root = create_bundle(tmp_path / "bundle")
    assert (
        ServingRuntime(allow_unbound_identity=True).activate(root).action_names
        == ACTION_NAMES
    )
    forced = ServingRuntime(allow_unbound_identity=True)
    forced.allow_unbound_identity = False
    with pytest.raises(RuntimeError, match="identity contract"):
        forced.activate(root)

    real = load_serving_bundle(root)
    fake = ServingBundle(
        root=real.root,
        manifest=real.manifest,
        release=real.release,
        normalizer=object(),
    )
    monkeypatch.setattr(runtime_module, "load_serving_bundle", lambda _: fake)
    with pytest.raises(ValueError, match="normalizer type"):
        ServingRuntime(identity_contract=contract).activate(root)


def test_runtime_predict_action_rejects_bad_inputs_and_outputs(tmp_path: Path) -> None:
    runtime = ServingRuntime(identity_contract=runtime_identity_contract())
    snapshot = runtime.activate(create_bundle(tmp_path / "baseline"))
    policy = ConstantPolicy(np.zeros(len(ACTION_NAMES), dtype=np.float32))

    for observation in (np.zeros(1), np.full(OBSERVATION_SIZE, np.nan)):
        with pytest.raises(ValueError, match="observation schema"):
            runtime._predict_action(policy, snapshot, None, observation)

    for action in (
        np.zeros(1),
        np.array([0.0, np.nan, 0.0]),
        np.array([0.0, 2.0, 0.0]),
    ):
        with pytest.raises(ValueError, match="action schema"):
            runtime._predict_action(
                ConstantPolicy(action), snapshot, None, np.zeros(OBSERVATION_SIZE)
            )
