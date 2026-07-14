from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.serving.helpers import (
    OBSERVATION_SIZE,
    create_bundle,
    runtime_identity_contract,
)
from trade_rl.domain.selection import PolicyMode
from trade_rl.serving.bundle import ServingBundle
from trade_rl.serving.runtime import LoadedPolicy, ServingRuntime


class RecordingPolicy:
    def __init__(self, action: np.ndarray) -> None:
        self.action = action
        self.observations: list[np.ndarray] = []

    def predict(self, observation: np.ndarray) -> np.ndarray:
        self.observations.append(np.asarray(observation).copy())
        return self.action.copy()


class Loader:
    def __init__(self, policy: RecordingPolicy) -> None:
        self.policy = policy

    def load(self, bundle: ServingBundle) -> LoadedPolicy:
        return self.policy


def test_runtime_applies_bundle_normalizer_before_policy(tmp_path: Path) -> None:
    policy = RecordingPolicy(np.zeros(3, dtype=np.float32))
    runtime = ServingRuntime(
        policy_loader=Loader(policy),
        identity_contract=runtime_identity_contract(
            normalizer_mean=2.0, normalizer_scale=4.0
        ),
    )
    runtime.activate(
        create_bundle(
            tmp_path / "normalized",
            policy_mode=PolicyMode.RESIDUAL_POLICY,
            normalizer_mean=2.0,
            normalizer_scale=4.0,
        )
    )
    policy.observations.clear()

    runtime.predict(np.full(OBSERVATION_SIZE, 6.0, dtype=np.float32))

    np.testing.assert_allclose(
        policy.observations[-1], np.ones(OBSERVATION_SIZE, dtype=np.float32)
    )


def test_activation_probes_policy_before_replacing_live_state(tmp_path: Path) -> None:
    runtime = ServingRuntime(identity_contract=runtime_identity_contract())
    original = runtime.activate(create_bundle(tmp_path / "baseline"))
    bad_policy = RecordingPolicy(np.array([0.0, np.nan, 0.0], dtype=np.float32))
    runtime.policy_loader = Loader(bad_policy)

    with pytest.raises(ValueError, match="action schema"):
        runtime.activate(
            create_bundle(
                tmp_path / "bad",
                policy_mode=PolicyMode.RESIDUAL_POLICY,
            )
        )

    assert runtime.snapshot() == original


def test_bundle_rejects_missing_normalizer_sidecar(tmp_path: Path) -> None:
    root = create_bundle(tmp_path / "missing")
    (root / "normalizer.json").unlink()
    with pytest.raises(ValueError, match="normalizer|missing"):
        from trade_rl.serving.bundle import load_serving_bundle

        load_serving_bundle(root)
