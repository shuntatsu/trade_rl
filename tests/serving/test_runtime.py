from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.serving.helpers import (
    ACTION_NAMES,
    OBSERVATION_SIZE,
    create_bundle,
    runtime_identity_contract,
)
from trade_rl.domain.selection import PolicyMode
from trade_rl.serving.bundle import ServingBundle
from trade_rl.serving.runtime import LoadedPolicy, ServingRuntime


class ConstantPolicy:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def predict(self, observation: np.ndarray) -> np.ndarray:
        return self.value.copy()


class Loader:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def load(self, bundle: ServingBundle) -> LoadedPolicy:
        return ConstantPolicy(self.value)


def test_runtime_requires_bound_identity_by_default() -> None:
    with pytest.raises(ValueError, match="explicit identity contract"):
        ServingRuntime()


def test_baseline_bundle_returns_dynamic_zero_identity_action(
    tmp_path: Path,
) -> None:
    runtime = ServingRuntime(identity_contract=runtime_identity_contract())
    snapshot = runtime.activate(create_bundle(tmp_path / "baseline"))
    action = runtime.predict(np.zeros(OBSERVATION_SIZE, dtype=np.float32))
    assert snapshot.action_names == ACTION_NAMES
    np.testing.assert_array_equal(
        action,
        np.zeros(len(ACTION_NAMES), dtype=np.float32),
    )


def test_runtime_fails_closed_on_identity_mismatch(tmp_path: Path) -> None:
    runtime = ServingRuntime(
        identity_contract=runtime_identity_contract(environment_digest="f" * 64)
    )
    with pytest.raises(ValueError, match="environment identity"):
        runtime.activate(create_bundle(tmp_path / "bundle"))


def test_runtime_rejects_wrong_shape_nonfinite_and_out_of_bounds_actions(
    tmp_path: Path,
) -> None:
    for name, value in (
        ("shape", np.array([0.0])),
        ("finite", np.array([0.0, np.nan, 0.0])),
        ("bounds", np.array([0.0, 1.1, 0.0])),
    ):
        runtime = ServingRuntime(
            policy_loader=Loader(value),
            identity_contract=runtime_identity_contract(),
        )
        runtime.activate(
            create_bundle(
                tmp_path / name,
                policy_mode=PolicyMode.RESIDUAL_POLICY,
            )
        )
        with pytest.raises(ValueError, match="action schema"):
            runtime.predict(np.zeros(OBSERVATION_SIZE, dtype=np.float32))
