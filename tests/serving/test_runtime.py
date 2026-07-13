from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.serving.helpers import create_bundle
from trade_rl.domain.selection import PolicyMode
from trade_rl.serving.bundle import ServingBundle
from trade_rl.serving.runtime import LoadedPolicy, RuntimeSnapshot, ServingRuntime


class ConstantPolicy:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def predict(self, observation: np.ndarray) -> np.ndarray:
        del observation
        return self.value.copy()


def _predict(
    runtime: ServingRuntime,
    snapshot: RuntimeSnapshot,
    observation: np.ndarray,
) -> np.ndarray:
    return runtime.predict(
        observation,
        dataset_id=snapshot.dataset_id,
        observation_schema_digest=snapshot.observation_schema_digest,
        market_inputs_digest=snapshot.market_inputs_digest,
    )


class FakeLoader:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[ServingBundle] = []

    def load(self, bundle: ServingBundle) -> LoadedPolicy:
        self.calls.append(bundle)
        if self.fail:
            raise RuntimeError("load failed")
        return ConstantPolicy(np.array([0.25, -0.5], dtype=np.float32))


def test_baseline_bundle_serves_identity_action_without_policy_loader(
    tmp_path: Path,
) -> None:
    runtime = ServingRuntime()

    snapshot = runtime.activate(create_bundle(tmp_path / "baseline"))
    action = _predict(runtime, snapshot, np.zeros(5, dtype=np.float32))

    assert snapshot.policy_mode is PolicyMode.BASELINE_ONLY
    np.testing.assert_array_equal(action, np.zeros(2, dtype=np.float32))


def test_residual_bundle_loads_policy_and_validates_action_schema(
    tmp_path: Path,
) -> None:
    loader = FakeLoader()
    runtime = ServingRuntime(policy_loader=loader)

    snapshot = runtime.activate(
        create_bundle(tmp_path / "residual", policy_mode=PolicyMode.RESIDUAL_POLICY)
    )
    action = _predict(runtime, snapshot, np.zeros(5, dtype=np.float32))

    assert snapshot.policy_mode is PolicyMode.RESIDUAL_POLICY
    assert len(loader.calls) == 1
    np.testing.assert_allclose(action, np.array([0.25, -0.5], dtype=np.float32))


def test_failed_hot_swap_preserves_previous_snapshot(tmp_path: Path) -> None:
    runtime = ServingRuntime(policy_loader=FakeLoader())
    first = runtime.activate(
        create_bundle(tmp_path / "first", policy_mode=PolicyMode.RESIDUAL_POLICY)
    )
    runtime.policy_loader = FakeLoader(fail=True)

    with pytest.raises(RuntimeError, match="load failed"):
        runtime.activate(
            create_bundle(tmp_path / "second", policy_mode=PolicyMode.RESIDUAL_POLICY)
        )

    assert runtime.snapshot() == first
    np.testing.assert_allclose(
        _predict(runtime, first, np.zeros(5, dtype=np.float32)),
        np.array([0.25, -0.5], dtype=np.float32),
    )


def test_predict_rejects_non_finite_or_wrong_shaped_action(tmp_path: Path) -> None:
    class BadLoader:
        def load(self, bundle: ServingBundle) -> LoadedPolicy:
            del bundle
            return ConstantPolicy(np.array([np.nan], dtype=np.float32))

    runtime = ServingRuntime(policy_loader=BadLoader())
    snapshot = runtime.activate(
        create_bundle(tmp_path / "bad", policy_mode=PolicyMode.RESIDUAL_POLICY)
    )

    with pytest.raises(ValueError, match="action schema"):
        _predict(runtime, snapshot, np.zeros(5, dtype=np.float32))


def test_raw_predict_rejects_identity_contract_bypass(tmp_path: Path) -> None:
    runtime = ServingRuntime()
    snapshot = runtime.activate(create_bundle(tmp_path / "contract"))
    vector = np.zeros(snapshot.observation_size, dtype=np.float32)

    with pytest.raises(ValueError, match="dataset identity"):
        runtime.predict(
            vector,
            dataset_id="f" * 64,
            observation_schema_digest=snapshot.observation_schema_digest,
            market_inputs_digest=snapshot.market_inputs_digest,
        )
    with pytest.raises(ValueError, match="observation schema"):
        runtime.predict(
            vector,
            dataset_id=snapshot.dataset_id,
            observation_schema_digest="f" * 64,
            market_inputs_digest=snapshot.market_inputs_digest,
        )
    with pytest.raises(ValueError, match="market inputs"):
        runtime.predict(
            vector,
            dataset_id=snapshot.dataset_id,
            observation_schema_digest=snapshot.observation_schema_digest,
            market_inputs_digest="f" * 64,
        )
