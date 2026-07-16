from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.serving.helpers import (
    TEST_TRUSTED_ATTESTATION_KEYS,
    create_authenticated_bundle,
    runtime_identity_contract,
)
from tests.serving.test_shared_observation_builder import market_dataset
from trade_rl.rl.observations import ObservationBuilder
from trade_rl.serving.runtime import ServingRuntime


def test_runtime_accepts_matching_bound_vector_contract(tmp_path: Path) -> None:
    runtime = ServingRuntime(
        identity_contract=runtime_identity_contract(),
        trusted_attestation_keys=TEST_TRUSTED_ATTESTATION_KEYS,
    )
    snapshot = runtime.activate(
        create_authenticated_bundle(tmp_path / "matching", observation_size=5)
    )

    action = runtime.predict(np.zeros(snapshot.observation_size, dtype=np.float32))

    np.testing.assert_array_equal(
        action, np.zeros(snapshot.action_size, dtype=np.float32)
    )


def test_runtime_rejects_wrong_observation_vector_size(tmp_path: Path) -> None:
    runtime = ServingRuntime(
        identity_contract=runtime_identity_contract(),
        trusted_attestation_keys=TEST_TRUSTED_ATTESTATION_KEYS,
    )
    runtime.activate(create_authenticated_bundle(tmp_path / "size", observation_size=5))

    with pytest.raises(ValueError, match="observation schema"):
        runtime.predict(np.zeros(4, dtype=np.float32))


def test_activation_rejects_environment_identity_mismatch(tmp_path: Path) -> None:
    runtime = ServingRuntime(
        identity_contract=runtime_identity_contract(),
        trusted_attestation_keys=TEST_TRUSTED_ATTESTATION_KEYS,
    )

    with pytest.raises(ValueError, match="environment identity"):
        runtime.activate(
            create_authenticated_bundle(
                tmp_path / "wrong-environment",
                environment_digest="f" * 64,
            )
        )


def test_observation_schema_digest_binds_layout_contract() -> None:
    market = market_dataset()

    three_actions = ObservationBuilder(action_size=3).schema_digest(market)
    four_actions = ObservationBuilder(action_size=4).schema_digest(market)

    assert three_actions != four_actions
