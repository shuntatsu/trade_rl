from __future__ import annotations

from pathlib import Path

import numpy as np

from trade_rl.learning.teacher_artifact import (
    SupervisedPolicyDataset,
    load_teacher_artifact,
    write_teacher_artifact,
)


def test_structured_teacher_artifact_round_trip_binds_key_order_and_shapes(
    tmp_path: Path,
) -> None:
    observations = {
        "active": np.ones((4, 3), dtype=np.float32),
        "sequence_15m_values": np.arange(4 * 3 * 2 * 5, dtype=np.float32).reshape(
            4, 3, 2, 5
        ),
    }
    dataset = SupervisedPolicyDataset(
        observations=observations,
        actions=np.zeros((4, 3), dtype=np.float32),
        dataset_id="a" * 64,
        train_start=5,
        train_stop=10,
        environment_digest="b" * 64,
        action_spec_digest="c" * 64,
        teacher_config_digest="d" * 64,
    )

    write_teacher_artifact(tmp_path, dataset)
    manifest, loaded = load_teacher_artifact(tmp_path)

    assert manifest.observation_keys == ("active", "sequence_15m_values")
    assert manifest.observation_shapes == {
        "active": (4, 3),
        "sequence_15m_values": (4, 3, 2, 5),
    }
    assert isinstance(loaded.observations, dict)
    for key in observations:
        np.testing.assert_array_equal(loaded.observations[key], observations[key])
