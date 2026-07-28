from __future__ import annotations

import pytest

from trade_rl.rl.training_modes import CudaRuntimeMode, ObservationEncoder


def test_observation_encoder_is_a_closed_string_enum() -> None:
    assert ObservationEncoder("flat_mlp") is ObservationEncoder.FLAT_MLP
    assert ObservationEncoder("asset_set") is ObservationEncoder.ASSET_SET
    assert (
        ObservationEncoder("hierarchical_sequence_v2")
        is ObservationEncoder.HIERARCHICAL_SEQUENCE_V2
    )
    with pytest.raises(ValueError):
        ObservationEncoder("sequence")


def test_cuda_runtime_mode_is_a_closed_string_enum() -> None:
    assert CudaRuntimeMode("deterministic") is CudaRuntimeMode.DETERMINISTIC
    assert CudaRuntimeMode("performance") is CudaRuntimeMode.PERFORMANCE
    with pytest.raises(ValueError):
        CudaRuntimeMode("auto")
