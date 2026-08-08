from __future__ import annotations

import pytest

from tools.nautilus_training_throughput_benchmark import _normalize_timesteps


def test_normalize_timesteps_accepts_scalar_and_canonicalizes_sequence() -> None:
    assert _normalize_timesteps(8) == (8,)
    assert _normalize_timesteps([32, 8, 32]) == (8, 32)


def test_normalize_timesteps_rejects_bool_values_explicitly() -> None:
    with pytest.raises(TypeError, match="timesteps must contain integers"):
        _normalize_timesteps(True)
    with pytest.raises(TypeError, match="timesteps must contain integers"):
        _normalize_timesteps([8, True])
