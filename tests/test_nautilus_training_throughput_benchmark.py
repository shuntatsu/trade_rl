from __future__ import annotations

import pytest

from tools.nautilus_training_throughput_benchmark import (
    _DEFAULT_TIMESTEPS,
    _normalize_timesteps,
)


def test_default_timesteps_cover_broader_performance_workloads() -> None:
    assert _DEFAULT_TIMESTEPS == (8, 32, 128)


def test_normalize_timesteps_accepts_scalar_and_canonicalizes_sequence() -> None:
    assert _normalize_timesteps(8) == (8,)
    assert _normalize_timesteps([32, 8, 32]) == (8, 32)


def test_normalize_timesteps_rejects_bool_values_explicitly() -> None:
    with pytest.raises(TypeError, match="timesteps must contain integers"):
        _normalize_timesteps(True)
    with pytest.raises(TypeError, match="timesteps must contain integers"):
        _normalize_timesteps([8, True])


def test_persisted_catalog_workload_requires_source_identity() -> None:
    from tools.nautilus_training_throughput_benchmark import _benchmark_source_digest

    synthetic = _benchmark_source_digest((8, 32, 128))
    assert len(synthetic) == 64
