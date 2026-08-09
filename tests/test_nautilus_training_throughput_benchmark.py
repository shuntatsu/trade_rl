from __future__ import annotations

import pytest

from tools.nautilus_training_throughput_benchmark import (
    _DEFAULT_TIMESTEPS,
    _benchmark_source_digest,
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


def test_benchmark_source_digest_binds_persisted_dataset_identity() -> None:
    first = _benchmark_source_digest((8, 32, 128), dataset_source_digest="a" * 64)
    second = _benchmark_source_digest((8, 32, 128), dataset_source_digest="b" * 64)

    assert first != second


def test_benchmark_source_digest_rejects_invalid_persisted_dataset_identity() -> None:
    with pytest.raises(ValueError, match="dataset_source_digest must be a SHA-256 digest"):
        _benchmark_source_digest((8,), dataset_source_digest="not-a-digest")
