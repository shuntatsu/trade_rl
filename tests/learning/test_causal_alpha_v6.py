from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v6 import (
    CausalAlphaV6Candidate,
    CausalAlphaV6SlowState,
    CausalAlphaV6TargetConfig,
    CausalAlphaV6TargetPath,
)


def _path(**overrides: object) -> CausalAlphaV6TargetPath:
    rows = 2
    defaults: dict[str, Any] = {
        "candidate": CausalAlphaV6Candidate.FAST_ONLY,
        "initial_weight": 0.0,
        "decision_indices": np.arange(rows),
        "targets": np.array([0.0, 0.025]),
        "fast_proposals": np.array([0.0, 0.025]),
        "expected_returns_4h": np.array([0.01, 0.02]),
        "expected_returns_24h": np.array([0.02, 0.03]),
        "expected_returns_72h": np.array([0.03, 0.04]),
        "direction_scores_4h": np.ones(rows),
        "uncertainties_4h": np.zeros(rows),
        "one_way_cost_rates": np.full(rows, 0.001),
        "liquidity_weight_caps": np.full(rows, 0.25),
        "risk_weight_caps": np.full(rows, 0.25),
        "objectives": np.array([0.0, 0.0003]),
        "confirmation_counts": np.array([1, 2]),
        "actionable_mask": np.ones(rows, dtype=np.bool_),
        "slow_states": (
            CausalAlphaV6SlowState.FLAT,
            CausalAlphaV6SlowState.FLAT,
        ),
        "reasons": ("confirmation_hold", "entry"),
        "reason_counts": (("confirmation_hold", 1), ("entry", 1)),
        "submitted_change_count": 1,
        "sign_flip_count": 0,
        "liquidity_deleveraging_count": 0,
        "risk_projection_count": 0,
        "forecast_digest": "1" * 64,
        "config_digest": CausalAlphaV6TargetConfig().digest,
    }
    defaults.update(overrides)
    return CausalAlphaV6TargetPath(**defaults)


def test_v6_target_config_is_exact_and_digest_stable() -> None:
    config = CausalAlphaV6TargetConfig()
    assert config.target_magnitudes == (0.0, 0.025, 0.05, 0.10, 0.25)
    assert config.maximum_absolute_target == 0.25
    assert config.maximum_target_delta == 0.125
    assert config.fast_rebalance_decisions == 4
    assert config.slow_context_decisions == 16
    assert config.confirmation_count == 2
    assert config.strong_reversal_threshold == 0.02
    assert config.digest == CausalAlphaV6TargetConfig().digest


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_magnitudes", (0.0, 0.05)),
        ("maximum_absolute_target", 0.5),
        ("fast_rebalance_decisions", True),
        ("slow_context_decisions", 8),
        ("edge_margin", 0.0),
    ],
)
def test_v6_target_config_rejects_drift(field: str, value: object) -> None:
    values = {
        item.name: getattr(CausalAlphaV6TargetConfig(), item.name)
        for item in fields(CausalAlphaV6TargetConfig)
        if item.init
    }
    values[field] = value
    with pytest.raises(ValueError, match="must remain"):
        CausalAlphaV6TargetConfig(**values)


def test_v6_target_path_canonicalizes_arrays_and_binds_digest() -> None:
    path = _path()
    assert path.decision_indices.dtype == np.int64
    assert path.actionable_mask.dtype == np.bool_
    assert path.targets.dtype == np.float64
    assert not path.targets.flags.writeable
    assert len(path.digest) == 64
    assert path.reason_counts == (("confirmation_hold", 1), ("entry", 1))


def test_v6_target_path_rejects_unaccounted_reason() -> None:
    with pytest.raises(ValueError, match="reasons"):
        _path(reasons=("not_a_v6_reason", "entry"))


def test_v6_target_path_rejects_misaligned_or_negative_evidence() -> None:
    with pytest.raises(ValueError, match="align"):
        _path(objectives=np.zeros(3))
    with pytest.raises(ValueError, match="non-negative"):
        _path(uncertainties_4h=np.array([-0.1, 0.0]))
    with pytest.raises(ValueError, match="reason counts"):
        _path(reason_counts=(("entry", 2),))


def test_v6_target_path_rejects_digest_or_target_bound_drift() -> None:
    with pytest.raises(ValueError, match="digest mismatch"):
        _path(digest="f" * 64)
    with pytest.raises(ValueError, match="absolute bound"):
        _path(targets=np.array([0.0, 0.30]))
