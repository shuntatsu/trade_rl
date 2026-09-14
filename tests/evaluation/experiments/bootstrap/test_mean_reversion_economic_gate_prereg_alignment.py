from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.evaluation.experiments.bootstrap.mean_reversion_economic_gate_prereg import (
    canonical_mean_reversion_economic_gate_protocol,
)


def test_protocol_freezes_calibration_eligibility_alignment() -> None:
    protocol = canonical_mean_reversion_economic_gate_protocol()

    assert protocol.feature_availability_decision_offset_bars == 0
    assert protocol.cost_execution_offset_bars == 1
    assert protocol.label_endpoint_offset_bars == 25
    assert protocol.require_asset_active_at_decision is True
    assert protocol.require_tradable_at_decision is True
    assert protocol.require_asset_active_through_label_window is True
    assert protocol.require_tradable_through_label_window is True
    assert protocol.label_window_start_offset_bars == 1
    assert protocol.label_window_stop_offset_bars_inclusive == 25
    assert protocol.require_label_end_strictly_before_fit_cutoff is True
    assert protocol.cost_arrays_read_from_execution_row is True


def test_protocol_rejects_calibration_alignment_drift() -> None:
    protocol = canonical_mean_reversion_economic_gate_protocol()
    mutations: tuple[dict[str, object], ...] = (
        {"feature_availability_decision_offset_bars": 1},
        {"cost_execution_offset_bars": 0},
        {"label_endpoint_offset_bars": 24},
        {"require_asset_active_at_decision": False},
        {"require_tradable_at_decision": False},
        {"require_asset_active_through_label_window": False},
        {"require_tradable_through_label_window": False},
        {"label_window_start_offset_bars": 0},
        {"label_window_stop_offset_bars_inclusive": 24},
        {"require_label_end_strictly_before_fit_cutoff": False},
        {"cost_arrays_read_from_execution_row": False},
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="preregistered"):
            replace(protocol, **mutation)
