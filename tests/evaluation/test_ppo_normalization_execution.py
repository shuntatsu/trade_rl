from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.evaluation import ppo_normalization_execution as module
from trade_rl.evaluation.directional_contract import DIRECTIONAL_BASE_EXECUTION_COST
from trade_rl.evaluation.ppo_normalization_execution import (
    claim_replication_slot,
    fit_replication_strategy,
    recompute_replication_decision,
    record_consumed_failure,
    record_prefit_failure,
    replication_arm_specs,
    replication_slot_state,
    replication_strategy_factory,
)


def test_replication_roster_is_exactly_ten_fresh_matched_seed_slots() -> None:
    specs = replication_arm_specs()

    assert tuple(spec.slot for spec in specs) == (
        "control_raw_seed0",
        "control_raw_seed1",
        "control_raw_seed2",
        "control_raw_seed3",
        "control_raw_seed4",
        "candidate_normalized_seed0",
        "candidate_normalized_seed1",
        "candidate_normalized_seed2",
        "candidate_normalized_seed3",
        "candidate_normalized_seed4",
    )
    assert len({spec.slot for spec in specs}) == 10

    for seed in range(5):
        control = specs[seed]
        candidate = specs[seed + 5]
        assert control.seed == candidate.seed == seed
        assert control.protocol_arm == "control_raw"
        assert candidate.protocol_arm == "candidate_normalized"
        assert control.normalize_features is False
        assert candidate.normalize_features is True


def test_fit_replication_strategy_changes_only_normalization_for_matched_seed(
    monkeypatch,
) -> None:
    timestamps = np.asarray(
        [
            "2022-12-31T21:00:00",
            "2022-12-31T22:00:00",
            "2022-12-31T23:00:00",
            "2023-01-01T00:00:00",
        ],
        dtype="datetime64[ns]",
    )
    dataset = SimpleNamespace(timestamps=timestamps, n_bars=len(timestamps))
    config = SimpleNamespace(
        feature_indices=(1, 3),
        fit_symbol_indices=(0, 2),
        fit_cutoff="2023-01-01T00:00:00",
    )
    calls: list[dict[str, object]] = []
    sentinel = object()

    def fake_fit(_dataset, **kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(module, "fit_ppo_strategy", fake_fit)
    control, candidate = replication_arm_specs()[0], replication_arm_specs()[5]

    assert fit_replication_strategy(dataset, config, control) is sentinel
    assert fit_replication_strategy(dataset, config, candidate) is sentinel

    first, second = calls
    common_keys = set(first) | set(second)
    changed = {key for key in common_keys if first.get(key) != second.get(key)}
    assert changed == {"normalize_features"}
    assert first["normalize_features"] is False
    assert second["normalize_features"] is True

    for call in calls:
        assert call["feature_indices"] == (1, 3)
        assert call["fit_symbol_indices"] == (0, 2)
        assert call["start_index"] == 0
        assert call["stop_index"] == 2
        assert call["gross_budget"] == 0.1
        assert call["total_timesteps"] == 262_144
        assert call["seed"] == 0
        assert call["initial_capital"] == 10_000.0
        assert call["execution_cost"] is DIRECTIONAL_BASE_EXECUTION_COST
        assert call["training_layout"] == "sequential"
        assert call["risk_config"] is None
        assert call["settle_terminal_position"] is True


def test_replication_strategy_factory_creates_fresh_wrappers() -> None:
    policy = object()
    normalizer = object()
    frozen = SimpleNamespace(
        policy=policy,
        feature_indices=(2, 4),
        feature_normalizer=normalizer,
    )

    factory = replication_strategy_factory(frozen)
    first = factory()
    second = factory()

    assert first is not second
    assert first.policy is second.policy is policy
    assert first.feature_indices == second.feature_indices == (2, 4)
    assert first.feature_normalizer is second.feature_normalizer is normalizer


def test_slot_boundary_distinguishes_prefit_from_consumed_failure(tmp_path) -> None:
    spec = replication_arm_specs()[0]

    record_prefit_failure(
        tmp_path,
        spec,
        attempt_id="runtime-preflight-1",
        error="trainer runtime missing",
    )
    before = replication_slot_state(tmp_path, spec)
    assert before["consumed"] is False
    assert before["failed"] is False
    assert before["prefit_failure_count"] == 1

    claim_replication_slot(
        tmp_path,
        spec,
        activation_digest="a" * 64,
        implementation_digest="b" * 64,
    )
    claimed = replication_slot_state(tmp_path, spec)
    assert claimed["consumed"] is True
    assert claimed["failed"] is False

    with pytest.raises(ValueError, match="consumed"):
        claim_replication_slot(
            tmp_path,
            spec,
            activation_digest="a" * 64,
            implementation_digest="b" * 64,
        )

    record_consumed_failure(tmp_path, spec, error="fit started then failed")
    failed = replication_slot_state(tmp_path, spec)
    assert failed["consumed"] is True
    assert failed["failed"] is True
    assert failed["prefit_failure_count"] == 1


def _screen_row(
    total_return: float,
    *,
    year_return: float,
    with_stress: bool,
) -> dict[str, object]:
    row: dict[str, object] = {
        "metrics": {"total_return": total_return},
        "ledger_max_drawdown": 0.1,
        "terminal_flat": True,
        "termination_reasons": [],
        "start_index": 0,
        "stop_index": 17_544,
        "returns": [0.0] * 17_544,
        "year_returns": {"2023": year_return, "2024": year_return},
        "qualified": False,
    }
    if with_stress:
        stress_row = {
            "metrics": {"total_return": 0.01},
            "ledger_max_drawdown": 0.1,
            "terminal_flat": True,
            "termination_reasons": [],
            "start_index": 0,
            "stop_index": 17_544,
            "returns": [0.0] * 17_544,
            "year_returns": {"2023": -0.5, "2024": -0.5},
        }
        row["stress"] = [
            {**stress_row, "cost_multiplier": 2.0, "latency_bars": 0},
            {**stress_row, "cost_multiplier": 1.0, "latency_bars": 1},
        ]
        row["by_symbol"] = {
            symbol: {}
            for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
        }
    return row


def test_decision_is_recomputed_without_trusting_qualified_flags() -> None:
    control = {
        seed: _screen_row(0.0, year_return=0.0, with_stress=False)
        for seed in range(5)
    }
    candidate = {
        seed: _screen_row(0.01, year_return=0.01, with_stress=True)
        for seed in range(5)
    }

    report = recompute_replication_decision(control, candidate)

    assert report["paired_win_count"] == 5
    assert report["relative_improvement"] is True
    assert report["candidate_base_pass_count"] == 5
    assert report["candidate_base_and_stress_pass_count"] == 5
    assert report["decision"] == "PROSPECTIVE_PAPER_REQUIRED"


def test_relative_improvement_does_not_claim_absolute_profitability() -> None:
    control = {
        seed: _screen_row(-0.02, year_return=-0.01, with_stress=False)
        for seed in range(5)
    }
    candidate = {
        seed: _screen_row(-0.01, year_return=-0.01, with_stress=False)
        for seed in range(5)
    }

    report = recompute_replication_decision(control, candidate)

    assert report["relative_improvement"] is True
    assert report["candidate_base_pass_count"] == 0
    assert report["decision"] == "RELATIVE_IMPROVEMENT_ONLY"
