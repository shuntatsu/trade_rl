from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from trade_rl.evaluation import ppo_normalization_execution as module
from trade_rl.evaluation.directional_contract import DIRECTIONAL_BASE_EXECUTION_COST
from trade_rl.evaluation.ppo_normalization_execution import (
    replication_arm_specs,
    replication_strategy_factory,
    fit_replication_strategy,
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
    dataset = SimpleNamespace(timestamps=timestamps)
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
