from __future__ import annotations

from tools.tier_d_nonrl.task18_recipe import build_task18_protocol

EXPECTED_PAIRS = [
    ("H24_RIDGE_A1", "BTCUSDT", "BASEL0"),
    ("H24_RIDGE_A1", "BTCUSDT", "COST2L0"),
    ("H24_RIDGE_A1", "BTCUSDT", "BASEL2"),
    ("H24_RIDGE_A1", "ETHUSDT", "BASEL0"),
    ("H24_RIDGE_A1", "ETHUSDT", "COST2L0"),
    ("H24_RIDGE_A1", "ETHUSDT", "BASEL2"),
    ("H24_RIDGE_A10", "BTCUSDT", "BASEL0"),
    ("H24_RIDGE_A10", "BTCUSDT", "COST2L0"),
    ("H24_RIDGE_A10", "BTCUSDT", "BASEL2"),
    ("H24_RIDGE_A10", "ETHUSDT", "BASEL0"),
    ("H24_RIDGE_A10", "ETHUSDT", "COST2L0"),
    ("H24_RIDGE_A10", "ETHUSDT", "BASEL2"),
    ("H72_RIDGE_A1", "BTCUSDT", "BASEL0"),
    ("H72_RIDGE_A1", "BTCUSDT", "COST2L0"),
    ("H72_RIDGE_A1", "BTCUSDT", "BASEL2"),
    ("H72_RIDGE_A1", "ETHUSDT", "BASEL0"),
    ("H72_RIDGE_A1", "ETHUSDT", "COST2L0"),
    ("H72_RIDGE_A1", "ETHUSDT", "BASEL2"),
    ("H72_RIDGE_A10", "BTCUSDT", "BASEL0"),
    ("H72_RIDGE_A10", "BTCUSDT", "COST2L0"),
    ("H72_RIDGE_A10", "BTCUSDT", "BASEL2"),
    ("H72_RIDGE_A10", "ETHUSDT", "BASEL0"),
    ("H72_RIDGE_A10", "ETHUSDT", "COST2L0"),
    ("H72_RIDGE_A10", "ETHUSDT", "BASEL2"),
]


def test_roster_is_exactly_ordered_24_pairs_and_48_nonrl_cash_arms() -> None:
    protocol = build_task18_protocol()

    assert protocol.scope == "CONSUMED_2025_DEVELOPMENT_NONRL48_ONLY"
    assert protocol.roster_dimension_note == (
        "4 configurations x 2 symbols x 3 scenarios = 24 pairs; three vintages "
        "are sequential model updates, not a pair dimension."
    )
    assert len(protocol.pairs) == 24
    assert len(protocol.arms) == 48
    assert [pair.pair_id for pair in protocol.pairs] == [
        f"PAIR-{index:02d}" for index in range(1, 25)
    ]
    assert [
        (pair.configuration, pair.symbol, pair.scenario) for pair in protocol.pairs
    ] == EXPECTED_PAIRS
    assert [(arm.arm_id, arm.pair_id, arm.role) for arm in protocol.arms[:6]] == [
        ("PAIR-01-NONRL", "PAIR-01", "NONRL"),
        ("PAIR-01-CASH", "PAIR-01", "CASH"),
        ("PAIR-02-NONRL", "PAIR-02", "NONRL"),
        ("PAIR-02-CASH", "PAIR-02", "CASH"),
        ("PAIR-03-NONRL", "PAIR-03", "NONRL"),
        ("PAIR-03-CASH", "PAIR-03", "CASH"),
    ]


def test_adapted_protocol_digest_is_separate_from_historical_reference() -> None:
    protocol = build_task18_protocol()

    assert protocol.historical_reference.source_commit == (
        "9f40cc93aec8439a3740aec4e2aa2b842ad893a3"
    )
    assert protocol.historical_reference.protocol_digest == (
        "9ce03a1a3361e77edb2a111ebb1364f83994e09400128a920bb4b3c38b6c9b79"
    )
    assert len(protocol.historical_reference.source_file_pins) == 14
    assert protocol.adapted_protocol_digest != (
        protocol.historical_reference.protocol_digest
    )
    assert len(protocol.adapted_protocol_digest) == 64


def test_frozen_indexes_carry_and_terminal_reserve_are_explicit() -> None:
    protocol = build_task18_protocol()

    assert protocol.index_plan.actor_start_inclusive == 2903
    assert protocol.index_plan.actor_stop_exclusive == 9430
    assert protocol.index_plan.actor_start_utc == "2025-04-01T00:00:00Z"
    assert protocol.index_plan.actor_last_bar_utc == "2025-12-28T22:00:00Z"
    assert protocol.index_plan.actor_stop_exclusive_utc == "2025-12-28T23:00:00Z"
    assert protocol.index_plan.vintage_start_indices == (2903, 5087, 7295)
    assert protocol.index_plan.ridge_vintage_count == 12
    assert protocol.index_plan.public_cost_preparation_fit_count == 4
    assert protocol.index_plan.vintages_are_pair_dimensions is False
    assert protocol.index_plan.account_continuity_across_vintages is True
    assert protocol.index_plan.terminal_stop_inclusive == 9502
    assert protocol.index_plan.terminal_advance_count == 72
    assert protocol.index_plan.terminal_time_utc == "2025-12-31T23:00:00Z"
    assert protocol.index_plan.unused_final_row == 9503
    assert protocol.index_plan.native_advances_per_arm == 6599
    assert protocol.index_plan.total_native_advances == 316752


def test_economics_and_risk_recipe_are_fixed_without_claiming_winner() -> None:
    protocol = build_task18_protocol()

    assert protocol.dataset.row_count == 9504
    assert protocol.dataset.symbols == ("BTCUSDT", "ETHUSDT")
    assert protocol.dataset.features_per_symbol == 6
    assert protocol.economics.initial_capital_per_symbol_usdt == 100_000
    assert protocol.economics.books_are_independent_by_symbol
    assert protocol.economics.base_fee_rate == 0.0005
    assert protocol.economics.base_spread_rate == 0.0002
    assert protocol.economics.impact_coefficient == 0.0001
    assert protocol.economics.impact_reference_notional_usdt == 100_000
    assert protocol.economics.maximum_participation_rate == 0.05
    assert protocol.economics.borrow_available is True
    assert protocol.economics.borrow_rate == 0.0
    assert protocol.scenarios[1].fee_multiplier == 2.0
    assert protocol.scenarios[1].spread_multiplier == 2.0
    assert protocol.scenarios[1].impact_multiplier == 2.0
    assert protocol.scenarios[2].execution_latency_bars == 2
    assert protocol.risk.max_gross == 1.0
    assert protocol.risk.max_abs_weight == 1.0
    assert protocol.risk.drawdown_stop == 0.20
    assert protocol.rollout_length == 32
    assert protocol.gamma == 1.0
    assert protocol.gae_lambda == 0.95
    assert protocol.gae_lambda_usage == "declared but unused for the NONRL baseline"
    assert protocol.index_plan.declared_seeds == 0
    assert protocol.hyperparameter_optimization is False
    assert protocol.gates.minimum_operational_threshold == "UNKNOWN"
    assert protocol.gates.adoption_status == "NOT_ESTABLISHED"
    assert protocol.gates.winner_status == "NOT_DECLARED"
    assert protocol.gates.combined_metric_denominator == (
        "2 * V0, with independent 100000 USDT books per symbol"
    )
    assert protocol.nonrl_policy == "residual KEEP_BASELINE code 2, scale 1"
    assert protocol.cash_policy == "direct HOLD code 0 from a flat book, scale 1"
    assert protocol.label_semantics == (
        "raw close-ratio simple-return labels; carry and costs are accounted "
        "in allocation/execution"
    )
