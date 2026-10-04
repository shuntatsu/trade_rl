"""Synthetic source-profile wiring through the bot's real execution path."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from tests.evaluation.test_forming_week_bot import _config as _forming_config
from tests.evaluation.test_forming_week_bot import _rebind, _source
from tests.evaluation.test_shared_cash_replay import _market
from tests.integrations.test_binance_market_order_profile import _profile
from trade_rl.data.features.forming_week_context import with_forming_week_context
from trade_rl.evaluation import bot as bot_module
from trade_rl.evaluation.bot import BotConfig, run_trading_bot
from trade_rl.evaluation.replay import SharedCashReplayResult
from trade_rl.simulation import ExecutionCostConfig, MarketExecutor


def _data():
    return _market(
        np.array([[100], [100], [1], [1], [1], [1]], dtype=float),
        symbols=("BTCUSDT",),
    ).with_content_identity()


def _config():
    return BotConfig(
        strategy_name="constant_long",
        initial_capital=10_000,
        gross_budget=0.1,
        minimum_hold_bars=0,
        execution_cost=replace(
            ExecutionCostConfig.zero(),
            fee_rate=0.001,
            max_participation_rate=1.0,
            processing_bar_volume_capacity=True,
        ),
    )


def _events(result: SharedCashReplayResult):
    assert result.ledger_evidence is not None
    return tuple(
        event
        for interval in result.ledger_evidence.intervals
        for event in interval.order_events
    )


def test_none_profile_preserves_omitted_returns_report_ledger_and_policy():
    data, config = _data(), _config()
    omitted, report = run_trading_bot(data, config)
    explicit_none, none_report = run_trading_bot(
        data, config, market_order_profile=None
    )

    assert explicit_none.returns == omitted.returns
    assert none_report == report
    assert explicit_none.diagnostics == omitted.diagnostics
    assert explicit_none.decisions == omitted.decisions
    assert explicit_none.ledger_evidence == omitted.ledger_evidence
    assert explicit_none.book.exact_quantities == omitted.book.exact_quantities
    assert explicit_none.book.cash == omitted.book.cash
    assert omitted.ledger_evidence is not None
    assert (
        omitted.ledger_evidence.execution_policy_digest
        == MarketExecutor(data, config.execution_cost).execution_policy_digest
    )
    assert omitted.ledger_evidence.schema_version == "shared_cash_replay_ledger_v2"


def test_matched_profiles_cost_only_bounded_actual_fills_and_keep_exact_residual():
    data, config = _data(), _config()
    ordinary_profile = _profile(data, reduce_only_exits=False)
    closing_profile = _profile(data, reduce_only_exits=True)
    assert ordinary_profile.source_sha256 == closing_profile.source_sha256
    assert ordinary_profile.rules == closing_profile.rules
    assert ordinary_profile.digest != closing_profile.digest

    ordinary, ordinary_report = run_trading_bot(
        data, config, market_order_profile=ordinary_profile
    )
    closing, closing_report = run_trading_bot(
        data, config, market_order_profile=closing_profile
    )

    assert ordinary.book.exact_quantities == (Fraction(10),)
    assert closing.book.exact_quantities == (Fraction(0),)
    assert not ordinary_report.terminal_settled
    assert closing_report.terminal_settled
    assert ordinary_report.fill_count == 1
    assert closing_report.fill_count == 2
    assert ordinary_report.total_execution_cost == pytest.approx(1.0)
    assert closing_report.total_execution_cost == pytest.approx(1.01)
    assert ordinary.ledger_evidence is not None
    assert closing.ledger_evidence is not None
    for result, profile in (
        (ordinary, ordinary_profile),
        (closing, closing_profile),
    ):
        assert (
            result.ledger_evidence.execution_policy_digest
            == MarketExecutor(
                data, config.execution_cost, market_order_profile=profile
            ).execution_policy_digest
        )
    assert (
        ordinary.ledger_evidence.execution_policy_digest
        != closing.ledger_evidence.execution_policy_digest
    )
    fills = [event for event in _events(closing) if event.filled_quantity]
    assert [(event.filled_quantity, event.reduce_only) for event in fills] == [
        (10.0, False),
        (-10.0, True),
    ]
    terminal = closing.ledger_evidence.intervals[-1]
    assert terminal.exact_quantities_before == ("10",)
    assert terminal.exact_quantities_after == ("0",)
    assert terminal.interval_cost == pytest.approx(10 * 1 * 0.001)
    assert not closing.ledger_evidence.active_order_remainders
    assert any(event.reason == "below_minimum_notional" for event in _events(ordinary))


def test_profile_keeps_small_opening_under_ordinary_notional_floor():
    data, config = _data(), replace(_config(), gross_budget=0.0004)
    result, report = run_trading_bot(data, config, market_order_profile=_profile(data))

    assert result.book.exact_quantities == (Fraction(0),)
    assert report.fill_count == 0
    assert report.total_execution_cost == 0
    assert any(event.reason == "below_minimum_notional" for event in _events(result))
    assert not any(event.reduce_only for event in _events(result))


def test_profile_keeps_reversal_under_ordinary_notional_floor():
    prices = np.full((6, 1), 100.0)
    data = _market(prices, symbols=("BTCUSDT",), open_values=prices)
    floors = np.full((6, 1), 50.0)
    floors[2:] = 5_000
    signals = np.full((6, 1, 1), 0.03, dtype=np.float32)
    signals[2:] = -0.03
    data = replace(
        data, minimum_notional=floors, features=signals
    ).with_content_identity()
    result, report = run_trading_bot(
        data,
        replace(_config(), strategy_name="trend"),
        market_order_profile=_profile(data),
    )

    rejected = [event for event in _events(result) if event.event_type == "rejected"]
    assert rejected
    assert all(event.reason == "below_minimum_notional" for event in rejected)
    assert all(not event.reduce_only for event in rejected)
    reversal = next(decision for decision in result.decisions if decision.index == 2)
    assert reversal.intents[0].value == -1
    assert (
        reversal.position_quantity_before == reversal.position_quantity_after == (10,)
    )
    assert report.fill_count == 2
    assert report.terminal_settled  # The later explicit flat close may use the waiver.


@pytest.mark.parametrize("constraint", ["runtime_floor", "sub_lot"])
def test_profile_leaves_unfillable_terminal_quantity_exact_and_visible(constraint):
    data, config = _data(), _config()
    if constraint == "runtime_floor":
        config = replace(
            config, execution_cost=replace(config.execution_cost, minimum_notional=20)
        )
        quantity, reason = Fraction(10), "below_minimum_notional"
    else:
        # A synthetic unit change creates real inventory below the source lot grid.
        splits = np.ones((6, 1))
        splits[4] = 0.00001
        data = replace(
            data, split_factor=splits, identity_payload_json=None
        ).with_content_identity()
        quantity, reason = Fraction("0.0001"), "zero_quantity_after_rounding"
    result, report = run_trading_bot(data, config, market_order_profile=_profile(data))

    assert result.book.exact_quantities == (quantity,)
    assert not report.terminal_settled
    assert report.fill_count == 1
    assert report.total_execution_cost == pytest.approx(1.0)
    assert any(
        event.reduce_only and event.reason == reason for event in _events(result)
    )


def test_profile_does_not_fill_before_latency_or_skip_costed_terminal_close():
    prices = np.array([[100], [100], [100], [1], [1], [1]], dtype=float)
    data = _market(
        prices, symbols=("BTCUSDT",), open_values=prices
    ).with_content_identity()
    config = replace(
        _config(),
        execution_cost=replace(_config().execution_cost, order_latency_bars=2),
    )
    result, report = run_trading_bot(data, config, market_order_profile=_profile(data))

    events = _events(result)
    assert [event.processing_index for event in events if event.filled_quantity] == [
        2,
        4,
    ]
    assert [
        event.processing_index for event in events if event.event_type == "latency_wait"
    ] == [1, 3]
    assert result.book.exact_quantities == (Fraction(0),)
    assert report.terminal_settled
    assert report.total_execution_cost == pytest.approx(1.01)


def test_profile_partial_capacity_close_keeps_residual_and_active_remainder():
    data = _data()
    volume = data.volume.copy()
    volume[-1] = 3
    data = replace(
        data, volume=volume, identity_payload_json=None
    ).with_content_identity()
    result, report = run_trading_bot(
        data, _config(), market_order_profile=_profile(data)
    )

    assert result.book.exact_quantities == (Fraction(7),)
    assert not report.terminal_settled
    assert report.total_execution_cost == pytest.approx(1.003)
    assert result.ledger_evidence is not None
    terminal = result.ledger_evidence.intervals[-1]
    partials = [
        event for event in terminal.order_events if event.event_type == "partial_fill"
    ]
    assert len(partials) == 1
    assert partials[0].reduce_only and partials[0].filled_quantity == -3
    assert [
        remainder for _, remainder in result.ledger_evidence.active_order_remainders
    ] == [-7]
    capacity = terminal.capacity_events[0]
    assert capacity.consumed_capacity_notional == 3
    assert capacity.consumed_capacity_notional <= capacity.initial_capacity_notional


@pytest.mark.parametrize("fault", ["different_dataset", "mutated_unverified_content"])
def test_profile_rejects_wrong_dataset_or_mutated_source_content(fault):
    data = _data()
    profile = _profile(data)
    if fault == "different_dataset":
        data = replace(
            data, volume=data.volume + 1, identity_payload_json=None
        ).with_content_identity()
    else:
        # Canonical construction already rejects retaining an identity after a
        # content edit. Removing that identity cannot preserve profile eligibility.
        data = replace(data, features=data.features + 1, identity_payload_json=None)
        assert not data.identity_verified
    with pytest.raises(ValueError, match="dataset|identity"):
        run_trading_bot(data, _config(), market_order_profile=profile)


@pytest.mark.parametrize("precomputed", [False, True])
def test_prepared_profile_reaches_real_replay_after_forming_augmentation(
    monkeypatch, precomputed
):
    source = _rebind(_source(), symbols=("BTCUSDT", "ETHUSDT"))
    prepared = with_forming_week_context(source)
    profile = _profile(prepared, selected_symbols=prepared.symbols)
    config = replace(_forming_config(), initial_capital=10_000, gross_budget=0.1)
    start = 19 * 168
    calls = []
    real_replay = bot_module.run_shared_cash_replay

    def recording_replay(dataset, strategies, **kwargs):
        calls.append((dataset.dataset_id, kwargs.get("market_order_profile")))
        return real_replay(dataset, strategies, **kwargs)

    monkeypatch.setattr(bot_module, "run_shared_cash_replay", recording_replay)
    result, report = run_trading_bot(
        prepared if precomputed else source,
        config,
        start_index=start,
        stop_index=start + 40,
        market_order_profile=profile,
    )

    assert len(calls) == 1
    assert calls[0][0] == prepared.dataset_id
    assert calls[0][1] is profile
    assert result.ledger_evidence is not None
    assert result.ledger_evidence.dataset_id == prepared.dataset_id
    assert (
        result.ledger_evidence.execution_policy_digest
        == MarketExecutor(
            prepared, config.execution_cost, market_order_profile=profile
        ).execution_policy_digest
    )
    # H24 unlock reapplies LONG after fee/funding drift. Each requested 0.0072
    # reduction fills seven source lots; the true profile allows its small notional.
    fills = sorted(
        (
            event.processing_index - start,
            event.symbol_index,
            event.filled_quantity,
            event.reduce_only,
        )
        for event in _events(result)
        if event.filled_quantity
    )
    assert fills == [
        (1, 0, 10.0, False),
        (1, 1, 10.0, False),
        (25, 0, -0.007, True),
        (25, 1, -0.007, True),
        (40, 0, -9.993, True),
        (40, 1, -9.993, True),
    ]
    assert report.fill_count == len(fills)
    unlock = result.ledger_evidence.intervals[24]
    terminal = result.ledger_evidence.intervals[-1]
    assert unlock.exact_quantities_before == ("10", "10")
    assert unlock.exact_quantities_after == ("9993/1000", "9993/1000")
    assert terminal.exact_quantities_before == unlock.exact_quantities_after
    assert terminal.exact_quantities_after == ("0", "0")
    assert result.book.exact_quantities == (Fraction(0), Fraction(0))
    # Constant price 100, two symbols, 24 funded intervals at 10 units and
    # 15 at 9.993 units; terminal flat pays no funding. Fees plus spread cost
    # 0.0012 of 4000 total notional; impact uses the three known fill sizes.
    expected_funding = -2 * 100 * 0.0001 * (24 * 10 + 15 * 9.993)
    expected_cost = 4.8 + 2 * 0.0001 * sum(
        notional * np.sqrt(quantity / 1_000_000_000)
        for quantity, notional in ((10, 1000), (0.007, 0.7), (9.993, 999.3))
    )
    assert report.funding_pnl == pytest.approx(expected_funding, rel=0, abs=1e-12)
    assert report.total_execution_cost == pytest.approx(expected_cost, rel=0, abs=1e-12)
    assert result.book.cash == pytest.approx(
        10_000 - expected_cost + expected_funding, rel=0, abs=1e-8
    )
    assert not result.ledger_evidence.active_order_remainders
    assert report.terminal_settled


def test_forming_autoaugmentation_rejects_original_dataset_profile_binding():
    source = _rebind(_source(), symbols=("BTCUSDT", "ETHUSDT"))
    profile = _profile(source, selected_symbols=source.symbols)
    assert with_forming_week_context(source).dataset_id != profile.dataset_id

    with pytest.raises(ValueError, match="dataset.*binding"):
        run_trading_bot(
            source,
            _forming_config(),
            start_index=19 * 168,
            stop_index=19 * 168 + 40,
            market_order_profile=profile,
        )
