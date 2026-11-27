"""Native replay and literal cash path oracles for the restricted spot audit."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.simulation.test_stateful_execution import _market
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.evaluation.strict_spot_cash_path import verify_strict_spot_cash_path
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig, runtime_execution_policy_digest
from trade_rl.strategies.allocation.buy_and_hold import BuyAndHoldPortfolioPolicy


def _fixture(**overrides):
    return _market(**overrides).with_content_identity({"fixture": "spot-cash-path-v1"})


def _run(data, *, cost=None, strict=True, policy=None):
    cost = cost or replace(
        ExecutionCostConfig.zero(), allow_short=False, fee_rate=0.001
    )
    batches = []
    result = run_shared_portfolio_replay(
        data,
        policy or BuyAndHoldPortfolioPolicy(symbols=data.symbols, gross_budget=0.5),
        start_index=0,
        stop_index=4,
        initial_capital=1000.0,
        execution_cost=cost,
        risk=PreTradeRisk(
            PreTradeRiskConfig(max_gross=1.0, max_abs_weight=1.0, max_turnover=None)
        ),
        settle_terminal_position=True,
        strict_spot_cash=strict,
        capture_ledger_evidence=True,
        fill_cash_observer=batches.append,
    )
    assert result.ledger_evidence is not None
    return result.ledger_evidence, tuple(batches), cost


def _verify(data, ledger, batches, cost, **overrides):
    args = {
        "expected_dataset_id": data.dataset_id,
        "expected_symbols": data.symbols,
        "expected_spot_symbols": data.symbols,
        "expected_start_index": 0,
        "expected_stop_index": 4,
        "expected_initial_cash": 1000.0,
        "execution_cost": cost,
        "expected_execution_policy_digest": runtime_execution_policy_digest(
            cost,
            funding_price_rate_present=data.funding_price_rate is not None,
            strict_spot_cash=True,
        ),
        "expected_ledger_digest": content_digest(ledger.to_mapping()),
        "expected_fill_observations_digest": content_digest(batches),
    }
    args.update(overrides)
    return verify_strict_spot_cash_path(data, ledger, batches, **args)


def test_native_flat_replay_cash_path_and_fees():
    data = _fixture()
    ledger, batches, cost = _run(data)
    proof = _verify(data, ledger, batches, cost)
    assert proof.schema_version == "strict_spot_cash_path_v1"
    assert proof.interval_count == 4 and proof.fill_count == 2
    assert proof.minimum_cash == 499.5
    assert proof.final_cash == 999.0
    assert proof.total_execution_cost == 1.0
    assert proof.total_dividend == proof.total_cash_interest == 0.0


def test_actual_dividend_and_cash_interest_after_fill():
    dividend = np.zeros((6, 1))
    dividend[1, 0] = 2.0
    cash_rate = np.zeros(6)
    cash_rate[1] = 0.876
    data = _fixture(dividend=dividend, cash_rate=cash_rate)
    ledger, batches, cost = _run(data)
    proof = _verify(data, ledger, batches, cost)
    first = ledger.intervals.intervals[0]
    assert first.cash_before == 1000.0
    assert batches[0].fills[0].cash_after == 499.5
    assert first.interval_dividend == 10.0
    assert first.interval_cash_interest == pytest.approx(0.05095)
    assert first.cash_after == pytest.approx(509.55095)
    assert proof.minimum_cash == 499.5
    assert proof.total_dividend == 10.0
    assert proof.total_cash_interest == pytest.approx(0.05095)


@pytest.mark.parametrize(
    "field",
    [
        "expected_dataset_id",
        "expected_ledger_digest",
        "expected_fill_observations_digest",
        "expected_execution_policy_digest",
    ],
)
def test_external_pins_required(field):
    data = _fixture()
    ledger, batches, cost = _run(data)
    with pytest.raises(ValueError):
        _verify(data, ledger, batches, cost, **{field: "0" * 64})


def test_missing_or_reordered_batches_fail_even_with_resealed_digest():
    data = _fixture()
    ledger, batches, cost = _run(data)
    for changed in (
        batches[:-1],
        batches[::-1],
        (replace(batches[0], fills=(), interval_cost=0.0), *batches[1:]),
    ):
        with pytest.raises(ValueError):
            _verify(data, ledger, changed, cost)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cash_before", 1001.0),
        ("cash_after", 999.0),
        ("interval_dividend", 1.0),
        ("interval_cash_interest", 1.0),
        ("interval_funding", 1.0),
        ("interval_borrow_cost", 1.0),
        ("exact_quantities_after", ("4",)),
        ("total_cost_after", 5.0),
        ("termination_reason", "insolvency"),
    ],
)
def test_resealed_wrong_native_ledger_refused(field, value):
    data = _fixture()
    ledger, batches, cost = _run(data)
    iv = ledger.intervals
    first = replace(iv.intervals[0], **{field: value})
    changed = replace(
        ledger, intervals=replace(iv, intervals=(first, *iv.intervals[1:]))
    )
    with pytest.raises(ValueError):
        _verify(data, changed, batches, cost)


def test_legacy_execution_digest_is_not_strict_spot_evidence():
    data = _fixture()
    ledger, batches, cost = _run(data, strict=False)
    with pytest.raises(ValueError, match="identity"):
        _verify(data, ledger, batches, cost)


def test_no_claim_for_unsupported_split():
    factors = np.ones((6, 1))
    factors[2, 0] = 2.0
    data = _fixture(split_factor=factors)
    ledger, batches, cost = _run(data)
    with pytest.raises(ValueError, match="split"):
        _verify(data, ledger, batches, cost)


def test_dataset_identity_is_recomputed_and_roster_is_explicit():
    data = _fixture()
    ledger, batches, cost = _run(data)
    with pytest.raises(ValueError, match="roster"):
        _verify(data, ledger, batches, cost, expected_spot_symbols=("OTHER",))
    raw = _market()
    with pytest.raises(ValueError, match="Dataset identity"):
        _verify(raw, ledger, batches, cost)


def test_partial_terminal_path_is_not_complete():
    data = _fixture()
    ledger, batches, cost = _run(data)
    changed = replace(
        ledger, intervals=replace(ledger.intervals, terminal_exact_quantities=("1",))
    )
    with pytest.raises(ValueError, match="terminal"):
        _verify(data, changed, batches, cost)


def _multi_market(*, placeholder=False):
    shape = (6, 2)
    prices = np.tile(np.array([100.0, 200.0]), (6, 1))
    active = np.ones(shape, dtype=np.bool_)
    if placeholder:
        active[:, 1] = False
    return _fixture(
        symbols=("A", "B"),
        features=np.zeros((6, 2, 1), dtype=np.float32),
        feature_available=active[:, :, None].copy(),
        open=prices.copy(),
        high=prices.copy(),
        low=prices.copy(),
        close=prices.copy(),
        volume=np.where(active, 1000.0, 0.0),
        funding_rate=np.zeros(shape),
        tradable=active.copy(),
        asset_active=active.copy(),
        information_available=active.copy(),
        contract_multipliers=np.array([1.0, 2.0]),
    )


def test_multiple_symbols_and_multiplier_cash_oracle():
    data = _multi_market()
    ledger, batches, cost = _run(data)
    proof = _verify(data, ledger, batches, cost)
    rows = [r for b in batches for r in b.fills]
    assert len(rows) == proof.fill_count == 4
    assert proof.minimum_cash == pytest.approx(499.5)
    assert proof.final_cash == pytest.approx(999)
    assert proof.total_execution_cost == pytest.approx(1.0)
    assert {r.contract_multiplier for r in rows} == {1.0, 2.0}


def test_explicit_inactive_nonspot_placeholder_is_allowed_but_not_called_spot():
    from tests.evaluation.test_shared_cash_fee_ledger_oracle import FixedPortfolioTarget

    data = _multi_market(placeholder=True)
    ledger, batches, cost = _run(data, policy=FixedPortfolioTarget((0.5, 0.0)))
    proof = _verify(data, ledger, batches, cost, expected_spot_symbols=("A",))
    assert proof.symbols == ("A", "B") and proof.spot_symbols == ("A",)
    assert proof.final_cash == 999.0
    assert all(row.event.symbol_index == 0 for batch in batches for row in batch.fills)
    with pytest.raises(ValueError, match="fixed active spot roster"):
        _verify(data, ledger, batches, cost)
    with pytest.raises(ValueError, match="roster"):
        _verify(data, ledger, batches, cost, expected_spot_symbols=("B", "A"))


def test_explicitly_pinned_nonzero_source_dividend_cannot_be_erased():
    dividend = np.zeros((6, 1))
    dividend[1, 0] = 2.0
    data = _fixture(dividend=dividend)
    ledger, batches, cost = _run(data)
    iv = ledger.intervals
    # Re-seal a changed report, rather than only exercising the hash mismatch.
    first = replace(iv.intervals[0], interval_dividend=0.0, cash_after=499.5)
    changed = replace(
        ledger, intervals=replace(iv, intervals=(first, *iv.intervals[1:]))
    )
    with pytest.raises(ValueError, match="dividend does not match source"):
        _verify(data, changed, batches, cost)


def test_resealed_negative_fill_is_not_a_nonnegative_cash_proof():
    from tests.evaluation.test_shared_cash_fee_ledger_oracle import FixedPortfolioTarget

    data = _fixture()
    cost = replace(ExecutionCostConfig.zero(), allow_short=False, fee_rate=0.001)
    batches = []
    result = run_shared_portfolio_replay(
        data,
        FixedPortfolioTarget((1.0,)),
        start_index=0,
        stop_index=4,
        initial_capital=1000.0,
        execution_cost=cost,
        risk=PreTradeRisk(
            PreTradeRiskConfig(max_gross=1, max_abs_weight=1, max_turnover=None)
        ),
        settle_terminal_position=True,
        strict_spot_cash=False,
        capture_ledger_evidence=True,
        fill_cash_observer=batches.append,
    )
    assert result.ledger_evidence is not None
    assert any(row.cash_after < 0 for batch in batches for row in batch.fills)
    # Even if every identity is relabelled to the strict-policy digest and the
    # caller pins those new bytes, the actual negative cash must be refused.
    policy = runtime_execution_policy_digest(
        cost, funding_price_rate_present=False, strict_spot_cash=True
    )
    iv = result.ledger_evidence.intervals
    changed_iv = tuple(
        replace(
            row,
            order_events=tuple(
                replace(e, execution_policy_digest=policy) for e in row.order_events
            ),
        )
        for row in iv.intervals
    )
    changed = replace(
        result.ledger_evidence,
        intervals=replace(iv, execution_policy_digest=policy, intervals=changed_iv),
    )
    altered = tuple(
        replace(
            batch,
            execution_policy_digest=policy,
            fills=tuple(
                replace(row, event=replace(row.event, execution_policy_digest=policy))
                for row in batch.fills
            ),
        )
        for batch in batches
    )
    with pytest.raises(ValueError, match="nonnegative"):
        _verify(data, changed, altered, cost)


def test_empty_fill_batches_still_reconcile_source_interest():
    from tests.evaluation.test_shared_cash_fee_ledger_oracle import FixedPortfolioTarget

    rate = np.zeros(6)
    rate[1] = 0.876
    data = _fixture(cash_rate=rate)
    ledger, batches, cost = _run(data, policy=FixedPortfolioTarget((0.0,)))
    proof = _verify(data, ledger, batches, cost)
    assert proof.fill_count == 0 and len(batches) == 4
    assert proof.minimum_cash == 1000.0
    assert proof.final_cash == pytest.approx(1000.1)
    assert proof.total_cash_interest == pytest.approx(0.1)


def test_session_clock_is_not_silently_treated_as_continuous():
    from trade_rl.data.contracts import MarketCalendarKind

    data = _fixture(calendar_kind=MarketCalendarKind.SESSION)
    ledger, batches, cost = _run(data)
    with pytest.raises(ValueError, match="continuous regular"):
        _verify(data, ledger, batches, cost)


def test_first_source_row_has_no_carry_event_and_future_rows_are_not_processed():
    # These source rates are outside the execution window; they must not be
    # charged or cause refusal of otherwise valid processed-row evidence.
    rate = np.zeros(6)
    rate[0] = rate[5] = -0.876
    data = _fixture(cash_rate=rate)
    ledger, batches, cost = _run(data)
    proof = _verify(data, ledger, batches, cost)
    assert proof.total_cash_interest == 0.0
    assert proof.final_cash == 999.0


def test_unknown_native_ledger_schema_refused_even_if_outer_digest_matches():
    data = _fixture()
    ledger, batches, cost = _run(data)
    changed = replace(
        ledger, intervals=replace(ledger.intervals, schema_version="unknown")
    )
    with pytest.raises(ValueError, match="identity"):
        _verify(data, changed, batches, cost)


@pytest.mark.parametrize("unit", ["h", "s", "ms", "us"])
def test_cash_path_reconciles_native_event_ns_for_coarser_dataset_storage(unit):
    timestamps = _market().timestamps.astype(f"datetime64[{unit}]")
    data = _fixture(timestamps=timestamps)
    assert data.identity_verified
    assert data.timestamps.dtype == timestamps.dtype
    ledger, batches, cost = _run(data)
    proof = _verify(data, ledger, batches, cost)
    assert proof.interval_count == 4
    assert proof.fill_count == 2
    assert proof.minimum_cash == 499.5
    assert proof.final_cash == 999.0
    assert proof.total_execution_cost == 1.0
    for iv in ledger.intervals.intervals:
        for event in iv.order_events:
            assert event.timestamp_ns == int(
                timestamps[event.processing_index]
                .astype("datetime64[ns]")
                .astype(np.int64)
            )


def test_coarse_storage_units_are_not_accepted_as_native_event_nanoseconds():
    data = _fixture(timestamps=_market().timestamps.astype("datetime64[s]"))
    ledger, batches, cost = _run(data)
    native = ledger.intervals
    first = native.intervals[0]
    event = first.order_events[0]
    changed_event = replace(
        event,
        timestamp_ns=int(data.timestamps[event.processing_index].astype(np.int64)),
    )
    changed_first = replace(
        first, order_events=(changed_event, *first.order_events[1:])
    )
    changed = replace(
        ledger,
        intervals=replace(native, intervals=(changed_first, *native.intervals[1:])),
    )
    # The independent event-time check, not a stale outer digest, must reject.
    with pytest.raises(ValueError, match="native order event identity"):
        _verify(data, changed, batches, cost)
