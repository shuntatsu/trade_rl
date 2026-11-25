"""Core-owned spot decision reserve and native cash-account regression."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from tests.strategies.test_portfolio_policy_adapter import SYMBOLS, _dataset, _snapshot
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.risk.spot_cash import spot_cash_target_factor
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.allocation.buy_and_hold import BuyAndHoldPortfolioPolicy
from trade_rl.strategies.allocation.spot_cash import SpotCashBudgetPolicy
from trade_rl.strategies.portfolio import PortfolioTargetProposal


def test_full_spot_entry_reserves_fees_in_native_book():
    cost = ExecutionCostConfig(
        fee_rate=0.001,
        spread_rate=0,
        impact_rate=0,
        allow_short=False,
        partial_fill_carry=False,
    )
    data = replace(_dataset(), volume=np.full((3, 3), 1e8))
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1,
            max_abs_weight=1,
            max_turnover=None,
            drawdown_start=1,
            drawdown_stop=1,
        )
    )

    def replay(policy):
        return run_shared_portfolio_replay(
            data,
            policy,
            start_index=0,
            stop_index=2,
            initial_capital=1000,
            execution_cost=cost,
            risk=risk,
            capture_ledger_evidence=True,
        )

    unreserved = replay(BuyAndHoldPortfolioPolicy(symbols=SYMBOLS, gross_budget=1))
    assert unreserved.ledger_evidence.intervals.intervals[0].cash_after < 0
    wrapped = SpotCashBudgetPolicy(
        BuyAndHoldPortfolioPolicy(symbols=SYMBOLS, gross_budget=1), reserve_rate=0.001
    )
    result = replay(wrapped)
    rows = result.ledger_evidence.intervals.intervals
    assert min(row.cash_after for row in rows) >= -1e-9
    assert result.book.fill_count == 3
    assert rows[0].exact_quantities_after == rows[1].exact_quantities_after
    assert sum(result.decisions[0].proposal_weights) == pytest.approx(1 / 1.001)
    assert result.book.total_cost == pytest.approx(1000 * 0.001 / 1.001)


def test_rotation_cannot_count_unexecuted_sell_proceeds():
    current = (0.0, 0.0, 0.6)
    target = (0.3, 0.3, 0.0)
    factor = spot_cash_target_factor(
        current, target, cash_fraction=0.4, gross_limit=1.0, reserve_rate=0.002
    )
    assert factor == pytest.approx(0.4 / (0.6 * 1.002))
    assert 0.6 * factor * 1.002 <= 0.4 + 1e-15


def test_projection_randomized_budget_and_gross_oracles():
    rng = np.random.default_rng(53)
    for _ in range(300):
        current = tuple(rng.dirichlet(np.ones(5)) * rng.uniform(0, 0.99))
        target = tuple(rng.dirichlet(np.ones(5)) * rng.uniform(0, 2))
        cash = max(0, 1 - sum(current))
        rate = float(rng.uniform(0, 0.05))
        cap = float(rng.uniform(0.1, 1))
        f = spot_cash_target_factor(
            current, target, cash_fraction=cash, gross_limit=cap, reserve_rate=rate
        )
        buys = sum(max(f * t - c, 0) for c, t in zip(current, target, strict=True))
        assert 0 <= f <= 1
        assert buys * (1 + rate) <= cash + 1e-14
        assert (
            f * sum(target) <= max(0, (cap - rate * sum(current)) / (1 + rate)) + 1e-14
        )


@pytest.mark.parametrize(
    "current,target,kwargs",
    [
        ((), (), {}),
        ((0.0,), (1.0, 0.0), {}),
        ((-0.1,), (1.0,), {}),
        ((0.0,), (float("nan"),), {}),
        ((0.0,), (True,), {}),
        ((0.0,), (1.0,), {"cash_fraction": float("inf")}),
        ((0.0,), (1.0,), {"reserve_rate": 1.0}),
        ((0.0,), (1.0,), {"gross_limit": True}),
    ],
)
def test_projection_rejects_invalid_inputs(current, target, kwargs):
    params = {"cash_fraction": 1.0, "reserve_rate": 0.001, "gross_limit": 1.0}
    params.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        spot_cash_target_factor(current, target, **params)


@pytest.mark.parametrize(
    "value",
    [
        Fraction(-1, 10**400),
        Fraction(1, 3),
        2**53 + 1,
        np.uint64(2**53 + 1),
        np.timedelta64(1, "ns"),
        np.bool_(True),
    ],
)
@pytest.mark.parametrize(
    "field", ["current", "target", "cash_fraction", "gross_limit", "reserve_rate"]
)
def test_spot_projection_refuses_non_native_scalars_before_narrowing(value, field):
    args = dict(
        current=(0.0,),
        target=(0.5,),
        cash_fraction=1.0,
        gross_limit=1.0,
        reserve_rate=0.0,
    )
    args[field] = (value,) if field in {"current", "target"} else value
    with pytest.raises((ValueError, TypeError)):
        spot_cash_target_factor(**args)


def test_wide_spot_inputs_cannot_round_to_valid_values():
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip("platform longdouble is binary64")
    tiny_negative = -np.nextafter(np.longdouble(0), np.longdouble(1))
    above_one = np.nextafter(np.longdouble(1), np.longdouble(2))
    for changes in (
        {"current": (tiny_negative,)},
        {"target": (tiny_negative,)},
        {"cash_fraction": above_one},
        {"gross_limit": above_one},
        {"reserve_rate": tiny_negative},
    ):
        args = dict(
            current=(0.0,),
            target=(0.5,),
            cash_fraction=1.0,
            gross_limit=1.0,
            reserve_rate=0.0,
        )
        with pytest.raises(ValueError, match="binary64-compatible"):
            spot_cash_target_factor(**(args | changes))


def test_spot_projection_refuses_scalar_subclass_with_overridden_conversion():
    class LossyFloat(np.float64):
        def __float__(self):
            return 0.0

    with pytest.raises(TypeError, match="native real scalar"):
        spot_cash_target_factor(
            (LossyFloat(-0.5),),
            (0.5,),
            cash_fraction=1.0,
            gross_limit=1.0,
            reserve_rate=0.0,
        )


def test_spot_projection_accepts_native_scalar_types_without_changing_result():
    assert (
        spot_cash_target_factor(
            (np.float32(0.0), np.int32(0)),
            (np.float64(0.75), np.float32(0.25)),
            cash_fraction=np.int64(1),
            gross_limit=np.float32(1.0),
            reserve_rate=np.float64(0.0),
        )
        == 1.0
    )


def test_policy_preserves_holds_and_rejects_short_spot():
    s = _snapshot()
    s = replace(
        s,
        account=replace(
            s.account,
            weights=(0.2, 0.0, 0.0),
            quantities=(2.0, 0.0, 0.0),
            exact_quantities=("2", "0", "0"),
            cash=800.0,
            portfolio_value=1000.0,
            peak_value=1000.0,
            max_drawdown=0.0,
        ),
    )

    class Fixed:
        def __init__(self, p):
            self.p = p

        def decide_portfolio(self, snapshot):
            return self.p

    hold = PortfolioTargetProposal(s.account.weights, (0, 1, 2))
    assert (
        SpotCashBudgetPolicy(Fixed(hold), reserve_rate=0.001).decide_portfolio(s)
        is hold
    )
    with pytest.raises(ValueError, match="nonnegative"):
        SpotCashBudgetPolicy(
            Fixed(PortfolioTargetProposal((0.0, -0.1, 0.0))), reserve_rate=0.001
        ).decide_portfolio(s)


def test_projection_preserves_zero_cash_sells_and_rejects_overflow():
    assert (
        spot_cash_target_factor(
            (0.8, 0.0),
            (0.0, 0.0),
            cash_fraction=-0.02,
            gross_limit=1.0,
            reserve_rate=0.001,
        )
        == 1.0
    )
    assert (
        spot_cash_target_factor(
            (0.8, 0.0),
            (0.0, 0.8),
            cash_fraction=0.0,
            gross_limit=1.0,
            reserve_rate=0.001,
        )
        == 0.0
    )
    with pytest.raises(ValueError, match="gross"):
        spot_cash_target_factor(
            (0.0, 0.0),
            (1.7e308, 1.7e308),
            cash_fraction=1.0,
            gross_limit=1.0,
            reserve_rate=0.001,
        )


def test_spot_wrapper_uses_native_drawdown_scale_once_and_rejects_inconsistent_hold():
    s = _snapshot()
    s = replace(
        s,
        risk=replace(
            s.risk,
            max_gross=1.0,
            max_abs_weight=1.0,
            drawdown_start=0.1,
            drawdown_stop=0.3,
            max_turnover=None,
        ),
        account=replace(
            s.account,
            weights=(0.2, 0.0, 0.0),
            quantities=(2.0, 0.0, 0.0),
            exact_quantities=("2", "0", "0"),
            cash=800.0,
            portfolio_value=1000.0,
            peak_value=1250.0,
            max_drawdown=0.2,
        ),
    )

    class Fixed:
        def __init__(self, p):
            self.p = p

        def decide_portfolio(self, snapshot):
            return self.p

    p = PortfolioTargetProposal((1.0, 1.0, 1.0))
    wrapped = SpotCashBudgetPolicy(Fixed(p), reserve_rate=0.001)
    target = wrapped.decide_portfolio(s)
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=1,
            max_abs_weight=1,
            max_turnover=None,
            drawdown_start=0.1,
            drawdown_stop=0.3,
        )
    )
    native = risk.constrain(
        np.array(target.target_weights),
        current=np.array(s.account.weights),
        drawdown=0.2,
    )
    # Gross reservation binds before the native risk cap. Check the
    # wrapper output, not merely a downstream cap that masks mutations.
    expected = (1 - 0.001 * 0.2) / (1.001 * 1.5)
    assert target.target_weights == pytest.approx((expected,) * 3)
    assert native.weights == pytest.approx((1 / 6,) * 3)
    with pytest.raises(ValueError, match="held spot target"):
        SpotCashBudgetPolicy(
            Fixed(PortfolioTargetProposal((0.3, 0.0, 0.0), (0,))), reserve_rate=0.001
        ).decide_portfolio(s)


@pytest.mark.parametrize(
    "held,weights,expected",
    [
        ((), (0.0, 1.0, 0.0), (0.0, 0.4, 0.0)),
        ((0,), (0.75, 1.0, 0.0), (0.6, 0.4, 0.0)),
    ],
)
def test_drawdown_cash_binding_releases_holds_and_scales_exactly_once(
    held, weights, expected
):
    s = _snapshot()
    s = replace(
        s,
        risk=replace(
            s.risk,
            max_gross=1.0,
            max_abs_weight=1.0,
            max_turnover=None,
            drawdown_start=0.0,
            drawdown_stop=1.0,
        ),
        account=replace(
            s.account,
            weights=(0.75, 0.0, 0.0),
            quantities=(7.5, 0.0, 0.0),
            exact_quantities=("15/2", "0", "0"),
            cash=250.0,
            portfolio_value=1000.0,
            peak_value=2000.0,
            max_drawdown=0.5,
        ),
    )

    class Fixed:
        def decide_portfolio(self, snapshot):
            return PortfolioTargetProposal(weights, held)

    # Literal oracle: cash 1/4, buy fee 1/4, thus buy budget 1/5.
    # At scale 1/2 the new buy target is 1/2, so factor = 2/5.
    # With a held 3/4 weight, its scaled effective target is 3/10.
    # Release the hold and invert drawdown scale exactly once.
    result = SpotCashBudgetPolicy(Fixed(), reserve_rate=0.25).decide_portfolio(s)
    assert result.target_weights == pytest.approx(expected)
    assert result.hold_current_quantity_symbols == ()
    for cap in (1.0, 0.5):
        risk = PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=1.0,
                max_abs_weight=cap,
                max_turnover=None,
                drawdown_start=0.0,
                drawdown_stop=1.0,
            )
        )
        native = risk.constrain(
            np.array(result.target_weights),
            current=np.array(s.account.weights),
            drawdown=0.5,
        )
        # Native risk clips each proposal before multiplying by scale.
        expected_native = tuple(min(v, cap) * 0.5 for v in expected)
        assert native.weights == pytest.approx(expected_native)
        assert native.weights[1] == pytest.approx(0.2)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_native_account_and_proposal_refuse_nonfinite_before_wrapper(bad):
    snapshot = _snapshot()
    for field in ("quantities", "weights"):
        with pytest.raises(ValueError, match="finite"):
            replace(snapshot.account, **{field: (bad, 0.0, 0.0)})
    with pytest.raises(ValueError, match="finite"):
        PortfolioTargetProposal((bad, 0.0, 0.0))


def test_native_proposal_refuses_negative_hold_index():
    with pytest.raises(ValueError, match="non-negative"):
        PortfolioTargetProposal((0.0, 0.0, 0.0), (-1,))


@pytest.mark.parametrize(
    "changes",
    [
        {"insolvent": True},
        {"termination_reason": "margin_call"},
    ],
)
def test_terminated_native_book_is_held_without_invoking_inner(changes):
    s = _snapshot()

    class Never:
        def decide_portfolio(self, snapshot):
            raise AssertionError("terminated account invoked inner policy")

    snap = replace(s, account=replace(s.account, **changes))
    p = SpotCashBudgetPolicy(Never(), reserve_rate=0.001).decide_portfolio(snap)
    assert p.target_weights == snap.account.weights
    assert p.hold_current_quantity_symbols == tuple(range(len(snap.symbols)))


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"portfolio_value": 0.0}, "portfolio value"),
        ({"portfolio_value": -1.0}, "portfolio value"),
        ({"weights": (0.2, 0.0, 0.0), "quantities": (2.0, -1.0, 0.0)}, "nonnegative"),
    ],
)
def test_invalid_native_book_refuses_before_inner(changes, reason):
    s = _snapshot()

    class Never:
        def decide_portfolio(self, snapshot):
            raise AssertionError("invalid account invoked inner policy")

    snap = replace(s, account=replace(s.account, **changes))
    with pytest.raises(ValueError, match=reason):
        SpotCashBudgetPolicy(Never(), reserve_rate=0.001).decide_portfolio(snap)


def test_pending_order_refuses_before_inner():
    from trade_rl.strategies.portfolio import PortfolioPendingOrderSnapshot

    s = _snapshot()
    account = replace(
        s.account,
        weights=(0.2, 0.0, 0.0),
        quantities=(2.0, 0.0, 0.0),
        exact_quantities=("2", "0", "0"),
    )
    pending = PortfolioPendingOrderSnapshot(
        symbol_index=0,
        order_type="market",
        time_in_force="gtc",
        requested_quantity=1.0,
        remaining_quantity=1.0,
        status="eligible",
        submit_index=0,
        eligible_index=1,
        last_processed_index=None,
        trigger_index=None,
        expiry_index=None,
        reduce_only=False,
    )

    class Never:
        def decide_portfolio(self, snapshot):
            raise AssertionError("pending account invoked inner policy")

    with pytest.raises(ValueError, match="active orders"):
        SpotCashBudgetPolicy(Never(), reserve_rate=0.001).decide_portfolio(
            replace(s, account=account, active_orders=(pending,))
        )
