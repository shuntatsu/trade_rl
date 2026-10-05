"""Independent synthetic oracles for one-account after-cost allocation."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from trade_rl.strategies.allocation import (
    AfterCostTargetAllocator,
    AllocationContext,
    AllocationInputs,
    AllocationProposal,
)

_TIME = np.datetime64("2024-01-01T00:00:00", "ns")


def _inputs(**changes: object) -> AllocationInputs:
    return replace(
        AllocationInputs(
            symbol="SYNTHETIC",
            decision_time=_TIME,
            available_at=_TIME,
            horizon_end=_TIME + np.timedelta64(24, "h"),
            source_identity="synthetic-simple-return-v1",
            expected_simple_return=0.0,
        ),
        **changes,
    )


def _context(weight: float = 0.0, **changes: object) -> AllocationContext:
    return replace(
        AllocationContext(
            account_id="independent-synthetic-account",
            symbol="SYNTHETIC",
            decision_time=_TIME,
            state_digest="a" * 64,
            current_weight=weight,
            quantity="0",
            cash=100.0,
            equity=100.0,
            pending_remaining="0",
        ),
        **changes,
    )


def _reference_score(
    inputs: AllocationInputs,
    *,
    weight: float,
    previous_weight: float,
    risk_aversion: float,
) -> float:
    """Sum separately computed contributions without production score/solver calls."""
    bought = max(0.0, weight - previous_weight)
    sold = max(0.0, previous_weight - weight)
    short = max(0.0, -weight)
    cash_growth = (1.0 - weight) * inputs.cash_return
    asset_growth = weight * inputs.expected_simple_return
    entry_buy_fee = bought * inputs.buy_cost
    entry_sell_fee = sold * inputs.sell_cost
    liquidation_fee = abs(weight) * inputs.exit_cost
    funding_paid = weight * inputs.funding_return
    borrowed_cost = short * inputs.borrow_return
    variance_preference = risk_aversion * inputs.return_variance * weight**2
    return math.fsum(
        (
            cash_growth,
            asset_growth,
            -entry_buy_fee,
            -entry_sell_fee,
            -liquidation_fee,
            -funding_paid,
            -borrowed_cost,
            -variance_preference,
        )
    )


@pytest.mark.parametrize(
    ("forecast", "target"),
    [(0.04, 1.0), (-0.04, -1.0)],
)
def test_useful_signal_trades_after_entry_and_terminal_costs(
    forecast: float, target: float
) -> None:
    proposal = AfterCostTargetAllocator().propose(
        _inputs(
            expected_simple_return=forecast,
            buy_cost=0.002,
            sell_cost=0.002,
            exit_cost=0.003,
        ),
        _context(),
    )
    assert isinstance(proposal, AllocationProposal)
    assert proposal.target_weight == target
    assert not proposal.is_hold
    assert proposal.objective_value == pytest.approx(0.035)


def test_fee_only_market_prefers_cash_over_unnecessary_turnover() -> None:
    proposal = AfterCostTargetAllocator().propose(
        _inputs(buy_cost=0.001, sell_cost=0.003, exit_cost=0.002), _context()
    )
    assert proposal.target_weight == 0.0
    assert proposal.is_hold
    assert proposal.objective_value == 0.0


def test_small_signal_changes_keep_actual_weight_in_the_no_trade_region() -> None:
    proposal = AfterCostTargetAllocator().propose(
        _inputs(
            expected_simple_return=0.0015,
            buy_cost=0.001,
            sell_cost=0.001,
            exit_cost=0.001,
        ),
        _context(0.25),
    )
    assert proposal.target_weight == 0.25
    assert proposal.is_hold
    assert proposal.objective_value == pytest.approx(0.000125)


def test_reversal_charges_close_and_open_notional_then_future_exit() -> None:
    proposal = AfterCostTargetAllocator().propose(
        _inputs(expected_simple_return=-0.05, sell_cost=0.004, exit_cost=0.002),
        _context(0.5),
    )
    assert proposal.target_weight == -1.0
    # Sell 1.5 units of current-equity notional, then liquidate one short unit.
    assert proposal.objective_value == pytest.approx(0.05 - 1.5 * 0.004 - 0.002)


def test_asymmetric_transaction_costs_change_the_optimal_direction() -> None:
    expensive_buy = _inputs(expected_simple_return=0.01, buy_cost=0.02)
    cheap_sell = _inputs(expected_simple_return=-0.01, sell_cost=0.002)
    allocator = AfterCostTargetAllocator()
    assert allocator.propose(expensive_buy, _context()).target_weight == 0.0
    assert allocator.propose(cheap_sell, _context()).target_weight == -1.0


def test_horizon_variance_preference_has_a_non_boundary_optimum() -> None:
    proposal = AfterCostTargetAllocator(risk_aversion=2.0).propose(
        _inputs(expected_simple_return=0.03, return_variance=0.01), _context()
    )
    # dU/dw = 0.03 - 2 * 2 * 0.01 * w.
    assert proposal.target_weight == pytest.approx(0.75)
    assert proposal.objective_value == pytest.approx(0.01125)


def test_stationary_point_on_the_short_side_includes_all_cost_terms() -> None:
    inputs = _inputs(
        expected_simple_return=-0.06,
        return_variance=0.02,
        sell_cost=0.004,
        exit_cost=0.006,
        funding_return=0.003,
        borrow_return=0.007,
        cash_return=0.005,
    )
    proposal = AfterCostTargetAllocator(risk_aversion=2.0).propose(inputs, _context())
    # For w < 0: dU/dw = -0.06 - 0.005 + 0.004 + 0.006 - 0.003 + 0.007 - 0.08*w.
    assert proposal.target_weight == pytest.approx(-0.6375)
    assert proposal.objective_value == pytest.approx(
        _reference_score(inputs, weight=-0.6375, previous_weight=0, risk_aversion=2)
    )


def test_cash_growth_and_signed_funding_are_in_the_same_horizon_objective() -> None:
    inputs = _inputs(
        expected_simple_return=0.015,
        cash_return=0.02,
        funding_return=-0.01,
        exit_cost=0.001,
    )
    proposal = AfterCostTargetAllocator(lower_weight=0.0).propose(inputs, _context())
    assert proposal.target_weight == 1.0
    assert proposal.objective_value == pytest.approx(0.024)


def test_mathematically_flat_segment_holds_despite_endpoint_rounding() -> None:
    inputs = _inputs(expected_simple_return=0.01, buy_cost=0.01)
    proposal = AfterCostTargetAllocator(lower_weight=0).propose(
        inputs, _context(current_weight=0.3)
    )
    assert proposal.target_weight == 0.3
    assert proposal.is_hold


def test_equal_utility_tie_uses_exact_turnover_outside_exposure_cap() -> None:
    proposal = AfterCostTargetAllocator(lower_weight=0, upper_weight=1e-16).propose(
        _inputs(expected_simple_return=0), _context(weight=2)
    )
    assert proposal.target_weight == 1e-16


@pytest.mark.parametrize("field", ["decision_time", "available_at", "horizon_end"])
@pytest.mark.parametrize(
    "time", [np.datetime64("3000-01-01", "D"), np.datetime64("9999-01-01", "D")]
)
def test_time_conversion_cannot_wrap_future_inputs_into_the_past(field, time) -> None:
    with pytest.raises(ValueError, match="time|horizon|available"):
        _inputs(**{field: time})


def test_context_time_cannot_wrap_and_subnanosecond_precision_cannot_be_lost() -> None:
    with pytest.raises(ValueError, match="time"):
        _context(decision_time=np.datetime64("3000-01-01", "D"))
    with pytest.raises(ValueError, match="time"):
        _inputs(available_at=np.datetime64("1970-01-01T00:00:00.000000000001", "ps"))


@pytest.mark.parametrize(
    ("forecast", "target"),
    [(0.1, 0.4), (-0.1, -0.3)],
)
def test_explicit_weight_limits_bound_the_optimum(
    forecast: float, target: float
) -> None:
    allocator = AfterCostTargetAllocator(lower_weight=-0.3, upper_weight=0.4)
    proposal = allocator.propose(_inputs(expected_simple_return=forecast), _context())
    assert proposal.target_weight == target


@pytest.mark.parametrize(("forecast", "target"), [(0.1, 0.5), (-0.1, 0.0)])
def test_turnover_cap_intersects_bounds_around_actual_weight(
    forecast: float, target: float
) -> None:
    allocator = AfterCostTargetAllocator(max_turnover=0.25)
    proposal = allocator.propose(
        _inputs(expected_simple_return=forecast), _context(0.25)
    )
    assert proposal.target_weight == target


def test_exact_objective_tie_prefers_hold() -> None:
    proposal = AfterCostTargetAllocator().propose(
        _inputs(expected_simple_return=0.125, buy_cost=0.125), _context(0.25)
    )
    assert proposal.target_weight == 0.25
    assert proposal.is_hold
    assert proposal.objective_value == 0.03125


def test_strictly_positive_marginal_profit_is_not_hidden_by_a_tie_epsilon() -> None:
    proposal = AfterCostTargetAllocator().propose(
        _inputs(expected_simple_return=math.nextafter(0.5, math.inf), buy_cost=0.5),
        _context(),
    )
    assert proposal.target_weight == 1.0
    assert not proposal.is_hold
    assert proposal.objective_value > 0.0


def test_infeasible_actual_hold_is_not_selected_even_when_all_scores_tie() -> None:
    proposal = AfterCostTargetAllocator().propose(_inputs(), _context(2.0))
    assert proposal.target_weight == 1.0
    assert not proposal.is_hold


def test_singleton_feasible_interval_is_a_valid_target() -> None:
    proposal = AfterCostTargetAllocator(lower_weight=0.4, upper_weight=0.4).propose(
        _inputs(expected_simple_return=-0.5), _context()
    )
    assert proposal.target_weight == 0.4


def test_empty_turnover_and_weight_intersection_fails_closed() -> None:
    with pytest.raises(ValueError):
        AfterCostTargetAllocator(max_turnover=0.0).propose(_inputs(), _context(2.0))


@pytest.mark.parametrize("seed", range(16))
def test_concave_allocator_matches_an_independent_bounded_numerical_oracle(
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    previous = float(rng.uniform(-0.7, 0.7))
    turnover = float(rng.uniform(0.05, 0.5))
    aversion = float(rng.uniform(0.0, 3.0))
    inputs = _inputs(
        expected_simple_return=float(rng.uniform(-0.08, 0.08)),
        return_variance=float(rng.uniform(0.001, 0.06)),
        buy_cost=float(rng.uniform(0, 0.015)),
        sell_cost=float(rng.uniform(0, 0.015)),
        exit_cost=float(rng.uniform(0, 0.01)),
        funding_return=float(rng.uniform(-0.01, 0.01)),
        borrow_return=float(rng.uniform(0, 0.01)),
        cash_return=float(rng.uniform(-0.01, 0.01)),
    )
    allocator = AfterCostTargetAllocator(
        lower_weight=-0.8,
        upper_weight=0.9,
        max_turnover=turnover,
        risk_aversion=aversion,
    )
    proposal = allocator.propose(inputs, _context(previous))
    lower = max(-0.8, previous - turnover)
    upper = min(0.9, previous + turnover)

    def reference(weight: float) -> float:
        return _reference_score(
            inputs, weight=weight, previous_weight=previous, risk_aversion=aversion
        )

    numerical = minimize_scalar(
        lambda weight: -reference(float(weight)),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-14},
    )
    assert numerical.success
    assert lower <= proposal.target_weight <= upper
    expected_value = reference(proposal.target_weight)
    assert proposal.objective_value == pytest.approx(expected_value, abs=1e-14)
    independently_best = max(reference(lower), reference(upper), -float(numerical.fun))
    assert proposal.objective_value >= independently_best - 1e-12


@pytest.mark.parametrize("unit", ["log_return", "simple_return", "bps", ""])
def test_non_expected_simple_return_units_are_rejected(unit: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        AfterCostTargetAllocator().propose(_inputs(return_unit=unit), _context())


@pytest.mark.parametrize("offset", [-1, 1])
def test_stale_or_future_forecast_as_of_does_not_match_the_decision(
    offset: int,
) -> None:
    as_of = _TIME + np.timedelta64(offset, "s")
    with pytest.raises(ValueError):
        AfterCostTargetAllocator().propose(
            _inputs(decision_time=as_of, available_at=as_of), _context()
        )


def test_forecast_for_another_symbol_is_rejected() -> None:
    with pytest.raises(ValueError):
        AfterCostTargetAllocator().propose(_inputs(symbol="OTHER"), _context())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("available_at", _TIME + np.timedelta64(1, "ns")),
        ("horizon_end", _TIME),
        ("horizon_end", _TIME - np.timedelta64(1, "ns")),
        ("decision_time", np.datetime64("NaT")),
        ("available_at", np.datetime64("NaT")),
        ("horizon_end", np.datetime64("NaT")),
        ("symbol", ""),
        ("source_identity", ""),
        ("expected_simple_return", -1.00001),
    ],
)
def test_invalid_input_identity_and_clocks_fail_closed(
    field: str, value: object
) -> None:
    with pytest.raises((ValueError, TypeError)):
        _inputs(**{field: value})


def test_minus_one_is_a_valid_simple_return_boundary() -> None:
    proposal = AfterCostTargetAllocator(lower_weight=0.0).propose(
        _inputs(expected_simple_return=-1.0), _context()
    )
    assert proposal.target_weight == 0.0


@pytest.mark.parametrize(
    "field",
    [
        "expected_simple_return",
        "return_variance",
        "buy_cost",
        "sell_cost",
        "exit_cost",
        "funding_return",
        "borrow_return",
        "cash_return",
    ],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True])
def test_nonfinite_or_boolean_input_numbers_fail_closed(
    field: str, value: object
) -> None:
    with pytest.raises((ValueError, TypeError)):
        _inputs(**{field: value})


@pytest.mark.parametrize(
    "field", ["return_variance", "buy_cost", "sell_cost", "exit_cost", "borrow_return"]
)
def test_negative_variance_or_costs_fail_closed(field: str) -> None:
    with pytest.raises(ValueError):
        _inputs(**{field: -0.001})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("account_id", ""),
        ("symbol", ""),
        ("state_digest", ""),
        ("state_digest", "x" * 64),
        ("state_digest", "a" * 63),
        ("decision_time", np.datetime64("NaT")),
        ("equity", 0.0),
        ("equity", -1.0),
    ],
)
def test_invalid_account_context_fails_closed(field: str, value: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        _context(**{field: value})


@pytest.mark.parametrize("field", ["current_weight", "cash", "equity"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True])
def test_nonfinite_or_boolean_account_numbers_fail_closed(
    field: str, value: object
) -> None:
    with pytest.raises((ValueError, TypeError)):
        _context(**{field: value})


@pytest.mark.parametrize(
    "field", ["lower_weight", "upper_weight", "max_turnover", "risk_aversion"]
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True])
def test_nonfinite_or_boolean_allocator_parameters_fail_closed(
    field: str, value: object
) -> None:
    with pytest.raises((ValueError, TypeError)):
        AfterCostTargetAllocator(**{field: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"lower_weight": 1.0, "upper_weight": 0.0},
        {"max_turnover": -0.1},
        {"risk_aversion": -0.1},
    ],
)
def test_invalid_allocator_limits_fail_closed(changes: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        AfterCostTargetAllocator(**changes)


def test_decision_contract_objects_are_frozen() -> None:
    inputs = _inputs()
    context = _context()
    allocator = AfterCostTargetAllocator()
    proposal = allocator.propose(inputs, context)
    for value, field in (
        (inputs, "expected_simple_return"),
        (context, "cash"),
        (allocator, "risk_aversion"),
        (proposal, "target_weight"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, 1.0)
