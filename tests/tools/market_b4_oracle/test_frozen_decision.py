from __future__ import annotations

import inspect
import math
import random
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

import tools.market_b4_oracle.frozen_decision as _FROZEN_DECISION
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.strategies.allocation import (
    AfterCostTargetAllocator,
    AllocationContext,
    AllocationInputs,
)
from trade_rl.strategies.allocation_action import (
    AllocationActionContract,
    AllocationDecision,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

_FROZEN_RISK_POLICY = {
    "max_gross": 1.0,
    "max_abs_weight": 1.0,
    "max_turnover": None,
    "drawdown_start": 0.10,
    "drawdown_stop": 0.20,
    "emergency_turnover_override": True,
    "fail_closed_tolerance": 1e-10,
}
_RISK = PreTradeRisk(PreTradeRiskConfig(**_FROZEN_RISK_POLICY))
_TOLERANCE = 1e-12


def _production_projection(
    proposal: float,
    current: float,
    drawdown: float,
    emergency_flatten: bool,
    reduce_only: bool,
    symbol_index: int,
):
    targets = np.zeros(2, dtype=np.float64)
    currents = np.zeros(2, dtype=np.float64)
    emergency = np.zeros(2, dtype=np.bool_)
    reduce_mask = np.zeros(2, dtype=np.bool_)
    targets[symbol_index] = proposal
    currents[symbol_index] = current
    emergency[symbol_index] = emergency_flatten
    reduce_mask[symbol_index] = reduce_only
    return _RISK.constrain(
        targets,
        current=currents,
        drawdown=drawdown,
        emergency_flatten_mask=emergency,
        reduce_only_mask=reduce_mask,
    )


def _assert_matches_both_vector_slots(
    proposal: float,
    current: float,
    drawdown: float,
    *,
    emergency_flatten: bool = False,
    reduce_only: bool = False,
):
    scalar = _FROZEN_DECISION.project_frozen_single_symbol(
        proposal,
        current_weight=current,
        drawdown=drawdown,
        emergency_flatten=emergency_flatten,
        reduce_only=reduce_only,
    )
    for symbol_index in (0, 1):
        vector = _production_projection(
            proposal,
            current,
            drawdown,
            emergency_flatten,
            reduce_only,
            symbol_index,
        )
        assert vector.weights[symbol_index] == scalar.weights
        assert vector.proposal_weights[symbol_index] == scalar.proposal_weights
        assert vector.pretrade_weights[symbol_index] == scalar.pretrade_weights
        assert vector.requested_turnover == scalar.requested_turnover
        assert vector.constrained_turnover == scalar.constrained_turnover
        assert vector.was_constrained is scalar.was_constrained
        assert vector.reasons == scalar.reasons
        assert vector.risk_scale == scalar.risk_scale
        assert vector.projection_l1 == scalar.projection_l1
        assert vector.turnover_overridden is scalar.turnover_overridden
        assert vector.max_gross == scalar.max_gross
        assert vector.drawdown_budget == scalar.drawdown_budget
        assert vector.max_abs_weight == scalar.max_abs_weight
        assert vector.fail_closed_tolerance == scalar.fail_closed_tolerance
        assert abs(scalar.weights) <= scalar.risk_scale + 1e-10
    return scalar


def _assert_both_slots_reject(
    proposal: float,
    current: float,
    drawdown: float,
    *,
    emergency_flatten: bool = False,
    reduce_only: bool = False,
) -> None:
    with pytest.raises(ValueError):
        _FROZEN_DECISION.project_frozen_single_symbol(
            proposal,
            current_weight=current,
            drawdown=drawdown,
            emergency_flatten=emergency_flatten,
            reduce_only=reduce_only,
        )
    for symbol_index in (0, 1):
        with pytest.raises(ValueError):
            _production_projection(
                proposal,
                current,
                drawdown,
                emergency_flatten,
                reduce_only,
                symbol_index,
            )


def test_frozen_decision_uses_current_checkout_native_source() -> None:
    independent_source = inspect.getsource(_FROZEN_DECISION)
    assert "trade_rl" not in independent_source
    assert (
        Path(inspect.getfile(PreTradeRisk)).resolve()
        == (REPOSITORY_ROOT / "trade_rl" / "risk" / "pretrade.py").resolve()
    )
    assert (
        Path(inspect.getfile(AllocationDecision)).resolve()
        == (
            REPOSITORY_ROOT / "trade_rl" / "strategies" / "allocation_action.py"
        ).resolve()
    )
    assert _FROZEN_RISK_POLICY == {
        "max_gross": 1.0,
        "max_abs_weight": 1.0,
        "max_turnover": None,
        "drawdown_start": 0.10,
        "drawdown_stop": 0.20,
        "emergency_turnover_override": True,
        "fail_closed_tolerance": 1e-10,
    }


@pytest.mark.parametrize(
    ("drawdown", "expected_scale"),
    [
        (0.0, 1.0),
        (0.10, 1.0),
        (
            math.nextafter(0.10, math.inf),
            1.0 - (math.nextafter(0.10, math.inf) - 0.10) / 0.10,
        ),
        (0.15, 1.0 - (0.15 - 0.10) / (0.20 - 0.10)),
        (
            math.nextafter(0.20, 0.0),
            1.0 - (math.nextafter(0.20, 0.0) - 0.10) / 0.10,
        ),
        (0.20, 0.0),
        (1.0, 0.0),
    ],
)
def test_drawdown_scale_thresholds_and_linear_interior(
    drawdown: float, expected_scale: float
) -> None:
    scale = _FROZEN_DECISION.frozen_risk_scale(drawdown)
    assert scale == expected_scale
    production_scale = _RISK.risk_scale(drawdown)
    assert scale == production_scale
    result = _assert_matches_both_vector_slots(0.8, 0.2, drawdown)
    assert result.risk_scale == expected_scale
    assert result.weights == 0.8 * expected_scale
    if expected_scale < 1.0:
        assert "drawdown_deleveraging" in result.reasons


def test_hold_proposal_above_hard_limit_is_clipped_and_records_both_turnovers() -> None:
    result = _assert_matches_both_vector_slots(1.5, 1.5, 0.0)
    assert result.weights == 1.0
    assert result.requested_turnover == 0.0
    assert result.constrained_turnover == 0.5
    assert result.projection_l1 == 0.5
    assert result.was_constrained is True
    assert result.reasons == ("max_abs_weight",)


@pytest.mark.parametrize(
    (
        "proposal",
        "current",
        "drawdown",
        "emergency_flatten",
        "reduce_only",
        "expected_weight",
        "expected_reasons",
        "expected_override",
    ),
    [
        (
            0.75,
            0.0,
            0.0,
            False,
            True,
            0.0,
            ("reduce_only", "reduce_only_satisfied"),
            False,
        ),
        (
            0.8,
            0.5,
            0.0,
            False,
            True,
            0.5,
            ("reduce_only", "reduce_only_satisfied"),
            False,
        ),
        (0.2, 0.5, 0.0, False, True, 0.2, ("reduce_only",), False),
        (
            0.8,
            0.5,
            0.0,
            True,
            False,
            0.0,
            ("emergency_flatten", "emergency_turnover_override"),
            True,
        ),
        (
            0.8,
            0.5,
            0.0,
            True,
            True,
            0.0,
            ("reduce_only", "emergency_flatten", "emergency_turnover_override"),
            True,
        ),
        (0.5, 0.5e-13, 0.0, True, False, 0.0, ("emergency_flatten",), False),
        (0.0, 0.0, 0.15, False, False, 0.0, ("drawdown_deleveraging",), False),
    ],
)
def test_reduce_only_emergency_and_drawdown_combinations(
    proposal: float,
    current: float,
    drawdown: float,
    emergency_flatten: bool,
    reduce_only: bool,
    expected_weight: float,
    expected_reasons: tuple[str, ...],
    expected_override: bool,
) -> None:
    result = _assert_matches_both_vector_slots(
        proposal,
        current,
        drawdown,
        emergency_flatten=emergency_flatten,
        reduce_only=reduce_only,
    )
    assert result.weights == expected_weight
    assert result.reasons == expected_reasons
    assert result.turnover_overridden is expected_override
    if emergency_flatten:
        assert result.requested_turnover == abs(proposal - current)


def test_reduce_only_wrong_sign_is_rejected_at_either_vector_slot() -> None:
    _assert_both_slots_reject(0.5, -0.5, 0.0, reduce_only=True)


def test_reduce_only_near_zero_inventory_flattens_before_sign_check() -> None:
    result = _assert_matches_both_vector_slots(-0.5, 0.5e-13, 0.0, reduce_only=True)
    assert result.weights == 0.0
    assert result.reasons == ("reduce_only", "reduce_only_satisfied")


@pytest.mark.parametrize("bad_bool", [0, 1, np.bool_(True), None])
def test_frozen_decision_requires_exact_python_booleans(bad_bool: object) -> None:
    with pytest.raises(ValueError):
        _FROZEN_DECISION.project_frozen_single_symbol(
            0.2,
            current_weight=0.0,
            drawdown=0.0,
            emergency_flatten=bad_bool,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        _FROZEN_DECISION.project_frozen_single_symbol(
            0.2,
            current_weight=0.0,
            drawdown=0.0,
            reduce_only=bad_bool,  # type: ignore[arg-type]
        )


def test_frozen_decision_result_is_frozen_and_has_narrow_arithmetic_scope() -> None:
    result = _assert_matches_both_vector_slots(0.2, 0.0, 0.0)
    assert result.scope == "SINGLE_ACTIVE_SYMBOL_FROZEN_H_RISK_ARITHMETIC_ONLY"
    assert not hasattr(result, "account_id")
    assert not hasattr(result, "action")
    with pytest.raises(FrozenInstanceError):
        result.weights = 0.5  # type: ignore[misc]


@pytest.mark.parametrize("seed", range(128))
def test_manufactured_finite_inputs_match_h_vector_projection(seed: int) -> None:
    rng = random.Random(0xF20A1 + seed)
    proposal = rng.uniform(-1.5, 1.5)
    current = rng.uniform(-1.5, 1.5)
    drawdown = rng.choice(
        (
            0.0,
            0.10,
            math.nextafter(0.10, math.inf),
            0.15,
            math.nextafter(0.20, 0.0),
            0.20,
            1.0,
        )
    )
    emergency_flatten = bool(rng.getrandbits(1))
    reduce_only = bool(rng.getrandbits(1))
    rejects = (
        reduce_only
        and not emergency_flatten
        and abs(current) > _TOLERANCE
        and proposal * current < -_TOLERANCE
    )
    if rejects:
        _assert_both_slots_reject(
            proposal,
            current,
            drawdown,
            emergency_flatten=emergency_flatten,
            reduce_only=reduce_only,
        )
        return

    _assert_matches_both_vector_slots(
        proposal,
        current,
        drawdown,
        emergency_flatten=emergency_flatten,
        reduce_only=reduce_only,
    )


# The combined target tests below compare Q's current public allocation/risk interfaces
# only as manufactured arithmetic differentials. No native execution is called.


def _new_inputs(**overrides: float):
    values = {
        "expected_simple_return": 0.06,
        "return_variance": 0.2,
        "buy_cost": 0.01,
        "sell_cost": 0.01,
        "exit_cost": 0.0,
        "funding_return": 0.0,
        "borrow_return": 0.0,
        "cash_return": 0.0,
    }
    values.update(overrides)
    return _FROZEN_DECISION.AllocationAssumptions(**values)


def _native_composition(
    inputs,
    *,
    current: float,
    lower: float,
    upper: float,
    drawdown: float,
    role: str,
    emergency: bool = False,
    reduce_only: bool = False,
    symbol_index: int = 0,
):
    decision_time = np.datetime64("2025-01-01T00:00:00", "ns")
    h_inputs = AllocationInputs(
        symbol="BTC",
        decision_time=decision_time,
        available_at=decision_time,
        horizon_end=decision_time + np.timedelta64(1, "D"),
        source_identity="a" * 64,
        expected_simple_return=inputs.expected_simple_return,
        return_variance=inputs.return_variance,
        buy_cost=inputs.buy_cost,
        sell_cost=inputs.sell_cost,
        exit_cost=inputs.exit_cost,
        funding_return=inputs.funding_return,
        borrow_return=inputs.borrow_return,
        cash_return=inputs.cash_return,
    )
    context = AllocationContext(
        account_id="synthetic-account",
        symbol="BTC",
        decision_time=decision_time,
        state_digest="b" * 64,
        current_weight=current,
        quantity="0",
        cash=10_000.0,
        equity=10_000.0,
        pending_remaining="0",
    )
    allocator = AfterCostTargetAllocator(
        lower_weight=lower,
        upper_weight=upper,
        max_turnover=None,
        risk_aversion=0.0,
    )
    baseline = allocator.propose(h_inputs, context)
    mode, action_code = ("residual", 2) if role == "NONRL" else ("direct", 0)
    decision = AllocationDecision(
        baseline=baseline,
        action_contract=AllocationActionContract(mode=mode, scale=1.0),
        feature_names=("synthetic-feature",),
        feature_values=(0.0,),
        max_drawdown=drawdown,
        initial_capital=10_000.0,
        remaining_steps=1,
        pending_gross=0.0,
        pending_count=0,
    )
    action = decision.propose(action_code)
    targets = np.zeros(2, dtype=np.float64)
    currents = np.zeros(2, dtype=np.float64)
    emergency_mask = np.zeros(2, dtype=np.bool_)
    reduce_mask = np.zeros(2, dtype=np.bool_)
    targets[symbol_index] = action.target_weight
    currents[symbol_index] = current
    emergency_mask[symbol_index] = emergency
    reduce_mask[symbol_index] = reduce_only
    risk = PreTradeRisk(PreTradeRiskConfig(**_FROZEN_RISK_POLICY)).constrain(
        targets,
        current=currents,
        drawdown=drawdown,
        emergency_flatten_mask=emergency_mask,
        reduce_only_mask=reduce_mask,
    )
    return baseline, action, risk


def _assert_composition_matches_native(
    inputs,
    *,
    current: float,
    lower: float = -1.0,
    upper: float = 1.0,
    drawdown: float = 0.0,
    role: str = "NONRL",
    emergency: bool = False,
    reduce_only: bool = False,
) -> object:
    result = _FROZEN_DECISION.compose_frozen_allocation_target(
        inputs,
        current_weight=current,
        lower_weight=lower,
        upper_weight=upper,
        drawdown=drawdown,
        role=role,
        emergency_flatten=emergency,
        reduce_only=reduce_only,
    )
    for symbol_index in (0, 1):
        baseline, action, risk = _native_composition(
            inputs,
            current=current,
            lower=lower,
            upper=upper,
            drawdown=drawdown,
            role=role,
            emergency=emergency,
            reduce_only=reduce_only,
            symbol_index=symbol_index,
        )
        assert result.baseline_weight == baseline.target_weight
        assert result.baseline_objective == baseline.objective_value
        assert result.action_mode == action.decision.action_contract.mode
        assert result.action_code == action.raw_action
        assert result.action_label == action.action
        assert result.raw_action_target_weight == action.raw_target_weight
        assert result.action_target_weight == action.target_weight
        assert result.is_hold is action.is_hold
        assert result.risk.weights == risk.weights[symbol_index]
        assert result.risk.requested_turnover == risk.requested_turnover
        assert result.risk.constrained_turnover == risk.constrained_turnover
        assert result.risk.was_constrained is risk.was_constrained
        assert result.risk.reasons == risk.reasons
        assert result.risk.risk_scale == risk.risk_scale
        assert result.risk.projection_l1 == risk.projection_l1
        assert result.risk.turnover_overridden is risk.turnover_overridden
    return result


def test_frozen_nonrl_composes_after_cost_baseline_then_hard_risk() -> None:
    result = _assert_composition_matches_native(
        _new_inputs(), current=0.1, drawdown=0.15, role="NONRL"
    )
    assert result.action_mode == "residual"
    assert result.action_code == 2
    assert result.action_label == "baseline"
    assert result.raw_action_target_weight == result.baseline_weight
    assert result.action_target_weight == result.baseline_weight
    expected_scale = 1.0 - (0.15 - 0.10) / (0.20 - 0.10)
    assert result.risk.risk_scale == expected_scale
    assert result.final_target_weight == result.action_target_weight * expected_scale
    assert "drawdown_deleveraging" in result.risk.reasons


@pytest.mark.parametrize(
    ("expected_return", "buy_cost", "expected_weight"),
    [
        (0.049, 0.05, 0.0),
        (0.05, 0.05, 0.0),
        (math.nextafter(0.05, math.inf), 0.05, 1.0),
    ],
)
def test_after_cost_positive_negative_and_exact_fee_edge(
    expected_return: float, buy_cost: float, expected_weight: float
) -> None:
    result = _assert_composition_matches_native(
        _new_inputs(
            expected_simple_return=expected_return,
            buy_cost=buy_cost,
            sell_cost=0.0,
            exit_cost=0.0,
            return_variance=0.0,
        ),
        current=0.0,
        role="NONRL",
    )
    assert result.baseline_weight == expected_weight
    assert result.action_target_weight == expected_weight


def test_cash_hold_bypasses_allocator_bounds_but_still_passes_final_risk() -> None:
    result = _assert_composition_matches_native(
        _new_inputs(expected_simple_return=0.4),
        current=1.2,
        lower=-0.3,
        upper=0.3,
        role="CASH",
    )
    assert result.baseline_weight == 0.3
    assert result.action_mode == "direct"
    assert result.action_code == 0
    assert result.action_label == "quantity_hold"
    assert result.raw_action_target_weight == 1.2
    assert result.action_target_weight == 1.2
    assert result.is_hold is True
    assert result.final_target_weight == 1.0
    assert result.risk.reasons == ("max_abs_weight",)


@pytest.mark.parametrize(
    ("drawdown", "expected_scale"),
    [
        (0.10, 1.0),
        (
            math.nextafter(0.10, math.inf),
            1.0 - (math.nextafter(0.10, math.inf) - 0.10) / 0.10,
        ),
        (math.nextafter(0.20, 0.0), 1.0 - (math.nextafter(0.20, 0.0) - 0.10) / 0.10),
        (0.20, 0.0),
    ],
)
def test_drawdown_ulp_boundaries_match_both_native_symbol_slots(
    drawdown: float, expected_scale: float
) -> None:
    result = _assert_composition_matches_native(
        _new_inputs(), current=0.0, drawdown=drawdown, role="NONRL"
    )
    assert result.risk.risk_scale == expected_scale
    assert result.final_target_weight == result.action_target_weight * expected_scale


def test_emergency_flatten_and_reduce_only_are_final_risk_overrides() -> None:
    emergency = _assert_composition_matches_native(
        _new_inputs(), current=0.4, role="NONRL", emergency=True
    )
    assert emergency.action_target_weight == emergency.baseline_weight
    assert emergency.final_target_weight == 0.0
    assert "emergency_flatten" in emergency.risk.reasons
    assert "emergency_turnover_override" in emergency.risk.reasons

    reduced = _assert_composition_matches_native(
        _new_inputs(expected_simple_return=0.5),
        current=0.4,
        role="NONRL",
        reduce_only=True,
    )
    assert reduced.final_target_weight == 0.4
    assert "reduce_only" in reduced.risk.reasons
    assert "reduce_only_satisfied" in reduced.risk.reasons


def test_reduce_only_sign_reversal_fails_closed() -> None:
    inputs = _new_inputs(expected_simple_return=-0.5)
    with pytest.raises(ValueError, match="reduce-only target cannot change sign"):
        _FROZEN_DECISION.compose_frozen_allocation_target(
            inputs,
            current_weight=0.4,
            lower_weight=-1.0,
            upper_weight=1.0,
            drawdown=0.0,
            role="NONRL",
            reduce_only=True,
        )
    for symbol_index in (0, 1):
        with pytest.raises(ValueError, match="reduce-only target cannot change sign"):
            _native_composition(
                inputs,
                current=0.4,
                lower=-1.0,
                upper=1.0,
                drawdown=0.0,
                role="NONRL",
                reduce_only=True,
                symbol_index=symbol_index,
            )


def test_tied_after_cost_baseline_prefers_current_hold() -> None:
    result = _assert_composition_matches_native(
        _new_inputs(
            expected_simple_return=0.0,
            return_variance=0.0,
            buy_cost=0.0,
            sell_cost=0.0,
            exit_cost=0.0,
            funding_return=0.0,
            borrow_return=0.0,
            cash_return=0.0,
        ),
        current=0.25,
        role="NONRL",
    )
    assert result.baseline_weight == 0.25
    assert result.action_label == "baseline"
    assert result.is_hold is True


def test_cash_negative_zero_survives_hold_action_and_noop_risk() -> None:
    result = _assert_composition_matches_native(
        _new_inputs(
            expected_simple_return=0.0,
            return_variance=0.0,
            buy_cost=0.0,
            sell_cost=0.0,
            exit_cost=0.0,
            funding_return=0.0,
            borrow_return=0.0,
            cash_return=0.0,
        ),
        current=-0.0,
        role="CASH",
    )
    assert math.copysign(1.0, result.raw_action_target_weight) == -1.0
    assert math.copysign(1.0, result.action_target_weight) == -1.0
    assert math.copysign(1.0, result.final_target_weight) == -1.0


def test_frozen_composition_rejects_unbound_roles_and_bad_numeric_inputs() -> None:
    inputs = _new_inputs()
    for role in ("ppo", "cash ", "nonrl", ""):
        with pytest.raises(ValueError):
            _FROZEN_DECISION.compose_frozen_allocation_target(
                inputs,
                current_weight=0.0,
                lower_weight=-1.0,
                upper_weight=1.0,
                drawdown=0.0,
                role=role,
            )
    for current, lower, upper, drawdown in (
        (True, -1.0, 1.0, 0.0),
        (0.0, 1.0, -1.0, 0.0),
        (0.0, -1.0, 1.0, math.nan),
    ):
        with pytest.raises(ValueError):
            _FROZEN_DECISION.compose_frozen_allocation_target(
                inputs,
                current_weight=current,
                lower_weight=lower,
                upper_weight=upper,
                drawdown=drawdown,
                role="NONRL",
            )


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (math.nan, 1.0),
        (-1.0, math.inf),
        (-math.inf, 1.0),
        (-1.0, np.float64(1.0)),
        (np.int64(-1), 1.0),
        (False, 1.0),
        (1.0, -1.0),
    ],
)
def test_frozen_bounds_reject_nonfinite_nonordinary_or_reversed_values(
    lower: object, upper: object
) -> None:
    with pytest.raises(ValueError):
        _FROZEN_DECISION.compose_frozen_allocation_target(
            _new_inputs(),
            current_weight=0.0,
            lower_weight=lower,
            upper_weight=upper,
            drawdown=0.0,
            role="NONRL",
        )


def test_negative_weight_at_drawdown_stop_retains_native_negative_zero() -> None:
    inputs = _new_inputs(
        expected_simple_return=-0.5,
        return_variance=0.0,
        buy_cost=0.0,
        sell_cost=0.0,
        exit_cost=0.0,
        funding_return=0.0,
        borrow_return=0.0,
        cash_return=0.0,
    )
    result = _assert_composition_matches_native(
        inputs, current=0.4, drawdown=0.20, role="NONRL"
    )
    assert result.baseline_weight == -1.0
    assert result.risk.risk_scale == 0.0
    assert result.final_target_weight == 0.0
    assert math.copysign(1.0, result.final_target_weight) == -1.0
    for symbol_index in (0, 1):
        _, _, native = _native_composition(
            inputs,
            current=0.4,
            lower=-1.0,
            upper=1.0,
            drawdown=0.20,
            role="NONRL",
            symbol_index=symbol_index,
        )
        assert math.copysign(1.0, native.weights[symbol_index]) == -1.0
