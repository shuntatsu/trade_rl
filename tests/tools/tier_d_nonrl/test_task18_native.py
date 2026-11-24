"""Native Task18 software conformance using the existing artificial v7 fixture."""

from dataclasses import replace
from fractions import Fraction
from unittest.mock import patch

import numpy as np
import pytest

from tests.tools.tier_d_nonrl.test_task18_frozen_inputs import (
    _binding,
    _identity,
    _market,
)
from tools.tier_d_nonrl.task18_recipe import build_task18_protocol

P = build_task18_protocol()


@pytest.mark.parametrize(
    "current,forecast,cost,funding,target",
    (
        (0.0, 0.1, 0.01, 0.0, 1.0),
        (0.0, -0.1, 0.01, 0.0, -1.0),
        (0.5, 0.0, 0.01, 0.0, 0.5),
        (0.5, 0.02, 0.01, 0.0, 0.5),
        (0.0, 0.02, 0.01, 0.0, 0.0),
        (0.0, np.nextafter(0.02, np.inf), 0.01, 0.0, 1.0),
        (0.0, 0.1, 0.01, 0.2, -1.0),
        (1.1, 0.1, 0.01, 0.0, 1.0),
    ),
)
def test_scalar_allocator_preserves_exact_tie_breaks(
    current, forecast, cost, funding, target
):
    from tools.tier_d_nonrl.task18_native import choose_target_weight

    assert (
        choose_target_weight(
            current_weight=current,
            expected_simple_return=forecast,
            one_way_cost=cost,
            funding_return=funding,
        )
        == target
    )


@pytest.fixture(scope="module")
def native_attempt(tmp_path_factory):
    from tools.tier_d_nonrl.task18_frozen_inputs import (
        prepare_frozen_inputs,
        publish_frozen_inputs,
    )
    from tools.tier_d_nonrl.task18_native import open_synthetic_native_attempt

    # Explicit venue grid, not a dust waiver. The original no-grid
    # fixture is exercised independently as a rejected terminal state.
    original = _market()
    market = _identity(
        replace(
            original,
            dataset_id="0" * 64,
            identity_payload_json=None,
            lot_size=np.full_like(original.close, 0.001),
        )
    )
    binding = _binding(market)
    plan = prepare_frozen_inputs(market, binding=binding)
    path = tmp_path_factory.mktemp("native-task18").resolve() / "plan"
    pin = publish_frozen_inputs(path, plan)
    return open_synthetic_native_attempt(
        market, path / "frozen-plan.bin", expected_binding=binding, expected_pin=pin
    )


def test_full_native_cash_arm_and_nonrl_cost_oracle(native_attempt):
    from tools.tier_d_nonrl.task18_native_oracle import verify_native_arm

    ids = (P.arms[0].arm_id, P.arms[1].arm_id)
    with (
        patch(
            "tools.tier_d_nonrl.task18_frozen_inputs.fit_simple_return_ridge",
            side_effect=AssertionError("must not fit"),
        ),
        patch(
            "tools.tier_d_nonrl.task18_frozen_inputs.build_cost_rates",
            side_effect=AssertionError("must not recost"),
        ),
        patch(
            "trade_rl.strategies.forecasts.simple_return.SimpleReturnRidgeModel.predict_simple_return",
            side_effect=AssertionError("must not predict"),
        ),
    ):
        nonrl, cash = tuple(native_attempt.iter_arms(ids))
    assert nonrl.arm_id == ids[0] and cash.arm_id == ids[1]
    for observed in (nonrl, cash):
        assert observed.verification.passed
        assert observed.verification.advances == 6599
        assert observed.verification.actor_decisions == 6527
        assert observed.verification.terminal_advances == 72
        assert observed.verification.terminal_flat
        assert observed.verification.pending_orders == 0
        assert observed.verification.cost_oracle_passed
        assert not observed.execution_authorized
        ledger = observed.result.ledger_evidence.intervals
        assert ledger.start_index == 2903 and ledger.stop_index == 9502
        assert all(Fraction(i.exact_quantities_after[1]) == 0 for i in ledger.intervals)
    assert cash.result.book.portfolio_value == 100000.0
    assert cash.result.book.total_cost == cash.result.book.funding_pnl == 0.0
    assert nonrl.result.book.total_cost > 0 and nonrl.result.book.n_trades > 0
    # Changing a fill's participation evidence must not pass the source cost oracle.
    evidence = nonrl.result.ledger_evidence
    ledger = evidence.intervals
    i = next(
        i
        for i, v in enumerate(ledger.intervals)
        if any(e.filled_quantity for e in v.order_events)
    )
    interval = ledger.intervals[i]
    j = next(j for j, e in enumerate(interval.order_events) if e.filled_quantity)
    events = list(interval.order_events)
    events[j] = replace(events[j], participation_rate=0.0)
    intervals = list(ledger.intervals)
    intervals[i] = replace(interval, order_events=tuple(events))
    tampered = replace(
        nonrl.result,
        ledger_evidence=replace(
            evidence, intervals=replace(ledger, intervals=tuple(intervals))
        ),
    )
    assert not verify_native_arm(
        native_attempt.dataset, nonrl.arm_id, tampered, frozen=native_attempt.frozen
    ).passed

    # The actual earlier no-lot run left a sub-tolerance nonzero exact
    # quantity. A portable falsifier preserves the refusal contract without
    # depending on a particular BLAS/rounding trajectory.
    residual = replace(
        nonrl.result,
        ledger_evidence=replace(
            evidence,
            intervals=replace(
                ledger, terminal_exact_quantities=("1/10000000000000", "0")
            ),
        ),
    )
    report = verify_native_arm(
        native_attempt.dataset, nonrl.arm_id, residual, frozen=native_attempt.frozen
    )
    assert not report.passed and not report.terminal_flat
    assert report.reason == "incomplete_native_coverage"

    decisions = list(evidence.decisions)
    first = decisions[0]
    decisions[0] = replace(first, proposal_weights=(0.123456, 0.0))
    forged = replace(
        nonrl.result,
        decisions=tuple(decisions),
        ledger_evidence=replace(evidence, decisions=tuple(decisions)),
    )
    report = verify_native_arm(
        native_attempt.dataset, nonrl.arm_id, forged, frozen=native_attempt.frozen
    )
    assert not report.passed and report.reason == "frozen_decision_mismatch"


@pytest.mark.parametrize(
    "ids",
    (
        (),
        ("foreign",),
        ("PAIR-01-CASH", "PAIR-01-CASH"),
        ("PAIR-02-NONRL", "PAIR-01-NONRL"),
    ),
)
def test_attempt_refuses_wrong_roster_before_any_dispatch(native_attempt, ids):
    with patch(
        "tools.tier_d_nonrl.task18_native.run_shared_portfolio_replay",
        side_effect=AssertionError("must not dispatch"),
    ):
        with pytest.raises(ValueError):
            tuple(native_attempt.iter_arms(ids))


def test_native_stress_and_latency_arms(native_attempt):
    # Stress BTC and latency-two ETH exercise both symbols without
    # sharing books across any pair or configuration.
    ids = (P.arms[2].arm_id, P.arms[10].arm_id)
    for observed in native_attempt.iter_arms(ids):
        assert observed.verification.passed
        assert observed.verification.advances == 6599
        ledger = observed.result.ledger_evidence.intervals
        events = tuple(
            e for interval in ledger.intervals for e in interval.order_events
        )
        if observed.arm_id == ids[0]:
            assert observed.result.book.n_trades > 0
            assert observed.result.book.total_cost > 0
        else:
            # A changing target can supersede a delayed order before it fills.
            # Do not force fills or change the frozen objective to rescue it.
            assert any(e.event_type == "latency_wait" for e in events)
            assert any(e.event_type == "cancelled" for e in events)
        assert observed.verification.pending_orders == 0
        assert observed.verification.terminal_flat


def test_direct_attempt_construction_cannot_bypass_frozen_pin(native_attempt):
    from tools.tier_d_nonrl.task18_native import SyntheticNativeAttempt

    with pytest.raises(ValueError, match="frozen plan pin"):
        SyntheticNativeAttempt(
            native_attempt.dataset,
            native_attempt.frozen,
            replace(native_attempt.plan_pin, sha256="f" * 64),
        )


def test_full_48_arm_dispatch_roster_uses_fresh_policy_risk_and_fixed_execution(
    native_attempt,
):
    from tools.tier_d_nonrl import task18_native as native

    # Dispatch contract only: the four actual full-size arm tests above
    # exercise the real shared replay; this test never simulates 48 results.
    seen = []
    result = object()
    report = native.NativeArmVerification(
        True, "dispatch-test-double", 6599, 6527, 72, True, 0, True
    )

    def dispatch(dataset, policy, **kwargs):
        assert dataset is native_attempt.dataset
        arm = P.arms[len(seen)]
        assert policy._arm == arm
        assert policy._next_index == 2903
        assert all(
            policy is not previous[0] and kwargs["risk"] is not previous[1]
            for previous in seen
        )
        scenario = next(s for s in P.scenarios if s.name == arm.scenario)
        cost = kwargs["execution_cost"]
        assert cost.multiplier == scenario.fee_multiplier
        assert cost.order_latency_bars == scenario.execution_latency_bars
        assert cost.processing_bar_volume_capacity is False
        assert kwargs["initial_capital"] == 100000.0
        assert kwargs["start_index"] == 2903
        assert kwargs["stop_index"] == 9502
        assert kwargs["terminal_reserve_bars"] == 72
        assert kwargs["hold_unchanged_risk_target"] is True
        assert kwargs["insolvency_valuation"] == "retain_debt"
        seen.append((policy, kwargs["risk"]))
        return result

    with (
        patch.object(native, "run_shared_portfolio_replay", side_effect=dispatch),
        patch.object(native, "verify_native_arm", return_value=report) as verifier,
    ):
        observed = tuple(native_attempt.iter_arms(tuple(arm.arm_id for arm in P.arms)))
    assert len(seen) == verifier.call_count == 48
    assert tuple(arm.arm_id for arm in observed) == tuple(arm.arm_id for arm in P.arms)
    assert all(
        arm.result is result and not arm.execution_authorized for arm in observed
    )
