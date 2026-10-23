"""Result-blind native payoff-clock controls; no actor or optional backend."""

import hashlib
import json
from dataclasses import fields, replace
from fractions import Fraction

import numpy as np
import pytest

from tests.evaluation import allocation_scheduled_delayed_fixture as fixture
from tests.simulation.test_stateful_execution_characterization import _normalize
from trade_rl.evaluation.rl_allocation.preprocessing import (
    validate_training_preprocessing,
)


def assert_same(left, right):
    assert _normalize(left) == _normalize(right)


@pytest.mark.parametrize("cue", [1, -1])
@pytest.mark.parametrize("payoff", [True, False])
def test_legacy_identity_arrays_recipe_and_native_trajectory(cue, payoff):
    old = fixture.delayed_raw_args(1, cue, payoff=payoff)
    explicit = fixture.delayed_raw_args(1, cue, payoff=payoff, payoff_index=22)
    golden = hashlib.sha256(
        json.dumps(
            {"scheduled_delayed_v1": 1, "cue": cue, "payoff": payoff},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert old["dataset"].dataset_id == explicit["dataset"].dataset_id == golden
    assert_same(old, explicit)
    marks = np.full((24, 1), 128.0)
    if payoff:
        marks[22:] = 128 * (1 + 0.08 * cue)
    np.testing.assert_array_equal(old["dataset"].mark_price, marks)
    left = fixture.delayed_env(1, cue, payoff=payoff)
    right = fixture.delayed_env(1, cue, payoff=payoff, payoff_index=22)
    assert_same(left.recipe, right.recipe)
    assert left.reset()[0].tobytes() == right.reset()[0].tobytes()
    for action in [3 if cue == 1 else 1] + [3, 2, 1, 0] * 3 + [0] * 3:
        a, b = left.step(action), right.step(action)
        assert a[0].tobytes() == b[0].tobytes() and a[1:4] == b[1:4]
        assert_same(a[4], b[4])
        assert_same(left.book, right.book)
        assert_same(left.order_book, right.order_book)
    assert_same(
        fixture.delayed_preprocessing(), fixture.delayed_preprocessing(payoff_index=22)
    )


@pytest.mark.parametrize("helper", ["raw", "preprocessing", "env", "schedule"])
@pytest.mark.parametrize("clock", [True, False, 8.0, 22.0, 7, "8", None, np.int64(8)])
def test_closed_clock_rejects_before_construction(helper, clock, monkeypatch):
    monkeypatch.setattr(fixture, "parameters", lambda: pytest.fail("constructed"))
    call = {
        "raw": lambda: fixture.delayed_raw_args(1, 1, payoff_index=clock),
        "preprocessing": lambda: fixture.delayed_preprocessing(payoff_index=clock),
        "env": lambda: fixture.delayed_env(1, 1, payoff_index=clock),
        "schedule": lambda: fixture.delayed_schedule(4, payoff_index=clock),
    }[helper]
    with pytest.raises(ValueError, match="payoff_index"):
        call()


@pytest.mark.parametrize("helper", [fixture.delayed_raw_args, fixture.delayed_env])
def test_no_payoff_cannot_select_early_clock(helper, monkeypatch):
    monkeypatch.setattr(fixture, "parameters", lambda: pytest.fail("constructed"))
    with pytest.raises(ValueError, match="payoff"):
        helper(1, 1, payoff=False, payoff_index=8)


@pytest.mark.parametrize("cue", [1, -1])
def test_only_mark_family_moves_and_actual_prefix_and_first_inputs_match(cue):
    early, late = (
        fixture.delayed_raw_args(1, cue, payoff_index=clock) for clock in (8, 22)
    )
    changed = {"mark_price", "index_price", "close", "high", "low"}
    for field in fields(early["dataset"]):
        if not field.init or field.name == "dataset_id":
            continue
        a, b = (getattr(args["dataset"], field.name) for args in (early, late))
        if field.name in changed:
            marks = []
            for clock in (8, 22):
                expected = np.full((24, 1), 128.0)
                expected[clock:] = 128 * (1 + 0.08 * cue)
                if field.name == "high":
                    expected = np.maximum(128, expected)
                elif field.name == "low":
                    expected = np.minimum(128, expected)
                marks.append(expected)
            np.testing.assert_array_equal(a, marks[0])
            np.testing.assert_array_equal(b, marks[1])
        else:
            assert_same(a, b)
    assert early["dataset"].dataset_id != late["dataset"].dataset_id
    np.testing.assert_array_equal(early["dataset"].open, np.full((24, 1), 128.0))
    assert np.flatnonzero(early["dataset"].volume[:, 0]).tolist() == [7]
    assert early["dataset"].volume[7, 0] == 1000
    for key in early.keys() - {"dataset", "stream", "bound"}:
        assert_same(early[key], late[key])
    assert early["bound"].clock == late["bound"].clock
    assert early["bound"].objective == late["bound"].objective
    prefixes = []
    for args in (early, late):
        vintage = args["stream"].vintages[0]
        trace = vintage.training.trace
        times = args["dataset"].timestamps
        np.testing.assert_array_equal(trace.start_times, times[:4])
        np.testing.assert_array_equal(trace.end_times, times[1:5])
        np.testing.assert_array_equal(trace.start_close, [128] * 4)
        np.testing.assert_array_equal(trace.end_close, [128] * 4)
        np.testing.assert_array_equal(vintage.training.labels, [0] * 4)
        np.testing.assert_array_equal(
            vintage.training.log_training.features[:, 0], [-1, 0, 1, cue]
        )
        assert vintage.block.fit_cutoff == times[5]
        prefixes.append((vintage.training, vintage.model))
    assert_same(*prefixes)
    a, b = (fixture.delayed_preprocessing(payoff_index=clock) for clock in (8, 22))
    assert a.normalizer.mean == b.normalizer.mean == (0.0,)
    assert (
        a.normalizer.scale
        == b.normalizer.scale
        == (float(np.sqrt(float(Fraction(2, 3)))),)
    )
    assert a.admitted_row_indices == b.admitted_row_indices == ((0, 1, 2),)
    assert a.fit_consumption_digest == b.fit_consumption_digest
    assert a.normalizer.source_dataset_id != b.normalizer.source_dataset_id
    assert a.digest != b.digest
    left, right = (fixture.delayed_env(1, cue, payoff_index=clock) for clock in (8, 22))
    assert left.recipe_digest != right.recipe_digest
    assert left.reset()[0].tobytes() == right.reset()[0].tobytes()
    entry = 3 if cue == 1 else 1
    assert left.step(entry)[0].tobytes() == right.step(entry)[0].tobytes()
    assert left.index == right.index == 7
    assert left.step(0)[0].tobytes() != right.step(0)[0].tobytes()
    assert left.index == right.index == 8


def test_canonical_early_owner_is_shared_and_supplied_content_is_not_replaced(
    monkeypatch,
):
    frozen = fixture.delayed_preprocessing(payoff_index=8)
    owner = fixture.delayed_raw_args(1, 1, payoff_index=8)["dataset"]
    validate_training_preprocessing(
        owner, frozen, feature_indices=(0,), symbol_index=0, first_decision_index=6
    )
    runtime = fixture.delayed_schedule(4, frozen=frozen, payoff_index=8)
    assert len(runtime.schedule.training_windows) == 6
    assert len(runtime.schedule.windows) == 10
    assert frozen.normalizer.source_dataset_id == owner.dataset_id
    assert owner.dataset_id in {w.dataset_id for w in runtime.schedule.training_windows}
    assert all(
        child.feature_preprocessing is frozen for child in runtime.training_environments
    )
    roster = fixture.TRAIN_ROSTER + fixture.HELDOUT_ROSTER
    identities = [
        fixture.delayed_raw_args(day, cue, payoff_index=8)["dataset"].dataset_id
        for day, cue in roster
    ]
    assert [w.dataset_id for w in runtime.schedule.windows] == identities
    children = runtime.training_environments + tuple(
        fixture.delayed_env(day, cue, frozen=frozen, payoff_index=8)
        for day, cue in fixture.HELDOUT_ROSTER
    )
    for child, (_, cue), identity in zip(children, roster, identities, strict=True):
        assert child.feature_preprocessing is frozen
        assert child.dataset.dataset_id == identity
        np.testing.assert_array_equal(
            child.dataset.mark_price[8:, 0], [128 * (1 + 0.08 * cue)] * 16
        )
    assert runtime.schedule.digest != fixture.delayed_schedule(4).schedule.digest
    assert_same(
        fixture.delayed_schedule(4).schedule,
        fixture.delayed_schedule(4, payoff_index=22).schedule,
    )
    late = fixture.delayed_preprocessing()
    inference = replace(
        late, normalizer=replace(late.normalizer, source_dataset_id="f" * 64)
    )
    assert (
        fixture.delayed_env(20, -1, frozen=inference).feature_preprocessing is inference
    )
    invalid = (
        late,
        replace(frozen, normalizer=replace(frozen.normalizer, mean=(1.0,))),
        replace(
            frozen,
            admitted_row_indices=((0, 1),),
            normalizer=replace(frozen.normalizer, usable_counts=((2,),)),
        ),
    )
    monkeypatch.setattr(
        fixture, "bind_frozen", lambda *a: pytest.fail("child constructed")
    )
    for declaration in invalid:
        with pytest.raises(ValueError, match="preprocessing|prefix"):
            fixture.delayed_env(20, -1, frozen=declaration, payoff_index=8)
        with pytest.raises(ValueError, match="preprocessing|prefix"):
            fixture.delayed_schedule(4, frozen=declaration, payoff_index=8)


@pytest.mark.parametrize("clock", [8, 22])
@pytest.mark.parametrize("cue", [1, -1])
@pytest.mark.parametrize("side", [1, -1, 0])
def test_fraction_native_amounts_rewards_and_drawdown(clock, cue, side):
    q = 4 * cue * side
    fee = Fraction(abs(q) * 128, 512)
    cash = Fraction(1024 - q * 128) - fee
    mark = 128 * (1 + Fraction(2 * cue, 25))
    final = cash + q * mark
    gain = (final - 1024) / 1024
    assert (
        gain
        == {1: Fraction(999, 25600), -1: Fraction(-1049, 25600), 0: Fraction(0)}[side]
    )
    env = fixture.delayed_env(1, cue, payoff_index=clock)
    env.reset()
    before, rewards = Fraction(1024), []
    peak, maximum = Fraction(1024), Fraction(0)
    expected_cash, expected_q = Fraction(1024), 0
    for processing in range(7, 23):
        action = (3 if q > 0 else 1 if q < 0 else 0) if processing == 7 else 0
        observation, reward, done, truncated, info = env.step(action)
        # Native order: value existing holdings at open, pay entry fee, then mark.
        opened = expected_cash + expected_q * 128
        if processing == 7:
            expected_cash -= q * 128 + fee
            expected_q = q
        after_entry = expected_cash + expected_q * 128
        value = expected_cash + expected_q * (mark if processing >= clock else 128)
        for checkpoint in (opened, after_entry, value):
            peak = max(peak, checkpoint)
            maximum = max(maximum, 1 - checkpoint / peak)
        current = 1 - value / peak
        if clock == 8 and side == 1 and processing in (8, 9):
            assert (
                maximum == {8: Fraction(1, 1024), 9: Fraction(1024, 26599)}[processing]
            )
        assert env.book.quantities.tolist() == [expected_q]
        assert env.book.cash == float(expected_cash)
        assert reward == pytest.approx(float((value - before) / 1024), rel=0, abs=1e-14)
        assert env.book.portfolio_value == pytest.approx(float(value), rel=0, abs=1e-10)
        assert env.book.peak_value == pytest.approx(float(peak), rel=0, abs=1e-10)
        assert env.book.max_drawdown == pytest.approx(float(maximum), rel=0, abs=1e-14)
        assert 1 - env.book.portfolio_value / env.book.peak_value == pytest.approx(
            float(current), rel=0, abs=1e-14
        )
        assert info["execution"].interval_cost == (float(fee) if processing == 7 else 0)
        assert info["execution"].filled_notional == (
            abs(q) * 128 if processing == 7 else 0
        )
        assert info["execution"].fill_count == (int(q != 0) if processing == 7 else 0)
        assert done is (processing == 22) and not truncated
        before = value
        rewards.append(reward)
    assert rewards[0] == float(-fee / 1024)
    assert rewards[clock - 7] == pytest.approx(
        float(Fraction(side, 25)), rel=0, abs=1e-14
    )
    assert sum(rewards) == pytest.approx(float(gain), rel=0, abs=1e-14)
    assert env.book.quantities.tolist() == [q]
    assert env.book.cash == float(cash) == {4: 511, -4: 1535, 0: 1024}[q]
    assert env.book.mark_prices.tolist() == [float(mark)]
    assert env.book.total_cost == fee and env.book.fill_count == int(q != 0)
    correct_dd = Fraction(1024, 26599) if clock == 8 else Fraction(1, 1024)
    assert maximum == {1: correct_dd, -1: Fraction(1049, 25600), 0: Fraction(0)}[side]
    assert env.book.max_drawdown < env.risk_config.drawdown_start == 0.10
    assert env.book.gross_exposure < env.risk_config.max_gross == 1
    assert env.book.termination_reason is None and env.index == 22
    assert not observation.any()


@pytest.mark.parametrize("clock", [8, 22])
@pytest.mark.parametrize("cue", [1, -1])
def test_every_late_first_entry_and_adversarial_pending_requests_have_zero_fills(
    clock, cue
):
    entry = 3 if cue == 1 else 1
    for first in range(7, 22):
        env = fixture.delayed_env(1, cue, payoff_index=clock)
        env.reset()
        for decision in range(6, 22):
            info = env.step(entry if decision >= first else 0)[4]
            assert info["execution"].fill_count == 0
            assert (
                info["execution"].filled_notional
                == info["execution"].interval_cost
                == 0
            )
        assert env.book.quantities.tolist() == [0]
        assert env.book.cash == env.book.portfolio_value == 1024
        assert env.order_book.active_orders
    env = fixture.delayed_env(1, cue, payoff_index=clock)
    env.reset()
    env.step(entry)
    pending = []
    for action in [entry, 2, 1 if entry == 3 else 3, 0] * 3 + [entry] * 3:
        info = env.step(action)[4]
        assert info["execution"].fill_count == 0
        assert info["execution"].filled_notional == info["execution"].interval_cost == 0
        assert env.book.quantities.tolist() == [cue * 4]
        assert env.book.cash == (511 if cue == 1 else 1535)
        assert env.book.total_cost == env.book.fill_count == 1
        assert info["risk_target"] is not None
        pending.append(bool(env.order_book.active_orders))
    assert any(pending) and not all(pending)


@pytest.mark.parametrize("clock", [8, 22])
def test_h16_schedule_rotates_only_after_true_terminal_without_liquidation(clock):
    runtime = fixture.delayed_schedule(4, payoff_index=clock)
    runtime.reset()
    child, identity = runtime.active_env, runtime.active_window_id
    for action in [3, 1, 1, 1]:
        assert runtime.step(action)[2:4] == (False, False)
    assert runtime.active_env is child and child.index == 10
    assert runtime.active_window_id == identity
    with pytest.raises(RuntimeError, match="true terminal"):
        runtime.reset()
    for _ in range(11):
        assert runtime.step(1)[2:4] == (False, False)
    orders = _normalize(child.order_book)
    observation, _, done, truncated, info = runtime.step(1)
    assert done and not truncated and not observation.any()
    assert child.index == child.book.as_of_index == 22
    assert child.book.quantities.tolist() == [4]
    assert (
        child.book.cash == 511 and child.book.total_cost == child.book.fill_count == 1
    )
    assert info["execution"].interval_cost == info["execution"].filled_notional == 0
    assert child.order_book.active_orders
    assert_same(orders, child.order_book)
    with pytest.raises(RuntimeError, match="reset"):
        runtime.step(0)
    with pytest.raises(RuntimeError, match="terminal"):
        child.step(0)
    runtime.reset()
    assert runtime.active_env is not child
    assert runtime.active_window_id == runtime.schedule.train_window_ids[1]
    assert child.index == 22
