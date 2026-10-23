"""Native zero-target reserve oracles; no fitting or market-data access."""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from importlib import import_module, util

import numpy as np
import pytest

from tests.evaluation.test_allocation_global_execution import declare
from tests.evaluation.test_allocation_nonrl_walk_forward import fixture
from tests.evaluation.test_allocation_terminal_execution import carrier
from trade_rl.artifacts import content_digest
from trade_rl.evaluation.rl_allocation.transition_facts import (
    native_allocation_account_facts,
)


def capability():
    name = "trade_rl.evaluation.allocation_reserved_closing"
    assert util.find_spec(name) is not None, "shared native reserve owner is missing"
    api = import_module(name)
    assert hasattr(api, "_iter_reserved_allocation_closing"), (
        "shared native reserve iterator is missing"
    )
    return api


def open_native_prefix(folds, env, *, kind="nonrl"):
    native = import_module("trade_rl.evaluation.allocation_global_execution")
    native.run_declared_global_allocation_execution(
        folds, env, declare(native, env, folds, kind=kind)
    )
    assert env.index == env.stop_index == 7
    return env


def reserve(
    api,
    env,
    *,
    stop=8,
    stages=None,
    before=None,
    binding=None,
):
    if stages is None:
        stages = []
    if before is None:

        def before(index, book, orders):
            return content_digest(
                {
                    "schema": "literal_reserve_before_state_v1",
                    "index": index,
                    "account": native_allocation_account_facts(book, orders),
                }
            )

    return api._iter_reserved_allocation_closing(
        env.executor,
        env.book,
        env.order_book,
        risk=env.risk,
        symbol_index=env.symbol_index,
        start_index=env.index,
        stop_index=stop,
        decision_digest=lambda index: content_digest(
            {"schema": "literal_reserved_decision_v1", "index": index}
        ),
        before_state=before,
        validate_binding=env.validate_binding if binding is None else binding,
        note_stage=lambda index, stage: stages.append((index, stage)),
    )


@pytest.mark.parametrize("side,final_cash", [(1, 1039.0), (-1, 1041.0)])
def test_native_reserved_long_short_charge_real_exit_and_keep_carrier(side, final_cash):
    api = capability()
    folds, env = carrier(side=side)
    open_native_prefix(folds, env)
    prefix_book, prefix_orders = env.book, env.order_book
    before = prefix_book.portfolio_value
    witness = "1" * 64
    (row,) = list(reserve(api, env, before=lambda _index, _book, _orders: witness))
    assert row.start_index == 7 and row.opening_equity == before
    assert row.before_state_digest == witness
    assert row.clock_valid and row.processing_time_ns == int(
        env.dataset.timestamps[8].astype("datetime64[ns]").astype("int64")
    )
    assert row.execution.next_index == 8 and row.execution.bars_advanced == 1
    assert row.risk_target.weights.tolist() == [0.0]
    assert row.execution.book.cash == pytest.approx(final_cash)
    assert row.execution.book.portfolio_value == pytest.approx(final_cash)
    assert row.execution.book.exact_quantities == (Fraction(0),)
    assert not row.execution.order_book.active_orders
    assert row.native["book"]["cash"] == pytest.approx(final_cash)
    assert row.native["book"]["exact_quantities"] == ["0"]
    assert (before - 1000.0) / 1000.0 + (
        row.execution.book.portfolio_value - row.opening_equity
    ) / 1000.0 == pytest.approx((final_cash - 1000.0) / 1000.0)
    assert env.book is prefix_book and env.order_book is prefix_orders
    assert env.index == env.stop_index == 7 and env._terminated


def test_native_fee_only_final_equity_uses_original_capital():
    api = capability()
    folds, (env,) = fixture(ranges=((6, 7),), signals=(1, 0, 0, 0), fee=0.002)
    for name in ("open", "high", "low", "close", "mark_price"):
        object.__setattr__(env.dataset, name, np.full((12, 1), 100.0))
    open_native_prefix(folds, env)
    assert env.book.cash == pytest.approx(499.0)
    assert env.book.portfolio_value == pytest.approx(999.0)
    (row,) = list(reserve(api, env))
    assert row.execution.book.cash == pytest.approx(998.0)
    assert row.execution.interval_cost == pytest.approx(1.0)
    assert (row.execution.book.portfolio_value - row.opening_equity) / 1000.0 == (
        pytest.approx(-0.001)
    )


def test_already_flat_cash_processes_every_native_reserved_interest_bar():
    api = capability()
    folds, env = carrier(kind="cash", cash_rate=0.876)
    open_native_prefix(folds, env, kind="cash")
    rows = list(reserve(api, env, stop=9))
    assert [row.start_index for row in rows] == [7, 8]
    assert [row.execution.next_index for row in rows] == [8, 9]
    assert rows[-1].execution.book.cash == pytest.approx(1000.0 * 1.0001**3)
    assert all(row.execution.book.exact_quantities == (Fraction(0),) for row in rows)
    assert all(row.execution.fill_count == 0 for row in rows)
    assert all(row.execution.interval_cash_interest > 0 for row in rows)


def test_profitable_partial_native_close_keeps_actual_inventory_and_pending_order():
    api = capability()
    folds, env = carrier(partial=True)
    open_native_prefix(folds, env)
    (row,) = list(reserve(api, env))
    assert row.execution.book.cash == pytest.approx(711.97)
    assert row.execution.book.portfolio_value == pytest.approx(1041.97)
    assert row.execution.book.exact_quantities == (Fraction(3),)
    assert row.execution.order_book.active_orders
    assert row.native["book"]["exact_quantities"] == ["3"]


@pytest.mark.parametrize("block", ["no_volume", "untradable", "dust"])
def test_unfilled_or_dust_close_never_erases_exact_quantity(block):
    api = capability()
    folds, env = carrier()
    name = "tradable" if block == "untradable" else "volume"
    values = env.dataset.resolved_array(name).copy()
    values[8] = (
        False
        if block == "untradable"
        else (0.0 if block == "no_volume" else 4.999999999999)
    )
    object.__setattr__(env.dataset, name, values)
    open_native_prefix(folds, env)
    (row,) = list(reserve(api, env))
    quantity = row.execution.book.exact_quantities[0]
    assert quantity > 0
    if block == "dust":
        assert quantity < Fraction(1, 10**9)
    else:
        assert quantity == 5
    assert row.native["book"]["exact_quantities"] != ["0"]


@pytest.mark.parametrize("first_volume,final_cash", [(0, 1038.45), (2, 1038.67)])
def test_deferred_close_reuses_pending_order_and_charges_held_carry(
    first_volume, final_cash
):
    api = capability()
    folds, env = carrier()
    volume = env.dataset.volume.copy()
    volume[8] = first_volume
    object.__setattr__(env.dataset, "volume", volume)
    open_native_prefix(folds, env)
    first, final = list(reserve(api, env, stop=9))
    pending = first.native["active_orders"][0]["intent"]["order_id"]
    assert any(
        event["order_id"] == pending and event["event_type"] == "filled"
        for event in final.native["order_events"]
    )
    assert not any(
        event["event_type"] == "submitted" for event in final.native["order_events"]
    )
    assert final.execution.book.cash == pytest.approx(final_cash)
    assert final.execution.book.exact_quantities == (Fraction(0),)
    assert not final.execution.order_book.active_orders


def test_actual_signed_debt_stops_reserve_without_fictional_later_bars():
    api = capability()
    folds, env = carrier(side=-1)
    for name in ("open", "high", "low", "close", "mark_price"):
        values = env.dataset.resolved_array(name).copy()
        values[8:] = 400.0
        object.__setattr__(env.dataset, name, values)
    open_native_prefix(folds, env)
    stages = []
    (row,) = list(reserve(api, env, stop=9, stages=stages))
    assert row.execution.book.portfolio_value == pytest.approx(-504.5)
    assert row.execution.book.termination_reason is not None
    assert row.native["book"]["equity"] == pytest.approx(-504.5)
    assert (row.execution.book.portfolio_value - 1000.0) / 1000.0 == (
        pytest.approx(-1.5045)
    )
    assert row.execution.next_index == 8
    assert stages[-1] == (7, "closing_validation")
    assert all(index == 7 for index, _ in stages)


def test_returned_native_objects_and_execution_rng_continue_across_reserve(
    monkeypatch,
):
    api = capability()
    folds, (env,) = fixture(
        ranges=((6, 7),),
        signals=(1, 0, 0, 0),
        cost_changes={"slippage_std": 0.001, "random_seed": 42},
    )
    volume = env.dataset.volume.copy()
    volume[8] = 2.0
    object.__setattr__(env.dataset, "volume", volume)
    open_native_prefix(folds, env)
    rng = env.executor._rng
    rng_before = deepcopy(rng.bit_generator.state)
    calls = []
    native = api._execute_allocation_target

    def execute(executor, book, orders, **kwargs):
        assert executor is env.executor and executor._rng is rng
        assert kwargs["pretrade_risk"] is env.risk
        assert book is (env.book if not calls else calls[-1].book)
        assert orders is (env.order_book if not calls else calls[-1].order_book)
        assert kwargs["target_weight"] == 0.0
        target, execution = native(executor, book, orders, **kwargs)
        calls.append(execution)
        return target, execution

    monkeypatch.setattr(api, "_execute_allocation_target", execute)
    monkeypatch.setattr(
        env.executor,
        "reset_random_state",
        lambda *_: pytest.fail("reserve reset execution RNG"),
    )
    iterator = reserve(api, env, stop=9)
    first = next(iterator)
    retained = deepcopy(first.native)
    final = next(iterator)
    with pytest.raises(StopIteration):
        next(iterator)
    assert len(calls) == 2 and final.execution is calls[-1]
    assert env.executor._rng is rng and rng.bit_generator.state != rng_before
    assert first.native == retained
    assert first.native["book"]["exact_quantities"] != ["0"]
    assert final.native["book"]["exact_quantities"] == ["0"]


@pytest.mark.parametrize(
    "change",
    [
        {"next_index": 99},
        {"next_index": 8.0},
        {"next_index": np.int64(8)},
        {"bars_advanced": True},
    ],
)
def test_returned_row_is_yielded_before_strict_final_clock_validation(
    change, monkeypatch
):
    api = capability()
    folds, env = carrier()
    open_native_prefix(folds, env)
    native = api._execute_allocation_target

    def execute(*args, **kwargs):
        target, execution = native(*args, **kwargs)
        return target, replace(execution, **change)

    monkeypatch.setattr(api, "_execute_allocation_target", execute)
    stages = []
    iterator = reserve(api, env, stages=stages)
    row = next(iterator)
    assert stages[-1] == (7, "closing_returned")
    assert not row.clock_valid
    assert row.native["book"]["cash"] == pytest.approx(1039.0)
    if "next_index" in change:
        assert row.processing_time_ns is None
    with pytest.raises(ValueError, match="one-bar clock"):
        next(iterator)
    assert stages[-1] == (7, "closing_validation")


def test_last_yield_does_not_skip_postflight_binding_or_exhaustion():
    api = capability()
    folds, env = carrier()
    open_native_prefix(folds, env)
    calls, stages = [], []
    failure = LookupError("reserve source changed after returned close")

    def validate():
        calls.append(True)
        if len(calls) == 2:
            raise failure
        env.validate_binding()

    iterator = reserve(api, env, stages=stages, binding=validate)
    row = next(iterator)
    assert row.clock_valid and row.native["book"]["cash"] == pytest.approx(1039.0)
    assert len(calls) == 1 and stages[-1] == (7, "closing_returned")
    with pytest.raises(LookupError) as caught:
        next(iterator)
    assert caught.value is failure and len(calls) == 2
    assert stages[-1] == (7, "closing_validation")


def test_before_state_failure_precedes_execution_and_keeps_original_exception(
    monkeypatch,
):
    api = capability()
    folds, env = carrier()
    open_native_prefix(folds, env)
    failure, stages = LookupError("original before-state failure"), []

    def before(_index, _book, _orders):
        raise failure

    monkeypatch.setattr(
        api,
        "_execute_allocation_target",
        lambda *_a, **_k: pytest.fail("execution before failed before-state witness"),
    )
    with pytest.raises(LookupError) as caught:
        next(reserve(api, env, stages=stages, before=before))
    assert caught.value is failure
    assert stages[-1] == (7, "closing_preflight")


def test_reserved_closing_reads_no_forecasts_features_or_gym_policy(monkeypatch):
    api = capability()
    folds, env = carrier()
    open_native_prefix(folds, env)

    class ForbiddenFeatures(np.ndarray):
        def __getitem__(self, _key):
            pytest.fail("reserve numerically read policy features")

    object.__setattr__(
        env.dataset, "features", env.dataset.features.view(ForbiddenFeatures)
    )
    object.__setattr__(
        env.dataset,
        "global_features",
        env.dataset.global_features.view(ForbiddenFeatures),
    )
    for name in ("_prepare", "_observation", "step", "reset"):
        monkeypatch.setattr(
            env,
            name,
            lambda *_a, **_k: pytest.fail("reserve called forecast/policy/reset hook"),
        )
    rows = list(reserve(api, env, stop=9))
    assert [row.start_index for row in rows] == [7, 8]
    assert rows[-1].native["book"]["cash"] == pytest.approx(1039.0)
    assert rows[-1].execution.next_index == 9
