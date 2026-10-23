"""Passive native progress oracles on declared arithmetic fixtures; no fit/data IO."""

from dataclasses import replace

import pytest

from tests.evaluation.test_allocation_nonrl_walk_forward import (
    fixture,
    global_control_fixture,
    global_control_folds,
)
from tests.evaluation.test_allocation_terminal_execution import carrier, plans
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation import allocation_reserved_closing as closing
from trade_rl.evaluation import allocation_terminal_execution as terminal
from trade_rl.evaluation.allocation_terminal_evidence import (
    TerminalAllocationExecutionReceipt,
)
from trade_rl.simulation.orders.model import OrderBookState


def declared():
    folds, (env,) = fixture(
        ranges=((6, 9),),
        signals=(1, 1, 1, 1),
        cost_changes={"slippage_std": 0.001, "random_seed": 42},
    )
    return folds, env, plans(terminal, env, folds, stop=11)


def change_mark(env, index):
    marks = env.dataset.resolved_array("mark_price").copy()
    marks[index, 0] += 1
    object.__setattr__(env.dataset, "mark_price", marks)


def test_progress_counts_real_actor_and_reserve_with_unchanged_carrier_index():
    folds, env, plan = declared()
    events, carrier_indices = [], []

    def observe(phase, start, processing):
        assert type(phase) is str and type(start) is type(processing) is int
        events.append((phase, start, processing))
        carrier_indices.append(env.index)

    raw = terminal.run_terminal_allocation_execution(
        folds, env, plan, on_advance=observe
    ).payload
    TerminalAllocationExecutionReceipt.from_payload(raw, expected_plan=plan)
    assert events == [
        ("actor", 6, 7),
        ("actor", 7, 8),
        ("actor", 8, 9),
        ("reserve", 9, 10),
        ("reserve", 10, 11),
    ]
    assert carrier_indices == [7, 8, 9, 9, 9]


def test_observer_preserves_canonical_receipt_and_execution_randomness():
    results = []
    for enabled in (False, True):
        folds, env, plan = declared()
        kwargs = {"on_advance": lambda *_: None} if enabled else {}
        observed = terminal.run_terminal_allocation_execution(
            folds, env, plan, **kwargs
        )
        results.append(
            (
                canonical_json_bytes(observed.payload),
                env.executor._rng.bit_generator.state,
            )
        )
    assert results[0] == results[1]


@pytest.mark.parametrize("callback", [1, False, object()])
def test_invalid_observer_is_rejected_before_reset(callback, monkeypatch):
    folds, env, plan = declared()
    monkeypatch.setattr(env, "reset", lambda **_: pytest.fail("reset before rejection"))
    with pytest.raises(ValueError, match="on_advance"):
        terminal.run_terminal_allocation_execution(
            folds, env, plan, on_advance=callback
        )


@pytest.mark.parametrize("phase,expected_rows", [("actor", 1), ("reserve", 2)])
def test_observer_failure_keeps_returned_prefix_and_primary_exception(
    phase, expected_rows
):
    folds, env, plan = declared()
    error, events = LookupError("observer failed"), []

    def observe(actual, start, processing):
        events.append((actual, start, processing))
        if actual == phase and (phase == "actor" or processing == 11):
            raise error

    with pytest.raises(LookupError) as caught:
        terminal.run_terminal_allocation_execution(folds, env, plan, on_advance=observe)
    assert caught.value is error
    raw = error.terminal_execution_receipt
    assert raw["status"] == "integrity_failure"
    assert raw["settled_profit_rate"] is None
    rows = raw["native_receipt"]["rows"] if phase == "actor" else raw["closing_rows"]
    assert len(rows) == expected_rows
    assert events[-1][0] == phase


def test_invalid_final_returned_clock_emits_no_validated_reserve_event(monkeypatch):
    folds, env, plan = declared()
    original, events = closing._execute_allocation_target, []

    def execute(*args, **kwargs):
        risk, execution = original(*args, **kwargs)
        if kwargs["start_index"] == 10:
            execution = replace(execution, bars_advanced=2)
        return risk, execution

    monkeypatch.setattr(closing, "_execute_allocation_target", execute)
    with pytest.raises(ValueError, match="one-bar clock") as caught:
        terminal.run_terminal_allocation_execution(
            folds, env, plan, on_advance=lambda *event: events.append(event)
        )
    assert events[-1] == ("reserve", 9, 10)
    assert len(events) == 4
    assert len(caught.value.terminal_execution_receipt["closing_rows"]) == 2


def test_final_yield_source_drift_is_checked_before_progress(monkeypatch):
    folds, env, plan = declared()
    original, events = terminal._iter_reserved_allocation_closing, []

    def changed(*args, **kwargs):
        for step in original(*args, **kwargs):
            yield step
            if step.start_index == 10:
                change_mark(env, 11)

    monkeypatch.setattr(terminal, "_iter_reserved_allocation_closing", changed)
    with pytest.raises(ValueError, match="source changed"):
        terminal.run_terminal_allocation_execution(
            folds, env, plan, on_advance=lambda *event: events.append(event)
        )
    assert len(events) == 4


@pytest.mark.parametrize("phase", ["actor", "reserve"])
@pytest.mark.parametrize("change", ["book", "orders", "rng", "source", "history"])
def test_callback_cannot_change_native_state_even_on_last_reserve(
    phase, change, monkeypatch
):
    folds, env, plan = declared()
    returned, events = {}, []
    original = closing._execute_allocation_target

    def execute(*args, **kwargs):
        target, result = original(*args, **kwargs)
        returned["execution"] = result
        return target, result

    monkeypatch.setattr(closing, "_execute_allocation_target", execute)

    def observe(actual, start, processing):
        events.append((actual, start, processing))
        if actual != phase or (phase == "reserve" and processing != 11):
            return
        book = env.book if phase == "actor" else returned["execution"].book
        if change == "book":
            book.cash += 1
        elif change == "orders":
            env.order_book = OrderBookState.empty()
        elif change == "rng":
            env.executor._rng.random()
        elif change == "source":
            change_mark(env, 11 if phase == "reserve" else 7)
        else:
            book.returns_history[0] += 1

    with pytest.raises(ValueError) as caught:
        terminal.run_terminal_allocation_execution(folds, env, plan, on_advance=observe)
    assert caught.value.terminal_execution_receipt["status"] == "integrity_failure"
    assert len(events) == (1 if phase == "actor" else 5)


def test_observer_exception_after_insolvency_remains_integrity_failure():
    env = global_control_fixture(signals=(-1, -1, -1, -1), insolvent_short=True)
    folds = global_control_folds(7)
    plan = plans(terminal, env, folds, stop=11)
    error = LookupError("failed observer on signed book")

    def observe(*_):
        raise error

    with pytest.raises(LookupError) as caught:
        terminal.run_terminal_allocation_execution(folds, env, plan, on_advance=observe)
    assert caught.value is error
    raw = error.terminal_execution_receipt
    assert raw["native_receipt"]["status"] == "integrity_failure"
    assert raw["status"] == "integrity_failure"
    assert raw["native_receipt"]["rows"][0]["native"]["book"]["cash"] == -501


def test_observer_failure_on_signed_reserve_retains_debt_and_no_later_advances():
    folds, env = carrier(side=-1)
    for name in ("open", "high", "low", "close", "mark_price"):
        values = env.dataset.resolved_array(name).copy()
        values[8:] = 400
        object.__setattr__(env.dataset, name, values)
    plan = plans(terminal, env, folds, stop=9)
    error, events = LookupError("observer failed on reserve debt"), []

    def observe(phase, start, processing):
        events.append((phase, start, processing))
        if phase == "reserve":
            raise error

    with pytest.raises(LookupError) as caught:
        terminal.run_terminal_allocation_execution(folds, env, plan, on_advance=observe)
    assert caught.value is error
    raw = error.terminal_execution_receipt
    assert events == [("actor", 6, 7), ("reserve", 7, 8)]
    assert raw["status"] == "integrity_failure"
    assert raw["native_receipt"]["status"] == "completed"
    assert len(raw["closing_rows"]) == 1
    assert raw["final_book"]["equity"] == pytest.approx(-504.5)
    assert raw["final_book"]["insolvent"]
    assert raw["settled_profit_rate"] is None


def test_reused_marked_error_does_not_change_legacy_status_precedence():
    _, _, plan = declared()
    error = LookupError("reused caller exception")
    error._allocation_advance_observer_failure = True
    closing_rows = [
        {
            "native": {
                "book": {
                    "equity": -1,
                    "termination_reason": "insolvency",
                    "exact_quantities": ["0"],
                    "max_drawdown": 1.001,
                },
                "active_orders": [],
            },
            "fixed_capital_reward": -1.001,
        }
    ]
    raw = terminal._observation(
        plan, {"rows": [], "status": "completed"}, closing_rows, None, error=error
    )
    assert raw["status"] == "economic_stop"


def test_diagnostic_annotation_cannot_replace_the_observer_exception():
    class DiagnosticRejectingError(LookupError):
        def __setattr__(self, name, value):
            if name == "_allocation_advance_observer_failure":
                raise RuntimeError("caller exception rejects diagnostic attributes")
            super().__setattr__(name, value)

    folds, env, plan = declared()
    error = DiagnosticRejectingError("primary observer error")

    def observe(*_):
        raise error

    with pytest.raises(DiagnosticRejectingError) as caught:
        terminal.run_terminal_allocation_execution(folds, env, plan, on_advance=observe)
    assert caught.value is error
