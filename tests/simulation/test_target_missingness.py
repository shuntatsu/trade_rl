"""Synthetic missingness must fail before target/risk/order state changes."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from tests.simulation.test_execution_v2 import market
from trade_rl.risk.pretrade import PreTradeRisk, RiskConstrainedTarget
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor
from trade_rl.simulation.native_transition import execute_native_target_transition
from trade_rl.simulation.orders.model import OrderBookState, OrderType, TimeInForce
from trade_rl.simulation.orders.reconciliation import reconcile_target
from trade_rl.simulation.stateful.portfolio_session import SharedPortfolioSession
from trade_rl.simulation.targets.execution import execute_target_statefully


@pytest.mark.parametrize(
    "field", ["target", "current", "emergency_flatten_mask", "reduce_only_mask"]
)
def test_risk_does_not_erase_missingness(field: str) -> None:
    inputs = {"target": np.array([0.5]), "current": np.array([0.2]), "drawdown": 0.0}
    hidden = True if field.endswith("mask") else 0.5
    inputs[field] = np.ma.array([hidden], mask=[True])
    with pytest.raises(ValueError, match="masked array"):
        PreTradeRisk.default_for_execution(max_leverage=1.0).constrain(**inputs)


@pytest.mark.parametrize("field", ["weights", "proposal_weights", "pretrade_weights"])
def test_risk_receipt_does_not_erase_missingness(field: str) -> None:
    inputs = dict(
        weights=np.array([0.5]),
        requested_turnover=0.5,
        constrained_turnover=0.5,
        was_constrained=False,
        reasons=(),
        risk_scale=1.0,
    )
    inputs[field] = np.ma.array([0.5], mask=[True])
    with pytest.raises(ValueError, match="masked array"):
        RiskConstrainedTarget(**inputs)


@pytest.mark.parametrize(
    "path", ["native", "compatibility", "stateful", "reconciliation", "session"]
)
@pytest.mark.parametrize(
    "wrapper",
    ["direct", "owner_view", "memoryview", "nested", "stride", "stride_memoryview"],
)
def test_target_paths_refuse_hidden_weight_before_execution(
    path: str, wrapper: str
) -> None:
    data = market()
    events = []
    executor = MarketExecutor(
        data, ExecutionCostConfig.zero(), execution_observer=events.append
    )
    book = BookState.zero(1, 1000.0, data.close[0])
    state = OrderBookState.empty()
    risk = PreTradeRisk.default_for_execution(max_leverage=1.0)
    masked = np.ma.array([0.5], mask=[True])
    if wrapper == "owner_view":
        hidden = masked.astype(np.float64).view(np.ndarray)
        assert np.ma.isMaskedArray(hidden.base)
    elif wrapper in {"stride", "stride_memoryview"}:
        owner_view = masked.astype(np.float64).view(np.ndarray)
        hidden = np.lib.stride_tricks.as_strided(owner_view)
        assert np.ma.isMaskedArray(hidden.base.base.base)
        if wrapper == "stride_memoryview":
            hidden = memoryview(hidden)
    elif wrapper == "memoryview":
        hidden = memoryview(masked)
    elif wrapper == "nested":
        hidden = [masked]
    else:
        hidden = masked
    session = SharedPortfolioSession(
        data,
        start_index=0,
        stop_index=3,
        initial_capital=1000.0,
        execution_observer=events.append,
    )
    with pytest.raises(ValueError, match="masked array|opaque array ownership"):
        if path == "native":
            execute_native_target_transition(
                executor, book, risk, hidden, start_index=0
            )
        elif path == "compatibility":
            executor.execute_interval(book, hidden, start_index=0, bars=1)
        elif path == "stateful":
            execute_target_statefully(
                executor,
                book,
                state,
                hidden,
                start_index=0,
                bars=1,
                target_identity="synthetic",
            )
        elif path == "reconciliation":
            reconcile_target(
                dataset_id=data.dataset_id,
                target_identity="synthetic",
                execution_policy_digest=executor.execution_policy_digest,
                target_weights=hidden,
                book=book,
                order_book=state,
                reference_prices=data.close[0],
                decision_equity=1000.0,
                submit_index=0,
                latency_bars=0,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
                expiry_index=None,
                limit_offset_rate=0.0,
            )
        else:
            session.advance(hidden)
    assert events == []
    assert book.exact_quantities == (Fraction(0),)
    assert book.cash == 1000.0
    assert executor.compatibility_order_book.active_orders == ()
    assert session.current_index == 0 and session.can_decide and not session.poisoned
    assert session.book.cash == 1000.0
    assert state.active_orders == ()


@pytest.mark.parametrize("field", ["emergency_flatten_mask", "reduce_only_mask"])
def test_native_target_rejects_masked_safety_control(field: str) -> None:
    data = market()
    events = []
    executor = MarketExecutor(
        data, ExecutionCostConfig.zero(), execution_observer=events.append
    )
    book = BookState.zero(1, 1000.0, data.close[0])
    risk = PreTradeRisk.default_for_execution(max_leverage=1.0)
    with pytest.raises(ValueError, match="masked array"):
        execute_native_target_transition(
            executor,
            book,
            risk,
            np.array([0.5]),
            start_index=0,
            **{field: np.ma.array([False], mask=[True])},
        )
    assert events == []
    assert book.exact_quantities == (Fraction(0),)


def test_native_target_accepts_valid_list_and_boolean_mask() -> None:
    data = market()
    executor = MarketExecutor(data, ExecutionCostConfig.zero())
    book = BookState.zero(1, 1000.0, data.close[0])
    risk = PreTradeRisk.default_for_execution(max_leverage=1.0)
    target, result, _ = execute_native_target_transition(
        executor,
        book,
        risk,
        [0.5],
        start_index=0,
        emergency_flatten_mask=[False],
        reduce_only_mask=(False,),
    )
    assert target.weights.tolist() == [0.5]
    assert result.book.exact_quantities == (Fraction(5),)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    "field", ["target", "emergency_flatten_mask", "reduce_only_mask"]
)
def test_snapshot_does_not_promote_object_array_to_valid_dtype(
    field: str, wrapped: bool
) -> None:
    raw = 0.5 if field == "target" else True
    value = [np.array(raw, dtype=object)] if wrapped else np.array([raw], dtype=object)
    assert np.asarray(value).dtype == np.dtype(object)
    data = market()
    executor = MarketExecutor(data, ExecutionCostConfig.zero())
    book = BookState.zero(1, 1000.0, data.close[0])
    risk = PreTradeRisk.default_for_execution(max_leverage=1.0)
    if field == "target":
        with pytest.raises(TypeError, match="object arrays"):
            execute_native_target_transition(executor, book, risk, value, start_index=0)
    else:
        with pytest.raises(TypeError, match="object arrays"):
            execute_native_target_transition(
                executor, book, risk, np.array([0.5]), start_index=0, **{field: value}
            )
        with pytest.raises(TypeError, match="object arrays"):
            risk.constrain(
                np.array([0.5]), current=np.array([0.0]), drawdown=0.0, **{field: value}
            )


@pytest.mark.parametrize(
    "field", ["target", "current", "emergency_flatten_mask", "reduce_only_mask"]
)
@pytest.mark.parametrize("wrap_memoryview", [False, True])
def test_risk_refuses_opaque_retained_mask_owner(
    field: str, wrap_memoryview: bool
) -> None:
    inputs = {"target": np.array([0.5]), "current": np.array([0.2]), "drawdown": 0.0}
    dtype = np.bool_ if field.endswith("mask") else np.float64
    owner = np.ma.array([True], mask=[True]).astype(dtype).view(np.ndarray)
    hidden = np.lib.stride_tricks.as_strided(owner)
    assert np.ma.isMaskedArray(hidden.base.base.base)
    inputs[field] = memoryview(hidden) if wrap_memoryview else hidden
    with pytest.raises(ValueError, match="masked array|opaque array ownership"):
        PreTradeRisk.default_for_execution(max_leverage=1.0).constrain(**inputs)
