from __future__ import annotations

import json
from dataclasses import asdict, fields

import numpy as np

from tests.evaluation.test_after_cost_allocation import market
from tools.market_b4_oracle.decision_identity import (
    OwnedAccountComponents,
    build_actor_identity,
)
from trade_rl.evaluation.allocation import _context, propose_nonrl_target
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import BookState, MarketExecutor
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders.model import OrderBookState
from trade_rl.strategies.allocation import AfterCostTargetAllocator, AllocationInputs


def _owned_account(
    executor, book, risk, *, account_id: str, symbol_index: int, index: int
):
    facts = {}
    for field in fields(book):
        if field.name == "_exact_quantities":
            continue
        value = getattr(book, field.name)
        facts[field.name] = value.tolist() if isinstance(value, np.ndarray) else value
    facts["exact_quantities"] = tuple(str(value) for value in book.exact_quantities)
    return OwnedAccountComponents(
        account_id=account_id,
        dataset_id=executor.dataset.dataset_id,
        symbols=tuple(executor.dataset.symbols),
        execution_policy_digest=executor.execution_policy_digest,
        risk_config=asdict(risk.config),
        symbol_index=symbol_index,
        decision_index=index,
        decision_time=str(executor.dataset.timestamps[index]),
        book=facts,
    )


def test_context_identity_matches_native_context_from_owned_synthetic_inputs():
    # Synthetic two-symbol Dataset; the frozen native allocation path only permits
    # one nonzero symbol here. The second slot preserves a signed-zero reporting
    # bit while exact inventory correctly remains zero.
    dataset = market(next_close=123.456789, n_symbols=2)
    index = 1
    book = BookState(
        quantities=np.asarray([0.1, -0.0], dtype=np.float64),
        cash=99987.65432110002,
        mark_prices=dataset.resolved_array("mark_price")[index],
        peak_value=100100.0,
        contract_multipliers=dataset.resolved_array("contract_multipliers"),
        as_of_index=index,
        as_of_dataset_id=dataset.dataset_id,
        _exact_quantities=("1/10", "0"),
    )
    book.refresh_drawdown()
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_abs_weight=1.0,
            max_turnover=None,
            drawdown_start=0.10,
            drawdown_stop=0.20,
        )
    )
    orders = OrderBookState.empty()
    account_id = "synthetic-owned-context"

    native_context = _context(
        executor,
        book,
        orders,
        account_id=account_id,
        pretrade_risk=risk,
        symbol_index=0,
        start_index=index,
    )
    decision_time = dataset.timestamps[index]
    inputs = AllocationInputs(
        symbol=dataset.symbols[0],
        decision_time=decision_time,
        available_at=decision_time,
        horizon_end=decision_time + np.timedelta64(24, "h"),
        source_identity="a" * 64,
        expected_simple_return=0.08,
        return_variance=0.01,
        buy_cost=0.001,
        sell_cost=0.001,
    )
    allocator = AfterCostTargetAllocator(
        lower_weight=-0.5,
        upper_weight=0.5,
        max_turnover=None,
        risk_aversion=1.5,
    )
    proposal = propose_nonrl_target(
        executor,
        book,
        orders,
        account_id=account_id,
        inputs=inputs,
        allocator=allocator,
        pretrade_risk=risk,
        symbol_index=0,
        start_index=index,
    )
    assert proposal.context == native_context

    owned = _owned_account(
        executor, book, risk, account_id=account_id, symbol_index=0, index=index
    )
    input_fields = asdict(inputs)
    for name in ("decision_time", "available_at", "horizon_end"):
        input_fields[name] = str(input_fields[name])
    decision_fields = {
        "feature_names": ("signal",),
        "feature_values": (-0.0,),
        "max_drawdown": book.max_drawdown,
        "initial_capital": book.peak_value,
        "remaining_steps": 4,
        "pending_gross": 0.0,
        "pending_count": 0,
    }
    built = build_actor_identity(
        owned,
        input_fields,
        asdict(proposal.allocator),
        independent_target_weight=proposal.target_weight,
        independent_objective_value=proposal.objective_value,
        decision_fields=decision_fields,
        action_contract={"mode": "residual", "scale": 0.5},
        action_code=2,
        final_risk_weights=tuple(float(value) for value in book.weights),
    )

    expected_context = asdict(native_context)
    expected_context["decision_time"] = str(expected_context["decision_time"])
    assert json.loads(built.context_bytes) == expected_context
    assert built.context_state_digest == native_context.state_digest
    assert built.proposal_digest == proposal.decision_digest
