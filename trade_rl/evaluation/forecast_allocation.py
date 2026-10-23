"""Bind same-decision simple-price forecasts to canonical allocation/execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.allocation import (
    NonRLExecutionResult,
    execute_nonrl_proposal,
    propose_nonrl_target,
)
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import MarketExecutor
from trade_rl.simulation.orders.model import OrderBookState
from trade_rl.strategies.allocation import (
    AfterCostTargetAllocator,
    AllocationInputs,
    AllocationProposal,
)
from trade_rl.strategies.forecasts.simple_stream import FrozenSimpleReturnStream


@dataclass(frozen=True, slots=True)
class HorizonCostEstimates:
    """Causally declared rates on initial notional; no venue calibration claim."""

    symbol: str
    decision_time: np.datetime64
    available_at: np.datetime64
    horizon_end: np.datetime64
    source_identity: str
    buy_cost: float = 0.0
    sell_cost: float = 0.0
    exit_cost: float = 0.0
    funding_return: float = 0.0
    borrow_return: float = 0.0
    cash_return: float = 0.0
    valuation_basis: str = "same_close_price_return"

    def __post_init__(self) -> None:
        if self.valuation_basis != "same_close_price_return":
            raise ValueError("cost estimates must use the declared same-close basis")
        checked = AllocationInputs(
            **self._arguments(),
            expected_simple_return=0.0,
        )
        for field in ("decision_time", "available_at", "horizon_end"):
            object.__setattr__(self, field, getattr(checked, field))

    def _arguments(self) -> dict:
        return {
            field: getattr(self, field)
            for field in (
                "symbol",
                "decision_time",
                "available_at",
                "horizon_end",
                "source_identity",
                "buy_cost",
                "sell_cost",
                "exit_cost",
                "funding_return",
                "borrow_return",
                "cash_return",
            )
        }

    def payload(self) -> dict[str, object]:
        arguments = self._arguments()
        for field in ("decision_time", "available_at", "horizon_end"):
            arguments[field] = int(arguments[field].astype(np.int64))
        return {
            "schema": "horizon_cost_estimates_v1",
            "valuation_basis": self.valuation_basis,
            "cost_basis": "declared_initial_notional_rates",
            **arguments,
        }


def propose_forecast_target(
    executor: MarketExecutor,
    book: BookState,
    order_book: OrderBookState,
    *,
    account_id: str,
    stream: FrozenSimpleReturnStream,
    estimates: HorizonCostEstimates,
    allocator: AfterCostTargetAllocator,
    pretrade_risk: PreTradeRisk,
    symbol_index: int,
    start_index: int,
    expected_horizon_seconds: int,
) -> AllocationProposal:
    """Use only this decision's available packet; never substitute a stale one."""
    dataset = executor.dataset
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, int)
        or not 0 <= start_index < dataset.n_bars - 1
        or isinstance(symbol_index, bool)
        or not isinstance(symbol_index, int)
        or not 0 <= symbol_index < dataset.n_symbols
    ):
        raise ValueError("forecast allocation indices are outside the dataset")
    if (
        isinstance(expected_horizon_seconds, bool)
        or not isinstance(expected_horizon_seconds, int)
        or expected_horizon_seconds <= 0
    ):
        raise ValueError("expected horizon must be positive integer seconds")
    if not isinstance(stream, FrozenSimpleReturnStream) or not isinstance(
        estimates, HorizonCostEstimates
    ):
        raise ValueError("forecast allocation requires a typed stream and estimates")
    if stream.dataset_id != dataset.dataset_id:
        raise ValueError("forecast lineage does not match the current dataset")
    book.validate_processing_clock(
        dataset_id=dataset.dataset_id, index=start_index, require_known=True
    )
    decision = dataset.timestamps[start_index]
    symbol = dataset.symbols[symbol_index]
    matching = tuple(
        p for p in stream.packets if p.symbol == symbol and p.as_of == decision
    )
    if len(matching) != 1:
        raise ValueError(
            "exact decision packet is missing; stale fallback is forbidden"
        )
    packet = matching[0]
    owner = next(v for v in stream.vintages if v.digest == packet.vintage_digest)
    if (
        packet.forecast_available_at != decision
        or owner.block.inference_delay_seconds != 0
        or packet.horizon_seconds != expected_horizon_seconds
        or estimates.symbol != symbol
        or estimates.decision_time != decision
        or estimates.horizon_end != packet.horizon_end
        or estimates.valuation_basis != packet.valuation_basis
    ):
        raise ValueError("forecast and cost clocks, horizon or valuation do not match")
    indices = owner.model.feature_indices
    if (
        any(i >= dataset.n_features for i in indices)
        or tuple(dataset.feature_names[i] for i in indices)
        != owner.training.selected_feature_names
        or not dataset.resolved_array("information_available")[
            start_index, symbol_index
        ]
        or not np.all(
            dataset.feature_available[start_index, symbol_index, list(indices)]
        )
        or packet.source_available_at
        != dataset.resolved_array("available_at")[start_index, symbol_index]
        or packet.feature_values
        != tuple(
            float(x) for x in dataset.features[start_index, symbol_index, list(indices)]
        )
    ):
        raise ValueError(
            "packet does not match the available selected decision snapshot"
        )
    if (
        packet.decision_close != dataset.close[start_index, symbol_index]
        or packet.decision_close
        != dataset.resolved_array("mark_price")[start_index, symbol_index]
        or book.mark_prices.shape != (dataset.n_symbols,)
        or packet.decision_close != book.mark_prices[symbol_index]
    ):
        raise ValueError(
            "same-close forecast requires matching close and account valuation"
        )
    arguments = estimates._arguments()
    arguments["available_at"] = max(
        packet.forecast_available_at, estimates.available_at
    )
    arguments["source_identity"] = content_digest(
        {
            "schema": "forecast_allocation_source_v1",
            "dataset_id": dataset.dataset_id,
            "packet": packet.digest,
            "vintage": owner.digest,
            "estimates": estimates.payload(),
        }
    )
    inputs = AllocationInputs(
        **arguments,
        expected_simple_return=packet.expected_simple_return,
        return_variance=packet.fit_prefix_marginal_variance,
    )
    return propose_nonrl_target(
        executor,
        book,
        order_book,
        account_id=account_id,
        inputs=inputs,
        allocator=allocator,
        pretrade_risk=pretrade_risk,
        symbol_index=symbol_index,
        start_index=start_index,
    )


def execute_forecast_proposal(
    executor: MarketExecutor,
    book: BookState,
    order_book: OrderBookState,
    proposal: AllocationProposal,
    *,
    account_id: str,
    stream: FrozenSimpleReturnStream,
    estimates: HorizonCostEstimates,
    allocator: AfterCostTargetAllocator,
    pretrade_risk: PreTradeRisk,
    symbol_index: int,
    start_index: int,
    expected_horizon_seconds: int,
) -> NonRLExecutionResult:
    """Rebind current forecast/cost/account inputs before canonical admission."""
    current = propose_forecast_target(
        executor,
        book,
        order_book,
        account_id=account_id,
        stream=stream,
        estimates=estimates,
        allocator=allocator,
        pretrade_risk=pretrade_risk,
        symbol_index=symbol_index,
        start_index=start_index,
        expected_horizon_seconds=expected_horizon_seconds,
    )
    if current.decision_digest != proposal.decision_digest:
        raise ValueError("forecast allocation proposal inputs have changed")
    return execute_nonrl_proposal(
        executor,
        book,
        order_book,
        proposal,
        account_id=account_id,
        pretrade_risk=pretrade_risk,
        start_index=start_index,
    )


__all__ = [
    "HorizonCostEstimates",
    "propose_forecast_target",
    "execute_forecast_proposal",
]
