"""Opt-in independent-account allocator → hard risk → canonical execution.

The current-close return forecast is a declared surrogate. MARKET latency0/2
changes eligible execution, not those beliefs. This adapter neither fits
forecasts nor runs a Study, and its supplied estimates are not economic evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from fractions import Fraction
from hashlib import sha256

import numpy as np

from trade_rl.artifacts import canonical, canonical_json_bytes, hashing
from trade_rl.artifacts.hashing import content_digest
from trade_rl.risk.pretrade import PreTradeRisk, RiskConstrainedTarget
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import MarketExecutor
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderType,
    TerminalOrderArchive,
)
from trade_rl.simulation.stateful.execution import StatefulExecutionResult
from trade_rl.simulation.targets.execution import (
    execute_quantity_hold_statefully,
    execute_target_statefully,
)
from trade_rl.strategies.allocation import (
    AfterCostTargetAllocator,
    AllocationContext,
    AllocationInputs,
    AllocationProposal,
)


def _archived_context_digest(
    payload: dict[str, object], archive: TerminalOrderArchive
) -> str:
    """SHA-256 the *same* canonical payload without decoding sealed history.

    The terminal_orders field is last under the current sorted-key JSON
    contract.  Verify that exact byte boundary, and fail over to the public
    serializer whenever its contract or bindings have been changed.
    """
    compatible = (
        type(archive) is TerminalOrderArchive
        and TerminalOrderArchive._canonical_order_json_chunks is _ARCHIVE_CHUNKS
        and TerminalOrderArchive._canonical_order_json_chunks.__code__
        is _ARCHIVE_CHUNKS_CODE
        and TerminalOrderArchive._ordered_nodes is _ARCHIVE_NODES
        and TerminalOrderArchive._ordered_nodes.__code__ is _ARCHIVE_NODES_CODE
        and content_digest is _CANONICAL_DIGEST
        and content_digest.__code__ is _CANONICAL_DIGEST_CODE
        and hashing.canonical_json_bytes is _CANONICAL_ENCODER
        and canonical_json_bytes is _CANONICAL_ENCODER
        and canonical_json_bytes.__code__ is _CANONICAL_ENCODER_CODE
        and canonical.to_json_value is _CANONICAL_VALUE
        and canonical.to_json_value.__code__ is _CANONICAL_VALUE_CODE
        and canonical._mapping_value is _CANONICAL_MAPPING_VALUE
        and canonical._mapping_value.__code__ is _CANONICAL_MAPPING_VALUE_CODE
        and canonical._dataclass_value is _CANONICAL_DATACLASS_VALUE
        and canonical._dataclass_value.__code__ is _CANONICAL_DATACLASS_VALUE_CODE
        and canonical._datetime_value is _CANONICAL_DATETIME_VALUE
        and canonical._datetime_value.__code__ is _CANONICAL_DATETIME_VALUE_CODE
        and canonical.is_dataclass is _CANONICAL_IS_DATACLASS
        and canonical.fields is _CANONICAL_FIELDS
        and canonical.json.dumps is _CANONICAL_DUMPS
        and canonical.json.dumps.__code__ is _CANONICAL_DUMPS_CODE
        and canonical.json.JSONEncoder is _CANONICAL_JSON_ENCODER
        and canonical.json.JSONEncoder.__init__ is _CANONICAL_JSON_INIT
        and canonical.json.JSONEncoder.__init__.__code__ is _CANONICAL_JSON_INIT_CODE
        and canonical.json.JSONEncoder.encode is _CANONICAL_JSON_ENCODE
        and canonical.json.JSONEncoder.encode.__code__ is _CANONICAL_JSON_ENCODE_CODE
        and canonical.json.JSONEncoder.iterencode is _CANONICAL_JSON_ITERENCODE
        and canonical.json.JSONEncoder.iterencode.__code__
        is _CANONICAL_JSON_ITERENCODE_CODE
        and canonical.json.JSONEncoder.default is _CANONICAL_JSON_DEFAULT
        and canonical.json.JSONEncoder.default.__code__ is _CANONICAL_JSON_DEFAULT_CODE
    )
    if compatible:
        encoded = canonical_json_bytes({**payload, "terminal_orders": ()})
        if encoded.endswith(_TERMINAL_JSON_SUFFIX):
            digest = sha256(encoded[:-2])
            separator = b""
            for item in archive._canonical_order_json_chunks():
                digest.update(separator)
                digest.update(item)
                separator = b","
            digest.update(b"]}")
            return digest.hexdigest()
    return content_digest({**payload, "terminal_orders": archive.asdict_tuple})


_ARCHIVE_CHUNKS = TerminalOrderArchive._canonical_order_json_chunks
_ARCHIVE_CHUNKS_CODE = TerminalOrderArchive._canonical_order_json_chunks.__code__
_ARCHIVE_NODES = TerminalOrderArchive._ordered_nodes
_ARCHIVE_NODES_CODE = TerminalOrderArchive._ordered_nodes.__code__
_CANONICAL_DIGEST = content_digest
_CANONICAL_DIGEST_CODE = content_digest.__code__
_CANONICAL_ENCODER = canonical_json_bytes
_CANONICAL_ENCODER_CODE = canonical_json_bytes.__code__
_CANONICAL_VALUE = canonical.to_json_value
_CANONICAL_VALUE_CODE = canonical.to_json_value.__code__
_CANONICAL_MAPPING_VALUE = canonical._mapping_value
_CANONICAL_MAPPING_VALUE_CODE = canonical._mapping_value.__code__
_CANONICAL_DATACLASS_VALUE = canonical._dataclass_value
_CANONICAL_DATACLASS_VALUE_CODE = canonical._dataclass_value.__code__
_CANONICAL_DATETIME_VALUE = canonical._datetime_value
_CANONICAL_DATETIME_VALUE_CODE = canonical._datetime_value.__code__
_CANONICAL_IS_DATACLASS = canonical.is_dataclass
_CANONICAL_FIELDS = canonical.fields
_CANONICAL_DUMPS = canonical.json.dumps
_CANONICAL_DUMPS_CODE = canonical.json.dumps.__code__
_CANONICAL_JSON_ENCODER = canonical.json.JSONEncoder
_CANONICAL_JSON_INIT = canonical.json.JSONEncoder.__init__
_CANONICAL_JSON_INIT_CODE = canonical.json.JSONEncoder.__init__.__code__
_CANONICAL_JSON_ENCODE = canonical.json.JSONEncoder.encode
_CANONICAL_JSON_ENCODE_CODE = canonical.json.JSONEncoder.encode.__code__
_CANONICAL_JSON_ITERENCODE = canonical.json.JSONEncoder.iterencode
_CANONICAL_JSON_ITERENCODE_CODE = canonical.json.JSONEncoder.iterencode.__code__
_CANONICAL_JSON_DEFAULT = canonical.json.JSONEncoder.default
_CANONICAL_JSON_DEFAULT_CODE = canonical.json.JSONEncoder.default.__code__
_TERMINAL_JSON_SUFFIX = b'"terminal_orders":[]}'


def _context(
    executor: MarketExecutor,
    book: BookState,
    order_book: OrderBookState,
    *,
    account_id: str,
    pretrade_risk: PreTradeRisk,
    symbol_index: int,
    start_index: int,
) -> AllocationContext:
    dataset = executor.dataset
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, int)
        or not 0 <= start_index < dataset.n_bars - 1
    ):
        raise ValueError("decision index is outside the executable dataset")
    if (
        isinstance(symbol_index, bool)
        or not isinstance(symbol_index, int)
        or not 0 <= symbol_index < dataset.n_symbols
    ):
        raise ValueError("selected symbol is outside the dataset")
    # Reject stale/foreign known clocks before generating a proposal, not only
    # when the canonical executor eventually attempts the next interval.
    # Unknown clocks remain valid for the legacy/bootstrap account contract.
    book.validate_processing_clock(dataset_id=dataset.dataset_id, index=start_index)
    latency = executor.cost.order_latency_bars
    if (
        executor.cost.order_type != "market"
        or type(latency) is not int
        or latency not in {0, 2}
    ):
        raise ValueError(
            "allocation supports only MARKET orders with integer latency0 or2"
        )
    if start_index + max(1, latency) >= dataset.n_bars:
        raise ValueError("first eligible processing bar is outside the source dataset")
    if (
        not pretrade_risk.config.drawdown_start
        < pretrade_risk.config.drawdown_stop
        <= 0.20
    ):
        raise ValueError("allocation requires the 20% drawdown risk guardrail")
    if (
        book.termination_reason is not None
        or book.insolvent
        or book.portfolio_value <= 0.0
    ):
        raise ValueError("allocation requires a live, solvent account")
    if book.quantities.shape != (dataset.n_symbols,) or not np.array_equal(
        np.asarray(book.contract_multipliers),
        dataset.resolved_array("contract_multipliers"),
    ):
        raise ValueError("book shape or contract multipliers do not match dataset")
    if not np.array_equal(
        book.mark_prices, dataset.resolved_array("mark_price")[start_index]
    ):
        raise ValueError("book marks do not match the current decision valuation")
    margin_check = book.clone()
    executor._update_margin(margin_check)
    if margin_check.termination_reason is not None or margin_check.insolvent:
        raise ValueError("canonical margin check rejects this initial account")
    margin_check.refresh_drawdown()
    if book.max_drawdown + 1e-12 < margin_check.max_drawdown:
        raise ValueError("book drawdown evidence has not been refreshed")
    quantities = book.exact_quantities
    if any(quantity != 0 for i, quantity in enumerate(quantities) if i != symbol_index):
        raise ValueError("allocation v1 supports only independent_symbol accounts")
    for order in order_book.active_orders:
        if order.intent.symbol_index != symbol_index:
            raise ValueError("independent account has another symbol's pending order")
        if order.intent.order_type is not OrderType.MARKET:
            raise ValueError("allocation v1 rejects non-MARKET/protective orders")
        if order.intent.dataset_id != dataset.dataset_id:
            raise ValueError("pending order belongs to another dataset")
        if (
            order.intent.submit_index > start_index
            or order.intent.eligible_index > start_index + 1
            or (
                order.intent.expiry_index is not None
                and order.intent.expiry_index < start_index + 1
            )
            or (
                order.last_processed_index is not None
                and order.last_processed_index > start_index
            )
            or (order.trigger_index is not None and order.trigger_index > start_index)
        ):
            raise ValueError("pending order clock is incompatible with this decision")
    terminal_clock_ahead = (
        order_book.terminal_orders.max_clock_index > start_index
        if isinstance(order_book.terminal_orders, TerminalOrderArchive)
        else any(
            order.intent.submit_index > start_index
            or (
                order.last_processed_index is not None
                and order.last_processed_index > start_index
            )
            or (order.trigger_index is not None and order.trigger_index > start_index)
            for order in order_book.terminal_orders
        )
    )
    if terminal_clock_ahead:
        raise ValueError("terminal order clock is ahead of the decision")
    # Bind the full canonical book, including accepted exact quantity evidence.
    # No free-cash reservation, reconstructed P&L or second ledger is invented.
    book_facts: dict[str, object] = {}
    for field in fields(book):
        if field.name == "_exact_quantities":
            continue
        value = getattr(book, field.name)
        book_facts[field.name] = (
            value.tolist() if isinstance(value, np.ndarray) else value
        )
    book_facts["exact_quantities"] = tuple(str(q) for q in quantities)
    state_payload: dict[str, object] = {
        "schema": "independent_allocation_context_v1",
        "account_id": account_id,
        "dataset_id": dataset.dataset_id,
        "execution_policy": executor.execution_policy_digest,
        "risk_config": pretrade_risk.config,
        "symbol_index": symbol_index,
        "decision_index": start_index,
        "decision_time": str(dataset.timestamps[start_index]),
        "book": book_facts,
        "active_orders": tuple(asdict(order) for order in order_book.active_orders),
    }
    archive = order_book.terminal_orders
    state_digest = (
        _archived_context_digest(state_payload, archive)
        if isinstance(archive, TerminalOrderArchive)
        else content_digest(
            {
                **state_payload,
                "terminal_orders": tuple(asdict(order) for order in archive),
            }
        )
    )
    remaining = sum(
        (Fraction(str(order.remaining_quantity)) for order in order_book.active_orders),
        Fraction(0),
    )
    return AllocationContext(
        account_id=account_id,
        symbol=dataset.symbols[symbol_index],
        decision_time=dataset.timestamps[start_index],
        state_digest=state_digest,
        current_weight=float(book.weights[symbol_index]),
        quantity=str(quantities[symbol_index]),
        cash=book.cash,
        equity=book.portfolio_value,
        pending_remaining=str(remaining),
    )


def propose_nonrl_target(
    executor: MarketExecutor,
    book: BookState,
    order_book: OrderBookState,
    *,
    account_id: str,
    inputs: AllocationInputs,
    allocator: AfterCostTargetAllocator,
    pretrade_risk: PreTradeRisk,
    symbol_index: int,
    start_index: int,
) -> AllocationProposal:
    """Optimize against actual holdings with static limits before hard risk."""
    context = _context(
        executor,
        book,
        order_book,
        account_id=account_id,
        pretrade_risk=pretrade_risk,
        symbol_index=symbol_index,
        start_index=start_index,
    )
    first_eligible = start_index + max(1, executor.cost.order_latency_bars)
    if inputs.horizon_end < executor.dataset.timestamps[first_eligible]:
        raise ValueError("return horizon ends before the first eligible processing bar")
    cap = min(pretrade_risk.config.max_abs_weight, executor.cost.max_leverage)
    # Static bounds include actual shorting permission. Drawdown projection occurs
    # only at execution, so a held position is not implicitly scaled twice.
    resolved = replace(
        allocator,
        lower_weight=max(
            allocator.lower_weight, -cap if executor.cost.allow_short else 0.0
        ),
        upper_weight=min(allocator.upper_weight, cap),
    )
    return resolved.propose(inputs, context)


@dataclass(frozen=True, slots=True)
class NonRLExecutionResult:
    proposal: AllocationProposal
    risk_target: RiskConstrainedTarget
    execution: StatefulExecutionResult


def execute_nonrl_proposal(
    executor: MarketExecutor,
    book: BookState,
    order_book: OrderBookState,
    proposal: AllocationProposal,
    *,
    account_id: str,
    pretrade_risk: PreTradeRisk,
    start_index: int,
) -> NonRLExecutionResult:
    """Reject changed context, apply final risk once, then advance one bar."""
    if proposal.context.symbol not in executor.dataset.symbols:
        raise ValueError("proposal symbol does not belong to this dataset")
    symbol_index = executor.dataset.symbols.index(proposal.context.symbol)
    fresh = propose_nonrl_target(
        executor,
        book,
        order_book,
        account_id=account_id,
        inputs=proposal.inputs,
        allocator=proposal.allocator,
        pretrade_risk=pretrade_risk,
        symbol_index=symbol_index,
        start_index=start_index,
    )
    if fresh != proposal:
        raise ValueError("stale or altered allocation proposal/context")
    target, execution = _execute_allocation_target(
        executor,
        book,
        order_book,
        target_weight=proposal.target_weight,
        decision_digest=proposal.decision_digest,
        pretrade_risk=pretrade_risk,
        symbol_index=symbol_index,
        start_index=start_index,
    )
    return NonRLExecutionResult(proposal, target, execution)


def _execute_allocation_target(
    executor: MarketExecutor,
    book: BookState,
    order_book: OrderBookState,
    *,
    target_weight: float,
    decision_digest: str,
    pretrade_risk: PreTradeRisk,
    symbol_index: int,
    start_index: int,
) -> tuple[RiskConstrainedTarget, StatefulExecutionResult]:
    """Execute an admitted scalar decision through final risk exactly once."""
    proposed_weights = np.zeros(executor.dataset.n_symbols, dtype=np.float64)
    proposed_weights[symbol_index] = target_weight
    target = pretrade_risk.constrain(
        proposed_weights, current=book.weights, drawdown=book.max_drawdown
    )
    if np.array_equal(target.weights, book.weights):
        execution = execute_quantity_hold_statefully(
            executor,
            book,
            order_book,
            symbol_index=symbol_index,
            start_index=start_index,
        )
    else:
        identity = content_digest(
            {
                "decision": decision_digest,
                "target": target.weights.tolist(),
            }
        )
        execution = execute_target_statefully(
            executor,
            book,
            order_book,
            target.weights,
            start_index=start_index,
            bars=1,
            target_identity=identity,
        )
    return target, execution


__all__ = ["NonRLExecutionResult", "propose_nonrl_target", "execute_nonrl_proposal"]
