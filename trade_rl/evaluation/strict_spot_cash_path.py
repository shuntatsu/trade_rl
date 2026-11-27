"""Read-only cash-path audit for a restricted, settled strict-spot replay.

No order simulation, second mutable book, source acquisition or study admission.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Final

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.canonical import to_json_value
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import MarketCalendarKind
from trade_rl.data.identity import parse_identity_json
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import (
    SharedCashLedgerIntervalEvidence,
    SharedCashPortfolioReplayLedgerEvidence,
    SharedCashReplayLedgerEvidence,
)
from trade_rl.simulation.diagnostics.fill_cash import (
    NativeFillCashObservation,
    NativeFillCashRecord,
)
from trade_rl.simulation.execution import (
    ExecutionCostConfig,
    runtime_execution_policy_digest,
)
from trade_rl.simulation.orders.model import OrderEvent
from trade_rl.simulation.quantities import parse_quantity, project_quantity

STRICT_SPOT_CASH_PATH_SCHEMA: Final = "strict_spot_cash_path_v1"


@dataclass(frozen=True, slots=True)
class StrictSpotCashPathEvidence:
    """Result of one source-backed, restricted cash-path reconciliation."""

    dataset_id: str
    execution_policy_digest: str
    ledger_digest: str
    fill_observations_digest: str
    symbols: tuple[str, ...]
    spot_symbols: tuple[str, ...]
    start_index: int
    stop_index: int
    interval_count: int
    fill_count: int
    initial_cash: float
    minimum_cash: float
    final_cash: float
    total_execution_cost: float
    total_dividend: float
    total_cash_interest: float
    schema_version: str = field(default=STRICT_SPOT_CASH_PATH_SCHEMA, init=False)

    @property
    def digest(self) -> str:
        return content_digest(self)


def _cash(value: object) -> float:
    """Refuse even a tiny negative cash boundary; no tolerance or fee credit."""
    if type(value) not in (float, int):
        raise ValueError("cash evidence requires a native number")
    if type(value) is int and abs(value) > 2**53:
        raise ValueError("cash evidence requires a binary64-compatible number")
    result = float(value)  # type: ignore[arg-type]
    if not math.isfinite(result) or result < 0:
        raise ValueError("cash boundary must be finite and nonnegative")
    return result


def _digest(expected: str, actual: str, *, name: str) -> None:
    if type(expected) is not str:
        raise ValueError(f"{name} requires an explicit expected digest")
    require_sha256(expected, field=name)
    if expected != actual:
        raise ValueError(f"{name} mismatch")


def _quantities(values: tuple[str, ...], *, size: int) -> tuple[Fraction, ...]:
    if (
        type(values) is not tuple
        or len(values) != size
        or any(type(value) is not str for value in values)
    ):
        raise ValueError("exact quantities must match the symbol roster")
    parsed = tuple(parse_quantity(value) for value in values)
    if any(q < 0 for q in parsed):
        raise ValueError("negative spot inventory")
    return parsed


def _zero_carry(iv: SharedCashLedgerIntervalEvidence) -> None:
    if any(
        value != 0.0
        for value in (
            iv.interval_funding,
            iv.interval_borrow_cost,
            iv.funding_pnl_before,
            iv.funding_pnl_after,
            iv.borrow_cost_before,
            iv.borrow_cost_after,
        )
    ):
        raise ValueError("strict spot cash path refuses funding or borrow")


def _check_roster(
    dataset: MarketDataset,
    index: int,
    active: np.ndarray,
) -> None:
    if not np.array_equal(dataset.resolved_array("asset_active")[index], active):
        raise ValueError("cash path requires fixed active spot roster")
    if np.any(dataset.tradable[index, ~active]) or np.any(
        dataset.resolved_array("information_available")[index, ~active]
    ):
        raise ValueError("non-spot placeholder must be inactive and unavailable")


def _check_row_source(dataset: MarketDataset, index: int, active: np.ndarray) -> None:
    _check_roster(dataset, index, active)
    if np.any(dataset.resolved_array("split_factor")[index] != 1.0):
        raise ValueError("cash path refuses corporate split")
    due = dataset.resolved_array("funding_due")[index]
    rate = (
        dataset.funding_rate[index]
        if dataset.funding_price_rate is None
        else dataset.funding_price_rate[index]
    )
    if np.any(due & (rate != 0.0)):
        raise ValueError("cash path refuses nonzero funding")
    if (
        np.any(dataset.resolved_array("dividend")[index] < 0.0)
        or dataset.resolved_array("cash_rate")[index] < 0.0
    ):
        raise ValueError("cash path refuses negative non-order flows")


def _positive_carry(
    dataset: MarketDataset,
    iv: SharedCashLedgerIntervalEvidence,
    quantities: tuple[Fraction, ...],
    cash_after_fills: float,
    multipliers: np.ndarray,
) -> tuple[float, float, float]:
    """Reconcile post-fill dividend then interest, using source inputs.

    The source model and native binary64 arithmetic are the accounting
    convention; this does not independently verify the source economic rates.
    """
    projected = np.array([project_quantity(q) for q in quantities], dtype=np.float64)
    dividends = dataset.resolved_array("dividend")[iv.next_index]
    chargeable = (projected != 0.0) & (dividends != 0.0)
    basis = np.zeros(projected.shape, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        basis[chargeable] = projected[chargeable] * multipliers[chargeable]
        terms = basis * dividends
        dividend = _cash(float(np.dot(basis, dividends)))
    if not np.isfinite(terms).all() or np.any(chargeable & (terms == 0.0)):
        raise ValueError("dividend is not representable")
    if dividend != iv.interval_dividend:
        raise ValueError("native dividend does not match source")
    cash_with_dividend = _cash(cash_after_fills + dividend)
    rate = float(dataset.resolved_array("cash_rate")[iv.next_index])
    fraction = dataset.elapsed_year_fraction(iv.start_index, iv.next_index)
    if cash_with_dividend == 0.0 or rate == 0.0 or fraction == 0.0:
        interest = 0.0
    else:
        interest = _cash(cash_with_dividend * rate * fraction)
        if interest == 0.0:
            raise ValueError("nonzero cash interest is not representable")
    if interest != iv.interval_cash_interest:
        raise ValueError("native interest does not match source")
    return dividend, interest, _cash(cash_with_dividend + interest)


def verify_strict_spot_cash_path(
    dataset: MarketDataset,
    ledger: SharedCashPortfolioReplayLedgerEvidence,
    observations: tuple[NativeFillCashObservation, ...],
    *,
    execution_cost: ExecutionCostConfig,
    expected_dataset_id: str,
    expected_execution_policy_digest: str,
    expected_ledger_digest: str,
    expected_fill_observations_digest: str,
    expected_symbols: tuple[str, ...],
    expected_spot_symbols: tuple[str, ...],
    expected_start_index: int,
    expected_stop_index: int,
    expected_initial_cash: float,
    market_order_profile_digest: str | None = None,
) -> StrictSpotCashPathEvidence:
    """Reconcile every cash boundary of a completed, restricted spot replay.

    Required independent pins are for the Dataset, strict execution policy,
    canonical ledger mapping, and ordered fill-observation tuple. Caller also
    supplies the reviewed execution config and declared ordered spot subset.
    Require a content-verified continuous regular Dataset, fixed active
    spot roster (other symbols strictly inactive placeholders), one native
    bar per interval, zero starting holdings/cumulative costs, full terminal
    flatness, and no outstanding orders or economic termination. Due funding,
    borrow, split, lifecycle changes, session gaps and non-order settlement
    are outside this first supported scope and fail closed. Nonnegative
    source dividends and interest are accounted in their actual post-fill
    order. Check each fill against native ledger events and every cash/quantity
    and cumulative cost transition using the binary64 accounting convention.

    This is an in-memory observer audit, not an untrusted JSON reader or
    producer, a full execution-price/fee/risk oracle, source/venue attestation,
    or economic authorization. It cannot authenticate the caller's pins.
    """
    if type(dataset) is not MarketDataset or not dataset.identity_verified:
        raise ValueError("cash path requires verified Dataset identity")
    assert dataset.identity_payload_json is not None
    if parse_identity_json(dataset.identity_payload_json).get(
        "dataset_contract"
    ) != to_json_value(dataset.identity_contract_payload()):
        raise ValueError("Dataset identity contract differs")
    _digest(expected_dataset_id, dataset.dataset_id, name="Dataset identity")
    if (
        dataset.calendar_kind is not MarketCalendarKind.CONTINUOUS
        or not dataset.regular_cadence
    ):
        raise ValueError("cash path requires continuous regular native bars")
    if (
        type(expected_symbols) is not tuple
        or expected_symbols != dataset.symbols
        or type(expected_spot_symbols) is not tuple
        or not expected_spot_symbols
        or expected_spot_symbols
        != tuple(s for s in expected_symbols if s in expected_spot_symbols)
    ):
        raise ValueError("expected ordered spot roster differs")
    active = np.asarray(
        [s in expected_spot_symbols for s in expected_symbols], dtype=np.bool_
    )
    if (
        type(expected_start_index) is not int
        or type(expected_stop_index) is not int
        or not 0 <= expected_start_index < expected_stop_index < dataset.n_bars
    ):
        raise ValueError("invalid expected cash path interval")
    initial_cash = _cash(expected_initial_cash)
    if initial_cash == 0.0:
        raise ValueError("cash path requires positive initial cash")
    policy = runtime_execution_policy_digest(
        execution_cost,
        funding_price_rate_present=dataset.funding_price_rate is not None,
        market_order_profile_digest=market_order_profile_digest,
        strict_spot_cash=True,
    )
    _digest(
        expected_execution_policy_digest,
        policy,
        name="strict execution policy identity",
    )
    if (
        type(ledger) is not SharedCashPortfolioReplayLedgerEvidence
        or type(ledger.intervals) is not SharedCashReplayLedgerEvidence
        or type(observations) is not tuple
        or any(type(row) is not NativeFillCashObservation for row in observations)
    ):
        raise ValueError("cash path requires typed native ledger and fill observations")
    _digest(
        expected_ledger_digest,
        content_digest(ledger.to_mapping()),
        name="ledger digest",
    )
    _digest(
        expected_fill_observations_digest,
        content_digest(observations),
        name="fill observations digest",
    )
    native = ledger.intervals
    count = expected_stop_index - expected_start_index
    if (
        native.schema_version
        not in ("shared_cash_replay_ledger_v1", "shared_cash_replay_ledger_v2")
        or native.dataset_id != expected_dataset_id
        or native.execution_policy_digest != policy
        or native.start_index != expected_start_index
        or native.stop_index != expected_stop_index
    ):
        raise ValueError("native ledger identity mismatch")
    if (
        type(native.intervals) is not tuple
        or len(native.intervals) != count
        or len(observations) != count
        or native.termination_reason is not None
    ):
        raise ValueError("cash path requires a complete nonterminated interval roster")
    multipliers = dataset.resolved_array("contract_multipliers")
    # Dataset admission already verifies lossless ns conversion, but retains
    # the original datetime64 storage unit. Native event evidence is in ns.
    timestamps_ns = (
        dataset.timestamps[expected_start_index : expected_stop_index + 1]
        .astype("datetime64[ns]")
        .astype(np.int64)
    )
    _check_roster(dataset, expected_start_index, active)
    cash = initial_cash
    lowest = cash
    total_cost = total_dividend = total_interest = 0.0
    fill_count = 0
    inventory = (Fraction(0),) * dataset.n_symbols
    for offset, (iv, batch) in enumerate(
        zip(native.intervals, observations, strict=True)
    ):
        start = expected_start_index + offset
        stop = start + 1
        if (
            type(iv) is not SharedCashLedgerIntervalEvidence
            or (iv.start_index, iv.next_index) != (start, stop)
            or (batch.start_index, batch.next_index) != (start, stop)
            or batch.dataset_id != expected_dataset_id
            or batch.execution_policy_digest != policy
            or iv.termination_reason is not None
        ):
            raise ValueError("cash path interval or observer identity mismatch")
        _check_row_source(dataset, stop, active)
        _zero_carry(iv)
        if (
            _cash(iv.cash_before) != cash
            or _quantities(iv.exact_quantities_before, size=dataset.n_symbols)
            != inventory
            or _cash(iv.total_cost_before) != total_cost
        ):
            raise ValueError("cash path is discontinuous")
        if type(iv.order_events) is not tuple:
            raise ValueError("cash path requires a native order event tuple")
        for sequence, event in enumerate(iv.order_events):
            if (
                type(event) is not OrderEvent
                or event.sequence != sequence
                or event.dataset_id != expected_dataset_id
                or event.execution_policy_digest != policy
                or not 0 <= event.symbol_index < dataset.n_symbols
                or not start <= event.processing_index <= stop
                or event.timestamp_ns
                != int(timestamps_ns[event.processing_index - expected_start_index])
            ):
                raise ValueError("native order event identity or sequence mismatch")
        events = tuple(
            event
            for event in iv.order_events
            if event.event_type in ("filled", "partial_fill")
        )
        if (
            tuple(row.event for row in batch.fills) != events
            or batch.interval_cost != iv.interval_cost
        ):
            raise ValueError("missing or mismatched native fill records")
        quantities = list(inventory)
        interval_cost = 0.0
        for row in batch.fills:
            if type(row) is not NativeFillCashRecord:
                raise ValueError("expected a native fill cash record")
            symbol = row.event.symbol_index
            before = parse_quantity(row.quantity_before)
            after = parse_quantity(row.quantity_after)
            if (
                not active[symbol]
                or not dataset.tradable[stop, symbol]
                or row.event.processing_index != stop
                or row.contract_multiplier != float(multipliers[symbol])
                or before != quantities[symbol]
                or after < 0
                or _cash(row.cash_before) != cash
                or _cash(row.total_cost_before) != total_cost
            ):
                raise ValueError(
                    "fill cash/quantity/source transition is discontinuous"
                )
            cost = _cash(row.cost_amount)
            principal = row.signed_notional
            if not math.isfinite(principal) or principal == 0.0:
                raise ValueError("invalid native fill principal")
            cash = _cash(cash - (principal + cost))
            total_cost = _cash(total_cost + cost)
            if cash != _cash(row.cash_after) or total_cost != _cash(
                row.total_cost_after
            ):
                raise ValueError("native fill cash or cost does not reconcile")
            lowest = min(lowest, cash)
            quantities[symbol] = after
            interval_cost = _cash(interval_cost + cost)
            fill_count += 1
        inventory = tuple(quantities)
        if (
            inventory != _quantities(iv.exact_quantities_after, size=dataset.n_symbols)
            or total_cost != _cash(iv.total_cost_after)
            or interval_cost != _cash(iv.interval_cost)
        ):
            raise ValueError("cash path inventory or execution cost does not reconcile")
        dividend, interest, cash = _positive_carry(
            dataset, iv, inventory, cash, multipliers
        )
        if cash != _cash(iv.cash_after):
            raise ValueError("unexplained non-order cash movement")
        total_dividend = _cash(total_dividend + dividend)
        total_interest = _cash(total_interest + interest)
    if (
        any(inventory)
        or _quantities(native.terminal_exact_quantities, size=dataset.n_symbols)
        != inventory
        or type(native.active_order_remainders) is not tuple
        or native.active_order_remainders
        or _cash(native.final_cash) != cash
        or _cash(native.final_portfolio_value) != cash
        or _cash(native.final_total_cost) != total_cost
        or native.final_funding_pnl != 0.0
        or native.final_borrow_cost != 0.0
    ):
        raise ValueError("cash path requires reconciled terminal flatness")
    return StrictSpotCashPathEvidence(
        dataset_id=expected_dataset_id,
        execution_policy_digest=policy,
        ledger_digest=expected_ledger_digest,
        fill_observations_digest=expected_fill_observations_digest,
        symbols=expected_symbols,
        spot_symbols=expected_spot_symbols,
        start_index=expected_start_index,
        stop_index=expected_stop_index,
        interval_count=count,
        fill_count=fill_count,
        initial_cash=initial_cash,
        minimum_cash=lowest,
        final_cash=cash,
        total_execution_cost=total_cost,
        total_dividend=total_dividend,
        total_cash_interest=total_interest,
    )


__all__ = [
    "STRICT_SPOT_CASH_PATH_SCHEMA",
    "StrictSpotCashPathEvidence",
    "verify_strict_spot_cash_path",
]
