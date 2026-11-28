"""Opt-in duration-bound session replay; no inferred calendar or new accounting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction

import numpy as np

from trade_rl._validation import _ordered_ns_durations
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.data.contracts import MarketCalendarKind
from trade_rl.data.market import MarketDataset
from trade_rl.data.native_execution_clock import frequency_payload
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.evaluation.series import SessionElapsedReturnSeries
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.simulation.execution import (
    ExecutionCostConfig,
    _preflight_session_execution_window,
)
from trade_rl.strategies.portfolio import PortfolioPolicy

SESSION_ELAPSED_RETURN_SCHEMA = "session_elapsed_return_series_v1"


def _session_boundaries(
    dataset: MarketDataset, *, start_index: int, stop_index: int
) -> tuple[int, ...]:
    if (
        type(dataset) is not MarketDataset
        or dataset.calendar_kind is not MarketCalendarKind.SESSION
        or dataset.native_clock is not None
        or not dataset.identity_verified
    ):
        raise ValueError("expected a verified declared session Dataset")
    if (
        type(start_index) is not int
        or type(stop_index) is not int
        or not 0 <= start_index < stop_index < dataset.n_bars
    ):
        raise ValueError("invalid session elapsed return window")
    timestamps = dataset.timestamps[start_index : stop_index + 1].astype(
        "datetime64[ns]"
    )
    boundaries = tuple(int(x) for x in timestamps.view(np.int64))
    _ordered_ns_durations(boundaries)
    return boundaries


def session_elapsed_return_payload(
    dataset: MarketDataset, returns: SessionElapsedReturnSeries, *, start_index: int
) -> dict[str, object]:
    if type(returns) is not SessionElapsedReturnSeries:
        raise ValueError("expected session elapsed returns")
    if type(start_index) is not int:
        raise ValueError("invalid session return start index")
    stop_index = start_index + len(returns.values)
    if returns.boundaries_ns != _session_boundaries(
        dataset, start_index=start_index, stop_index=stop_index
    ):
        raise ValueError("session returns do not match the Dataset interval")
    assert isinstance(returns.periods_per_year, Fraction)
    return {
        "schema_version": SESSION_ELAPSED_RETURN_SCHEMA,
        "dataset_id": dataset.dataset_id,
        "start_index": start_index,
        "stop_index": stop_index,
        "kind": returns.kind.value,
        "boundaries_ns": list(returns.boundaries_ns),
        "periods_per_year": frequency_payload(returns.periods_per_year),
        "elapsed_years": frequency_payload(returns.exact_elapsed_years),
        "values": list(returns.values),
    }


def load_session_elapsed_return_payload(
    dataset: MarketDataset, payload: Mapping[str, object]
) -> SessionElapsedReturnSeries:
    if (
        type(payload) is not dict
        or payload.get("schema_version") != SESSION_ELAPSED_RETURN_SCHEMA
    ):
        raise ValueError("expected a session elapsed return payload")
    values, start = payload.get("values"), payload.get("start_index")
    if (
        type(values) is not list
        or any(type(x) is not float for x in values)
        or type(start) is not int
    ):
        raise ValueError("invalid session elapsed return payload")
    returns = SessionElapsedReturnSeries.from_boundaries(
        tuple(values),
        boundaries_ns=_session_boundaries(
            dataset, start_index=start, stop_index=start + len(values)
        ),
    )
    if canonical_json_bytes(payload) != canonical_json_bytes(
        session_elapsed_return_payload(dataset, returns, start_index=start)
    ):
        raise ValueError(
            "session elapsed return payload differs from the Dataset interval"
        )
    return returns


def run_session_elapsed_portfolio_replay(
    dataset: MarketDataset,
    policy: PortfolioPolicy,
    *,
    start_index: int,
    stop_index: int,
    expected_boundaries_ns: tuple[int, ...],
    expected_nominal_bar_seconds: int,
    initial_capital: float,
    execution_cost: ExecutionCostConfig,
    risk: PreTradeRisk | None = None,
    native_session_linear_carry: bool = False,
) -> SharedCashPortfolioReplayResult:
    """Replay a caller-pinned session grid, returning duration-bound evidence.

    The expected sequence includes the initial decision mark and every
    processing close. The sequence and nominal processing duration must be
    established independently by the caller; equality is not venue-calendar
    authentication or missing-bar discovery.
    Existing session gap/processing carry and next-open execution are used
    without alteration. The default is Spot; explicit native_session_linear_carry
    admits verified contract-unit data with one declared funding event per
    processing close, available borrow, and the same deterministic restrictions
    as native calendar carry. Only complete settled windows with whole-second
    nominal processing bars are admitted.
    """
    boundaries = _preflight_session_execution_window(
        dataset,
        execution_cost,
        start_index=start_index,
        stop_index=stop_index,
        expected_boundaries_ns=expected_boundaries_ns,
        expected_nominal_bar_seconds=expected_nominal_bar_seconds,
        native_session_linear_carry=native_session_linear_carry,
    )
    replay = run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=start_index,
        stop_index=stop_index,
        initial_capital=initial_capital,
        execution_cost=execution_cost,
        risk=risk,
        settle_terminal_position=True,
        strict_spot_cash=not native_session_linear_carry,
        native_session_linear_carry=native_session_linear_carry,
        capture_ledger_evidence=True,
    )
    ledger = replay.ledger_evidence
    if (
        ledger is None
        or len(replay.returns.values) != stop_index - start_index
        or ledger.intervals.termination_reason is not None
        or ledger.intervals.active_order_remainders
        or any(q != 0 for q in replay.book.exact_quantities)
        or tuple(row.next_index for row in ledger.intervals.intervals)
        != tuple(range(start_index + 1, stop_index + 1))
    ):
        raise ValueError(
            "session elapsed replay did not complete its declared settled window"
        )
    if native_session_linear_carry:
        # The signed native replay already produces the duration-bound series.
        if (
            type(replay.returns) is not SessionElapsedReturnSeries
            or replay.returns.boundaries_ns != boundaries
        ):
            raise RuntimeError("signed session replay returned inconsistent boundaries")
        return replay
    return replace(
        replay,
        returns=SessionElapsedReturnSeries.from_boundaries(
            replay.returns.values, boundaries_ns=boundaries
        ),
    )


__all__ = [
    "SESSION_ELAPSED_RETURN_SCHEMA",
    "load_session_elapsed_return_payload",
    "run_session_elapsed_portfolio_replay",
    "session_elapsed_return_payload",
]
