"""Publication-gated coarse decisions on actual fine-grid spot-style replay.

The existing overlay owns causal feature values and the shared replay owns
every execution, risk, and accounting transition. No coarse OHLC is expanded
into fabricated execution bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import MarketCalendarKind, VolumeUnit
from trade_rl.data.features.clocked_overlay import (
    ClockedNativeFeatureGroup,
    clocked_native_source_digest,
)
from trade_rl.data.identity import parse_identity_json
from trade_rl.data.market import MarketDataset
from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.data.source import NativeRawMarketSeries
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.stateful.portfolio_session import resolve_agent_stop_index
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioPolicy,
    PortfolioTargetProposal,
)

_NS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True, slots=True)
class NativePublicationSlot:
    """One declared coarse event and its planned fine decision, if admitted."""

    close_ns: int
    available_at_ns: int
    status: Literal["scheduled", "expired", "outside_window"]
    decision_index: int | None = None
    decision_time_ns: int | None = None


@dataclass(frozen=True, slots=True)
class ClockedNativeReplayResult:
    """Planned slots and actual actor calls, distinct from completed fills."""

    dataset_id: str
    slots: tuple[NativePublicationSlot, ...]
    policy_decision_indices: tuple[int, ...]
    replay: SharedCashPortfolioReplayResult


def _verify_group(dataset: MarketDataset, group: ClockedNativeFeatureGroup) -> None:
    if type(dataset) is not MarketDataset or not dataset.identity_verified:
        raise ValueError(
            "native decision schedule requires a verified Dataset identity"
        )
    if (
        dataset.native_clock is not None
        or dataset.calendar_kind is not MarketCalendarKind.CONTINUOUS
        or not dataset.regular_cadence
    ):
        raise ValueError(
            "native decision schedule requires a continuous fine execution grid"
        )
    if type(group) is not ClockedNativeFeatureGroup:
        raise ValueError("expected a declared native feature group")
    if tuple(source.contract.symbol for source in group.sources) != dataset.symbols:
        raise ValueError("native decision group roster differs from Dataset")
    assert dataset.identity_payload_json is not None
    identity = parse_identity_json(dataset.identity_payload_json)
    overlay = identity.get("clocked_native_overlay")
    if (
        not isinstance(overlay, dict)
        or overlay.get("schema") != "clocked_native_feature_overlay_v1"
    ):
        raise ValueError("native decision schedule requires a clocked feature overlay")
    definition = {
        key: overlay.get(key)
        for key in ("schema", "groups", "base_feature_config_digest")
    }
    groups, pins = definition["groups"], overlay.get("group_source_digests")
    if (
        not isinstance(groups, list)
        or not isinstance(pins, list)
        or len(groups) != len(pins)
        or content_digest(definition) != dataset.feature_config_digest
    ):
        raise ValueError("native overlay identity is inconsistent")
    clock = group.clock
    expected_group = canonical_json_bytes(
        {
            "clock": {
                "kind": clock.kind.value,
                "fixed_seconds": clock.fixed_seconds,
                "fixed_anchor_ns": None
                if clock.fixed_anchor is None
                else int(clock.fixed_anchor.astype(np.int64)),
            },
            "specs": tuple(spec.canonical_payload() for spec in group.specs),
        }
    )
    expected_pins = canonical_json_bytes(
        tuple(
            {"symbol": source.contract.symbol, "sha256": source.expected_source_digest}
            for source in group.sources
        )
    )
    if (
        sum(
            canonical_json_bytes(declared_group) == expected_group
            and canonical_json_bytes(roster) == expected_pins
            for declared_group, roster in zip(groups, pins, strict=True)
        )
        != 1
    ):
        raise ValueError("native decision group must match one bound overlay group")
    closes = group.sources[0].raw.timestamps
    starts, ends = clock.interval_bounds(closes)
    first = int(dataset.timestamps[0].astype(np.int64))
    last = int(dataset.timestamps[-1].astype(np.int64))
    step = int(dataset.timestamps[1].astype(np.int64)) - first
    for start, end in zip(starts.view(np.int64), ends.view(np.int64), strict=True):
        lo, hi = int(start), int(end)
        if (lo - first) % step or (hi - first) % step or hi - lo <= step:
            raise ValueError(
                "native decision clock must be larger and aligned to fine grid"
            )
    if int(ends[-1].astype(np.int64)) > last:
        raise ValueError("native decision source exceeds fine Dataset")
    for source in group.sources:
        if type(source.raw) is not NativeRawMarketSeries:
            raise ValueError(
                "native decision schedule requires strict explicit receipt sources"
            )
        if not np.array_equal(source.raw.timestamps, closes):
            raise ValueError(
                "native decision sources must share the same complete close grid"
            )
        actual = clocked_native_source_digest(
            source.raw, contract=source.contract, clock=clock
        )
        if actual != source.expected_source_digest:
            raise ValueError("native decision source digest mismatch")


def _expired(clock: NativeBarClock, close_ns: int, decision_ns: int) -> bool:
    # Expiry is a property of the declared clock, not future observed receipts.
    if clock.kind is NativeClockKind.UTC_MONTH:
        return bool(
            np.datetime64(close_ns, "ns").astype("datetime64[M]")
            != np.datetime64(decision_ns, "ns").astype("datetime64[M]")
        )
    seconds = (
        7 * 86400 if clock.kind is NativeClockKind.UTC_WEEK else clock.fixed_seconds
    )
    assert seconds is not None
    return decision_ns - close_ns >= seconds * _NS_PER_SECOND


def clocked_native_decision_schedule(
    dataset: MarketDataset,
    group: ClockedNativeFeatureGroup,
    *,
    start_index: int,
    stop_index: int,
) -> tuple[NativePublicationSlot, ...]:
    """Map each common complete coarse bar to its first observable fine close.

    The actor window is [start_index, stop_index); the final Dataset row has
    no next interval. All symbols must have published the bar. Exact ties are
    admitted under the existing close/next-open convention; a receipt one
    nanosecond later waits until the next fine close. If this first eligible
    fine close is at/after the next coarse close, the event expires, even when
    the newer bar has not yet arrived. No catch-up at window start and no
    invented fills. Earlier warm-up events outside the fine grid do not act.
    Only raw publication is gated; the current as-of feature masks, warm-up
    and staleness still belong to the overlay and the policy.
    """
    _verify_group(dataset, group)
    if (
        type(start_index) is not int
        or type(stop_index) is not int
        or not 0 <= start_index < stop_index < dataset.n_bars
    ):
        raise ValueError("invalid native decision window")
    fine = dataset.timestamps.view(np.int64)
    closes = group.sources[0].raw.timestamps.view(np.int64)
    # Every source already passed the same complete clock/grid and pin checks.
    published = closes.copy()
    for source in group.sources:
        assert source.raw.available_at is not None
        np.maximum(published, source.raw.available_at.view(np.int64), out=published)
    candidate_indices = np.searchsorted(fine, published, side="left")
    slots: list[NativePublicationSlot] = []
    for close, receipt, candidate in zip(
        closes, published, candidate_indices, strict=True
    ):
        close_ns, available_ns, index = int(close), int(receipt), int(candidate)
        if close_ns < int(fine[0]) or not start_index <= index < stop_index:
            slots.append(
                NativePublicationSlot(close_ns, available_ns, "outside_window")
            )
            continue
        decision_ns = int(fine[index])
        if _expired(group.clock, close_ns, decision_ns):
            slots.append(NativePublicationSlot(close_ns, available_ns, "expired"))
        else:
            slots.append(
                NativePublicationSlot(
                    close_ns, available_ns, "scheduled", index, decision_ns
                )
            )
    return tuple(slots)


class _PublicationGatedPolicy:
    def __init__(
        self, policy: PortfolioPolicy, slots: tuple[NativePublicationSlot, ...]
    ) -> None:
        self._policy = policy
        self._decisions = {
            slot.decision_index: slot.decision_time_ns
            for slot in slots
            if slot.status == "scheduled"
        }
        self.called: list[int] = []

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        if snapshot.index not in self._decisions:
            return PortfolioTargetProposal(
                snapshot.account.weights, tuple(range(len(snapshot.symbols)))
            )
        if self._decisions[snapshot.index] != snapshot.timestamp_ns:
            raise ValueError("native publication schedule timestamp mismatch")
        self.called.append(snapshot.index)
        return self._policy.decide_portfolio(snapshot)


def run_clocked_native_portfolio_replay(
    dataset: MarketDataset,
    policy: PortfolioPolicy,
    *,
    decision_group: ClockedNativeFeatureGroup,
    start_index: int,
    stop_index: int,
    initial_capital: float,
    execution_cost: ExecutionCostConfig,
    risk: PreTradeRisk | None = None,
    capture_ledger_evidence: bool = False,
) -> ClockedNativeReplayResult:
    """Run sparse actor calls while the canonical engine advances every fine bar.

    The first scope is strict spot cash, deterministic MARKET, zero additional
    latency, previous-bar capacity, no pending partial-fill carry. Hard risk may
    still override off-schedule exact HOLD and the native terminal tail attempts
    flattening. The caller must inspect actual settlement/termination.
    Delayed/unavailable fine bars and non-spot carry are not modeled
    by this entrypoint. This is a software contract, not execution authority.
    """
    if type(execution_cost) is not ExecutionCostConfig or (
        execution_cost.order_type != "market"
        or execution_cost.order_latency_bars != 0
        or execution_cost.partial_fill_carry
        or execution_cost.processing_bar_volume_capacity
        or execution_cost.allow_short
        or execution_cost.max_leverage > 1.0
        or execution_cost.collateral_haircut != 1.0
        or execution_cost.slippage_std != 0.0
        or execution_cost.tail_slippage_probability != 0.0
    ):
        raise ValueError(
            "scheduled replay requires deterministic strict spot MARKET without pending carry"
        )
    if type(dataset) is not MarketDataset:
        raise TypeError("expected a MarketDataset")
    if not callable(getattr(policy, "decide_portfolio", None)):
        raise TypeError("portfolio policy must define decide_portfolio")
    if (
        type(start_index) is not int
        or type(stop_index) is not int
        or not 0 <= start_index < stop_index < dataset.n_bars
    ):
        raise ValueError("invalid native replay window")
    actor_stop = resolve_agent_stop_index(
        start_index=start_index,
        stop_index=stop_index,
        execution_cost=execution_cost,
        settle_terminal_position=True,
    )
    slots = clocked_native_decision_schedule(
        dataset, decision_group, start_index=start_index, stop_index=actor_stop
    )
    span = slice(start_index, stop_index + 1)
    usable = (
        dataset.tradable[span]
        & dataset.resolved_array("asset_active")[span]
        & dataset.resolved_array("information_available")[span]
        & (
            dataset.resolved_array("available_at")[span]
            <= dataset.timestamps[span, None]
        )
    )
    if not np.all(usable):
        raise ValueError(
            "scheduled replay requires complete available fine execution bars"
        )
    if (
        np.any(dataset.resolved_array("funding_event_count")[span])
        or np.any(dataset.resolved_array("funding_due")[span])
        or np.any(dataset.funding_rate[span] != 0.0)
        or (
            dataset.funding_price_rate is not None
            and np.any(dataset.funding_price_rate[span] != 0.0)
        )
        or np.any(dataset.resolved_array("contract_multipliers") != 1.0)
        or any(
            unit not in (VolumeUnit.BASE_ASSET, VolumeUnit.QUOTE_NOTIONAL)
            for unit in dataset.volume_units
        )
    ):
        raise ValueError("scheduled replay requires spot-style fine execution data")
    gated = _PublicationGatedPolicy(policy, slots)
    replay = run_shared_portfolio_replay(
        dataset,
        gated,
        start_index=start_index,
        stop_index=stop_index,
        initial_capital=initial_capital,
        execution_cost=execution_cost,
        risk=risk,
        settle_terminal_position=True,
        strict_spot_cash=True,
        capture_ledger_evidence=capture_ledger_evidence,
    )
    return ClockedNativeReplayResult(
        dataset.dataset_id, slots, tuple(gated.called), replay
    )


__all__ = [
    "ClockedNativeReplayResult",
    "NativePublicationSlot",
    "clocked_native_decision_schedule",
    "run_clocked_native_portfolio_replay",
]
