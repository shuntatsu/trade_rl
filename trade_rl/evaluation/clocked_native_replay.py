"""Publication-gated coarse decisions on actual fine-grid native replay.

The existing builders/overlay own causal features and the shared replay owns
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
from trade_rl.data.native_execution_clock import execution_clock_payload
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


def _decision_group_payload(group: ClockedNativeFeatureGroup) -> dict[str, object]:
    clock = group.clock
    return {
        "clock": {
            "kind": clock.kind.value,
            "fixed_seconds": clock.fixed_seconds,
            "fixed_anchor_ns": None
            if clock.fixed_anchor is None
            else int(clock.fixed_anchor.astype(np.int64)),
        },
        "specs": tuple(spec.canonical_payload() for spec in group.specs),
    }


def _verify_overlay_binding(
    dataset: MarketDataset,
    group: ClockedNativeFeatureGroup,
    identity: dict[str, object],
) -> None:
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
    expected_group = canonical_json_bytes(_decision_group_payload(group))
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


def _verify_native_fixed_binding(
    dataset: MarketDataset,
    group: ClockedNativeFeatureGroup,
    identity: dict[str, object],
) -> None:
    """Match the existing v8 spot builder's independent source roster."""
    assert dataset.native_clock is not None
    config = identity.get("config")
    if (
        not isinstance(config, dict)
        or config.get("schema_version") != "native_fixed_spot_build_v1"
        or content_digest(config) != dataset.feature_config_digest
        or config.get("execution_clock")
        != execution_clock_payload(dataset.native_clock)
    ):
        raise ValueError("native fixed decision schedule requires the bound spot build")
    features = config.get("features")
    if (
        not isinstance(features, list)
        or any(not isinstance(spec, dict) for spec in features)
        or tuple(spec.get("name") for spec in features) != dataset.feature_names
        or canonical_json_bytes(config.get("global_feature_names"))
        != canonical_json_bytes(dataset.global_feature_names)
    ):
        raise ValueError("native fixed build feature roster differs from Dataset")
    groups = config.get("groups")
    pins = identity.get("feature_source_digests")
    if (
        not isinstance(groups, list)
        or not isinstance(pins, list)
        or len(groups) != len(pins)
        or not groups
        or any(
            not isinstance(roster, list) or len(roster) != dataset.n_symbols
            for roster in pins
        )
    ):
        raise ValueError("native fixed feature source binding is inconsistent")
    if canonical_json_bytes(identity.get("instruments")) != canonical_json_bytes(
        tuple(source.contract.canonical_payload() for source in group.sources)
    ):
        raise ValueError(
            "native fixed decision group instrument contracts differ from build"
        )
    expected_group = canonical_json_bytes(_decision_group_payload(group))
    expected_pins = tuple(source.expected_source_digest for source in group.sources)
    if (
        sum(
            canonical_json_bytes(definition) == expected_group
            and tuple(roster) == expected_pins
            for definition, roster in zip(groups, pins, strict=True)
        )
        != 1
    ):
        raise ValueError("native decision group must match one bound spot build group")


def _verify_group(
    dataset: MarketDataset, group: ClockedNativeFeatureGroup
) -> np.ndarray:
    if type(dataset) is not MarketDataset or not dataset.identity_verified:
        raise ValueError(
            "native decision schedule requires a verified Dataset identity"
        )
    if (
        dataset.calendar_kind is not MarketCalendarKind.CONTINUOUS
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
    native_fixed = dataset.native_clock is not None
    if native_fixed:
        _verify_native_fixed_binding(dataset, group, identity)
    else:
        _verify_overlay_binding(dataset, group, identity)
    clock = group.clock
    closes = group.sources[0].raw.timestamps
    starts, ends = clock.interval_bounds(closes)
    # MarketDataset admits exact coarser datetime64 units and preserves
    # their dtype. Its constructor already checked lossless ns conversion.
    fine = dataset.timestamps.astype("datetime64[ns]").view(np.int64)
    first = int(fine[0])
    last = int(fine[-1])
    step = int(fine[1]) - first
    for start, end in zip(starts.view(np.int64), ends.view(np.int64), strict=True):
        lo, hi = int(start), int(end)
        if hi - lo <= step or (
            not native_fixed and ((lo - first) % step or (hi - first) % step)
        ):
            raise ValueError(
                "native decision clock must be larger and, for overlays, aligned to fine grid"
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
    return fine


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
    Native fixed spot builds also admit independently pinned larger clocks
    whose boundaries need not coincide with the execution grid. Legacy
    overlays retain their aligned-clock admission.
    Only raw publication is gated; the current as-of feature masks, warm-up
    and staleness still belong to the overlay and the policy.
    """
    fine = _verify_group(dataset, group)
    if (
        type(start_index) is not int
        or type(stop_index) is not int
        or not 0 <= start_index < stop_index < dataset.n_bars
    ):
        raise ValueError("invalid native decision window")
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


def _require_linear_carry_data(
    dataset: MarketDataset, group: ClockedNativeFeatureGroup, span: slice
) -> None:
    """Admit explicit, close-assigned single funding events on the fine grid.

    A rate-price product binds the declared economic input, not the venue
    event's time or independent mark provenance. Multiple events in one
    execution interval cannot be modeled from this contract.
    """
    if dataset.funding_price_rate is None:
        raise ValueError("linear carry requires explicit fine funding price-rate data")
    count = dataset.resolved_array("funding_event_count")[span]
    due = dataset.resolved_array("funding_due")[span]
    rate = dataset.funding_rate[span]
    price_rate = dataset.funding_price_rate[span]
    if (
        np.any(count > 1)
        or np.any(due != (count == 1))
        or not np.isfinite(rate).all()
        or np.any(~due & ((rate != 0.0) | (price_rate != 0.0)))
        or np.any((rate == 0.0) != (price_rate == 0.0))
    ):
        raise ValueError("linear carry requires consistent single-event funding rows")
    nonzero = due & (rate != 0.0)
    # A nonzero single-event rate requires a representable positive event
    # price. Do not substitute the close or try to reconstruct event timing.
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        implied_price = price_rate[nonzero] / rate[nonzero]
    if not np.isfinite(implied_price).all() or np.any(implied_price <= 0.0):
        raise ValueError("linear carry requires a finite positive funding event price")
    if tuple(
        source.contract.volume_unit for source in group.sources
    ) != dataset.volume_units or not np.array_equal(
        dataset.resolved_array("contract_multipliers"),
        [source.contract.contract_multiplier for source in group.sources],
    ):
        raise ValueError(
            "linear carry requires matching fine and coarse instrument units"
        )
    if np.any(dataset.resolved_array("split_factor")[span] != 1.0) or np.any(
        dataset.resolved_array("dividend")[span] != 0.0
    ):
        raise ValueError("linear carry does not admit corporate actions")


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
    execution_mode: Literal["strict_spot", "linear_carry"] = "strict_spot",
    terminal_reserve_bars: int | None = None,
    continue_pending_orders: bool = False,
) -> ClockedNativeReplayResult:
    """Run sparse actor calls while the canonical engine advances every fine bar.

    The default preserves strict long-only spot cash without funding. Explicit
    linear_carry selects the shared native signed-position/cross-margin and
    carry accounting with at most one close-assigned funding event per symbol
    per fine row, explicit price-rate products and identity-bound borrow
    inputs. Funding applies after the row's open fills, never before them.
    This is a declared discrete linear model, not exchange-specific event
    timing, leverage or provider admission. Both modes require deterministic
    unlevered MARKET and previous-bar capacity. The default refuses additional
    latency and partial-fill carry. Explicit continue_pending_orders=True
    admits cross-margin fine-bar latency and pending MARKET residuals using
    the shared no-new-intent transition; the coarse policy is not recalled.
    Hard risk may override off-schedule exact HOLD or cancel pending exposure;
    the terminal tail attempts flattening. An explicit terminal_reserve_bars
    counts fine execution intervals, extends the native default, and excludes
    those intervals from the coarse actor schedule. It does not guarantee
    settlement or override capacity. Inspect actual settlement.
    Native fixed spot builds retain their spot-only execution restrictions
    and exact rational annualization. Delayed/unavailable fine bars and
    intrabar carry are not admitted.
    This is a software contract, not execution authority.
    """
    mode: object = execution_mode
    if type(mode) is not str or mode not in ("strict_spot", "linear_carry"):
        raise ValueError("invalid clocked native execution_mode")
    strict_spot = mode == "strict_spot"
    if type(continue_pending_orders) is not bool:
        raise ValueError("continue_pending_orders must be boolean")
    if type(execution_cost) is not ExecutionCostConfig or (
        execution_cost.order_type != "market"
        or (
            not continue_pending_orders
            and (
                execution_cost.order_latency_bars != 0
                or execution_cost.partial_fill_carry
            )
        )
        or execution_cost.processing_bar_volume_capacity
        or (strict_spot and execution_cost.allow_short)
        or (
            (not strict_spot or continue_pending_orders)
            and execution_cost.margin_mode != "cross"
        )
        or execution_cost.max_leverage > 1.0
        or execution_cost.collateral_haircut != 1.0
        or execution_cost.slippage_std != 0.0
        or execution_cost.tail_slippage_probability != 0.0
    ):
        raise ValueError(
            "scheduled replay requires deterministic unlevered MARKET; pending carry requires explicit opt-in; "
            "strict spot is long-only and linear carry requires cross margin"
        )
    if type(dataset) is not MarketDataset:
        raise TypeError("expected a MarketDataset")
    if dataset.native_clock is not None and not strict_spot:
        raise ValueError("native fixed scheduled replay requires strict spot")
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
        terminal_reserve_bars=terminal_reserve_bars,
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
    if not strict_spot:
        _require_linear_carry_data(dataset, decision_group, span)
    elif (
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
        terminal_reserve_bars=terminal_reserve_bars,
        strict_spot_cash=strict_spot,
        actor_decision_indices=(
            tuple(
                slot.decision_index
                for slot in slots
                if slot.status == "scheduled" and slot.decision_index is not None
            )
            if continue_pending_orders
            else None
        ),
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
