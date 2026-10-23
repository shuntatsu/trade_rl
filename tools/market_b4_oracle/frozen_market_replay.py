"""Owned fixed-profile MARKET/GTC replay for bounded B4 comparisons.

The driver consumes caller-pinned Dataset and forecast inputs, owns its account
and order ledger, and derives fills from raw market rows. It is a software
comparison component only: callers still own source provenance, outer-process
authentication, and research authorization.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Mapping, TypedDict, cast

import numpy as np

from tools.market_b4_oracle.allocation_choice import AllocationAssumptions
from tools.market_b4_oracle.binary64_account import (
    AccountSnapshot,
    BorrowEvent,
    CashInterestEvent,
    DividendEvent,
    DrawdownObservation,
    ExecutionEvent,
    FundingMarkEvent,
    MarginEvent,
    OwnedBinary64Account,
    RetainedTerminationEvent,
    RevaluationEvent,
)
from tools.market_b4_oracle.decision_identity import (
    ActorIdentity,
    ActorIntentBasis,
    OwnedAccountComponents,
    OwnedIntentComponents,
    OwnedPendingOrder,
    _actor_intent_basis_from_identity,
    _build_actor_identity_core,
    build_actor_identity,
    canonical_json_bytes,
)
from tools.market_b4_oracle.expected_intent import (
    ExpectedIntent,
    ExpectedPending,
    exact_quantity,
    project_quantity,
    reconcile_expected,
)
from tools.market_b4_oracle.frozen_decision import compose_frozen_allocation_target
from tools.market_b4_oracle.frozen_forecast import OwnedDecisionSnapshot
from tools.market_b4_oracle.frozen_forecast_index import FrozenForecastIndex
from tools.market_replay_oracle import DeclaredProtocolPair, VerifiedDatasetContent

_TOLERANCE = 1e-12
_QUANTITY_TOLERANCE = 1e-12
_QUANTITY_ULPS = 32
_SYMBOL_COUNT = 2
_NS_PER_HOUR = 3_600_000_000_000
_COST_IMPACT_RATE = 0.0001
_COST_MULTIPLIER = 2.0
_COST_MAX_PARTICIPATION = 0.05
_COST_TRIGGER_OPEN_FRACTION = 1.0
_FROZEN_SCENARIOS = {
    "BASEL0": (1.0, 0),
    "COST2L0": (2.0, 0),
    "BASEL2": (1.0, 2),
}
_FROZEN_RISK = {
    "drawdown_start": 0.1,
    "drawdown_stop": 0.2,
    "emergency_turnover_override": True,
    "fail_closed_tolerance": 1e-10,
    "max_abs_weight": 1.0,
    "max_gross": 1.0,
    "max_turnover": None,
}
_ACTIVE_STATUSES = frozenset(
    {"submitted", "latency_wait", "eligible", "partially_filled"}
)
_TERMINAL_STATUSES = frozenset({"filled", "cancelled", "expired", "rejected"})
_TERMINAL_IDENTITY_CACHE_MAX_ORDER_ENTRIES = 8192
_TERMINAL_IDENTITY_CACHE_MAX_MAPPING_BYTES = 16 * 1024 * 1024
_SHA = frozenset("0123456789abcdef")
_RISK_KEYS = frozenset(
    {
        "drawdown_start",
        "drawdown_stop",
        "emergency_turnover_override",
        "fail_closed_tolerance",
        "max_abs_weight",
        "max_gross",
        "max_turnover",
    }
)
_POLICY_KEYS = frozenset(
    {
        "allow_short",
        "borrow_rate_multiplier",
        "collateral_haircut",
        "fee_rate",
        "impact_rate",
        "limit_offset_rate",
        "lot_size",
        "maintenance_margin_rate",
        "maker_fee_rate",
        "margin_mode",
        "max_leverage",
        "max_participation_rate",
        "minimum_notional",
        "multiplier",
        "order_latency_bars",
        "order_type",
        "partial_fill_carry",
        "path_mode",
        "processing_bar_volume_capacity",
        "random_seed",
        "schema_version",
        "slippage_std",
        "spread_rate",
        "tail_slippage_multiplier",
        "tail_slippage_probability",
        "taker_fee_rate",
        "tick_size",
        "trigger_volume_fractions",
    }
)


class _ExecutionEconomics(TypedDict):
    fee_rate: float
    maker_fee_rate: float
    taker_fee_rate: float
    spread_rate: float
    max_participation_rate: float
    borrow_available: bool
    borrow_rate: float


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite non-boolean number")
    result = float(cast(int | float, value))
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} is outside its finite range")
    return result


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be an ordinary integer in range")
    return value


def _digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _sha_matches(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA for character in value)
    )


def _quantity_completion_tolerance(*values: float) -> float:
    """Match PendingOrder's source tolerance for binary64 lot arithmetic."""
    return max(
        _QUANTITY_TOLERANCE,
        *(
            _QUANTITY_ULPS * math.ulp(abs(float(value)))
            for value in values
            if math.isfinite(float(value))
        ),
    )


def _transition_order(order: _OwnedOrder, **changes: Any) -> _OwnedOrder:
    """Apply one source-equivalent order state transition and evidence tick."""
    return replace(order, evidence_version=order.evidence_version + 1, **changes)


def _apply_owned_fill_progress(
    order: _OwnedOrder,
    *,
    exact_fill: Fraction,
    filled_notional: float,
    processing_index: int,
) -> _OwnedOrder:
    """Record a native-shaped fill transition, including tolerance completion."""
    exact_cumulative = order.exact_filled_quantity + exact_fill
    exact_remaining = exact_quantity(order.intent.requested_quantity) - exact_cumulative
    cumulative = project_quantity(exact_cumulative)
    remaining = project_quantity(exact_remaining)
    tolerance = _quantity_completion_tolerance(
        order.intent.requested_quantity,
        cumulative,
        remaining,
    )
    if abs(remaining) <= tolerance:
        exact_cumulative = exact_quantity(order.intent.requested_quantity)
        status = "filled"
        terminal_reason = "filled"
    else:
        status = "partially_filled"
        terminal_reason = None
    return _transition_order(
        order,
        exact_filled_quantity=exact_cumulative,
        cumulative_filled_notional=order.cumulative_filled_notional + filled_notional,
        status=status,
        last_processed_index=processing_index,
        terminal_reason=terminal_reason,
    )


def _native_json_sha256(value: object) -> str:
    """Match tier_d_protocol.json_sha256 for the original H/v3 declaration."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("protocol JSON contains a duplicate key")
        result[key] = value
    return result


def _json_reject_constant(value: str) -> None:
    raise ValueError(f"protocol JSON contains nonfinite constant {value!r}")


def _json_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_json_unique_pairs,
            parse_constant=_json_reject_constant,
        )
        if not isinstance(value, dict):
            raise ValueError("protocol JSON must be an object")
        canonical_json_bytes(value)
        return value
    except (RecursionError, UnicodeError) as error:
        raise ValueError("protocol JSON is invalid or too deeply nested") from error


def _effective_profile(
    scenario: str, actor_advances: int, risk: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive only the three source-frozen MARKET/GTC profile variants."""
    if scenario not in _FROZEN_SCENARIOS:
        raise ValueError("unsupported frozen market scenario")
    if type(actor_advances) is not int or actor_advances <= 0:
        raise ValueError("actor advances must be a positive integer")
    if canonical_json_bytes(risk) != canonical_json_bytes(_FROZEN_RISK):
        raise ValueError("risk assumptions differ from the frozen protocol")
    multiplier, latency = _FROZEN_SCENARIOS[scenario]
    policy = {
        "allow_short": True,
        "borrow_rate_multiplier": 1.0,
        "collateral_haircut": 1.0,
        "fee_rate": 0.0,
        "impact_rate": _COST_IMPACT_RATE,
        "limit_offset_rate": 0.0005,
        "lot_size": 0.0,
        "maintenance_margin_rate": 0.25,
        "maker_fee_rate": 0.0,
        "margin_mode": "cross",
        "max_leverage": 1.0,
        "max_participation_rate": _COST_MAX_PARTICIPATION,
        "minimum_notional": 0.0,
        "multiplier": multiplier,
        "order_latency_bars": latency,
        "order_type": "market",
        "partial_fill_carry": True,
        "path_mode": "conservative",
        "processing_bar_volume_capacity": False,
        "random_seed": 0,
        "schema_version": "execution_policy_v2",
        "slippage_std": 0.0,
        "spread_rate": 0.0,
        "tail_slippage_multiplier": 5.0,
        "tail_slippage_probability": 0.0,
        "taker_fee_rate": 0.0,
        "tick_size": 0.0,
        "trigger_volume_fractions": [1.0, 0.5, 0.25, 0.0],
    }
    cost_digest = _native_json_sha256(policy)
    execution_digest = _native_json_sha256(
        {
            "schema_version": "insolvency_execution_policy_v1",
            "base_policy_digest": cost_digest,
            "insolvency_valuation": "retain_debt",
        }
    )
    return {
        "cost_policy": policy,
        "cost_policy_digest": cost_digest,
        "execution_policy_digest": execution_digest,
        "market_order_profile": None,
        "rule_stress": {
            "adverse_tick_rounding": False,
            "lot_size_factor": 1.0,
            "minimum_notional_factor": 1.0,
            "name": "nominal",
            "schema_version": "execution_rule_stress_v1",
            "tick_size_factor": 1.0,
        },
        "runtime_profile": {
            "schema": "allocation_runtime_profile_v1",
            "economics_digest": execution_digest,
            "risk_digest": _native_json_sha256(risk),
            "initial_capital": 100_000.0,
            "currency": "USDT",
            "decision_interval_seconds": 3600,
            "economic_horizon_seconds": actor_advances * 3600,
            "insolvency_valuation": "retain_debt",
            "calendar_kind": "continuous_24_7",
            "execution_bar_hours": 1.0,
        },
    }


def _protocol_window(
    document: Mapping[str, Any], content: VerifiedDatasetContent
) -> FrozenMarketWindow:
    actor = document.get("actor")
    terminal = document.get("terminal")
    timestamps = content.arrays.get("timestamps")
    if (
        not isinstance(actor, dict)
        or not isinstance(terminal, dict)
        or not isinstance(timestamps, np.ndarray)
        or timestamps.ndim != 1
        or timestamps.dtype.kind != "M"
    ):
        raise ValueError("frozen protocol clock is unavailable")
    start = _integer(actor.get("start"), "actor.start", minimum=0)
    stop = _integer(actor.get("stop"), "actor.stop", minimum=1)
    actor_rows = _integer(actor.get("rows"), "actor.rows", minimum=1)
    reserve_stop = _integer(
        terminal.get("stop_inclusive"), "terminal.stop_inclusive", minimum=1
    )
    reserve_advances = _integer(
        terminal.get("advances"), "terminal.advances", minimum=1
    )
    if (
        not start < stop <= reserve_stop < timestamps.size
        or stop - start != actor_rows
        or reserve_stop != stop + reserve_advances
        or reserve_stop + 1 != timestamps.size - 1
        or np.datetime_as_string(timestamps[start], unit="s") != actor.get("start_utc")
        or np.datetime_as_string(timestamps[stop], unit="s") != actor.get("stop_utc")
        or np.datetime_as_string(timestamps[reserve_stop], unit="s")
        != terminal.get("utc")
    ):
        raise ValueError("frozen actor or terminal window differs from Dataset clock")
    vintage_starts = actor.get("vintage_starts")
    vintage_times = actor.get("vintage_start_utc")
    if (
        type(vintage_starts) is not list
        or type(vintage_times) is not list
        or len(vintage_starts) != 3
        or len(vintage_times) != 3
    ):
        raise ValueError("frozen protocol requires its three-vintage schedule")
    for index, declared_time in zip(vintage_starts, vintage_times, strict=True):
        vintage_index = _integer(index, "actor.vintage_start", minimum=0)
        if (
            vintage_index < start
            or vintage_index >= stop
            or np.datetime_as_string(timestamps[vintage_index], unit="s")
            != declared_time
        ):
            raise ValueError("frozen vintage boundary differs from Dataset clock")
    return FrozenMarketWindow(start, stop, reserve_stop)


def _two(value: object, name: str) -> tuple[Any, Any]:
    if type(value) not in (tuple, list, np.ndarray):
        raise ValueError(f"{name} must preserve the two-symbol account shape")
    sequence = cast(tuple[Any, Any] | list[Any] | np.ndarray, value)
    if len(sequence) != _SYMBOL_COUNT or (
        isinstance(sequence, np.ndarray) and sequence.ndim != 1
    ):
        raise ValueError(f"{name} must preserve the two-symbol account shape")
    return sequence[0], sequence[1]


def _pair_numbers(
    value: object, name: str, *, minimum: float | None = None
) -> tuple[float, float]:
    first, second = _two(value, name)
    return (
        _finite(first, f"{name}[0]", minimum=minimum),
        _finite(second, f"{name}[1]", minimum=minimum),
    )


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if type(value) not in (tuple, list):
        raise ValueError(f"{name} must be an immutable sequence of strings")
    values = cast(tuple[object, ...] | list[object], value)
    if any(type(item) is not str or not item.strip() for item in values):
        raise ValueError(f"{name} must contain only nonempty strings")
    return tuple(cast(str, item) for item in values)


@dataclass(frozen=True, slots=True)
class FrozenMarketPolicy:
    """Closed MARKET/GTC execution and risk assumptions for one comparison arm."""

    dataset_id: str
    execution_policy_digest: str
    order_latency_bars: int
    fee_rate: float
    taker_fee_rate: float
    spread_rate: float
    impact_rate: float
    multiplier: float
    max_participation_rate: float
    allow_short: bool
    max_leverage: float
    maintenance_margin_rate: float
    collateral_haircut: float
    borrow_rate_multiplier: float
    maximum_gross: float
    drawdown_start: float
    drawdown_stop: float
    insolvency_valuation: str
    margin_mode: str
    tick_size_floor: float
    lot_size_floor: float
    minimum_notional_floor: float
    tick_size_factor: float
    lot_size_factor: float
    minimum_notional_factor: float
    adverse_tick_rounding: bool

    def __post_init__(self) -> None:
        _digest(self.dataset_id, "dataset_id")
        _digest(self.execution_policy_digest, "execution_policy_digest")
        if type(self.order_latency_bars) is not int or self.order_latency_bars < 0:
            raise ValueError("order_latency_bars must be a non-negative integer")
        for name in (
            "fee_rate",
            "taker_fee_rate",
            "spread_rate",
            "impact_rate",
            "tick_size_floor",
            "lot_size_floor",
            "minimum_notional_floor",
        ):
            if _finite(getattr(self, name), name, minimum=0.0) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        multiplier = _finite(self.multiplier, "multiplier", minimum=0.0)
        if multiplier not in {1.0, 2.0}:
            raise ValueError("multiplier is outside the frozen scenario allowlist")
        if self.order_latency_bars not in {0, 2}:
            raise ValueError("latency is outside the frozen scenario allowlist")
        if (
            not 0.0
            < _finite(self.max_participation_rate, "max_participation_rate")
            <= 1.0
        ):
            raise ValueError("max_participation_rate must be in (0, 1]")
        if type(self.allow_short) is not bool:
            raise ValueError("allow_short must be boolean")
        if _finite(self.max_leverage, "max_leverage", minimum=0.0) <= 0.0:
            raise ValueError("max_leverage must be positive")
        for name in ("maintenance_margin_rate", "collateral_haircut"):
            if not 0.0 <= _finite(getattr(self, name), name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if (
            _finite(self.borrow_rate_multiplier, "borrow_rate_multiplier", minimum=0.0)
            < 0.0
        ):
            raise ValueError("borrow_rate_multiplier must be non-negative")
        if _finite(self.maximum_gross, "maximum_gross", minimum=0.0) <= 0.0:
            raise ValueError("maximum_gross must be positive")
        start = _finite(self.drawdown_start, "drawdown_start", minimum=0.0)
        stop = _finite(self.drawdown_stop, "drawdown_stop", minimum=0.0)
        if start >= stop or stop > 1.0:
            raise ValueError("drawdown bounds must be ordered within [0, 1]")
        if self.insolvency_valuation != "retain_debt" or self.margin_mode != "cross":
            raise ValueError("only cross margin with retain_debt is supported")
        for name in ("tick_size_factor", "lot_size_factor", "minimum_notional_factor"):
            if _finite(getattr(self, name), name, minimum=1.0) < 1.0:
                raise ValueError(f"{name} must be at least one")
        if type(self.adverse_tick_rounding) is not bool:
            raise ValueError("adverse_tick_rounding must be boolean")
        if self.adverse_tick_rounding:
            raise ValueError(
                "this fixed MARKET/GTC profile requires nominal tick rounding"
            )
        if (
            self.fee_rate != 0.0
            or self.taker_fee_rate != 0.0
            or self.spread_rate != 0.0
            or self.impact_rate != _COST_IMPACT_RATE
            or self.max_participation_rate != _COST_MAX_PARTICIPATION
            or self.borrow_rate_multiplier != 1.0
        ):
            raise ValueError("policy is outside the frozen market cost recipe")

    @classmethod
    def from_effective_profile(
        cls,
        *,
        dataset_id: str,
        profile: Mapping[str, object],
        risk: Mapping[str, object],
    ) -> FrozenMarketPolicy:
        """Bind the closed protocol profile shape without claiming provenance."""
        if type(profile) is not dict or type(risk) is not dict:
            raise ValueError("profile and risk must be parsed dictionaries")
        cost = profile.get("cost_policy")
        stress = profile.get("rule_stress")
        if type(cost) is not dict or set(cost) != _POLICY_KEYS:
            raise ValueError("execution policy fields are not closed")
        if type(stress) is not dict or set(stress) != {
            "adverse_tick_rounding",
            "lot_size_factor",
            "minimum_notional_factor",
            "name",
            "schema_version",
            "tick_size_factor",
        }:
            raise ValueError("execution rule stress fields are not closed")
        if set(risk) != _RISK_KEYS:
            raise ValueError("risk profile fields are not closed")
        if (
            profile.get("market_order_profile") is not None
            or cost["order_type"] != "market"
            or cost["slippage_std"] != 0.0
            or cost["tail_slippage_probability"] != 0.0
            or cost["margin_mode"] != "cross"
            or cost["path_mode"] != "conservative"
            or cost["partial_fill_carry"] is not True
            or cost["processing_bar_volume_capacity"] is not False
            or stress["name"] != "nominal"
            or stress["adverse_tick_rounding"] is not False
        ):
            raise ValueError("profile is outside the fixed MARKET/GTC scope")
        runtime = profile.get("runtime_profile")
        if (
            type(runtime) is not dict
            or runtime.get("insolvency_valuation") != "retain_debt"
        ):
            raise ValueError("runtime profile must retain debt on insolvency")
        return cls(
            dataset_id=dataset_id,
            execution_policy_digest=_digest(
                profile.get("execution_policy_digest"), "execution_policy_digest"
            ),
            order_latency_bars=cost["order_latency_bars"],
            fee_rate=cost["fee_rate"],
            taker_fee_rate=cost["taker_fee_rate"],
            spread_rate=cost["spread_rate"],
            impact_rate=cost["impact_rate"],
            multiplier=cost["multiplier"],
            max_participation_rate=cost["max_participation_rate"],
            allow_short=cost["allow_short"],
            max_leverage=cost["max_leverage"],
            maintenance_margin_rate=cost["maintenance_margin_rate"],
            collateral_haircut=cost["collateral_haircut"],
            borrow_rate_multiplier=cost["borrow_rate_multiplier"],
            maximum_gross=risk["max_gross"],
            drawdown_start=risk["drawdown_start"],
            drawdown_stop=risk["drawdown_stop"],
            insolvency_valuation="retain_debt",
            margin_mode="cross",
            tick_size_floor=cost["tick_size"],
            lot_size_floor=cost["lot_size"],
            minimum_notional_floor=cost["minimum_notional"],
            tick_size_factor=stress["tick_size_factor"],
            lot_size_factor=stress["lot_size_factor"],
            minimum_notional_factor=stress["minimum_notional_factor"],
            adverse_tick_rounding=stress["adverse_tick_rounding"],
        )


@dataclass(frozen=True, slots=True)
class FrozenMarketWindow:
    """Decision start and inclusive processing stops from a fixed binding."""

    actor_start: int
    actor_stop: int
    terminal_stop: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int
            for value in (self.actor_start, self.actor_stop, self.terminal_stop)
        ):
            raise ValueError("window indexes must be ordinary integers")
        if not 0 <= self.actor_start < self.actor_stop < self.terminal_stop:
            raise ValueError("actor and reserve windows are not strictly ordered")

    @classmethod
    def from_binding_payload(cls, payload: Mapping[str, object]) -> FrozenMarketWindow:
        clock = payload.get("clock")
        if type(clock) is not dict:
            raise ValueError("binding has no closed clock object")
        return cls(clock["actor_start"], clock["actor_stop"], clock["settlement_stop"])

    @property
    def actor_advances(self) -> int:
        return self.actor_stop - self.actor_start

    @property
    def reserve_advances(self) -> int:
        return self.terminal_stop - self.actor_stop


@dataclass(frozen=True, slots=True)
class FrozenMarketBinding:
    """One selected arm derived from the frozen native protocol roster.

    A validated H/v3 declaration supplies account, scenario, role, horizon and
    row boundaries. This static binding does not authenticate who supplied the
    protocol or Dataset. Manufactured fixtures use a separate explicit scope.
    """

    protocol_canonical: bytes
    account_id: str
    configuration_id: str
    scenario: str
    native_role: str
    selected_symbol_index: int
    expected_horizon_seconds: int
    observation_feature_indices: tuple[int, ...]
    window: FrozenMarketWindow
    protocol_pair: DeclaredProtocolPair | None = None
    comparison_only: bool = False

    def __post_init__(self) -> None:
        if type(self.protocol_canonical) is not bytes:
            raise ValueError("binding requires canonical protocol bytes")
        document = _json_object(self.protocol_canonical)
        if canonical_json_bytes(document) != self.protocol_canonical:
            raise ValueError("binding protocol bytes are not canonical")
        if type(self.comparison_only) is not bool:
            raise ValueError("comparison_only must be boolean")
        if self.comparison_only:
            if (
                self.protocol_pair is not None
                or document.get("scope") != "ARTIFICIAL_COMPARISON_ONLY"
                or not (
                    self.configuration_id.startswith("TEST_")
                    or (
                        document.get("schema")
                        == "issue810_artificial_h_component_q_binding_v1"
                        and self.configuration_id == "H24_RIDGE_A1"
                        and _sha_matches(document.get("component_spec_sha256"))
                        and _sha_matches(document.get("source_protocol_sha256"))
                    )
                )
            ):
                raise ValueError("synthetic binding must remain explicitly artificial")
        else:
            if type(self.protocol_pair) is not DeclaredProtocolPair:
                raise ValueError("native binding requires a validated protocol pair")
            self.protocol_pair.validate()
            if (
                self.protocol_canonical != self.protocol_pair.native_canonical
                or document.get("schema") != "issue810_tier_d_nonrl48_result_blind_v3"
                or document.get("scope") != "CONSUMED_2025_DEVELOPMENT_NONRL48_ONLY"
            ):
                raise ValueError("native binding differs from the frozen H/v3 protocol")
            if self.scenario not in _FROZEN_SCENARIOS:
                raise ValueError("native scenario is outside the frozen allowlist")

        if (
            type(self.selected_symbol_index) is not int
            or self.selected_symbol_index not in (0, 1)
            or type(self.expected_horizon_seconds) is not int
            or self.expected_horizon_seconds <= 0
            or type(self.observation_feature_indices) is not tuple
            or type(self.window) is not FrozenMarketWindow
        ):
            raise ValueError("binding selection fields are malformed")
        dataset_document = document.get("dataset", {})
        is_artificial_component = (
            document.get("schema") == "issue810_artificial_h_component_q_binding_v1"
        )
        symbols = (
            dataset_document.get("symbols")
            if is_artificial_component
            else dataset_document.get("fit_symbols")
        )
        fit_symbols = dataset_document.get("fit_symbols")
        prediction_symbols = dataset_document.get("prediction_symbols")
        feature_names = document.get("dataset", {}).get("feature_names")
        configurations = document.get("configurations")
        accounts = document.get("accounts")
        scenarios = document.get("scenarios")
        roles = document.get("roles")
        if (
            type(symbols) is not list
            or len(symbols) != 2
            or any(type(symbol) is not str for symbol in symbols)
            or type(feature_names) is not list
            or not feature_names
            or type(configurations) is not list
            or type(accounts) is not list
            or type(scenarios) is not list
            or type(roles) is not list
        ):
            raise ValueError("protocol arm roster is incomplete")
        if is_artificial_component and (
            fit_symbols != ["BTCUSDT"]
            or prediction_symbols != ["BTCUSDT", "ETHUSDT"]
            or symbols != prediction_symbols
        ):
            raise ValueError(
                "artificial fit-source and prediction rosters differ from fixed scope"
            )
        matches = [
            item
            for item in configurations
            if type(item) is dict and item.get("id") == self.configuration_id
        ]
        if len(matches) != 1:
            raise ValueError("configuration is absent or duplicated in protocol")
        configuration = matches[0]
        symbol = symbols[self.selected_symbol_index]
        expected_account = f"{self.configuration_id}_{symbol}"
        if (
            self.account_id != expected_account
            or self.account_id not in accounts
            or self.scenario not in scenarios
            or self.native_role not in roles
            or self.native_role not in {"nonrl", "cash"}
        ):
            raise ValueError("selected account/scenario/role is outside frozen roster")
        horizon_hours = _integer(
            configuration.get("horizon_hours"), "configuration.horizon_hours", minimum=1
        )
        if (
            self.expected_horizon_seconds != horizon_hours * 3600
            or self.observation_feature_indices != tuple(range(len(feature_names)))
        ):
            raise ValueError("binding horizon or feature roster differs from protocol")
        actor = document.get("actor")
        terminal = document.get("terminal")
        if (
            type(actor) is not dict
            or type(terminal) is not dict
            or self.window
            != FrozenMarketWindow(
                actor.get("start", -1),
                actor.get("stop", -1),
                terminal.get("stop_inclusive", -1),
            )
        ):
            raise ValueError("binding row window differs from protocol declaration")
        policy = document.get("policy")
        risk = policy.get("risk") if type(policy) is dict else None
        if type(risk) is not dict:
            raise ValueError("protocol risk declaration is unavailable")
        try:
            expected_profile = _effective_profile(
                self.scenario, self.window.actor_advances, cast(dict[str, Any], risk)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("native policy/risk declaration is unsupported") from error
        expected_multiplier, expected_latency = _FROZEN_SCENARIOS[self.scenario]
        if (
            expected_profile["cost_policy"]["multiplier"] != expected_multiplier
            or expected_profile["cost_policy"]["order_latency_bars"] != expected_latency
            or expected_profile["cost_policy"]["impact_rate"] != _COST_IMPACT_RATE
            or expected_profile["cost_policy"]["max_participation_rate"]
            != _COST_MAX_PARTICIPATION
        ):
            raise ValueError("selected protocol profile differs from frozen scenario")

    @classmethod
    def from_declared_protocol_pair(
        cls,
        *,
        protocol_pair: DeclaredProtocolPair,
        content: VerifiedDatasetContent,
        configuration_id: str,
        scenario: str,
        native_role: str,
        selected_symbol_index: int,
    ) -> FrozenMarketBinding:
        """Select exactly one account/configuration/scenario/role roster row."""
        if type(protocol_pair) is not DeclaredProtocolPair:
            raise ValueError("requires the exact declared protocol-pair type")
        protocol_pair.validate()
        if type(content) is not VerifiedDatasetContent:
            raise ValueError("arm selection requires caller-verified Dataset content")
        document = _json_object(protocol_pair.native_canonical)
        dataset = document.get("dataset")
        configurations = document.get("configurations")
        accounts = document.get("accounts")
        scenarios = document.get("scenarios")
        roles = document.get("roles")
        if (
            type(dataset) is not dict
            or type(configurations) is not list
            or type(accounts) is not list
            or type(scenarios) is not list
            or type(roles) is not list
            or type(selected_symbol_index) is not int
            or selected_symbol_index not in (0, 1)
        ):
            raise ValueError("frozen protocol arm roster is malformed")
        symbols = _string_tuple(dataset.get("fit_symbols"), "protocol fit symbols")
        features = _string_tuple(dataset.get("feature_names"), "protocol feature names")
        if (
            content.dataset_id != dataset.get("dataset_id")
            or _string_tuple(content.dataset.get("symbols"), "Dataset symbols")
            != symbols
            or _string_tuple(content.dataset.get("feature_names"), "Dataset features")
            != features
        ):
            raise ValueError("Dataset identity differs from the frozen protocol")
        configuration_rows = [
            row
            for row in configurations
            if type(row) is dict and row.get("id") == configuration_id
        ]
        if (
            len(configuration_rows) != 1
            or scenario not in scenarios
            or native_role not in roles
            or native_role not in {"nonrl", "cash"}
        ):
            raise ValueError("selection does not resolve to one frozen arm")
        configuration = configuration_rows[0]
        account_id = f"{configuration_id}_{symbols[selected_symbol_index]}"
        if account_id not in accounts:
            raise ValueError("selected account is outside the frozen roster")
        horizon_seconds = (
            _integer(
                configuration.get("horizon_hours"),
                "configuration.horizon_hours",
                minimum=1,
            )
            * 3600
        )
        window = _protocol_window(document, content)
        return cls(
            protocol_canonical=protocol_pair.native_canonical,
            account_id=account_id,
            configuration_id=configuration_id,
            scenario=scenario,
            native_role=native_role,
            selected_symbol_index=selected_symbol_index,
            expected_horizon_seconds=horizon_seconds,
            observation_feature_indices=tuple(
                range(len(document["dataset"]["feature_names"]))
            ),
            window=window,
            protocol_pair=protocol_pair,
        )

    @classmethod
    def _comparison_fixture(
        cls,
        *,
        content: VerifiedDatasetContent,
        window: FrozenMarketWindow,
        selected_symbol_index: int,
        expected_horizon_seconds: int,
        observation_feature_indices: tuple[int, ...],
        native_role: str,
        initial_capital: float,
        scenario: str = "COST2L0",
    ) -> FrozenMarketBinding:
        """Manufacture a clearly artificial binding for small software tests."""
        symbols = _string_tuple(content.dataset.get("symbols"), "Dataset symbols")
        features = _string_tuple(
            content.dataset.get("feature_names"), "Dataset feature names"
        )
        if selected_symbol_index not in (0, 1) or expected_horizon_seconds % 3600:
            raise ValueError("synthetic binding requires a two-slot whole-hour arm")
        if scenario not in _FROZEN_SCENARIOS:
            raise ValueError("synthetic scenario is outside the frozen allowlist")
        horizon_hours = expected_horizon_seconds // 3600
        configuration_id = f"TEST_H{horizon_hours}_RIDGE_A1"
        account_id = f"{configuration_id}_{symbols[selected_symbol_index]}"
        role = native_role.lower()
        protocol = {
            "schema": "artificial_frozen_market_test_v1",
            "scope": "ARTIFICIAL_COMPARISON_ONLY",
            "dataset": {
                "dataset_id": content.dataset_id,
                "fit_symbols": list(symbols),
                "feature_names": list(features),
            },
            "configurations": [
                {
                    "id": configuration_id,
                    "horizon_hours": horizon_hours,
                    "ridge_alpha": 1.0,
                }
            ],
            "accounts": [account_id],
            "scenarios": [scenario],
            "roles": [role],
            "economics": {"reference_notional": 100_000.0},
            "funding": {
                "coverage_bounds": {
                    "lookback_hours": 24,
                    "minimum_counted_events": 2,
                    "maximum_counted_close_gap_hours": 9,
                    "maximum_counted_close_age_hours": 8,
                }
            },
            "actor": {"start": window.actor_start, "stop": window.actor_stop},
            "terminal": {"stop_inclusive": window.terminal_stop},
            "policy": {
                "risk": {
                    "max_gross": 1.0,
                    "max_abs_weight": 1.0,
                    "max_turnover": None,
                    "drawdown_start": 0.1,
                    "drawdown_stop": 0.2,
                    "emergency_turnover_override": True,
                    "fail_closed_tolerance": 1e-10,
                }
            },
            "capital": {"per_account": _finite(initial_capital, "initial_capital")},
        }
        return cls(
            protocol_canonical=canonical_json_bytes(protocol),
            account_id=account_id,
            configuration_id=configuration_id,
            scenario=scenario,
            native_role=role,
            selected_symbol_index=selected_symbol_index,
            expected_horizon_seconds=expected_horizon_seconds,
            observation_feature_indices=observation_feature_indices,
            window=window,
            comparison_only=True,
        )

    @property
    def protocol_document(self) -> dict[str, Any]:
        return _json_object(self.protocol_canonical)

    @property
    def initial_capital(self) -> float:
        return _finite(
            self.protocol_document["capital"]["per_account"], "capital.per_account"
        )

    @property
    def assumptions_identity(self) -> str:
        document = self.protocol_document
        if document.get("schema") == "issue810_artificial_h_component_q_binding_v1":
            return _digest(
                document.get("source_protocol_sha256"),
                "source protocol identity",
            )
        return _native_json_sha256(document)

    @property
    def effective_profile(self) -> dict[str, Any]:
        document = self.protocol_document
        return _effective_profile(
            self.scenario,
            self.window.actor_advances,
            document["policy"]["risk"],
        )

    def assert_content(self, content: VerifiedDatasetContent) -> None:
        document = self.protocol_document
        dataset = document["dataset"]
        expected_symbols = (
            dataset.get("symbols")
            if document.get("schema") == "issue810_artificial_h_component_q_binding_v1"
            else dataset.get("fit_symbols")
        )
        if (
            content.dataset_id != dataset["dataset_id"]
            or _string_tuple(content.dataset.get("symbols"), "Dataset symbols")
            != tuple(expected_symbols)
            or _string_tuple(content.dataset.get("feature_names"), "Dataset features")
            != tuple(dataset["feature_names"])
        ):
            raise ValueError(
                "Dataset identity or arm feature roster differs from binding"
            )


@dataclass(frozen=True, slots=True)
class ObservedFill:
    """Observed fill fields used only for equality comparison, never account input."""

    order_id: str
    processing_index: int
    exact_quantity: Fraction
    execution_price: float
    filled_lot_count: int | None = None
    lot_size: float = 0.0

    def __post_init__(self) -> None:
        _digest(self.order_id, "order_id")
        if type(self.processing_index) is not int or self.processing_index < 1:
            raise ValueError("processing_index must be positive")
        if type(self.exact_quantity) is not Fraction or not self.exact_quantity:
            raise ValueError("observed exact_quantity must be a nonzero Fraction")
        if _finite(self.execution_price, "execution_price", minimum=0.0) <= 0.0:
            raise ValueError("execution_price must be positive")
        if self.filled_lot_count is not None and (
            type(self.filled_lot_count) is not int or self.filled_lot_count == 0
        ):
            raise ValueError("filled_lot_count must be a nonzero integer or null")
        if _finite(self.lot_size, "lot_size", minimum=0.0) < 0.0:
            raise ValueError("lot_size must be non-negative")


@dataclass(frozen=True, slots=True)
class OwnedFill:
    order_id: str
    processing_index: int
    exact_quantity: Fraction
    execution_price: float
    filled_notional: float
    cost_amount: float
    participation_rate: float
    filled_lot_count: int | None
    lot_size: float

    def observed_projection(self) -> ObservedFill:
        return ObservedFill(
            self.order_id,
            self.processing_index,
            self.exact_quantity,
            self.execution_price,
            self.filled_lot_count,
            self.lot_size,
        )


@dataclass(frozen=True, slots=True)
class ReplayAdvance:
    decision_index: int
    processing_index: int
    phase: str
    actor_identity: str | None
    target_identity: str | None
    new_intent_ids: tuple[str, ...]
    cancelled_order_ids: tuple[str, ...]
    owned_fills: tuple[OwnedFill, ...]
    scope: str = "OWNED_RAW_MARKET_COMPARISON_ONLY"
    actor_context_state_digest: str | None = None
    actor_proposal_digest: str | None = None
    actor_decision_digest: str | None = None
    actor_action_digest: str | None = None
    actor_context_bytes: bytes | None = None
    actor_proposal_bytes: bytes | None = None
    actor_decision_bytes: bytes | None = None
    actor_action_bytes: bytes | None = None


@dataclass(frozen=True, slots=True)
class _OwnedOrder:
    intent: ExpectedIntent
    actor_identity: ActorIntentBasis | None
    exact_filled_quantity: Fraction = Fraction(0)
    cumulative_filled_notional: float = 0.0
    status: str = "submitted"
    last_processed_index: int | None = None
    terminal_reason: str | None = None
    evidence_version: int = 0

    @property
    def remaining_exact(self) -> Fraction:
        return (
            exact_quantity(self.intent.requested_quantity) - self.exact_filled_quantity
        )

    @property
    def remaining_quantity(self) -> float:
        return project_quantity(self.remaining_exact)

    def active_projection(self) -> ExpectedPending:
        return ExpectedPending.from_intent(
            self.intent,
            exact_filled_quantity=self.exact_filled_quantity,
            status=self.status,
        )


@dataclass(slots=True)
class _TerminalProjectionEntry:
    """One unique order entry with strong direct predecessor/source references."""

    source: _OwnedOrder
    projection: OwnedPendingOrder
    predecessor_id: str | None
    predecessor_source: _OwnedOrder | None
    mapping_bytes: bytes | None = None


class _TerminalIdentityMemo:
    """Bounded per-replay memo for stable terminal order identity projections."""

    __slots__ = (
        "scope",
        "entries",
        "by_projection_id",
        "mapping_bytes",
        "disabled",
    )

    def __init__(self) -> None:
        self.scope: tuple[object, ...] | None = None
        self.entries: dict[str, _TerminalProjectionEntry] = {}
        self.by_projection_id: dict[int, _TerminalProjectionEntry] = {}
        self.mapping_bytes = 0
        self.disabled = False

    def ensure_scope(self, scope: tuple[object, ...]) -> None:
        if self.scope != scope:
            self.clear()
            self.scope = scope

    def clear(self, *, disable: bool = False) -> None:
        self.entries.clear()
        self.by_projection_id.clear()
        self.mapping_bytes = 0
        self.disabled = disable

    def transaction(self) -> _TerminalIdentityMemoTransaction:
        return _TerminalIdentityMemoTransaction(self)


class _TerminalIdentityMemoTransaction:
    """Stage projection and canonical-map entries until identity construction succeeds."""

    __slots__ = ("memo", "projections", "by_projection_id", "mappings")

    def __init__(self, memo: _TerminalIdentityMemo) -> None:
        self.memo = memo
        self.projections: dict[str, _TerminalProjectionEntry] = {}
        self.by_projection_id: dict[int, _TerminalProjectionEntry] = {}
        self.mappings: dict[int, tuple[OwnedPendingOrder, bytes]] = {}

    def projection_for(
        self, order_id: str, source: _OwnedOrder
    ) -> OwnedPendingOrder | None:
        if self.memo.disabled:
            return None
        entry = self.projections.get(order_id) or self.memo.entries.get(order_id)
        if entry is None or entry.source is not source:
            return None
        return entry.projection

    def stage_projection(self, entry: _TerminalProjectionEntry) -> None:
        if self.memo.disabled:
            return
        current = self.projections.get(entry.source.intent.order_id)
        if current is None:
            current = self.memo.entries.get(entry.source.intent.order_id)
        if current is not None:
            if (
                current.source is not entry.source
                or current.projection is not entry.projection
            ):
                return
            return
        if entry.predecessor_id is not None:
            predecessor = self.projections.get(
                entry.predecessor_id
            ) or self.memo.entries.get(entry.predecessor_id)
            if (
                predecessor is None
                or predecessor.source is not entry.predecessor_source
                or entry.projection.intent.replaced_order is not predecessor.projection
            ):
                return
        elif (
            entry.predecessor_source is not None
            or entry.projection.intent.replaced_order is not None
        ):
            return
        self.projections[entry.source.intent.order_id] = entry
        self.by_projection_id[id(entry.projection)] = entry

    def admits(self, order: OwnedPendingOrder) -> bool:
        if self.memo.disabled or type(order) is not OwnedPendingOrder:
            return False
        entry = self.by_projection_id.get(id(order))
        if entry is not None and entry.projection is order:
            return type(order.status) is str and order.status in _TERMINAL_STATUSES
        entry = self.memo.by_projection_id.get(id(order))
        return (
            entry is not None
            and entry.projection is order
            and type(order.status) is str
            and order.status in _TERMINAL_STATUSES
        )

    def mapping_bytes(self, order: OwnedPendingOrder) -> bytes | None:
        if self.memo.disabled:
            return None
        staged = self.mappings.get(id(order))
        if staged is not None and staged[0] is order:
            return staged[1]
        entry = self.memo.by_projection_id.get(id(order))
        if entry is None or entry.projection is not order:
            return None
        return entry.mapping_bytes

    def stage_mapping(self, order: OwnedPendingOrder, payload: bytes) -> None:
        if self.admits(order):
            self.mappings[id(order)] = (order, payload)

    def commit(self) -> bool:
        """Commit only a complete identity; return false on bounded-cache fallback."""
        if self.memo.disabled:
            return True
        staged_bytes = 0
        for order, payload in self.mappings.values():
            entry = self.by_projection_id.get(id(order))
            if entry is None or entry.projection is not order:
                entry = self.memo.by_projection_id.get(id(order))
            if entry is None or entry.projection is not order:
                continue
            if entry.mapping_bytes is None:
                staged_bytes += len(payload)
            elif entry.mapping_bytes != payload:
                self.memo.clear(disable=True)
                return False
        if (
            len(self.memo.entries) + len(self.projections)
            > _TERMINAL_IDENTITY_CACHE_MAX_ORDER_ENTRIES
            or self.memo.mapping_bytes + staged_bytes
            > _TERMINAL_IDENTITY_CACHE_MAX_MAPPING_BYTES
        ):
            self.memo.clear(disable=True)
            return False
        for order_id, entry in self.projections.items():
            self.memo.entries[order_id] = entry
            self.memo.by_projection_id[id(entry.projection)] = entry
        for order, payload in self.mappings.values():
            entry = self.by_projection_id.get(id(order))
            if entry is None or entry.projection is not order:
                entry = self.memo.by_projection_id.get(id(order))
            if entry is None or entry.projection is not order:
                continue
            if entry.mapping_bytes is None:
                self.memo.mapping_bytes += len(payload)
                entry.mapping_bytes = payload
        return True


def _cacheable_terminal_projection(
    source: _OwnedOrder,
    projection: OwnedPendingOrder,
    predecessor: OwnedPendingOrder | None,
) -> bool:
    """Admit only the exact immutable record graph emitted by this replay."""
    if (
        type(source) is not _OwnedOrder
        or type(source.intent) is not ExpectedIntent
        or type(source.actor_identity) is not ActorIntentBasis
        or type(source.actor_identity.target_identity) is not str
        or type(projection) is not OwnedPendingOrder
        or type(projection.intent) is not OwnedIntentComponents
        or type(projection.intent.identity_source) is not ActorIntentBasis
        or projection.intent.identity_source is not source.actor_identity
        or type(source.status) is not str
        or source.status not in _TERMINAL_STATUSES
        or type(source.exact_filled_quantity) is not Fraction
        or type(source.cumulative_filled_notional) is not float
        or (
            source.last_processed_index is not None
            and type(source.last_processed_index) is not int
        )
        or type(source.terminal_reason) is not str
        or not source.terminal_reason
        or type(source.evidence_version) is not int
    ):
        return False
    intent = source.intent
    if (
        any(
            type(value) is not str
            for value in (
                intent.order_id,
                intent.dataset_id,
                intent.target_identity,
                intent.execution_policy_digest,
                intent.order_type,
                intent.time_in_force,
            )
        )
        or any(
            type(value) is not int
            for value in (
                intent.symbol_index,
                intent.submit_index,
                intent.eligible_index,
            )
        )
        or any(
            type(value) is not float
            for value in (
                intent.requested_quantity,
                intent.submission_reference_price,
                intent.decision_equity,
            )
        )
        or (intent.limit_price is not None and type(intent.limit_price) is not float)
        or (intent.stop_price is not None and type(intent.stop_price) is not float)
        or (intent.expiry_index is not None and type(intent.expiry_index) is not int)
        or (
            intent.replaced_order_id is not None
            and type(intent.replaced_order_id) is not str
        )
        or type(intent.reduce_only) is not bool
    ):
        return False
    projected_intent = projection.intent
    return (
        projected_intent.dataset_id == intent.dataset_id
        and projected_intent.execution_policy_digest == intent.execution_policy_digest
        and projected_intent.symbol_index == intent.symbol_index
        and projected_intent.requested_quantity == intent.requested_quantity
        and projected_intent.order_type == intent.order_type
        and projected_intent.time_in_force == intent.time_in_force
        and projected_intent.limit_price == intent.limit_price
        and projected_intent.stop_price == intent.stop_price
        and projected_intent.submit_index == intent.submit_index
        and projected_intent.eligible_index == intent.eligible_index
        and projected_intent.expiry_index == intent.expiry_index
        and projected_intent.submission_reference_price
        == intent.submission_reference_price
        and projected_intent.decision_equity == intent.decision_equity
        and projected_intent.replaced_order is predecessor
        and projected_intent.reduce_only is intent.reduce_only
        and type(projection.remaining_quantity) is float
        and projection.remaining_quantity == source.remaining_quantity
        and type(projection.cumulative_filled_quantity) is float
        and projection.cumulative_filled_quantity
        == project_quantity(source.exact_filled_quantity)
        and type(projection.cumulative_filled_notional) is float
        and projection.cumulative_filled_notional == source.cumulative_filled_notional
        and type(projection.status) is str
        and projection.status == source.status
        and projection.trigger_index is None
        and projection.last_processed_index == source.last_processed_index
        and projection.terminal_reason == source.terminal_reason
        and projection.evidence_version == source.evidence_version
        and projection.exact_cumulative_filled_quantity
        == str(source.exact_filled_quantity)
    )


class FrozenMarketReplay:
    """Own one fixed two-slot account, selected-symbol order, and row sequence."""

    __slots__ = (
        "content",
        "policy",
        "binding",
        "window",
        "selected_symbol_index",
        "initial_capital",
        "expected_horizon_seconds",
        "observation_feature_indices",
        "_account",
        "_active",
        "_terminal",
        "_all_orders",
        "_terminal_identity_memo",
        "_current_index",
        "_last_time_ns",
        "_terminated_early",
        "reserve_plan_digest",
    )

    def __init__(
        self,
        *,
        content: VerifiedDatasetContent,
        policy: FrozenMarketPolicy,
        binding: FrozenMarketBinding,
        reserve_plan_digest: str,
    ) -> None:
        if type(content) is not VerifiedDatasetContent:
            raise ValueError("requires exact caller-verified Dataset content")
        if (
            type(policy) is not FrozenMarketPolicy
            or type(binding) is not FrozenMarketBinding
        ):
            raise ValueError("requires exact frozen policy and protocol binding")
        binding.assert_content(content)
        if content.dataset_id != policy.dataset_id:
            raise ValueError("Dataset content and execution policy identities differ")
        if not binding.comparison_only:
            expected_policy = FrozenMarketPolicy.from_effective_profile(
                dataset_id=content.dataset_id,
                profile=binding.effective_profile,
                risk=binding.protocol_document["policy"]["risk"],
            )
            if policy != expected_policy:
                raise ValueError("execution policy differs from selected native arm")
        expected_multiplier, expected_latency = _FROZEN_SCENARIOS[binding.scenario]
        if (
            policy.multiplier != expected_multiplier
            or policy.order_latency_bars != expected_latency
        ):
            raise ValueError("execution policy differs from the frozen scenario")
        self.reserve_plan_digest = _digest(reserve_plan_digest, "reserve_plan_digest")
        self._validate_content(
            content, binding.window, binding.observation_feature_indices
        )
        capital = _finite(binding.initial_capital, "initial_capital", minimum=0.0)
        if capital <= 0.0:
            raise ValueError("initial capital must be positive")
        multipliers = _pair_numbers(
            content.arrays["contract_multipliers"], "contract_multipliers", minimum=0.0
        )
        if any(value <= 0.0 for value in multipliers):
            raise ValueError("contract multipliers must be positive")
        opening_marks = self._array_pair(
            content, "mark_price", binding.window.actor_start
        )
        self.content = content
        self.policy = policy
        self.binding = binding
        self.window = binding.window
        self.selected_symbol_index = binding.selected_symbol_index
        self.initial_capital = capital
        self.expected_horizon_seconds = binding.expected_horizon_seconds
        self.observation_feature_indices = binding.observation_feature_indices
        self._account = OwnedBinary64Account(
            cash=capital,
            exact_quantities=(Fraction(0), Fraction(0)),
            marks=opening_marks,
            multipliers=multipliers,
        )
        self._account.apply(
            MarginEvent(
                max_leverage=self.policy.max_leverage,
                maintenance_margin_rate=self.policy.maintenance_margin_rate,
                collateral_haircut=self.policy.collateral_haircut,
                insolvency_valuation=self.policy.insolvency_valuation,
                margin_mode=self.policy.margin_mode,
            )
        )
        self._active: dict[str, _OwnedOrder] = {}
        self._terminal: dict[str, _OwnedOrder] = {}
        self._all_orders: dict[str, _OwnedOrder] = {}
        self._terminal_identity_memo = _TerminalIdentityMemo()
        self._current_index = binding.window.actor_start
        self._last_time_ns = self._time_ns(content, binding.window.actor_start)
        self._terminated_early = False

    @staticmethod
    def _validate_content(
        content: VerifiedDatasetContent,
        window: FrozenMarketWindow,
        observation_feature_indices: tuple[int, ...],
    ) -> None:
        symbols = _string_tuple(content.dataset.get("symbols"), "Dataset symbols")
        feature_names = _string_tuple(
            content.dataset.get("feature_names"), "Dataset feature names"
        )
        volume_units = _string_tuple(
            content.dataset.get("volume_units"), "Dataset volume units"
        )
        if len(symbols) != 2:
            raise ValueError("fixed market replay requires exactly two symbols")
        if not feature_names:
            raise ValueError("Dataset feature names are required")
        if len(volume_units) != 2 or any(
            unit not in {"base_asset", "contracts", "quote_notional"}
            for unit in volume_units
        ):
            raise ValueError("Dataset volume units do not match the two-slot scope")
        arrays = content.arrays
        required = {
            "timestamps",
            "available_at",
            "information_available",
            "features",
            "feature_available",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tradable",
            "symbol_active",
            "asset_active",
            "fee_rate",
            "taker_fee_rate",
            "spread_rate",
            "max_participation_rate",
            "minimum_notional",
            "lot_size",
            "tick_size",
            "borrow_available",
            "borrow_rate",
            "funding_price_rate",
            "funding_rate",
            "funding_event_count",
            "funding_due",
            "buy_allowed",
            "sell_allowed",
            "mark_price",
            "dividend",
            "split_factor",
            "cash_rate",
            "contract_multipliers",
        }
        if not required.issubset(arrays):
            raise ValueError("verified Dataset is missing raw market/accounting arrays")
        timestamps = arrays["timestamps"]
        if (
            timestamps.ndim != 1
            or timestamps.dtype.kind != "M"
            or np.isnat(timestamps).any()
        ):
            raise ValueError("Dataset timestamps must be a complete datetime64 vector")
        count = timestamps.size
        if window.terminal_stop >= count:
            raise ValueError("window processing stop exceeds the Dataset source rows")
        if any(array.flags.writeable for array in arrays.values()):
            raise ValueError("verified Dataset arrays must remain read-only")
        shapes = {
            "available_at": (count, 2),
            "information_available": (count, 2),
            "features": (count, 2, len(feature_names)),
            "feature_available": (count, 2, len(feature_names)),
            "cash_rate": (count,),
            "contract_multipliers": (2,),
        }
        for name in required - {"timestamps"}:
            expected = shapes.get(name, (count, 2))
            if arrays[name].shape != expected:
                raise ValueError(f"Dataset {name} array has an unexpected shape")
        for index in observation_feature_indices:
            if index >= len(feature_names):
                raise ValueError("observation feature index exceeds Dataset features")
        times = timestamps.astype("int64")
        if count < 2:
            raise ValueError("Dataset must contain multiple source rows")
        step = int(times[1] - times[0])
        if step <= 0 or any(
            int(right - left) != step
            for left, right in zip(times[:-1], times[1:], strict=True)
        ):
            raise ValueError("Dataset clock is not strictly regular")
        if not math.isclose(
            step / 3_600_000_000_000,
            _finite(content.dataset["nominal_bar_hours"], "nominal_bar_hours"),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("Dataset nominal cadence differs from raw clocks")

    @staticmethod
    def _time_ns(content: VerifiedDatasetContent, index: int) -> int:
        return int(content.arrays["timestamps"][index].astype("int64"))

    @staticmethod
    def _array_pair(
        content: VerifiedDatasetContent, name: str, index: int
    ) -> tuple[float, float]:
        return _pair_numbers(content.arrays[name][index], f"{name}[{index}]")

    @property
    def account_snapshot(self) -> AccountSnapshot:
        return self._account.snapshot()

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def active_order_count(self) -> int:
        return len(self._active)

    @property
    def actor_advances(self) -> int:
        return max(
            0,
            min(self._current_index, self.window.actor_stop) - self.window.actor_start,
        )

    @property
    def reserve_advances(self) -> int:
        return max(0, self._current_index - self.window.actor_stop)

    @property
    def source_scope(self) -> str:
        return "CALLER_PINNED_INPUTS_UNAUTHENTICATED"

    def advance(
        self,
        *,
        forecast_index: FrozenForecastIndex,
        role: str,
        observed_fills: tuple[ObservedFill, ...] | None = None,
    ) -> ReplayAdvance:
        """Derive one decision, order transition and raw-row fill atomically."""
        if self._terminated_early or self._account.insolvent:
            raise ValueError("economic termination is absorbing")
        if self._current_index >= self.window.terminal_stop:
            raise ValueError("frozen replay has consumed its full window")
        if type(forecast_index) is not FrozenForecastIndex:
            raise ValueError("requires an exact frozen forecast index")
        if observed_fills is not None and (
            type(observed_fills) is not tuple
            or any(type(fill) is not ObservedFill for fill in observed_fills)
        ):
            raise ValueError("observed_fills must be an immutable tuple of comparisons")
        expected_role = {"nonrl": "NONRL", "cash": "CASH"}[self.binding.native_role]
        if role != expected_role:
            raise ValueError("decision role differs from the bound frozen arm")

        decision_index = self._current_index
        processing_index = decision_index + 1
        decision_time = self._time_ns(self.content, decision_index)
        processing_time = self._time_ns(self.content, processing_index)
        expected_step = self._time_ns(self.content, 1) - self._time_ns(self.content, 0)
        if (
            decision_time != self._last_time_ns
            or processing_time - decision_time != expected_step
        ):
            raise ValueError("raw row clock jumped or moved backwards")
        if not bool(
            self.content.arrays["asset_active"][
                processing_index, self.selected_symbol_index
            ]
        ):
            raise ValueError("inactive selected asset row is outside the fixed profile")
        phase = "actor" if processing_index <= self.window.actor_stop else "reserve"
        source = self._decision_snapshot(decision_index)
        before = self._account.snapshot()
        current_weights = self._weights(before)
        staged_active = dict(self._active)
        staged_terminal = dict(self._terminal)
        new_intent_ids: list[str] = []
        cancelled_ids: list[str] = []
        actor_identity: ActorIdentity | None = None
        target_identity: str | None

        if phase == "actor":
            allocation_costs, cost_source_identity = self._derive_cost_assumptions(
                source, decision_index=decision_index
            )
            assumptions = forecast_index.lookup(
                source,
                cost_assumptions=allocation_costs,
            )
            allocation_source_identity = forecast_index.allocation_source_identity(
                source,
                cost_estimates_payload=self._cost_estimates_payload(
                    source, allocation_costs, cost_source_identity
                ),
            )
            target = compose_frozen_allocation_target(
                assumptions,
                current_weight=current_weights[self.selected_symbol_index],
                lower_weight=-self.policy.maximum_gross,
                upper_weight=self.policy.maximum_gross,
                drawdown=before.max_drawdown,
                role=role,
                emergency_flatten=before.max_drawdown >= self.policy.drawdown_stop,
                reduce_only=before.max_drawdown >= self.policy.drawdown_stop,
            )
            final_weights = (
                (target.final_target_weight, 0.0)
                if self.selected_symbol_index == 0
                else (0.0, target.final_target_weight)
            )
            built_identity = self._build_identity(
                source=source,
                before=before,
                current_weights=current_weights,
                final_weights=final_weights,
                baseline_weight=target.baseline_weight,
                objective=target.baseline_objective,
                action_mode=target.action_mode,
                action_code=target.action_code,
                drawdown=before.max_drawdown,
                assumptions=assumptions,
                allocation_source_identity=allocation_source_identity,
                active=staged_active,
                terminal=staged_terminal,
            )
            actor_identity = built_identity
            target_identity = built_identity.target_identity
        else:
            final_weights = (0.0, 0.0)
            target_identity = self._reserve_target_identity(decision_index)
        transition = reconcile_expected(
            exact_quantities=before.exact_quantities,
            current_weights=current_weights,
            final_risk_weights=final_weights,
            decision_equity=before.wealth,
            valuation_prices=before.marks,
            multipliers=before.multipliers,
            reference_prices=self._array_pair(
                self.content, "mark_price", decision_index
            ),
            dataset_id=self.policy.dataset_id,
            target_identity=target_identity,
            execution_policy_digest=self.policy.execution_policy_digest,
            submit_index=decision_index,
            latency_bars=self.policy.order_latency_bars,
            pending_orders=tuple(
                order.active_projection() for order in staged_active.values()
            ),
        )
        for cancellation in transition.cancelled_orders:
            order = staged_active.pop(cancellation.order_id)
            staged_terminal[order.intent.order_id] = _transition_order(
                order,
                status="cancelled",
                last_processed_index=decision_index,
                terminal_reason=cancellation.reason,
            )
            cancelled_ids.append(order.intent.order_id)
        for kept_order in transition.kept_orders:
            if kept_order.order_id not in staged_active:
                raise ValueError(
                    "reconciliation kept an order outside owned active state"
                )
        for intent in transition.new_intents:
            if (
                intent.order_id in self._all_orders
                or intent.order_id in staged_active
                or intent.order_id in staged_terminal
            ):
                raise ValueError("order identity was reused after an earlier lifecycle")
            if intent.symbol_index != self.selected_symbol_index:
                raise ValueError(
                    "one-symbol driver proposed an order for the other slot"
                )
            new_order = _OwnedOrder(
                intent, _actor_intent_basis_from_identity(actor_identity)
            )
            staged_active[intent.order_id] = new_order
            new_intent_ids.append(intent.order_id)
        if len(staged_active) > 1:
            raise ValueError(
                "fixed one-symbol replay exceeded its one-active-order invariant"
            )

        observation = self._processing_observation(
            decision_index, processing_index, source
        )
        open_marks = cast(tuple[float, float], observation["open"])
        open_probe = copy.deepcopy(self._account)
        open_probe.apply(RevaluationEvent(open_marks))
        if open_probe.insolvent:
            event_batch: list[object] = [
                RevaluationEvent(open_marks),
                DrawdownObservation(),
                RetainedTerminationEvent(
                    mark_prices=open_marks,
                    period_start_value=max(before.wealth, _TOLERANCE),
                ),
            ]
            fills: list[OwnedFill] = []
            for order_id, order in tuple(staged_active.items()):
                staged_terminal[order_id] = _transition_order(
                    order,
                    status="cancelled",
                    last_processed_index=processing_index,
                    terminal_reason="economic_termination",
                )
                del staged_active[order_id]
                cancelled_ids.append(order_id)
        else:
            event_batch, fills, staged_active, staged_terminal = (
                self._derive_row_events(
                    observation,
                    before,
                    staged_active,
                    staged_terminal,
                )
            )
        expected_observed = tuple(fill.observed_projection() for fill in fills)
        if observed_fills is not None and observed_fills != expected_observed:
            raise ValueError(
                "observed fill batch differs from independently derived raw-row fills"
            )

        working_account = copy.deepcopy(self._account)
        for event_index, event in enumerate(event_batch):
            working_account.apply(event)
            if working_account.insolvent:
                if isinstance(event, FundingMarkEvent):
                    # Native carry application is followed by the final margin
                    # check; it owns any post-funding flatten without a second
                    # zero-funding return observation.
                    continue
                if (
                    isinstance(event, RevaluationEvent)
                    and event_index + 1 < len(event_batch)
                    and isinstance(event_batch[event_index + 1], DrawdownObservation)
                ):
                    # Native begin_bar records open-mark drawdown before retained
                    # termination is finalized.
                    continue
                if (
                    isinstance(event, MarginEvent)
                    and event_index == len(event_batch) - 1
                ):
                    break
                if not isinstance(event, RetainedTerminationEvent):
                    working_account.apply(
                        RetainedTerminationEvent(
                            mark_prices=working_account.snapshot().marks,
                            period_start_value=max(before.wealth, _TOLERANCE),
                        )
                    )
                break
        account_after = working_account.snapshot()
        other_slot = 1 - self.selected_symbol_index
        if account_after.exact_quantities[other_slot] != 0:
            raise ValueError("nonselected account slot acquired a position")
        if any(
            order.intent.symbol_index != self.selected_symbol_index
            for order in staged_active.values()
        ):
            raise ValueError("owned pending order escaped the selected symbol slot")
        if working_account.insolvent:
            for order_id, order in tuple(staged_active.items()):
                staged_terminal[order_id] = _transition_order(
                    order,
                    status="cancelled",
                    last_processed_index=processing_index,
                    terminal_reason="economic_termination",
                )
                del staged_active[order_id]
                cancelled_ids.append(order_id)
            self._terminated_early = True
        self._account = working_account
        self._active = staged_active
        self._terminal = staged_terminal
        for order_id, order in (*staged_active.items(), *staged_terminal.items()):
            self._all_orders[order_id] = order
        self._current_index = processing_index
        self._last_time_ns = processing_time
        return ReplayAdvance(
            decision_index=decision_index,
            processing_index=processing_index,
            phase=phase,
            actor_identity=(
                None
                if actor_identity is None
                else actor_identity.actor_decision_identity
            ),
            target_identity=target_identity,
            new_intent_ids=tuple(new_intent_ids),
            cancelled_order_ids=tuple(cancelled_ids),
            owned_fills=tuple(fills),
            actor_context_state_digest=(
                None if actor_identity is None else actor_identity.context_state_digest
            ),
            actor_proposal_digest=(
                None if actor_identity is None else actor_identity.proposal_digest
            ),
            actor_decision_digest=(
                None if actor_identity is None else actor_identity.decision_digest
            ),
            actor_action_digest=(
                None if actor_identity is None else actor_identity.action_digest
            ),
            actor_context_bytes=(
                None if actor_identity is None else actor_identity.context_bytes
            ),
            actor_proposal_bytes=(
                None if actor_identity is None else actor_identity.proposal_bytes
            ),
            actor_decision_bytes=(
                None if actor_identity is None else actor_identity.decision_bytes
            ),
            actor_action_bytes=(
                None if actor_identity is None else actor_identity.action_bytes
            ),
        )

    def _decision_snapshot(self, index: int) -> OwnedDecisionSnapshot:
        arrays = self.content.arrays
        selected = self.selected_symbol_index
        names = _string_tuple(
            self.content.dataset["feature_names"], "Dataset feature names"
        )
        features = tuple(float(value) for value in arrays["features"][index, selected])
        availability = tuple(
            bool(value) for value in arrays["feature_available"][index, selected]
        )
        return OwnedDecisionSnapshot(
            dataset_id=self.content.dataset_id,
            symbol=_string_tuple(self.content.dataset["symbols"], "Dataset symbols")[
                selected
            ],
            decision_time_ns=self._time_ns(self.content, index),
            source_available_at_ns=int(
                arrays["available_at"][index, selected].astype("int64")
            ),
            information_available=bool(
                arrays["information_available"][index, selected]
            ),
            feature_names=names,
            feature_values=features,
            feature_available=availability,
            dataset_close=float(arrays["close"][index, selected]),
            book_mark=float(arrays["mark_price"][index, selected]),
            expected_horizon_seconds=self.expected_horizon_seconds,
            observation_feature_indices=self.observation_feature_indices,
        )

    @staticmethod
    def _weights(snapshot: AccountSnapshot) -> tuple[float, float]:
        if not math.isfinite(snapshot.wealth) or snapshot.wealth <= 0.0:
            raise ValueError("owned account equity must be finite and positive")
        position_values = np.asarray(snapshot.projected_quantities, dtype=np.float64)
        position_values *= np.asarray(snapshot.marks, dtype=np.float64)
        position_values *= np.asarray(snapshot.multipliers, dtype=np.float64)
        return (
            float(position_values[0] / snapshot.wealth),
            float(position_values[1] / snapshot.wealth),
        )

    def _build_identity(
        self,
        *,
        source: OwnedDecisionSnapshot,
        before: AccountSnapshot,
        current_weights: tuple[float, float],
        final_weights: tuple[float, float],
        baseline_weight: float,
        objective: float,
        action_mode: str,
        action_code: int,
        drawdown: float,
        assumptions: AllocationAssumptions,
        allocation_source_identity: str,
        active: Mapping[str, _OwnedOrder],
        terminal: Mapping[str, _OwnedOrder],
    ) -> ActorIdentity:
        symbols = _string_tuple(self.content.dataset["symbols"], "Dataset symbols")
        terminal_memo = self._terminal_identity_memo
        terminal_memo.ensure_scope(
            (
                self.binding.account_id,
                self.content.dataset_id,
                self.policy.execution_policy_digest,
                symbols,
                self.selected_symbol_index,
            )
        )
        if not self._terminal_identity_memo_is_current():
            terminal_memo.clear()
        memo_transaction = terminal_memo.transaction()
        projection_memo: dict[int, tuple[_OwnedOrder, OwnedPendingOrder]] = {}
        risk = {
            "max_gross": self.policy.maximum_gross,
            "max_abs_weight": self.policy.maximum_gross,
            "max_turnover": None,
            "drawdown_start": self.policy.drawdown_start,
            "drawdown_stop": self.policy.drawdown_stop,
            "emergency_turnover_override": True,
            "fail_closed_tolerance": 1e-10,
        }
        book = {
            "quantities": before.projected_quantities,
            "cash": before.cash,
            "mark_prices": before.marks,
            "peak_value": before.peak_value,
            "contract_multipliers": before.multipliers,
            "max_drawdown": before.max_drawdown,
            "turnover_total": before.turnover_total,
            "total_cost": before.total_cost,
            "funding_pnl": before.funding_pnl,
            "fill_count": before.fill_count,
            "rebalance_events": before.rebalance_events,
            "returns_history": before.returns_history,
            "borrow_cost": before.borrow_cost,
            "margin_used": before.margin_used,
            "maintenance_margin": before.maintenance_margin,
            "maintenance_requirement": before.maintenance_requirement,
            "margin_deficit": before.margin_deficit,
            "insolvent": before.insolvent,
            "termination_reason": before.termination_reason,
            "as_of_index": self._current_index,
            "as_of_dataset_id": self.policy.dataset_id,
            "exact_quantities": tuple(str(value) for value in before.exact_quantities),
        }
        account = OwnedAccountComponents(
            account_id=self.binding.account_id,
            dataset_id=self.policy.dataset_id,
            symbols=(symbols[0], symbols[1]),
            execution_policy_digest=self.policy.execution_policy_digest,
            risk_config=risk,
            symbol_index=self.selected_symbol_index,
            decision_index=self._current_index,
            decision_time=str(
                np.datetime_as_string(
                    np.datetime64(source.decision_time_ns, "ns"), unit="ns"
                )
            ),
            book=book,
            active_orders=tuple(
                self._owned_projection(
                    item,
                    active,
                    terminal,
                    projection_memo,
                    memo_transaction,
                )
                for item in active.values()
            ),
            terminal_orders=tuple(
                self._owned_projection(
                    item,
                    active,
                    terminal,
                    projection_memo,
                    memo_transaction,
                )
                for item in terminal.values()
            ),
        )
        allocation_inputs = {
            "symbol": source.symbol,
            "decision_time": str(
                np.datetime_as_string(
                    np.datetime64(source.decision_time_ns, "ns"), unit="ns"
                )
            ),
            "available_at": str(
                np.datetime_as_string(
                    np.datetime64(source.source_available_at_ns, "ns"), unit="ns"
                )
            ),
            "horizon_end": str(
                np.datetime_as_string(
                    np.datetime64(
                        source.decision_time_ns
                        + source.expected_horizon_seconds * 1_000_000_000,
                        "ns",
                    ),
                    unit="ns",
                )
            ),
            "source_identity": allocation_source_identity,
            "expected_simple_return": assumptions.expected_simple_return,
            "return_variance": assumptions.return_variance,
            "buy_cost": assumptions.buy_cost,
            "sell_cost": assumptions.sell_cost,
            "exit_cost": assumptions.exit_cost,
            "funding_return": assumptions.funding_return,
            "borrow_return": assumptions.borrow_return,
            "cash_return": assumptions.cash_return,
            "return_unit": "expected_simple_return",
        }
        allocator = {
            "lower_weight": -self.policy.maximum_gross,
            "upper_weight": self.policy.maximum_gross,
            "max_turnover": None,
            "risk_aversion": 0.0,
        }
        decision_fields = {
            "feature_names": tuple(
                source.feature_names[index]
                for index in self.observation_feature_indices
            ),
            "feature_values": tuple(
                source.feature_values[index]
                for index in self.observation_feature_indices
            ),
            "max_drawdown": drawdown,
            "initial_capital": self.initial_capital,
            "remaining_steps": max(0, self.window.actor_stop - self._current_index),
            "pending_gross": (
                float(
                    sum(
                        (
                            abs(Fraction(str(order.remaining_quantity)))
                            for order in active.values()
                        ),
                        Fraction(0),
                    )
                )
                * before.marks[self.selected_symbol_index]
                * before.multipliers[self.selected_symbol_index]
                / self.initial_capital
            ),
            "pending_count": len(active),
        }
        identity = _build_actor_identity_core(
            account,
            allocation_inputs,
            allocator,
            independent_target_weight=baseline_weight,
            independent_objective_value=objective,
            decision_fields=decision_fields,
            action_contract={"mode": action_mode, "scale": 1.0},
            action_code=action_code,
            final_risk_weights=final_weights,
            terminal_memo=memo_transaction,
        )
        if memo_transaction.commit():
            return identity
        return build_actor_identity(
            account,
            allocation_inputs,
            allocator,
            independent_target_weight=baseline_weight,
            independent_objective_value=objective,
            decision_fields=decision_fields,
            action_contract={"mode": action_mode, "scale": 1.0},
            action_code=action_code,
            final_risk_weights=final_weights,
        )

    def _terminal_identity_memo_is_current(self) -> bool:
        memo = self._terminal_identity_memo
        for order_id, entry in memo.entries.items():
            source = entry.source
            if (
                source.intent.order_id != order_id
                or self._terminal.get(order_id) is not source
                or self._all_orders.get(order_id) is not source
            ):
                return False
            predecessor_id = source.intent.replaced_order_id
            if predecessor_id != entry.predecessor_id:
                return False
            if predecessor_id is None:
                predecessor_projection = None
                if entry.predecessor_source is not None:
                    return False
            else:
                if predecessor_id == order_id:
                    return False
                predecessor = self._terminal.get(predecessor_id)
                predecessor_entry = memo.entries.get(predecessor_id)
                if (
                    predecessor is None
                    or predecessor is not entry.predecessor_source
                    or self._all_orders.get(predecessor_id) is not predecessor
                    or predecessor_entry is None
                    or predecessor_entry.source is not predecessor
                ):
                    return False
                predecessor_projection = predecessor_entry.projection
            if not _cacheable_terminal_projection(
                source, entry.projection, predecessor_projection
            ):
                return False
        return True

    def _owned_projection(
        self,
        order: _OwnedOrder,
        active: Mapping[str, _OwnedOrder],
        terminal: Mapping[str, _OwnedOrder],
        memo: dict[int, tuple[_OwnedOrder, OwnedPendingOrder]] | None = None,
        terminal_memo: _TerminalIdentityMemoTransaction | None = None,
    ) -> OwnedPendingOrder:
        cache = {} if memo is None else memo
        cached = cache.get(id(order))
        if cached is not None and cached[0] is order:
            return cached[1]
        if terminal_memo is not None:
            found = terminal_memo.projection_for(order.intent.order_id, order)
            if found is not None:
                cache[id(order)] = (order, found)
                return found
        replaced = None
        predecessor: _OwnedOrder | None = None
        if order.intent.replaced_order_id is not None:
            prior = self._all_orders.get(order.intent.replaced_order_id)
            if prior is None:
                prior = terminal.get(order.intent.replaced_order_id) or active.get(
                    order.intent.replaced_order_id
                )
            if prior is None:
                raise ValueError(
                    "replaced order identity is missing from owned history"
                )
            predecessor = prior
            replaced = self._owned_projection(
                prior, active, terminal, cache, terminal_memo
            )
        if order.actor_identity is None:
            raise ValueError(
                "reserve identity cannot be projected as an actor identity"
            )
        intent = OwnedIntentComponents(
            identity_source=order.actor_identity,
            dataset_id=order.intent.dataset_id,
            execution_policy_digest=order.intent.execution_policy_digest,
            symbol_index=order.intent.symbol_index,
            requested_quantity=order.intent.requested_quantity,
            order_type=order.intent.order_type,
            time_in_force=order.intent.time_in_force,
            limit_price=order.intent.limit_price,
            stop_price=order.intent.stop_price,
            submit_index=order.intent.submit_index,
            eligible_index=order.intent.eligible_index,
            expiry_index=order.intent.expiry_index,
            submission_reference_price=order.intent.submission_reference_price,
            decision_equity=order.intent.decision_equity,
            replaced_order=replaced,
            reduce_only=order.intent.reduce_only,
        )
        projection = OwnedPendingOrder(
            intent=intent,
            remaining_quantity=order.remaining_quantity,
            cumulative_filled_quantity=project_quantity(order.exact_filled_quantity),
            cumulative_filled_notional=order.cumulative_filled_notional,
            status=order.status,
            last_processed_index=order.last_processed_index,
            terminal_reason=order.terminal_reason,
            evidence_version=order.evidence_version,
            exact_cumulative_filled_quantity=str(order.exact_filled_quantity),
        )
        cache[id(order)] = (order, projection)
        order_id = order.intent.order_id
        current_owner = (
            self._all_orders.get(order_id) is order
            and self._terminal.get(order_id) is order
            and (
                predecessor is None
                or (
                    self._all_orders.get(predecessor.intent.order_id) is predecessor
                    and self._terminal.get(predecessor.intent.order_id) is predecessor
                )
            )
        )
        if (
            terminal_memo is not None
            and current_owner
            and _cacheable_terminal_projection(order, projection, replaced)
        ):
            terminal_memo.stage_projection(
                _TerminalProjectionEntry(
                    source=order,
                    projection=projection,
                    predecessor_id=order.intent.replaced_order_id,
                    predecessor_source=predecessor,
                )
            )
        return projection

    def _processing_observation(
        self, decision_index: int, processing_index: int, source: OwnedDecisionSnapshot
    ) -> dict[str, Any]:
        arrays = self.content.arrays
        index = self.selected_symbol_index

        def rule(name: str) -> float:
            return max(
                float(arrays[name][processing_index, index]),
                getattr(self.policy, f"{name}_floor"),
            )

        open_prices = self._array_pair(self.content, "open", processing_index)
        high = self._array_pair(self.content, "high", processing_index)
        low = self._array_pair(self.content, "low", processing_index)
        close = self._array_pair(self.content, "close", processing_index)
        mark = self._array_pair(self.content, "mark_price", processing_index)
        prior_close = self._array_pair(self.content, "close", decision_index)
        prior_volume = self._array_pair(self.content, "volume", decision_index)
        tick = rule("tick_size") * self.policy.tick_size_factor
        lot = rule("lot_size") * self.policy.lot_size_factor
        minimum = rule("minimum_notional") * self.policy.minimum_notional_factor
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (*open_prices, *high, *low, *close, *mark)
        ):
            raise ValueError("processing raw prices must be finite and positive")
        if (
            high[index] < max(open_prices[index], close[index])
            or low[index] > min(open_prices[index], close[index])
            or low[index] > high[index]
        ):
            raise ValueError("processing OHLC is internally inconsistent")
        if not np.all(arrays["split_factor"][processing_index] == 1.0):
            raise ValueError("split actions are outside this fixed market profile")
        if float(arrays["cash_rate"][processing_index]) != 0.0:
            raise ValueError("fixed market profile requires zero raw cash interest")
        return {
            "decision_index": decision_index,
            "processing_index": processing_index,
            "decision_time_ns": source.decision_time_ns,
            "processing_time_ns": self._time_ns(self.content, processing_index),
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "mark": mark,
            "prior_close": prior_close,
            "prior_volume": prior_volume,
            "volume_units": _string_tuple(
                self.content.dataset["volume_units"], "Dataset volume units"
            ),
            "tradable": tuple(
                bool(value) for value in arrays["tradable"][processing_index]
            ),
            "active": tuple(
                bool(value) for value in arrays["asset_active"][processing_index]
            ),
            "buy_allowed": tuple(
                bool(value) for value in arrays["buy_allowed"][processing_index]
            ),
            "sell_allowed": tuple(
                bool(value) for value in arrays["sell_allowed"][processing_index]
            ),
            "borrow_available": tuple(
                bool(value) for value in arrays["borrow_available"][processing_index]
            ),
            "tick": tick,
            "lot": lot,
            "minimum": minimum,
            "fee_rate": float(arrays["fee_rate"][processing_index, index]),
            "taker_fee_rate": float(arrays["taker_fee_rate"][processing_index, index]),
            "spread_rate": float(arrays["spread_rate"][processing_index, index]),
            "participation": float(
                arrays["max_participation_rate"][processing_index, index]
            ),
            "borrow_rate": self._array_pair(
                self.content, "borrow_rate", processing_index
            ),
            "funding_rate": self._array_pair(
                self.content, "funding_price_rate", processing_index
            ),
            "funding_due": tuple(
                bool(value) for value in arrays["funding_due"][processing_index]
            ),
            "dividend": self._array_pair(self.content, "dividend", processing_index),
            "cash_rate": float(arrays["cash_rate"][processing_index]),
            "multipliers": _pair_numbers(
                arrays["contract_multipliers"], "contract_multipliers", minimum=0.0
            ),
            "split_factor": _pair_numbers(
                arrays["split_factor"][processing_index], "split_factor"
            ),
            "year_fraction": (
                self._time_ns(self.content, processing_index)
                - self._time_ns(self.content, decision_index)
            )
            / (365.0 * 24 * 3600 * 1_000_000_000),
            "period_start_value": max(self._account.wealth, _TOLERANCE),
        }

    def _derive_row_events(
        self,
        row: dict[str, Any],
        before: AccountSnapshot,
        active: dict[str, _OwnedOrder],
        terminal: dict[str, _OwnedOrder],
    ) -> tuple[
        list[object],
        list[OwnedFill],
        dict[str, _OwnedOrder],
        dict[str, _OwnedOrder],
    ]:
        processing = row["processing_index"]
        slot = self.selected_symbol_index
        open_prices = list(row["open"])
        execution_price = self._tick_price(open_prices[slot], row["tick"], 1)
        open_prices[slot] = execution_price
        # Current open revaluation precedes admission; no observed BookState is read.
        open_marks = self._array_pair(self.content, "open", processing)
        working_cash = before.cash
        opening_quantities = before.exact_quantities
        open_position_values = tuple(
            project_quantity(quantity) * open_marks[index] * before.multipliers[index]
            for index, quantity in enumerate(opening_quantities)
        )
        open_equity = working_cash + float(
            np.asarray(open_position_values, dtype=np.float64).sum()
        )
        if not math.isfinite(open_equity):
            raise ValueError("open equity became non-finite")

        fills: list[OwnedFill] = []
        account_deltas = [Fraction(0), Fraction(0)]
        staged_events: list[object] = [
            RevaluationEvent(open_marks),
            DrawdownObservation(),
        ]
        for order_id, order in tuple(active.items()):
            intent = order.intent
            if row["processing_time_ns"] - row["decision_time_ns"] <= 0:
                raise ValueError("processing clock must follow decision clock")
            if processing < intent.eligible_index:
                active[order_id] = _transition_order(
                    order,
                    status="latency_wait",
                    last_processed_index=processing,
                )
                continue
            if (
                intent.dataset_id != self.policy.dataset_id
                or intent.execution_policy_digest != self.policy.execution_policy_digest
            ):
                rejected = _transition_order(
                    order,
                    status="rejected",
                    last_processed_index=processing,
                    terminal_reason="identity_mismatch",
                )
                terminal[order_id] = rejected
                del active[order_id]
                continue
            reason: str | None = None
            if not row["active"][slot]:
                reason = "inactive_asset"
            elif not row["tradable"][slot]:
                reason = "non_tradable_market"
            elif intent.requested_quantity > 0.0 and not row["buy_allowed"][slot]:
                reason = "buy_disabled"
            elif intent.requested_quantity < 0.0 and not row["sell_allowed"][slot]:
                reason = "sell_disabled"
            else:
                reason = self._admission_reason(
                    order, row, open_equity, opening_quantities, open_marks
                )
            if reason is not None:
                status = (
                    "expired"
                    if order.status in {"triggered", "partially_filled"}
                    else "rejected"
                )
                terminal[order_id] = _transition_order(
                    order,
                    status=status,
                    last_processed_index=processing,
                    terminal_reason=reason,
                )
                del active[order_id]
                continue
            if order.status in {"submitted", "latency_wait"}:
                order = _transition_order(
                    order,
                    status="eligible",
                    last_processed_index=processing,
                )
            remaining = order.remaining_quantity
            if abs(remaining) <= _TOLERANCE:
                raise ValueError("active order has no remaining exact quantity")

            prior_volume = row["prior_volume"][slot]
            prior_close = row["prior_close"][slot]
            unit = row["volume_units"][slot]
            multiplier = row["multipliers"][slot]
            if unit == "quote_notional":
                market_notional = prior_volume
                quantity_capacity = None
            elif unit == "base_asset":
                market_notional = prior_volume * prior_close
                quantity_capacity = prior_volume / multiplier
            elif unit == "contracts":
                market_notional = prior_volume * prior_close * multiplier
                quantity_capacity = prior_volume
            else:  # guarded when Dataset content is loaded
                raise ValueError("unsupported raw volume unit")
            participation_limit = min(
                self.policy.max_participation_rate, row["participation"]
            )
            capacity = market_notional * participation_limit
            admitted, admitted_count = self._quantize(remaining, row["lot"])
            if abs(admitted) <= _TOLERANCE:
                active[order_id] = order
                continue
            if quantity_capacity is not None:
                raw_quantity_cap = quantity_capacity * participation_limit
            else:
                raw_quantity_cap = math.inf
            max_by_notional = (
                capacity / (execution_price * multiplier) if capacity > 0.0 else 0.0
            )
            amount, count = self._capacity_fill(
                admitted,
                admitted_count,
                row["lot"],
                min(max_by_notional, raw_quantity_cap),
                execution_price,
                multiplier,
            )
            if abs(amount) <= _TOLERANCE:
                active[order_id] = order
                continue
            exact = self._exact_fill(amount, row["lot"], count)
            account_delta = exact
            if count is None and amount == -project_quantity(opening_quantities[slot]):
                # Native BookState consumes the exact inventory when an
                # unquantized fill equals its projected full-close quantity.
                account_delta = -opening_quantities[slot]
            account_deltas[slot] += account_delta
            notional = abs(amount) * execution_price * multiplier
            if notional + _TOLERANCE < row["minimum"]:
                active[order_id] = order
                continue
            if intent.reduce_only and (
                opening_quantities[slot] * exact >= 0
                or abs(exact) > abs(opening_quantities[slot])
            ):
                terminal[order_id] = replace(
                    order,
                    status="expired",
                    last_processed_index=processing,
                    terminal_reason="reduce_only_exhausted",
                )
                del active[order_id]
                continue
            notional_fraction = (
                0.0 if market_notional <= _TOLERANCE else notional / market_notional
            )
            if unit == "base_asset":
                quantity_fraction = (
                    abs(amount) / prior_volume if prior_volume > 0.0 else 0.0
                )
            elif unit == "contracts":
                quantity_fraction = (
                    abs(amount) / prior_volume if prior_volume > 0.0 else 0.0
                )
            else:
                quantity_fraction = 0.0
            fill_participation = max(notional_fraction, quantity_fraction)
            impact = self.policy.impact_rate * math.sqrt(fill_participation)
            unit_cost = self.policy.multiplier * (
                self.policy.fee_rate
                + row["fee_rate"]
                + self.policy.taker_fee_rate
                + row["taker_fee_rate"]
                + self.policy.spread_rate
                + row["spread_rate"]
                + impact
            )
            cost = notional * unit_cost
            if not math.isfinite(cost):
                raise ValueError("independent market costs must be finite")
            updated_order = _apply_owned_fill_progress(
                order,
                exact_fill=exact,
                filled_notional=notional,
                processing_index=processing,
            )
            if updated_order.status == "filled":
                terminal[order_id] = updated_order
                del active[order_id]
            else:
                active[order_id] = updated_order
            fills.append(
                OwnedFill(
                    order_id,
                    processing,
                    exact,
                    execution_price,
                    notional,
                    cost,
                    fill_participation,
                    count,
                    row["lot"] if count is not None else 0.0,
                )
            )

        if fills:
            fill_prices = list(open_marks)
            for fill in fills:
                fill_prices[slot] = fill.execution_price
            total_notional = sum(fill.filled_notional for fill in fills)
            total_cost = sum(fill.cost_amount for fill in fills)
            staged_events.append(
                ExecutionEvent(
                    deltas=(account_deltas[0], account_deltas[1]),
                    fill_prices=(fill_prices[0], fill_prices[1]),
                    valuation_prices=(fill_prices[0], fill_prices[1]),
                    cost_amount=total_cost,
                    turnover=total_notional
                    / max(row["period_start_value"], _TOLERANCE),
                )
            )
            staged_events.append(
                MarginEvent(
                    max_leverage=self.policy.max_leverage,
                    maintenance_margin_rate=self.policy.maintenance_margin_rate,
                    collateral_haircut=self.policy.collateral_haircut,
                    insolvency_valuation=self.policy.insolvency_valuation,
                    margin_mode=self.policy.margin_mode,
                )
            )

        # No split or delisting settlement is admitted by this frozen profile.
        staged_events.extend(
            (
                DividendEvent(row["dividend"]),
                CashInterestEvent(row["cash_rate"], row["year_fraction"]),
                BorrowEvent(
                    row["borrow_rate"],
                    row["year_fraction"],
                    self.policy.borrow_rate_multiplier,
                ),
                FundingMarkEvent(
                    row["funding_rate"],
                    row["funding_due"],
                    row["mark"],
                    row["period_start_value"],
                ),
                MarginEvent(
                    max_leverage=self.policy.max_leverage,
                    maintenance_margin_rate=self.policy.maintenance_margin_rate,
                    collateral_haircut=self.policy.collateral_haircut,
                    insolvency_valuation=self.policy.insolvency_valuation,
                    margin_mode=self.policy.margin_mode,
                ),
            )
        )
        return staged_events, fills, active, terminal

    def _admission_reason(
        self,
        order: _OwnedOrder,
        row: dict[str, Any],
        equity: float,
        quantities: tuple[Fraction, Fraction],
        open_marks: tuple[float, float],
    ) -> str | None:
        intent = order.intent
        slot = self.selected_symbol_index
        if (
            intent.order_type != "market"
            or intent.time_in_force != "gtc"
            or intent.limit_price is not None
            or intent.stop_price is not None
        ):
            return "profile_requires_market_order"
        requested = order.remaining_quantity
        if not math.isfinite(requested) or abs(requested) <= _TOLERANCE:
            return "zero_quantity_after_rounding"
        exact_requested = exact_quantity(requested)
        if intent.reduce_only:
            position = quantities[slot]
            if not position:
                return "reduce_only_no_position"
            if position * exact_requested >= 0:
                return "reduce_only_wrong_direction"
            if abs(exact_requested) > abs(position):
                return "reduce_only_exceeds_position"
        current = project_quantity(quantities[slot])
        projected_unrounded = current + requested
        if projected_unrounded < -_TOLERANCE and not self.policy.allow_short:
            return "shorting_disabled"
        if (
            projected_unrounded < min(current, 0.0) - _TOLERANCE
            and not row["borrow_available"][slot]
        ):
            return "borrow_unavailable"
        admitted, _ = self._quantize(requested, row["lot"])
        if abs(admitted) <= _TOLERANCE:
            return "zero_quantity_after_rounding"
        notional = abs(admitted) * open_marks[slot] * row["multipliers"][slot]
        if notional + _TOLERANCE < row["minimum"]:
            return "below_minimum_notional"
        projected_quantities = np.asarray(
            [project_quantity(quantity) for quantity in quantities],
            dtype=np.float64,
        )
        projected_quantities[slot] += admitted
        prices = np.asarray(open_marks, dtype=np.float64)
        multipliers = np.asarray(row["multipliers"], dtype=np.float64)
        gross = float(np.sum(np.abs(projected_quantities * prices * multipliers)))
        projected_equity = equity
        if (
            not math.isfinite(projected_equity)
            or projected_equity <= _TOLERANCE
            or gross > projected_equity * self.policy.max_leverage + _TOLERANCE
        ):
            return "pretrade_leverage_exceeded"
        return None

    @staticmethod
    def _tick_price(price: float, tick: float, direction: int) -> float:
        if tick <= 0.0:
            return price
        del direction
        rounded = max(tick, float(np.round(price / tick) * tick))
        if not math.isfinite(rounded) or rounded <= 0.0:
            raise ValueError("market tick price is invalid")
        return rounded

    @staticmethod
    def _quantize(quantity: float, lot: float) -> tuple[float, int | None]:
        exact = exact_quantity(quantity)
        if lot <= 0.0:
            return quantity, None
        step = exact_quantity(lot)
        count = int(abs(exact) // step) * (-1 if exact < 0 else 1)
        return project_quantity(count * step), count

    @classmethod
    def _capacity_fill(
        cls,
        quantity: float,
        count: int | None,
        lot: float,
        quantity_capacity: float,
        price: float,
        multiplier: float,
    ) -> tuple[float, int | None]:
        if quantity_capacity <= 0.0:
            return 0.0, (0 if count is not None else None)
        bound = min(abs(quantity), quantity_capacity)
        direction = -1.0 if quantity < 0.0 else 1.0
        if count is None:
            amount = direction * bound
            for _ in range(8):
                if (
                    abs(amount) * price * multiplier
                    <= quantity_capacity * price * multiplier
                ):
                    break
                amount = math.nextafter(amount, 0.0)
            return amount, None
        step = exact_quantity(lot)
        maximum_count = min(abs(count), int(exact_quantity(quantity_capacity) // step))
        projected = project_quantity(maximum_count * step)
        while (
            maximum_count
            and projected * price * multiplier > quantity_capacity * price * multiplier
        ):
            maximum_count -= 1
            projected = project_quantity(maximum_count * step)
        signed_count = maximum_count * (-1 if count < 0 else 1)
        return project_quantity(signed_count * step), signed_count

    @staticmethod
    def _exact_fill(amount: float, lot: float, count: int | None) -> Fraction:
        if count is None:
            return exact_quantity(amount)
        if lot <= 0.0 or count == 0:
            return Fraction(0)
        exact = count * exact_quantity(lot)
        if project_quantity(exact) != amount:
            raise ValueError("owned fill differs from exact lot evidence")
        return exact

    def _derive_cost_assumptions(
        self,
        source: OwnedDecisionSnapshot,
        *,
        decision_index: int,
    ) -> tuple[AllocationAssumptions, str]:
        """Derive six rates and native recipe identity from frozen raw inputs."""
        if type(source) is not OwnedDecisionSnapshot:
            raise ValueError(
                "cost derivation requires an exact owned decision snapshot"
            )
        if (
            type(decision_index) is not int
            or not 0 <= decision_index < len(self.content.arrays["timestamps"]) - 1
        ):
            raise ValueError("cost decision index is outside raw source rows")
        expected = self._decision_snapshot(decision_index)
        if source != expected:
            raise ValueError("cost decision snapshot differs from pinned raw source")
        if self.expected_horizon_seconds % 3600:
            raise ValueError("fixed cost recipe requires an integer-hour horizon")

        arrays = self.content.arrays
        selected = self.selected_symbol_index
        symbols = _string_tuple(self.content.dataset["symbols"], "Dataset symbols")
        units = _string_tuple(
            self.content.dataset["volume_units"], "Dataset volume units"
        )
        if units[selected] != "quote_notional":
            raise ValueError("fixed cost recipe requires quote-notional raw volume")
        time_ns = self._time_ns(self.content, decision_index)
        timestamps = arrays["timestamps"].astype("int64")
        bar_ns = int(timestamps[1] - timestamps[0])
        recipe = self._cost_recipe_declarations()
        lookback_hours = _integer(
            recipe["funding_lookback_hours"], "funding_lookback_hours", minimum=1
        )
        minimum_events = _integer(
            recipe["funding_minimum_event_count"],
            "funding_minimum_event_count",
            minimum=1,
        )
        maximum_gap_hours = _finite(
            recipe["funding_maximum_counted_bar_close_gap_hours"],
            "funding_maximum_counted_bar_close_gap_hours",
            minimum=0.0,
        )
        maximum_age_hours = _finite(
            recipe["funding_maximum_counted_bar_close_age_hours"],
            "funding_maximum_counted_bar_close_age_hours",
            minimum=0.0,
        )
        reference_notional = _finite(
            recipe["reference_notional"], "reference_notional", minimum=0.0
        )
        lookback_ns = lookback_hours * _NS_PER_HOUR
        if bar_ns <= 0 or lookback_ns % bar_ns:
            raise ValueError("fixed funding lookback must align to the raw clock")
        lookback_bars = lookback_ns // bar_ns
        left = decision_index - lookback_bars
        if left < 0:
            raise ValueError("funding history lacks complete lookback intervals")
        boundary_ns = int(timestamps[left])
        if time_ns - boundary_ns != lookback_ns:
            raise ValueError("funding interval boundary differs from fixed lookback")
        coverage_start = _integer(
            recipe["funding_coverage_start"], "funding_coverage_start"
        )
        coverage_stop = _integer(
            recipe["funding_coverage_stop"], "funding_coverage_stop"
        )
        assumptions_available_at = _integer(
            recipe["assumptions_available_at"], "assumptions_available_at"
        )
        if (
            assumptions_available_at > time_ns
            or coverage_start >= time_ns
            or coverage_start > boundary_ns
            or coverage_stop < time_ns
        ):
            raise ValueError("declared cost source is unavailable or lacks coverage")

        profile = self._raw_execution_economics()
        raw_identity = self.content.dataset.get("identity_payload_json")
        if type(raw_identity) is not str:
            raise ValueError("pinned Dataset lacks execution-economics identity bytes")
        execution_identity = _json_object(raw_identity.encode("utf-8")).get(
            "execution_economics"
        )
        if type(execution_identity) is not dict:
            raise ValueError("Dataset execution-economics identity is not an object")
        window = slice(left + 1, decision_index + 1)
        decision_time = arrays["timestamps"][decision_index]
        if (
            not np.all(arrays["information_available"][window, selected])
            or np.any(arrays["available_at"][window, selected] > decision_time)
            or float(arrays["close"][decision_index, selected])
            != float(arrays["mark_price"][decision_index, selected])
        ):
            raise ValueError(
                "cost inputs require complete available intervals and same-close marks"
            )
        for name in (
            "fee_rate",
            "maker_fee_rate",
            "taker_fee_rate",
            "spread_rate",
            "max_participation_rate",
            "borrow_available",
            "borrow_rate",
        ):
            observed = arrays[name][decision_index, selected]
            if name == "borrow_available":
                matches = bool(observed) == profile[name]
            else:
                matches = float(observed) == profile[name]
            if not matches:
                raise ValueError("raw execution rows disagree with Dataset economics")
        if float(arrays["cash_rate"][decision_index]) != 0.0:
            raise ValueError("fixed cost recipe requires explicit zero cash interest")

        prefix_available = arrays["information_available"][
            : decision_index + 1, selected
        ] & (arrays["available_at"][: decision_index + 1, selected] <= decision_time)
        prefix_counts = arrays["funding_event_count"][: decision_index + 1, selected]
        prefix_due = arrays["funding_due"][: decision_index + 1, selected]
        prefix_rates = arrays["funding_rate"][: decision_index + 1, selected]
        if np.any(prefix_available & (prefix_due != (prefix_counts > 0))) or np.any(
            prefix_available & (prefix_counts == 0) & (prefix_rates != 0)
        ):
            raise ValueError("funding due/count/rates disagree in the available prefix")

        counts = arrays["funding_event_count"][window, selected]
        if sum(int(count) for count in counts) < minimum_events:
            raise ValueError("funding history has insufficient counted events")
        counted_closes = timestamps[window][counts > 0]
        gap_ns = np.diff(np.concatenate((np.asarray([boundary_ns]), counted_closes)))
        age_ns = time_ns - int(counted_closes[-1])
        if np.any(gap_ns > maximum_gap_hours * _NS_PER_HOUR):
            raise ValueError("funding counted-bar-close gap exceeds fixed recipe")
        if age_ns > maximum_age_hours * _NS_PER_HOUR:
            raise ValueError("funding counted-bar-close age exceeds fixed recipe")

        row_volume = float(arrays["volume"][decision_index, selected])
        if row_volume <= 0.0:
            raise ValueError("reference-size cost requires positive quote volume")
        execution_cost = cast(dict[str, object], recipe["execution_cost"])
        cost_multiplier = _finite(execution_cost["multiplier"], "cost.multiplier")
        cost_impact_rate = _finite(
            execution_cost["impact_rate"], "cost.impact_rate", minimum=0.0
        )
        cost_participation = _finite(
            execution_cost["max_participation_rate"],
            "cost.max_participation_rate",
            minimum=0.0,
        )
        cost_trigger_fractions = cast(
            list[float], execution_cost["trigger_volume_fractions"]
        )
        if (
            not cost_trigger_fractions
            or _finite(cost_trigger_fractions[0], "cost.trigger_volume_fractions[0]")
            != _COST_TRIGGER_OPEN_FRACTION
            or cost_multiplier != self.policy.multiplier
            or cost_impact_rate != _COST_IMPACT_RATE
            or cost_participation != _COST_MAX_PARTICIPATION
        ):
            raise ValueError("declared recipe differs from the frozen scenario scope")
        capacity = (
            row_volume
            * min(cost_participation, profile["max_participation_rate"])
            * cost_trigger_fractions[0]
        )
        if reference_notional > capacity:
            raise ValueError(
                "reference notional exceeds raw decision-row OPEN capacity"
            )
        participation = reference_notional / row_volume
        expected_slippage = (
            _finite(execution_cost["slippage_std"], "cost.slippage_std")
            * math.sqrt(2 / math.pi)
            * (
                1
                + _finite(
                    execution_cost["tail_slippage_probability"],
                    "cost.tail_slippage_probability",
                )
                * (
                    _finite(
                        execution_cost["tail_slippage_multiplier"],
                        "cost.tail_slippage_multiplier",
                    )
                    - 1
                )
            )
        )
        one_way = cost_multiplier * (
            _finite(execution_cost["fee_rate"], "cost.fee_rate")
            + profile["fee_rate"]
            + _finite(execution_cost["taker_fee_rate"], "cost.taker_fee_rate")
            + profile["taker_fee_rate"]
            + _finite(execution_cost["spread_rate"], "cost.spread_rate")
            + profile["spread_rate"]
            + cost_impact_rate * math.sqrt(participation)
            + expected_slippage
        )
        horizon_hours = self.expected_horizon_seconds // 3600
        funding = (
            math.fsum(float(rate) for rate in arrays["funding_rate"][window, selected])
            * horizon_hours
            / lookback_hours
        )
        borrow = (
            profile["borrow_rate"]
            * horizon_hours
            / (365 * 24)
            * _finite(
                execution_cost["borrow_rate_multiplier"],
                "cost.borrow_rate_multiplier",
            )
        )
        if _finite(arrays["cash_rate"][decision_index], "cash_rate") != 0.0:
            raise ValueError("fixed cost recipe requires explicit zero cash interest")
        assumptions = AllocationAssumptions(
            expected_simple_return=0.0,
            return_variance=0.0,
            buy_cost=one_way,
            sell_cost=one_way,
            exit_cost=one_way,
            funding_return=funding,
            borrow_return=borrow,
            cash_return=0.0,
        )
        if (
            not bool(profile["borrow_available"])
            or float(profile["borrow_rate"]) != 0.0
            or float(arrays["cash_rate"][decision_index]) != 0.0
        ):
            raise ValueError("fixed recipe requires available zero-rate borrow/cash")
        source_identity = hashlib.sha256(
            canonical_json_bytes(
                {
                    "dataset_id": self.content.dataset_id,
                    "execution_economics": execution_identity,
                    "recipe": recipe,
                    "symbol": symbols[selected],
                    "decision_time": time_ns,
                    "horizon_hours": horizon_hours,
                    "profile_origin": "caller_declared_simulation_v1",
                }
            )
        ).hexdigest()
        return assumptions, source_identity

    def _cost_estimates_payload(
        self,
        source: OwnedDecisionSnapshot,
        assumptions: AllocationAssumptions,
        source_identity: str,
    ) -> dict[str, object]:
        """Recreate the declared native estimate payload from owned raw inputs."""
        decision = source.decision_time_ns
        return {
            "schema": "horizon_cost_estimates_v1",
            "valuation_basis": "same_close_price_return",
            "cost_basis": "declared_initial_notional_rates",
            "symbol": source.symbol,
            "decision_time": decision,
            "available_at": decision,
            "horizon_end": decision + source.expected_horizon_seconds * 1_000_000_000,
            "source_identity": source_identity,
            "buy_cost": assumptions.buy_cost,
            "sell_cost": assumptions.sell_cost,
            "exit_cost": assumptions.exit_cost,
            "funding_return": assumptions.funding_return,
            "borrow_return": assumptions.borrow_return,
            "cash_return": assumptions.cash_return,
        }

    def _cost_recipe_declarations(self) -> dict[str, object]:
        """Recreate tier_d_native.cost_rows declarations from frozen inputs."""
        protocol = self.binding.protocol_document
        cost_scenario = (
            "BASEL0" if self.binding.scenario == "BASEL2" else self.binding.scenario
        )
        profile = _effective_profile(
            cost_scenario,
            self.window.actor_advances,
            cast(dict[str, Any], protocol["policy"]["risk"]),
        )
        bounds = protocol["funding"]["coverage_bounds"]
        timestamps = self.content.arrays["timestamps"]
        if timestamps.size < 2:
            raise ValueError("cost recipe requires Dataset coverage timestamps")
        return {
            "execution_cost": profile["cost_policy"],
            "expected_execution_policy_digest": profile["cost_policy_digest"],
            "reference_notional": protocol["economics"]["reference_notional"],
            "assumptions_identity": self.binding.assumptions_identity,
            "assumptions_available_at": int(timestamps[0].astype("int64")),
            "fee_assumption": (
                "Declared Task18 generic fee .0005; overlay0; stress multiplier2."
            ),
            "spread_assumption": (
                "Declared Task18 full spread .0002; overlay0; stress multiplier2."
            ),
            "impact_assumption": (
                "Declared square-root participation coefficient .0001, not calibrated."
            ),
            "slippage_assumption": (
                "Deterministic zero stochastic slippage declaration."
            ),
            "borrow_assumption": (
                "Borrow available; zero declared borrow rate; multiplier1."
            ),
            "funding_source_identity": (
                f"Declared bar-close-rate proxy:{self.content.dataset_id}"
            ),
            "funding_coverage_start": int(timestamps[0].astype("int64")),
            "funding_coverage_stop": int(timestamps[-1].astype("int64")),
            "funding_lookback_hours": bounds["lookback_hours"],
            "funding_minimum_event_count": bounds["minimum_counted_events"],
            "funding_maximum_counted_bar_close_gap_hours": float(
                bounds["maximum_counted_close_gap_hours"]
            ),
            "funding_maximum_counted_bar_close_age_hours": float(
                bounds["maximum_counted_close_age_hours"]
            ),
            "cash_zero_rationale": (
                "Nominal USDT quote cash assumed zero yield, USDT FX parity1."
            ),
        }

    def _raw_execution_economics(self) -> _ExecutionEconomics:
        raw = self.content.dataset.get("identity_payload_json")
        if type(raw) is not str:
            raise ValueError("pinned Dataset lacks execution-economics identity bytes")
        identity = _json_object(raw.encode("utf-8"))
        profile = identity.get("execution_economics")
        required = {
            "schema_version",
            "name",
            "fee_rate",
            "maker_fee_rate",
            "taker_fee_rate",
            "spread_rate",
            "max_participation_rate",
            "borrow_available",
            "borrow_rate",
        }
        if type(profile) is not dict or set(profile) != required:
            raise ValueError("execution-economics identity is not a closed profile")
        profile_values = cast(dict[str, object], profile)
        if (
            profile_values["schema_version"] != "execution_economics_profile_v1"
            or identity.get("schema") != "market_dataset_identity_v7"
            or type(profile_values["name"]) is not str
            or not cast(str, profile_values["name"]).strip()
            or type(profile_values["borrow_available"]) is not bool
        ):
            raise ValueError("execution-economics identity metadata is invalid")
        result: _ExecutionEconomics = {
            "fee_rate": _finite(
                profile_values["fee_rate"], "execution_economics.fee_rate", minimum=0.0
            ),
            "maker_fee_rate": _finite(
                profile_values["maker_fee_rate"],
                "execution_economics.maker_fee_rate",
                minimum=0.0,
            ),
            "taker_fee_rate": _finite(
                profile_values["taker_fee_rate"],
                "execution_economics.taker_fee_rate",
                minimum=0.0,
            ),
            "spread_rate": _finite(
                profile_values["spread_rate"],
                "execution_economics.spread_rate",
                minimum=0.0,
            ),
            "max_participation_rate": 0.0,
            "borrow_available": cast(bool, profile_values["borrow_available"]),
            "borrow_rate": _finite(
                profile_values["borrow_rate"],
                "execution_economics.borrow_rate",
                minimum=0.0,
            ),
        }
        participation_rate = _finite(
            profile_values["max_participation_rate"],
            "execution_economics.max_participation_rate",
            minimum=0.0,
        )
        if not 0.0 < participation_rate <= 1.0:
            raise ValueError("execution-economics participation is outside (0, 1]")
        if not result["borrow_available"] and result["borrow_rate"] != 0.0:
            raise ValueError("unavailable borrow cannot declare a nonzero rate")
        if not result["borrow_available"] or result["borrow_rate"] != 0.0:
            raise ValueError("fixed cost recipe requires available zero-rate borrow")
        if result["fee_rate"] > 0.0 and (
            result["maker_fee_rate"] > 0.0 or result["taker_fee_rate"] > 0.0
        ):
            raise ValueError("generic fee cannot combine with maker or taker fees")
        result["max_participation_rate"] = participation_rate
        result["max_participation_rate"] = participation_rate
        return result

    def _reserve_target_identity(self, closing_index: int) -> str:
        if type(closing_index) is not int or closing_index < 0:
            raise ValueError("reserve closing index must be non-negative")
        decision_digest = hashlib.sha256(
            canonical_json_bytes(
                {"plan": self.reserve_plan_digest, "closing_index": closing_index}
            )
        ).hexdigest()
        return hashlib.sha256(
            canonical_json_bytes({"decision": decision_digest, "target": [0.0, 0.0]})
        ).hexdigest()

    def completion(self) -> dict[str, object]:
        """Return an explicit non-authoritative status; never a B4 PASS."""
        finished = self._current_index == self.window.terminal_stop
        flat = self._account.exact_quantities == (Fraction(0), Fraction(0))
        no_pending = not self._active
        return {
            "status": "comparison_window_consumed"
            if finished and flat and no_pending
            else "incomplete",
            "full_b4_authorized": False,
            "source_scope": self.source_scope,
            "processed_actor_advances": self.actor_advances,
            "expected_actor_advances": self.window.actor_advances,
            "processed_reserve_advances": self.reserve_advances,
            "expected_reserve_advances": self.window.reserve_advances,
            "terminal_flat": flat,
            "no_active_orders": no_pending,
            "economic_termination": self._terminated_early,
        }


__all__ = [
    "FrozenMarketBinding",
    "FrozenMarketPolicy",
    "FrozenMarketReplay",
    "FrozenMarketWindow",
    "ObservedFill",
    "OwnedFill",
    "ReplayAdvance",
]
