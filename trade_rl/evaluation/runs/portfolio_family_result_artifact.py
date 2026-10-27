"""Strict write-once result artifact for fixed four-family portfolio attempts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from trade_rl._validation import require_git_sha, require_sha256
from trade_rl.artifacts.atomic_write import atomic_rename_directory, atomic_write_bytes
from trade_rl.artifacts.canonical import (
    canonical_json_bytes,
    freeze_json_value,
    to_json_value,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import open_regular_binary, read_verified_bytes
from trade_rl.data.market_order_rules import MarketOrderProfile
from trade_rl.evaluation.runs.portfolio_family_comparison import (
    FourFamilyPortfolioReplayResult,
    PortfolioFamilyReplayOutcome,
)
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig, ExecutionRuleStress
from trade_rl.simulation.objectives.binding import BoundObjectiveClock
from trade_rl.strategies.forecasts.ridge_artifact import (
    RidgeForecastArtifactBinding,
)
from trade_rl.strategies.rules.trend import TrendIntentConfig

_SCHEMA = "four_family_portfolio_result_v1"
_DIGEST_DOMAIN = b"trade_rl.four_family_portfolio_result_v1\0"
_IDENTITY_SCHEMA = "four_family_portfolio_attempt_identity_v1"
_ROW_SCHEMA = "four_family_portfolio_family_row_v1"
_FAMILY_NAMES = ("cash", "fixed_trend", "ridge_forecast", "joint_ppo")
_ATTEMPT_FILES = frozenset(
    {
        "started/record.json",
        "provenance/before/record.json",
        "provenance/after/record.json",
        "result/manifest.json",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_ATTEMPT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", re.ASCII)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty canonical string")
    return value


def _number(value: object, field: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a non-boolean number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{field} must be finite{' and positive' if positive else ''}")
    return result


@dataclass(frozen=True, slots=True)
class FourFamilyPortfolioAttemptIdentity:
    """Immutable externally declared identity for one fixed four-family attempt."""

    attempt_id: str
    research_context_digest: str
    source_repository: str
    source_commit: str
    source_tree: str
    expected_implementation_digest: str
    expected_runtime_environment_digest: str
    expected_dataset_artifact_sha256: str
    expected_symbols: tuple[str, ...]
    ridge_artifact_sha256: str
    expected_ridge_binding: RidgeForecastArtifactBinding
    expected_bundle_sha256: str
    expected_manifest_sha256: str
    expected_completion_sha256: str
    expected_training_context_sha256: str
    expected_checkpoint_sha256: str
    bound: BoundObjectiveClock
    start_index: int
    stop_index: int
    initial_capital: float
    target_abs_weights: tuple[float, ...]
    trend_config: TrendIntentConfig
    ridge_entry_threshold: float
    ridge_exit_threshold: float
    ridge_one_way_switch_cost: float | None
    execution_cost: ExecutionCostConfig
    risk_config: PreTradeRiskConfig
    market_order_profile: MarketOrderProfile | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.attempt_id, str)
            or _ATTEMPT_ID.fullmatch(self.attempt_id) is None
        ):
            raise ValueError("attempt_id must be a canonical identifier")
        for field, value in (
            ("research_context_digest", self.research_context_digest),
            ("expected_implementation_digest", self.expected_implementation_digest),
            (
                "expected_runtime_environment_digest",
                self.expected_runtime_environment_digest,
            ),
            ("expected_dataset_artifact_sha256", self.expected_dataset_artifact_sha256),
            ("ridge_artifact_sha256", self.ridge_artifact_sha256),
            ("expected_bundle_sha256", self.expected_bundle_sha256),
            ("expected_manifest_sha256", self.expected_manifest_sha256),
            ("expected_completion_sha256", self.expected_completion_sha256),
            ("expected_training_context_sha256", self.expected_training_context_sha256),
            ("expected_checkpoint_sha256", self.expected_checkpoint_sha256),
        ):
            require_sha256(value, field=field)
        require_git_sha(self.source_commit, field="source_commit")
        require_git_sha(self.source_tree, field="source_tree")
        repository = _string(self.source_repository, "source_repository")
        if repository.count("/") != 1 or any(
            part in {"", ".", ".."} for part in repository.split("/")
        ):
            raise ValueError("source_repository must use owner/repository form")
        if type(self.expected_symbols) is not tuple or not self.expected_symbols:
            raise TypeError("expected_symbols must be a non-empty tuple")
        symbols = tuple(
            _string(item, "expected_symbols[]") for item in self.expected_symbols
        )
        if len(set(symbols)) != len(symbols):
            raise ValueError("expected_symbols must be unique")
        if type(self.expected_ridge_binding) is not RidgeForecastArtifactBinding:
            raise TypeError(
                "expected_ridge_binding must be a RidgeForecastArtifactBinding"
            )
        if (
            self.expected_ridge_binding.dataset_artifact_sha256
            != self.expected_dataset_artifact_sha256
        ):
            raise ValueError(
                "Ridge binding Dataset digest differs from its external pin"
            )
        if type(self.bound) is not BoundObjectiveClock:
            raise TypeError("bound must be a BoundObjectiveClock")
        if type(self.trend_config) is not TrendIntentConfig:
            raise TypeError("trend_config must be a TrendIntentConfig")
        if type(self.execution_cost) is not ExecutionCostConfig:
            raise TypeError("execution_cost must be an ExecutionCostConfig")
        if type(self.risk_config) is not PreTradeRiskConfig:
            raise TypeError("risk_config must be a PreTradeRiskConfig")
        if (
            self.market_order_profile is not None
            and type(self.market_order_profile) is not MarketOrderProfile
        ):
            raise TypeError("market_order_profile must be a MarketOrderProfile or None")
        if (
            type(self.start_index) is not int
            or type(self.stop_index) is not int
            or self.start_index < 0
            or self.stop_index <= self.start_index
        ):
            raise ValueError("range must satisfy integer 0 <= start_index < stop_index")
        capital = _number(self.initial_capital, "initial_capital", positive=True)
        if capital != self.bound.objective.capital.total_initial_capital:
            raise ValueError("initial_capital differs from the bound objective")
        if type(self.target_abs_weights) is not tuple or len(
            self.target_abs_weights
        ) != len(symbols):
            raise ValueError("target_abs_weights must match expected_symbols")
        weights = tuple(
            _number(item, "target_abs_weights[]", positive=True)
            for item in self.target_abs_weights
        )
        if any(weight > 1.0 for weight in weights):
            raise ValueError("target_abs_weights must be within (0, 1]")
        entry = _number(self.ridge_entry_threshold, "ridge_entry_threshold")
        exit_ = _number(self.ridge_exit_threshold, "ridge_exit_threshold")
        if entry < 0.0 or exit_ < 0.0 or exit_ > entry:
            raise ValueError("Ridge thresholds must satisfy 0 <= exit <= entry")
        if self.ridge_one_way_switch_cost is not None:
            switch_cost = _number(
                self.ridge_one_way_switch_cost, "ridge_one_way_switch_cost"
            )
            if switch_cost < 0.0:
                raise ValueError("ridge_one_way_switch_cost must be non-negative")
            object.__setattr__(self, "ridge_one_way_switch_cost", switch_cost)
        if self.trend_config.signal_index < 0:
            raise ValueError("trend signal index must be non-negative")
        object.__setattr__(self, "expected_symbols", symbols)
        object.__setattr__(self, "initial_capital", capital)
        object.__setattr__(self, "target_abs_weights", weights)
        object.__setattr__(self, "ridge_entry_threshold", entry)
        object.__setattr__(self, "ridge_exit_threshold", exit_)


def four_family_portfolio_attempt_identity_payload(
    identity: FourFamilyPortfolioAttemptIdentity,
) -> dict[str, object]:
    """Return the canonical identity mapping; filesystem roots are excluded."""
    if type(identity) is not FourFamilyPortfolioAttemptIdentity:
        raise TypeError("identity must be a FourFamilyPortfolioAttemptIdentity")
    profile: object = None
    if identity.market_order_profile is not None:
        profile = {
            "declaration": identity.market_order_profile.canonical_payload(),
            "digest": identity.market_order_profile.digest,
        }
    return {
        "attempt_id": identity.attempt_id,
        "bound": identity.bound.payload(),
        "expected_bundle_sha256": identity.expected_bundle_sha256,
        "expected_checkpoint_sha256": identity.expected_checkpoint_sha256,
        "expected_completion_sha256": identity.expected_completion_sha256,
        "expected_dataset_artifact_sha256": identity.expected_dataset_artifact_sha256,
        "expected_implementation_digest": identity.expected_implementation_digest,
        "expected_manifest_sha256": identity.expected_manifest_sha256,
        "expected_ridge_binding": to_json_value(identity.expected_ridge_binding),
        "expected_runtime_environment_digest": identity.expected_runtime_environment_digest,
        "expected_symbols": list(identity.expected_symbols),
        "expected_training_context_sha256": identity.expected_training_context_sha256,
        "execution_cost": to_json_value(identity.execution_cost),
        "initial_capital": identity.initial_capital,
        "market_order_profile": profile,
        "research_context_digest": identity.research_context_digest,
        "ridge_artifact_sha256": identity.ridge_artifact_sha256,
        "ridge_entry_threshold": identity.ridge_entry_threshold,
        "ridge_exit_threshold": identity.ridge_exit_threshold,
        "ridge_one_way_switch_cost": identity.ridge_one_way_switch_cost,
        "risk_config": to_json_value(identity.risk_config),
        "source_commit": identity.source_commit,
        "source_repository": identity.source_repository,
        "source_tree": identity.source_tree,
        "start_index": identity.start_index,
        "stop_index": identity.stop_index,
        "target_abs_weights": list(identity.target_abs_weights),
        "trend_config": to_json_value(identity.trend_config),
        "schema_version": _IDENTITY_SCHEMA,
    }


def four_family_portfolio_attempt_identity_digest(
    identity: FourFamilyPortfolioAttemptIdentity,
) -> str:
    return content_digest(four_family_portfolio_attempt_identity_payload(identity))


def _identity_payload_matches_expected(
    persisted: object, expected: Mapping[str, object]
) -> bool:
    """Match current identities or the exact v1 optional-cost representation."""
    if not isinstance(persisted, dict) or set(persisted) != set(expected):
        return False
    if canonical_json_bytes(persisted) == canonical_json_bytes(expected):
        return True

    persisted_cost = persisted.get("ridge_one_way_switch_cost")
    expected_cost = expected.get("ridge_one_way_switch_cost")
    if (
        type(persisted_cost) is not int
        or persisted_cost < 0
        or type(expected_cost) is not float
        or persisted_cost != expected_cost
    ):
        return False
    legacy_expected = dict(expected)
    legacy_expected["ridge_one_way_switch_cost"] = persisted_cost
    return canonical_json_bytes(persisted) == canonical_json_bytes(legacy_expected)


@dataclass(frozen=True, slots=True)
class LoadedFourFamilyPortfolioResultArtifact:
    root: Path
    artifact_digest: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        require_sha256(self.artifact_digest, field="artifact_digest")
        frozen = freeze_json_value(self.payload)
        if not isinstance(frozen, Mapping):
            raise TypeError("result payload must be an object")
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "payload", cast(Mapping[str, object], frozen))


def _strict_json(raw: bytes, *, field: str) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{field} contains a duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{field} contains non-finite JSON number {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not valid UTF-8 JSON") from error
    if canonical_json_bytes(value) != raw:
        raise ValueError(f"{field} is not canonical JSON")
    return value


def _read_file(path: Path, *, field: str) -> bytes:
    if path.is_symlink():
        raise ValueError(f"{field} must not be a symlink")
    with open_regular_binary(path, field=field) as handle:
        return handle.read()


def _require_object(value: object, *, field: str, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{field} field closure mismatch")
    return cast(dict[str, object], value)


def _require_int(value: object, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be boolean")
    return value


def _require_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _require_array(
    value: object, field: str, *, length: int | None = None
) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    if length is not None and len(value) != length:
        raise ValueError(f"{field} length differs from expected roster")
    return value


def _validate_numeric_array(value: object, field: str, *, length: int) -> None:
    for index, item in enumerate(_require_array(value, field, length=length)):
        _require_number(item, f"{field}[{index}]")


def _validate_string_array(
    value: object, field: str, *, length: int | None = None
) -> None:
    for index, item in enumerate(_require_array(value, field, length=length)):
        if not isinstance(item, str):
            raise ValueError(f"{field}[{index}] must be a string")


def _validate_pairs(value: object, field: str, *, numeric_value: bool) -> None:
    for index, pair_value in enumerate(_require_array(value, field)):
        pair = _require_array(pair_value, f"{field}[{index}]", length=2)
        if not isinstance(pair[0], str):
            raise ValueError(f"{field}[{index}][0] must be a string")
        if numeric_value:
            _require_number(pair[1], f"{field}[{index}][1]")
        elif not isinstance(pair[1], str):
            raise ValueError(f"{field}[{index}][1] must be a string")


def _expected_execution_policy_digest(
    identity: FourFamilyPortfolioAttemptIdentity,
) -> str:
    """Derive the native policy digest declared by the fixed replay identity."""
    digest = identity.execution_cost.execution_policy_digest
    profile = identity.market_order_profile
    if profile is None:
        return digest
    return content_digest(
        {
            "schema_version": "profile_market_execution_v1",
            "base_policy_digest": digest,
            "market_order_profile_digest": profile.digest,
            "rule_stress": ExecutionRuleStress().digest_payload(),
        }
    )


def _validate_ledger_shape(
    value: object, *, identity: FourFamilyPortfolioAttemptIdentity
) -> dict[str, object]:
    ledger = _require_object(
        value,
        field="family.ledger",
        keys={
            "active_order_remainders",
            "dataset_id",
            "execution_policy_digest",
            "final_borrow_cost",
            "final_cash",
            "final_funding_pnl",
            "final_max_drawdown",
            "final_portfolio_value",
            "final_total_cost",
            "final_turnover_total",
            "intervals",
            "portfolio_decisions",
            "schema_version",
            "start_index",
            "stop_index",
            "terminal_exact_quantities",
            "terminal_order_reasons",
            "termination_reason",
        },
    )
    if ledger["schema_version"] != "shared_cash_portfolio_replay_ledger_v1":
        raise ValueError("family ledger schema is unsupported")
    if ledger["dataset_id"] != identity.expected_ridge_binding.dataset_id:
        raise ValueError("family ledger Dataset identity differs from expected binding")
    execution_policy_digest = ledger["execution_policy_digest"]
    if not isinstance(execution_policy_digest, str):
        raise ValueError("execution_policy_digest must be a string")
    require_sha256(execution_policy_digest, field="execution_policy_digest")
    if execution_policy_digest != _expected_execution_policy_digest(identity):
        raise ValueError("ledger execution policy digest differs from attempt identity")
    if (ledger["start_index"], ledger["stop_index"]) != (
        identity.start_index,
        identity.stop_index,
    ):
        raise ValueError("family ledger range differs from attempt identity")
    intervals = ledger["intervals"]
    decisions = ledger["portfolio_decisions"]
    if not isinstance(intervals, list) or not isinstance(decisions, list):
        raise ValueError(
            "family ledger intervals and portfolio_decisions must be arrays"
        )
    if len(intervals) > identity.stop_index - identity.start_index:
        raise ValueError("family ledger contains intervals beyond the fixed window")
    if len(decisions) > identity.stop_index - identity.start_index:
        raise ValueError("family decisions exceed the fixed window")
    from fractions import Fraction

    terminal_quantities = _require_array(
        ledger["terminal_exact_quantities"],
        "terminal_exact_quantities",
        length=len(identity.expected_symbols),
    )
    for quantity in terminal_quantities:
        if not isinstance(quantity, str):
            raise ValueError("terminal exact quantities must be strings")
        try:
            parsed = Fraction(quantity)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError("terminal exact quantity is invalid") from error
        if str(parsed) != quantity:
            raise ValueError("terminal exact quantity is not canonical")
    _validate_pairs(
        ledger["active_order_remainders"],
        "active_order_remainders",
        numeric_value=True,
    )
    _validate_pairs(
        ledger["terminal_order_reasons"],
        "terminal_order_reasons",
        numeric_value=False,
    )
    if ledger["termination_reason"] is not None and not isinstance(
        ledger["termination_reason"], str
    ):
        raise ValueError("termination_reason must be a string or null")
    for name in (
        "final_borrow_cost",
        "final_cash",
        "final_funding_pnl",
        "final_max_drawdown",
        "final_portfolio_value",
        "final_total_cost",
        "final_turnover_total",
    ):
        _require_number(ledger[name], f"ledger.{name}")
    for offset, interval_value in enumerate(intervals):
        interval = _require_object(
            interval_value,
            field=f"ledger.intervals[{offset}]",
            keys={
                "borrow_cost_after",
                "borrow_cost_before",
                "cash_after",
                "cash_before",
                "capacity_events",
                "exact_quantities_after",
                "exact_quantities_before",
                "funding_events",
                "funding_pnl_after",
                "funding_pnl_before",
                "interval_borrow_cost",
                "interval_cash_interest",
                "interval_cost",
                "interval_dividend",
                "interval_funding",
                "interval_net_return",
                "max_drawdown_after",
                "max_drawdown_before",
                "next_index",
                "order_events",
                "portfolio_value_after",
                "portfolio_value_before",
                "start_index",
                "termination_reason",
                "total_cost_after",
                "total_cost_before",
                "turnover_total_after",
                "turnover_total_before",
            },
        )
        start = _require_int(interval["start_index"], f"interval[{offset}].start_index")
        next_index = _require_int(
            interval["next_index"], f"interval[{offset}].next_index"
        )
        if not identity.start_index <= start < next_index <= identity.stop_index:
            raise ValueError("native interval falls outside the fixed comparison range")
        if start != identity.start_index + offset:
            raise ValueError(
                "native intervals must form a prefix from the requested replay index"
            )
        if next_index != start + 1:
            raise ValueError("native interval must advance exactly one native bar")
        for name in (
            "borrow_cost_after",
            "borrow_cost_before",
            "cash_after",
            "cash_before",
            "funding_pnl_after",
            "funding_pnl_before",
            "interval_borrow_cost",
            "interval_cash_interest",
            "interval_cost",
            "interval_dividend",
            "interval_funding",
            "interval_net_return",
            "max_drawdown_after",
            "max_drawdown_before",
            "portfolio_value_after",
            "portfolio_value_before",
            "total_cost_after",
            "total_cost_before",
            "turnover_total_after",
            "turnover_total_before",
        ):
            _require_number(interval[name], f"interval[{offset}].{name}")
        for name in ("exact_quantities_after", "exact_quantities_before"):
            _validate_string_array(
                interval[name],
                f"interval[{offset}].{name}",
                length=len(identity.expected_symbols),
            )
        for name in ("capacity_events", "funding_events", "order_events"):
            events = _require_array(interval[name], f"interval[{offset}].{name}")
            if any(not isinstance(item, dict) for item in events):
                raise ValueError(f"interval[{offset}].{name} entries must be objects")
        interval_reason = interval["termination_reason"]
        if interval_reason is not None and not isinstance(interval_reason, str):
            raise ValueError("interval termination_reason must be string or null")
        if offset < len(intervals) - 1 and interval_reason is not None:
            raise ValueError("only the final native interval may terminate")
        if (
            offset == len(intervals) - 1
            and interval_reason != ledger["termination_reason"]
        ):
            raise ValueError("final native interval termination differs from ledger")
    for offset, decision_value in enumerate(decisions):
        decision = _require_object(
            decision_value,
            field=f"portfolio_decisions[{offset}]",
            keys={
                "effective_hold_symbols",
                "index",
                "proposal_weights",
                "requested_hold_symbols",
                "risk_reasons",
                "target_weights",
            },
        )
        _require_int(decision["index"], f"portfolio_decisions[{offset}].index")
        for name in ("proposal_weights", "target_weights"):
            _validate_numeric_array(
                decision[name],
                f"portfolio_decisions[{offset}].{name}",
                length=len(identity.expected_symbols),
            )
        for name in ("requested_hold_symbols", "effective_hold_symbols"):
            indexes = _require_array(
                decision[name], f"portfolio_decisions[{offset}].{name}"
            )
            if any(
                type(item) is not int or not 0 <= item < len(identity.expected_symbols)
                for item in indexes
            ):
                raise ValueError(f"portfolio_decisions[{offset}].{name} is invalid")
        _validate_string_array(
            decision["risk_reasons"], f"portfolio_decisions[{offset}].risk_reasons"
        )
    # This boundary validates envelope, range, and native-owned top-level shape.
    # Nested native event mappings are preserved as canonical JSON, not reinterpreted.
    return ledger


def _family_payload(outcome: PortfolioFamilyReplayOutcome) -> dict[str, object]:
    if type(outcome) is not PortfolioFamilyReplayOutcome:
        raise TypeError("family outcome must be a PortfolioFamilyReplayOutcome")
    replay = outcome.replay
    ledger = replay.ledger_evidence
    if ledger is None:
        raise ValueError("every family must retain the native ledger evidence")
    book = replay.book
    contract_multipliers = book.contract_multipliers
    if contract_multipliers is None:
        raise ValueError("family book must retain contract multipliers")
    payload: dict[str, object] = {
        "actual_interval_count": outcome.actual_interval_count,
        "book": {
            "borrow_cost": book.borrow_cost,
            "cash": book.cash,
            "contract_multipliers": contract_multipliers.tolist(),
            "exact_quantities": [str(item) for item in book.exact_quantities],
            "fill_count": book.fill_count,
            "funding_pnl": book.funding_pnl,
            "insolvent": book.insolvent,
            "maintenance_margin": book.maintenance_margin,
            "maintenance_requirement": book.maintenance_requirement,
            "margin_deficit": book.margin_deficit,
            "margin_used": book.margin_used,
            "mark_prices": book.mark_prices.tolist(),
            "max_drawdown": book.max_drawdown,
            "peak_value": book.peak_value,
            "quantities": book.quantities.tolist(),
            "rebalance_events": book.rebalance_events,
            "returns_history": list(book.returns_history),
            "termination_reason": (
                None
                if book.termination_reason is None
                else str(book.termination_reason)
            ),
            "total_cost": book.total_cost,
            "turnover_total": book.turnover_total,
        },
        "diagnostics": replay.diagnostics.digest_payload(),
        "expected_interval_count": outcome.expected_interval_count,
        "family": outcome.family,
        "insolvent": outcome.insolvent,
        "ledger": ledger.to_mapping(),
        "no_active_orders": outcome.no_active_orders,
        "returns": {
            "elapsed_years": replay.returns.elapsed_years,
            "kind": replay.returns.kind.value,
            "periods_per_year": replay.returns.periods_per_year,
            "values": list(replay.returns.values),
        },
        "schema_version": _ROW_SCHEMA,
        "terminal_flat": outcome.terminal_flat,
        "termination_reason": outcome.termination_reason,
        "window_complete": outcome.window_complete,
        "decisions": [decision.to_mapping() for decision in replay.decisions],
    }
    return cast(dict[str, object], json.loads(canonical_json_bytes(payload)))


def _validate_family_payload(
    value: object, *, expected_family: str, identity: FourFamilyPortfolioAttemptIdentity
) -> dict[str, object]:
    family = _require_object(
        value,
        field="family row",
        keys={
            "actual_interval_count",
            "book",
            "decisions",
            "diagnostics",
            "expected_interval_count",
            "family",
            "insolvent",
            "ledger",
            "no_active_orders",
            "returns",
            "schema_version",
            "terminal_flat",
            "termination_reason",
            "window_complete",
        },
    )
    if family["schema_version"] != _ROW_SCHEMA or family["family"] != expected_family:
        raise ValueError("family row schema/order mismatch")
    if (
        _require_int(family["expected_interval_count"], "expected_interval_count")
        != identity.stop_index - identity.start_index
    ):
        raise ValueError("family expected interval count differs from identity")
    actual = _require_int(family["actual_interval_count"], "actual_interval_count")
    expected = _require_int(
        family["expected_interval_count"], "expected_interval_count"
    )
    if actual > expected:
        raise ValueError("family actual interval count exceeds expected count")
    for name in ("insolvent", "terminal_flat", "no_active_orders", "window_complete"):
        _require_bool(family[name], name)
    reason = family["termination_reason"]
    if reason is not None and not isinstance(reason, str):
        raise ValueError("family termination_reason must be string or null")
    ledger = _validate_ledger_shape(family["ledger"], identity=identity)
    ledger_intervals = ledger["intervals"]
    if len(cast(list[object], ledger_intervals)) != actual:
        raise ValueError("family interval count differs from native ledger")
    terminal_quantities = _require_array(
        ledger["terminal_exact_quantities"], "terminal_exact_quantities"
    )
    if family["terminal_flat"] != all(
        quantity == "0" for quantity in terminal_quantities
    ):
        raise ValueError("family terminal flat flag differs from native ledger")
    active_orders = _require_array(
        ledger["active_order_remainders"], "active_order_remainders"
    )
    if family["no_active_orders"] != (len(active_orders) == 0):
        raise ValueError("family pending-order flag differs from native ledger")
    if family["termination_reason"] != ledger["termination_reason"]:
        raise ValueError("family termination reason differs from native ledger")
    if family["window_complete"] != (
        actual == expected and reason is None and family["insolvent"] is False
    ):
        raise ValueError("family completeness flag differs from native outcome")
    for field in ("decisions",):
        if not isinstance(family[field], list):
            raise ValueError(f"family {field} must be an array")
    symbol_count = len(identity.expected_symbols)
    book = _require_object(
        family["book"],
        field="family.book",
        keys={
            "borrow_cost",
            "cash",
            "contract_multipliers",
            "exact_quantities",
            "fill_count",
            "funding_pnl",
            "insolvent",
            "maintenance_margin",
            "maintenance_requirement",
            "margin_deficit",
            "margin_used",
            "mark_prices",
            "max_drawdown",
            "peak_value",
            "quantities",
            "rebalance_events",
            "returns_history",
            "termination_reason",
            "total_cost",
            "turnover_total",
        },
    )
    if _require_bool(book["insolvent"], "book.insolvent") != family["insolvent"]:
        raise ValueError("book insolvency flag differs from family outcome")
    for name in ("contract_multipliers", "mark_prices", "quantities"):
        _validate_numeric_array(book[name], f"family.book.{name}", length=symbol_count)
    _validate_string_array(
        book["exact_quantities"], "family.book.exact_quantities", length=symbol_count
    )
    _validate_numeric_array(
        book["returns_history"],
        "family.book.returns_history",
        length=actual,
    )
    for name in (
        "borrow_cost",
        "cash",
        "funding_pnl",
        "maintenance_margin",
        "maintenance_requirement",
        "margin_deficit",
        "margin_used",
        "max_drawdown",
        "peak_value",
        "total_cost",
        "turnover_total",
    ):
        _require_number(book[name], f"family.book.{name}")
    for name in ("fill_count", "rebalance_events"):
        _require_int(book[name], f"family.book.{name}")
    if book["termination_reason"] is not None and not isinstance(
        book["termination_reason"], str
    ):
        raise ValueError("family book termination_reason must be string or null")
    returns = _require_object(
        family["returns"],
        field="family.returns",
        keys={"elapsed_years", "kind", "periods_per_year", "values"},
    )
    if returns["kind"] not in {"base_bar", "decision_step"}:
        raise ValueError("family return kind is unsupported")
    _require_int(returns["periods_per_year"], "returns.periods_per_year", minimum=1)
    if returns["elapsed_years"] is not None:
        _require_number(returns["elapsed_years"], "returns.elapsed_years")
    return_values = _require_array(returns["values"], "returns.values")
    if len(return_values) != actual:
        raise ValueError("return count differs from native interval count")
    for offset, item in enumerate(return_values):
        _require_number(item, f"returns.values[{offset}]")
    decisions = cast(list[object], family["decisions"])
    for offset, decision_value in enumerate(decisions):
        decision = _require_object(
            decision_value,
            field=f"family.decisions[{offset}]",
            keys={
                "effective_hold_symbols",
                "index",
                "proposal_weights",
                "requested_hold_symbols",
                "risk_reasons",
                "target_weights",
            },
        )
        _require_int(decision["index"], f"family.decisions[{offset}].index")
        for name in ("proposal_weights", "target_weights"):
            _validate_numeric_array(
                decision[name],
                f"family.decisions[{offset}].{name}",
                length=symbol_count,
            )
        for name in ("requested_hold_symbols", "effective_hold_symbols"):
            indexes = _require_array(
                decision[name], f"family.decisions[{offset}].{name}"
            )
            if any(
                type(item) is not int or not 0 <= item < symbol_count
                for item in indexes
            ):
                raise ValueError(f"family.decisions[{offset}].{name} is invalid")
        _validate_string_array(
            decision["risk_reasons"], f"family.decisions[{offset}].risk_reasons"
        )
    diagnostics = _require_object(
        family["diagnostics"],
        field="family.diagnostics",
        keys={
            "borrow_cost",
            "closed_trades",
            "funding_pnl",
            "n_trades",
            "rebalance_events",
            "termination_reasons",
            "total_cost",
            "turnover_total",
        },
    )
    for name in ("borrow_cost", "funding_pnl", "total_cost", "turnover_total"):
        _require_number(diagnostics[name], f"diagnostics.{name}")
    for name in ("n_trades", "rebalance_events"):
        _require_int(diagnostics[name], f"diagnostics.{name}")
    _validate_string_array(
        diagnostics["termination_reasons"], "diagnostics.termination_reasons"
    )
    expected_termination_reasons = [] if reason is None else [reason]
    if diagnostics["termination_reasons"] != expected_termination_reasons:
        raise ValueError("diagnostic termination reasons differ from family outcome")
    if not isinstance(diagnostics["closed_trades"], dict):
        raise ValueError("diagnostics.closed_trades must be an object")
    return family


def _validate_provenance_record(
    root: Path,
    position: str,
    *,
    identity: FourFamilyPortfolioAttemptIdentity,
    expected_digest: object,
) -> dict[str, object]:
    raw = _read_file(
        root / "provenance" / position / "record.json",
        field=f"{position} provenance record",
    )
    record = _require_object(
        _strict_json(raw, field=f"{position} provenance record"),
        field=f"{position} provenance record",
        keys={"provenance", "schema_version"},
    )
    if record["schema_version"] != "four_family_portfolio_provenance_v1":
        raise ValueError("unsupported provenance record schema")
    provenance = _require_object(
        record["provenance"],
        field=f"{position} provenance",
        keys={
            "implementation",
            "implementation_digest",
            "research_context_digest",
            "runtime_environment",
            "runtime_environment_digest",
            "schema_version",
        },
    )
    if content_digest(provenance) != expected_digest:
        raise ValueError(f"{position} provenance digest differs from result manifest")
    if provenance["schema_version"] != "candidate_run_provenance_v1":
        raise ValueError("candidate provenance schema is unsupported")
    implementation = _require_object(
        provenance["implementation"],
        field=f"{position} provenance implementation",
        keys={"files", "schema_version"},
    )
    runtime = _require_object(
        provenance["runtime_environment"],
        field=f"{position} provenance runtime",
        keys={"machine", "os", "packages", "python", "schema_version"},
    )
    if provenance["implementation_digest"] != content_digest(
        implementation
    ) or provenance["runtime_environment_digest"] != content_digest(runtime):
        raise ValueError(f"{position} provenance internal digest mismatch")
    if (
        provenance["implementation_digest"] != identity.expected_implementation_digest
        or provenance["runtime_environment_digest"]
        != identity.expected_runtime_environment_digest
        or provenance["research_context_digest"] != identity.research_context_digest
    ):
        raise ValueError(f"{position} provenance differs from expected pins")
    return provenance


def _publish_record(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"attempt record already exists: {path}")
    path.parent.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{path.parent.name}.staging-",
            dir=path.parent.parent,
        )
    )
    try:
        atomic_write_bytes(staging / path.name, raw)
        atomic_rename_directory(staging, path.parent)
    except BaseException:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink(missing_ok=True)
            staging.rmdir()
        raise


def _attempt_file_inventory(root: Path) -> set[str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("attempt root must be a regular directory")
    found: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("attempt artifact tree must not contain symlinks")
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            allowed_directories = {
                "families",
                "provenance",
                "result",
                "started",
                "complete",
                "failure",
                "provenance/before",
                "provenance/after",
                *(f"families/{name}" for name in _FAMILY_NAMES),
            }
            if relative not in allowed_directories:
                raise ValueError(
                    "attempt artifact tree contains an unexpected directory"
                )
            continue
        if not path.is_file():
            raise ValueError("attempt artifact tree contains a non-regular entry")
        found.add(relative)
    return found


def _validate_started_record(
    raw: bytes, *, expected_identity: Mapping[str, object]
) -> None:
    started = _require_object(
        _strict_json(raw, field="started record"),
        field="started record",
        keys={"declared_roots", "identity", "schema_version"},
    )
    started_identity = _require_object(
        started["identity"],
        field="started.identity",
        keys=set(expected_identity),
    )
    if canonical_json_bytes(started_identity) != canonical_json_bytes(
        expected_identity
    ):
        raise ValueError("started record identity differs from result")
    if started["schema_version"] != "four_family_portfolio_attempt_started_v1":
        raise ValueError("started record schema is unsupported")
    declared_roots = _require_object(
        started["declared_roots"],
        field="started.declared_roots",
        keys={"bundle_root", "dataset_artifact_root", "ridge_artifact_root"},
    )
    if any(
        not isinstance(value, str) or not value for value in declared_roots.values()
    ):
        raise ValueError("started record roots must be non-empty strings")


def _validate_premanifest_inventory(root: Path) -> None:
    result_directory = root / "result"
    if result_directory.exists() or result_directory.is_symlink():
        raise FileExistsError("result directory already exists before publication")
    expected = set(_ATTEMPT_FILES) - {"result/manifest.json"}
    expected.update(f"families/{name}/record.json" for name in _FAMILY_NAMES)
    if _attempt_file_inventory(root) != expected:
        raise ValueError("attempt files differ from pre-result inventory")


def publish_four_family_portfolio_result_artifact(
    attempt_root: str | Path,
    result: FourFamilyPortfolioReplayResult,
    identity: FourFamilyPortfolioAttemptIdentity,
    *,
    before_provenance: Mapping[str, object],
    after_provenance: Mapping[str, object],
) -> str:
    """Publish a canonical manifest after the immediate family rows are durable."""
    if type(result) is not FourFamilyPortfolioReplayResult:
        raise TypeError("result must be a FourFamilyPortfolioReplayResult")
    if type(identity) is not FourFamilyPortfolioAttemptIdentity:
        raise TypeError("identity must be a FourFamilyPortfolioAttemptIdentity")
    root = Path(attempt_root)
    identity_payload = four_family_portfolio_attempt_identity_payload(identity)
    _validate_premanifest_inventory(root)
    if (
        result.dataset_id != identity.expected_ridge_binding.dataset_id
        or result.dataset_artifact.artifact_digest
        != identity.expected_dataset_artifact_sha256
        or result.dataset_artifact.schema_version
        != identity.expected_ridge_binding.dataset_schema_version
        or tuple(result.symbols) != identity.expected_symbols
        or (result.start_index, result.stop_index)
        != (identity.start_index, identity.stop_index)
        or result.ridge_artifact_digest != identity.ridge_artifact_sha256
        or result.ppo_bundle_sha256 != identity.expected_bundle_sha256
    ):
        raise ValueError("comparison result differs from the declared attempt identity")
    names = tuple(item.family for item in result.families)
    if names != _FAMILY_NAMES:
        raise ValueError("result must contain all four fixed families in order")
    started_path = root / "started" / "record.json"
    started_raw = _read_file(started_path, field="started record")
    _validate_started_record(started_raw, expected_identity=identity_payload)
    before_digest = content_digest(before_provenance)
    after_digest = content_digest(after_provenance)
    before_record = _validate_provenance_record(
        root,
        "before",
        identity=identity,
        expected_digest=before_digest,
    )
    after_record = _validate_provenance_record(
        root,
        "after",
        identity=identity,
        expected_digest=after_digest,
    )
    if canonical_json_bytes(before_record) != canonical_json_bytes(
        before_provenance
    ) or canonical_json_bytes(after_record) != canonical_json_bytes(after_provenance):
        raise ValueError("provenance records differ from supplied provenance")
    if canonical_json_bytes(before_record) != canonical_json_bytes(after_record):
        raise ValueError("before/after provenance records differ")
    rows: list[dict[str, object]] = []
    for outcome, family in zip(result.families, _FAMILY_NAMES, strict=True):
        family_path = root / "families" / family / "record.json"
        raw = _read_file(family_path, field=f"{family} family record")
        row = _validate_family_payload(
            _strict_json(raw, field=f"{family} family record"),
            expected_family=family,
            identity=identity,
        )
        if row != _family_payload(outcome):
            raise ValueError("durable family row differs from returned native outcome")
        rows.append(
            {
                "family": family,
                "path": f"families/{family}/record.json",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    before_digest = content_digest(before_record)
    after_digest = content_digest(after_record)
    content: dict[str, object] = {
        "after_provenance_digest": after_digest,
        "before_provenance_digest": before_digest,
        "dataset": {
            "artifact_digest": result.dataset_artifact.artifact_digest,
            "dataset_id": result.dataset_id,
            "schema_version": result.dataset_artifact.schema_version,
            "symbols": list(result.symbols),
            "start_index": result.start_index,
            "stop_index": result.stop_index,
        },
        "families": rows,
        "identity": identity_payload,
        "identity_digest": four_family_portfolio_attempt_identity_digest(identity),
        "schema_version": _SCHEMA,
        "started_record": {
            "path": "started/record.json",
            "sha256": hashlib.sha256(started_raw).hexdigest(),
            "size_bytes": len(started_raw),
        },
    }
    result_digest = hashlib.sha256(
        _DIGEST_DOMAIN + canonical_json_bytes(content)
    ).hexdigest()
    manifest = {"artifact_digest": result_digest, "payload": content}
    _publish_record(root / "result" / "manifest.json", canonical_json_bytes(manifest))
    return result_digest


def load_four_family_portfolio_result_artifact(
    attempt_root: str | Path,
    *,
    expected_digest: str,
    expected_identity: FourFamilyPortfolioAttemptIdentity,
) -> LoadedFourFamilyPortfolioResultArtifact:
    """Strictly reload the result before completion; no complete marker is needed."""
    require_sha256(expected_digest, field="expected_digest")
    root = Path(attempt_root)
    inventory = _attempt_file_inventory(root)
    allowed = set(_ATTEMPT_FILES)
    allowed.update(f"families/{name}/record.json" for name in _FAMILY_NAMES)
    allowed.update({"complete/record.json", "failure/record.json"})
    if not set(_ATTEMPT_FILES).issubset(inventory) or inventory - allowed:
        raise ValueError("attempt tree has missing or unexpected result files")
    if "failure/record.json" in inventory:
        raise ValueError("failed attempts cannot be loaded as result artifacts")
    for name in _FAMILY_NAMES:
        if f"families/{name}/record.json" not in inventory:
            raise ValueError("attempt is missing a completed family row")
    manifest_raw = _read_file(
        root / "result" / "manifest.json", field="result manifest"
    )
    envelope = _require_object(
        _strict_json(manifest_raw, field="result manifest"),
        field="result manifest",
        keys={"artifact_digest", "payload"},
    )
    digest = envelope["artifact_digest"]
    if not isinstance(digest, str):
        raise ValueError("artifact_digest must be a string")
    require_sha256(digest, field="artifact_digest")
    payload = _require_object(
        envelope["payload"],
        field="result payload",
        keys={
            "after_provenance_digest",
            "before_provenance_digest",
            "dataset",
            "families",
            "identity",
            "identity_digest",
            "schema_version",
            "started_record",
        },
    )
    if (
        digest != expected_digest
        or digest
        != hashlib.sha256(_DIGEST_DOMAIN + canonical_json_bytes(payload)).hexdigest()
    ):
        raise ValueError("result artifact digest mismatch")
    if payload["schema_version"] != _SCHEMA:
        raise ValueError("unsupported four-family result schema")
    identity_payload = four_family_portfolio_attempt_identity_payload(expected_identity)
    persisted_identity = _require_object(
        payload["identity"], field="result identity", keys=set(identity_payload)
    )
    if not _identity_payload_matches_expected(persisted_identity, identity_payload):
        raise ValueError("result identity differs from the expected identity")
    identity_digest = content_digest(persisted_identity)
    if payload["identity_digest"] != identity_digest:
        raise ValueError("result identity digest mismatch")
    started_ref = _require_object(
        payload["started_record"],
        field="started_record",
        keys={"path", "sha256", "size_bytes"},
    )
    if started_ref["path"] != "started/record.json":
        raise ValueError("started record path is unsupported")
    started_raw = _read_file(root / "started" / "record.json", field="started record")
    if started_ref["sha256"] != hashlib.sha256(started_raw).hexdigest() or started_ref[
        "size_bytes"
    ] != len(started_raw):
        raise ValueError("started record bytes differ from result manifest")
    _validate_started_record(started_raw, expected_identity=persisted_identity)
    before = _validate_provenance_record(
        root,
        "before",
        identity=expected_identity,
        expected_digest=payload["before_provenance_digest"],
    )
    after = _validate_provenance_record(
        root,
        "after",
        identity=expected_identity,
        expected_digest=payload["after_provenance_digest"],
    )
    if canonical_json_bytes(before) != canonical_json_bytes(after):
        raise ValueError("before/after provenance records differ")
    rows = payload["families"]
    if not isinstance(rows, list) or len(rows) != len(_FAMILY_NAMES):
        raise ValueError("result must list exactly four family records")
    for row_ref, family in zip(rows, _FAMILY_NAMES, strict=True):
        reference = _require_object(
            row_ref,
            field="family reference",
            keys={"family", "path", "sha256", "size_bytes"},
        )
        if (
            reference["family"] != family
            or reference["path"] != f"families/{family}/record.json"
        ):
            raise ValueError("family reference order or path is invalid")
        raw = read_verified_bytes(
            root / cast(str, reference["path"]),
            expected_digest=cast(str, reference["sha256"]),
            expected_size_bytes=_require_int(reference["size_bytes"], "family size"),
            field=f"{family} family record",
        )
        _validate_family_payload(
            _strict_json(raw, field=f"{family} family record"),
            expected_family=family,
            identity=expected_identity,
        )
    dataset = _require_object(
        payload["dataset"],
        field="result.dataset",
        keys={
            "artifact_digest",
            "dataset_id",
            "schema_version",
            "symbols",
            "start_index",
            "stop_index",
        },
    )
    if (
        dataset["artifact_digest"] != expected_identity.expected_dataset_artifact_sha256
        or dataset["dataset_id"] != expected_identity.expected_ridge_binding.dataset_id
        or dataset["schema_version"]
        != expected_identity.expected_ridge_binding.dataset_schema_version
        or dataset["symbols"] != list(expected_identity.expected_symbols)
        or dataset["start_index"] != expected_identity.start_index
        or dataset["stop_index"] != expected_identity.stop_index
    ):
        raise ValueError("result Dataset identity differs from expected identity")
    for name in ("before_provenance_digest", "after_provenance_digest"):
        value = payload[name]
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        require_sha256(value, field=name)
    return LoadedFourFamilyPortfolioResultArtifact(
        root=root / "result",
        artifact_digest=digest,
        payload=payload,
    )


__all__ = [
    "FourFamilyPortfolioAttemptIdentity",
    "LoadedFourFamilyPortfolioResultArtifact",
    "four_family_portfolio_attempt_identity_digest",
    "four_family_portfolio_attempt_identity_payload",
    "load_four_family_portfolio_result_artifact",
    "publish_four_family_portfolio_result_artifact",
]
