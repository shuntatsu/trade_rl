"""Immutable, ledger-complete Ridge shared-cash comparison artifacts."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from trade_rl._validation import require_sha256
from trade_rl.artifacts.atomic_write import atomic_rename_directory, atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes, freeze_json_value
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import open_regular_binary
from trade_rl.data.market_order_rules import MarketOrderProfile
from trade_rl.evaluation.comparison.strategies import SharedCashStrategyComparisonEntry
from trade_rl.evaluation.evidence import ExecutionDiagnostics
from trade_rl.evaluation.metrics import PerformanceMetrics, evaluate_performance
from trade_rl.evaluation.robustness.closed_trades import ClosedTradeDiagnostics
from trade_rl.evaluation.runs.provenance import PROVENANCE_SCHEMA
from trade_rl.evaluation.runs.ridge_artifact_comparison import (
    RidgeArtifactComparison,
    RidgeArtifactComparisonConfig,
)
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.simulation.diagnostics.funding import FundingBoundaryEvidence
from trade_rl.simulation.execution import ExecutionCostConfig, ExecutionRuleStress
from trade_rl.simulation.orders.model import OrderEvent
from trade_rl.strategies.forecasts import RidgeForecastArtifactBinding
from trade_rl.strategies.position_intent import PositionIntent

_SCHEMA = "ridge_shared_cash_comparison_v1"
_DIGEST_DOMAIN = b"trade_rl.ridge_shared_cash_comparison_artifact_v1\0"
_FILENAME = "comparison.json"
_ARM_NAMES = (
    "cash",
    "legacy_hysteresis",
    "position_aware_zero_cost",
    "position_aware_cost",
)
_PROVENANCE_FIELDS = {
    "schema_version",
    "implementation",
    "implementation_digest",
    "runtime_environment",
    "runtime_environment_digest",
    "research_context_digest",
}


@dataclass(frozen=True, slots=True)
class LoadedRidgeComparisonArtifact:
    """Deeply immutable validated content from one comparison artifact."""

    root: Path
    artifact_digest: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        require_sha256(self.artifact_digest, field="artifact_digest")
        frozen = freeze_json_value(self.payload)
        if not isinstance(frozen, Mapping):
            raise TypeError("comparison payload must be an object")
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "payload", cast(Mapping[str, object], frozen))


def _object(
    value: object, *, field: str, keys: set[str] | None = None
) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a JSON object")
    if keys is not None and set(value) != keys:
        raise ValueError(f"{field} field closure mismatch")
    return cast(dict[str, object], value)


def _sequence(value: object, *, field: str) -> list[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be an array")
    return list(value)


def _string(value: object, *, field: str, non_empty: bool = True) -> str:
    if not isinstance(value, str) or (non_empty and not value):
        raise ValueError(f"{field} must be a string")
    return value


def _integer(value: object, *, field: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _close(actual: float, expected: float, *, field: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"{field} is inconsistent")


def _binding_payload(binding: RidgeForecastArtifactBinding) -> dict[str, object]:
    return cast(dict[str, object], json.loads(canonical_json_bytes(binding)))


def _profile_declaration(profile: MarketOrderProfile | None) -> object:
    if profile is None:
        return None
    return {"digest": profile.digest, "declaration": profile.canonical_payload()}


def _config_payload(config: RidgeArtifactComparisonConfig) -> dict[str, object]:
    return {
        "entry_threshold": config.entry_threshold,
        "execution_cost": asdict(config.execution_cost),
        "exit_threshold": config.exit_threshold,
        "gross_budget": config.gross_budget,
        "initial_capital": config.initial_capital,
        "market_order_profile": _profile_declaration(config.market_order_profile),
        "minimum_hold_bars": config.minimum_hold_bars,
        "one_way_switch_cost": config.one_way_switch_cost,
        "risk": asdict(config.risk),
        "start_index": config.start_index,
        "stop_index": config.stop_index,
    }


def _return_payload(returns: ReturnSeries) -> dict[str, object]:
    return {
        "elapsed_years": returns.elapsed_years,
        "kind": returns.kind.value,
        "periods_per_year": returns.periods_per_year,
        "values": list(returns.values),
    }


def _metrics_payload(metrics: PerformanceMetrics) -> dict[str, object]:
    return {
        "borrow_cost": metrics.borrow_cost,
        "funding_pnl": metrics.funding_pnl,
        "max_drawdown": metrics.max_drawdown,
        "n_periods": metrics.n_periods,
        "n_trades": metrics.n_trades,
        "periods_per_year": metrics.periods_per_year,
        "rebalance_events": metrics.rebalance_events,
        "return_kind": metrics.return_kind.value,
        "sharpe": metrics.sharpe,
        "sortino": metrics.sortino,
        "termination_count": metrics.termination_count,
        "total_cost": metrics.total_cost,
        "total_return": metrics.total_return,
        "turnover_total": metrics.turnover_total,
    }


def _comparison_payload(
    comparison: RidgeArtifactComparison,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    if type(comparison) is not RidgeArtifactComparison:
        raise TypeError("comparison must be a RidgeArtifactComparison")
    if type(comparison.config) is not RidgeArtifactComparisonConfig:
        raise TypeError("comparison config is invalid")
    if type(comparison.binding) is not RidgeForecastArtifactBinding:
        raise TypeError("comparison binding is invalid")
    if (
        len(comparison.entries) != len(_ARM_NAMES)
        or tuple(entry.name for entry in comparison.entries) != _ARM_NAMES
    ):
        raise ValueError("comparison must contain the fixed four arms in order")
    if type(comparison.dataset_id) is not str or not comparison.dataset_id:
        raise ValueError("comparison Dataset identity is invalid")
    profile = comparison.config.market_order_profile
    if profile is not None and (
        profile.dataset_id != comparison.dataset_id
        or profile.dataset_symbols != comparison.symbols
    ):
        raise ValueError("declared market profile differs from comparison Dataset")

    arms: list[dict[str, object]] = []
    for entry in comparison.entries:
        if type(entry) is not SharedCashStrategyComparisonEntry:
            raise TypeError("comparison entry is invalid")
        if entry.replay.ledger_evidence is None:
            raise ValueError("all comparison arms require full ledger evidence")
        arms.append(
            {
                "diagnostics": entry.replay.diagnostics.digest_payload(),
                "ledger": entry.replay.ledger_evidence.to_mapping(),
                "metrics": _metrics_payload(entry.metrics),
                "name": entry.name,
                "returns": _return_payload(entry.replay.returns),
            }
        )
    dataset = comparison.dataset_artifact
    return {
        "arms": arms,
        "config": _config_payload(comparison.config),
        "dataset": {
            "artifact_digest": dataset.artifact_digest,
            "dataset_id": comparison.dataset_id,
            "evaluation_start_index": comparison.config.start_index,
            "evaluation_start_ns": comparison.evaluation_start_ns,
            "evaluation_stop_index": comparison.config.stop_index,
            "evaluation_stop_ns": comparison.evaluation_stop_ns,
            "schema_version": dataset.schema_version,
            "symbols": list(comparison.symbols),
        },
        "model": {
            "artifact_digest": comparison.ridge_artifact_digest,
            "binding": _binding_payload(comparison.binding),
        },
        "provenance": dict(provenance),
        "schema_version": _SCHEMA,
    }


def _validate_provenance(value: object) -> dict[str, object]:
    provenance = _object(value, field="provenance", keys=_PROVENANCE_FIELDS)
    if provenance["schema_version"] != PROVENANCE_SCHEMA:
        raise ValueError("unsupported comparison provenance schema")
    implementation = _object(
        provenance["implementation"],
        field="provenance.implementation",
        keys={"schema_version", "files"},
    )
    if implementation["schema_version"] != "candidate_run_implementation_v1":
        raise ValueError("unsupported implementation manifest")
    files = _sequence(implementation["files"], field="implementation.files")
    paths: list[str] = []
    for index, item in enumerate(files):
        entry = _object(
            item, field=f"implementation.files[{index}]", keys={"path", "sha256"}
        )
        path = _string(entry["path"], field="implementation path")
        if (
            path.startswith("/")
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise ValueError("implementation path must be a normalized relative path")
        require_sha256(
            _string(entry["sha256"], field="implementation digest"),
            field="implementation digest",
        )
        paths.append(path)
    if not paths or paths != sorted(set(paths)):
        raise ValueError("implementation files must be sorted and unique")
    if provenance["implementation_digest"] != content_digest(implementation):
        raise ValueError("implementation provenance digest mismatch")

    runtime = _object(
        provenance["runtime_environment"],
        field="provenance.runtime_environment",
        keys={"schema_version", "python", "os", "machine", "packages"},
    )
    if runtime["schema_version"] != "candidate_run_runtime_v1":
        raise ValueError("unsupported runtime manifest")
    python = _object(
        runtime["python"], field="runtime.python", keys={"implementation", "version"}
    )
    _string(python["implementation"], field="runtime.python.implementation")
    _string(python["version"], field="runtime.python.version")
    operating_system = _object(
        runtime["os"], field="runtime.os", keys={"family", "release"}
    )
    _string(operating_system["family"], field="runtime.os.family")
    _string(operating_system["release"], field="runtime.os.release")
    _string(runtime["machine"], field="runtime.machine")
    packages = _object(
        runtime["packages"],
        field="runtime.packages",
        keys={
            "trade-rl",
            "numpy",
            "gymnasium",
            "lightgbm",
            "stable-baselines3",
            "torch",
        },
    )
    for name, version in packages.items():
        if version is not None:
            _string(version, field=f"runtime.packages.{name}")
    if provenance["runtime_environment_digest"] != content_digest(runtime):
        raise ValueError("runtime provenance digest mismatch")
    context = provenance["research_context_digest"]
    if context is not None:
        require_sha256(
            _string(context, field="research_context_digest"),
            field="research_context_digest",
        )
    return provenance


def _validate_binding(
    value: object,
    *,
    dataset_id: str,
    dataset_schema: str,
    dataset_digest: str,
    symbols: tuple[str, ...],
) -> RidgeForecastArtifactBinding:
    expected = {field.name for field in fields(RidgeForecastArtifactBinding)}
    raw = _object(value, field="model.binding", keys=expected)
    for name in (
        "fit_symbol_indices",
        "fit_symbol_names",
        "feature_names",
        "feature_indices",
    ):
        _sequence(raw[name], field=f"model.binding.{name}")
    binding = RidgeForecastArtifactBinding(
        dataset_id=_string(raw["dataset_id"], field="binding.dataset_id"),
        dataset_schema_version=_string(
            raw["dataset_schema_version"], field="binding.dataset_schema_version"
        ),
        dataset_artifact_sha256=_string(
            raw["dataset_artifact_sha256"], field="binding.dataset_artifact_sha256"
        ),
        fit_symbol_indices=tuple(
            _integer(item, field="fit_symbol_indices[]", minimum=0)
            for item in cast(list[object], raw["fit_symbol_indices"])
        ),
        fit_symbol_names=tuple(
            _string(item, field="fit_symbol_names[]")
            for item in cast(list[object], raw["fit_symbol_names"])
        ),
        feature_names=tuple(
            _string(item, field="feature_names[]")
            for item in cast(list[object], raw["feature_names"])
        ),
        feature_indices=tuple(
            _integer(item, field="feature_indices[]", minimum=0)
            for item in cast(list[object], raw["feature_indices"])
        ),
        fit_cutoff_ns=_integer(raw["fit_cutoff_ns"], field="fit_cutoff_ns"),
        horizon_hours=_integer(raw["horizon_hours"], field="horizon_hours"),
        alpha_hex=_string(raw["alpha_hex"], field="alpha_hex"),
        n_samples=_integer(raw["n_samples"], field="n_samples"),
        vintage_id=_string(raw["vintage_id"], field="vintage_id"),
        source_commit=_string(raw["source_commit"], field="source_commit"),
        implementation_sha256=_string(
            raw["implementation_sha256"], field="implementation_sha256"
        ),
        runtime_sha256=_string(raw["runtime_sha256"], field="runtime_sha256"),
    )
    if (
        binding.dataset_id,
        binding.dataset_schema_version,
        binding.dataset_artifact_sha256,
    ) != (
        dataset_id,
        dataset_schema,
        dataset_digest,
    ):
        raise ValueError("Dataset identity differs from model binding")
    if any(index >= len(symbols) for index in binding.fit_symbol_indices):
        raise ValueError("model fit symbol index is outside ordered Dataset symbols")
    if (
        tuple(symbols[index] for index in binding.fit_symbol_indices)
        != binding.fit_symbol_names
    ):
        raise ValueError("model fit symbol names differ from ordered Dataset symbols")
    return binding


def _validate_profile(
    value: object, *, dataset_id: str, symbols: tuple[str, ...]
) -> dict[str, object] | None:
    if value is None:
        return None
    profile = _object(
        value, field="config.market_order_profile", keys={"digest", "declaration"}
    )
    digest = _string(profile["digest"], field="market profile digest")
    require_sha256(digest, field="market profile digest")
    declaration = _object(
        profile["declaration"],
        field="market profile declaration",
        keys={
            "schema",
            "venue",
            "historical_application",
            "dataset_minimum_role",
            "reversal_mode",
            "dataset_id",
            "dataset_symbols",
            "rules",
            "source_uri",
            "source_sha256",
            "retrieved_at",
            "account_mode",
            "reduce_only_exits",
        },
    )
    if (
        declaration["schema"],
        declaration["venue"],
        declaration["historical_application"],
        declaration["dataset_minimum_role"],
        declaration["reversal_mode"],
    ) != (
        "market_order_profile_v1",
        "binance_usdm_perpetual",
        "current_snapshot_assumption",
        "venue_constraint",
        "ordinary",
    ):
        raise ValueError("unsupported market profile declaration")
    if (
        declaration["dataset_id"] != dataset_id
        or tuple(
            _string(item, field="profile.dataset_symbols[]")
            for item in _sequence(
                declaration["dataset_symbols"], field="profile.dataset_symbols"
            )
        )
        != symbols
    ):
        raise ValueError("market profile Dataset identity mismatch")
    _string(declaration["source_uri"], field="profile.source_uri")
    require_sha256(
        _string(declaration["source_sha256"], field="profile.source_sha256"),
        field="profile.source_sha256",
    )
    if (
        declaration["account_mode"] != "one_way"
        or type(declaration["reduce_only_exits"]) is not bool
    ):
        raise ValueError("market profile account declaration is invalid")
    retrieved_at = _string(declaration["retrieved_at"], field="profile.retrieved_at")
    try:
        parsed_time = datetime.fromisoformat(retrieved_at)
    except ValueError as error:
        raise ValueError("market profile retrieval time is invalid") from error
    if parsed_time.utcoffset() is None or parsed_time.isoformat() != retrieved_at:
        raise ValueError("market profile retrieval time must be canonical and aware")
    rules = _sequence(declaration["rules"], field="profile.rules")
    indices: list[int] = []
    rule_keys = {
        "symbol_index",
        "symbol",
        "lot_size",
        "minimum_quantity",
        "maximum_quantity",
        "minimum_notional",
    }
    for index, item in enumerate(rules):
        rule = _object(item, field=f"profile.rules[{index}]", keys=rule_keys)
        symbol_index = _integer(
            rule["symbol_index"], field="profile.rule.symbol_index", minimum=0
        )
        if symbol_index >= len(symbols) or rule["symbol"] != symbols[symbol_index]:
            raise ValueError("market profile rule does not match Dataset symbol order")
        indices.append(symbol_index)
        for name in (
            "lot_size",
            "minimum_quantity",
            "maximum_quantity",
            "minimum_notional",
        ):
            _number(rule[name], field=f"profile.rule.{name}")
        lot_size = cast(float, rule["lot_size"])
        minimum_quantity = cast(float, rule["minimum_quantity"])
        maximum_quantity = cast(float, rule["maximum_quantity"])
        minimum_notional = cast(float, rule["minimum_notional"])
        if (
            lot_size <= 0.0
            or maximum_quantity <= 0.0
            or minimum_quantity < 0.0
            or minimum_notional < 0.0
            or math.ceil(minimum_quantity / lot_size) * lot_size > maximum_quantity
        ):
            raise ValueError("market profile quantity rule is invalid")
    if not rules or indices != sorted(set(indices)):
        raise ValueError("market profile rules must be non-empty and ordered")
    if sha256(canonical_json_bytes(declaration)).hexdigest() != digest:
        raise ValueError("market profile declaration digest mismatch")
    return declaration


def _validate_config(
    value: object,
    *,
    dataset_id: str,
    symbols: tuple[str, ...],
) -> tuple[dict[str, object], str]:
    config = _object(
        value,
        field="config",
        keys={
            "entry_threshold",
            "execution_cost",
            "exit_threshold",
            "gross_budget",
            "initial_capital",
            "market_order_profile",
            "minimum_hold_bars",
            "one_way_switch_cost",
            "risk",
            "start_index",
            "stop_index",
        },
    )
    cost_raw = _object(
        config["execution_cost"],
        field="config.execution_cost",
        keys={field.name for field in fields(ExecutionCostConfig)},
    )
    cost_values = dict(cost_raw)
    fractions = _sequence(
        cost_values["trigger_volume_fractions"], field="trigger_volume_fractions"
    )
    if len(fractions) != 4:
        raise ValueError("trigger_volume_fractions must contain four rates")
    cost_values["trigger_volume_fractions"] = tuple(
        _number(item, field="trigger_volume_fractions[]") for item in fractions
    )
    for name in ("allow_short", "processing_bar_volume_capacity", "partial_fill_carry"):
        if type(cost_values[name]) is not bool:
            raise ValueError(f"execution_cost.{name} must be boolean")
    for name in ("random_seed", "order_latency_bars"):
        _integer(cost_values[name], field=f"execution_cost.{name}", minimum=0)
    for name in ("margin_mode", "order_type", "path_mode"):
        _string(cost_values[name], field=f"execution_cost.{name}")
    for name in set(cost_values) - {
        "trigger_volume_fractions",
        "allow_short",
        "processing_bar_volume_capacity",
        "partial_fill_carry",
        "random_seed",
        "order_latency_bars",
        "margin_mode",
        "order_type",
        "path_mode",
    }:
        _number(cost_values[name], field=f"execution_cost.{name}")
    cost = ExecutionCostConfig(**cast(Any, cost_values))

    risk_raw = _object(
        config["risk"],
        field="config.risk",
        keys={field.name for field in fields(PreTradeRiskConfig)},
    )
    risk_values = dict(risk_raw)
    for name in (
        "max_gross",
        "max_abs_weight",
        "drawdown_start",
        "drawdown_stop",
        "fail_closed_tolerance",
    ):
        _number(risk_values[name], field=f"risk.{name}")
    if risk_values["max_turnover"] is not None:
        _number(risk_values["max_turnover"], field="risk.max_turnover")
    if type(risk_values["emergency_turnover_override"]) is not bool:
        raise ValueError("risk.emergency_turnover_override must be boolean")
    risk = PreTradeRiskConfig(**cast(Any, risk_values))

    profile = _validate_profile(
        config["market_order_profile"], dataset_id=dataset_id, symbols=symbols
    )
    start = _integer(config["start_index"], field="config.start_index", minimum=0)
    stop = _integer(config["stop_index"], field="config.stop_index", minimum=1)
    minimum_hold = _integer(
        config["minimum_hold_bars"], field="config.minimum_hold_bars", minimum=0
    )
    validated = RidgeArtifactComparisonConfig(
        start_index=start,
        stop_index=stop,
        gross_budget=_number(config["gross_budget"], field="config.gross_budget"),
        initial_capital=_number(
            config["initial_capital"], field="config.initial_capital"
        ),
        entry_threshold=_number(
            config["entry_threshold"], field="config.entry_threshold"
        ),
        exit_threshold=_number(config["exit_threshold"], field="config.exit_threshold"),
        one_way_switch_cost=_number(
            config["one_way_switch_cost"], field="config.one_way_switch_cost"
        ),
        execution_cost=cost,
        risk=risk,
        minimum_hold_bars=minimum_hold,
    )
    if profile is None:
        execution_digest = cost.execution_policy_digest
    else:
        execution_digest = content_digest(
            {
                "schema_version": "profile_market_execution_v1",
                "base_policy_digest": cost.execution_policy_digest,
                "market_order_profile_digest": cast(
                    str,
                    _object(config["market_order_profile"], field="profile")["digest"],
                ),
                "rule_stress": ExecutionRuleStress().digest_payload(),
            }
        )
    return _config_payload(validated) | {
        "market_order_profile": config["market_order_profile"]
    }, execution_digest


def _validate_dataset_model(
    payload: dict[str, object],
) -> tuple[dict[str, object], RidgeForecastArtifactBinding, tuple[str, ...]]:
    dataset = _object(
        payload["dataset"],
        field="dataset",
        keys={
            "artifact_digest",
            "dataset_id",
            "evaluation_start_index",
            "evaluation_start_ns",
            "evaluation_stop_index",
            "evaluation_stop_ns",
            "schema_version",
            "symbols",
        },
    )
    dataset_id = _string(dataset["dataset_id"], field="dataset.dataset_id")
    require_sha256(dataset_id, field="dataset.dataset_id")
    dataset_digest = _string(
        dataset["artifact_digest"], field="dataset.artifact_digest"
    )
    require_sha256(dataset_digest, field="dataset.artifact_digest")
    schema = _string(dataset["schema_version"], field="dataset.schema_version")
    symbols = tuple(
        _string(item, field="dataset.symbols[]")
        for item in _sequence(dataset["symbols"], field="dataset.symbols")
    )
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("Dataset symbols must be non-empty and unique")
    start = _integer(
        dataset["evaluation_start_index"], field="evaluation_start_index", minimum=0
    )
    stop = _integer(
        dataset["evaluation_stop_index"], field="evaluation_stop_index", minimum=1
    )
    start_ns = _integer(dataset["evaluation_start_ns"], field="evaluation_start_ns")
    stop_ns = _integer(dataset["evaluation_stop_ns"], field="evaluation_stop_ns")
    minimum_ns = -(2**63) + 1
    maximum_ns = 2**63 - 1
    if (
        start >= stop
        or start_ns >= stop_ns
        or not minimum_ns <= start_ns <= maximum_ns
        or not minimum_ns <= stop_ns <= maximum_ns
    ):
        raise ValueError("Dataset evaluation range is invalid")
    model = _object(
        payload["model"], field="model", keys={"artifact_digest", "binding"}
    )
    model_digest = _string(model["artifact_digest"], field="model.artifact_digest")
    require_sha256(model_digest, field="model.artifact_digest")
    binding = _validate_binding(
        model["binding"],
        dataset_id=dataset_id,
        dataset_schema=schema,
        dataset_digest=dataset_digest,
        symbols=symbols,
    )
    config = _object(payload["config"], field="config")
    if (start, stop) != (config.get("start_index"), config.get("stop_index")):
        raise ValueError("Dataset boundary indices differ from comparison config")
    return dataset, binding, symbols


def _validate_fraction_vector(
    value: object, *, field: str, size: int
) -> tuple[Fraction, ...]:
    raw = _sequence(value, field=field)
    if len(raw) != size:
        raise ValueError(f"{field} length differs from Dataset symbols")
    results: list[Fraction] = []
    for item in raw:
        text = _string(item, field=f"{field}[]")
        try:
            parsed = Fraction(text)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(f"{field} contains an invalid exact quantity") from error
        if str(parsed) != text:
            raise ValueError(f"{field} quantities must use canonical fractions")
        results.append(parsed)
    return tuple(results)


def _validate_decision(
    value: object, *, index: int, symbols_count: int
) -> dict[str, object]:
    decision = _object(
        value,
        field="ledger decision",
        keys={
            "changed_intents",
            "effective_intents",
            "index",
            "intents",
            "minimum_hold_suppressed",
            "minimum_hold_unlocked",
            "position_age_bars_after",
            "position_age_bars_before",
            "position_quantity_after",
            "position_quantity_before",
            "proposal_weights",
            "risk_reasons",
            "target_weights",
        },
    )
    if _integer(decision["index"], field="decision.index") != index:
        raise ValueError("ledger decisions do not cover the requested range")
    for name in ("intents", "effective_intents"):
        values = _sequence(decision[name], field=f"decision.{name}")
        if len(values) != symbols_count:
            raise ValueError("decision intent roster differs from Dataset")
        for item in values:
            try:
                PositionIntent(_integer(item, field=f"decision.{name}[]"))
            except ValueError as error:
                raise ValueError("decision contains an unsupported intent") from error
    for name in ("changed_intents", "minimum_hold_suppressed", "minimum_hold_unlocked"):
        values = _sequence(decision[name], field=f"decision.{name}")
        if len(values) != symbols_count or any(
            type(item) is not bool for item in values
        ):
            raise ValueError(f"decision.{name} roster is invalid")
    for name in ("position_age_bars_before", "position_age_bars_after"):
        values = _sequence(decision[name], field=f"decision.{name}")
        if len(values) != symbols_count:
            raise ValueError(f"decision.{name} roster is invalid")
        for item in values:
            _integer(item, field=f"decision.{name}[]", minimum=0)
    for name in (
        "position_quantity_before",
        "position_quantity_after",
        "proposal_weights",
        "target_weights",
    ):
        values = _sequence(decision[name], field=f"decision.{name}")
        if len(values) != symbols_count:
            raise ValueError(f"decision.{name} roster is invalid")
        for item in values:
            _number(item, field=f"decision.{name}[]")
    reasons = _sequence(decision["risk_reasons"], field="decision.risk_reasons")
    for reason in reasons:
        _string(reason, field="decision.risk_reasons[]")
    return decision


def _validate_interval(
    value: object,
    *,
    index: int,
    dataset_id: str,
    policy_digest: str,
    symbols_count: int,
    start_ns: int,
    stop_ns: int,
) -> tuple[dict[str, object], tuple[Fraction, ...], tuple[Fraction, ...]]:
    interval = _object(
        value,
        field="ledger interval",
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
    if (
        _integer(interval["start_index"], field="interval.start_index"),
        _integer(interval["next_index"], field="interval.next_index"),
    ) != (index, index + 1):
        raise ValueError("ledger intervals are not contiguous")
    before_q = _validate_fraction_vector(
        interval["exact_quantities_before"],
        field="exact_quantities_before",
        size=symbols_count,
    )
    after_q = _validate_fraction_vector(
        interval["exact_quantities_after"],
        field="exact_quantities_after",
        size=symbols_count,
    )
    numeric_names = {
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
    }
    for name in numeric_names:
        _number(interval[name], field=f"interval.{name}")
    if interval["termination_reason"] is not None:
        raise ValueError("completed comparisons cannot contain early termination")
    if (
        cast(float, interval["total_cost_before"]) < 0
        or cast(float, interval["total_cost_after"]) < 0
    ):
        raise ValueError("ledger cumulative cost cannot be negative")
    if cast(float, interval["interval_cost"]) < 0.0:
        raise ValueError("ledger interval cost cannot be negative")
    if (
        cast(float, interval["borrow_cost_before"]) < 0
        or cast(float, interval["borrow_cost_after"]) < 0
    ):
        raise ValueError("ledger cumulative borrow cost cannot be negative")
    if cast(float, interval["interval_borrow_cost"]) < 0.0:
        raise ValueError("ledger interval borrow cost cannot be negative")
    if (
        cast(float, interval["turnover_total_before"]) < 0
        or cast(float, interval["turnover_total_after"]) < 0
    ):
        raise ValueError("ledger cumulative turnover cannot be negative")
    if (
        not 0.0
        <= cast(float, interval["max_drawdown_before"])
        <= cast(float, interval["max_drawdown_after"])
        <= 1.0
    ):
        raise ValueError("ledger maximum drawdown chain is invalid")
    if (
        cast(float, interval["portfolio_value_before"]) <= 0.0
        or cast(float, interval["portfolio_value_after"]) <= 0.0
    ):
        raise ValueError("ledger portfolio equity must stay positive")
    events = _sequence(interval["order_events"], field="interval.order_events")
    for event_value in events:
        event = _object(event_value, field="order event")
        parsed = OrderEvent.from_mapping(event)
        if (
            parsed.dataset_id != dataset_id
            or parsed.execution_policy_digest != policy_digest
        ):
            raise ValueError("order event identity differs from ledger")
        if (
            parsed.symbol_index >= symbols_count
            or not index <= parsed.processing_index <= index + 1
        ):
            raise ValueError("order event falls outside its ledger interval")
    capacities = _sequence(
        interval["capacity_events"], field="interval.capacity_events"
    )
    capacity_fields = {
        "processing_volume",
        "capacity_reference_price",
        "contract_multiplier",
        "participation_limit",
        "market_notional",
        "initial_capacity_notional",
        "consumed_capacity_notional",
        "remaining_capacity_notional",
    }
    for capacity_value in capacities:
        capacity = _object(capacity_value, field="capacity event", keys=capacity_fields)
        for name in capacity_fields:
            _number(capacity[name], field=f"capacity.{name}")
        if (
            cast(float, capacity["capacity_reference_price"]) <= 0
            or cast(float, capacity["contract_multiplier"]) <= 0
        ):
            raise ValueError("capacity reference price and multiplier must be positive")
        if not 0 <= cast(float, capacity["participation_limit"]) <= 1:
            raise ValueError("capacity participation limit is invalid")
    funding = _sequence(interval["funding_events"], field="interval.funding_events")
    for funding_value in funding:
        boundary = FundingBoundaryEvidence.from_mapping(
            _object(funding_value, field="funding event")
        )
        if (
            len(boundary.funding_due) != symbols_count
            or not index <= boundary.processing_index <= index + 1
            or not start_ns <= boundary.timestamp_ns <= stop_ns
        ):
            raise ValueError("funding boundary is outside comparison scope")
    return interval, before_q, after_q


def _validate_ledger(
    value: object,
    *,
    dataset_id: str,
    symbols: tuple[str, ...],
    start: int,
    stop: int,
    start_ns: int,
    stop_ns: int,
    initial_capital: float,
    expected_policy_digest: str,
    expected_decision_count: int,
) -> tuple[dict[str, object], tuple[float, ...]]:
    ledger = _object(
        value,
        field="ledger",
        keys={
            "active_order_remainders",
            "dataset_id",
            "decisions",
            "execution_policy_digest",
            "final_borrow_cost",
            "final_cash",
            "final_funding_pnl",
            "final_max_drawdown",
            "final_portfolio_value",
            "final_total_cost",
            "final_turnover_total",
            "intervals",
            "schema_version",
            "start_index",
            "stop_index",
            "terminal_exact_quantities",
            "terminal_order_reasons",
            "termination_reason",
        },
    )
    if ledger["schema_version"] != "shared_cash_replay_ledger_v2":
        raise ValueError("complete comparison requires ledger v2")
    if ledger["dataset_id"] != dataset_id:
        raise ValueError("ledger Dataset identity differs from comparison")
    policy = _string(ledger["execution_policy_digest"], field="execution_policy_digest")
    require_sha256(policy, field="execution_policy_digest")
    if policy != expected_policy_digest:
        raise ValueError("ledger execution policy differs from configuration")
    if (
        _integer(ledger["start_index"], field="ledger.start_index"),
        _integer(ledger["stop_index"], field="ledger.stop_index"),
    ) != (start, stop):
        raise ValueError("ledger range differs from comparison request")
    if ledger["termination_reason"] is not None:
        raise ValueError("terminated ledger cannot be published as complete")
    intervals = _sequence(ledger["intervals"], field="ledger.intervals")
    decisions_raw = _sequence(ledger["decisions"], field="ledger.decisions")
    if len(intervals) != stop - start or len(decisions_raw) != expected_decision_count:
        raise ValueError("ledger does not cover requested intervals and decisions")
    parsed_intervals: list[
        tuple[dict[str, object], tuple[Fraction, ...], tuple[Fraction, ...]]
    ] = []
    for offset, item in enumerate(intervals):
        parsed_intervals.append(
            _validate_interval(
                item,
                index=start + offset,
                dataset_id=dataset_id,
                policy_digest=policy,
                symbols_count=len(symbols),
                start_ns=start_ns,
                stop_ns=stop_ns,
            )
        )
    decisions = [
        _validate_decision(item, index=start + offset, symbols_count=len(symbols))
        for offset, item in enumerate(decisions_raw)
    ]
    zeroes = tuple(Fraction(0) for _ in symbols)
    previous: dict[str, object] | None = None
    returned: list[float] = []
    for offset, (interval, before_q, after_q) in enumerate(parsed_intervals):
        if before_q != (
            zeroes
            if previous is None
            else _validate_fraction_vector(
                previous["exact_quantities_after"],
                field="prior exact quantities",
                size=len(symbols),
            )
        ):
            raise ValueError("ledger exact-position chain is discontinuous")
        if previous is None:
            for field in ("cash_before", "portfolio_value_before"):
                _close(cast(float, interval[field]), initial_capital, field=field)
            for field in (
                "total_cost_before",
                "funding_pnl_before",
                "borrow_cost_before",
                "turnover_total_before",
                "max_drawdown_before",
            ):
                _close(cast(float, interval[field]), 0.0, field=field)
        else:
            for field in (
                "cash_before",
                "portfolio_value_before",
                "total_cost_before",
                "funding_pnl_before",
                "borrow_cost_before",
                "turnover_total_before",
                "max_drawdown_before",
            ):
                _close(
                    cast(float, interval[field]),
                    cast(float, previous[field.removesuffix("_before") + "_after"]),
                    field=field,
                )
        _close(
            cast(float, interval["total_cost_after"])
            - cast(float, interval["total_cost_before"]),
            cast(float, interval["interval_cost"]),
            field="interval cost chain",
        )
        _close(
            cast(float, interval["funding_pnl_after"])
            - cast(float, interval["funding_pnl_before"]),
            cast(float, interval["interval_funding"]),
            field="interval funding chain",
        )
        _close(
            cast(float, interval["borrow_cost_after"])
            - cast(float, interval["borrow_cost_before"]),
            cast(float, interval["interval_borrow_cost"]),
            field="interval borrow chain",
        )
        if cast(float, interval["turnover_total_after"]) + 1e-9 < cast(
            float, interval["turnover_total_before"]
        ):
            raise ValueError("ledger turnover must not decrease")
        if offset < len(decisions):
            decision = decisions[offset]
            before_values = _sequence(
                decision["position_quantity_before"],
                field="decision.position_quantity_before",
            )
            after_values = _sequence(
                decision["position_quantity_after"],
                field="decision.position_quantity_after",
            )
            for actual, exact in zip(before_values, before_q, strict=True):
                _close(
                    cast(float, actual),
                    float(exact),
                    field="decision pre-execution quantity",
                )
            for actual, exact in zip(after_values, after_q, strict=True):
                _close(
                    cast(float, actual),
                    float(exact),
                    field="decision post-execution quantity",
                )
        _close(
            cast(float, interval["portfolio_value_after"])
            / cast(float, interval["portfolio_value_before"])
            - 1.0,
            cast(float, interval["interval_net_return"]),
            field="interval return/equity",
        )
        returned.append(cast(float, interval["interval_net_return"]))
        previous = interval
    terminal = _validate_fraction_vector(
        ledger["terminal_exact_quantities"],
        field="terminal_exact_quantities",
        size=len(symbols),
    )
    if terminal != zeroes or parsed_intervals[-1][2] != zeroes:
        raise ValueError("comparison ledger must finish exactly flat")
    if _sequence(ledger["active_order_remainders"], field="active_order_remainders"):
        raise ValueError("comparison ledger has active order remainders")
    terminal_orders = _sequence(
        ledger["terminal_order_reasons"], field="terminal_order_reasons"
    )
    for item in terminal_orders:
        pair = _sequence(item, field="terminal order reason")
        if len(pair) != 2:
            raise ValueError("terminal order reason must be an ID/reason pair")
        _string(pair[0], field="terminal order ID")
        _string(pair[1], field="terminal order reason")
    final_map = {
        "final_cash": "cash_after",
        "final_portfolio_value": "portfolio_value_after",
        "final_total_cost": "total_cost_after",
        "final_funding_pnl": "funding_pnl_after",
        "final_borrow_cost": "borrow_cost_after",
        "final_turnover_total": "turnover_total_after",
        "final_max_drawdown": "max_drawdown_after",
    }
    last = parsed_intervals[-1][0]
    for final_name, interval_name in final_map.items():
        final_value = _number(ledger[final_name], field=final_name)
        _close(final_value, cast(float, last[interval_name]), field=final_name)
    _close(
        cast(float, ledger["final_cash"]),
        cast(float, ledger["final_portfolio_value"]),
        field="flat terminal cash/equity",
    )
    return ledger, tuple(returned)


def _validate_diagnostics(value: object) -> ExecutionDiagnostics:
    raw = _object(
        value,
        field="diagnostics",
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
    closed_raw = _object(
        raw["closed_trades"],
        field="diagnostics.closed_trades",
        keys={
            "average_realized_pnl",
            "breakeven_trades",
            "closed_trades",
            "gross_loss",
            "gross_profit",
            "losing_trades",
            "net_realized_pnl",
            "open_positions",
            "profit_factor",
            "win_rate",
            "winning_trades",
        },
    )
    closed = ClosedTradeDiagnostics(
        closed_trades=_integer(
            closed_raw["closed_trades"], field="closed_trades", minimum=0
        ),
        winning_trades=_integer(
            closed_raw["winning_trades"], field="winning_trades", minimum=0
        ),
        losing_trades=_integer(
            closed_raw["losing_trades"], field="losing_trades", minimum=0
        ),
        breakeven_trades=_integer(
            closed_raw["breakeven_trades"], field="breakeven_trades", minimum=0
        ),
        net_realized_pnl=_number(
            closed_raw["net_realized_pnl"], field="net_realized_pnl"
        ),
        gross_profit=_number(closed_raw["gross_profit"], field="gross_profit"),
        gross_loss=_number(closed_raw["gross_loss"], field="gross_loss"),
        open_positions=_integer(
            closed_raw["open_positions"], field="open_positions", minimum=0
        ),
    )
    expected_closed = closed.digest_payload()
    if canonical_json_bytes(expected_closed) != canonical_json_bytes(closed_raw):
        raise ValueError("closed-trade diagnostic summaries are inconsistent")
    reasons = _sequence(raw["termination_reasons"], field="termination_reasons")
    diagnostics = ExecutionDiagnostics(
        turnover_total=_number(raw["turnover_total"], field="turnover_total"),
        total_cost=_number(raw["total_cost"], field="total_cost"),
        funding_pnl=_number(raw["funding_pnl"], field="funding_pnl"),
        borrow_cost=_number(raw["borrow_cost"], field="borrow_cost"),
        n_trades=_integer(raw["n_trades"], field="n_trades", minimum=0),
        rebalance_events=_integer(
            raw["rebalance_events"], field="rebalance_events", minimum=0
        ),
        termination_reasons=tuple(
            _string(item, field="termination reason") for item in reasons
        ),
        closed_trades=closed,
    )
    if canonical_json_bytes(diagnostics.digest_payload()) != canonical_json_bytes(raw):
        raise ValueError("execution diagnostics are inconsistent")
    return diagnostics


def _validate_arm(
    value: object,
    *,
    expected_name: str,
    dataset_id: str,
    symbols: tuple[str, ...],
    start: int,
    stop: int,
    start_ns: int,
    stop_ns: int,
    initial_capital: float,
    execution_digest: str,
    decision_count: int,
) -> tuple[str, dict[str, object]]:
    arm = _object(
        value,
        field="comparison arm",
        keys={"diagnostics", "ledger", "metrics", "name", "returns"},
    )
    if arm["name"] != expected_name:
        raise ValueError("comparison arm order or name is invalid")
    ledger, ledger_returns = _validate_ledger(
        arm["ledger"],
        dataset_id=dataset_id,
        symbols=symbols,
        start=start,
        stop=stop,
        start_ns=start_ns,
        stop_ns=stop_ns,
        initial_capital=initial_capital,
        expected_policy_digest=execution_digest,
        expected_decision_count=decision_count,
    )
    return_map = _object(
        arm["returns"],
        field="returns",
        keys={"elapsed_years", "kind", "periods_per_year", "values"},
    )
    try:
        kind = ReturnKind(_string(return_map["kind"], field="returns.kind"))
    except ValueError as error:
        raise ValueError("unsupported return kind") from error
    periods = _integer(
        return_map["periods_per_year"], field="periods_per_year", minimum=1
    )
    elapsed = (
        None
        if return_map["elapsed_years"] is None
        else _number(return_map["elapsed_years"], field="elapsed_years")
    )
    values_raw = _sequence(return_map["values"], field="return values")
    values = tuple(_number(item, field="return value") for item in values_raw)
    returns = ReturnSeries(values, kind, periods, elapsed)
    if len(returns.values) != stop - start:
        raise ValueError("return series does not cover requested interval")
    for actual, expected in zip(returns.values, ledger_returns, strict=True):
        _close(actual, expected, field="return/ledger interval")
    diagnostics = _validate_diagnostics(arm["diagnostics"])
    if diagnostics.termination_reasons:
        raise ValueError("complete comparison cannot contain termination diagnostics")
    metrics = _object(
        arm["metrics"],
        field="metrics",
        keys={
            "borrow_cost",
            "funding_pnl",
            "max_drawdown",
            "n_periods",
            "n_trades",
            "periods_per_year",
            "rebalance_events",
            "return_kind",
            "sharpe",
            "sortino",
            "termination_count",
            "total_cost",
            "total_return",
            "turnover_total",
        },
    )
    recomputed = _metrics_payload(
        evaluate_performance(
            returns,
            turnover_total=diagnostics.turnover_total,
            total_cost=diagnostics.total_cost,
            funding_pnl=diagnostics.funding_pnl,
            borrow_cost=diagnostics.borrow_cost,
            n_trades=diagnostics.n_trades,
            rebalance_events=diagnostics.rebalance_events,
            termination_count=diagnostics.termination_count,
        )
    )
    if canonical_json_bytes(metrics) != canonical_json_bytes(recomputed):
        raise ValueError("stored metrics do not match the public metric owner")
    if (
        diagnostics.total_cost != ledger["final_total_cost"]
        or diagnostics.funding_pnl != ledger["final_funding_pnl"]
        or diagnostics.borrow_cost != ledger["final_borrow_cost"]
        or diagnostics.turnover_total != ledger["final_turnover_total"]
    ):
        raise ValueError("execution diagnostics differ from terminal ledger aggregates")
    _close(
        cast(float, _object(arm["metrics"], field="metrics")["total_return"]),
        cast(float, ledger["final_portfolio_value"]) / initial_capital - 1.0,
        field="terminal return/equity",
    )
    _close(
        cast(float, _object(arm["metrics"], field="metrics")["max_drawdown"]),
        cast(float, ledger["final_max_drawdown"]),
        field="terminal drawdown",
    )
    return expected_name, arm


def _validate_payload(value: object) -> dict[str, object]:
    payload = _object(
        value,
        field="comparison",
        keys={"arms", "config", "dataset", "model", "provenance", "schema_version"},
    )
    if payload["schema_version"] != _SCHEMA:
        raise ValueError("unsupported Ridge comparison artifact schema")
    dataset, binding, symbols = _validate_dataset_model(payload)
    dataset_id = cast(str, dataset["dataset_id"])
    start = cast(int, dataset["evaluation_start_index"])
    stop = cast(int, dataset["evaluation_stop_index"])
    start_ns = cast(int, dataset["evaluation_start_ns"])
    stop_ns = cast(int, dataset["evaluation_stop_ns"])
    config, execution_digest = _validate_config(
        payload["config"], dataset_id=dataset_id, symbols=symbols
    )
    if (config["start_index"], config["stop_index"]) != (start, stop):
        raise ValueError("config and requested Dataset boundaries differ")
    if binding.fit_cutoff_ns > start_ns:
        raise ValueError("Ridge fit cutoff is after requested decision start")
    execution_cost = _object(config["execution_cost"], field="config.execution_cost")
    decision_count = (
        stop
        - start
        - _integer(
            execution_cost["order_latency_bars"],
            field="execution_cost.order_latency_bars",
            minimum=0,
        )
        - 1
    )
    if decision_count <= 0:
        raise ValueError("comparison range has no agent decision interval")
    arms = _sequence(payload["arms"], field="arms")
    if len(arms) != len(_ARM_NAMES):
        raise ValueError("comparison must contain exactly four arms")
    for name, arm in zip(_ARM_NAMES, arms, strict=True):
        _validate_arm(
            arm,
            expected_name=name,
            dataset_id=dataset_id,
            symbols=symbols,
            start=start,
            stop=stop,
            start_ns=start_ns,
            stop_ns=stop_ns,
            initial_capital=cast(float, config["initial_capital"]),
            execution_digest=execution_digest,
            decision_count=decision_count,
        )
    _validate_provenance(payload["provenance"])
    return payload


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is unsupported: {value}")


def _load_raw(root: Path) -> bytes:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("comparison root must be a regular directory")
    entries = tuple(root.iterdir())
    if len(entries) != 1 or entries[0].name != _FILENAME:
        raise ValueError("comparison artifact must contain exactly comparison.json")
    with open_regular_binary(entries[0], field="comparison.json") as handle:
        return handle.read()


def load_ridge_comparison_artifact(
    root: str | Path,
    *,
    expected_digest: str,
) -> LoadedRidgeComparisonArtifact:
    """Load and strictly validate one externally digest-bound artifact."""

    require_sha256(expected_digest, field="expected_digest")
    artifact_root = Path(root)
    raw = _load_raw(artifact_root)
    digest = sha256(_DIGEST_DOMAIN + raw).hexdigest()
    if digest != expected_digest:
        raise ValueError("comparison artifact digest mismatch")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("comparison artifact is not valid JSON") from error
    validated = _validate_payload(payload)
    if canonical_json_bytes(validated) != raw:
        raise ValueError("comparison artifact JSON is not canonical")
    return LoadedRidgeComparisonArtifact(artifact_root, digest, validated)


def publish_ridge_comparison_artifact(
    root: str | Path,
    comparison: RidgeArtifactComparison,
    *,
    provenance: Mapping[str, object],
) -> str:
    """Write one complete comparison artifact without replacing any destination.

    The supplied provenance is validated as a declaration; it is not an
    attestation that this function captured execution before and after replay.
    """

    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping")
    payload = _comparison_payload(comparison, provenance)
    validated = _validate_payload(payload)
    raw = canonical_json_bytes(validated)
    digest = sha256(_DIGEST_DOMAIN + raw).hexdigest()
    output = Path(root)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"comparison destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        atomic_write_bytes(staging / _FILENAME, raw)
        load_ridge_comparison_artifact(staging, expected_digest=digest)
        atomic_rename_directory(staging, output)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return digest


__all__ = [
    "LoadedRidgeComparisonArtifact",
    "load_ridge_comparison_artifact",
    "publish_ridge_comparison_artifact",
]
