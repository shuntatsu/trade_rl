"""Research-only raw-evidence evaluator for sealed PPO training-layout arms."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields
from typing import Any, Literal, cast

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation import compare_strategies_by_symbol, compound_return
from trade_rl.evaluation.experiments.ppo_interleaved_evaluation_prereg import (
    canonical_ppo_interleaved_evaluation_protocol,
)
from trade_rl.evaluation.metrics import evaluate_performance
from trade_rl.evaluation.runs import execution_cost_for_overlay
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.interface import SingleSymbolStrategy
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.ppo import fit_ppo_strategy

Arm = Literal["baseline", "candidate"]

_SPEC_SCHEMA = "ppo_interleaved_evaluator_spec_v1"
_PATH_SCHEMA = "ppo_interleaved_return_path_evidence_v1"
_SYMBOL_SCHEMA = "ppo_interleaved_symbol_evidence_v1"
_SEED_SCHEMA = "ppo_interleaved_seed_evidence_v1"

_PROTOCOL_HEAD = "f1187dacae78e679a322cc53cbf03f3371f457b1"
_PROTOCOL_MODULE_BLOB = "62a71b1ca8b7d22fdfc14282508754c44d96091a"
_PROTOCOL_DIGEST = "a34aee66bf3f51ce02675b955f835c292b023770aab841f25b46815f9b399c2c"
_PROTOCOL_SEAL_RUN_ID = 35_205_354_305
_PROTOCOL_PRIMARY_ARTIFACT_ID = 10_489_866_637
_PROTOCOL_PRIMARY_ARTIFACT_DIGEST = (
    "4ca30f9884b642f32443bbb143b15e4fecbbc7a25d262e89d63649acbef00387"
)
_PROTOCOL_FRESH_ARTIFACT_ID = 10_489_661_856
_PROTOCOL_FRESH_ARTIFACT_DIGEST = (
    "2c9d9a71db2315dc66237c9ba4b975f0bdf5a5234ad87cf8bbd05b947867afbb"
)
_PROTOCOL_JSON_SHA256 = _PROTOCOL_DIGEST
_PROTOCOL_SEAL_JSON_SHA256 = (
    "3e55fdcf772bbf4063d537e3f1912a5d1588611ce51e586180c8d1be13a186fe"
)
_IMPLEMENTATION_HEAD = "cddef3532dd582d0f1066a61006f64265f0cabbe"


def _canonical_spec_values() -> dict[str, object]:
    protocol = canonical_ppo_interleaved_evaluation_protocol()
    return {
        "schema_version": _SPEC_SCHEMA,
        "issue_number": 632,
        "protocol_issue_number": 629,
        "protocol_head": _PROTOCOL_HEAD,
        "protocol_module_blob": _PROTOCOL_MODULE_BLOB,
        "protocol_digest": _PROTOCOL_DIGEST,
        "protocol_seal_run_id": _PROTOCOL_SEAL_RUN_ID,
        "protocol_primary_artifact_id": _PROTOCOL_PRIMARY_ARTIFACT_ID,
        "protocol_primary_artifact_digest": _PROTOCOL_PRIMARY_ARTIFACT_DIGEST,
        "protocol_fresh_artifact_id": _PROTOCOL_FRESH_ARTIFACT_ID,
        "protocol_fresh_artifact_digest": _PROTOCOL_FRESH_ARTIFACT_DIGEST,
        "protocol_json_sha256": _PROTOCOL_JSON_SHA256,
        "protocol_seal_json_sha256": _PROTOCOL_SEAL_JSON_SHA256,
        "implementation_head": _IMPLEMENTATION_HEAD,
        "successor_bundle_run_id": protocol.successor_bundle_run_id,
        "successor_bundle_artifact_id": protocol.successor_bundle_artifact_id,
        "successor_bundle_artifact_digest": protocol.successor_bundle_artifact_digest,
        "dataset_id": protocol.dataset_id,
        "dataset_artifact_digest": protocol.dataset_artifact_digest,
        "study_digest": protocol.study_digest,
        "execution_overlay": protocol.execution_overlay,
        "symbols": protocol.symbols,
        "feature_names": protocol.feature_names,
        "feature_indices": protocol.feature_indices,
        "fit_symbol_names": protocol.fit_symbol_names,
        "fit_cutoff": protocol.fit_cutoff,
        "evaluation_start": protocol.evaluation_start,
        "evaluation_stop_exclusive": protocol.evaluation_stop_exclusive,
        "gross_budget": protocol.gross_budget,
        "initial_capital": protocol.initial_capital,
        "slippage_std": protocol.slippage_std,
        "ppo_seeds": protocol.ppo_seeds,
        "ppo_total_timesteps": protocol.ppo_total_timesteps,
        "baseline_training_layout": protocol.baseline_training_layout,
        "baseline_rollout_steps_per_env": protocol.baseline_rollout_steps_per_env,
        "candidate_training_layout": protocol.candidate_training_layout,
        "candidate_rollout_steps_per_env": protocol.candidate_rollout_steps_per_env,
        "baseline_training_authorized": False,
        "candidate_training_authorized": False,
        "economic_result_inspected": False,
        "final_test_accessed": False,
        "shared_cash_profitability_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }


@dataclass(frozen=True, slots=True)
class PPOInterleavedEvaluatorSpec:
    schema_version: str
    issue_number: int
    protocol_issue_number: int
    protocol_head: str
    protocol_module_blob: str
    protocol_digest: str
    protocol_seal_run_id: int
    protocol_primary_artifact_id: int
    protocol_primary_artifact_digest: str
    protocol_fresh_artifact_id: int
    protocol_fresh_artifact_digest: str
    protocol_json_sha256: str
    protocol_seal_json_sha256: str
    implementation_head: str
    successor_bundle_run_id: int
    successor_bundle_artifact_id: int
    successor_bundle_artifact_digest: str
    dataset_id: str
    dataset_artifact_digest: str
    study_digest: str
    execution_overlay: str
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_indices: tuple[int, ...]
    fit_symbol_names: tuple[str, ...]
    fit_cutoff: str
    evaluation_start: str
    evaluation_stop_exclusive: str
    gross_budget: float
    initial_capital: float
    slippage_std: float
    ppo_seeds: tuple[int, ...]
    ppo_total_timesteps: int
    baseline_training_layout: str
    baseline_rollout_steps_per_env: None
    candidate_training_layout: str
    candidate_rollout_steps_per_env: int
    baseline_training_authorized: bool
    candidate_training_authorized: bool
    economic_result_inspected: bool
    final_test_accessed: bool
    shared_cash_profitability_established: bool
    production_eligible: bool
    live_trading_authorized: bool
    merge_authorized: bool

    def __post_init__(self) -> None:
        expected = _canonical_spec_values()
        for item in fields(self):
            actual = getattr(self, item.name)
            canonical = expected[item.name]
            if type(actual) is not type(canonical) or actual != canonical:
                raise ValueError(
                    f"{item.name} differs from sealed PPO evaluator authority"
                )

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            payload[item.name] = list(value) if isinstance(value, tuple) else value
        return payload

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def canonical_ppo_interleaved_evaluator_spec() -> PPOInterleavedEvaluatorSpec:
    protocol = canonical_ppo_interleaved_evaluation_protocol()
    if protocol.digest != _PROTOCOL_DIGEST:
        raise ValueError("sealed PPO interleaved protocol digest drifted")
    return PPOInterleavedEvaluatorSpec(**_canonical_spec_values())  # type: ignore[arg-type]


def _strict_return_values(values: object) -> tuple[float, ...]:
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    normalized: list[float] = []
    for value in raw.tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError("returns must contain only real numeric values")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("return path values must be finite")
        normalized.append(number)
    if not normalized:
        raise ValueError("return path must not be empty")
    if any(value <= -1.0 for value in normalized):
        raise ValueError("return path values must be greater than -1")
    return tuple(normalized)


def return_path_sha256(values: object) -> str:
    """Hash one exact one-dimensional float64 return path."""

    normalized = _strict_return_values(values)
    array = np.asarray(normalized, dtype=np.float64)
    little_endian = np.asarray(array, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(int(array.size).to_bytes(8, "big", signed=False))
    digest.update(little_endian.tobytes(order="C"))
    return digest.hexdigest()


def _finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


@dataclass(frozen=True, slots=True)
class PPOReturnPathEvidence:
    strategy_name: str
    returns: tuple[float, ...]
    return_sha256: str
    total_return: float
    total_cost: float
    turnover_total: float
    max_drawdown: float
    termination_count: int
    termination_reasons: tuple[str, ...]
    n_periods: int
    periods_per_year: int
    schema_version: str = _PATH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _PATH_SCHEMA:
            raise ValueError("unsupported PPO return-path evidence schema")
        if not isinstance(self.strategy_name, str) or not self.strategy_name:
            raise ValueError("strategy_name must be non-empty")
        normalized = _strict_return_values(self.returns)
        series = ReturnSeries(
            values=normalized,
            kind=ReturnKind.BASE_BAR,
            periods_per_year=self.periods_per_year,
        )
        object.__setattr__(self, "returns", normalized)
        if self.return_sha256 != return_path_sha256(normalized):
            raise ValueError("return_sha256 does not match exact raw returns")
        if isinstance(self.n_periods, bool) or not isinstance(self.n_periods, int):
            raise ValueError("n_periods must be an integer")
        if self.n_periods != len(normalized):
            raise ValueError("n_periods does not match raw return path")
        if (
            isinstance(self.periods_per_year, bool)
            or not isinstance(self.periods_per_year, int)
            or self.periods_per_year <= 0
        ):
            raise ValueError("periods_per_year must be a positive integer")
        if (
            isinstance(self.termination_count, bool)
            or not isinstance(self.termination_count, int)
            or self.termination_count < 0
        ):
            raise ValueError("termination_count must be a non-negative integer")
        reasons = tuple(self.termination_reasons)
        if any(not isinstance(reason, str) or not reason for reason in reasons):
            raise ValueError("termination reasons must contain non-empty strings")
        if self.termination_count != len(reasons):
            raise ValueError("termination count/reasons mismatch")
        object.__setattr__(self, "termination_reasons", reasons)
        cost = _finite_nonnegative(self.total_cost, field="total_cost")
        turnover = _finite_nonnegative(self.turnover_total, field="turnover_total")
        drawdown = _finite_nonnegative(self.max_drawdown, field="max_drawdown")
        if drawdown > 1.0:
            raise ValueError("max_drawdown must be within [0, 1]")
        if isinstance(self.total_return, bool) or not isinstance(
            self.total_return, (int, float)
        ):
            raise ValueError("total_return must be finite")
        total_return = float(self.total_return)
        if not math.isfinite(total_return):
            raise ValueError("total_return must be finite")
        metrics = evaluate_performance(
            series,
            turnover_total=turnover,
            total_cost=cost,
            termination_count=self.termination_count,
        )
        if total_return != metrics.total_return:
            raise ValueError("total_return does not match raw return path")
        if drawdown != metrics.max_drawdown:
            raise ValueError("max_drawdown does not match raw return path")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "strategy_name": self.strategy_name,
            "returns": list(self.returns),
            "return_sha256": self.return_sha256,
            "total_return": self.total_return,
            "total_cost": self.total_cost,
            "turnover_total": self.turnover_total,
            "max_drawdown": self.max_drawdown,
            "termination_count": self.termination_count,
            "termination_reasons": list(self.termination_reasons),
            "n_periods": self.n_periods,
            "periods_per_year": self.periods_per_year,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class PPOSymbolEvidence:
    symbol_index: int
    symbol: str
    ppo: PPOReturnPathEvidence
    cash: PPOReturnPathEvidence
    constant_long: PPOReturnPathEvidence
    constant_short: PPOReturnPathEvidence
    ppo_returns_equal_cash: bool
    ppo_returns_equal_constant_long: bool
    ppo_returns_equal_constant_short: bool
    schema_version: str = _SYMBOL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _SYMBOL_SCHEMA:
            raise ValueError("unsupported PPO symbol evidence schema")
        if (
            isinstance(self.symbol_index, bool)
            or not isinstance(self.symbol_index, int)
            or self.symbol_index < 0
        ):
            raise ValueError("symbol_index must be a non-negative integer")
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("symbol must be non-empty")
        expected_names = (
            (self.ppo, "ppo"),
            (self.cash, "cash"),
            (self.constant_long, "constant_long"),
            (self.constant_short, "constant_short"),
        )
        for evidence, expected in expected_names:
            if evidence.strategy_name != expected:
                raise ValueError("strategy_name differs from PPO evidence role")
        expected_flags = (
            self.ppo.returns == self.cash.returns,
            self.ppo.returns == self.constant_long.returns,
            self.ppo.returns == self.constant_short.returns,
        )
        actual_flags = (
            self.ppo_returns_equal_cash,
            self.ppo_returns_equal_constant_long,
            self.ppo_returns_equal_constant_short,
        )
        if any(type(value) is not bool for value in actual_flags):
            raise ValueError("policy-mode equality flags must be booleans")
        labels = ("cash", "constant_long", "constant_short")
        for label, actual, expected_flag in zip(
            labels, actual_flags, expected_flags, strict=True
        ):
            if actual is not expected_flag:
                raise ValueError(f"{label} equality flag differs from exact raw paths")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "symbol_index": self.symbol_index,
            "symbol": self.symbol,
            "ppo": self.ppo.to_payload(),
            "cash": self.cash.to_payload(),
            "constant_long": self.constant_long.to_payload(),
            "constant_short": self.constant_short.to_payload(),
            "ppo_returns_equal_cash": self.ppo_returns_equal_cash,
            "ppo_returns_equal_constant_long": self.ppo_returns_equal_constant_long,
            "ppo_returns_equal_constant_short": self.ppo_returns_equal_constant_short,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class PPOSeedEvidence:
    spec_digest: str
    protocol_head: str
    protocol_digest: str
    implementation_head: str
    successor_bundle_run_id: int
    successor_bundle_artifact_id: int
    successor_bundle_artifact_digest: str
    dataset_id: str
    dataset_artifact_digest: str
    study_digest: str
    execution_overlay: str
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_indices: tuple[int, ...]
    fit_symbol_names: tuple[str, ...]
    fit_cutoff: str
    evaluation_start: str
    evaluation_stop_exclusive: str
    gross_budget: float
    initial_capital: float
    arm: Arm
    seed: int
    training_layout: str
    rollout_steps_per_env: int | None
    caller_total_timesteps: int
    realized_num_timesteps: int
    by_symbol: tuple[PPOSymbolEvidence, ...]
    protocol_seal_run_id: int = _PROTOCOL_SEAL_RUN_ID
    protocol_primary_artifact_id: int = _PROTOCOL_PRIMARY_ARTIFACT_ID
    protocol_primary_artifact_digest: str = _PROTOCOL_PRIMARY_ARTIFACT_DIGEST
    protocol_fresh_artifact_id: int = _PROTOCOL_FRESH_ARTIFACT_ID
    protocol_fresh_artifact_digest: str = _PROTOCOL_FRESH_ARTIFACT_DIGEST
    protocol_json_sha256: str = _PROTOCOL_JSON_SHA256
    protocol_seal_json_sha256: str = _PROTOCOL_SEAL_JSON_SHA256
    baseline_training_authorized: bool = False
    candidate_training_authorized: bool = False
    economic_result_inspected: bool = False
    final_test_accessed: bool = False
    shared_cash_profitability_established: bool = False
    production_eligible: bool = False
    live_trading_authorized: bool = False
    merge_authorized: bool = False
    schema_version: str = _SEED_SCHEMA

    def __post_init__(self) -> None:
        spec = canonical_ppo_interleaved_evaluator_spec()
        if self.schema_version != _SEED_SCHEMA:
            raise ValueError("unsupported PPO seed evidence schema")
        expected_identity = {
            "spec_digest": spec.digest,
            "protocol_head": spec.protocol_head,
            "protocol_digest": spec.protocol_digest,
            "implementation_head": spec.implementation_head,
            "successor_bundle_run_id": spec.successor_bundle_run_id,
            "successor_bundle_artifact_id": spec.successor_bundle_artifact_id,
            "successor_bundle_artifact_digest": spec.successor_bundle_artifact_digest,
            "dataset_id": spec.dataset_id,
            "dataset_artifact_digest": spec.dataset_artifact_digest,
            "study_digest": spec.study_digest,
            "execution_overlay": spec.execution_overlay,
            "symbols": spec.symbols,
            "feature_names": spec.feature_names,
            "feature_indices": spec.feature_indices,
            "fit_symbol_names": spec.fit_symbol_names,
            "fit_cutoff": spec.fit_cutoff,
            "evaluation_start": spec.evaluation_start,
            "evaluation_stop_exclusive": spec.evaluation_stop_exclusive,
            "gross_budget": spec.gross_budget,
            "initial_capital": spec.initial_capital,
            "protocol_seal_run_id": spec.protocol_seal_run_id,
            "protocol_primary_artifact_id": spec.protocol_primary_artifact_id,
            "protocol_primary_artifact_digest": spec.protocol_primary_artifact_digest,
            "protocol_fresh_artifact_id": spec.protocol_fresh_artifact_id,
            "protocol_fresh_artifact_digest": spec.protocol_fresh_artifact_digest,
            "protocol_json_sha256": spec.protocol_json_sha256,
            "protocol_seal_json_sha256": spec.protocol_seal_json_sha256,
        }
        for field_name, expected in expected_identity.items():
            actual = getattr(self, field_name)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    f"{field_name} differs from sealed PPO evidence identity"
                )
        if self.arm not in ("baseline", "candidate"):
            raise ValueError("arm must be baseline or candidate")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed not in spec.ppo_seeds
        ):
            raise ValueError("seed must be one of the sealed PPO seeds")
        expected_layout = (
            spec.baseline_training_layout
            if self.arm == "baseline"
            else spec.candidate_training_layout
        )
        expected_rollout = (
            spec.baseline_rollout_steps_per_env
            if self.arm == "baseline"
            else spec.candidate_rollout_steps_per_env
        )
        if (
            type(self.training_layout) is not str
            or self.training_layout != expected_layout
        ):
            raise ValueError("training_layout differs from sealed arm authority")
        if (
            type(self.rollout_steps_per_env) is not type(expected_rollout)
            or self.rollout_steps_per_env != expected_rollout
        ):
            raise ValueError("rollout_steps_per_env differs from sealed arm authority")
        if (
            type(self.caller_total_timesteps) is not int
            or self.caller_total_timesteps != spec.ppo_total_timesteps
        ):
            raise ValueError("caller_total_timesteps differs from sealed authority")
        if (
            isinstance(self.realized_num_timesteps, bool)
            or not isinstance(self.realized_num_timesteps, int)
            or self.realized_num_timesteps <= 0
        ):
            raise ValueError("realized_num_timesteps must be a positive integer")
        rows = tuple(self.by_symbol)
        expected_symbols = tuple(
            (index, symbol) for index, symbol in enumerate(spec.symbols)
        )
        actual_symbols = tuple((item.symbol_index, item.symbol) for item in rows)
        if actual_symbols != expected_symbols:
            raise ValueError("symbol roster/order differs from sealed PPO evidence")
        object.__setattr__(self, "by_symbol", rows)
        for field_name in (
            "baseline_training_authorized",
            "candidate_training_authorized",
            "economic_result_inspected",
            "final_test_accessed",
            "shared_cash_profitability_established",
            "production_eligible",
            "live_trading_authorized",
            "merge_authorized",
        ):
            value = getattr(self, field_name)
            if type(value) is not bool or value:
                raise ValueError(
                    "research evaluator cannot authorize production/final/shared-cash use"
                )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "spec_digest": self.spec_digest,
            "protocol_head": self.protocol_head,
            "protocol_digest": self.protocol_digest,
            "protocol_seal_run_id": self.protocol_seal_run_id,
            "protocol_primary_artifact_id": self.protocol_primary_artifact_id,
            "protocol_primary_artifact_digest": self.protocol_primary_artifact_digest,
            "protocol_fresh_artifact_id": self.protocol_fresh_artifact_id,
            "protocol_fresh_artifact_digest": self.protocol_fresh_artifact_digest,
            "protocol_json_sha256": self.protocol_json_sha256,
            "protocol_seal_json_sha256": self.protocol_seal_json_sha256,
            "implementation_head": self.implementation_head,
            "successor_bundle_run_id": self.successor_bundle_run_id,
            "successor_bundle_artifact_id": self.successor_bundle_artifact_id,
            "successor_bundle_artifact_digest": self.successor_bundle_artifact_digest,
            "dataset_id": self.dataset_id,
            "dataset_artifact_digest": self.dataset_artifact_digest,
            "study_digest": self.study_digest,
            "execution_overlay": self.execution_overlay,
            "symbols": list(self.symbols),
            "feature_names": list(self.feature_names),
            "feature_indices": list(self.feature_indices),
            "fit_symbol_names": list(self.fit_symbol_names),
            "fit_cutoff": self.fit_cutoff,
            "evaluation_start": self.evaluation_start,
            "evaluation_stop_exclusive": self.evaluation_stop_exclusive,
            "gross_budget": self.gross_budget,
            "initial_capital": self.initial_capital,
            "arm": self.arm,
            "seed": self.seed,
            "training_layout": self.training_layout,
            "rollout_steps_per_env": self.rollout_steps_per_env,
            "caller_total_timesteps": self.caller_total_timesteps,
            "realized_num_timesteps": self.realized_num_timesteps,
            "by_symbol": [item.to_payload() for item in self.by_symbol],
            "baseline_training_authorized": self.baseline_training_authorized,
            "candidate_training_authorized": self.candidate_training_authorized,
            "economic_result_inspected": self.economic_result_inspected,
            "final_test_accessed": self.final_test_accessed,
            "shared_cash_profitability_established": self.shared_cash_profitability_established,
            "production_eligible": self.production_eligible,
            "live_trading_authorized": self.live_trading_authorized,
            "merge_authorized": self.merge_authorized,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def _datetime64(value: str) -> np.datetime64:
    return np.datetime64(value.removesuffix("Z"), "ns")


def _exact_index(dataset: MarketDataset, value: str, *, field: str) -> int:
    matches = np.flatnonzero(
        np.asarray(dataset.timestamps, dtype="datetime64[ns]") == _datetime64(value)
    )
    if matches.size != 1:
        raise ValueError(f"{field} must identify exactly one dataset row")
    return int(matches[0])


def _validate_dataset(
    dataset: MarketDataset,
    spec: PPOInterleavedEvaluatorSpec,
) -> tuple[int, int, int, tuple[int, ...]]:
    if dataset.dataset_id != spec.dataset_id:
        raise ValueError("dataset id differs from sealed PPO evaluator identity")
    if tuple(dataset.symbols) != spec.symbols:
        raise ValueError(
            "dataset symbol roster differs from sealed PPO evaluator identity"
        )
    for index, name in zip(spec.feature_indices, spec.feature_names, strict=True):
        if index >= len(dataset.feature_names) or dataset.feature_names[index] != name:
            raise ValueError(
                "dataset feature identity differs from sealed PPO evaluator identity"
            )
    fit_cutoff = _datetime64(spec.fit_cutoff)
    eligible = np.flatnonzero(
        np.asarray(dataset.timestamps, dtype="datetime64[ns]") < fit_cutoff
    )
    if eligible.size < 2:
        raise ValueError(
            "dataset must contain at least two pre-cutoff PPO training rows"
        )
    train_stop = int(eligible[-1])
    evaluation_start = _exact_index(
        dataset, spec.evaluation_start, field="evaluation_start"
    )
    evaluation_stop = _exact_index(
        dataset,
        spec.evaluation_stop_exclusive,
        field="evaluation_stop_exclusive",
    )
    if not 0 <= train_stop < evaluation_start < evaluation_stop < dataset.n_bars:
        raise ValueError("PPO training/evaluation clock differs from sealed authority")
    symbol_indices = tuple(
        dataset.symbols.index(name) for name in spec.fit_symbol_names
    )
    if symbol_indices != tuple(range(len(spec.symbols))):
        raise ValueError(
            "fit symbol identity differs from sealed PPO evaluator authority"
        )
    return train_stop, evaluation_start, evaluation_stop, symbol_indices


def _path_from_entry(name: str, entry: Any) -> PPOReturnPathEvidence:
    if getattr(entry, "name", None) != name:
        raise RuntimeError("strategy comparison entry name drifted")
    returns = tuple(float(value) for value in entry.replay.returns.values)
    metrics = entry.metrics
    diagnostics = entry.replay.diagnostics
    reasons = tuple(diagnostics.termination_reasons)
    if metrics.n_periods != len(returns):
        raise RuntimeError("return period count differs from metrics")
    if metrics.total_return != compound_return(returns):
        raise RuntimeError("total return differs from exact raw return path")
    if metrics.total_cost != diagnostics.total_cost:
        raise RuntimeError("total cost differs from replay diagnostics")
    if metrics.turnover_total != diagnostics.turnover_total:
        raise RuntimeError("turnover differs from replay diagnostics")
    if metrics.termination_count != len(reasons):
        raise RuntimeError("termination count differs from replay diagnostics")
    if diagnostics.termination_count != len(reasons):
        raise RuntimeError("termination diagnostics are inconsistent")
    return PPOReturnPathEvidence(
        strategy_name=name,
        returns=returns,
        return_sha256=return_path_sha256(returns),
        total_return=float(metrics.total_return),
        total_cost=float(metrics.total_cost),
        turnover_total=float(metrics.turnover_total),
        max_drawdown=float(metrics.max_drawdown),
        termination_count=int(metrics.termination_count),
        termination_reasons=reasons,
        n_periods=int(metrics.n_periods),
        periods_per_year=int(entry.replay.returns.periods_per_year),
    )


def _realized_num_timesteps(strategy: Any) -> int:
    value = getattr(getattr(strategy, "policy", None), "num_timesteps", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("realized_num_timesteps must be a positive integer")
    return value


def evaluate_ppo_training_seed(
    dataset: MarketDataset,
    spec: PPOInterleavedEvaluatorSpec,
    *,
    arm: str,
    seed: int,
) -> PPOSeedEvidence:
    """Train exactly one sealed PPO arm/seed and emit uninterpreted raw evidence."""

    canonical = canonical_ppo_interleaved_evaluator_spec()
    if spec != canonical:
        raise ValueError("spec differs from sealed PPO evaluator authority")
    if arm not in ("baseline", "candidate"):
        raise ValueError("arm must be baseline or candidate")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed not in spec.ppo_seeds
    ):
        raise ValueError("seed must be one of the sealed PPO seeds")

    train_stop, evaluation_start, evaluation_stop, fit_symbol_indices = (
        _validate_dataset(dataset, spec)
    )
    execution_cost = execution_cost_for_overlay(spec.execution_overlay)
    if (
        not isinstance(execution_cost.slippage_std, (int, float))
        or float(execution_cost.slippage_std) != spec.slippage_std
        or execution_cost.processing_bar_volume_capacity is not False
    ):
        raise ValueError("execution cost differs from sealed calibrated PPO authority")

    training_layout = (
        spec.baseline_training_layout
        if arm == "baseline"
        else spec.candidate_training_layout
    )
    rollout_steps = (
        spec.baseline_rollout_steps_per_env
        if arm == "baseline"
        else spec.candidate_rollout_steps_per_env
    )
    strategy = fit_ppo_strategy(
        dataset,
        feature_indices=spec.feature_indices,
        fit_symbol_indices=fit_symbol_indices,
        start_index=0,
        stop_index=train_stop,
        gross_budget=spec.gross_budget,
        total_timesteps=spec.ppo_total_timesteps,
        seed=seed,
        initial_capital=spec.initial_capital,
        execution_cost=execution_cost,
        training_layout=training_layout,
        rollout_steps_per_env=rollout_steps,
    )
    realized_num_timesteps = _realized_num_timesteps(strategy)

    strategies: dict[str, SingleSymbolStrategy] = {
        "ppo": strategy,
        "cash": ConstantIntentStrategy(PositionIntent.FLAT),
        "constant_long": ConstantIntentStrategy(PositionIntent.LONG),
        "constant_short": ConstantIntentStrategy(PositionIntent.SHORT),
    }
    comparison = compare_strategies_by_symbol(
        dataset,
        strategies,
        start_index=evaluation_start,
        stop_index=evaluation_stop,
        gross_budget=spec.gross_budget,
        initial_capital=spec.initial_capital,
        execution_cost=execution_cost,
        risk=None,
    )

    rows: list[PPOSymbolEvidence] = []
    if len(comparison.by_symbol) != len(spec.symbols):
        raise RuntimeError("paired PPO replay did not return every sealed symbol")
    for expected_index, (expected_symbol, result) in enumerate(
        zip(spec.symbols, comparison.by_symbol, strict=True)
    ):
        if result.symbol_index != expected_index or result.symbol != expected_symbol:
            raise RuntimeError("paired PPO replay symbol roster/order drifted")
        entries = {entry.name: entry for entry in result.comparison.entries}
        expected_entries = ("ppo", "cash", "constant_long", "constant_short")
        if tuple(entry.name for entry in result.comparison.entries) != expected_entries:
            raise RuntimeError("paired PPO replay strategy roster/order drifted")
        if len(entries) != len(expected_entries):
            raise RuntimeError("paired PPO replay strategy names are duplicated")
        ppo = _path_from_entry("ppo", entries["ppo"])
        cash = _path_from_entry("cash", entries["cash"])
        constant_long = _path_from_entry("constant_long", entries["constant_long"])
        constant_short = _path_from_entry("constant_short", entries["constant_short"])
        rows.append(
            PPOSymbolEvidence(
                symbol_index=expected_index,
                symbol=expected_symbol,
                ppo=ppo,
                cash=cash,
                constant_long=constant_long,
                constant_short=constant_short,
                ppo_returns_equal_cash=ppo.returns == cash.returns,
                ppo_returns_equal_constant_long=ppo.returns == constant_long.returns,
                ppo_returns_equal_constant_short=ppo.returns == constant_short.returns,
            )
        )

    return PPOSeedEvidence(
        spec_digest=spec.digest,
        protocol_head=spec.protocol_head,
        protocol_digest=spec.protocol_digest,
        implementation_head=spec.implementation_head,
        successor_bundle_run_id=spec.successor_bundle_run_id,
        successor_bundle_artifact_id=spec.successor_bundle_artifact_id,
        successor_bundle_artifact_digest=spec.successor_bundle_artifact_digest,
        dataset_id=spec.dataset_id,
        dataset_artifact_digest=spec.dataset_artifact_digest,
        study_digest=spec.study_digest,
        execution_overlay=spec.execution_overlay,
        symbols=spec.symbols,
        feature_names=spec.feature_names,
        feature_indices=spec.feature_indices,
        fit_symbol_names=spec.fit_symbol_names,
        fit_cutoff=spec.fit_cutoff,
        evaluation_start=spec.evaluation_start,
        evaluation_stop_exclusive=spec.evaluation_stop_exclusive,
        gross_budget=spec.gross_budget,
        initial_capital=spec.initial_capital,
        arm=cast(Arm, arm),
        seed=seed,
        training_layout=training_layout,
        rollout_steps_per_env=rollout_steps,
        caller_total_timesteps=spec.ppo_total_timesteps,
        realized_num_timesteps=realized_num_timesteps,
        by_symbol=tuple(rows),
    )


__all__ = [
    "PPOInterleavedEvaluatorSpec",
    "PPOReturnPathEvidence",
    "PPOSeedEvidence",
    "PPOSymbolEvidence",
    "canonical_ppo_interleaved_evaluator_spec",
    "evaluate_ppo_training_seed",
    "return_path_sha256",
]
