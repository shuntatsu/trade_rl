"""Immutable candidate-run publication, loading, and semantic identity."""

from __future__ import annotations

import io
import json
import math
import shutil
import tempfile
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.atomic_write import atomic_rename_directory, atomic_write_bytes
from trade_rl.artifacts.canonical import freeze_json_value
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import file_digest_and_size, read_verified_bytes
from trade_rl.evaluation.metrics import PerformanceMetrics
from trade_rl.evaluation.replay import _agent_stop_index
from trade_rl.evaluation.runs.config import LEGACY_DATASET_EXECUTION_OVERLAY
from trade_rl.evaluation.runs.execute import (
    CandidateRunResult,
    _execution_cost_for_overlay,
)
from trade_rl.evaluation.runs.provenance import PROVENANCE_SCHEMA
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.risk.pretrade import should_rebind_strategy_proposal
from trade_rl.simulation.diagnostics.funding import FundingBoundaryEvidence
from trade_rl.simulation.orders.model import OrderEvent
from trade_rl.strategies.position_duration import (
    constrain_intent_for_minimum_hold,
    next_position_age_bars,
)
from trade_rl.strategies.position_intent import PositionIntent, target_weight_for_intent
from trade_rl.strategies.rl.intent import (
    PPO_OBSERVATION_SCHEMA_V3,
    ppo_observation_contract_payload,
)
from trade_rl.strategies.rl.ppo_training import expected_ppo_realized_timesteps

_RESULT_SCHEMA_V1 = "lean_candidate_result_v1"
_RESULT_SCHEMA_V2 = "lean_candidate_result_v2"
_RESULT_SCHEMA_V3 = "lean_candidate_result_v3"
_RESULT_SCHEMA_V4 = "lean_candidate_result_v4"
_RESULT_SCHEMA_V5 = "lean_candidate_result_v5"
_RESULT_SCHEMA_V6 = "lean_candidate_result_v6"
_RESULT_SCHEMA_V7 = "lean_candidate_result_v7"
_RESULT_SCHEMA_V8 = "lean_candidate_result_v8"
_RESULT_SCHEMA_V9 = "lean_candidate_result_v9"
_RESULT_SCHEMA_V10 = "lean_candidate_result_v10"
_RESULT_SCHEMA_V11 = "lean_candidate_result_v11"
_SUPPORTED_RESULT_SCHEMAS = frozenset(
    {
        _RESULT_SCHEMA_V1,
        _RESULT_SCHEMA_V2,
        _RESULT_SCHEMA_V3,
        _RESULT_SCHEMA_V4,
        _RESULT_SCHEMA_V5,
        _RESULT_SCHEMA_V6,
        _RESULT_SCHEMA_V7,
        _RESULT_SCHEMA_V8,
        _RESULT_SCHEMA_V9,
        _RESULT_SCHEMA_V10,
        _RESULT_SCHEMA_V11,
    }
)
_ARTIFACT_IDENTITY_SCHEMA = "candidate_run_artifact_identity_v1"
_REQUIRED_FILES = frozenset({"summary.json", "returns.npz", "provenance.json"})

FileEvidence = tuple[str, int]
ArtifactFileEvidence = tuple[FileEvidence, FileEvidence, FileEvidence]


@dataclass(frozen=True, slots=True)
class PublishedCandidateRun:
    """Paths of one immutably published candidate comparison run."""

    root: Path
    summary_path: Path
    returns_path: Path
    provenance_path: Path


@dataclass(frozen=True, slots=True)
class LoadedCandidateRun:
    """Validated semantic content of one published candidate-run artifact."""

    root: Path
    summary: Mapping[str, object]
    returns: Mapping[str, np.ndarray]
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        frozen_summary = freeze_json_value(self.summary)
        frozen_provenance = freeze_json_value(self.provenance)
        if not isinstance(frozen_summary, Mapping) or not isinstance(
            frozen_provenance, Mapping
        ):
            raise TypeError("candidate summary/provenance must be JSON objects")

        immutable_returns: dict[str, np.ndarray] = {}
        for key, value in self.returns.items():
            contiguous = np.ascontiguousarray(value)
            immutable_returns[key] = np.frombuffer(
                contiguous.tobytes(order="C"),
                dtype=contiguous.dtype,
            ).reshape(contiguous.shape)

        object.__setattr__(self, "summary", frozen_summary)
        object.__setattr__(self, "returns", MappingProxyType(immutable_returns))
        object.__setattr__(self, "provenance", frozen_provenance)

    @property
    def has_verified_full_evaluation_coverage(self) -> bool:
        """Whether this artifact schema verifies every requested evaluation period."""
        schema = self.summary.get("schema_version")
        if schema not in {
            _RESULT_SCHEMA_V6,
            _RESULT_SCHEMA_V7,
            _RESULT_SCHEMA_V8,
            _RESULT_SCHEMA_V9,
            _RESULT_SCHEMA_V10,
            _RESULT_SCHEMA_V11,
        }:
            return False
        evaluation = self.summary.get("evaluation")
        by_symbol = self.summary.get("by_symbol")
        if (
            not isinstance(evaluation, Mapping)
            or not isinstance(by_symbol, Sequence)
            or isinstance(by_symbol, (str, bytes, bytearray))
        ):
            return False
        expected_periods = evaluation.get("expected_periods")
        if (
            isinstance(expected_periods, bool)
            or not isinstance(expected_periods, int)
            or expected_periods <= 0
        ):
            return False
        for symbol in by_symbol:
            strategies = (
                symbol.get("strategies") if isinstance(symbol, Mapping) else None
            )
            if not isinstance(strategies, Sequence) or isinstance(
                strategies, (str, bytes, bytearray)
            ):
                return False
            for strategy in strategies:
                if not isinstance(strategy, Mapping):
                    return False
                metrics = strategy.get("metrics")
                key = strategy.get("return_key")
                values = self.returns.get(key) if isinstance(key, str) else None
                n_periods = (
                    metrics.get("n_periods") if isinstance(metrics, Mapping) else None
                )
                if (
                    values is None
                    or isinstance(n_periods, bool)
                    or not isinstance(n_periods, int)
                    or n_periods != expected_periods
                    or values.size != expected_periods
                ):
                    return False
        if schema in {
            _RESULT_SCHEMA_V7,
            _RESULT_SCHEMA_V8,
            _RESULT_SCHEMA_V9,
            _RESULT_SCHEMA_V10,
            _RESULT_SCHEMA_V11,
        }:
            portfolio = self.summary.get("shared_cash_ppo")
            if not isinstance(portfolio, Mapping):
                return False
            key = portfolio.get("return_key")
            values = self.returns.get(key) if isinstance(key, str) else None
            metrics = portfolio.get("metrics")
            n_periods = (
                metrics.get("n_periods") if isinstance(metrics, Mapping) else None
            )
            if (
                values is None
                or isinstance(n_periods, bool)
                or not isinstance(n_periods, int)
                or n_periods != expected_periods
                or values.size != expected_periods
            ):
                return False
        return True


@dataclass(frozen=True, slots=True)
class CandidateRunArtifactIdentity:
    """Stable semantic identity plus exact file-level evidence."""

    schema_version: str
    result_schema_version: str
    artifact_digest: str
    summary_file_sha256: str
    summary_file_size: int
    returns_file_sha256: str
    returns_file_size: int
    provenance_file_sha256: str
    provenance_file_size: int


def _metrics_payload(metrics: PerformanceMetrics) -> dict[str, object]:
    return {
        "total_return": metrics.total_return,
        "sharpe": metrics.sharpe,
        "sortino": metrics.sortino,
        "max_drawdown": metrics.max_drawdown,
        "turnover_total": metrics.turnover_total,
        "total_cost": metrics.total_cost,
        "funding_pnl": metrics.funding_pnl,
        "borrow_cost": metrics.borrow_cost,
        "n_trades": metrics.n_trades,
        "rebalance_events": metrics.rebalance_events,
        "termination_count": metrics.termination_count,
        "n_periods": metrics.n_periods,
        "return_kind": metrics.return_kind.value,
        "periods_per_year": metrics.periods_per_year,
    }


def _evaluation_payload(result: CandidateRunResult) -> dict[str, object]:
    spec = result.spec
    config = spec.config
    payload: dict[str, object] = {
        "start": str(config.evaluation_start),
        "stop_exclusive": str(config.evaluation_stop_exclusive),
        "expected_periods": spec.evaluation_stop_index - spec.evaluation_start_index,
        "gross_budget": config.gross_budget,
        "initial_capital": config.initial_capital,
        "ppo_settle_terminal_position": config.ppo_settle_terminal_position,
        "pretrade_risk_config": (
            None
            if config.pretrade_risk_config is None
            else asdict(config.pretrade_risk_config)
        ),
        "execution_overlay": getattr(
            spec, "execution_overlay", LEGACY_DATASET_EXECUTION_OVERLAY
        ),
    }
    if result.comparison.shared_cash_ppo is not None:
        cost = _execution_cost_for_overlay(spec.execution_overlay)
        payload["evaluation_start_index"] = spec.evaluation_start_index
        payload["evaluation_stop_index"] = spec.evaluation_stop_index
        payload["ppo_policy_decision_stop_index"] = _agent_stop_index(
            start_index=spec.evaluation_start_index,
            stop_index=spec.evaluation_stop_index,
            execution_cost=cost,
            settle_terminal_position=config.ppo_settle_terminal_position,
        )
        ledger = result.comparison.shared_cash_ppo.replay.ledger_evidence
        if ledger is not None and ledger.schema_version in {
            "shared_cash_replay_ledger_v3",
            "shared_cash_replay_ledger_v4",
        }:
            payload["ppo_minimum_hold_bars"] = config.ppo_minimum_hold_bars
    return payload


def _result_payload(
    result: CandidateRunResult,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    spec = result.spec
    config = spec.config
    lean_config = spec.lean_config
    expected_periods = spec.evaluation_stop_index - spec.evaluation_start_index
    if expected_periods <= 0:
        raise ValueError("candidate evaluation must contain at least one interval")
    returns: dict[str, np.ndarray] = {}
    symbols_payload: list[dict[str, object]] = []
    for symbol_result in result.comparison.by_symbol:
        strategies_payload: list[dict[str, object]] = []
        for strategy_index, entry in enumerate(symbol_result.comparison.entries):
            return_key = (
                f"symbol_{symbol_result.symbol_index}_strategy_{strategy_index}"
            )
            returns[return_key] = np.asarray(
                entry.replay.returns.values,
                dtype=np.float64,
            )
            diagnostics = entry.replay.diagnostics
            final_quantities = [float(value) for value in entry.replay.book.quantities]
            active_order_remainders = [
                {"order_id": order_id, "remaining_quantity": float(quantity)}
                for order_id, quantity in entry.replay.active_order_remainders
            ]
            terminal_order_reasons = [
                {"order_id": order_id, "reason": reason}
                for order_id, reason in entry.replay.terminal_order_reasons
            ]
            minimum_hold_audit = [
                {
                    "index": decision.index,
                    "requested_intent": int(decision.intent),
                    "effective_intent": int(decision.effective_intent),
                    "position_age_bars": decision.position_age_bars,
                    "position_age_bars_after": decision.position_age_bars_after,
                    "position_quantity_before": decision.position_quantity_before,
                    "position_quantity_after": decision.position_quantity_after,
                    "target_weight": decision.target_weight,
                    "minimum_hold_suppressed": decision.minimum_hold_suppressed,
                    "minimum_hold_unlocked": decision.minimum_hold_unlocked,
                    "risk_reasons": list(decision.risk_reasons),
                }
                for decision in entry.replay.decisions
                if decision.minimum_hold_suppressed or decision.minimum_hold_unlocked
            ]
            strategies_payload.append(
                {
                    "name": entry.name,
                    "return_key": return_key,
                    "metrics": _metrics_payload(entry.metrics),
                    "diagnostics": {
                        "turnover_total": diagnostics.turnover_total,
                        "total_cost": diagnostics.total_cost,
                        "funding_pnl": diagnostics.funding_pnl,
                        "borrow_cost": diagnostics.borrow_cost,
                        "n_trades": diagnostics.n_trades,
                        "rebalance_events": diagnostics.rebalance_events,
                        "termination_reasons": list(diagnostics.termination_reasons),
                        "minimum_hold_suppressed_count": sum(
                            decision.minimum_hold_suppressed
                            for decision in entry.replay.decisions
                        ),
                        "minimum_hold_unlocked_count": sum(
                            decision.minimum_hold_unlocked
                            for decision in entry.replay.decisions
                        ),
                    },
                    "final_portfolio_value": entry.replay.book.portfolio_value,
                    "fill_count": entry.replay.book.fill_count,
                    "final_quantities": final_quantities,
                    "active_order_remainders": active_order_remainders,
                    "terminal_order_reasons": terminal_order_reasons,
                    "terminal_settlement_complete": (
                        config.ppo_settle_terminal_position
                        and not any(quantity != 0.0 for quantity in final_quantities)
                        and not active_order_remainders
                        and len(entry.replay.returns.values) == expected_periods
                        and entry.metrics.termination_count == 0
                        and not diagnostics.termination_reasons
                    ),
                    "minimum_hold_audit": minimum_hold_audit,
                }
            )
        symbols_payload.append(
            {
                "symbol_index": symbol_result.symbol_index,
                "symbol": symbol_result.symbol,
                "strategies": strategies_payload,
            }
        )

    summary: dict[str, object] = {
        "schema_version": (
            _RESULT_SCHEMA_V11
            if result.comparison.shared_cash_ppo is not None
            and result.comparison.shared_cash_ppo.replay.ledger_evidence is not None
            and result.comparison.shared_cash_ppo.replay.ledger_evidence.schema_version
            == "shared_cash_replay_ledger_v4"
            else (
                _RESULT_SCHEMA_V10
                if result.comparison.shared_cash_ppo is not None
                and result.comparison.shared_cash_ppo.replay.ledger_evidence is not None
                and result.comparison.shared_cash_ppo.replay.ledger_evidence.schema_version
                == "shared_cash_replay_ledger_v3"
                else (
                    _RESULT_SCHEMA_V9
                    if result.comparison.shared_cash_ppo is not None
                    else _RESULT_SCHEMA_V6
                )
            )
        ),
        "ppo_observation": ppo_observation_contract_payload(
            config.ppo_observation_schema
        ),
        "dataset_id": spec.dataset_id,
        "dataset_artifact": {
            "schema_version": spec.dataset_artifact_schema,
            "artifact_digest": spec.dataset_artifact_digest,
        },
        "symbols": list(result.symbols),
        "candidate_config": {
            "signal_name": config.signal_name,
            "signal_index": lean_config.signal_index,
            "feature_names": list(config.feature_names),
            "feature_indices": list(lean_config.feature_indices),
            "fit_symbol_names": list(config.fit_symbol_names),
            "fit_symbol_indices": list(lean_config.fit_symbol_indices),
            "fit_cutoff": str(lean_config.fit_cutoff),
            "rule_entry_threshold": lean_config.rule_entry_threshold,
            "rule_exit_threshold": lean_config.rule_exit_threshold,
            "forecast_entry_threshold": lean_config.forecast_entry_threshold,
            "forecast_exit_threshold": lean_config.forecast_exit_threshold,
            "ppo_total_timesteps": lean_config.ppo_total_timesteps,
            "ppo_seed": lean_config.ppo_seed,
            "ppo_training_layout": lean_config.ppo_training_layout,
            "ppo_rollout_steps_per_env": lean_config.ppo_rollout_steps_per_env,
            "ppo_minimum_hold_bars": lean_config.ppo_minimum_hold_bars,
            "ppo_observation_schema": config.ppo_observation_schema,
            "ppo_settle_terminal_position": config.ppo_settle_terminal_position,
            "pretrade_risk_config": (
                None
                if config.pretrade_risk_config is None
                else asdict(config.pretrade_risk_config)
            ),
            "ppo_training_timesteps": result.ppo_training_timesteps,
            "ppo_training_minimum_hold_suppressed_count": (
                result.ppo_training_minimum_hold_suppressed_count
            ),
        },
        "evaluation": _evaluation_payload(result),
        "by_symbol": symbols_payload,
    }
    shared_cash_ppo = result.comparison.shared_cash_ppo
    if shared_cash_ppo is not None:
        replay = shared_cash_ppo.replay
        ledger = replay.ledger_evidence
        if ledger is None:
            raise ValueError("shared-cash PPO result requires ledger evidence")
        ledger_payload = ledger.to_mapping()
        return_key = "shared_cash_ppo"
        returns[return_key] = np.asarray(replay.returns.values, dtype=np.float64)
        diagnostics = replay.diagnostics
        final_quantities = [float(value) for value in replay.book.quantities]
        remainders = ledger.active_order_remainders
        terminal_complete = (
            config.ppo_settle_terminal_position
            and not any(quantity != 0.0 for quantity in final_quantities)
            and not remainders
            and len(replay.returns.values) == expected_periods
            and shared_cash_ppo.metrics.termination_count == 0
            and not diagnostics.termination_reasons
        )
        summary["shared_cash_ppo"] = {
            "name": shared_cash_ppo.name,
            "return_key": return_key,
            "metrics": _metrics_payload(shared_cash_ppo.metrics),
            "diagnostics": {
                "turnover_total": diagnostics.turnover_total,
                "total_cost": diagnostics.total_cost,
                "funding_pnl": diagnostics.funding_pnl,
                "borrow_cost": diagnostics.borrow_cost,
                "n_trades": diagnostics.n_trades,
                "rebalance_events": diagnostics.rebalance_events,
                "termination_reasons": list(diagnostics.termination_reasons),
            },
            "final_portfolio_value": float(replay.book.portfolio_value),
            "final_cash": float(replay.book.cash),
            "fill_count": replay.book.fill_count,
            "final_quantities": final_quantities,
            "active_order_remainders": [
                {"order_id": order_id, "remaining_quantity": float(quantity)}
                for order_id, quantity in remainders
            ],
            "terminal_order_reasons": [
                {"order_id": order_id, "reason": reason}
                for order_id, reason in ledger.terminal_order_reasons
            ],
            "terminal_settlement_complete": terminal_complete,
            "ledger_evidence": {
                "schema_version": ledger.schema_version,
                "digest": content_digest(ledger_payload),
                "interval_count": len(ledger.intervals),
                "decision_count": len(ledger.decisions),
                "terminal_exact_quantities": list(ledger.terminal_exact_quantities),
                "payload": ledger_payload,
            },
        }
    return summary, returns


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")


def publish_candidate_run(
    output_root: str | Path,
    result: CandidateRunResult,
    provenance: Mapping[str, object],
) -> PublishedCandidateRun:
    """Publish one exact three-file candidate artifact without overwriting."""

    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"candidate run destination already exists: {output}")
    provenance_payload = dict(provenance)
    _validate_provenance(provenance_payload)
    summary, returns = _result_payload(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        atomic_write_bytes(staging / "summary.json", _json_bytes(summary))
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **returns)
        atomic_write_bytes(staging / "returns.npz", buffer.getvalue())
        atomic_write_bytes(staging / "provenance.json", _json_bytes(provenance_payload))
        atomic_rename_directory(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return PublishedCandidateRun(
        root=output,
        summary_path=output / "summary.json",
        returns_path=output / "returns.npz",
        provenance_path=output / "provenance.json",
    )


def _validate_root(root: Path) -> tuple[Path, Path, Path]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("candidate artifact root must be a regular directory")
    for name in _REQUIRED_FILES:
        if (root / name).is_symlink():
            raise ValueError(
                f"candidate artifact {name} must be a regular file, not a symlink"
            )
    try:
        names = {entry.name for entry in root.iterdir()}
    except OSError as error:
        raise ValueError("candidate artifact root cannot be read") from error
    if names != _REQUIRED_FILES:
        raise ValueError(
            "candidate artifact root must contain exactly "
            "summary.json, returns.npz, provenance.json"
        )
    summary_path = root / "summary.json"
    returns_path = root / "returns.npz"
    provenance_path = root / "provenance.json"
    for path in (summary_path, returns_path, provenance_path):
        if not path.is_file():
            raise ValueError(
                f"candidate artifact {path.name} must be a regular file, not a symlink"
            )
    return summary_path, returns_path, provenance_path


def _verified_bytes(path: Path, *, label: str) -> tuple[bytes, str, int]:
    digest, size = file_digest_and_size(path, field=f"candidate {label}")
    payload = read_verified_bytes(
        path,
        expected_digest=digest,
        expected_size_bytes=size,
        field=f"candidate {label}",
    )
    return payload, digest, size


def _read_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        raw = cast(object, json.loads(payload.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed candidate {label} JSON") from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError(f"candidate {label} must be a JSON object")
    return cast(dict[str, object], raw)


def _validate_provenance(provenance: dict[str, object]) -> None:
    if provenance.get("schema_version") != PROVENANCE_SCHEMA:
        raise ValueError("unsupported candidate provenance schema")
    implementation = provenance.get("implementation")
    runtime = provenance.get("runtime_environment")
    if implementation is None or runtime is None:
        raise ValueError("candidate provenance manifests are required")
    if provenance.get("implementation_digest") != content_digest(implementation):
        raise ValueError("candidate provenance implementation digest mismatch")
    if provenance.get("runtime_environment_digest") != content_digest(runtime):
        raise ValueError("candidate provenance runtime environment digest mismatch")
    context = provenance.get("research_context_digest")
    if context is not None:
        if not isinstance(context, str):
            raise ValueError("research_context_digest must be a SHA-256 string")
        require_sha256(context, field="research_context_digest")


def _expected_return_keys(summary: dict[str, object]) -> frozenset[str]:
    by_symbol = summary.get("by_symbol")
    if not isinstance(by_symbol, list):
        raise ValueError("candidate summary by_symbol must be an array")
    keys: list[str] = []
    for symbol_entry in by_symbol:
        if not isinstance(symbol_entry, dict):
            raise ValueError("candidate summary symbol entry must be an object")
        strategies = symbol_entry.get("strategies")
        if not isinstance(strategies, list):
            raise ValueError("candidate summary strategies must be an array")
        for strategy in strategies:
            if not isinstance(strategy, dict):
                raise ValueError("candidate summary strategy entry must be an object")
            key = strategy.get("return_key")
            if not isinstance(key, str) or not key:
                raise ValueError("candidate summary return_key must be non-empty")
            keys.append(key)
    shared_cash_ppo = summary.get("shared_cash_ppo")
    if shared_cash_ppo is not None:
        if not isinstance(shared_cash_ppo, dict):
            raise ValueError("candidate shared-cash PPO result must be an object")
        key = shared_cash_ppo.get("return_key")
        if not isinstance(key, str) or not key:
            raise ValueError("candidate shared-cash return_key must be non-empty")
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError("candidate summary return keys must be unique")
    return frozenset(keys)


def _validate_ppo_training_evidence(
    summary: Mapping[str, object],
    *,
    result_schema: str,
) -> None:
    candidate_config = summary.get("candidate_config")
    if not isinstance(candidate_config, Mapping):
        raise ValueError("candidate PPO training config is malformed")
    requested = candidate_config.get("ppo_total_timesteps")
    realized = candidate_config.get("ppo_training_timesteps")
    layout = candidate_config.get("ppo_training_layout")
    rollout_steps = candidate_config.get("ppo_rollout_steps_per_env")
    fit_symbols = candidate_config.get("fit_symbol_indices")
    minimum_hold_bars = candidate_config.get("ppo_minimum_hold_bars", 0)
    observation_schema = candidate_config.get(
        "ppo_observation_schema", "ppo_observation_v2"
    )
    settle_terminal_position = candidate_config.get(
        "ppo_settle_terminal_position", False
    )
    if (
        isinstance(requested, bool)
        or not isinstance(requested, int)
        or isinstance(realized, bool)
        or not isinstance(realized, int)
        or not isinstance(layout, str)
        or (rollout_steps is not None and isinstance(rollout_steps, bool))
        or (rollout_steps is not None and not isinstance(rollout_steps, int))
        or not isinstance(fit_symbols, list)
        or not fit_symbols
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in fit_symbols
        )
    ):
        raise ValueError("candidate PPO training evidence is malformed")
    if len(set(fit_symbols)) != len(fit_symbols):
        raise ValueError("candidate PPO fit symbol roster is duplicated")
    expected = expected_ppo_realized_timesteps(
        requested,
        training_layout=layout,
        rollout_steps_per_env=rollout_steps,
        n_envs=len(fit_symbols),
    )
    if realized != expected:
        raise ValueError("candidate PPO realized timesteps do not match rollout budget")
    if result_schema in {
        _RESULT_SCHEMA_V4,
        _RESULT_SCHEMA_V5,
        _RESULT_SCHEMA_V6,
        _RESULT_SCHEMA_V7,
        _RESULT_SCHEMA_V8,
        _RESULT_SCHEMA_V9,
        _RESULT_SCHEMA_V10,
        _RESULT_SCHEMA_V11,
    }:
        if not {
            "ppo_minimum_hold_bars",
            "ppo_observation_schema",
            "ppo_settle_terminal_position",
            "ppo_training_minimum_hold_suppressed_count",
        }.issubset(candidate_config):
            raise ValueError("candidate PPO duration config is incomplete")
        if (
            isinstance(minimum_hold_bars, bool)
            or not isinstance(minimum_hold_bars, int)
            or minimum_hold_bars < 0
        ):
            raise ValueError("candidate PPO minimum hold duration is malformed")
        if not isinstance(observation_schema, str):
            raise ValueError("candidate PPO observation schema is malformed")
        if not isinstance(settle_terminal_position, bool):
            raise ValueError("candidate PPO terminal settlement is malformed")
        suppressed_count = candidate_config.get(
            "ppo_training_minimum_hold_suppressed_count"
        )
        if (
            isinstance(suppressed_count, bool)
            or not isinstance(suppressed_count, int)
            or suppressed_count < 0
        ):
            raise ValueError("candidate PPO suppression count is malformed")
        if minimum_hold_bars > 0 and observation_schema != "ppo_observation_v3":
            raise ValueError(
                "candidate PPO duration requires the age-aware observation"
            )
        evaluation = summary.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise ValueError("candidate evaluation config is malformed")
        if (
            evaluation.get("ppo_settle_terminal_position")
            is not settle_terminal_position
        ):
            raise ValueError("candidate terminal settlement config is inconsistent")
        if summary.get("ppo_observation") != ppo_observation_contract_payload(
            observation_schema
        ):
            raise ValueError("candidate PPO observation contract mismatch")
    if result_schema in {
        _RESULT_SCHEMA_V5,
        _RESULT_SCHEMA_V6,
        _RESULT_SCHEMA_V7,
        _RESULT_SCHEMA_V8,
        _RESULT_SCHEMA_V9,
        _RESULT_SCHEMA_V10,
        _RESULT_SCHEMA_V11,
    }:
        if "pretrade_risk_config" not in candidate_config:
            raise ValueError("candidate PPO risk config is incomplete")
        risk_config = candidate_config["pretrade_risk_config"]
        if risk_config is not None:
            if not isinstance(risk_config, Mapping):
                raise ValueError("candidate pre-trade risk config is malformed")
            required_risk_fields = {
                "max_gross",
                "max_abs_weight",
                "max_turnover",
                "drawdown_start",
                "drawdown_stop",
                "emergency_turnover_override",
                "fail_closed_tolerance",
            }
            if set(risk_config) != required_risk_fields:
                raise ValueError("candidate pre-trade risk config is malformed")
            numeric_fields = required_risk_fields - {
                "max_turnover",
                "emergency_turnover_override",
            }
            if any(
                isinstance(risk_config[name], bool)
                or not isinstance(risk_config[name], (int, float))
                or not math.isfinite(float(risk_config[name]))
                for name in numeric_fields
            ):
                raise ValueError("candidate pre-trade risk config is malformed")
            max_turnover = risk_config["max_turnover"]
            if max_turnover is not None and (
                isinstance(max_turnover, bool)
                or not isinstance(max_turnover, (int, float))
                or not math.isfinite(float(max_turnover))
            ):
                raise ValueError("candidate pre-trade risk config is malformed")
            if not isinstance(risk_config["emergency_turnover_override"], bool):
                raise ValueError("candidate pre-trade risk config is malformed")
            try:
                PreTradeRiskConfig(**dict(risk_config))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "candidate pre-trade risk config is malformed"
                ) from error
        evaluation = summary.get("evaluation")
        if (
            not isinstance(evaluation, Mapping)
            or evaluation.get("pretrade_risk_config") != risk_config
        ):
            raise ValueError("candidate pre-trade risk config is inconsistent")
        if observation_schema == PPO_OBSERVATION_SCHEMA_V3:
            if not settle_terminal_position:
                raise ValueError(
                    "age-aware PPO comparison requires terminal settlement"
                )
            if risk_config is None:
                raise ValueError(
                    "age-aware PPO comparison requires explicit pre-trade risk config"
                )
            if float(risk_config["drawdown_stop"]) > 0.20:
                raise ValueError("PPO drawdown stop must not exceed 20%")


def _validate_replay_evidence(
    summary: Mapping[str, object],
    *,
    require_full_coverage: bool,
) -> None:
    symbols = summary.get("symbols")
    by_symbol = summary.get("by_symbol")
    evaluation = summary.get("evaluation")
    if not isinstance(symbols, list) or not isinstance(by_symbol, list):
        raise ValueError("candidate replay evidence is malformed")
    if not isinstance(evaluation, Mapping):
        raise ValueError("candidate replay evidence is malformed")
    terminal_settlement = evaluation.get("ppo_settle_terminal_position")
    if not isinstance(terminal_settlement, bool):
        raise ValueError("candidate replay evidence is malformed")
    expected_periods: int | None = None
    if require_full_coverage:
        raw_expected_periods = evaluation.get("expected_periods")
        if (
            isinstance(raw_expected_periods, bool)
            or not isinstance(raw_expected_periods, int)
            or raw_expected_periods <= 0
        ):
            raise ValueError("candidate replay expected-period evidence is malformed")
        expected_periods = raw_expected_periods

    replay_fields = {
        "final_quantities",
        "active_order_remainders",
        "terminal_order_reasons",
        "terminal_settlement_complete",
        "minimum_hold_audit",
    }
    audit_fields = {
        "index",
        "requested_intent",
        "effective_intent",
        "position_age_bars",
        "position_age_bars_after",
        "position_quantity_before",
        "position_quantity_after",
        "target_weight",
        "minimum_hold_suppressed",
        "minimum_hold_unlocked",
        "risk_reasons",
    }
    for symbol_entry in by_symbol:
        if not isinstance(symbol_entry, Mapping):
            raise ValueError("candidate replay evidence is malformed")
        strategies = symbol_entry.get("strategies")
        if not isinstance(strategies, list):
            raise ValueError("candidate replay evidence is malformed")
        for strategy in strategies:
            if not isinstance(strategy, Mapping) or not replay_fields.issubset(
                strategy
            ):
                raise ValueError("candidate replay evidence is incomplete")
            n_periods: int | None = None
            termination_count: int | None = None
            termination_reasons: list[object] | None = None
            if require_full_coverage:
                metrics = strategy.get("metrics")
                diagnostics = strategy.get("diagnostics")
                if not isinstance(metrics, Mapping) or not isinstance(
                    diagnostics, Mapping
                ):
                    raise ValueError("candidate replay metrics are malformed")
                raw_n_periods = metrics.get("n_periods")
                raw_termination_count = metrics.get("termination_count")
                raw_termination_reasons = diagnostics.get("termination_reasons")
                if (
                    isinstance(raw_n_periods, bool)
                    or not isinstance(raw_n_periods, int)
                    or raw_n_periods < 0
                    or isinstance(raw_termination_count, bool)
                    or not isinstance(raw_termination_count, int)
                    or raw_termination_count < 0
                    or not isinstance(raw_termination_reasons, list)
                    or any(
                        not isinstance(reason, str) or not reason
                        for reason in raw_termination_reasons
                    )
                ):
                    raise ValueError(
                        "candidate replay completion evidence is malformed"
                    )
                n_periods = raw_n_periods
                termination_count = raw_termination_count
                termination_reasons = raw_termination_reasons
            quantities = strategy["final_quantities"]
            remainders = strategy["active_order_remainders"]
            terminal_reasons = strategy["terminal_order_reasons"]
            terminal_complete = strategy["terminal_settlement_complete"]
            audit = strategy["minimum_hold_audit"]
            if (
                not isinstance(quantities, list)
                or len(quantities) != len(symbols)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in quantities
                )
                or not isinstance(remainders, list)
                or not isinstance(terminal_reasons, list)
                or not isinstance(terminal_complete, bool)
                or not isinstance(audit, list)
            ):
                raise ValueError("candidate replay evidence is malformed")
            for order in remainders:
                if (
                    not isinstance(order, Mapping)
                    or set(order) != {"order_id", "remaining_quantity"}
                    or not isinstance(order["order_id"], str)
                    or not order["order_id"]
                    or isinstance(order["remaining_quantity"], bool)
                    or not isinstance(order["remaining_quantity"], (int, float))
                    or not math.isfinite(float(order["remaining_quantity"]))
                ):
                    raise ValueError("candidate replay order remainder is malformed")
            for terminal_reason in terminal_reasons:
                if (
                    not isinstance(terminal_reason, Mapping)
                    or set(terminal_reason) != {"order_id", "reason"}
                    or not isinstance(terminal_reason["order_id"], str)
                    or not terminal_reason["order_id"]
                    or not isinstance(terminal_reason["reason"], str)
                    or not terminal_reason["reason"]
                ):
                    raise ValueError("candidate terminal order evidence is malformed")
            expected_terminal_complete = (
                terminal_settlement
                and all(float(value) == 0.0 for value in quantities)
                and not remainders
            )
            if require_full_coverage:
                expected_terminal_complete = (
                    expected_terminal_complete
                    and n_periods == expected_periods
                    and termination_count == 0
                    and not termination_reasons
                )
            if terminal_complete is not expected_terminal_complete:
                raise ValueError(
                    "candidate terminal settlement evidence is inconsistent"
                )
            for event in audit:
                if (
                    not isinstance(event, Mapping)
                    or set(event) != audit_fields
                    or isinstance(event["index"], bool)
                    or not isinstance(event["index"], int)
                    or event["index"] < 0
                    or any(
                        isinstance(event[field], bool)
                        or not isinstance(event[field], int)
                        or event[field] < 0
                        for field in ("position_age_bars", "position_age_bars_after")
                    )
                    or any(
                        isinstance(event[field], bool)
                        or not isinstance(event[field], int)
                        or event[field] not in {-1, 0, 1}
                        for field in ("requested_intent", "effective_intent")
                    )
                    or any(
                        isinstance(event[field], bool)
                        or not isinstance(event[field], (int, float))
                        or not math.isfinite(float(event[field]))
                        for field in (
                            "position_quantity_before",
                            "position_quantity_after",
                            "target_weight",
                        )
                    )
                    or not isinstance(event["minimum_hold_suppressed"], bool)
                    or not isinstance(event["minimum_hold_unlocked"], bool)
                    or not (
                        event["minimum_hold_suppressed"]
                        or event["minimum_hold_unlocked"]
                    )
                    or not isinstance(event["risk_reasons"], list)
                    or any(
                        not isinstance(reason, str) for reason in event["risk_reasons"]
                    )
                ):
                    raise ValueError("candidate minimum-hold audit is malformed")


def _validate_v5_replay_evidence(summary: Mapping[str, object]) -> None:
    _validate_replay_evidence(summary, require_full_coverage=False)


def _validate_v6_replay_evidence(summary: Mapping[str, object]) -> None:
    _validate_replay_evidence(summary, require_full_coverage=True)


def _validate_v7_replay_evidence(
    summary: Mapping[str, object],
    *,
    require_ledger_payload: bool = False,
    strict_ledger_semantics: bool = False,
) -> None:
    _validate_v6_replay_evidence(summary)
    symbols = summary.get("symbols")
    evaluation = summary.get("evaluation")
    portfolio = summary.get("shared_cash_ppo")
    if (
        not isinstance(symbols, list)
        or not isinstance(evaluation, Mapping)
        or not isinstance(portfolio, Mapping)
    ):
        raise ValueError("candidate shared-cash PPO evidence is malformed")
    expected_periods = evaluation.get("expected_periods")
    if (
        isinstance(expected_periods, bool)
        or not isinstance(expected_periods, int)
        or expected_periods <= 0
    ):
        raise ValueError("candidate shared-cash period count is malformed")
    expected_fields = {
        "name",
        "return_key",
        "metrics",
        "diagnostics",
        "final_portfolio_value",
        "final_cash",
        "fill_count",
        "final_quantities",
        "active_order_remainders",
        "terminal_order_reasons",
        "terminal_settlement_complete",
        "ledger_evidence",
    }
    if set(portfolio) != expected_fields:
        raise ValueError("candidate shared-cash PPO evidence fields are malformed")
    if portfolio["name"] != "ppo" or portfolio["return_key"] != "shared_cash_ppo":
        raise ValueError("candidate shared-cash PPO identity is malformed")
    metrics = portfolio["metrics"]
    diagnostics = portfolio["diagnostics"]
    if not isinstance(metrics, Mapping) or not isinstance(diagnostics, Mapping):
        raise ValueError("candidate shared-cash metrics are malformed")
    metric_fields = {
        "total_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "turnover_total",
        "total_cost",
        "funding_pnl",
        "borrow_cost",
        "n_trades",
        "rebalance_events",
        "termination_count",
        "n_periods",
        "return_kind",
        "periods_per_year",
    }
    diagnostic_fields = {
        "turnover_total",
        "total_cost",
        "funding_pnl",
        "borrow_cost",
        "n_trades",
        "rebalance_events",
        "termination_reasons",
    }
    if set(metrics) != metric_fields or set(diagnostics) != diagnostic_fields:
        raise ValueError("candidate shared-cash metrics are incomplete")
    if metrics["return_kind"] != "base_bar":
        raise ValueError("candidate shared-cash returns must use base bars")
    if metrics["n_periods"] != expected_periods:
        raise ValueError("candidate shared-cash return period count is inconsistent")
    for field in (
        "total_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "turnover_total",
        "total_cost",
        "funding_pnl",
        "borrow_cost",
    ):
        value = metrics[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("candidate shared-cash metric is malformed")
    termination_count = metrics["termination_count"]
    if (
        isinstance(termination_count, bool)
        or not isinstance(termination_count, int)
        or termination_count < 0
    ):
        raise ValueError("candidate shared-cash termination count is malformed")
    reasons = diagnostics["termination_reasons"]
    if not isinstance(reasons, list) or any(
        not isinstance(reason, str) or not reason for reason in reasons
    ):
        raise ValueError("candidate shared-cash termination reasons are malformed")
    quantities = portfolio["final_quantities"]
    remainders = portfolio["active_order_remainders"]
    terminal_reasons = portfolio["terminal_order_reasons"]
    terminal_complete = portfolio["terminal_settlement_complete"]
    if (
        not isinstance(quantities, list)
        or len(quantities) != len(symbols)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in quantities
        )
        or not isinstance(remainders, list)
        or not isinstance(terminal_reasons, list)
        or not isinstance(terminal_complete, bool)
    ):
        raise ValueError("candidate shared-cash terminal evidence is malformed")
    for order in remainders:
        if (
            not isinstance(order, Mapping)
            or set(order) != {"order_id", "remaining_quantity"}
            or not isinstance(order["order_id"], str)
            or not order["order_id"]
            or isinstance(order["remaining_quantity"], bool)
            or not isinstance(order["remaining_quantity"], (int, float))
            or not math.isfinite(float(order["remaining_quantity"]))
        ):
            raise ValueError("candidate shared-cash order remainder is malformed")
    for order in terminal_reasons:
        if (
            not isinstance(order, Mapping)
            or set(order) != {"order_id", "reason"}
            or not isinstance(order["order_id"], str)
            or not order["order_id"]
            or not isinstance(order["reason"], str)
            or not order["reason"]
        ):
            raise ValueError("candidate shared-cash terminal order is malformed")
    expected_terminal_complete = (
        evaluation.get("ppo_settle_terminal_position") is True
        and all(float(value) == 0.0 for value in quantities)
        and not remainders
        and metrics["n_periods"] == expected_periods
        and termination_count == 0
        and not reasons
    )
    if terminal_complete is not expected_terminal_complete:
        raise ValueError("candidate shared-cash settlement evidence is inconsistent")
    for field in ("final_portfolio_value", "final_cash"):
        value = portfolio[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("candidate shared-cash final account value is malformed")
    fill_count = portfolio["fill_count"]
    if (
        isinstance(fill_count, bool)
        or not isinstance(fill_count, int)
        or fill_count < 0
    ):
        raise ValueError("candidate shared-cash fill count is malformed")
    ledger = portfolio["ledger_evidence"]
    ledger_fields = {
        "schema_version",
        "digest",
        "interval_count",
        "decision_count",
        "terminal_exact_quantities",
    }
    if require_ledger_payload:
        ledger_fields.add("payload")
    if not isinstance(ledger, Mapping) or set(ledger) != ledger_fields:
        raise ValueError("candidate shared-cash ledger evidence is malformed")
    if ledger["schema_version"] not in {
        "shared_cash_replay_ledger_v1",
        "shared_cash_replay_ledger_v2",
        *(
            ("shared_cash_replay_ledger_v3", "shared_cash_replay_ledger_v4")
            if strict_ledger_semantics
            else ()
        ),
    }:
        raise ValueError("candidate shared-cash ledger schema is unsupported")
    digest = ledger["digest"]
    if not isinstance(digest, str):
        raise ValueError("candidate shared-cash ledger digest is malformed")
    require_sha256(digest, field="candidate shared-cash ledger digest")
    for field in ("interval_count", "decision_count"):
        value = ledger[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("candidate shared-cash ledger count is malformed")
    if (
        ledger["interval_count"] != expected_periods
        or ledger["decision_count"] > expected_periods
    ):
        raise ValueError("candidate shared-cash ledger coverage is incomplete")
    exact_quantities = ledger["terminal_exact_quantities"]
    if not isinstance(exact_quantities, list) or len(exact_quantities) != len(symbols):
        raise ValueError(
            "candidate shared-cash exact terminal quantities are malformed"
        )
    if any(not isinstance(value, str) for value in exact_quantities):
        raise ValueError(
            "candidate shared-cash exact terminal quantities are malformed"
        )
    if terminal_complete and any(float(value) != 0.0 for value in exact_quantities):
        raise ValueError("candidate shared-cash exact terminal quantities are nonzero")
    if require_ledger_payload:
        payload = ledger.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("candidate shared-cash ledger payload is malformed")
        _validate_shared_cash_ledger_payload(
            payload,
            ledger=ledger,
            summary=summary,
            symbols=symbols,
            expected_periods=expected_periods,
            strict_semantics=strict_ledger_semantics,
        )


def _validate_shared_cash_ledger_payload(
    payload: Mapping[str, object],
    *,
    ledger: Mapping[str, object],
    summary: Mapping[str, object],
    symbols: list[object],
    expected_periods: int,
    strict_semantics: bool = False,
) -> None:
    schema = ledger["schema_version"]
    expected_fields = {
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
        "schema_version",
        "start_index",
        "stop_index",
        "terminal_exact_quantities",
        "terminal_order_reasons",
        "termination_reason",
    }
    if schema == "shared_cash_replay_ledger_v2":
        expected_fields.add("decisions")
    elif schema in {
        "shared_cash_replay_ledger_v3",
        "shared_cash_replay_ledger_v4",
    }:
        expected_fields.update(
            {"contract_multipliers", "decisions", "initial_mark_prices"}
        )
    if set(payload) != expected_fields:
        raise ValueError("candidate shared-cash ledger payload fields are malformed")
    if payload["schema_version"] != schema:
        raise ValueError("candidate shared-cash ledger payload schema is inconsistent")
    if payload["dataset_id"] != summary.get("dataset_id"):
        raise ValueError(
            "candidate shared-cash ledger dataset identity is inconsistent"
        )
    if schema in {
        "shared_cash_replay_ledger_v3",
        "shared_cash_replay_ledger_v4",
    }:
        for field in ("contract_multipliers", "initial_mark_prices"):
            values = payload[field]
            if (
                not isinstance(values, (list, tuple))
                or len(values) != len(symbols)
                or any(
                    not _is_finite_number(value) or float(value) <= 0.0
                    for value in values
                )
            ):
                raise ValueError(
                    "candidate shared-cash ledger market vectors are malformed"
                )
    digest = ledger.get("digest")
    try:
        payload_digest = content_digest(payload)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "candidate shared-cash ledger payload is not canonical"
        ) from error
    if payload_digest != digest:
        raise ValueError("candidate shared-cash ledger digest does not match payload")
    policy_digest = payload["execution_policy_digest"]
    if not isinstance(policy_digest, str):
        raise ValueError("candidate shared-cash ledger policy digest is malformed")
    require_sha256(policy_digest, field="candidate shared-cash execution policy digest")

    start = payload["start_index"]
    stop = payload["stop_index"]
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(stop, bool)
        or not isinstance(stop, int)
        or stop - start != expected_periods
    ):
        raise ValueError("candidate shared-cash ledger index coverage is malformed")
    intervals = payload["intervals"]
    if (
        not isinstance(intervals, (list, tuple))
        or len(intervals) != expected_periods
        or len(intervals) != ledger["interval_count"]
    ):
        raise ValueError("candidate shared-cash ledger intervals are incomplete")
    interval_fields = {
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
    }
    if schema in {
        "shared_cash_replay_ledger_v3",
        "shared_cash_replay_ledger_v4",
    }:
        interval_fields.add("accounting_transitions")
    previous_next_index = start
    for interval in intervals:
        if not isinstance(interval, Mapping) or set(interval) != interval_fields:
            raise ValueError("candidate shared-cash ledger interval is malformed")
        interval_start = interval["start_index"]
        interval_next = interval["next_index"]
        if (
            isinstance(interval_start, bool)
            or not isinstance(interval_start, int)
            or interval_start != previous_next_index
            or isinstance(interval_next, bool)
            or not isinstance(interval_next, int)
            or interval_next != interval_start + 1
        ):
            raise ValueError("candidate shared-cash ledger interval chain is broken")
        previous_next_index = interval_next
        for field in ("exact_quantities_before", "exact_quantities_after"):
            values = interval[field]
            if (
                not isinstance(values, (list, tuple))
                or len(values) != len(symbols)
                or any(not isinstance(value, str) for value in values)
            ):
                raise ValueError(
                    "candidate shared-cash ledger quantities are malformed"
                )
        for field in ("order_events", "capacity_events", "funding_events"):
            if not isinstance(interval[field], (list, tuple)):
                raise ValueError("candidate shared-cash ledger events are malformed")
        if schema in {
            "shared_cash_replay_ledger_v3",
            "shared_cash_replay_ledger_v4",
        } and not isinstance(interval["accounting_transitions"], (list, tuple)):
            raise ValueError(
                "candidate shared-cash accounting transitions are malformed"
            )
    if previous_next_index != stop:
        raise ValueError("candidate shared-cash ledger interval chain is incomplete")

    if strict_semantics:
        interval_financial_fields = (
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
        )
        for interval in intervals:
            assert isinstance(interval, Mapping)
            if any(
                not _is_finite_number(interval[field])
                for field in interval_financial_fields
            ):
                raise ValueError(
                    "candidate shared-cash ledger interval financial value is malformed"
                )
        for field in (
            "final_borrow_cost",
            "final_cash",
            "final_funding_pnl",
            "final_max_drawdown",
            "final_portfolio_value",
            "final_total_cost",
            "final_turnover_total",
        ):
            if not _is_finite_number(payload[field]):
                raise ValueError(
                    "candidate shared-cash ledger final value is malformed"
                )

        _validate_shared_cash_ledger_continuity(intervals)

    terminal_quantities = payload["terminal_exact_quantities"]
    if (
        not isinstance(terminal_quantities, (list, tuple))
        or len(terminal_quantities) != len(symbols)
        or any(not isinstance(value, str) for value in terminal_quantities)
        or list(terminal_quantities) != ledger["terminal_exact_quantities"]
    ):
        raise ValueError(
            "candidate shared-cash ledger terminal quantities are malformed"
        )
    if schema in {
        "shared_cash_replay_ledger_v2",
        "shared_cash_replay_ledger_v3",
        "shared_cash_replay_ledger_v4",
    }:
        decisions = payload["decisions"]
        if (
            not isinstance(decisions, (list, tuple))
            or len(decisions) != ledger["decision_count"]
            or len(decisions) > expected_periods
        ):
            raise ValueError("candidate shared-cash ledger decisions are incomplete")
        decision_fields = {
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
        }
        previous_decision_index = start - 1
        intent_values = {-1, 0, 1}
        symbol_sequences = (
            "changed_intents",
            "effective_intents",
            "intents",
            "minimum_hold_suppressed",
            "minimum_hold_unlocked",
            "position_age_bars_after",
            "position_age_bars_before",
            "position_quantity_after",
            "position_quantity_before",
            "proposal_weights",
            "target_weights",
        )
        for decision in decisions:
            if not isinstance(decision, Mapping) or set(decision) != decision_fields:
                raise ValueError("candidate shared-cash ledger decision is malformed")
            index = decision["index"]
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not start <= index < stop
                or index <= previous_decision_index
            ):
                raise ValueError(
                    "candidate shared-cash ledger decision index is invalid"
                )
            previous_decision_index = index
            for field in symbol_sequences:
                values = decision[field]
                if not isinstance(values, (list, tuple)) or len(values) != len(symbols):
                    raise ValueError(
                        "candidate shared-cash ledger decision roster is malformed"
                    )
            for field in ("intents", "effective_intents"):
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value not in intent_values
                    for value in decision[field]
                ):
                    raise ValueError("candidate shared-cash ledger intent is invalid")
            for field in (
                "changed_intents",
                "minimum_hold_suppressed",
                "minimum_hold_unlocked",
            ):
                if any(not isinstance(value, bool) for value in decision[field]):
                    raise ValueError(
                        "candidate shared-cash ledger action flags are malformed"
                    )
            for field in ("position_age_bars_before", "position_age_bars_after"):
                if any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in decision[field]
                ):
                    raise ValueError("candidate shared-cash ledger ages are malformed")
            for field in (
                "position_quantity_before",
                "position_quantity_after",
                "proposal_weights",
                "target_weights",
            ):
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in decision[field]
                ):
                    raise ValueError(
                        "candidate shared-cash ledger decision values are malformed"
                    )
            if not isinstance(decision["risk_reasons"], (list, tuple)) or any(
                not isinstance(reason, str) for reason in decision["risk_reasons"]
            ):
                raise ValueError(
                    "candidate shared-cash ledger risk reasons are malformed"
                )
        if strict_semantics:
            _validate_v9_evaluation_index_link(
                start=start,
                stop=stop,
                evaluation=summary.get("evaluation"),
            )
            _validate_v9_initial_ledger_link(
                intervals=intervals,
                evaluation=summary.get("evaluation"),
            )
            _validate_v9_interval_accounting(
                intervals,
                symbols=symbols,
                dataset_id=cast(str, payload["dataset_id"]),
                execution_policy_digest=policy_digest,
            )
            _validate_v9_decision_coverage(
                decisions,
                start=start,
                stop=stop,
                evaluation=summary.get("evaluation"),
            )
            _validate_v9_decision_interval_links(
                decisions,
                intervals=intervals,
                start=start,
            )
            _validate_shared_cash_terminal_links(
                payload,
                intervals=intervals,
                summary=summary,
            )


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _numbers_are_close(left: object, right: object) -> bool:
    if (
        isinstance(left, bool)
        or not isinstance(left, (int, float))
        or not math.isfinite(float(left))
        or isinstance(right, bool)
        or not isinstance(right, (int, float))
        or not math.isfinite(float(right))
    ):
        return False
    return math.isclose(
        float(left),
        float(right),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _validate_shared_cash_ledger_continuity(
    intervals: Sequence[object],
) -> None:
    state_fields = (
        ("cash_after", "cash_before"),
        ("portfolio_value_after", "portfolio_value_before"),
        ("total_cost_after", "total_cost_before"),
        ("funding_pnl_after", "funding_pnl_before"),
        ("borrow_cost_after", "borrow_cost_before"),
        ("turnover_total_after", "turnover_total_before"),
        ("max_drawdown_after", "max_drawdown_before"),
    )
    for previous, current in zip(intervals, intervals[1:], strict=False):
        if not isinstance(previous, Mapping) or not isinstance(current, Mapping):
            raise ValueError("candidate shared-cash ledger interval is malformed")
        if previous["exact_quantities_after"] != current["exact_quantities_before"]:
            raise ValueError(
                "candidate shared-cash ledger interval continuity is broken"
            )
        if any(
            not _numbers_are_close(previous[after], current[before])
            for after, before in state_fields
        ):
            raise ValueError(
                "candidate shared-cash ledger interval continuity is broken"
            )


def _validate_v9_evaluation_index_link(
    *,
    start: int,
    stop: int,
    evaluation: object,
) -> None:
    if not isinstance(evaluation, Mapping):
        raise ValueError("candidate shared-cash evaluation index link is malformed")
    persisted_start = evaluation.get("evaluation_start_index")
    persisted_stop = evaluation.get("evaluation_stop_index")
    if (
        isinstance(persisted_start, bool)
        or not isinstance(persisted_start, int)
        or persisted_start != start
        or isinstance(persisted_stop, bool)
        or not isinstance(persisted_stop, int)
        or persisted_stop != stop
    ):
        raise ValueError("candidate shared-cash evaluation index link is inconsistent")


def _validate_v9_initial_ledger_link(
    *,
    intervals: Sequence[object],
    evaluation: object,
) -> None:
    if not isinstance(evaluation, Mapping) or not intervals:
        raise ValueError("candidate shared-cash initial ledger link is malformed")
    initial_capital = evaluation.get("initial_capital")
    first_interval = intervals[0]
    if (
        not _is_finite_number(initial_capital)
        or float(cast(int | float, initial_capital)) <= 0.0
        or not isinstance(first_interval, Mapping)
        or not _numbers_are_close(first_interval.get("cash_before"), initial_capital)
        or not _numbers_are_close(
            first_interval.get("portfolio_value_before"), initial_capital
        )
    ):
        raise ValueError("candidate shared-cash initial ledger link is inconsistent")
    quantities_before = first_interval.get("exact_quantities_before")
    if not isinstance(quantities_before, (list, tuple)) or any(
        value != "0" for value in quantities_before
    ):
        raise ValueError("candidate shared-cash initial ledger link is inconsistent")


def _validate_v9_interval_accounting(
    intervals: Sequence[object],
    *,
    symbols: Sequence[object],
    dataset_id: str,
    execution_policy_digest: str,
) -> None:
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
    fill_event_types = {"filled", "partial_fill"}

    for interval in intervals:
        if not isinstance(interval, Mapping):
            raise ValueError("candidate shared-cash interval accounting is malformed")
        start_index = cast(int, interval["start_index"])
        next_index = cast(int, interval["next_index"])
        portfolio_value_before = cast(float, interval["portfolio_value_before"])
        portfolio_value_after = cast(float, interval["portfolio_value_after"])
        if portfolio_value_before <= 0.0:
            raise ValueError(
                "candidate shared-cash interval return accounting is inconsistent"
            )
        expected_net_return = max(
            max(portfolio_value_after, 0.0) / portfolio_value_before - 1.0,
            -1.0 + 1e-12,
        )
        if not _numbers_are_close(interval["interval_net_return"], expected_net_return):
            raise ValueError(
                "candidate shared-cash interval return accounting is inconsistent"
            )

        cumulative_links = (
            ("total_cost_after", "total_cost_before", "interval_cost"),
            ("funding_pnl_after", "funding_pnl_before", "interval_funding"),
            ("borrow_cost_after", "borrow_cost_before", "interval_borrow_cost"),
        )
        if any(
            not _numbers_are_close(
                cast(float, interval[after]) - cast(float, interval[before]),
                interval[amount],
            )
            for after, before, amount in cumulative_links
        ):
            raise ValueError(
                "candidate shared-cash interval financial accounting is inconsistent"
            )

        order_events = interval["order_events"]
        fill_notionals: list[float] = []
        for event_index, raw_event in enumerate(order_events):
            if not isinstance(raw_event, Mapping):
                raise ValueError("candidate shared-cash order event is malformed")
            try:
                event = OrderEvent.from_mapping(cast(Mapping[str, object], raw_event))
            except (OverflowError, TypeError, ValueError) as error:
                raise ValueError(
                    "candidate shared-cash order event is malformed"
                ) from error
            if (
                event.sequence != event_index
                or event.dataset_id != dataset_id
                or event.execution_policy_digest != execution_policy_digest
                or not 0 <= event.symbol_index < len(symbols)
                or not start_index <= event.processing_index <= next_index
            ):
                raise ValueError(
                    "candidate shared-cash order event identity is inconsistent"
                )
            is_fill = event.event_type in fill_event_types
            if is_fill:
                if (
                    event.filled_quantity == 0.0
                    or event.execution_price is None
                    or event.filled_notional <= 0.0
                    or event.filled_quantity * event.requested_quantity <= 0.0
                    or abs(event.filled_quantity)
                    > abs(event.requested_quantity) + 1e-12
                ):
                    raise ValueError(
                        "candidate shared-cash order event fill is inconsistent"
                    )
                fill_notionals.append(event.filled_notional)
            elif (
                event.filled_quantity != 0.0
                or event.execution_price is not None
                or event.filled_notional != 0.0
            ):
                raise ValueError(
                    "candidate shared-cash order event fill is inconsistent"
                )

        capacity_events = interval["capacity_events"]
        consumed_capacity: list[float] = []
        capacity_tolerances: list[float] = []
        for raw_capacity in capacity_events:
            if (
                not isinstance(raw_capacity, Mapping)
                or set(raw_capacity) != capacity_fields
                or any(
                    not _is_finite_number(raw_capacity[field])
                    for field in capacity_fields
                )
            ):
                raise ValueError("candidate shared-cash capacity event is malformed")
            capacity = {
                field: float(cast(int | float, raw_capacity[field]))
                for field in capacity_fields
            }
            initial = capacity["initial_capacity_notional"]
            remaining = capacity["remaining_capacity_notional"]
            market_notional = capacity["market_notional"]
            participation_limit = capacity["participation_limit"]
            if (
                capacity["processing_volume"] < 0.0
                or capacity["capacity_reference_price"] <= 0.0
                or capacity["contract_multiplier"] <= 0.0
                or not 0.0 <= participation_limit <= 1.0
                or market_notional < 0.0
                or initial < 0.0
                or capacity["consumed_capacity_notional"] < 0.0
                or remaining < 0.0
            ):
                raise ValueError("candidate shared-cash capacity event is malformed")
            tolerance = max(
                1e-9,
                8.0 * math.ulp(initial),
                8.0 * math.ulp(remaining),
            )
            if not _numbers_are_close(
                initial, market_notional * participation_limit
            ) or not math.isclose(
                capacity["consumed_capacity_notional"] + remaining,
                initial,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise ValueError(
                    "candidate shared-cash capacity event accounting is inconsistent"
                )
            consumed_capacity.append(capacity["consumed_capacity_notional"])
            capacity_tolerances.append(tolerance)

        try:
            total_fill_notional = math.fsum(fill_notionals)
            total_consumed_capacity = math.fsum(consumed_capacity)
            aggregate_capacity_tolerance = max(
                1e-9,
                math.fsum(capacity_tolerances),
            )
        except OverflowError as error:
            raise ValueError(
                "candidate shared-cash capacity event accounting is inconsistent"
            ) from error
        if not consumed_capacity and total_fill_notional != 0.0:
            raise ValueError(
                "candidate shared-cash execution capacity link is inconsistent"
            )
        if not math.isclose(
            total_fill_notional,
            total_consumed_capacity,
            rel_tol=0.0,
            abs_tol=aggregate_capacity_tolerance,
        ):
            raise ValueError(
                "candidate shared-cash execution capacity link is inconsistent"
            )

        funding_amounts: list[float] = []
        for raw_funding in interval["funding_events"]:
            if not isinstance(raw_funding, Mapping):
                raise ValueError("candidate shared-cash funding event is malformed")
            try:
                funding_event = FundingBoundaryEvidence.from_mapping(
                    cast(Mapping[str, object], raw_funding)
                )
            except (OverflowError, TypeError, ValueError) as error:
                raise ValueError(
                    "candidate shared-cash funding event is malformed"
                ) from error
            if funding_event.processing_index != next_index or len(
                funding_event.funding_due
            ) != len(symbols):
                raise ValueError(
                    "candidate shared-cash funding event identity is inconsistent"
                )
            funding_amounts.append(funding_event.funding_amount)
        try:
            evidenced_funding = math.fsum(funding_amounts)
        except OverflowError as error:
            raise ValueError(
                "candidate shared-cash funding event accounting is inconsistent"
            ) from error
        if not _numbers_are_close(
            evidenced_funding, interval["interval_funding"]
        ) or not _numbers_are_close(
            cast(float, interval["turnover_total_after"])
            - cast(float, interval["turnover_total_before"]),
            total_fill_notional / max(portfolio_value_before, 1e-12),
        ):
            raise ValueError(
                "candidate shared-cash interval execution accounting is inconsistent"
            )


def _validate_v9_decision_interval_links(
    decisions: Sequence[object],
    *,
    intervals: Sequence[object],
    start: int,
) -> None:
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise ValueError(
                "candidate shared-cash decision interval link is malformed"
            )
        decision_index = cast(int, decision["index"])
        interval_offset = decision_index - start
        if not 0 <= interval_offset < len(intervals):
            raise ValueError("candidate shared-cash decision interval link is invalid")
        interval = intervals[interval_offset]
        if not isinstance(interval, Mapping):
            raise ValueError(
                "candidate shared-cash decision interval link is malformed"
            )
        for decision_field, interval_field in (
            ("position_quantity_before", "exact_quantities_before"),
            ("position_quantity_after", "exact_quantities_after"),
        ):
            decision_quantities = decision[decision_field]
            exact_quantities = interval[interval_field]
            if (
                not isinstance(decision_quantities, (list, tuple))
                or not isinstance(exact_quantities, (list, tuple))
                or len(decision_quantities) != len(exact_quantities)
            ):
                raise ValueError(
                    "candidate shared-cash decision interval link is malformed"
                )
            if any(
                exact_quantity is None
                or not _numbers_are_close(decision_quantity, exact_quantity)
                for decision_quantity, exact_value in zip(
                    decision_quantities,
                    exact_quantities,
                    strict=True,
                )
                for exact_quantity in (_quantity_as_float(exact_value),)
            ):
                raise ValueError(
                    "candidate shared-cash decision interval link is inconsistent"
                )


def _validate_v9_decision_coverage(
    decisions: Sequence[object],
    *,
    start: int,
    stop: int,
    evaluation: object,
) -> None:
    if not isinstance(evaluation, Mapping):
        raise ValueError("candidate shared-cash decision coverage is malformed")
    settle_terminal_position = evaluation.get("ppo_settle_terminal_position")
    execution_overlay = evaluation.get("execution_overlay")
    if not isinstance(settle_terminal_position, bool) or not isinstance(
        execution_overlay, str
    ):
        raise ValueError("candidate shared-cash decision coverage is malformed")
    try:
        expected_stop = _agent_stop_index(
            start_index=start,
            stop_index=stop,
            execution_cost=_execution_cost_for_overlay(execution_overlay),
            settle_terminal_position=settle_terminal_position,
        )
    except ValueError as error:
        raise ValueError(
            "candidate shared-cash decision coverage boundary is malformed"
        ) from error
    persisted_stop = evaluation.get("ppo_policy_decision_stop_index")
    if (
        isinstance(persisted_stop, bool)
        or not isinstance(persisted_stop, int)
        or persisted_stop != expected_stop
    ):
        raise ValueError(
            "candidate shared-cash decision coverage boundary is inconsistent"
        )
    if any(
        not isinstance(decision, Mapping) or decision.get("index") != start + offset
        for offset, decision in enumerate(decisions)
    ):
        raise ValueError("candidate shared-cash decision coverage is incomplete")
    if len(decisions) != expected_stop - start:
        raise ValueError("candidate shared-cash decision coverage is incomplete")


def _validate_v10_accounting_transitions(
    *,
    payload: Mapping[str, object],
    intervals: Sequence[object],
    symbols: Sequence[object],
    summary: Mapping[str, object],
    require_ohlc_stress: bool = False,
) -> None:
    require_ohlc_stress = require_ohlc_stress or (
        payload.get("schema_version") == "shared_cash_replay_ledger_v4"
    )
    initial_marks = _finite_positive_vector(
        payload.get("initial_mark_prices"),
        size=len(symbols),
        field="initial mark prices",
    )
    multipliers = _finite_positive_vector(
        payload.get("contract_multipliers"),
        size=len(symbols),
        field="contract multipliers",
    )
    evaluation = summary.get("evaluation")
    initial_capital = (
        evaluation.get("initial_capital") if isinstance(evaluation, Mapping) else None
    )
    if not _is_finite_number(initial_capital):
        raise ValueError("candidate v10 accounting initial capital is malformed")
    initial_capital_value = float(cast(int | float, initial_capital))
    if initial_capital_value <= 0.0:
        raise ValueError("candidate v10 accounting initial capital is malformed")
    peak_value = initial_capital_value
    maximum_drawdown = 0.0

    previous_interval_end_state: (
        tuple[float, tuple[Fraction, ...], tuple[float, ...], tuple[float, ...]] | None
    ) = None
    for interval in intervals:
        if not isinstance(interval, Mapping):
            raise ValueError("candidate v10 accounting interval is malformed")
        if not _numbers_are_close(
            interval.get("max_drawdown_before"), maximum_drawdown
        ):
            raise ValueError("candidate v10 accounting drawdown link is inconsistent")
        raw_transitions = interval.get("accounting_transitions")
        raw_events = interval.get("order_events")
        raw_funding_events = interval.get("funding_events")
        if (
            not isinstance(raw_transitions, (list, tuple))
            or not raw_transitions
            or not isinstance(raw_events, (list, tuple))
            or not isinstance(raw_funding_events, (list, tuple))
        ):
            raise ValueError("candidate v10 accounting transition stream is incomplete")

        parsed_events: dict[int, OrderEvent] = {}
        for raw_event in raw_events:
            if not isinstance(raw_event, Mapping):
                raise ValueError("candidate v10 accounting order event is malformed")
            try:
                event = OrderEvent.from_mapping(cast(Mapping[str, object], raw_event))
            except (OverflowError, TypeError, ValueError) as error:
                raise ValueError(
                    "candidate v10 accounting order event is malformed"
                ) from error
            parsed_events[event.sequence] = event

        funding_events: list[FundingBoundaryEvidence] = []
        for raw_funding in raw_funding_events:
            if not isinstance(raw_funding, Mapping):
                raise ValueError("candidate v10 accounting funding event is malformed")
            try:
                funding_events.append(
                    FundingBoundaryEvidence.from_mapping(
                        cast(Mapping[str, object], raw_funding)
                    )
                )
            except (OverflowError, TypeError, ValueError) as error:
                raise ValueError(
                    "candidate v10 accounting funding event is malformed"
                ) from error

        start_index = cast(int, interval["start_index"])
        processing_index = cast(int, interval["next_index"])
        previous_state: (
            tuple[float, tuple[Fraction, ...], tuple[float, ...], tuple[float, ...]]
            | None
        ) = None
        transition_types: list[str] = []
        transition_costs: list[float] = []
        transition_funding: list[float] = []
        transition_borrow: list[float] = []
        transition_dividend: list[float] = []
        transition_interest: list[float] = []
        transition_turnover: list[float] = []
        fill_sequences: set[int] = set()
        gap_carry_delta = 0.0
        open_mark_seen = False
        dividend_seen = False
        processing_interest_seen = False
        processing_borrow_seen = False
        funding_mark_seen = False
        ohlc_stress_phases: list[str] = []
        ohlc_stress_fill_sequences: set[int] = set()
        pending_fill_stress_sequence: int | None = None

        for sequence, raw_transition in enumerate(raw_transitions):
            transition = _accounting_transition_mapping(raw_transition)
            if (
                transition["sequence"] != sequence
                or transition["processing_index"] != processing_index
            ):
                raise ValueError(
                    "candidate v10 accounting transition order is inconsistent"
                )
            transition_type = transition["transition_type"]
            if not isinstance(transition_type, str):
                raise ValueError(
                    "candidate v10 accounting transition type is malformed"
                )
            if (
                require_ohlc_stress
                and pending_fill_stress_sequence is not None
                and transition_type != "ohlc_drawdown_stress"
            ):
                raise ValueError("candidate v11 post-fill OHLC stress is missing")
            transition_types.append(transition_type)
            state_before = _accounting_state_mapping(
                transition["state_before"], size=len(symbols)
            )
            state_after = _accounting_state_mapping(
                transition["state_after"], size=len(symbols)
            )
            for state in (state_before, state_after):
                value = max(_accounting_nav(state), 0.0)
                peak_value = max(peak_value, value)
                maximum_drawdown = max(
                    maximum_drawdown,
                    1.0 - value / max(peak_value, float(np.finfo(np.float64).tiny)),
                )
            if previous_state is not None and not _accounting_states_match(
                previous_state, state_before
            ):
                raise ValueError("candidate v10 accounting transition chain is broken")
            previous_state = state_after
            if any(
                not _numbers_are_close(actual, expected)
                for actual, expected in zip(state_before[3], multipliers, strict=True)
            ) or any(
                not _numbers_are_close(actual, expected)
                for actual, expected in zip(state_after[3], multipliers, strict=True)
            ):
                raise ValueError(
                    "candidate v10 accounting transition multiplier link is inconsistent"
                )

            evidence = transition["evidence"]
            if not isinstance(evidence, Mapping):
                raise ValueError(
                    "candidate v10 accounting transition evidence is malformed"
                )
            event_sequence = transition["order_event_sequence"]
            if transition_type != "fill" and event_sequence is not None:
                raise ValueError("candidate v10 accounting event sequence is invalid")
            cash_before, quantities_before, marks_before, state_multipliers = (
                state_before
            )
            cash_after, quantities_after, marks_after, _ = state_after
            before_nav = _accounting_nav(state_before)
            expected_quantities = quantities_before
            expected_cash = cash_before
            expected_marks = marks_before

            if transition_type == "split":
                if open_mark_seen or dividend_seen:
                    raise ValueError(
                        "candidate v10 split transition order is inconsistent"
                    )
                _require_evidence_fields(evidence, {"split_factors"})
                factors = _finite_positive_vector(
                    evidence["split_factors"],
                    size=len(symbols),
                    field="split factors",
                )
                expected_quantities = tuple(
                    quantity * Fraction(str(factor))
                    for quantity, factor in zip(quantities_before, factors, strict=True)
                )
                expected_marks = tuple(
                    mark / factor
                    for mark, factor in zip(marks_before, factors, strict=True)
                )
            elif transition_type == "delisting_settlement":
                if open_mark_seen or dividend_seen:
                    raise ValueError(
                        "candidate v10 settlement transition order is inconsistent"
                    )
                _require_evidence_fields(
                    evidence, {"inactive_mask", "open_prices", "delisting_recovery"}
                )
                inactive = evidence["inactive_mask"]
                if (
                    not isinstance(inactive, (list, tuple))
                    or len(inactive) != len(symbols)
                    or any(not isinstance(value, bool) for value in inactive)
                ):
                    raise ValueError("candidate v10 delisting mask is malformed")
                prices = _finite_positive_vector(
                    evidence["open_prices"],
                    size=len(symbols),
                    field="settlement prices",
                )
                recovery = _finite_vector(
                    evidence["delisting_recovery"],
                    size=len(symbols),
                    field="delisting recovery",
                )
                if any(value < 0.0 or value > 1.0 for value in recovery):
                    raise ValueError("candidate v10 delisting recovery is invalid")
                proceeds = math.fsum(
                    float(quantity) * price * multiplier * recovered
                    for quantity, price, multiplier, recovered, settle in zip(
                        quantities_before,
                        prices,
                        state_multipliers,
                        recovery,
                        inactive,
                        strict=True,
                    )
                    if settle
                )
                expected_cash += proceeds
                expected_quantities = tuple(
                    Fraction(0) if settle else quantity
                    for quantity, settle in zip(
                        quantities_before, inactive, strict=True
                    )
                )
                expected_marks = prices
            elif transition_type == "mark_revaluation":
                _require_evidence_fields(evidence, {"mark_phase", "mark_prices"})
                if evidence["mark_phase"] != "open" or open_mark_seen or dividend_seen:
                    raise ValueError("candidate v10 mark phase is malformed")
                open_mark_seen = True
                expected_marks = _finite_positive_vector(
                    evidence["mark_prices"], size=len(symbols), field="mark prices"
                )
            elif transition_type == "ohlc_drawdown_stress":
                if not require_ohlc_stress:
                    raise ValueError(
                        "candidate v10 accounting transition type is unsupported"
                    )
                phase = evidence["phase"]
                if phase == "pre_fill":
                    _require_evidence_fields(
                        evidence,
                        {
                            "adverse_prices",
                            "favorable_prices",
                            "high_prices",
                            "low_prices",
                            "phase",
                        },
                    )
                    if (
                        not open_mark_seen
                        or dividend_seen
                        or fill_sequences
                        or ohlc_stress_phases
                    ):
                        raise ValueError(
                            "candidate v11 pre-fill OHLC stress order is inconsistent"
                        )
                elif phase == "after_fill":
                    _require_evidence_fields(
                        evidence,
                        {
                            "adverse_prices",
                            "fill_event_sequence",
                            "favorable_prices",
                            "high_prices",
                            "low_prices",
                            "phase",
                        },
                    )
                    fill_event_sequence = evidence["fill_event_sequence"]
                    fill_event = (
                        parsed_events.get(fill_event_sequence)
                        if isinstance(fill_event_sequence, int)
                        and not isinstance(fill_event_sequence, bool)
                        else None
                    )
                    if (
                        not open_mark_seen
                        or dividend_seen
                        or funding_mark_seen
                        or not ohlc_stress_phases
                        or ohlc_stress_phases[0] != "pre_fill"
                        or any(
                            previous_phase != "after_fill"
                            for previous_phase in ohlc_stress_phases[1:]
                        )
                        or isinstance(fill_event_sequence, bool)
                        or not isinstance(fill_event_sequence, int)
                        or pending_fill_stress_sequence != fill_event_sequence
                        or fill_event_sequence not in fill_sequences
                        or fill_event_sequence in ohlc_stress_fill_sequences
                        or fill_event is None
                        or fill_event.event_type not in {"filled", "partial_fill"}
                        or fill_event.processing_index != processing_index
                    ):
                        raise ValueError(
                            "candidate v11 OHLC stress fill event link is inconsistent"
                        )
                    ohlc_stress_fill_sequences.add(fill_event_sequence)
                    pending_fill_stress_sequence = None
                elif phase == "post_fill":
                    _require_evidence_fields(
                        evidence,
                        {
                            "adverse_prices",
                            "favorable_prices",
                            "high_prices",
                            "low_prices",
                            "phase",
                        },
                    )
                    if (
                        not funding_mark_seen
                        or not ohlc_stress_phases
                        or ohlc_stress_phases[0] != "pre_fill"
                        or any(
                            previous_phase != "after_fill"
                            for previous_phase in ohlc_stress_phases[1:]
                        )
                        or pending_fill_stress_sequence is not None
                    ):
                        raise ValueError(
                            "candidate v11 post-fill OHLC stress order is inconsistent"
                        )
                else:
                    raise ValueError("candidate v11 OHLC stress phase is malformed")
                if not _accounting_states_match(state_before, state_after):
                    raise ValueError(
                        "candidate v11 OHLC stress changed the account state"
                    )
                highs = _finite_positive_vector(
                    evidence["high_prices"],
                    size=len(symbols),
                    field="OHLC stress highs",
                )
                lows = _finite_positive_vector(
                    evidence["low_prices"],
                    size=len(symbols),
                    field="OHLC stress lows",
                )
                if any(low > high for low, high in zip(lows, highs, strict=True)):
                    raise ValueError("candidate v11 OHLC stress range is invalid")
                adverse_prices = _finite_positive_vector(
                    evidence["adverse_prices"],
                    size=len(symbols),
                    field="OHLC adverse prices",
                )
                favorable_prices = _finite_positive_vector(
                    evidence["favorable_prices"],
                    size=len(symbols),
                    field="OHLC favorable prices",
                )
                expected_adverse_prices = tuple(
                    low if quantity > 0 else high if quantity < 0 else mark
                    for quantity, low, high, mark in zip(
                        quantities_before,
                        lows,
                        highs,
                        marks_before,
                        strict=True,
                    )
                )
                if any(
                    not _numbers_are_close(actual, expected)
                    for actual, expected in zip(
                        adverse_prices, expected_adverse_prices, strict=True
                    )
                ):
                    raise ValueError(
                        "candidate v11 OHLC adverse prices are inconsistent"
                    )
                expected_favorable_prices = tuple(
                    high if quantity > 0 else low if quantity < 0 else mark
                    for quantity, low, high, mark in zip(
                        quantities_before,
                        lows,
                        highs,
                        marks_before,
                        strict=True,
                    )
                )
                if any(
                    not _numbers_are_close(actual, expected)
                    for actual, expected in zip(
                        favorable_prices, expected_favorable_prices, strict=True
                    )
                ):
                    raise ValueError(
                        "candidate v11 OHLC favorable prices are inconsistent"
                    )
                stressed_value = _accounting_nav(
                    (
                        cash_before,
                        quantities_before,
                        adverse_prices,
                        state_multipliers,
                    )
                )
                favorable_value = _accounting_nav(
                    (
                        cash_before,
                        quantities_before,
                        favorable_prices,
                        state_multipliers,
                    )
                )
                peak_value = max(peak_value, max(favorable_value, 0.0))
                maximum_drawdown = max(
                    maximum_drawdown,
                    1.0
                    - max(stressed_value, 0.0)
                    / max(peak_value, float(np.finfo(np.float64).tiny)),
                )
                ohlc_stress_phases.append(cast(str, phase))
            elif transition_type == "fill":
                if not open_mark_seen or dividend_seen:
                    raise ValueError(
                        "candidate v10 fill transition order is inconsistent"
                    )
                legacy_fill_fields = {
                    "cost_amount",
                    "execution_price",
                    "filled_notional",
                    "filled_quantity",
                    "filled_quantity_exact",
                    "order_id",
                    "symbol_index",
                    "turnover",
                }
                exact_fill_fields = legacy_fill_fields | {
                    "book_applied_quantity_exact",
                    "filled_lot_count",
                    "filled_lot_size",
                }
                evidence_fields = frozenset(evidence)
                if evidence_fields not in {
                    frozenset(legacy_fill_fields),
                    frozenset(exact_fill_fields),
                }:
                    raise ValueError(
                        "candidate v10 accounting transition evidence is malformed"
                    )
                has_exact_fill_evidence = set(evidence) == exact_fill_fields
                if (
                    isinstance(event_sequence, bool)
                    or not isinstance(event_sequence, int)
                    or event_sequence in fill_sequences
                    or pending_fill_stress_sequence is not None
                ):
                    raise ValueError(
                        "candidate v10 accounting fill event link is malformed"
                    )
                fill_event = parsed_events.get(event_sequence)
                symbol_index = evidence["symbol_index"]
                quantity = evidence["filled_quantity"]
                raw_exact_quantity = evidence["filled_quantity_exact"]
                price = evidence["execution_price"]
                notional = evidence["filled_notional"]
                cost_amount = evidence["cost_amount"]
                turnover = evidence["turnover"]
                if not isinstance(raw_exact_quantity, str):
                    raise ValueError("candidate v10 exact fill quantity is malformed")
                try:
                    exact_fill_quantity = Fraction(raw_exact_quantity)
                except (TypeError, ValueError, ZeroDivisionError) as error:
                    raise ValueError(
                        "candidate v10 exact fill quantity is malformed"
                    ) from error
                exact_fill_projection = _quantity_as_float(raw_exact_quantity)
                applied_fill_quantity: Fraction | None = None
                if has_exact_fill_evidence:
                    raw_applied_quantity = evidence["book_applied_quantity_exact"]
                    raw_lot_count = evidence["filled_lot_count"]
                    lot_size = _finite_number(
                        evidence["filled_lot_size"], "filled lot size"
                    )
                    if not isinstance(raw_applied_quantity, str):
                        raise ValueError(
                            "candidate v10 applied fill quantity is malformed"
                        )
                    try:
                        applied_fill_quantity = Fraction(raw_applied_quantity)
                    except (TypeError, ValueError, ZeroDivisionError) as error:
                        raise ValueError(
                            "candidate v10 applied fill quantity is malformed"
                        ) from error
                    if (
                        str(applied_fill_quantity) != raw_applied_quantity
                        or isinstance(raw_lot_count, bool)
                        or (
                            raw_lot_count is not None
                            and (
                                not isinstance(raw_lot_count, int) or raw_lot_count == 0
                            )
                        )
                        or lot_size < 0.0
                    ):
                        raise ValueError(
                            "candidate v10 applied fill quantity is malformed"
                        )
                    if raw_lot_count is None:
                        if lot_size != 0.0:
                            raise ValueError(
                                "candidate v10 fill lot evidence is inconsistent"
                            )
                        expected_exact_fill = Fraction(str(float(quantity)))
                    else:
                        if lot_size <= 0.0:
                            raise ValueError(
                                "candidate v10 fill lot evidence is inconsistent"
                            )
                        expected_exact_fill = Fraction(
                            cast(int, raw_lot_count)
                        ) * Fraction(str(lot_size))
                    if (
                        exact_fill_quantity != expected_exact_fill
                        or _project_exact_quantity(exact_fill_quantity) != quantity
                    ):
                        raise ValueError(
                            "candidate v10 exact fill quantity is inconsistent"
                        )
                if (
                    fill_event is None
                    or fill_event.event_type not in {"filled", "partial_fill"}
                    or fill_event.processing_index != processing_index
                    or fill_event.order_id != evidence["order_id"]
                    or isinstance(symbol_index, bool)
                    or not isinstance(symbol_index, int)
                    or symbol_index != fill_event.symbol_index
                    or not 0 <= symbol_index < len(symbols)
                    or not _is_finite_number(quantity)
                    or not _numbers_are_close(quantity, fill_event.filled_quantity)
                    or str(exact_fill_quantity) != raw_exact_quantity
                    or exact_fill_projection is None
                    or not _numbers_are_close(exact_fill_projection, quantity)
                    or not _is_finite_number(price)
                    or fill_event.execution_price is None
                    or not _numbers_are_close(price, fill_event.execution_price)
                    or not _is_finite_number(notional)
                    or not _numbers_are_close(notional, fill_event.filled_notional)
                    or not _numbers_are_close(
                        notional,
                        abs(float(cast(int | float, quantity)))
                        * float(cast(int | float, price))
                        * state_multipliers[cast(int, symbol_index)],
                    )
                    or not _is_finite_number(cost_amount)
                    or float(cast(int | float, cost_amount)) < 0.0
                    or not _is_finite_number(turnover)
                    or float(cast(int | float, turnover)) < 0.0
                ):
                    raise ValueError(
                        "candidate v10 accounting fill event link is inconsistent"
                    )
                fill_sequences.add(event_sequence)
                if require_ohlc_stress:
                    pending_fill_stress_sequence = event_sequence
                old_quantity = quantities_before[symbol_index]
                if has_exact_fill_evidence:
                    if applied_fill_quantity is None:
                        raise ValueError(
                            "candidate v10 applied fill quantity is malformed"
                        )
                    projected_close = _project_exact_quantity(old_quantity)
                    expected_applied_quantity = (
                        -old_quantity
                        if raw_lot_count is None
                        and old_quantity != 0
                        and quantity == -projected_close
                        else exact_fill_quantity
                    )
                    if applied_fill_quantity != expected_applied_quantity:
                        raise ValueError(
                            "candidate v10 applied fill quantity is inconsistent"
                        )
                    fill_delta = applied_fill_quantity
                else:
                    # Older v10 artifacts did not persist lot allocation or the
                    # book-applied delta, so retain their original strict contract.
                    actual_delta = quantities_after[symbol_index] - old_quantity
                    if actual_delta != exact_fill_quantity:
                        raise ValueError(
                            "candidate v10 applied fill quantity is inconsistent"
                        )
                    fill_delta = exact_fill_quantity
                expected_quantities = tuple(
                    old + fill_delta if index == symbol_index else old
                    for index, old in enumerate(quantities_before)
                )
                expected_cash -= float(cast(int | float, quantity)) * float(
                    cast(int | float, price)
                ) * state_multipliers[symbol_index] + float(
                    cast(int | float, cost_amount)
                )
                expected_marks = tuple(
                    float(cast(int | float, price)) if index == symbol_index else mark
                    for index, mark in enumerate(marks_before)
                )
                transition_costs.append(float(cast(int | float, cost_amount)))
                transition_turnover.append(float(cast(int | float, turnover)))
            elif transition_type == "dividend":
                if not open_mark_seen or dividend_seen:
                    raise ValueError(
                        "candidate v10 dividend transition order is inconsistent"
                    )
                dividend_seen = True
                _require_evidence_fields(
                    evidence, {"dividend_per_unit", "dividend_amount"}
                )
                dividends = _finite_vector(
                    evidence["dividend_per_unit"],
                    size=len(symbols),
                    field="dividend amounts",
                )
                amount = math.fsum(
                    float(quantity) * multiplier * dividend
                    for quantity, multiplier, dividend in zip(
                        quantities_before, state_multipliers, dividends, strict=True
                    )
                )
                if not _numbers_are_close(evidence["dividend_amount"], amount):
                    raise ValueError(
                        "candidate v10 dividend accounting is inconsistent"
                    )
                expected_cash += amount
                transition_dividend.append(amount)
            elif transition_type == "cash_interest":
                _require_evidence_fields(
                    evidence,
                    {
                        "annual_rate",
                        "basis_adjustment",
                        "carry_phase",
                        "year_fraction",
                    },
                )
                carry_phase = evidence["carry_phase"]
                annual_rate = _finite_number(evidence["annual_rate"], "cash rate")
                basis_adjustment = _finite_number(
                    evidence["basis_adjustment"], "cash interest basis adjustment"
                )
                year_fraction = _finite_number(
                    evidence["year_fraction"], "cash interest year fraction"
                )
                if (
                    year_fraction < 0.0
                    or not isinstance(carry_phase, str)
                    or carry_phase not in {"gap", "processing"}
                    or (carry_phase == "gap" and (open_mark_seen or dividend_seen))
                    or (
                        carry_phase == "processing"
                        and (
                            not dividend_seen
                            or processing_interest_seen
                            or not _numbers_are_close(basis_adjustment, gap_carry_delta)
                        )
                    )
                    or (
                        carry_phase == "gap"
                        and not _numbers_are_close(basis_adjustment, 0.0)
                    )
                ):
                    raise ValueError("candidate v10 cash interest inputs are invalid")
                if carry_phase == "processing":
                    processing_interest_seen = True
                amount = (cash_before - basis_adjustment) * annual_rate * year_fraction
                expected_cash += amount
                transition_interest.append(amount)
                if carry_phase == "gap":
                    gap_carry_delta += amount
            elif transition_type == "borrow_charge":
                _require_evidence_fields(
                    evidence,
                    {
                        "borrow_amount",
                        "borrow_rate",
                        "borrow_rate_multiplier",
                        "carry_phase",
                        "year_fraction",
                    },
                )
                rates = _finite_vector(
                    evidence["borrow_rate"], size=len(symbols), field="borrow rates"
                )
                rate_multiplier = _finite_number(
                    evidence["borrow_rate_multiplier"], "borrow rate multiplier"
                )
                year_fraction = _finite_number(
                    evidence["year_fraction"], "borrow year fraction"
                )
                carry_phase = evidence["carry_phase"]
                if (
                    any(rate < 0.0 for rate in rates)
                    or rate_multiplier < 0.0
                    or year_fraction < 0.0
                    or not isinstance(carry_phase, str)
                    or carry_phase not in {"gap", "processing"}
                    or (carry_phase == "gap" and (open_mark_seen or dividend_seen))
                    or (
                        carry_phase == "processing"
                        and (not processing_interest_seen or processing_borrow_seen)
                    )
                ):
                    raise ValueError("candidate v10 borrow inputs are invalid")
                amount = (
                    math.fsum(
                        max(-float(quantity) * mark * multiplier, 0.0) * rate
                        for quantity, mark, multiplier, rate in zip(
                            quantities_before,
                            marks_before,
                            state_multipliers,
                            rates,
                            strict=True,
                        )
                    )
                    * year_fraction
                    * rate_multiplier
                )
                if not _numbers_are_close(evidence["borrow_amount"], amount):
                    raise ValueError("candidate v10 borrow accounting is inconsistent")
                expected_cash -= amount
                transition_borrow.append(amount)
                if carry_phase == "gap":
                    gap_carry_delta -= amount
                else:
                    processing_borrow_seen = True
            elif transition_type == "funding_mark":
                if (
                    not processing_interest_seen
                    or not processing_borrow_seen
                    or funding_mark_seen
                ):
                    raise ValueError(
                        "candidate v10 funding transition order is inconsistent"
                    )
                funding_mark_seen = True
                _require_evidence_fields(evidence, {"funding_amount", "mark_prices"})
                amount = _finite_number(evidence["funding_amount"], "funding amount")
                prices = _finite_positive_vector(
                    evidence["mark_prices"], size=len(symbols), field="mark prices"
                )
                if len(funding_events) > 1:
                    raise ValueError(
                        "candidate v10 funding boundary count is malformed"
                    )
                evidenced_funding = math.fsum(
                    event.funding_amount for event in funding_events
                )
                if not _numbers_are_close(amount, evidenced_funding):
                    raise ValueError("candidate v10 funding accounting is inconsistent")
                for funding_event in funding_events:
                    if (
                        funding_event.processing_index != processing_index
                        or any(
                            not _numbers_are_close(actual, expected)
                            for actual, expected in zip(
                                funding_event.signed_quantities,
                                (float(value) for value in quantities_before),
                                strict=True,
                            )
                        )
                        or any(
                            not _numbers_are_close(actual, expected)
                            for actual, expected in zip(
                                funding_event.mark_prices, prices, strict=True
                            )
                        )
                        or any(
                            not _numbers_are_close(actual, expected)
                            for actual, expected in zip(
                                funding_event.contract_multipliers,
                                state_multipliers,
                                strict=True,
                            )
                        )
                    ):
                        raise ValueError(
                            "candidate v10 funding transition link is inconsistent"
                        )
                expected_cash += amount
                expected_marks = prices
                transition_funding.append(amount)
            elif transition_type == "termination_flatten":
                _require_evidence_fields(
                    evidence, {"liquidation_prices", "nav_before", "reason"}
                )
                prices = _finite_positive_vector(
                    evidence["liquidation_prices"],
                    size=len(symbols),
                    field="liquidation prices",
                )
                reason = evidence["reason"]
                if (
                    not isinstance(reason, str)
                    or not reason
                    or reason != interval.get("termination_reason")
                ):
                    raise ValueError("candidate v10 termination reason is malformed")
                if not _numbers_are_close(evidence["nav_before"], before_nav):
                    raise ValueError("candidate v10 termination NAV is inconsistent")
                expected_cash = max(before_nav, 0.0)
                expected_quantities = tuple(Fraction(0) for _ in symbols)
                expected_marks = prices
            else:
                raise ValueError(
                    "candidate v10 accounting transition type is unsupported"
                )

            if (
                any(
                    expected != actual
                    for expected, actual in zip(
                        expected_quantities, quantities_after, strict=True
                    )
                )
                or not _numbers_are_close(cash_after, expected_cash)
                or any(
                    not _numbers_are_close(actual, expected)
                    for actual, expected in zip(
                        marks_after, expected_marks, strict=True
                    )
                )
            ):
                raise ValueError(
                    "candidate v10 accounting transition balance is inconsistent"
                )

        if previous_state is None:
            raise ValueError("candidate v10 accounting transition stream is empty")
        first_transition = _accounting_state_mapping(
            _accounting_transition_mapping(raw_transitions[0])["state_before"],
            size=len(symbols),
        )
        interval_cash_before, interval_quantities_before = (
            _accounting_interval_position_state(
                interval, before=True, size=len(symbols)
            )
        )
        interval_cash_after, interval_quantities_after = (
            _accounting_interval_position_state(
                interval, before=False, size=len(symbols)
            )
        )
        if (
            not _numbers_are_close(first_transition[0], interval_cash_before)
            or first_transition[1] != interval_quantities_before
            or not _numbers_are_close(previous_state[0], interval_cash_after)
            or previous_state[1] != interval_quantities_after
        ):
            raise ValueError(
                "candidate v10 accounting transition interval link is inconsistent"
            )
        if previous_interval_end_state is not None and not _accounting_states_match(
            previous_interval_end_state, first_transition
        ):
            raise ValueError(
                "candidate v10 accounting interval state continuity is broken"
            )
        if start_index == cast(int, payload["start_index"]):
            if (
                not _numbers_are_close(first_transition[0], initial_capital)
                or any(value != Fraction(0) for value in first_transition[1])
                or any(
                    not _numbers_are_close(actual, expected)
                    for actual, expected in zip(
                        first_transition[2], initial_marks, strict=True
                    )
                )
            ):
                raise ValueError(
                    "candidate v10 initial accounting state is inconsistent"
                )
        if not _numbers_are_close(
            _accounting_nav(first_transition), interval["portfolio_value_before"]
        ) or not _numbers_are_close(
            _accounting_nav(previous_state), interval["portfolio_value_after"]
        ):
            raise ValueError("candidate v10 portfolio value accounting is inconsistent")
        if not _numbers_are_close(interval.get("max_drawdown_after"), maximum_drawdown):
            raise ValueError("candidate v10 accounting drawdown link is inconsistent")
        previous_interval_end_state = previous_state

        if (
            transition_types.count("mark_revaluation") != 1
            or transition_types.count("funding_mark") != 1
            or transition_types.count("dividend") != 1
            or not processing_interest_seen
            or not processing_borrow_seen
            or (
                require_ohlc_stress
                and (
                    ohlc_stress_phases
                    != [
                        "pre_fill",
                        *("after_fill" for _ in fill_sequences),
                        "post_fill",
                    ]
                    or ohlc_stress_fill_sequences != fill_sequences
                    or pending_fill_stress_sequence is not None
                )
            )
        ):
            raise ValueError(
                "candidate v10 accounting transition coverage is incomplete"
            )
        if interval.get("termination_reason") is not None:
            if "termination_flatten" not in transition_types:
                raise ValueError(
                    "candidate v10 termination flatten evidence is missing"
                )
            if any(quantity != Fraction(0) for quantity in previous_state[1]):
                raise ValueError(
                    "candidate v10 termination did not leave a flat position"
                )
        aggregate_links = (
            (transition_costs, interval["interval_cost"], "cost"),
            (transition_funding, interval["interval_funding"], "funding"),
            (transition_borrow, interval["interval_borrow_cost"], "borrow"),
            (transition_dividend, interval["interval_dividend"], "dividend"),
            (transition_interest, interval["interval_cash_interest"], "cash interest"),
            (
                transition_turnover,
                cast(float, interval["turnover_total_after"])
                - cast(float, interval["turnover_total_before"]),
                "turnover",
            ),
        )
        for amounts, persisted_amount, label in aggregate_links:
            try:
                total_amount = math.fsum(amounts)
            except OverflowError as error:
                raise ValueError(
                    f"candidate v10 {label} transition total is malformed"
                ) from error
            if not _numbers_are_close(total_amount, persisted_amount):
                raise ValueError(
                    f"candidate v10 {label} transition total is inconsistent"
                )
        if fill_sequences != {
            event.sequence
            for event in parsed_events.values()
            if event.event_type in {"filled", "partial_fill"}
        }:
            raise ValueError("candidate v10 fill transition coverage is incomplete")

    portfolio = summary.get("shared_cash_ppo")
    metrics = portfolio.get("metrics") if isinstance(portfolio, Mapping) else None
    if (
        not _numbers_are_close(payload.get("final_max_drawdown"), maximum_drawdown)
        or not isinstance(metrics, Mapping)
        or not _numbers_are_close(metrics.get("max_drawdown"), maximum_drawdown)
    ):
        raise ValueError("candidate v10 accounting drawdown link is inconsistent")


def _accounting_transition_mapping(value: object) -> Mapping[str, object]:
    fields = {
        "evidence",
        "order_event_sequence",
        "processing_index",
        "sequence",
        "state_after",
        "state_before",
        "transition_type",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("candidate v10 accounting transition is malformed")
    for field in ("sequence", "processing_index"):
        index = value[field]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("candidate v10 accounting transition index is malformed")
    if value["order_event_sequence"] is not None and (
        isinstance(value["order_event_sequence"], bool)
        or not isinstance(value["order_event_sequence"], int)
        or value["order_event_sequence"] < 0
    ):
        raise ValueError("candidate v10 accounting event sequence is malformed")
    return value


def _accounting_state_mapping(
    value: object,
    *,
    size: int,
) -> tuple[float, tuple[Fraction, ...], tuple[float, ...], tuple[float, ...]]:
    fields = {"cash", "contract_multipliers", "exact_quantities", "mark_prices"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("candidate v10 accounting state is malformed")
    cash = _finite_number(value["cash"], "account cash")
    raw_quantities = value["exact_quantities"]
    if (
        not isinstance(raw_quantities, (list, tuple))
        or len(raw_quantities) != size
        or any(not isinstance(quantity, str) for quantity in raw_quantities)
    ):
        raise ValueError("candidate v10 accounting quantities are malformed")
    quantities: list[Fraction] = []
    try:
        for raw_quantity in raw_quantities:
            quantity = Fraction(raw_quantity)
            if (
                str(quantity) != raw_quantity
                or _quantity_as_float(raw_quantity) is None
            ):
                raise ValueError("non-canonical exact quantity")
            quantities.append(quantity)
    except (OverflowError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("candidate v10 accounting quantities are malformed") from error
    marks = _finite_positive_vector(value["mark_prices"], size=size, field="marks")
    multipliers = _finite_positive_vector(
        value["contract_multipliers"], size=size, field="multipliers"
    )
    return cash, tuple(quantities), marks, multipliers


def _accounting_interval_position_state(
    interval: Mapping[str, object],
    *,
    before: bool,
    size: int,
) -> tuple[float, tuple[Fraction, ...]]:
    suffix = "before" if before else "after"
    raw_quantities = interval[f"exact_quantities_{suffix}"]
    if (
        not isinstance(raw_quantities, (list, tuple))
        or len(raw_quantities) != size
        or any(not isinstance(value, str) for value in raw_quantities)
    ):
        raise ValueError("candidate v10 interval quantities are malformed")
    try:
        quantities = tuple(Fraction(value) for value in raw_quantities)
        if any(
            str(quantity) != raw or _quantity_as_float(raw) is None
            for quantity, raw in zip(quantities, raw_quantities, strict=True)
        ):
            raise ValueError("non-canonical exact quantity")
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("candidate v10 interval quantities are malformed") from error
    cash = _finite_number(interval[f"cash_{suffix}"], "interval cash")
    return cash, quantities


def _accounting_nav(
    state: tuple[float, tuple[Fraction, ...], tuple[float, ...], tuple[float, ...]],
) -> float:
    cash, quantities, marks, multipliers = state
    try:
        value = cash + math.fsum(
            float(quantity) * mark * multiplier
            for quantity, mark, multiplier in zip(
                quantities, marks, multipliers, strict=True
            )
        )
    except OverflowError as error:
        raise ValueError("candidate v10 portfolio value is malformed") from error
    if not math.isfinite(value):
        raise ValueError("candidate v10 portfolio value is malformed")
    return value


def _accounting_states_match(
    left: tuple[float, tuple[Fraction, ...], tuple[float, ...], tuple[float, ...]],
    right: tuple[float, tuple[Fraction, ...], tuple[float, ...], tuple[float, ...]],
) -> bool:
    return (
        _numbers_are_close(left[0], right[0])
        and left[1] == right[1]
        and len(left[2]) == len(right[2])
        and len(left[3]) == len(right[3])
        and all(
            _numbers_are_close(actual, expected)
            for actual, expected in zip(left[2], right[2], strict=True)
        )
        and all(
            _numbers_are_close(actual, expected)
            for actual, expected in zip(left[3], right[3], strict=True)
        )
    )


def _require_evidence_fields(
    evidence: Mapping[str, object],
    expected_fields: set[str],
) -> None:
    if set(evidence) != expected_fields:
        raise ValueError("candidate v10 accounting transition evidence is malformed")


def _finite_number(value: object, field: str) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"candidate v10 {field} is malformed")
    return float(cast(int | float, value))


def _finite_vector(value: object, *, size: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"candidate v10 {field} vector is malformed")
    return tuple(_finite_number(item, field) for item in value)


def _finite_positive_vector(
    value: object, *, size: int, field: str
) -> tuple[float, ...]:
    result = _finite_vector(value, size=size, field=field)
    if any(item <= 0.0 for item in result):
        raise ValueError(f"candidate v10 {field} vector must be positive")
    return result


def _quantity_as_float(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        quantity = float(Fraction(value))
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    return quantity if math.isfinite(quantity) else None


def _project_exact_quantity(value: Fraction) -> float:
    """Match the conservative float projection used by exact book quantities."""

    try:
        projected = float(value)
    except OverflowError as error:
        raise ValueError(
            "candidate v10 exact quantity is outside float range"
        ) from error
    if not math.isfinite(projected):
        raise ValueError("candidate v10 exact quantity is outside float range")
    while abs(Fraction(str(projected))) > abs(value):
        projected = math.nextafter(projected, 0.0)
    return projected


def _validate_shared_cash_terminal_links(
    payload: Mapping[str, object],
    *,
    intervals: Sequence[object],
    summary: Mapping[str, object],
) -> None:
    if not intervals or not isinstance(intervals[-1], Mapping):
        raise ValueError("candidate shared-cash terminal ledger link is inconsistent")
    terminal_interval = intervals[-1]
    ledger_links = (
        ("cash_after", "final_cash"),
        ("portfolio_value_after", "final_portfolio_value"),
        ("total_cost_after", "final_total_cost"),
        ("funding_pnl_after", "final_funding_pnl"),
        ("borrow_cost_after", "final_borrow_cost"),
        ("turnover_total_after", "final_turnover_total"),
        ("max_drawdown_after", "final_max_drawdown"),
    )
    if (
        any(
            not _numbers_are_close(terminal_interval[row_field], payload[ledger_field])
            for row_field, ledger_field in ledger_links
        )
        or terminal_interval["exact_quantities_after"]
        != payload["terminal_exact_quantities"]
    ):
        raise ValueError("candidate shared-cash terminal ledger link is inconsistent")

    portfolio = summary.get("shared_cash_ppo")
    if not isinstance(portfolio, Mapping):
        raise ValueError("candidate shared-cash portfolio ledger link is inconsistent")
    metrics = portfolio.get("metrics")
    diagnostics = portfolio.get("diagnostics")
    if not isinstance(metrics, Mapping) or not isinstance(diagnostics, Mapping):
        raise ValueError("candidate shared-cash portfolio ledger link is inconsistent")
    portfolio_links = (
        ("final_cash", "final_cash"),
        ("final_portfolio_value", "final_portfolio_value"),
    )
    diagnostic_links = (
        ("final_total_cost", "total_cost"),
        ("final_funding_pnl", "funding_pnl"),
        ("final_borrow_cost", "borrow_cost"),
        ("final_turnover_total", "turnover_total"),
    )
    if (
        any(
            not _numbers_are_close(payload[ledger_field], portfolio[summary_field])
            for ledger_field, summary_field in portfolio_links
        )
        or any(
            not _numbers_are_close(payload[ledger_field], diagnostics[diagnostic_field])
            for ledger_field, diagnostic_field in diagnostic_links
        )
        or not _numbers_are_close(
            payload["final_max_drawdown"], metrics.get("max_drawdown")
        )
    ):
        raise ValueError("candidate shared-cash portfolio ledger link is inconsistent")

    exact_quantities = payload.get("terminal_exact_quantities")
    final_quantities = portfolio.get("final_quantities")
    if (
        not isinstance(exact_quantities, (list, tuple))
        or not isinstance(final_quantities, list)
        or len(exact_quantities) != len(final_quantities)
        or any(
            not _numbers_are_close(_quantity_as_float(exact), quantity)
            for exact, quantity in zip(exact_quantities, final_quantities, strict=True)
        )
    ):
        raise ValueError("candidate shared-cash portfolio ledger link is inconsistent")

    payload_remainders = payload.get("active_order_remainders")
    summary_remainders = portfolio.get("active_order_remainders")
    if (
        not isinstance(payload_remainders, (list, tuple))
        or not isinstance(summary_remainders, list)
        or len(payload_remainders) != len(summary_remainders)
    ):
        raise ValueError("candidate shared-cash portfolio ledger link is inconsistent")
    for payload_order, summary_order in zip(
        payload_remainders,
        summary_remainders,
        strict=True,
    ):
        if (
            not isinstance(payload_order, (list, tuple))
            or len(payload_order) != 2
            or not isinstance(summary_order, Mapping)
            or payload_order[0] != summary_order.get("order_id")
            or not _numbers_are_close(
                payload_order[1],
                summary_order.get("remaining_quantity"),
            )
        ):
            raise ValueError(
                "candidate shared-cash portfolio ledger link is inconsistent"
            )

    payload_reasons = payload.get("terminal_order_reasons")
    summary_reasons = portfolio.get("terminal_order_reasons")
    if (
        not isinstance(payload_reasons, (list, tuple))
        or not isinstance(summary_reasons, list)
        or len(payload_reasons) != len(summary_reasons)
        or any(
            not isinstance(payload_reason, (list, tuple))
            or len(payload_reason) != 2
            or not isinstance(summary_reason, Mapping)
            or payload_reason[0] != summary_reason.get("order_id")
            or payload_reason[1] != summary_reason.get("reason")
            for payload_reason, summary_reason in zip(
                payload_reasons,
                summary_reasons,
                strict=True,
            )
        )
    ):
        raise ValueError("candidate shared-cash portfolio ledger link is inconsistent")

    termination_reason = payload.get("termination_reason")
    reasons = diagnostics.get("termination_reasons")
    termination_count = metrics.get("termination_count")
    if (
        (termination_reason is not None and not isinstance(termination_reason, str))
        or (isinstance(termination_reason, str) and not termination_reason)
        or not isinstance(reasons, list)
        or reasons != ([] if termination_reason is None else [termination_reason])
        or termination_count != len(reasons)
    ):
        raise ValueError("candidate shared-cash portfolio ledger link is inconsistent")


def _validate_v8_replay_evidence(summary: Mapping[str, object]) -> None:
    _validate_v7_replay_evidence(summary, require_ledger_payload=True)


def _validate_v9_replay_evidence(summary: Mapping[str, object]) -> None:
    _validate_v7_replay_evidence(
        summary,
        require_ledger_payload=True,
        strict_ledger_semantics=True,
    )
    portfolio = summary.get("shared_cash_ppo")
    ledger = (
        portfolio.get("ledger_evidence") if isinstance(portfolio, Mapping) else None
    )
    if not isinstance(ledger, Mapping) or ledger.get("schema_version") != (
        "shared_cash_replay_ledger_v2"
    ):
        raise ValueError("candidate v9 shared-cash ledger schema must be v2")


def _validate_v10_replay_evidence(summary: Mapping[str, object]) -> None:
    _validate_v7_replay_evidence(
        summary,
        require_ledger_payload=True,
        strict_ledger_semantics=True,
    )
    portfolio = summary.get("shared_cash_ppo")
    ledger = (
        portfolio.get("ledger_evidence") if isinstance(portfolio, Mapping) else None
    )
    if not isinstance(ledger, Mapping) or ledger.get("schema_version") != (
        "shared_cash_replay_ledger_v3"
    ):
        raise ValueError("candidate v10 shared-cash ledger schema must be v3")
    payload = ledger.get("payload")
    symbols = summary.get("symbols")
    intervals = payload.get("intervals") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or not isinstance(symbols, list)
        or not isinstance(intervals, (list, tuple))
    ):
        raise ValueError("candidate v10 accounting evidence is malformed")
    _validate_v10_accounting_transitions(
        payload=payload,
        intervals=intervals,
        symbols=symbols,
        summary=summary,
    )
    _validate_v10_decision_content(
        payload=payload,
        intervals=intervals,
        summary=summary,
    )


def _validate_v11_replay_evidence(summary: Mapping[str, object]) -> None:
    _validate_v7_replay_evidence(
        summary,
        require_ledger_payload=True,
        strict_ledger_semantics=True,
    )
    portfolio = summary.get("shared_cash_ppo")
    ledger = (
        portfolio.get("ledger_evidence") if isinstance(portfolio, Mapping) else None
    )
    if not isinstance(ledger, Mapping) or ledger.get("schema_version") != (
        "shared_cash_replay_ledger_v4"
    ):
        raise ValueError("candidate v11 shared-cash ledger schema must be v4")
    payload = ledger.get("payload")
    symbols = summary.get("symbols")
    intervals = payload.get("intervals") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or not isinstance(symbols, list)
        or not isinstance(intervals, (list, tuple))
    ):
        raise ValueError("candidate v11 accounting evidence is malformed")
    _validate_v10_accounting_transitions(
        payload=payload,
        intervals=intervals,
        symbols=symbols,
        summary=summary,
        require_ohlc_stress=True,
    )
    _validate_v10_decision_content(
        payload=payload,
        intervals=intervals,
        summary=summary,
    )


def _validate_v10_decision_content(
    *,
    payload: Mapping[str, object],
    intervals: Sequence[object],
    summary: Mapping[str, object],
) -> None:
    evaluation = summary.get("evaluation")
    symbols = summary.get("symbols")
    if not isinstance(evaluation, Mapping) or not isinstance(symbols, list):
        raise ValueError("candidate v10 decision evidence is malformed")
    gross_budget = evaluation.get("gross_budget")
    minimum_hold_bars = evaluation.get("ppo_minimum_hold_bars")
    raw_risk_config = evaluation.get("pretrade_risk_config")
    if (
        not _is_finite_number(gross_budget)
        or not 0.0 < float(cast(int | float, gross_budget)) <= 1.0
        or isinstance(minimum_hold_bars, bool)
        or not isinstance(minimum_hold_bars, int)
        or minimum_hold_bars < 0
        or not isinstance(raw_risk_config, Mapping)
        or set(raw_risk_config)
        != {
            "max_gross",
            "max_abs_weight",
            "max_turnover",
            "drawdown_start",
            "drawdown_stop",
            "emergency_turnover_override",
            "fail_closed_tolerance",
        }
    ):
        raise ValueError("candidate v10 decision policy config is malformed")
    try:
        risk_config = PreTradeRiskConfig(**raw_risk_config)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("candidate v10 decision risk config is malformed") from error
    risk = PreTradeRisk(risk_config)
    decisions = payload.get("decisions")
    policy_stop = evaluation.get("ppo_policy_decision_stop_index")
    start = payload.get("start_index")
    if (
        not isinstance(decisions, (list, tuple))
        or isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(policy_stop, bool)
        or not isinstance(policy_stop, int)
        or len(decisions) != policy_stop - start
    ):
        raise ValueError("candidate v10 decision coverage is malformed")

    current_intents = [PositionIntent.FLAT for _ in symbols]
    position_ages = [0 for _ in symbols]
    minimum_hold_locked = [False for _ in symbols]
    desired_quantities = [0.0 for _ in symbols]
    for decision_offset, raw_decision in enumerate(decisions):
        if not isinstance(raw_decision, Mapping):
            raise ValueError("candidate v10 decision content is malformed")
        decision_index = raw_decision.get("index")
        interval_offset = decision_offset
        interval = intervals[interval_offset]
        if (
            decision_index != start + decision_offset
            or not isinstance(interval, Mapping)
            or interval.get("start_index") != decision_index
        ):
            raise ValueError("candidate v10 decision interval link is inconsistent")
        vector_fields = (
            "intents",
            "effective_intents",
            "changed_intents",
            "minimum_hold_suppressed",
            "minimum_hold_unlocked",
            "position_age_bars_before",
            "position_age_bars_after",
            "position_quantity_before",
            "position_quantity_after",
            "proposal_weights",
            "target_weights",
        )
        vectors: dict[str, Sequence[object]] = {}
        for field in vector_fields:
            value = raw_decision.get(field)
            if not isinstance(value, (list, tuple)) or len(value) != len(symbols):
                raise ValueError("candidate v10 decision vector is malformed")
            vectors[field] = value
        for field in (
            "changed_intents",
            "minimum_hold_suppressed",
            "minimum_hold_unlocked",
        ):
            if any(not isinstance(value, bool) for value in vectors[field]):
                raise ValueError("candidate v10 decision flags are malformed")
        for field in ("position_age_bars_before", "position_age_bars_after"):
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in vectors[field]
            ):
                raise ValueError("candidate v10 decision ages are malformed")
        requested_intents: list[PositionIntent] = []
        effective_intents: list[PositionIntent] = []
        for field, destination in (
            ("intents", requested_intents),
            ("effective_intents", effective_intents),
        ):
            for value in vectors[field]:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("candidate v10 decision intent is malformed")
                try:
                    destination.append(PositionIntent(value))
                except ValueError as error:
                    raise ValueError(
                        "candidate v10 decision intent is malformed"
                    ) from error
        quantity_before = _finite_vector(
            vectors["position_quantity_before"],
            size=len(symbols),
            field="decision quantities before",
        )
        quantity_after = _finite_vector(
            vectors["position_quantity_after"],
            size=len(symbols),
            field="decision quantities after",
        )
        proposal_weights = _finite_vector(
            vectors["proposal_weights"],
            size=len(symbols),
            field="decision proposal weights",
        )
        target_weights = _finite_vector(
            vectors["target_weights"],
            size=len(symbols),
            field="decision target weights",
        )
        raw_transitions = interval.get("accounting_transitions")
        if not isinstance(raw_transitions, (list, tuple)) or not raw_transitions:
            raise ValueError("candidate v10 decision state is unavailable")
        first_transition = _accounting_transition_mapping(raw_transitions[0])
        state_before = _accounting_state_mapping(
            first_transition["state_before"], size=len(symbols)
        )
        nav = _accounting_nav(state_before)
        if nav <= 0.0:
            raise ValueError("candidate v10 decision state has non-positive equity")
        if any(
            not _numbers_are_close(actual, float(expected))
            for actual, expected in zip(quantity_before, state_before[1], strict=True)
        ):
            raise ValueError("candidate v10 decision quantity link is inconsistent")
        current_weights = np.asarray(
            [
                float(quantity) * mark * multiplier / nav
                for quantity, mark, multiplier in zip(
                    state_before[1], state_before[2], state_before[3], strict=True
                )
            ],
            dtype=np.float64,
        )
        expected_proposal_weights: list[float] = []
        for symbol_index, requested_intent in enumerate(requested_intents):
            if (
                vectors["position_age_bars_before"][symbol_index]
                != position_ages[symbol_index]
            ):
                raise ValueError("candidate v10 decision age link is inconsistent")
            hold_decision = constrain_intent_for_minimum_hold(
                requested_intent,
                current_quantity=quantity_before[symbol_index],
                position_age_bars=position_ages[symbol_index],
                minimum_hold_bars=minimum_hold_bars,
            )
            effective = hold_decision.effective_intent
            changed = effective is not current_intents[symbol_index]
            unlocked = (
                minimum_hold_locked[symbol_index] and not hold_decision.suppressed
            )
            if (
                effective != effective_intents[symbol_index]
                or changed is not vectors["changed_intents"][symbol_index]
                or hold_decision.suppressed
                is not vectors["minimum_hold_suppressed"][symbol_index]
                or unlocked is not vectors["minimum_hold_unlocked"][symbol_index]
            ):
                raise ValueError("candidate v10 decision intent link is inconsistent")
            if hold_decision.target_quantity_override is not None:
                desired_quantities[symbol_index] = (
                    hold_decision.target_quantity_override
                )
            elif changed or unlocked:
                desired_quantities[symbol_index] = (
                    target_weight_for_intent(
                        effective,
                        gross_budget=float(cast(int | float, gross_budget)),
                    )
                    * nav
                    / (state_before[2][symbol_index] * state_before[3][symbol_index])
                )
            expected_proposal_weights.append(
                desired_quantities[symbol_index]
                * state_before[2][symbol_index]
                * state_before[3][symbol_index]
                / nav
            )
        if any(
            not _numbers_are_close(actual, expected)
            for actual, expected in zip(
                proposal_weights, expected_proposal_weights, strict=True
            )
        ):
            raise ValueError("candidate v10 decision proposal link is inconsistent")
        try:
            constrained = risk.constrain(
                np.asarray(proposal_weights, dtype=np.float64),
                current=current_weights,
                drawdown=float(interval["max_drawdown_before"]),
            )
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError("candidate v10 decision risk link is malformed") from error
        if any(
            not _numbers_are_close(actual, expected)
            for actual, expected in zip(
                target_weights, constrained.weights, strict=True
            )
        ) or raw_decision.get("risk_reasons") != list(constrained.reasons):
            raise ValueError("candidate v10 decision risk link is inconsistent")
        if should_rebind_strategy_proposal(constrained):
            for symbol_index in range(len(symbols)):
                desired_quantities[symbol_index] = (
                    float(constrained.weights[symbol_index])
                    * nav
                    / (state_before[2][symbol_index] * state_before[3][symbol_index])
                )
        for symbol_index in range(len(symbols)):
            expected_age_after = next_position_age_bars(
                position_ages[symbol_index],
                previous_quantity=quantity_before[symbol_index],
                filled_quantity=quantity_after[symbol_index],
            )
            if vectors["position_age_bars_after"][symbol_index] != expected_age_after:
                raise ValueError("candidate v10 decision age link is inconsistent")
            position_ages[symbol_index] = expected_age_after
            minimum_hold_locked[symbol_index] = (
                bool(vectors["minimum_hold_suppressed"][symbol_index])
                and quantity_after[symbol_index] != 0.0
            )
            current_intents[symbol_index] = effective_intents[symbol_index]


def _validate_v9_interval_return_binding(
    summary: Mapping[str, object],
    returns: Mapping[str, np.ndarray],
) -> None:
    portfolio = summary.get("shared_cash_ppo")
    ledger = (
        portfolio.get("ledger_evidence") if isinstance(portfolio, Mapping) else None
    )
    payload = ledger.get("payload") if isinstance(ledger, Mapping) else None
    intervals = payload.get("intervals") if isinstance(payload, Mapping) else None
    values = returns.get("shared_cash_ppo")
    if (
        not isinstance(intervals, (list, tuple))
        or values is None
        or values.size != len(intervals)
    ):
        raise ValueError("candidate shared-cash return ledger link is inconsistent")
    if any(
        not _numbers_are_close(interval["interval_net_return"], values[index])
        for index, interval in enumerate(intervals)
        if isinstance(interval, Mapping)
    ) or any(not isinstance(interval, Mapping) for interval in intervals):
        raise ValueError("candidate shared-cash return ledger link is inconsistent")


def _load_returns(
    payload: bytes,
    *,
    expected_keys: frozenset[str],
) -> dict[str, np.ndarray]:
    loaded: dict[str, np.ndarray] = {}
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            keys = frozenset(archive.files)
            if keys != expected_keys:
                raise ValueError("candidate return keys do not match summary")
            for key in sorted(keys):
                try:
                    value = np.asarray(archive[key])
                except ValueError as error:
                    raise ValueError(
                        "candidate return arrays must be numeric and pickle-free"
                    ) from error
                if not np.issubdtype(value.dtype, np.number):
                    raise ValueError("candidate return arrays must be numeric")
                if value.ndim != 1:
                    raise ValueError("candidate return arrays must be one-dimensional")
                if not np.isfinite(value).all():
                    raise ValueError("candidate return arrays must be finite")
                loaded[key] = np.ascontiguousarray(value).copy()
    except (OSError, EOFError, zipfile.BadZipFile, zlib.error) as error:
        raise ValueError("malformed candidate returns archive") from error
    return loaded


def _validate_v6_return_coverage(
    summary: Mapping[str, object],
    returns: Mapping[str, np.ndarray],
) -> None:
    evaluation = summary.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("candidate return coverage evidence is malformed")
    expected_periods = evaluation.get("expected_periods")
    by_symbol = summary.get("by_symbol")
    if (
        isinstance(expected_periods, bool)
        or not isinstance(expected_periods, int)
        or expected_periods <= 0
        or not isinstance(by_symbol, list)
    ):
        raise ValueError("candidate return coverage evidence is malformed")
    for symbol in by_symbol:
        if not isinstance(symbol, Mapping) or not isinstance(
            symbol.get("strategies"), list
        ):
            raise ValueError("candidate return coverage evidence is malformed")
        for strategy in symbol["strategies"]:
            if not isinstance(strategy, Mapping):
                raise ValueError("candidate return coverage evidence is malformed")
            key = strategy.get("return_key")
            metrics = strategy.get("metrics")
            values = returns.get(key) if isinstance(key, str) else None
            n_periods = (
                metrics.get("n_periods") if isinstance(metrics, Mapping) else None
            )
            if (
                values is None
                or isinstance(n_periods, bool)
                or not isinstance(n_periods, int)
                or n_periods < 0
                or n_periods > expected_periods
                or values.size != n_periods
            ):
                raise ValueError("candidate return period count is inconsistent")


def _validate_v7_return_coverage(
    summary: Mapping[str, object],
    returns: Mapping[str, np.ndarray],
) -> None:
    _validate_v6_return_coverage(summary, returns)
    evaluation = summary.get("evaluation")
    portfolio = summary.get("shared_cash_ppo")
    if not isinstance(evaluation, Mapping) or not isinstance(portfolio, Mapping):
        raise ValueError("candidate shared-cash return evidence is malformed")
    expected_periods = evaluation.get("expected_periods")
    key = portfolio.get("return_key")
    values = returns.get(key) if isinstance(key, str) else None
    metrics = portfolio.get("metrics")
    if (
        isinstance(expected_periods, bool)
        or not isinstance(expected_periods, int)
        or values is None
        or values.size != expected_periods
        or not isinstance(metrics, Mapping)
    ):
        raise ValueError("candidate shared-cash return coverage is incomplete")
    wealth = 1.0
    peak = 1.0
    maximum_drawdown = 0.0
    for value in values:
        wealth *= 1.0 + float(value)
        peak = max(peak, wealth)
        maximum_drawdown = max(maximum_drawdown, 1.0 - wealth / peak)
    total_return = wealth - 1.0
    checked_metrics = [("total_return", total_return)]
    if summary.get("schema_version") not in {
        _RESULT_SCHEMA_V10,
        _RESULT_SCHEMA_V11,
    }:
        checked_metrics.append(("max_drawdown", maximum_drawdown))
    for field, actual in checked_metrics:
        reported = metrics.get(field)
        if (
            isinstance(reported, bool)
            or not isinstance(reported, (int, float))
            or not math.isclose(
                float(reported),
                actual,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("candidate shared-cash return metrics are inconsistent")


def _semantic_returns_payload(
    returns: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for key in sorted(returns):
        value = np.ascontiguousarray(returns[key])
        payload.append(
            {
                "key": key,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": sha256(value.tobytes(order="C")).hexdigest(),
            }
        )
    return payload


def _load_with_evidence(
    root: str | Path,
) -> tuple[LoadedCandidateRun, ArtifactFileEvidence]:
    artifact_root = Path(root)
    summary_path, returns_path, provenance_path = _validate_root(artifact_root)

    summary_evidence = _verified_bytes(summary_path, label="summary")
    returns_evidence = _verified_bytes(returns_path, label="returns")
    provenance_evidence = _verified_bytes(provenance_path, label="provenance")
    summary_bytes, summary_hash, summary_size = summary_evidence
    returns_bytes, returns_hash, returns_size = returns_evidence
    provenance_bytes, provenance_hash, provenance_size = provenance_evidence

    summary = _read_json_object(summary_bytes, label="summary")
    result_schema = summary.get("schema_version")
    if result_schema not in _SUPPORTED_RESULT_SCHEMAS:
        raise ValueError("unsupported candidate result schema")
    if (
        result_schema in {_RESULT_SCHEMA_V2, _RESULT_SCHEMA_V3}
        and summary.get("ppo_observation") != ppo_observation_contract_payload()
    ):
        raise ValueError("candidate PPO observation contract mismatch")
    if result_schema == _RESULT_SCHEMA_V3:
        _validate_ppo_training_evidence(summary, result_schema=result_schema)
    if result_schema == _RESULT_SCHEMA_V4:
        _validate_ppo_training_evidence(summary, result_schema=result_schema)
    if result_schema == _RESULT_SCHEMA_V5:
        _validate_ppo_training_evidence(summary, result_schema=result_schema)
        _validate_v5_replay_evidence(summary)
    if result_schema == _RESULT_SCHEMA_V6:
        _validate_ppo_training_evidence(summary, result_schema=result_schema)
        _validate_v6_replay_evidence(summary)
    if result_schema == _RESULT_SCHEMA_V7:
        _validate_ppo_training_evidence(summary, result_schema=result_schema)
        _validate_v7_replay_evidence(summary)
    if result_schema == _RESULT_SCHEMA_V8:
        _validate_ppo_training_evidence(summary, result_schema=result_schema)
        _validate_v8_replay_evidence(summary)
    if result_schema == _RESULT_SCHEMA_V9:
        _validate_ppo_training_evidence(summary, result_schema=result_schema)
        _validate_v9_replay_evidence(summary)
    if result_schema == _RESULT_SCHEMA_V10:
        _validate_ppo_training_evidence(summary, result_schema=result_schema)
        _validate_v10_replay_evidence(summary)
    if result_schema == _RESULT_SCHEMA_V11:
        _validate_ppo_training_evidence(summary, result_schema=result_schema)
        _validate_v11_replay_evidence(summary)
    dataset_id = summary.get("dataset_id")
    if isinstance(dataset_id, str):
        require_sha256(dataset_id, field="candidate dataset_id")
    provenance = _read_json_object(provenance_bytes, label="provenance")
    _validate_provenance(provenance)
    returns = _load_returns(
        returns_bytes,
        expected_keys=_expected_return_keys(summary),
    )
    if result_schema == _RESULT_SCHEMA_V6:
        _validate_v6_return_coverage(summary, returns)
    if result_schema == _RESULT_SCHEMA_V7:
        _validate_v7_return_coverage(summary, returns)
    if result_schema == _RESULT_SCHEMA_V8:
        _validate_v7_return_coverage(summary, returns)
    if result_schema == _RESULT_SCHEMA_V9:
        _validate_v7_return_coverage(summary, returns)
        _validate_v9_interval_return_binding(summary, returns)
    if result_schema == _RESULT_SCHEMA_V10:
        _validate_v7_return_coverage(summary, returns)
        _validate_v9_interval_return_binding(summary, returns)
    if result_schema == _RESULT_SCHEMA_V11:
        _validate_v7_return_coverage(summary, returns)
        _validate_v9_interval_return_binding(summary, returns)
    loaded = LoadedCandidateRun(
        root=artifact_root,
        summary=summary,
        returns=returns,
        provenance=provenance,
    )
    evidence: ArtifactFileEvidence = (
        (summary_hash, summary_size),
        (returns_hash, returns_size),
        (provenance_hash, provenance_size),
    )
    return loaded, evidence


def load_candidate_run_artifact(root: str | Path) -> LoadedCandidateRun:
    """Load one exact three-file candidate artifact and validate semantic content."""

    loaded, _ = _load_with_evidence(root)
    return loaded


def inspect_candidate_run_artifact(root: str | Path) -> CandidateRunArtifactIdentity:
    """Return stable semantic identity plus raw file digests/sizes."""

    loaded, evidence = _load_with_evidence(root)
    summary_evidence, returns_evidence, provenance_evidence = evidence
    summary_hash, summary_size = summary_evidence
    returns_hash, returns_size = returns_evidence
    provenance_hash, provenance_size = provenance_evidence
    semantic_payload = {
        "schema_version": _ARTIFACT_IDENTITY_SCHEMA,
        "summary": loaded.summary,
        "returns": _semantic_returns_payload(loaded.returns),
        "provenance": loaded.provenance,
    }
    return CandidateRunArtifactIdentity(
        schema_version=_ARTIFACT_IDENTITY_SCHEMA,
        result_schema_version=cast(str, loaded.summary["schema_version"]),
        artifact_digest=content_digest(semantic_payload),
        summary_file_sha256=summary_hash,
        summary_file_size=summary_size,
        returns_file_sha256=returns_hash,
        returns_file_size=returns_size,
        provenance_file_sha256=provenance_hash,
        provenance_file_size=provenance_size,
    )


__all__ = [
    "CandidateRunArtifactIdentity",
    "LoadedCandidateRun",
    "PublishedCandidateRun",
    "inspect_candidate_run_artifact",
    "load_candidate_run_artifact",
    "publish_candidate_run",
]
