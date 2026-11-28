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
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.atomic_write import atomic_rename_directory, atomic_write_bytes
from trade_rl.artifacts.canonical import freeze_json_value, strict_json_loads
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import file_digest_and_size, read_verified_bytes
from trade_rl.evaluation.metrics import PerformanceMetrics
from trade_rl.evaluation.runs.config import LEGACY_DATASET_EXECUTION_OVERLAY
from trade_rl.evaluation.runs.execute import CandidateRunResult
from trade_rl.evaluation.runs.provenance import PROVENANCE_SCHEMA
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.strategies.forecasts.supervised import (
    MAX_SUPERVISED_FORECAST_HORIZON_HOURS,
)
from trade_rl.strategies.rl.intent import (
    PPO_OBSERVATION_SCHEMA_V3,
    ppo_observation_contract_payload,
)
from trade_rl.strategies.rl.ppo_training import expected_ppo_realized_timesteps

_ARCHIVE_DECODE_ERRORS: tuple[type[Exception], ...] = (
    OSError,
    EOFError,
    zipfile.BadZipFile,
    zlib.error,
    NotImplementedError,
)
try:
    from lzma import LZMAError
except ImportError:
    pass
else:
    _ARCHIVE_DECODE_ERRORS += (LZMAError,)

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
_SHARED_CASH_RESULT_SCHEMAS = frozenset(
    {_RESULT_SCHEMA_V7, _RESULT_SCHEMA_V9, _RESULT_SCHEMA_V10}
)


def supports_shared_cash_result_schema(result_schema: object) -> bool:
    """Accept only native shared-cash schema strings, not overridden aliases."""

    return type(result_schema) is str and result_schema in _SHARED_CASH_RESULT_SCHEMAS


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
    }
)
_ARTIFACT_IDENTITY_SCHEMA = "candidate_run_artifact_identity_v1"
_REQUIRED_FILES = frozenset({"summary.json", "returns.npz", "provenance.json"})

FileEvidence = tuple[str, int]
ArtifactFileEvidence = tuple[FileEvidence, FileEvidence, FileEvidence]


def _require_coverage_roster(summary: Mapping[str, object]) -> frozenset[str]:
    """Bind full-coverage evidence to the declared ordered comparison matrix."""
    symbols = summary.get("symbols")
    by_symbol = summary.get("by_symbol")
    if (
        not isinstance(symbols, (list, tuple))
        or not symbols
        or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
        or len(symbols) != len(set(symbols))
        or not isinstance(by_symbol, (list, tuple))
        or len(by_symbol) != len(symbols)
    ):
        raise ValueError("candidate coverage symbol roster is incomplete or malformed")
    keys: set[str] = set()
    expected_names: tuple[str, ...] | None = None
    for index, (symbol, row) in enumerate(zip(symbols, by_symbol, strict=True)):
        if (
            not isinstance(row, Mapping)
            or row.get("symbol") != symbol
            or type(row.get("symbol_index")) is not int
            or row.get("symbol_index") != index
        ):
            raise ValueError("candidate coverage symbol row does not match roster")
        strategies = row.get("strategies")
        if not isinstance(strategies, (list, tuple)) or not strategies:
            raise ValueError("candidate coverage strategy roster is empty or malformed")
        names: list[str] = []
        for strategy in strategies:
            if not isinstance(strategy, Mapping):
                raise ValueError("candidate coverage strategy row is malformed")
            name = strategy.get("name")
            key = strategy.get("return_key")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("candidate coverage strategy names must be unique")
            if not isinstance(key, str) or not key or key in keys:
                raise ValueError("candidate coverage return keys must be unique")
            names.append(name)
            keys.add(key)
        if expected_names is None:
            expected_names = tuple(names)
        elif tuple(names) != expected_names:
            raise ValueError("candidate coverage strategy rosters do not match")
    schema = summary.get("schema_version")
    if schema in (_RESULT_SCHEMA_V7, _RESULT_SCHEMA_V9) or "shared_cash_ppo" in summary:
        portfolio = summary.get("shared_cash_ppo")
        if not supports_shared_cash_result_schema(schema) or not isinstance(
            portfolio, Mapping
        ):
            raise ValueError("candidate coverage shared-cash row is malformed")
        key = portfolio.get("return_key")
        if not isinstance(key, str) or not key or key in keys:
            raise ValueError("candidate coverage return keys must be unique")
        keys.add(key)
    return frozenset(keys)


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
            raw = np.asarray(value)
            contiguous = np.ascontiguousarray(raw)
            immutable_returns[key] = np.frombuffer(
                contiguous.tobytes(order="C"),
                dtype=contiguous.dtype,
            ).reshape(raw.shape)

        object.__setattr__(self, "summary", frozen_summary)
        object.__setattr__(self, "returns", MappingProxyType(immutable_returns))
        object.__setattr__(self, "provenance", frozen_provenance)

    @property
    def artifact_digest(self) -> str:
        """Semantic identity of this loaded content, without reading its path.

        This uses the same versioned payload as artifact inspection; it is not
        evidence of current filesystem contents or external source authenticity.
        """
        return content_digest(
            {
                "schema_version": _ARTIFACT_IDENTITY_SCHEMA,
                "summary": self.summary,
                "returns": _semantic_returns_payload(self.returns),
                "provenance": self.provenance,
            }
        )

    @property
    def has_verified_full_evaluation_coverage(self) -> bool:
        """Whether this artifact schema verifies every requested evaluation period."""
        schema = self.summary.get("schema_version")
        if type(schema) is not str or schema not in {
            _RESULT_SCHEMA_V6,
            _RESULT_SCHEMA_V7,
            _RESULT_SCHEMA_V8,
            _RESULT_SCHEMA_V9,
            _RESULT_SCHEMA_V10,
        }:
            return False
        try:
            keys = _require_coverage_roster(self.summary)
        except ValueError:
            return False
        if keys != set(self.returns):
            return False
        if any(
            values.ndim != 1
            or values.dtype.kind not in {"i", "u", "f"}
            or not np.isfinite(values).all()
            for values in self.returns.values()
        ):
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
        if type(expected_periods) is not int or expected_periods <= 0:
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
                    or type(n_periods) is not int
                    or n_periods != expected_periods
                    or values.size != expected_periods
                ):
                    return False
        if schema in {_RESULT_SCHEMA_V7, _RESULT_SCHEMA_V9} or (
            schema == _RESULT_SCHEMA_V10 and "shared_cash_ppo" in self.summary
        ):
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
                or type(n_periods) is not int
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
            terminal_complete = (
                config.ppo_settle_terminal_position
                and not any(quantity != 0.0 for quantity in final_quantities)
                and not active_order_remainders
                and len(entry.replay.returns.values) == expected_periods
                and entry.metrics.termination_count == 0
                and not diagnostics.termination_reasons
            )
            # The existing per-symbol schema stores reporting quantities only.
            # Refuse a completed claim that would erase an exact residual.
            if terminal_complete and any(entry.replay.book.exact_quantities):
                raise ValueError("candidate exact terminal quantities are nonzero")
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
                    "terminal_settlement_complete": terminal_complete,
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
        "schema_version": _RESULT_SCHEMA_V10,
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
            "supervised_forecast_horizon_hours": (
                lean_config.supervised_forecast_horizon_hours
            ),
            "forecast_switch_cost": lean_config.forecast_switch_cost,
        },
        "evaluation": {
            "start": str(config.evaluation_start),
            "stop_exclusive": str(config.evaluation_stop_exclusive),
            "expected_periods": expected_periods,
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
        },
        "by_symbol": symbols_payload,
    }
    shared_cash_ppo = result.comparison.shared_cash_ppo
    if shared_cash_ppo is not None:
        replay = shared_cash_ppo.replay
        ledger = replay.ledger_evidence
        if ledger is None:
            raise ValueError("shared-cash PPO result requires ledger evidence")
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
        if terminal_complete and (
            any(replay.book.exact_quantities)
            or ledger.terminal_exact_quantities != ("0",) * len(final_quantities)
        ):
            raise ValueError(
                "candidate shared-cash exact terminal quantities are nonzero"
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
                "digest": content_digest(ledger.to_mapping()),
                "interval_count": len(ledger.intervals),
                "decision_count": len(ledger.decisions),
                "terminal_exact_quantities": list(ledger.terminal_exact_quantities),
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
    expected_keys = _require_coverage_roster(summary)
    if expected_keys != set(returns):
        raise ValueError("candidate return keys do not match summary")
    expected_counts = _expected_return_period_counts(summary)
    for key, values in returns.items():
        _validate_return_vector(values, expected_count=expected_counts[key])
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
        raw = strict_json_loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed candidate {label} JSON") from error
    except ValueError as error:
        raise ValueError(f"candidate {label} JSON: {error}") from error
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
    if "shared_cash_ppo" in summary:
        _validate_shared_cash_result_schema(
            summary,
            result_schema=summary.get("schema_version"),
        )
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


def _validate_shared_cash_result_schema(
    summary: Mapping[str, object],
    *,
    result_schema: object,
) -> None:
    if "shared_cash_ppo" in summary and not supports_shared_cash_result_schema(
        result_schema
    ):
        raise ValueError(
            "candidate shared-cash PPO evidence is unsupported by result schema: "
            f"{result_schema}"
        )


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
        if result_schema in {_RESULT_SCHEMA_V8, _RESULT_SCHEMA_V9, _RESULT_SCHEMA_V10}:
            horizon = candidate_config.get("supervised_forecast_horizon_hours")
            if (
                isinstance(horizon, bool)
                or not isinstance(horizon, int)
                or horizon <= 0
                or horizon > MAX_SUPERVISED_FORECAST_HORIZON_HOURS
            ):
                raise ValueError("candidate supervised forecast horizon is malformed")
        if result_schema == _RESULT_SCHEMA_V10:
            switch_cost = candidate_config.get("forecast_switch_cost")
            if switch_cost is not None and (
                isinstance(switch_cost, bool)
                or not isinstance(switch_cost, (int, float))
                or not math.isfinite(float(switch_cost))
                or float(switch_cost) < 0.0
            ):
                raise ValueError("candidate forecast switch cost is malformed")
            if "forecast_switch_cost" not in candidate_config:
                raise ValueError("candidate forecast switch cost identity is missing")
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
        _require_coverage_roster(summary)
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


def _validate_v7_replay_evidence(summary: Mapping[str, object]) -> None:
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
    if not isinstance(ledger, Mapping) or set(ledger) != ledger_fields:
        raise ValueError("candidate shared-cash ledger evidence is malformed")
    if ledger["schema_version"] not in {
        "shared_cash_replay_ledger_v1",
        "shared_cash_replay_ledger_v2",
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
    # Every historical shared-cash writer serializes Fraction quantities with
    # str(value). Exact zero is therefore "0", not a value rounded to zero
    # by binary64 conversion (or a noncanonical alternate spelling).
    if terminal_complete and any(value != "0" for value in exact_quantities):
        raise ValueError("candidate shared-cash exact terminal quantities are nonzero")


def _validate_v8_replay_evidence(summary: Mapping[str, object]) -> None:
    _validate_shared_cash_result_schema(summary, result_schema=_RESULT_SCHEMA_V8)
    _validate_v6_replay_evidence(summary)


def _validate_v9_replay_evidence(summary: Mapping[str, object]) -> None:
    _validate_v7_replay_evidence(summary)


def _validate_v10_replay_evidence(summary: Mapping[str, object]) -> None:
    _validate_shared_cash_result_schema(summary, result_schema=_RESULT_SCHEMA_V10)
    if "shared_cash_ppo" in summary:
        _validate_v7_replay_evidence(summary)
    else:
        _validate_v8_replay_evidence(summary)


def _validate_return_member_extent(
    archive: zipfile.ZipFile, member: zipfile.ZipInfo
) -> int:
    """Reject inconsistent NPY extents before array materialization.

    ZIP's uncompressed byte count is not an independent resource budget.
    Match it to a bounded header using Python's exact integer arithmetic.
    """
    try:
        with archive.open(member) as handle:
            version = np.lib.format.read_magic(handle)
            if version not in {(1, 0), (2, 0), (3, 0)}:
                raise ValueError("unsupported NPY version")
            width = 2 if version == (1, 0) else 4
            length_bytes = handle.read(width)
            if len(length_bytes) != width:
                raise ValueError("short NPY header length")
            header_length = int.from_bytes(length_bytes, "little")
            if (
                header_length > 10_000
                or header_length > member.file_size - handle.tell()
            ):
                raise ValueError("invalid NPY header extent")
            header = handle.read(header_length)
            if len(header) != header_length:
                raise ValueError("short NPY header")
            # The admitted primitive numeric dtype header is ASCII even in
            # NPY v3, whose length field matches v2. Avoid private NumPy APIs.
            if version == (3, 0) and not header.isascii():
                raise ValueError("non-ASCII primitive NPY header")
            read_header = (
                np.lib.format.read_array_header_1_0
                if version == (1, 0)
                else np.lib.format.read_array_header_2_0
            )
            shape, _, dtype = read_header(io.BytesIO(length_bytes + header))
            header_end = handle.tell()
    except (ValueError, TypeError, OverflowError, RecursionError, SyntaxError) as error:
        raise ValueError(
            "malformed candidate returns archive: invalid NPY header extent"
        ) from error
    if dtype.kind not in {"i", "u", "f"}:
        raise ValueError("candidate return arrays must be real numeric and pickle-free")
    if len(shape) != 1:
        raise ValueError("candidate return arrays must be one-dimensional")
    if type(shape[0]) is not int or shape[0] < 0:
        raise ValueError("candidate return array shape is invalid")
    if shape[0] * dtype.itemsize != member.file_size - header_end:
        raise ValueError(
            "candidate return array byte extent does not match shape and dtype"
        )
    return shape[0]


def _validate_return_vector(value: np.ndarray, *, expected_count: int) -> None:
    """Apply the common persisted return-vector contract before publication or load."""
    if value.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("candidate return arrays must be real numeric")
    if value.ndim != 1:
        raise ValueError("candidate return arrays must be one-dimensional")
    if value.size != expected_count:
        raise ValueError("candidate return period count is inconsistent")
    if not np.isfinite(value).all():
        raise ValueError("candidate return arrays must be finite")


def _load_returns(
    payload: bytes,
    *,
    expected_keys: frozenset[str],
    expected_period_counts: Mapping[str, int] | None = None,
) -> dict[str, np.ndarray]:
    loaded: dict[str, np.ndarray] = {}
    # A bare NPY would be materialized rather than opened as a lazy archive.
    # Require the ZIP signature used by the NPZ writer before array decoding.
    if not payload.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        raise ValueError("malformed candidate returns archive: expected NPZ")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            members = archive.zip.namelist()
            if len(members) != len(set(members)):
                raise ValueError("candidate return archive has duplicate members")
            if set(members) != {f"{key}.npy" for key in expected_keys}:
                raise ValueError("candidate return keys do not match summary")
            # Reject unsupported physical member features before NumPy opens
            # or decodes an array. Keep all four standard Python ZIP codecs,
            # so valid historical archives need not be re-encoded.
            for member in archive.zip.infolist():
                if member.flag_bits & (1 | (1 << 5) | (1 << 6)):
                    raise ValueError(
                        "malformed candidate returns archive: unsupported member flags"
                    )
                if member.compress_type not in {
                    zipfile.ZIP_STORED,
                    zipfile.ZIP_DEFLATED,
                    zipfile.ZIP_BZIP2,
                    zipfile.ZIP_LZMA,
                }:
                    raise ValueError(
                        "malformed candidate returns archive: unsupported compression"
                    )
            declared_periods: dict[str, int] = {}
            for member in archive.zip.infolist():
                key = member.filename.removesuffix(".npy")
                count = _validate_return_member_extent(archive.zip, member)
                if (
                    expected_period_counts is not None
                    and expected_period_counts.get(key) != count
                ):
                    raise ValueError("candidate return period count is inconsistent")
                declared_periods[key] = count
            for key in sorted(expected_keys):
                try:
                    # Bind the physical member, not NumPy's suffix alias.
                    value = np.asarray(archive[f"{key}.npy"])
                except ValueError as error:
                    raise ValueError(
                        "candidate return arrays must be real numeric and pickle-free"
                    ) from error
                _validate_return_vector(value, expected_count=declared_periods[key])
                loaded[key] = np.ascontiguousarray(value).copy()
    except _ARCHIVE_DECODE_ERRORS as error:
        raise ValueError("malformed candidate returns archive") from error
    return loaded


def _expected_return_period_counts(
    summary: Mapping[str, object],
) -> dict[str, int]:
    """Derive modern per-member counts before opening the array payloads."""
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
    counts: dict[str, int] = {}
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
            n_periods = (
                metrics.get("n_periods") if isinstance(metrics, Mapping) else None
            )
            if (
                not isinstance(key, str)
                or isinstance(n_periods, bool)
                or not isinstance(n_periods, int)
                or n_periods < 0
                or n_periods > expected_periods
            ):
                raise ValueError("candidate return period count is inconsistent")
            counts[key] = n_periods
    if "shared_cash_ppo" in summary:
        portfolio = summary["shared_cash_ppo"]
        if not isinstance(portfolio, Mapping) or not isinstance(
            portfolio.get("return_key"), str
        ):
            raise ValueError("candidate shared-cash return evidence is malformed")
        metrics = portfolio.get("metrics")
        n_periods = metrics.get("n_periods") if isinstance(metrics, Mapping) else None
        if type(n_periods) is not int or n_periods != expected_periods:
            raise ValueError(
                "candidate shared-cash return period count is inconsistent"
            )
        counts[portfolio["return_key"]] = expected_periods
    return counts


def _validate_shared_cash_return_metrics(
    summary: Mapping[str, object],
    returns: Mapping[str, np.ndarray],
) -> None:
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
    for field, actual in (
        ("total_return", total_return),
        ("max_drawdown", maximum_drawdown),
    ):
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
    if (
        not isinstance(result_schema, str)
        or result_schema not in _SUPPORTED_RESULT_SCHEMAS
    ):
        raise ValueError("unsupported candidate result schema")
    candidate_config = summary.get("candidate_config")
    if (
        result_schema != _RESULT_SCHEMA_V10
        and isinstance(candidate_config, Mapping)
        and "forecast_switch_cost" in candidate_config
    ):
        raise ValueError("legacy candidate result cannot define forecast switch cost")
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
    dataset_id = summary.get("dataset_id")
    if isinstance(dataset_id, str):
        require_sha256(dataset_id, field="candidate dataset_id")
    provenance = _read_json_object(provenance_bytes, label="provenance")
    _validate_provenance(provenance)
    expected_keys = _expected_return_keys(summary)
    expected_counts = (
        _expected_return_period_counts(summary)
        if result_schema
        in {
            _RESULT_SCHEMA_V6,
            _RESULT_SCHEMA_V7,
            _RESULT_SCHEMA_V8,
            _RESULT_SCHEMA_V9,
            _RESULT_SCHEMA_V10,
        }
        else None
    )
    returns = _load_returns(
        returns_bytes,
        expected_keys=expected_keys,
        expected_period_counts=expected_counts,
    )
    if expected_counts is not None and "shared_cash_ppo" in summary:
        _validate_shared_cash_return_metrics(summary, returns)
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
    return CandidateRunArtifactIdentity(
        schema_version=_ARTIFACT_IDENTITY_SCHEMA,
        result_schema_version=cast(str, loaded.summary["schema_version"]),
        artifact_digest=loaded.artifact_digest,
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
    "supports_shared_cash_result_schema",
]
