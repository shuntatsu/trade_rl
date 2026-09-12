"""Immutable candidate-run publication, loading, and semantic identity."""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import file_digest_and_size, read_verified_bytes
from trade_rl.evaluation.metrics import PerformanceMetrics
from trade_rl.evaluation.runs.execute import CandidateRunResult
from trade_rl.evaluation.runs.provenance import PROVENANCE_SCHEMA

_RESULT_SCHEMA = "lean_candidate_result_v1"
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
    summary: dict[str, object]
    returns: dict[str, np.ndarray]
    provenance: dict[str, object]


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
                    },
                    "final_portfolio_value": entry.replay.book.portfolio_value,
                    "fill_count": entry.replay.book.fill_count,
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
        "schema_version": _RESULT_SCHEMA,
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
        },
        "evaluation": {
            "start": str(config.evaluation_start),
            "stop_exclusive": str(config.evaluation_stop_exclusive),
            "gross_budget": config.gross_budget,
            "initial_capital": config.initial_capital,
            "execution_overlay": "zero_overlay_dataset_fields_authoritative",
        },
        "by_symbol": symbols_payload,
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
        staging.rename(output)
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
    if len(keys) != len(set(keys)):
        raise ValueError("candidate summary return keys must be unique")
    return frozenset(keys)


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
                immutable = np.ascontiguousarray(value).copy()
                immutable.setflags(write=False)
                loaded[key] = immutable
    except (OSError, EOFError, zipfile.BadZipFile, zlib.error) as error:
        raise ValueError("malformed candidate returns archive") from error
    return loaded


def _semantic_returns_payload(
    returns: dict[str, np.ndarray],
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
    if summary.get("schema_version") != _RESULT_SCHEMA:
        raise ValueError("unsupported candidate result schema")
    dataset_id = summary.get("dataset_id")
    if isinstance(dataset_id, str):
        require_sha256(dataset_id, field="candidate dataset_id")
    provenance = _read_json_object(provenance_bytes, label="provenance")
    _validate_provenance(provenance)
    returns = _load_returns(
        returns_bytes,
        expected_keys=_expected_return_keys(summary),
    )
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
        result_schema_version=_RESULT_SCHEMA,
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
