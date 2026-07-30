"""Operator command for one resumable Binance symbol-triplet training stage."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.integrations.binance import BinanceMarket, BinancePublicTransport
from trade_rl.release.asymmetric import load_public_verification_keys
from trade_rl.workflows.binance_metadata_modes import (
    BinanceMetadataMode,
    BinanceMetadataResolution,
    VerifiedBinanceRuleHistory,
    load_verified_binance_rule_history,
    resolution_from_historical_signed,
    resolve_conservative_static,
    resolve_frozen_snapshot,
)
from trade_rl.workflows.binance_symbol_triplet_stage_runner import (
    binance_symbol_triplet_stage_root,
    current_binance_symbol_triplet_stage_request,
    execute_binance_symbol_triplet_postgres_stage,
)
from trade_rl.workflows.symbol_triplet_manifest import (
    SymbolTripletManifest,
    load_symbol_triplet_manifest,
)
from trade_rl.workflows.symbol_triplet_stage_orchestrator import (
    SymbolTripletStageRequest,
)
from trade_rl.workflows.symbol_triplet_training_cursor import (
    SymbolTripletTrainingPlan,
    load_symbol_triplet_training_plan,
)


def _json_object(path: Path, *, field: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} must be valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must be a JSON object: {path}")
    return dict(payload)


def _required_file_from_environment(name: str) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise ValueError(f"{name} is required")
    path = Path(raw)
    if not path.is_file():
        raise FileNotFoundError(f"{name} file is missing: {path}")
    return path


def _verified_history_from_environment() -> VerifiedBinanceRuleHistory:
    history_path = _required_file_from_environment("TRADE_RL_BINANCE_RULE_HISTORY")
    public_keys_path = _required_file_from_environment("TRADE_RL_METADATA_PUBLIC_KEYS")
    trusted_keys = load_public_verification_keys(public_keys_path)
    return load_verified_binance_rule_history(
        _json_object(history_path, field="signed Binance rule history"),
        trusted_keys=trusted_keys,
        trusted_now=datetime.now(UTC),
    )


def _validate_resolution_scope(
    request: SymbolTripletStageRequest,
    resolution: BinanceMetadataResolution,
) -> BinanceMetadataResolution:
    missing_metadata = tuple(
        symbol for symbol in request.symbols if symbol not in resolution.metadata
    )
    if missing_metadata:
        raise ValueError(
            "Binance metadata resolution is missing current stage symbols: "
            + ", ".join(missing_metadata)
        )
    histories = resolution.execution_rule_histories
    if histories is not None:
        missing_histories = tuple(
            symbol for symbol in request.symbols if symbol not in histories
        )
        if missing_histories:
            raise ValueError(
                "Binance execution-rule history is missing current stage symbols: "
                + ", ".join(missing_histories)
            )
    return resolution


def resolve_binance_symbol_triplet_stage_metadata(
    request: SymbolTripletStageRequest,
    *,
    mode: BinanceMetadataMode | str,
    transport: Any,
    start_time: datetime,
    end_time: datetime,
    conservative_static_path: Path | None = None,
    verified_history: VerifiedBinanceRuleHistory | None = None,
) -> BinanceMetadataResolution:
    """Resolve metadata for the current stage without consulting a fixed slot."""

    resolved_mode = BinanceMetadataMode(mode)
    if resolved_mode is BinanceMetadataMode.HISTORICAL_SIGNED:
        if verified_history is None:
            raise ValueError("historical_signed metadata requires verified history")
        resolution = resolution_from_historical_signed(
            verified_history,
            start_time=start_time,
            end_time=end_time,
        )
    elif resolved_mode is BinanceMetadataMode.FROZEN_SNAPSHOT:
        resolution = resolve_frozen_snapshot(
            transport=transport,
            market=BinanceMarket.USDS_M,
            symbols=request.symbols,
            start_time=start_time,
            end_time=end_time,
        )
    else:
        if conservative_static_path is None:
            raise ValueError(
                "conservative_static metadata requires conservative_static_path"
            )
        resolution = resolve_conservative_static(
            path=conservative_static_path,
            symbols=request.symbols,
            start_time=start_time,
            end_time=end_time,
        )
    return _validate_resolution_scope(request, resolution)


def _resolve_metadata_for_request(
    request: SymbolTripletStageRequest,
    *,
    mode: BinanceMetadataMode | str,
    cache_root: Path,
    start_time: datetime,
    end_time: datetime,
    conservative_static_path: Path | None,
) -> BinanceMetadataResolution:
    resolved_mode = BinanceMetadataMode(mode)
    verified_history = (
        _verified_history_from_environment()
        if resolved_mode is BinanceMetadataMode.HISTORICAL_SIGNED
        else None
    )
    transport = BinancePublicTransport(cache_root=cache_root)
    return resolve_binance_symbol_triplet_stage_metadata(
        request,
        mode=resolved_mode,
        transport=transport,
        start_time=start_time,
        end_time=end_time,
        conservative_static_path=conservative_static_path,
        verified_history=verified_history,
    )


def write_or_validate_binance_metadata_resolution(
    root: str | Path,
    resolution: BinanceMetadataResolution,
) -> Path:
    """Write metadata evidence once or require exact byte-for-byte reuse."""

    resolved_root = Path(root)
    report_path = resolved_root / "exchange-info.json"
    raw_path = resolved_root / "exchange-info.raw.json"
    expected_report = canonical_json_bytes(resolution.report_payload()) + b"\n"
    expected_raw = resolution.raw_payload

    if not report_path.exists() and not raw_path.exists():
        resolution.write_artifacts(resolved_root)
        return report_path
    if not report_path.is_file() or report_path.read_bytes() != expected_report:
        raise ValueError("Binance metadata evidence report differs from the stage binding")
    if expected_raw is None:
        if raw_path.exists():
            raise ValueError("Binance metadata evidence has an unexpected raw payload")
    elif not raw_path.is_file() or raw_path.read_bytes() != expected_raw:
        raise ValueError("Binance metadata evidence raw payload differs from the stage binding")
    return report_path


@contextmanager
def _postgres_connection(database_url: str) -> Iterator[Any]:
    try:
        import psycopg
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("PostgreSQL symbol-triplet training requires psycopg") from error
    with psycopg.connect(database_url) as connection:
        yield connection


def _manifest_universe(
    manifest: SymbolTripletManifest | object,
    request: SymbolTripletStageRequest,
) -> tuple[str, ...]:
    raw = getattr(manifest, "universe", request.symbols)
    universe = tuple(cast(Any, raw))
    if not universe or any(not isinstance(symbol, str) or not symbol for symbol in universe):
        raise ValueError("symbol-triplet manifest universe is invalid")
    return universe


def execute_binance_symbol_triplet_stage_command(
    *,
    manifest_path: str | Path,
    plan_path: str | Path,
    cursor_path: str | Path,
    base_config_path: str | Path,
    work_root: str | Path,
    cache_root: str | Path,
    metadata_mode: BinanceMetadataMode | str,
    start_time: datetime,
    end_time: datetime,
    conservative_static_path: str | Path | None = None,
    database_url: str | None = None,
) -> object | None:
    """Execute exactly one active Plan/Cursor stage and return its result."""

    manifest = load_symbol_triplet_manifest(manifest_path)
    plan = load_symbol_triplet_training_plan(plan_path, manifest=manifest)
    request = current_binance_symbol_triplet_stage_request(
        plan,
        cursor_path,
        base_config_path,
        work_root,
    )
    if request is None:
        return None

    resolved_static_path = (
        None if conservative_static_path is None else Path(conservative_static_path)
    )
    resolution = _resolve_metadata_for_request(
        request,
        mode=metadata_mode,
        cache_root=Path(cache_root),
        start_time=start_time,
        end_time=end_time,
        conservative_static_path=resolved_static_path,
    )
    metadata_root = binance_symbol_triplet_stage_root(work_root, request) / "metadata"
    write_or_validate_binance_metadata_resolution(metadata_root, resolution)

    resolved_database_url = (database_url or os.environ.get("TRADE_RL_DATABASE_URL", "")).strip()
    if not resolved_database_url:
        raise ValueError("TRADE_RL_DATABASE_URL is required for PostgreSQL market data")
    with _postgres_connection(resolved_database_url) as connection:
        return execute_binance_symbol_triplet_postgres_stage(
            connection=connection,
            plan=cast(SymbolTripletTrainingPlan, plan),
            cursor_path=cursor_path,
            base_config_path=base_config_path,
            work_root=work_root,
            symbol_vocabulary=_manifest_universe(manifest, request),
            start_time=start_time,
            end_time=end_time,
            metadata=cast(Mapping[str, Mapping[str, object]], resolution.metadata),
            metadata_evidence_digest=resolution.evidence_digest,
            execution_rule_histories=resolution.execution_rule_histories,
        )


__all__ = [
    "execute_binance_symbol_triplet_stage_command",
    "resolve_binance_symbol_triplet_stage_metadata",
    "write_or_validate_binance_metadata_resolution",
]
