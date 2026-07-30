from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trade_rl.workflows.binance_metadata_modes import (
    BinanceMetadataMode,
    BinanceMetadataResolution,
)
from trade_rl.workflows.symbol_triplet_manifest import build_symbol_triplet_manifest
from trade_rl.workflows.symbol_triplet_stage_orchestrator import (
    build_symbol_triplet_stage_request,
)
from trade_rl.workflows.symbol_triplet_training_cursor import (
    build_symbol_triplet_training_plan,
    initial_symbol_triplet_training_cursor,
)

_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "DOTUSDT",
    "AVAXUSDT",
    "UNIUSDT",
    "TRXUSDT",
    "ETCUSDT",
)
_START = datetime(2024, 12, 1, tzinfo=UTC)
_END = datetime(2026, 7, 1, tzinfo=UTC)


def _request():
    manifest = build_symbol_triplet_manifest(_SYMBOLS, seed=31)
    plan = build_symbol_triplet_training_plan(
        manifest,
        cycles=1,
        slot_symbols=("SLOT0", "SLOT1", "SLOT2"),
    )
    request = build_symbol_triplet_stage_request(
        plan,
        initial_symbol_triplet_training_cursor(plan),
        training_seeds=(0, 1),
        previous_completion=None,
    )
    assert request is not None
    return request


def _resolution(symbols: tuple[str, ...]) -> BinanceMetadataResolution:
    metadata = {
        symbol: {
            "listed_at": "2020-01-01T00:00:00+00:00",
            "tick_size": 0.1,
            "lot_size": 0.001,
            "minimum_notional": 5.0,
        }
        for symbol in symbols
    }
    identity = {
        "schema_version": "test_metadata_v1",
        "symbols": symbols,
    }
    return BinanceMetadataResolution(
        mode=BinanceMetadataMode.FROZEN_SNAPSHOT,
        metadata=metadata,
        execution_rule_histories=None,
        identity_evidence=identity,
        evidence_digest="a" * 64,
        source_uri="test://metadata",
        raw_payload=b"{}",
    )


def test_metadata_resolution_uses_only_current_request_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.workflows.binance_symbol_triplet_stage_command as module

    request = _request()
    observed: dict[str, Any] = {}

    def resolve_frozen(**kwargs: Any) -> BinanceMetadataResolution:
        observed.update(kwargs)
        return _resolution(request.symbols)

    monkeypatch.setattr(module, "resolve_frozen_snapshot", resolve_frozen)

    result = module.resolve_binance_symbol_triplet_stage_metadata(
        request,
        mode=BinanceMetadataMode.FROZEN_SNAPSHOT,
        transport=object(),
        start_time=_START,
        end_time=_END,
    )

    assert result.metadata.keys() == set(request.symbols)
    assert observed["symbols"] == request.symbols
    assert observed["start_time"] == _START
    assert observed["end_time"] == _END


def test_completed_plan_returns_before_metadata_and_database(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import trade_rl.workflows.binance_symbol_triplet_stage_command as module

    monkeypatch.setattr(
        module, "load_symbol_disjoint_triplet_manifest", lambda _: object()
    )
    monkeypatch.setattr(
        module,
        "load_symbol_triplet_training_plan",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        module,
        "current_binance_symbol_triplet_stage_request",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "_resolve_metadata_for_request",
        lambda *_args, **_kwargs: pytest.fail(
            "completed plan must not resolve metadata"
        ),
    )
    monkeypatch.setattr(
        module,
        "_postgres_connection",
        lambda *_args, **_kwargs: pytest.fail("completed plan must not touch database"),
    )

    assert (
        module.execute_binance_symbol_triplet_stage_command(
            manifest_path=tmp_path / "manifest.json",
            plan_path=tmp_path / "plan.json",
            cursor_path=tmp_path / "cursor.json",
            base_config_path=tmp_path / "config.json",
            work_root=tmp_path / "work",
            cache_root=tmp_path / "cache",
            metadata_mode=BinanceMetadataMode.HISTORICAL_SIGNED,
            start_time=_START,
            end_time=_END,
        )
        is None
    )


def test_active_command_persists_metadata_and_executes_postgres_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import trade_rl.workflows.binance_symbol_triplet_stage_command as module

    request = _request()
    resolution = _resolution(request.symbols)
    sentinel = object()
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        module, "load_symbol_disjoint_triplet_manifest", lambda _: object()
    )
    monkeypatch.setattr(
        module,
        "load_symbol_triplet_training_plan",
        lambda *_args, **_kwargs: "plan",
    )
    monkeypatch.setattr(
        module,
        "current_binance_symbol_triplet_stage_request",
        lambda *_args, **_kwargs: request,
    )
    monkeypatch.setattr(
        module,
        "_resolve_metadata_for_request",
        lambda *_args, **_kwargs: resolution,
    )

    @contextmanager
    def connection(_database_url: str):
        yield "connection"

    monkeypatch.setattr(module, "_postgres_connection", connection)

    def execute(**kwargs: Any) -> object:
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        module, "execute_binance_symbol_triplet_postgres_stage", execute
    )
    monkeypatch.setenv("TRADE_RL_DATABASE_URL", "postgresql://example.invalid/trade_rl")

    result = module.execute_binance_symbol_triplet_stage_command(
        manifest_path=tmp_path / "manifest.json",
        plan_path=tmp_path / "plan.json",
        cursor_path=tmp_path / "cursor.json",
        base_config_path=tmp_path / "config.json",
        work_root=tmp_path / "work",
        cache_root=tmp_path / "cache",
        metadata_mode=BinanceMetadataMode.FROZEN_SNAPSHOT,
        start_time=_START,
        end_time=_END,
    )

    assert result is sentinel
    assert observed["connection"] == "connection"
    assert observed["plan"] == "plan"
    assert observed["metadata"] == resolution.metadata
    assert observed["metadata_evidence_digest"] == resolution.evidence_digest
    metadata_root = (
        module.binance_symbol_triplet_stage_root(tmp_path / "work", request)
        / "metadata"
    )
    assert (metadata_root / "exchange-info.json").is_file()
    assert (metadata_root / "exchange-info.raw.json").read_bytes() == b"{}"


def test_metadata_artifact_reuse_rejects_drift(tmp_path: Path) -> None:
    import trade_rl.workflows.binance_symbol_triplet_stage_command as module

    resolution = _resolution(_request().symbols)
    root = tmp_path / "metadata"

    module.write_or_validate_binance_metadata_resolution(root, resolution)
    module.write_or_validate_binance_metadata_resolution(root, resolution)
    (root / "exchange-info.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="metadata evidence"):
        module.write_or_validate_binance_metadata_resolution(root, resolution)


def test_missing_database_url_fails_after_metadata_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import trade_rl.workflows.binance_symbol_triplet_stage_command as module

    request = _request()
    monkeypatch.setattr(
        module, "load_symbol_disjoint_triplet_manifest", lambda _: object()
    )
    monkeypatch.setattr(
        module,
        "load_symbol_triplet_training_plan",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        module,
        "current_binance_symbol_triplet_stage_request",
        lambda *_args, **_kwargs: request,
    )
    monkeypatch.setattr(
        module,
        "_resolve_metadata_for_request",
        lambda *_args, **_kwargs: _resolution(request.symbols),
    )
    monkeypatch.delenv("TRADE_RL_DATABASE_URL", raising=False)

    with pytest.raises(ValueError, match="TRADE_RL_DATABASE_URL"):
        module.execute_binance_symbol_triplet_stage_command(
            manifest_path=tmp_path / "manifest.json",
            plan_path=tmp_path / "plan.json",
            cursor_path=tmp_path / "cursor.json",
            base_config_path=tmp_path / "config.json",
            work_root=tmp_path / "work",
            cache_root=tmp_path / "cache",
            metadata_mode=BinanceMetadataMode.FROZEN_SNAPSHOT,
            start_time=_START,
            end_time=_END,
        )


def test_stage_a_command_rejects_legacy_all_symbol_triplet_manifest(
    tmp_path: Path,
) -> None:
    import trade_rl.workflows.binance_symbol_triplet_stage_command as module
    from trade_rl.workflows.symbol_triplet_manifest import (
        write_symbol_triplet_manifest,
    )

    legacy_path = write_symbol_triplet_manifest(
        tmp_path / "legacy-manifest.json",
        build_symbol_triplet_manifest(_SYMBOLS, seed=31),
    )
    with pytest.raises(ValueError, match="field closure|symbol-disjoint"):
        module.execute_binance_symbol_triplet_stage_command(
            manifest_path=legacy_path,
            plan_path=tmp_path / "plan.json",
            cursor_path=tmp_path / "cursor.json",
            base_config_path=tmp_path / "config.json",
            work_root=tmp_path / "work",
            cache_root=tmp_path / "cache",
            metadata_mode=BinanceMetadataMode.HISTORICAL_SIGNED,
            start_time=_START,
            end_time=_END,
        )
