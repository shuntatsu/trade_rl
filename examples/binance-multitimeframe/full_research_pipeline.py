"""Typed helpers for the maintained Binance multi-timeframe research pipeline."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import publish_market_dataset_artifact
from trade_rl.data.contracts import InstrumentExecutionRule
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.confirmation import load_confirmation_evidence
from trade_rl.evaluation.research_gate import (
    ResearchEvidenceRequirements,
    ResearchReturnGate,
    evaluate_research_return_gate,
    paired_block_bootstrap_excess_lower_bound,
)
from trade_rl.integrations.binance import (
    BinanceExchangeInfoSnapshot,
    BinanceMarket,
    BinancePublicTransport,
    BinanceTransportMode,
    binance_multitimeframe_feature_specs,
    build_binance_market_dataset,
)
from trade_rl.integrations.postgres_market_dataset import (
    build_postgres_market_dataset,
)
from trade_rl.release.asymmetric import (
    PublicVerificationKey,
    load_public_verification_keys,
)
from trade_rl.rl.checkpointing import checkpoint_manifests
from trade_rl.workflows.binance_metadata_modes import (
    BinanceMetadataMode,
    BinanceMetadataResolution,
    VerifiedBinanceRuleHistory,
    load_verified_binance_rule_history,
    resolution_from_historical_signed,
    resolve_conservative_static,
    resolve_frozen_snapshot,
)
from trade_rl.workflows.symbol_disjoint_manifest import (
    build_symbol_disjoint_manifest,
    write_symbol_disjoint_manifest,
)
from trade_rl.workflows.symbol_disjoint_triplet_manifest import (
    build_symbol_disjoint_triplet_manifest,
    write_symbol_disjoint_triplet_manifest,
)
from trade_rl.workflows.symbol_triplet_manifest import SymbolTripletSlot

_SYMBOL_POOL = (
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
_SLOT_SYMBOLS = ("SLOT0", "SLOT1", "SLOT2")
_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT")
_ACTIVE_SYMBOL_TRIPLET: dict[str, object] | None = None
_NATIVE_TIMEFRAMES = ("15m", "1h", "4h", "1d")
_FEATURE_TIMEFRAMES = ("1h", "4h", "1d")
_START = "2021-01-01T00:00:00Z"
_END = "2026-07-01T00:00:00Z"
_EXPECTED_15M_BARS = 192_672
_EXPECTED_POLICY_OBSERVATIONS = 217_886
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_TRAIN_RUN_COMMAND = ("train", "run")
_WALK_FORWARD_RUN_COMMAND = ("walk-forward", "run")


class _PersistentFrozenSnapshotTransport:
    """Persist one audited exchange-info snapshot across supervised retries."""

    def __init__(self, delegate: BinancePublicTransport, root: Path) -> None:
        self._delegate = delegate
        self._root = root

    def load_exchange_information_snapshot(
        self,
        *,
        market: BinanceMarket | str,
        mode: BinanceTransportMode | str = BinanceTransportMode.AUTO,
    ) -> BinanceExchangeInfoSnapshot:
        resolved_market = BinanceMarket(market)
        raw_path = self._root / "exchange-info.raw.json"
        manifest_path = self._root / "manifest.json"
        if raw_path.exists() != manifest_path.exists():
            raise RuntimeError("frozen metadata cache is incomplete")
        if raw_path.exists():
            raw = raw_path.read_bytes()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("frozen metadata cache manifest must be an object")
            digest = sha256(raw).hexdigest()
            if manifest.get("schema_version") != "frozen_metadata_cache_v1":
                raise ValueError("frozen metadata cache schema mismatch")
            if manifest.get("market") != resolved_market.value:
                raise ValueError("frozen metadata cache market mismatch")
            if manifest.get("raw_payload_sha256") != digest:
                raise ValueError("frozen metadata cache digest mismatch")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("cached exchange information must be an object")
            return BinanceExchangeInfoSnapshot(
                payload=payload,
                raw_payload=raw,
                source_uri=str(manifest["source_uri"]),
                retrieved_at=_parse_utc(str(manifest["retrieved_at"])),
                raw_payload_sha256=digest,
            )

        snapshot = self._delegate.load_exchange_information_snapshot(
            market=resolved_market,
            mode=mode,
        )
        self._root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "market": resolved_market.value,
            "raw_payload_sha256": snapshot.raw_payload_sha256,
            "retrieved_at": snapshot.retrieved_at.isoformat(),
            "schema_version": "frozen_metadata_cache_v1",
            "source_uri": snapshot.source_uri,
        }
        raw_temporary = raw_path.with_suffix(f".raw.{os.getpid()}.tmp")
        manifest_temporary = manifest_path.with_suffix(f".{os.getpid()}.tmp")
        raw_temporary.write_bytes(snapshot.raw_payload)
        manifest_temporary.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        raw_temporary.replace(raw_path)
        manifest_temporary.replace(manifest_path)
        return snapshot


def _activate_symbol_triplet(
    *, work_root: Path, seed: int, train_slot: int
) -> SymbolTripletSlot:
    """Bind one complete Generation to a stable generic three-slot mapping."""

    global _ACTIVE_SYMBOL_TRIPLET, _SYMBOLS
    source_manifest = build_symbol_disjoint_manifest(
        _SYMBOL_POOL,
        seed=seed,
        validation_count=3,
        test_count=3,
    )
    manifest = build_symbol_disjoint_triplet_manifest(source_manifest)
    train_slots = manifest.slots_for("train")
    if (
        isinstance(train_slot, bool)
        or not isinstance(train_slot, int)
        or not 0 <= train_slot < len(train_slots)
    ):
        raise ValueError("triplet train slot is outside the balanced train split")
    selected = train_slots[train_slot]
    _SYMBOLS = selected.symbols
    _ACTIVE_SYMBOL_TRIPLET = {
        "manifest_digest": manifest.digest,
        "schedule_identity": manifest.schedule_identity,
        "source_manifest_digest": source_manifest.digest,
        "selected": selected.digest_payload(),
    }
    write_symbol_disjoint_manifest(work_root / "symbol-disjoint.json", source_manifest)
    write_symbol_disjoint_triplet_manifest(work_root / "symbol-triplets.json", manifest)
    _write_json(
        work_root / "selected-symbol-triplet.json",
        {
            "manifest_digest": manifest.digest,
            "schedule_identity": manifest.schedule_identity,
            "source_manifest_digest": source_manifest.digest,
            "schema_version": "selected_symbol_triplet_v1",
            "selected": selected.digest_payload(),
            "slot_symbols": _SLOT_SYMBOLS,
            "symbol_vocabulary": _SYMBOL_POOL,
        },
    )
    return selected


def _policy_observation_count(dataset: MarketDataset) -> int:
    from trade_rl.rl.observations import ObservationBuilder
    from trade_rl.rl.sequence_observations import SequenceObservationBuilder

    n_symbols = dataset.n_symbols
    flat = (
        ObservationBuilder(action_size=3, n_factors=0, finite_horizon=True)
        .layout(dataset)
        .size
    )
    sequence = SequenceObservationBuilder().schema_payload(dataset)
    raw_windows = sequence.get("windows")
    if not isinstance(raw_windows, (tuple, list)):
        raise RuntimeError("sequence schema windows must be ordered")
    return flat + sum(
        n_symbols
        * int(dict(window)["length"])
        * len(tuple(dict(window)["feature_names"]))
        * 3
        for window in raw_windows
    )


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("time range must include a timezone")
    return parsed.astimezone(UTC)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return dict(payload)


def _training_policy_digest(payload: object) -> str:
    if not isinstance(payload, dict):
        raise ValueError("training result must be a JSON object")
    value = payload.get("policy_digest")
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("training result policy_digest is missing or invalid")
    return value


def _align_workflow_to_full_dataset(workflow: dict[str, Any], *, n_bars: int) -> None:
    """Use the full immutable timeline while retaining the requested fold count."""

    required = (
        "checkpoint_bars",
        "max_folds",
        "purge_bars",
        "selection_bars",
        "step_bars",
        "test_bars",
    )
    values: dict[str, int] = {}
    for field in required:
        value = workflow.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"workflow.{field} must be an integer")
        values[field] = value
    if n_bars <= 0 or any(value <= 0 for value in values.values()):
        raise ValueError("dataset bars and full-range workflow values must be positive")
    reserved = (
        (values["max_folds"] - 1) * values["step_bars"]
        + values["checkpoint_bars"]
        + values["selection_bars"]
        + values["test_bars"]
        + 3 * values["purge_bars"]
    )
    train_bars = n_bars - reserved
    if train_bars <= 0:
        raise ValueError("dataset is too short for the requested full-range workflow")
    workflow["train_bars"] = train_bars


def _packaged_git_provenance() -> tuple[str, bool]:
    commit = os.environ.get("TRADE_RL_GIT_COMMIT", "")
    if not _GIT_COMMIT_PATTERN.fullmatch(commit):
        raise ValueError(
            "TRADE_RL_GIT_COMMIT must be a 40-character lowercase Git commit"
        )
    dirty = os.environ.get("TRADE_RL_GIT_DIRTY")
    if dirty not in {"true", "false"}:
        raise ValueError("TRADE_RL_GIT_DIRTY must be exactly true or false")
    return commit, dirty == "true"


def _prepare_run_roots(*, work_root: Path, cache_root: Path) -> tuple[Path, Path]:
    resolved_work_root = work_root.resolve()
    resolved_cache_root = cache_root.resolve()
    if resolved_work_root.exists():
        raise FileExistsError(
            f"run generation already exists; choose a new --work-root: "
            f"{resolved_work_root}"
        )
    if (
        resolved_cache_root == resolved_work_root
        or resolved_work_root in resolved_cache_root.parents
        or resolved_cache_root in resolved_work_root.parents
    ):
        raise ValueError(
            f"cache root must be outside the run generation: {resolved_cache_root}"
        )
    try:
        resolved_work_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"run generation already exists; choose a new --work-root: "
            f"{resolved_work_root}"
        ) from error
    resolved_cache_root.mkdir(parents=True, exist_ok=True)
    return resolved_work_root, resolved_cache_root


def _write_run_config(
    *,
    template_path: Path,
    output_path: Path,
) -> Path:
    payload = _load_json(template_path)
    git_commit, git_dirty = _packaged_git_provenance()
    payload["git_commit"] = git_commit
    payload["git_dirty"] = git_dirty
    for candidate in payload.get("candidates", ()):
        if not isinstance(candidate, dict):
            continue
        run = candidate.get("run")
        if not isinstance(run, dict):
            continue
        run["git_commit"] = git_commit
        run["git_dirty"] = git_dirty
    _write_json(output_path, payload)
    return output_path


def _run_cli(arguments: list[str], *, root: Path, log_path: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "trade_rl.cli.app", *arguments]
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "command: "
        + repr(command)
        + "\n\nstdout:\n"
        + completed.stdout
        + "\n\nstderr:\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed; see {log_path}: {command!r}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"command produced no JSON output: {command!r}")
    payload = json.loads(lines[-1])
    if not isinstance(payload, dict):
        raise RuntimeError(f"command JSON result must be an object: {command!r}")
    return dict(payload)


def _metadata_public_keys_path() -> Path:
    raw = os.environ.get("TRADE_RL_METADATA_PUBLIC_KEYS", "")
    if not raw:
        raise RuntimeError(
            "TRADE_RL_METADATA_PUBLIC_KEYS is required for signed history verification"
        )
    path = Path(raw)
    if not path.is_file():
        raise FileNotFoundError(f"metadata public-key store is missing: {path}")
    return path


def _load_rule_history(
    *,
    trusted_now: datetime | None = None,
) -> VerifiedBinanceRuleHistory:
    """Load and verify the complete signed instrument rule history document."""

    raw_path = os.environ.get("TRADE_RL_BINANCE_RULE_HISTORY", "")
    if not raw_path:
        raise RuntimeError(
            "TRADE_RL_BINANCE_RULE_HISTORY is required for strict point-in-time research"
        )
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"signed Binance rule history is missing: {path}")
    document = _load_json(path)
    keys: dict[str, PublicVerificationKey] = load_public_verification_keys(
        _metadata_public_keys_path()
    )
    return load_verified_binance_rule_history(
        document,
        trusted_keys=keys,
        trusted_now=trusted_now or datetime.now(UTC),
    )


def _resolve_metadata(
    *,
    mode: BinanceMetadataMode,
    transport: BinancePublicTransport,
    conservative_static_path: Path | None,
) -> BinanceMetadataResolution:
    start_time = _parse_utc(_START)
    end_time = _parse_utc(_END)
    if mode is BinanceMetadataMode.HISTORICAL_SIGNED:
        verified = _load_rule_history()
        return resolution_from_historical_signed(
            verified_history=verified,
            start_time=start_time,
            end_time=end_time,
        )
    if mode is BinanceMetadataMode.FROZEN_SNAPSHOT:
        snapshot_transport: Any = transport
        cache_root = os.environ.get("TRADE_RL_FROZEN_METADATA_CACHE_ROOT", "").strip()
        if cache_root:
            snapshot_transport = _PersistentFrozenSnapshotTransport(
                transport,
                Path(cache_root),
            )
        return resolve_frozen_snapshot(
            transport=snapshot_transport,
            market=BinanceMarket.USDS_M,
            symbols=_SYMBOLS,
            start_time=start_time,
            end_time=end_time,
        )
    if conservative_static_path is None:
        raise ValueError(
            "--conservative-static-path is required for conservative_static mode"
        )
    return resolve_conservative_static(
        path=conservative_static_path,
        symbols=_SYMBOLS,
        start_time=start_time,
        end_time=end_time,
    )


def validate_maintained_dataset_preset(
    dataset: MarketDataset,
    *,
    use_postgres: bool,
) -> None:
    if dataset.n_bars != _EXPECTED_15M_BARS:
        raise RuntimeError(
            f"expected {_EXPECTED_15M_BARS:,} 15-minute bars, observed {dataset.n_bars}"
        )
    expected_dataset_symbols = _SLOT_SYMBOLS if use_postgres else _SYMBOLS
    if dataset.symbols != expected_dataset_symbols:
        raise RuntimeError(f"unexpected symbol order: {dataset.symbols}")
    expected_features = tuple(
        spec.name
        for spec in binance_multitimeframe_feature_specs(
            base_timeframe="15m",
            feature_timeframes=_FEATURE_TIMEFRAMES,
        )
    )
    if len(expected_features) != 226:
        raise RuntimeError(
            f"extended feature contract must contain 226 features, got {len(expected_features)}"
        )
    expected_dataset_features = expected_features
    if dataset.feature_names != expected_dataset_features:
        raise RuntimeError(f"unexpected feature contract: {dataset.feature_names}")


def _build_dataset(
    *,
    output: Path,
    transport: BinancePublicTransport,
    metadata: Mapping[str, Mapping[str, str | float]],
    execution_rule_histories: Mapping[str, tuple[InstrumentExecutionRule, ...]] | None,
    metadata_evidence: Mapping[str, object],
    metadata_mode: BinanceMetadataMode,
    metadata_evidence_digest: str,
) -> dict[str, object]:
    use_postgres = os.environ.get("TRADE_RL_POSTGRES_MARKET_DATA", "").lower() == "true"
    if use_postgres:
        database_url = os.environ.get("TRADE_RL_DATABASE_URL", "").strip()
        if not database_url:
            raise ValueError(
                "TRADE_RL_DATABASE_URL is required for PostgreSQL market data"
            )
        try:
            import psycopg
        except ImportError as error:  # pragma: no cover - optional dependency boundary
            raise RuntimeError("PostgreSQL market data requires psycopg") from error
        with psycopg.connect(database_url) as connection:
            dataset = build_postgres_market_dataset(
                connection,
                symbols=_SYMBOLS,
                symbol_vocabulary=_SYMBOL_POOL,
                slot_symbols=_SLOT_SYMBOLS,
                start_time=_parse_utc(_START),
                end_time=_parse_utc(_END),
                metadata=metadata,
                metadata_evidence_digest=metadata_evidence_digest,
                execution_rule_histories=execution_rule_histories,
                symbol_triplet_provenance=_ACTIVE_SYMBOL_TRIPLET,
            )
        feature_timeframes = _NATIVE_TIMEFRAMES
        sources_used = (
            "postgres:market_raw.binance_usds_m_klines_202101_202606",
            "postgres:market_raw.binance_usds_m_funding_202101_202606",
            "postgres:market_raw.binance_usds_m_indicator_artifacts_202101_202606",
        )
    else:
        result = build_binance_market_dataset(
            market=BinanceMarket.USDS_M,
            symbols=_SYMBOLS,
            interval="15m",
            feature_timeframes=_FEATURE_TIMEFRAMES,
            start_time=_parse_utc(_START),
            end_time=_parse_utc(_END),
            transport_mode=BinanceTransportMode.VISION,
            transport=transport,
            tick_sizes=tuple(
                float(metadata[symbol]["tick_size"]) for symbol in _SYMBOLS
            ),
            lot_sizes=tuple(float(metadata[symbol]["lot_size"]) for symbol in _SYMBOLS),
            minimum_notionals=tuple(
                float(metadata[symbol]["minimum_notional"]) for symbol in _SYMBOLS
            ),
            listed_ats=tuple(
                _parse_utc(str(metadata[symbol]["listed_at"])) for symbol in _SYMBOLS
            ),
            execution_rule_histories=execution_rule_histories,
            metadata_evidence=metadata_evidence,
        )
        dataset = result.dataset
        feature_timeframes = result.feature_timeframes
        sources_used = result.sources_used
    validate_maintained_dataset_preset(dataset, use_postgres=use_postgres)
    published = publish_market_dataset_artifact(output, dataset)
    return {
        "artifact_digest": published.artifact_digest,
        "dataset_id": dataset.dataset_id,
        "feature_names": list(dataset.feature_names),
        "metadata_mode": metadata_mode.value,
        "metadata_evidence_digest": metadata_evidence_digest,
        "feature_timeframes": list(feature_timeframes),
        "n_bars": dataset.n_bars,
        "n_features": dataset.n_features,
        "raw_feature_count": dataset.n_features,
        "policy_observation_count": _policy_observation_count(dataset),
        "n_symbols": dataset.n_symbols,
        "selected_symbols": list(_SYMBOLS),
        "slot_symbols": list(dataset.symbols),
        "sources_used": list(sources_used),
        "symbols": list(dataset.symbols),
    }


def _require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"required artifact file is missing or empty: {path}")


def _verify_training(path: Path) -> None:
    for relative in ("run.json", "ensemble.json", "environment.json"):
        _require_file(path / relative)
    ensemble = _load_json(path / "ensemble.json")
    expected_members = ensemble.get("expected_members")
    if (
        isinstance(expected_members, bool)
        or not isinstance(expected_members, int)
        or expected_members <= 0
    ):
        raise RuntimeError("training ensemble expected_members is invalid")
    members = ensemble.get("members")
    if not isinstance(members, list) or len(members) != expected_members:
        raise RuntimeError("training ensemble member evidence is incomplete")
    for index in range(expected_members):
        member = path / f"members/member-{index:03d}"
        _require_file(member / "policy.zip")
        checkpoints = checkpoint_manifests(member / "checkpoints")
        if not checkpoints:
            raise RuntimeError(f"member {index} has no retained checkpoints")


def _independent_fold_maximum_drawdown(folds: object) -> float | None:
    if not isinstance(folds, list) or not folds:
        return None
    maximum = 0.0
    for fold in folds:
        if not isinstance(fold, dict):
            return None
        selected_returns = fold.get("selected_returns")
        if not isinstance(selected_returns, (list, tuple)) or not selected_returns:
            return None
        wealth = 1.0
        peak = 1.0
        for value in selected_returns:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            try:
                resolved = float(value)
            except (OverflowError, TypeError, ValueError):
                return None
            if not np.isfinite(resolved) or resolved < -1.0:
                return None
            wealth *= 1.0 + resolved
            if not np.isfinite(wealth):
                return None
            peak = max(peak, wealth)
            if not np.isfinite(peak):
                return None
            drawdown = 1.0 - wealth / peak
            if not np.isfinite(drawdown):
                return None
            maximum = max(maximum, drawdown)
    return maximum


def _summary_mean(payload: dict[str, Any], name: str) -> object:
    summary = payload.get(name)
    if not isinstance(summary, dict):
        return None
    return summary.get("mean_fold_return")


def _selected_fold_policy_digests(folds: object) -> object:
    if not isinstance(folds, list) or not folds:
        return None
    identities: list[object] = []
    for fold in folds:
        if not isinstance(fold, dict):
            return None
        identities.append(fold.get("selected_policy_digest"))
    return tuple(identities)


def _maximum_fold_metric(folds: object, name: str) -> float | None:
    if not isinstance(folds, list) or not folds:
        return None
    values: list[float] = []
    for fold in folds:
        if not isinstance(fold, dict):
            return None
        raw = fold.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        value = float(raw)
        if not np.isfinite(value) or value < 0.0:
            return None
        values.append(value)
    return max(values)


def _selection_stability_passed(folds: object) -> bool:
    if not isinstance(folds, list) or not folds:
        return False
    selected_configurations: list[str] = []
    selected_seed_recipes: list[tuple[int, ...]] = []
    for fold in folds:
        if not isinstance(fold, dict):
            return False
        selected = fold.get("selected_configuration")
        aggregates = fold.get("candidate_aggregates")
        raw_member_seeds = fold.get("selected_member_seeds")
        if not isinstance(selected, str) or selected == "baseline":
            return False
        if (
            not isinstance(raw_member_seeds, (list, tuple))
            or len(raw_member_seeds) < 2
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in raw_member_seeds
            )
            or len(set(raw_member_seeds)) != len(raw_member_seeds)
        ):
            return False
        selected_configurations.append(selected)
        selected_seed_recipes.append(tuple(int(seed) for seed in raw_member_seeds))
        if not isinstance(aggregates, (list, tuple)):
            return False
        matched = [
            item
            for item in aggregates
            if isinstance(item, dict) and item.get("configuration") == selected
        ]
        if len(matched) != 1 or matched[0].get("eligible") is not True:
            return False
    return (
        len(set(selected_configurations)) == 1 and len(set(selected_seed_recipes)) == 1
    )


def _fold_daily_returns(
    folds: object,
    *,
    field: str,
) -> tuple[float, ...] | None:
    if not isinstance(folds, list) or not folds:
        return None
    periods_per_day = 96
    daily: list[float] = []
    for fold in folds:
        if not isinstance(fold, dict):
            return None
        raw_returns = fold.get(field)
        if not isinstance(raw_returns, (list, tuple)):
            return None
        values: list[float] = []
        for raw in raw_returns:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                return None
            value = float(raw)
            if not np.isfinite(value) or value < -1.0:
                return None
            values.append(value)
        if len(values) % periods_per_day != 0:
            return None
        for offset in range(0, len(values), periods_per_day):
            wealth = 1.0
            for value in values[offset : offset + periods_per_day]:
                wealth *= 1.0 + value
            if not np.isfinite(wealth):
                return None
            daily.append(wealth - 1.0)
    return tuple(daily)


def _confirmation_evidence(
    path: Path | None,
    *,
    expected_policy_digest: str | None,
    expected_dataset_id: str | None,
    expected_environment_digest: str | None,
    expected_training_run_digest: str | None,
    expected_required_after: datetime | None,
    trusted_now: datetime | None,
    trusted_confirmation_keys: Mapping[str, PublicVerificationKey] | None,
) -> tuple[bool, float]:
    if path is None or not path.is_file():
        return False, 0.0
    if expected_required_after is None or trusted_now is None:
        return False, 0.0
    try:
        evidence = load_confirmation_evidence(path)
        evidence.verify(
            trusted_confirmation_keys or {},
            expected_required_after=expected_required_after,
            trusted_now=trusted_now,
        )
    except (OSError, ValueError):
        return False, 0.0
    passed = (
        evidence.total_return > 0.0
        and 0.0 <= evidence.maximum_drawdown <= 0.20
        and (
            expected_policy_digest is None
            or evidence.policy_digest == expected_policy_digest
        )
        and (expected_dataset_id is None or evidence.dataset_id == expected_dataset_id)
        and (
            expected_environment_digest is None
            or evidence.environment_digest == expected_environment_digest
        )
        and (
            expected_training_run_digest is None
            or evidence.training_run_digest == expected_training_run_digest
        )
    )
    return passed, evidence.days


def _evaluate_walk_forward_research_gate(
    path: Path,
    *,
    strict: bool = False,
    require_confirmation: bool = False,
    confirmation_path: Path | None = None,
    expected_policy_digest: str | None = None,
    expected_dataset_id: str | None = None,
    expected_environment_digest: str | None = None,
    expected_training_run_digest: str | None = None,
    expected_required_after: datetime | None = None,
    trusted_now: datetime | None = None,
    trusted_confirmation_keys: Mapping[str, PublicVerificationKey] | None = None,
) -> ResearchReturnGate:
    try:
        payload = _load_json(path / "walk-forward.json")
    except (OSError, ValueError):
        payload = {}
    folds = payload.get("folds")
    requirements = None
    fold_count = None
    oos_days = None
    bootstrap_lower_bound = None
    confirmation_passed = None
    confirmation_days = None
    if strict:
        requirements = ResearchEvidenceRequirements(
            required_fold_count=6,
            minimum_oos_days=180.0,
            require_positive_bootstrap_lower_bound=True,
            require_confirmation=require_confirmation,
            minimum_confirmation_days=30.0,
            minimum_baseline_uplift=0.005,
        )
        fold_count = len(folds) if isinstance(folds, list) else None
        selected_daily_returns = _fold_daily_returns(folds, field="selected_returns")
        baseline_daily_returns = _fold_daily_returns(folds, field="baseline_returns")
        if selected_daily_returns is not None:
            oos_days = float(len(selected_daily_returns))
        if (
            selected_daily_returns is not None
            and baseline_daily_returns is not None
            and len(selected_daily_returns) >= 2
        ):
            bootstrap_lower_bound = paired_block_bootstrap_excess_lower_bound(
                selected_daily_returns,
                baseline_daily_returns,
                samples=2_000,
                block_size=5,
                seed=0,
            )
        if require_confirmation:
            confirmation_passed, confirmation_days = _confirmation_evidence(
                confirmation_path,
                expected_policy_digest=expected_policy_digest,
                expected_dataset_id=expected_dataset_id,
                expected_environment_digest=expected_environment_digest,
                expected_training_run_digest=expected_training_run_digest,
                expected_required_after=expected_required_after,
                trusted_now=trusted_now,
                trusted_confirmation_keys=trusted_confirmation_keys,
            )
    return evaluate_research_return_gate(
        selected_mean_return=_summary_mean(
            payload,
            "selected_independent_summary",
        ),
        baseline_mean_return=_summary_mean(
            payload,
            "baseline_independent_summary",
        ),
        maximum_fold_drawdown=_independent_fold_maximum_drawdown(folds),
        selected_policy_digests=_selected_fold_policy_digests(folds),
        maximum_turnover_per_day=_maximum_fold_metric(
            folds, "selected_turnover_per_day"
        ),
        maximum_cost_fraction=_maximum_fold_metric(folds, "selected_cost_fraction"),
        selection_stability_passed=_selection_stability_passed(folds),
        sealed_fold_count=fold_count,
        oos_days=oos_days,
        bootstrap_lower_bound=bootstrap_lower_bound,
        confirmation_passed=confirmation_passed,
        confirmation_days=confirmation_days,
        requirements=requirements,
    )


def _execution_sensitivity_gate(path: Path) -> tuple[bool, dict[str, object]]:
    try:
        walk_forward = _load_json(path / "walk-forward.json")
    except (OSError, ValueError):
        return False, {
            "passed": False,
            "reason": "walk-forward evidence is missing or invalid",
        }
    declared_digest = walk_forward.get("execution_sensitivity_digest")
    if declared_digest is None:
        return True, {
            "passed": True,
            "required": False,
            "reason": "execution sensitivity is not configured",
        }
    if not isinstance(declared_digest, str):
        return False, {
            "passed": False,
            "reason": "walk-forward sensitivity digest is invalid",
        }
    try:
        payload = _load_json(path / "execution-sensitivity.json")
    except (OSError, ValueError):
        return False, {
            "passed": False,
            "reason": "execution sensitivity artifact is missing or invalid",
        }
    artifact_digest = payload.get("artifact_digest")
    if not isinstance(artifact_digest, str):
        return False, {
            "passed": False,
            "reason": "execution sensitivity artifact digest is missing",
        }
    digest_payload = dict(payload)
    digest_payload.pop("artifact_digest", None)
    if content_digest(digest_payload) != artifact_digest:
        return False, {
            "passed": False,
            "reason": "execution sensitivity artifact digest mismatch",
        }
    if artifact_digest != declared_digest:
        return False, {
            "passed": False,
            "reason": "walk-forward sensitivity digest binding mismatch",
        }
    for field in ("dataset_id", "experiment_plan_digest"):
        expected = walk_forward.get(field)
        observed = payload.get(field)
        if not isinstance(expected, str) or observed != expected:
            return False, {
                "passed": False,
                "reason": f"execution sensitivity {field} binding mismatch",
            }
    if payload.get("schema_version") != "execution_sensitivity_v1":
        return False, {
            "passed": False,
            "reason": "execution sensitivity schema is invalid",
        }
    gate = payload.get("gate")
    if not isinstance(gate, dict) or not isinstance(gate.get("passed"), bool):
        return False, {
            "passed": False,
            "reason": "execution sensitivity gate is missing",
        }
    return bool(gate["passed"]), {**dict(gate), "required": True}


def _selected_walk_forward_recipe(
    walk_forward_path: Path,
    walk_forward_config_path: Path,
    output_path: Path,
) -> tuple[str, tuple[int, ...], Path]:
    evidence = _load_json(walk_forward_path / "walk-forward.json")
    folds = evidence.get("folds")
    if not isinstance(folds, list) or not folds:
        raise RuntimeError("walk-forward evidence has no folds")
    selected = tuple(
        fold.get("selected_configuration") for fold in folds if isinstance(fold, dict)
    )

    if len(selected) != len(folds) or any(
        not isinstance(name, str) or not name for name in selected
    ):
        raise RuntimeError("walk-forward selected configuration evidence is invalid")
    if len(set(selected)) != 1:
        raise RuntimeError(
            "walk-forward folds did not agree on one final training recipe"
        )
    selected_name = str(selected[0])
    if selected_name == "baseline":
        raise RuntimeError(
            "walk-forward selected baseline; final RL training is blocked"
        )
    config = _load_json(walk_forward_config_path)
    candidates = config.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("walk-forward config candidates are invalid")
    matches = [
        item
        for item in candidates
        if isinstance(item, dict) and item.get("name") == selected_name
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("run"), dict):
        raise RuntimeError("selected walk-forward recipe is missing from config")
    selected_run = dict(matches[0]["run"])
    training = selected_run.get("training")
    if not isinstance(training, dict):
        raise RuntimeError("selected training recipe has no training object")
    training = dict(training)
    raw_seeds = training.get("seeds")
    if (
        not isinstance(raw_seeds, list)
        or len(raw_seeds) < 2
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in raw_seeds
        )
    ):
        raise RuntimeError("selected training recipe requires multiple fixed seeds")
    seeds = tuple(int(seed) for seed in raw_seeds)
    selected_seed_recipes: list[tuple[int, ...]] = []
    for fold in folds:
        if not isinstance(fold, dict):
            raise RuntimeError("walk-forward fold evidence is invalid")
        raw_members = fold.get("selected_member_seeds")
        if not isinstance(raw_members, (list, tuple)) or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in raw_members
        ):
            raise RuntimeError("walk-forward seed ensemble evidence is invalid")
        selected_seed_recipes.append(tuple(int(seed) for seed in raw_members))
    if len(set(selected_seed_recipes)) != 1 or selected_seed_recipes[0] != seeds:
        raise RuntimeError(
            "walk-forward folds did not agree on the configured seed ensemble"
        )
    selected_run["training"] = training
    _write_json(output_path, selected_run)
    return selected_name, seeds, output_path


def _finalize_research_run(
    *,
    work_root: Path,
    walk_forward_path: Path,
    summary: dict[str, object],
    strict: bool = False,
    require_confirmation: bool = False,
    expected_policy_digest: str | None = None,
    expected_dataset_id: str | None = None,
    expected_environment_digest: str | None = None,
    expected_training_run_digest: str | None = None,
    expected_required_after: datetime | None = None,
    trusted_now: datetime | None = None,
    trusted_confirmation_keys: Mapping[str, PublicVerificationKey] | None = None,
) -> int:
    gate = asdict(
        _evaluate_walk_forward_research_gate(
            walk_forward_path,
            strict=strict,
            require_confirmation=require_confirmation,
            confirmation_path=work_root / "confirmation-evidence.json",
            expected_policy_digest=expected_policy_digest,
            expected_dataset_id=expected_dataset_id,
            expected_environment_digest=expected_environment_digest,
            expected_training_run_digest=expected_training_run_digest,
            expected_required_after=expected_required_after,
            trusted_now=trusted_now,
            trusted_confirmation_keys=trusted_confirmation_keys,
        )
    )
    sensitivity_passed, sensitivity_gate = _execution_sensitivity_gate(
        walk_forward_path
    )
    gate["execution_sensitivity"] = sensitivity_gate
    gate["passed"] = bool(gate["passed"]) and sensitivity_passed
    summary["execution_sensitivity"] = sensitivity_gate
    summary["research_gate"] = gate
    _write_json(work_root / "research-gate.json", gate)
    _write_json(work_root / "summary.json", summary)
    return 0 if gate["passed"] else 1


# Public aliases used by the state runner and tests.
parse_utc = _parse_utc
activate_symbol_triplet = _activate_symbol_triplet
align_workflow_to_full_dataset = _align_workflow_to_full_dataset
policy_observation_count = _policy_observation_count
write_json = _write_json
load_json = _load_json
training_policy_digest = _training_policy_digest
prepare_run_roots = _prepare_run_roots
write_run_config = _write_run_config
run_cli = _run_cli
resolve_metadata = _resolve_metadata
build_dataset = _build_dataset
require_file = _require_file
verify_training = _verify_training
evaluate_walk_forward_research_gate = _evaluate_walk_forward_research_gate
execution_sensitivity_gate = _execution_sensitivity_gate
selected_walk_forward_recipe = _selected_walk_forward_recipe
finalize_research_run = _finalize_research_run

__all__ = [
    "_END",
    "_EXPECTED_POLICY_OBSERVATIONS",
    "_FEATURE_TIMEFRAMES",
    "_NATIVE_TIMEFRAMES",
    "_SLOT_SYMBOLS",
    "_START",
    "_SYMBOL_POOL",
    "_SYMBOLS",
    "_TRAIN_RUN_COMMAND",
    "_WALK_FORWARD_RUN_COMMAND",
    "build_dataset",
    "activate_symbol_triplet",
    "evaluate_walk_forward_research_gate",
    "execution_sensitivity_gate",
    "finalize_research_run",
    "load_json",
    "parse_utc",
    "policy_observation_count",
    "prepare_run_roots",
    "require_file",
    "resolve_metadata",
    "run_cli",
    "selected_walk_forward_recipe",
    "training_policy_digest",
    "verify_training",
    "write_json",
    "write_run_config",
]
