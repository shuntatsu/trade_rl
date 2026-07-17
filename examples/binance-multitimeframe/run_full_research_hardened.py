#!/usr/bin/env python3
"""Run the maintained full pipeline with hardened trust and selection boundaries."""

from __future__ import annotations

import argparse
import json
import os
import re
import runpy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import load_market_dataset_artifact
from trade_rl.data.contracts import InstrumentExecutionRule
from trade_rl.integrations.binance import BinanceMarket, BinancePublicTransport
from trade_rl.release.signing import (
    AuthenticatedEnvelope,
    VerificationKey,
    verify_payload,
)
from trade_rl.rl.observations import ObservationBuilder
from trade_rl.rl.sequence_observations import SequenceObservationBuilder
from trade_rl.workflows.binance_metadata_modes import (
    BinanceHistoricalSignedScope,
    BinanceMetadataMode,
    BinanceMetadataResolution,
    resolution_from_historical_signed,
    resolve_conservative_static,
    resolve_frozen_snapshot,
)
from trade_rl.workflows.selection_authorization import (
    SelectionAuthorization,
    write_selection_authorization,
)
from trade_rl.workflows.training_run import TrainingRunConfig

_ROOT = Path(__file__).resolve().parents[2]
_LEGACY_PATH = Path(__file__).with_name("run_full_research.py")
_LEGACY = runpy.run_path(str(_LEGACY_PATH))

_SYMBOLS = tuple(_LEGACY["_SYMBOLS"])
_NATIVE_TIMEFRAMES = tuple(_LEGACY["_NATIVE_TIMEFRAMES"])
_START = str(_LEGACY["_START"])
_END = str(_LEGACY["_END"])
_EXPECTED_POLICY_OBSERVATIONS = int(_LEGACY["_EXPECTED_POLICY_OBSERVATIONS"])
_TRAIN_RUN_COMMAND = tuple(_LEGACY["_TRAIN_RUN_COMMAND"])
_WALK_FORWARD_RUN_COMMAND = tuple(_LEGACY["_WALK_FORWARD_RUN_COMMAND"])
_PARSE_UTC = _LEGACY["_parse_utc"]
_WRITE_JSON = _LEGACY["_write_json"]
_LOAD_JSON = _LEGACY["_load_json"]
_PREPARE_RUN_ROOTS = _LEGACY["_prepare_run_roots"]
_WRITE_RUN_CONFIG = _LEGACY["_write_run_config"]
_RUN_CLI = _LEGACY["_run_cli"]
_BUILD_DATASET = _LEGACY["_build_dataset"]
_REQUIRE_FILE = _LEGACY["_require_file"]
_VERIFY_TRAINING = _LEGACY["_verify_training"]
_EVALUATE_GATE = _LEGACY["_evaluate_walk_forward_research_gate"]
_SENSITIVITY_GATE = _LEGACY["_execution_sensitivity_gate"]
_SELECTED_RECIPE = _LEGACY["_selected_walk_forward_recipe"]
_FINALIZE = _LEGACY["_finalize_research_run"]
_TRAINING_POLICY_DIGEST = _LEGACY["_training_policy_digest"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _metadata_keys() -> dict[str, VerificationKey]:
    raw = os.environ.get("TRADE_RL_METADATA_KEYS", "")
    payload = json.loads(raw) if raw else {}
    if not isinstance(payload, dict):
        raise ValueError("TRADE_RL_METADATA_KEYS must be a JSON object")
    result: dict[str, VerificationKey] = {}
    for key_id, value in payload.items():
        if not isinstance(key_id, str) or not isinstance(value, str):
            raise ValueError("metadata key IDs and values must be strings")
        result[key_id] = VerificationKey(
            key_id=key_id,
            key=value.encode("utf-8"),
            purpose="metadata-verification",
        )
    return result


def _load_signed_history() -> tuple[
    dict[str, dict[str, str | float]],
    dict[str, tuple[InstrumentExecutionRule, ...]],
    BinanceHistoricalSignedScope,
]:
    raw_path = os.environ.get("TRADE_RL_BINANCE_RULE_HISTORY", "")
    if not raw_path:
        raise RuntimeError(
            "TRADE_RL_BINANCE_RULE_HISTORY is required for strict point-in-time research"
        )
    path = Path(raw_path)
    document = _LOAD_JSON(path)
    signed = document.get("payload")
    envelope_raw = document.get("envelope")
    if not isinstance(signed, dict) or not isinstance(envelope_raw, dict):
        raise ValueError("Binance rule history must contain payload and envelope objects")
    envelope = AuthenticatedEnvelope(
        key_id=str(envelope_raw.get("key_id", "")),
        payload_digest=str(envelope_raw.get("payload_digest", "")),
        signature=str(envelope_raw.get("signature", "")),
        schema_version=str(envelope_raw.get("schema_version", "")),
    )
    verify_payload(
        signed,
        envelope,
        trusted_keys=_metadata_keys(),
        required_purpose="metadata-verification",
    )
    if signed.get("schema_version") != "binance_instrument_rule_history_v3":
        raise ValueError("unsupported Binance execution rule history schema")
    if signed.get("policy_version") != "binance_metadata_modes_v1":
        raise ValueError("unsupported Binance metadata policy version")
    raw_symbols = signed.get("symbols")
    if not isinstance(raw_symbols, dict) or tuple(raw_symbols) != _SYMBOLS:
        raise ValueError("Binance execution rule history symbol order is invalid")
    coverage = signed.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError("Binance execution rule history coverage is missing")
    scope = BinanceHistoricalSignedScope(
        market=str(signed.get("market", "")),
        symbols=tuple(raw_symbols),
        coverage_start=_PARSE_UTC(str(coverage.get("start_time", ""))),
        coverage_end=_PARSE_UTC(str(coverage.get("end_time", ""))),
        issued_at=_PARSE_UTC(str(signed.get("issued_at", ""))),
        source_uri=str(signed.get("source_uri", "")),
        payload_digest=envelope.payload_digest,
        policy_version=str(signed.get("policy_version", "")),
    )
    metadata: dict[str, dict[str, str | float]] = {}
    histories: dict[str, tuple[InstrumentExecutionRule, ...]] = {}
    for symbol in _SYMBOLS:
        raw_entry = raw_symbols.get(symbol)
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Binance execution rule history is missing {symbol}")
        listed_at = _PARSE_UTC(str(raw_entry.get("listed_at", "")))
        raw_rules = raw_entry.get("rules")
        if not isinstance(raw_rules, list) or not raw_rules:
            raise ValueError(f"Binance execution rule history is missing {symbol} rules")
        rules = tuple(
            InstrumentExecutionRule(
                effective_at=_PARSE_UTC(str(item["effective_at"])),
                tick_size=float(item["tick_size"]),
                lot_size=float(item["lot_size"]),
                minimum_notional=float(item["minimum_notional"]),
            )
            for item in raw_rules
            if isinstance(item, dict)
        )
        if len(rules) != len(raw_rules):
            raise ValueError(f"Binance execution rule history for {symbol} is invalid")
        latest = rules[-1]
        histories[symbol] = rules
        metadata[symbol] = {
            "listed_at": listed_at.isoformat(),
            "tick_size": latest.tick_size,
            "lot_size": latest.lot_size,
            "minimum_notional": latest.minimum_notional,
        }
    return metadata, histories, scope


def _resolve_metadata(
    *,
    mode: BinanceMetadataMode,
    transport: BinancePublicTransport,
    conservative_static_path: Path | None,
) -> BinanceMetadataResolution:
    start_time = _PARSE_UTC(_START)
    end_time = _PARSE_UTC(_END)
    if mode is BinanceMetadataMode.HISTORICAL_SIGNED:
        metadata, histories, scope = _load_signed_history()
        return resolution_from_historical_signed(
            metadata=metadata,
            execution_rule_histories=histories,
            signed_scope=scope,
            start_time=start_time,
            end_time=end_time,
        )
    if mode is BinanceMetadataMode.FROZEN_SNAPSHOT:
        return resolve_frozen_snapshot(
            transport=transport,
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


def _required_digest(payload: object, *, field: str) -> str:
    if not isinstance(payload, str) or _SHA256.fullmatch(payload) is None:
        raise ValueError(f"{field} is missing or invalid")
    return payload


def _selection_authorization(
    *,
    walk_forward: dict[str, Any],
    preliminary_gate: object,
    sensitivity_gate: dict[str, object],
    selected_configuration: str,
    selected_seeds: tuple[int, ...],
    training_config_path: Path,
    dataset_id: str,
    output_path: Path,
) -> Path:
    config = TrainingRunConfig.from_json(training_config_path)
    authorization = SelectionAuthorization.create(
        walk_forward_run_digest=_required_digest(
            walk_forward.get("run_digest"), field="walk-forward run_digest"
        ),
        gate_evidence_digest=content_digest(
            {
                "research_gate": asdict(preliminary_gate),
                "execution_sensitivity": sensitivity_gate,
            }
        ),
        dataset_id=_required_digest(dataset_id, field="dataset_id"),
        selected_configuration=selected_configuration,
        candidate_config_digest=content_digest(config.candidate_digest_payload()),
        seeds=selected_seeds,
    )
    return write_selection_authorization(output_path, authorization)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--cache-root", type=Path, default=Path("var/cache/binance-vision")
    )
    parser.add_argument(
        "--metadata-mode",
        choices=tuple(mode.value for mode in BinanceMetadataMode),
        default=os.environ.get(
            "TRADE_RL_METADATA_MODE", BinanceMetadataMode.FROZEN_SNAPSHOT.value
        ),
    )
    parser.add_argument(
        "--conservative-static-path",
        type=Path,
        default=(
            Path(os.environ["TRADE_RL_CONSERVATIVE_STATIC_PATH"])
            if os.environ.get("TRADE_RL_CONSERVATIVE_STATIC_PATH")
            else None
        ),
    )
    args = parser.parse_args()

    requested_work_root = (
        args.work_root if args.work_root.is_absolute() else _ROOT / args.work_root
    )
    requested_cache_root = (
        args.cache_root if args.cache_root.is_absolute() else _ROOT / args.cache_root
    )
    work_root, cache_root = _PREPARE_RUN_ROOTS(
        work_root=requested_work_root,
        cache_root=requested_cache_root,
    )
    transport = BinancePublicTransport(
        timeout_seconds=60.0,
        max_attempts=4,
        retry_backoff_seconds=0.5,
        cache_root=cache_root,
    )
    resolution = _resolve_metadata(
        mode=BinanceMetadataMode(args.metadata_mode),
        transport=transport,
        conservative_static_path=args.conservative_static_path,
    )
    resolution.write_artifacts(work_root)
    metadata_report = _LOAD_JSON(work_root / "exchange-info.json")

    dataset_a_path = work_root / "dataset-a"
    dataset_b_path = work_root / "dataset-b"
    dataset_a = _BUILD_DATASET(
        output=dataset_a_path,
        transport=transport,
        metadata=resolution.metadata,
        execution_rule_histories=resolution.execution_rule_histories,
        metadata_evidence=resolution.identity_evidence,
        metadata_mode=resolution.mode,
        metadata_evidence_digest=resolution.evidence_digest,
    )
    dataset_b = _BUILD_DATASET(
        output=dataset_b_path,
        transport=transport,
        metadata=resolution.metadata,
        execution_rule_histories=resolution.execution_rule_histories,
        metadata_evidence=resolution.identity_evidence,
        metadata_mode=resolution.mode,
        metadata_evidence_digest=resolution.evidence_digest,
    )
    if dataset_a["dataset_id"] != dataset_b["dataset_id"]:
        raise RuntimeError("repeated multi-timeframe builds produced different dataset IDs")
    if dataset_a["artifact_digest"] != dataset_b["artifact_digest"]:
        raise RuntimeError(
            "repeated multi-timeframe builds produced different artifact digests"
        )

    dataset = load_market_dataset_artifact(dataset_a_path)
    flat_observation_count = (
        ObservationBuilder(action_size=3, n_factors=0, finite_horizon=True)
        .layout(dataset)
        .size
    )
    sequence_payload = SequenceObservationBuilder().schema_payload(dataset)
    raw_sequence_windows = sequence_payload.get("windows")
    if not isinstance(raw_sequence_windows, (tuple, list)):
        raise RuntimeError("sequence schema windows must be ordered")
    sequence_observation_count = sum(
        dataset.n_symbols
        * int(dict(window)["length"])
        * len(tuple(dict(window)["feature_names"]))
        * 3
        for window in raw_sequence_windows
    )
    policy_observation_count = flat_observation_count + sequence_observation_count
    if policy_observation_count != _EXPECTED_POLICY_OBSERVATIONS:
        raise RuntimeError(
            f"expected {_EXPECTED_POLICY_OBSERVATIONS:,} policy observations, "
            f"observed {policy_observation_count:,}"
        )

    walk_forward_config_path = _WRITE_RUN_CONFIG(
        template_path=_ROOT / "examples/binance-multitimeframe/walk-forward-full.json",
        output_path=work_root / "walk-forward-full.json",
    )
    artifact_root = work_root / "artifacts"
    walk_forward = _RUN_CLI(
        [
            *_WALK_FORWARD_RUN_COMMAND,
            "--config",
            str(walk_forward_config_path),
            "--dataset",
            str(dataset_a_path),
            "--output",
            str(artifact_root),
            "--run-id",
            "binance-multitimeframe-full-walk-forward",
        ],
        root=_ROOT,
        log_path=work_root / "walk-forward.log",
    )
    walk_forward_path = Path(str(walk_forward["artifact_path"]))
    if not walk_forward_path.is_absolute():
        walk_forward_path = _ROOT / walk_forward_path
    _REQUIRE_FILE(walk_forward_path / "run.json")

    summary: dict[str, object] = {
        "dataset": dataset_a,
        "dataset_repeat": dataset_b,
        "end_time": _END,
        "metadata": metadata_report,
        "metadata_error": None,
        "metadata_source": resolution.source_uri,
        "metadata_mode": resolution.mode.value,
        "metadata_evidence_digest": resolution.evidence_digest,
        "native_timeframes": list(_NATIVE_TIMEFRAMES),
        "decision_hours": 0.25,
        "raw_feature_count": dataset.n_features,
        "flat_observation_count": flat_observation_count,
        "sequence_observation_count": sequence_observation_count,
        "policy_observation_count": policy_observation_count,
        "production_status": "NO-GO",
        "schema": "binance_multitimeframe_complete_research_v5_authorized",
        "start_time": _START,
        "training": None,
        "walk_forward": walk_forward,
    }
    preliminary_gate = _EVALUATE_GATE(walk_forward_path, strict=True)
    preliminary_sensitivity_passed, preliminary_sensitivity_gate = _SENSITIVITY_GATE(
        walk_forward_path
    )
    summary["execution_sensitivity"] = preliminary_sensitivity_gate
    if not preliminary_gate.passed or not preliminary_sensitivity_passed:
        exit_code = _FINALIZE(
            work_root=work_root,
            walk_forward_path=walk_forward_path,
            summary=summary,
            strict=True,
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return exit_code

    selected_configuration, selected_seeds, training_config_path = _SELECTED_RECIPE(
        walk_forward_path,
        walk_forward_config_path,
        work_root / "training-selected.json",
    )
    authorization_path = _selection_authorization(
        walk_forward=walk_forward,
        preliminary_gate=preliminary_gate,
        sensitivity_gate=preliminary_sensitivity_gate,
        selected_configuration=selected_configuration,
        selected_seeds=selected_seeds,
        training_config_path=training_config_path,
        dataset_id=str(dataset_a["dataset_id"]),
        output_path=work_root / "selection-authorization.json",
    )
    training = _RUN_CLI(
        [
            *_TRAIN_RUN_COMMAND,
            "--config",
            str(training_config_path),
            "--dataset",
            str(dataset_a_path),
            "--output",
            str(artifact_root),
            "--run-id",
            "binance-multitimeframe-selected-training",
            "--selection-authorization",
            str(authorization_path),
            "--require-selection-authorization",
        ],
        root=_ROOT,
        log_path=work_root / "training.log",
    )
    if training.get("run_kind") != "research_selected_final":
        raise RuntimeError("final training did not retain selected-final authorization")
    training_path = Path(str(training["artifact_path"]))
    if not training_path.is_absolute():
        training_path = _ROOT / training_path
    _VERIFY_TRAINING(training_path)

    summary["selected_training_configuration"] = selected_configuration
    summary["selected_training_seeds"] = list(selected_seeds)
    summary["selection_authorization_digest"] = training.get(
        "selection_authorization_digest"
    )
    summary["confirmation_required_from"] = _END
    summary["training"] = training
    expected_policy_digest = _TRAINING_POLICY_DIGEST(training)
    training_ensemble = _LOAD_JSON(training_path / "ensemble.json")
    expected_environment_digest = training_ensemble.get("environment_digest")
    expected_dataset_id = training.get("dataset_id")
    expected_training_run_digest = training.get("run_digest")
    for name, value in (
        ("environment_digest", expected_environment_digest),
        ("dataset_id", expected_dataset_id),
        ("run_digest", expected_training_run_digest),
    ):
        _required_digest(value, field=f"training {name}")
    exit_code = _FINALIZE(
        work_root=work_root,
        walk_forward_path=walk_forward_path,
        summary=summary,
        strict=True,
        require_confirmation=True,
        expected_policy_digest=expected_policy_digest,
        expected_dataset_id=expected_dataset_id,
        expected_environment_digest=expected_environment_digest,
        expected_training_run_digest=expected_training_run_digest,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
