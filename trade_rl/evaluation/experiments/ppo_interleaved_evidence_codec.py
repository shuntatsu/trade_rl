"""Canonical serialization boundary for PPO interleaved seed evidence."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.evaluation.experiments.ppo_interleaved_evaluation import (
    PPOReturnPathEvidence,
    PPOSeedEvidence,
    PPOSymbolEvidence,
)

_T = TypeVar("_T")


def _mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a JSON object")
    return dict(value)


def _exact_keys(payload: dict[str, object], cls: type[_T], *, field: str) -> None:
    expected = {item.name for item in fields(cls)}
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{field} keys differ from schema; missing={missing}, unknown={unknown}"
        )


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    return float(value)


def _bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    result = tuple(_string(item, field=field) for item in value)
    return result


def _int_tuple(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    return tuple(_integer(item, field=field) for item in value)


def _float_tuple(value: object, *, field: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    return tuple(_number(item, field=field) for item in value)


def _path_from_payload(value: object, *, field: str) -> PPOReturnPathEvidence:
    payload = _mapping(value, field=field)
    _exact_keys(payload, PPOReturnPathEvidence, field=field)
    return PPOReturnPathEvidence(
        strategy_name=_string(payload["strategy_name"], field=f"{field}.strategy_name"),
        returns=_float_tuple(payload["returns"], field=f"{field}.returns"),
        return_sha256=_string(
            payload["return_sha256"], field=f"{field}.return_sha256"
        ),
        total_return=_number(payload["total_return"], field=f"{field}.total_return"),
        total_cost=_number(payload["total_cost"], field=f"{field}.total_cost"),
        turnover_total=_number(
            payload["turnover_total"], field=f"{field}.turnover_total"
        ),
        max_drawdown=_number(payload["max_drawdown"], field=f"{field}.max_drawdown"),
        termination_count=_integer(
            payload["termination_count"], field=f"{field}.termination_count"
        ),
        termination_reasons=_string_tuple(
            payload["termination_reasons"], field=f"{field}.termination_reasons"
        ),
        n_periods=_integer(payload["n_periods"], field=f"{field}.n_periods"),
        periods_per_year=_integer(
            payload["periods_per_year"], field=f"{field}.periods_per_year"
        ),
        schema_version=_string(
            payload["schema_version"], field=f"{field}.schema_version"
        ),
    )


def _symbol_from_payload(value: object, *, field: str) -> PPOSymbolEvidence:
    payload = _mapping(value, field=field)
    _exact_keys(payload, PPOSymbolEvidence, field=field)
    return PPOSymbolEvidence(
        symbol_index=_integer(payload["symbol_index"], field=f"{field}.symbol_index"),
        symbol=_string(payload["symbol"], field=f"{field}.symbol"),
        ppo=_path_from_payload(payload["ppo"], field=f"{field}.ppo"),
        cash=_path_from_payload(payload["cash"], field=f"{field}.cash"),
        constant_long=_path_from_payload(
            payload["constant_long"], field=f"{field}.constant_long"
        ),
        constant_short=_path_from_payload(
            payload["constant_short"], field=f"{field}.constant_short"
        ),
        ppo_returns_equal_cash=_bool(
            payload["ppo_returns_equal_cash"],
            field=f"{field}.ppo_returns_equal_cash",
        ),
        ppo_returns_equal_constant_long=_bool(
            payload["ppo_returns_equal_constant_long"],
            field=f"{field}.ppo_returns_equal_constant_long",
        ),
        ppo_returns_equal_constant_short=_bool(
            payload["ppo_returns_equal_constant_short"],
            field=f"{field}.ppo_returns_equal_constant_short",
        ),
        schema_version=_string(
            payload["schema_version"], field=f"{field}.schema_version"
        ),
    )


def ppo_seed_evidence_from_payload(value: object) -> PPOSeedEvidence:
    """Reconstruct one seed evidence object from an exact JSON-compatible payload."""

    payload = _mapping(value, field="seed_evidence")
    _exact_keys(payload, PPOSeedEvidence, field="seed_evidence")
    by_symbol_raw = payload["by_symbol"]
    if not isinstance(by_symbol_raw, list):
        raise ValueError("seed_evidence.by_symbol must be a JSON array")
    by_symbol = tuple(
        _symbol_from_payload(item, field=f"seed_evidence.by_symbol[{index}]")
        for index, item in enumerate(by_symbol_raw)
    )
    rollout_raw = payload["rollout_steps_per_env"]
    rollout_steps = (
        None
        if rollout_raw is None
        else _integer(rollout_raw, field="seed_evidence.rollout_steps_per_env")
    )
    return PPOSeedEvidence(
        spec_digest=_string(payload["spec_digest"], field="seed_evidence.spec_digest"),
        protocol_head=_string(
            payload["protocol_head"], field="seed_evidence.protocol_head"
        ),
        protocol_digest=_string(
            payload["protocol_digest"], field="seed_evidence.protocol_digest"
        ),
        implementation_head=_string(
            payload["implementation_head"], field="seed_evidence.implementation_head"
        ),
        successor_bundle_run_id=_integer(
            payload["successor_bundle_run_id"],
            field="seed_evidence.successor_bundle_run_id",
        ),
        successor_bundle_artifact_id=_integer(
            payload["successor_bundle_artifact_id"],
            field="seed_evidence.successor_bundle_artifact_id",
        ),
        successor_bundle_artifact_digest=_string(
            payload["successor_bundle_artifact_digest"],
            field="seed_evidence.successor_bundle_artifact_digest",
        ),
        dataset_id=_string(payload["dataset_id"], field="seed_evidence.dataset_id"),
        dataset_artifact_digest=_string(
            payload["dataset_artifact_digest"],
            field="seed_evidence.dataset_artifact_digest",
        ),
        study_digest=_string(
            payload["study_digest"], field="seed_evidence.study_digest"
        ),
        execution_overlay=_string(
            payload["execution_overlay"], field="seed_evidence.execution_overlay"
        ),
        symbols=_string_tuple(payload["symbols"], field="seed_evidence.symbols"),
        feature_names=_string_tuple(
            payload["feature_names"], field="seed_evidence.feature_names"
        ),
        feature_indices=_int_tuple(
            payload["feature_indices"], field="seed_evidence.feature_indices"
        ),
        fit_symbol_names=_string_tuple(
            payload["fit_symbol_names"], field="seed_evidence.fit_symbol_names"
        ),
        fit_cutoff=_string(payload["fit_cutoff"], field="seed_evidence.fit_cutoff"),
        evaluation_start=_string(
            payload["evaluation_start"], field="seed_evidence.evaluation_start"
        ),
        evaluation_stop_exclusive=_string(
            payload["evaluation_stop_exclusive"],
            field="seed_evidence.evaluation_stop_exclusive",
        ),
        gross_budget=_number(
            payload["gross_budget"], field="seed_evidence.gross_budget"
        ),
        initial_capital=_number(
            payload["initial_capital"], field="seed_evidence.initial_capital"
        ),
        arm=_string(payload["arm"], field="seed_evidence.arm"),  # type: ignore[arg-type]
        seed=_integer(payload["seed"], field="seed_evidence.seed"),
        training_layout=_string(
            payload["training_layout"], field="seed_evidence.training_layout"
        ),
        rollout_steps_per_env=rollout_steps,
        caller_total_timesteps=_integer(
            payload["caller_total_timesteps"],
            field="seed_evidence.caller_total_timesteps",
        ),
        realized_num_timesteps=_integer(
            payload["realized_num_timesteps"],
            field="seed_evidence.realized_num_timesteps",
        ),
        by_symbol=by_symbol,
        protocol_seal_run_id=_integer(
            payload["protocol_seal_run_id"], field="seed_evidence.protocol_seal_run_id"
        ),
        protocol_primary_artifact_id=_integer(
            payload["protocol_primary_artifact_id"],
            field="seed_evidence.protocol_primary_artifact_id",
        ),
        protocol_primary_artifact_digest=_string(
            payload["protocol_primary_artifact_digest"],
            field="seed_evidence.protocol_primary_artifact_digest",
        ),
        protocol_fresh_artifact_id=_integer(
            payload["protocol_fresh_artifact_id"],
            field="seed_evidence.protocol_fresh_artifact_id",
        ),
        protocol_fresh_artifact_digest=_string(
            payload["protocol_fresh_artifact_digest"],
            field="seed_evidence.protocol_fresh_artifact_digest",
        ),
        protocol_json_sha256=_string(
            payload["protocol_json_sha256"], field="seed_evidence.protocol_json_sha256"
        ),
        protocol_seal_json_sha256=_string(
            payload["protocol_seal_json_sha256"],
            field="seed_evidence.protocol_seal_json_sha256",
        ),
        baseline_training_authorized=_bool(
            payload["baseline_training_authorized"],
            field="seed_evidence.baseline_training_authorized",
        ),
        candidate_training_authorized=_bool(
            payload["candidate_training_authorized"],
            field="seed_evidence.candidate_training_authorized",
        ),
        economic_result_inspected=_bool(
            payload["economic_result_inspected"],
            field="seed_evidence.economic_result_inspected",
        ),
        final_test_accessed=_bool(
            payload["final_test_accessed"], field="seed_evidence.final_test_accessed"
        ),
        shared_cash_profitability_established=_bool(
            payload["shared_cash_profitability_established"],
            field="seed_evidence.shared_cash_profitability_established",
        ),
        production_eligible=_bool(
            payload["production_eligible"], field="seed_evidence.production_eligible"
        ),
        live_trading_authorized=_bool(
            payload["live_trading_authorized"],
            field="seed_evidence.live_trading_authorized",
        ),
        merge_authorized=_bool(
            payload["merge_authorized"], field="seed_evidence.merge_authorized"
        ),
        schema_version=_string(
            payload["schema_version"], field="seed_evidence.schema_version"
        ),
    )


def load_ppo_seed_evidence(path: Path) -> PPOSeedEvidence:
    """Load one seed evidence file only if bytes and semantic evidence are canonical."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    raw = path.read_bytes()
    try:
        decoded: Any = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("PPO seed evidence JSON is malformed") from error
    if raw != canonical_json_bytes(decoded):
        raise ValueError("PPO seed evidence must use canonical JSON bytes")
    evidence = ppo_seed_evidence_from_payload(decoded)
    if raw != canonical_json_bytes(evidence.to_payload()):
        raise ValueError("PPO seed evidence bytes differ from reconstructed evidence")
    return evidence


__all__ = ["load_ppo_seed_evidence", "ppo_seed_evidence_from_payload"]
