#!/usr/bin/env python3
"""Run the maintained three-asset native multi-timeframe research pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.data import publish_market_dataset_artifact
from trade_rl.data.contracts import InstrumentExecutionRule
from trade_rl.evaluation.confirmation import load_confirmation_evidence
from trade_rl.evaluation.research_gate import (
    ResearchEvidenceRequirements,
    ResearchReturnGate,
    evaluate_research_return_gate,
    paired_block_bootstrap_excess_lower_bound,
)
from trade_rl.integrations.binance import (
    BinanceMarket,
    BinancePublicTransport,
    BinanceTransportMode,
    binance_multitimeframe_feature_specs,
    build_binance_market_dataset,
)
from trade_rl.release.signing import AuthenticatedEnvelope, verify_payload
from trade_rl.rl.checkpointing import checkpoint_manifests
from trade_rl.rl.observations import ObservationBuilder

_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT")
_NATIVE_TIMEFRAMES = ("15m", "1h", "4h", "1d")
_FEATURE_TIMEFRAMES = ("1h", "4h", "1d")
_START = "2024-12-01T00:00:00Z"
_END = "2026-07-01T00:00:00Z"
_EXPECTED_15M_BARS = 55_392
_EXPECTED_POLICY_OBSERVATIONS = 230_999
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_TRAIN_RUN_COMMAND = ("train", "run")
_WALK_FORWARD_RUN_COMMAND = ("walk-forward", "run")


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


def _load_rule_history() -> tuple[
    dict[str, dict[str, str | float]],
    dict[str, tuple[InstrumentExecutionRule, ...]],
    str,
]:
    """Load the authenticated, deterministic instrument metadata history."""

    raw_path = os.environ.get("TRADE_RL_BINANCE_RULE_HISTORY", "")
    if not raw_path:
        raise RuntimeError(
            "TRADE_RL_BINANCE_RULE_HISTORY is required for strict point-in-time research"
        )
    path = Path(raw_path)
    payload = _load_json(path)
    signed = payload.get("payload")
    envelope_raw = payload.get("envelope")
    if not isinstance(signed, dict) or not isinstance(envelope_raw, dict):
        raise ValueError(
            "Binance rule history must contain payload and envelope objects"
        )
    envelope = AuthenticatedEnvelope(
        key_id=str(envelope_raw.get("key_id", "")),
        payload_digest=str(envelope_raw.get("payload_digest", "")),
        signature=str(envelope_raw.get("signature", "")),
        schema_version=str(envelope_raw.get("schema_version", "")),
    )
    keys_raw = os.environ.get("TRADE_RL_METADATA_KEYS", "")
    keys_payload = json.loads(keys_raw) if keys_raw else {}
    if not isinstance(keys_payload, dict):
        raise ValueError("TRADE_RL_METADATA_KEYS must be a JSON object")
    trusted_keys = {
        str(key): str(value).encode("utf-8") for key, value in keys_payload.items()
    }
    verify_payload(signed, envelope, trusted_keys=trusted_keys)
    if signed.get("schema_version") != "binance_instrument_rule_history_v2":
        raise ValueError("unsupported Binance execution rule history schema")
    raw_symbols = signed.get("symbols")
    if not isinstance(raw_symbols, dict):
        raise ValueError("Binance execution rule history symbols are missing")
    metadata: dict[str, dict[str, str | float]] = {}
    histories: dict[str, tuple[InstrumentExecutionRule, ...]] = {}
    for symbol in _SYMBOLS:
        raw_entry = raw_symbols.get(symbol)
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Binance execution rule history is missing {symbol}")
        listed_at = _parse_utc(str(raw_entry.get("listed_at", "")))
        raw_rules = raw_entry.get("rules")
        if not isinstance(raw_rules, list) or not raw_rules:
            raise ValueError(
                f"Binance execution rule history is missing {symbol} rules"
            )
        rules = tuple(
            InstrumentExecutionRule(
                effective_at=_parse_utc(str(item["effective_at"])),
                tick_size=float(item["tick_size"]),
                lot_size=float(item["lot_size"]),
                minimum_notional=float(item["minimum_notional"]),
            )
            for item in raw_rules
            if isinstance(item, dict)
        )
        if len(rules) != len(raw_rules):
            raise ValueError(f"Binance execution rule history for {symbol} is invalid")
        ordered = tuple(sorted(rules, key=lambda item: item.effective_at))
        latest = ordered[-1]
        histories[symbol] = ordered
        metadata[symbol] = {
            "listed_at": listed_at.isoformat(),
            "tick_size": latest.tick_size,
            "lot_size": latest.lot_size,
            "minimum_notional": latest.minimum_notional,
        }
    unknown = set(raw_symbols) - set(_SYMBOLS)
    if unknown:
        raise ValueError(
            f"Binance execution rule history contains unknown symbols: {sorted(unknown)}"
        )
    return metadata, histories, envelope.payload_digest


def _build_dataset(
    *,
    output: Path,
    transport: BinancePublicTransport,
    metadata: dict[str, dict[str, str | float]],
    execution_rule_histories: dict[str, tuple[InstrumentExecutionRule, ...]],
) -> dict[str, object]:
    result = build_binance_market_dataset(
        market=BinanceMarket.USDS_M,
        symbols=_SYMBOLS,
        interval="15m",
        feature_timeframes=_FEATURE_TIMEFRAMES,
        start_time=_parse_utc(_START),
        end_time=_parse_utc(_END),
        transport_mode=BinanceTransportMode.VISION,
        transport=transport,
        tick_sizes=tuple(float(metadata[symbol]["tick_size"]) for symbol in _SYMBOLS),
        lot_sizes=tuple(float(metadata[symbol]["lot_size"]) for symbol in _SYMBOLS),
        minimum_notionals=tuple(
            float(metadata[symbol]["minimum_notional"]) for symbol in _SYMBOLS
        ),
        listed_ats=tuple(
            _parse_utc(str(metadata[symbol]["listed_at"])) for symbol in _SYMBOLS
        ),
        execution_rule_histories=execution_rule_histories,
    )
    published = publish_market_dataset_artifact(output, result.dataset)
    if result.dataset.n_bars != _EXPECTED_15M_BARS:
        raise RuntimeError(
            "expected "
            f"{_EXPECTED_15M_BARS:,} 15-minute bars, observed {result.dataset.n_bars}"
        )
    if result.dataset.symbols != _SYMBOLS:
        raise RuntimeError(f"unexpected symbol order: {result.dataset.symbols}")
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
    if result.dataset.feature_names != expected_features:
        raise RuntimeError(
            f"unexpected feature contract: {result.dataset.feature_names}"
        )
    return {
        "artifact_digest": published.artifact_digest,
        "dataset_id": result.dataset.dataset_id,
        "feature_names": list(result.dataset.feature_names),
        "feature_timeframes": list(result.feature_timeframes),
        "n_bars": result.dataset.n_bars,
        "n_features": result.dataset.n_features,
        "n_symbols": result.dataset.n_symbols,
        "sources_used": list(result.sources_used),
        "symbols": list(result.dataset.symbols),
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


def _trusted_confirmation_keys() -> dict[str, bytes]:
    raw = os.environ.get("TRADE_RL_CONFIRMATION_KEYS", "")
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("TRADE_RL_CONFIRMATION_KEYS must be a JSON object")
    result: dict[str, bytes] = {}
    for key_id, value in payload.items():
        if not isinstance(key_id, str) or not isinstance(value, str):
            raise ValueError("confirmation key IDs and values must be strings")
        result[key_id] = value.encode("utf-8")
    return result


def _confirmation_evidence(
    path: Path | None,
    *,
    expected_policy_digest: str | None,
    expected_dataset_id: str | None,
    expected_environment_digest: str | None,
    expected_training_run_digest: str | None,
) -> tuple[bool, float]:
    if path is None or not path.is_file():
        return False, 0.0
    try:
        evidence = load_confirmation_evidence(path)
        evidence.verify(_trusted_confirmation_keys())
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
        )
    )
    summary["research_gate"] = gate
    _write_json(work_root / "research-gate.json", gate)
    _write_json(work_root / "summary.json", summary)
    return 0 if gate["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--cache-root", type=Path, default=Path("var/cache/binance-vision")
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    requested_work_root = (
        args.work_root if args.work_root.is_absolute() else root / args.work_root
    )
    requested_cache_root = (
        args.cache_root if args.cache_root.is_absolute() else root / args.cache_root
    )
    work_root, cache_root = _prepare_run_roots(
        work_root=requested_work_root,
        cache_root=requested_cache_root,
    )
    transport = BinancePublicTransport(
        timeout_seconds=60.0,
        max_attempts=4,
        retry_backoff_seconds=0.5,
        cache_root=cache_root,
    )
    (
        metadata,
        execution_rule_histories,
        metadata_evidence_digest,
    ) = _load_rule_history()
    metadata_source = "authenticated-rule-history"
    metadata_error = None
    _write_json(
        work_root / "exchange-info.json",
        {
            "evidence_digest": metadata_evidence_digest,
            "metadata": metadata,
            "source": metadata_source,
        },
    )

    dataset_a_path = work_root / "dataset-a"
    dataset_b_path = work_root / "dataset-b"
    dataset_a = _build_dataset(
        output=dataset_a_path,
        transport=transport,
        metadata=metadata,
        execution_rule_histories=execution_rule_histories,
    )
    dataset_b = _build_dataset(
        output=dataset_b_path,
        transport=transport,
        metadata=metadata,
        execution_rule_histories=execution_rule_histories,
    )
    if dataset_a["dataset_id"] != dataset_b["dataset_id"]:
        raise RuntimeError(
            "repeated multi-timeframe builds produced different dataset IDs"
        )
    if dataset_a["artifact_digest"] != dataset_b["artifact_digest"]:
        raise RuntimeError(
            "repeated multi-timeframe builds produced different artifact digests"
        )
    from trade_rl.data import load_market_dataset_artifact

    dataset = load_market_dataset_artifact(dataset_a_path)
    flat_observation_count = (
        ObservationBuilder(
            action_size=3,
            n_factors=0,
            finite_horizon=True,
        )
        .layout(dataset)
        .size
    )
    from trade_rl.rl.sequence_observations import SequenceObservationBuilder

    sequence_payload = SequenceObservationBuilder().schema_payload(dataset)
    raw_sequence_windows = sequence_payload.get("windows")
    if not isinstance(raw_sequence_windows, (tuple, list)):
        raise RuntimeError("sequence schema windows must be ordered")
    sequence_observation_count = 0
    for window in raw_sequence_windows:
        item = dict(window)
        sequence_observation_count += (
            dataset.n_symbols
            * int(item["length"])
            * len(tuple(item["feature_names"]))
            * 3
        )
    policy_observation_count = flat_observation_count + sequence_observation_count
    if policy_observation_count != _EXPECTED_POLICY_OBSERVATIONS:
        raise RuntimeError(
            "expected "
            f"{_EXPECTED_POLICY_OBSERVATIONS:,} policy observations, observed "
            f"{policy_observation_count:,}"
        )
    walk_forward_config_path = _write_run_config(
        template_path=root / "examples/binance-multitimeframe/walk-forward-full.json",
        output_path=work_root / "walk-forward-full.json",
    )

    artifact_root = work_root / "artifacts"
    walk_forward = _run_cli(
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
        root=root,
        log_path=work_root / "walk-forward.log",
    )
    walk_forward_path = Path(str(walk_forward["artifact_path"]))
    if not walk_forward_path.is_absolute():
        walk_forward_path = root / walk_forward_path
    _require_file(walk_forward_path / "run.json")

    summary: dict[str, object] = {
        "dataset": dataset_a,
        "dataset_repeat": dataset_b,
        "end_time": _END,
        "metadata_error": metadata_error,
        "metadata_source": metadata_source,
        "native_timeframes": list(_NATIVE_TIMEFRAMES),
        "decision_hours": 0.25,
        "raw_feature_count": dataset.n_features,
        "flat_observation_count": flat_observation_count,
        "sequence_observation_count": sequence_observation_count,
        "policy_observation_count": policy_observation_count,
        "production_status": "NO-GO",
        "schema": "binance_multitimeframe_complete_research_v4_seed_stable",
        "start_time": _START,
        "training": None,
        "walk_forward": walk_forward,
    }
    preliminary_gate = _evaluate_walk_forward_research_gate(
        walk_forward_path,
        strict=True,
    )
    if not preliminary_gate.passed:
        exit_code = _finalize_research_run(
            work_root=work_root,
            walk_forward_path=walk_forward_path,
            summary=summary,
            strict=True,
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return exit_code

    (
        selected_configuration,
        selected_seeds,
        training_config_path,
    ) = _selected_walk_forward_recipe(
        walk_forward_path,
        walk_forward_config_path,
        work_root / "training-selected.json",
    )
    training = _run_cli(
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
        ],
        root=root,
        log_path=work_root / "training.log",
    )
    training_path = Path(str(training["artifact_path"]))
    if not training_path.is_absolute():
        training_path = root / training_path
    _verify_training(training_path)

    summary["selected_training_configuration"] = selected_configuration
    summary["selected_training_seeds"] = list(selected_seeds)
    summary["confirmation_required_from"] = _END
    summary["training"] = training
    expected_policy_digest = _training_policy_digest(training)
    training_ensemble = _load_json(training_path / "ensemble.json")
    expected_environment_digest = training_ensemble.get("environment_digest")
    expected_dataset_id = training.get("dataset_id")
    expected_training_run_digest = training.get("run_digest")
    for name, value in (
        ("environment_digest", expected_environment_digest),
        ("dataset_id", expected_dataset_id),
        ("run_digest", expected_training_run_digest),
    ):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"training {name} is missing or invalid")
    exit_code = _finalize_research_run(
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
