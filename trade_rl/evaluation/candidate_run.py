"""Filesystem entry point for one immutable universal lean candidate run."""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.data import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.candidate_suite import (
    LeanCandidateConfig,
    run_lean_candidate_suite,
)
from trade_rl.evaluation.metrics import PerformanceMetrics
from trade_rl.evaluation.strategy_comparison import UniversalStrategyComparison

_RESULT_SCHEMA = "lean_candidate_result_v1"
_ALLOWED_CONFIG_KEYS = frozenset(
    {
        "signal_name",
        "feature_names",
        "fit_cutoff",
        "evaluation_start",
        "evaluation_stop_exclusive",
        "rule_entry_threshold",
        "rule_exit_threshold",
        "forecast_entry_threshold",
        "forecast_exit_threshold",
        "ppo_total_timesteps",
        "ppo_seed",
        "gross_budget",
        "initial_capital",
    }
)


@dataclass(frozen=True, slots=True)
class PublishedCandidateRun:
    """Paths of one immutably published candidate comparison run."""

    root: Path
    summary_path: Path
    returns_path: Path


def _read_json_object(path: str | Path) -> dict[str, object]:
    source = Path(path)
    raw = cast(object, json.loads(source.read_text(encoding="utf-8")))
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("candidate run config must be a JSON object")
    resolved = cast(dict[str, object], raw)
    unknown = sorted(set(resolved) - _ALLOWED_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")
    return resolved


def _required_string(config: dict[str, object], name: str) -> str:
    value = config.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_float(config: dict[str, object], name: str) -> float:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be a finite number")
    return resolved


def _required_int(config: dict[str, object], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _required_string_tuple(config: dict[str, object], name: str) -> tuple[str, ...]:
    value = config.get(name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty string array")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must be a non-empty string array")
    result = tuple(cast(str, item) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _timestamp(config: dict[str, object], name: str) -> np.datetime64:
    text = _required_string(config, name)
    try:
        value = np.datetime64(text, "ns")
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO datetime") from error
    if np.isnat(value):
        raise ValueError(f"{name} must not be NaT")
    return value


def _feature_index(dataset: MarketDataset, name: str) -> int:
    try:
        return dataset.feature_names.index(name)
    except ValueError as error:
        raise ValueError(f"unknown feature name: {name}") from error


def _exact_timestamp_index(
    dataset: MarketDataset,
    timestamp: np.datetime64,
    *,
    field: str,
) -> int:
    values = dataset.timestamps.astype("datetime64[ns]")
    matches = np.flatnonzero(values == np.datetime64(timestamp, "ns"))
    if matches.size != 1:
        raise ValueError(f"{field} must exactly match one dataset timestamp")
    return int(matches[0])


def _resolved_run_config(
    dataset: MarketDataset,
    raw: dict[str, object],
) -> tuple[LeanCandidateConfig, int, int, float, float]:
    signal_name = _required_string(raw, "signal_name")
    feature_names = _required_string_tuple(raw, "feature_names")
    feature_indices = tuple(_feature_index(dataset, name) for name in feature_names)
    fit_cutoff = _timestamp(raw, "fit_cutoff")
    start_time = _timestamp(raw, "evaluation_start")
    stop_time = _timestamp(raw, "evaluation_stop_exclusive")
    start_index = _exact_timestamp_index(dataset, start_time, field="evaluation_start")
    stop_index = _exact_timestamp_index(
        dataset,
        stop_time,
        field="evaluation_stop_exclusive",
    )
    config = LeanCandidateConfig(
        signal_index=_feature_index(dataset, signal_name),
        feature_indices=feature_indices,
        fit_cutoff=fit_cutoff,
        rule_entry_threshold=_required_float(raw, "rule_entry_threshold"),
        rule_exit_threshold=_required_float(raw, "rule_exit_threshold"),
        forecast_entry_threshold=_required_float(raw, "forecast_entry_threshold"),
        forecast_exit_threshold=_required_float(raw, "forecast_exit_threshold"),
        ppo_total_timesteps=_required_int(raw, "ppo_total_timesteps"),
        ppo_seed=_required_int(raw, "ppo_seed"),
    )
    gross_budget = _required_float(raw, "gross_budget")
    initial_capital = _required_float(raw, "initial_capital")
    return config, start_index, stop_index, gross_budget, initial_capital


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


def _comparison_payload(
    dataset: MarketDataset,
    comparison: UniversalStrategyComparison,
    *,
    dataset_artifact_schema: str,
    dataset_artifact_digest: str,
    config: LeanCandidateConfig,
    start_index: int,
    stop_index: int,
    gross_budget: float,
    initial_capital: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    returns: dict[str, np.ndarray] = {}
    symbols_payload: list[dict[str, object]] = []
    for symbol_result in comparison.by_symbol:
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
        "dataset_id": dataset.dataset_id,
        "dataset_artifact": {
            "schema_version": dataset_artifact_schema,
            "artifact_digest": dataset_artifact_digest,
        },
        "symbols": list(dataset.symbols),
        "candidate_config": {
            "signal_name": dataset.feature_names[config.signal_index],
            "signal_index": config.signal_index,
            "feature_names": [
                dataset.feature_names[index] for index in config.feature_indices
            ],
            "feature_indices": list(config.feature_indices),
            "fit_cutoff": str(config.fit_cutoff),
            "rule_entry_threshold": config.rule_entry_threshold,
            "rule_exit_threshold": config.rule_exit_threshold,
            "forecast_entry_threshold": config.forecast_entry_threshold,
            "forecast_exit_threshold": config.forecast_exit_threshold,
            "ppo_total_timesteps": config.ppo_total_timesteps,
            "ppo_seed": config.ppo_seed,
        },
        "evaluation": {
            "start": str(dataset.timestamps[start_index]),
            "stop_exclusive": str(dataset.timestamps[stop_index]),
            "gross_budget": gross_budget,
            "initial_capital": initial_capital,
            "execution_overlay": "zero_overlay_dataset_fields_authoritative",
        },
        "by_symbol": symbols_payload,
    }
    return summary, returns


def _publish_run(
    output_root: str | Path,
    summary: dict[str, object],
    returns: dict[str, np.ndarray],
) -> PublishedCandidateRun:
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"candidate run destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        summary_path = staging / "summary.json"
        returns_path = staging / "returns.npz"
        summary_bytes = json.dumps(
            summary,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        atomic_write_bytes(summary_path, summary_bytes)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **returns)
        atomic_write_bytes(returns_path, buffer.getvalue())
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return PublishedCandidateRun(
        root=output,
        summary_path=output / "summary.json",
        returns_path=output / "returns.npz",
    )


def run_candidate_artifact(
    *,
    dataset_root: str | Path,
    config_path: str | Path,
    output_root: str | Path,
) -> PublishedCandidateRun:
    """Load one dataset artifact, run the universal suite, and publish evidence."""

    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"candidate run destination already exists: {output}")
    artifact = inspect_published_market_dataset_artifact(dataset_root)
    dataset = load_market_dataset_artifact(dataset_root)
    raw = _read_json_object(config_path)
    config, start_index, stop_index, gross_budget, initial_capital = (
        _resolved_run_config(dataset, raw)
    )
    comparison = run_lean_candidate_suite(
        dataset,
        config,
        start_index=start_index,
        stop_index=stop_index,
        gross_budget=gross_budget,
        initial_capital=initial_capital,
        execution_cost=None,
        risk=None,
    )
    summary, returns = _comparison_payload(
        dataset,
        comparison,
        dataset_artifact_schema=artifact.schema_version,
        dataset_artifact_digest=artifact.artifact_digest,
        config=config,
        start_index=start_index,
        stop_index=stop_index,
        gross_budget=gross_budget,
        initial_capital=initial_capital,
    )
    return _publish_run(output, summary, returns)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the universal lean candidate suite from filesystem artifacts."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    artifact = run_candidate_artifact(
        dataset_root=cast(str, args.dataset),
        config_path=cast(str, args.config),
        output_root=cast(str, args.output),
    )
    print(artifact.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PublishedCandidateRun", "main", "run_candidate_artifact"]
