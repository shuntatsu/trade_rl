"""Development-only one-seed PPO smoke for a fixed 4h indicator roster."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.data.artifacts import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.directional import CloseAtEndStrategy
from trade_rl.evaluation.directional_contract import DIRECTIONAL_BASE_EXECUTION_COST
from trade_rl.evaluation.directional_study import (
    SOURCE_ARTIFACT_DIGEST,
    SOURCE_DATASET_ID,
    SOURCE_STUDY_DIGEST,
    development_indices,
)
from trade_rl.evaluation.experiments import inspect_study
from trade_rl.evaluation.metrics import compound_return
from trade_rl.evaluation.replay import run_single_symbol_replay
from trade_rl.evaluation.runs import build_candidate_run_provenance
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.strategies.rl.ppo import PPOIntentStrategy, fit_ppo_strategy
from trade_rl.strategies.rl.ppo_artifact import save_ppo_inference_bundle

SCHEMA = "ppo_4h_indicator_smoke_protocol_v1"
RESULT_SCHEMA = "ppo_4h_indicator_smoke_result_v1"
SEED = 0
REQUESTED_TIMESTEPS = 100_000
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
FEATURE_NAMES = (
    "4h__macd_line_12_26",
    "4h__macd_signal_12_26_9",
    "4h__macd_histogram_12_26_9",
    "4h__atr_pct_14bar",
    "4h__plus_di_14bar",
    "4h__minus_di_14bar",
    "4h__ichimoku_tenkan_distance_9bar",
    "4h__ichimoku_kijun_distance_26bar",
    "4h__ichimoku_cloud_position_9_26_52",
    "4h__ichimoku_cloud_thickness_9_26_52",
)
OBSERVATION_WIDTH = 3 * len(FEATURE_NAMES) + 2
SCENARIOS = {
    "base": {"cost_multiplier": 1.0, "latency_bars": 0},
    "cost_2x": {"cost_multiplier": 2.0, "latency_bars": 0},
    "latency_1": {"cost_multiplier": 1.0, "latency_bars": 1},
}
RISK_CONFIG = PreTradeRiskConfig(
    max_gross=0.5,
    max_abs_weight=0.1,
    max_turnover=None,
    drawdown_start=0.1,
    drawdown_stop=0.2,
)
INITIAL_CAPITAL = 10_000.0
GROSS_BUDGET = 0.1


def static_protocol_contract() -> dict[str, object]:
    """Return the result-blind contract that can be reviewed without source bytes."""
    static_contract = static_protocol_contract()
    return {
        **static_contract,
        "static_contract_digest": content_digest(static_contract),
        "source_dataset_id": SOURCE_DATASET_ID,
        "source_artifact_digest": SOURCE_ARTIFACT_DIGEST,
        "source_study_digest": SOURCE_STUDY_DIGEST,
        "symbols": list(SYMBOLS),
        "feature_names": list(FEATURE_NAMES),
        "observation_width": OBSERVATION_WIDTH,
        "training": {
            "seed": SEED,
            "requested_timesteps": REQUESTED_TIMESTEPS,
            "layout": "sequential",
            "normalize_features": False,
            "initial_capital": INITIAL_CAPITAL,
            "gross_budget": GROSS_BUDGET,
            "risk": asdict(RISK_CONFIG),
        },
        "evaluation": {
            "development_window": ["2023-01-01T00", "2025-01-01T00"],
            "one_independent_account_per_symbol": True,
            "initial_capital": INITIAL_CAPITAL,
            "gross_budget": GROSS_BUDGET,
            "scenarios": SCENARIOS,
            "terminal_flat_required": True,
            "drawdown_limit": 0.2,
        },
        "promotion": {
            "hard_guards_all_cells": True,
            "positive_base_symbols_min": 4,
            "base_median_total_return_gt": 0.0,
            "base_year_medians_gt": 0.0,
            "stress_median_total_returns_gt": 0.0,
            "next_stage_only": "full_preregistered_5_seed_study",
        },
        "development_only": True,
        "unused_data": False,
        "production_eligible": False,
    }


def _write_once(path: Path, payload: object) -> bytes:
    raw = canonical_json_bytes(payload)
    with path.open("xb") as stream:
        stream.write(raw)
    return raw


def resolve_feature_indices(dataset: Any) -> tuple[int, ...]:
    """Resolve the fixed roster by semantic name and preserve preregistered order."""
    names = tuple(getattr(dataset, "feature_names", ()))
    if len(names) != len(set(names)):
        raise ValueError("Dataset feature schema contains duplicate names")
    try:
        indices = tuple(names.index(name) for name in FEATURE_NAMES)
    except ValueError as error:
        raise ValueError("a preregistered 4h indicator feature is unavailable") from error
    if tuple(names[index] for index in indices) != FEATURE_NAMES:
        raise ValueError("4h indicator feature order differs from preregistration")
    return indices


def _median(values: list[float]) -> float:
    if len(values) != len(SYMBOLS) or any(not np.isfinite(value) for value in values):
        raise ValueError("smoke comparison requires five finite symbol values")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def promotion_decision(
    results: dict[str, dict[str, dict[str, object]]],
) -> dict[str, object]:
    """Apply the frozen one-seed promotion rule without tuning after results."""
    if tuple(results) != SYMBOLS:
        raise ValueError("smoke result symbol roster differs from preregistration")
    hard_guards_pass = True
    for symbol in SYMBOLS:
        scenarios = results[symbol]
        if tuple(scenarios) != tuple(SCENARIOS):
            raise ValueError("smoke scenario roster differs from preregistration")
        for cell in scenarios.values():
            drawdown = float(cell["ledger_max_drawdown"])
            if (
                not np.isfinite(drawdown)
                or drawdown > 0.2
                or cell.get("terminal_flat") is not True
                or bool(cell.get("termination_reasons"))
                or cell.get("complete", True) is not True
            ):
                hard_guards_pass = False

    base_returns = [float(results[symbol]["base"]["total_return"]) for symbol in SYMBOLS]
    positive_base_symbols = sum(value > 0.0 for value in base_returns)
    base_median = _median(base_returns)
    year_medians = {
        year: _median(
            [
                float(results[symbol]["base"]["year_returns"][year])  # type: ignore[index]
                for symbol in SYMBOLS
            ]
        )
        for year in ("2023", "2024")
    }
    stress_medians = {
        scenario: _median(
            [float(results[symbol][scenario]["total_return"]) for symbol in SYMBOLS]
        )
        for scenario in ("cost_2x", "latency_1")
    }
    promote = bool(
        hard_guards_pass
        and positive_base_symbols >= 4
        and base_median > 0.0
        and all(value > 0.0 for value in year_medians.values())
        and all(value > 0.0 for value in stress_medians.values())
    )
    return {
        "decision": (
            "PROMOTE_TO_FULL_5_SEED_STUDY" if promote else "STOP_AFTER_SMOKE"
        ),
        "hard_guards_pass": hard_guards_pass,
        "positive_base_symbols": positive_base_symbols,
        "base_median_total_return": base_median,
        "base_year_median_returns": year_medians,
        "stress_median_total_returns": stress_medians,
        "production_eligible": False,
        "unused_data_evidence": False,
    }


def expected_protocol(source: Path) -> dict[str, Any]:
    """Resolve the exact result-blind protocol against the frozen source artifact."""
    identity = inspect_published_market_dataset_artifact(source / "dataset")
    dataset = load_market_dataset_artifact(source / "dataset")
    if (
        dataset.dataset_id != SOURCE_DATASET_ID
        or identity.artifact_digest != SOURCE_ARTIFACT_DIGEST
    ):
        raise ValueError("source is not the frozen directional Dataset artifact")
    if tuple(dataset.symbols) != SYMBOLS:
        raise ValueError("source symbol roster differs from the smoke contract")
    indices = resolve_feature_indices(dataset)
    start, stop = development_indices(dataset)
    plan = inspect_study(source / "study").plan
    if plan.digest != SOURCE_STUDY_DIGEST or plan.dataset_id != SOURCE_DATASET_ID:
        raise ValueError("source StudyPlan is not the frozen directional Study")
    config = plan.baseline_config
    if tuple(config.fit_symbol_names) != SYMBOLS:
        raise ValueError("fit symbol roster differs from the smoke contract")
    if tuple(config.fit_symbol_indices) != tuple(range(len(SYMBOLS))):
        raise ValueError("smoke fit scope must include all five symbols")
    fit_cutoff = np.datetime64(config.fit_cutoff)
    if fit_cutoff > np.datetime64("2023-01-01T00", "ns"):
        raise ValueError("smoke fit cutoff overlaps development evaluation")
    return {
        "schema": SCHEMA,
        "source_dataset_id": SOURCE_DATASET_ID,
        "source_artifact_digest": SOURCE_ARTIFACT_DIGEST,
        "source_study_digest": SOURCE_STUDY_DIGEST,
        "source_plan_sha256": sha256(
            (source / "study" / "plan.json").read_bytes()
        ).hexdigest(),
        "dataset_feature_names": list(dataset.feature_names),
        "symbols": list(SYMBOLS),
        "feature_names": list(FEATURE_NAMES),
        "feature_indices": list(indices),
        "observation_width": OBSERVATION_WIDTH,
        "training": {
            "seed": SEED,
            "requested_timesteps": REQUESTED_TIMESTEPS,
            "layout": "sequential",
            "normalize_features": False,
            "fit_symbol_indices": list(config.fit_symbol_indices),
            "fit_symbol_names": list(config.fit_symbol_names),
            "fit_cutoff": config.fit_cutoff,
            "initial_capital": INITIAL_CAPITAL,
            "gross_budget": GROSS_BUDGET,
            "risk": asdict(RISK_CONFIG),
        },
        "evaluation": {
            "start_index": start,
            "stop_index": stop,
            "start_timestamp": str(dataset.timestamps[start]),
            "stop_timestamp": str(dataset.timestamps[stop]),
            "interval_count": stop - start,
            "year_counts": {"2023": 8_760, "2024": 8_784},
            "one_independent_account_per_symbol": True,
            "initial_capital": INITIAL_CAPITAL,
            "gross_budget": GROSS_BUDGET,
            "scenarios": SCENARIOS,
            "terminal_flat_required": True,
            "drawdown_limit": 0.2,
        },
        "promotion": {
            "hard_guards_all_cells": True,
            "positive_base_symbols_min": 4,
            "base_median_total_return_gt": 0.0,
            "base_year_medians_gt": 0.0,
            "stress_median_total_returns_gt": 0.0,
            "next_stage_only": "full_preregistered_5_seed_study",
        },
        "development_only": True,
        "unused_data": False,
        "production_eligible": False,
        "provenance": build_candidate_run_provenance(),
    }


def prepare_smoke(source: Path, output: Path) -> dict[str, Any]:
    """Write the result-blind protocol once; do not fit or replay."""
    output.mkdir(parents=True, exist_ok=False)
    protocol = expected_protocol(source)
    raw = _write_once(output / "protocol.json", protocol)
    _write_once(
        output / "protocol.digest.json",
        {"digest": content_digest(protocol), "sha256": sha256(raw).hexdigest()},
    )
    return protocol


def _validate_protocol(source: Path, output: Path) -> dict[str, Any]:
    expected = expected_protocol(source)
    protocol_path = output / "protocol.json"
    digest_path = output / "protocol.digest.json"
    if (
        not protocol_path.is_file()
        or protocol_path.is_symlink()
        or not digest_path.is_file()
        or digest_path.is_symlink()
    ):
        raise ValueError("smoke protocol is missing")
    raw = protocol_path.read_bytes()
    if raw != canonical_json_bytes(expected):
        raise ValueError("smoke protocol differs from the current frozen contract")
    expected_digest = {
        "digest": content_digest(expected),
        "sha256": sha256(raw).hexdigest(),
    }
    if digest_path.read_bytes() != canonical_json_bytes(expected_digest):
        raise ValueError("smoke protocol digest differs")
    return expected


def _new_strategy(strategy: PPOIntentStrategy) -> PPOIntentStrategy:
    return PPOIntentStrategy(
        strategy.policy,
        feature_indices=strategy.feature_indices,
        feature_names=strategy.feature_names,
        feature_normalizer=strategy.feature_normalizer,
    )


def _cell_result(
    dataset: MarketDataset,
    strategy: PPOIntentStrategy,
    *,
    symbol_index: int,
    start: int,
    stop: int,
    cost_multiplier: float,
    latency_bars: int,
) -> dict[str, object]:
    close_index = stop - latency_bars - 1
    wrapped = CloseAtEndStrategy(_new_strategy(strategy), close_index=close_index)
    execution = replace(
        DIRECTIONAL_BASE_EXECUTION_COST,
        multiplier=cost_multiplier,
        order_latency_bars=latency_bars,
    )
    replay = run_single_symbol_replay(
        dataset,
        wrapped,
        symbol_index=symbol_index,
        start_index=start,
        stop_index=stop,
        gross_budget=GROSS_BUDGET,
        initial_capital=INITIAL_CAPITAL,
        execution_cost=execution,
        risk=PreTradeRisk(RISK_CONFIG),
    )
    values = np.asarray(replay.returns.values, dtype=np.float64)
    complete = len(values) == stop - start
    interval_years = (
        dataset.timestamps[start : start + len(values)]
        .astype("datetime64[Y]")
        .astype(str)
    )
    year_returns = {
        year: compound_return(tuple(float(x) for x in values[interval_years == year]))
        for year in ("2023", "2024")
    }
    terminal_flat = bool(np.all(np.abs(replay.book.quantities) <= 1e-10))
    return {
        "total_return": compound_return(tuple(float(x) for x in values)),
        "year_returns": year_returns,
        "ledger_max_drawdown": float(replay.book.max_drawdown),
        "terminal_flat": terminal_flat,
        "termination_reasons": list(replay.diagnostics.termination_reasons),
        "complete": complete,
        "turnover_total": float(replay.diagnostics.turnover_total),
        "total_cost": float(replay.diagnostics.total_cost),
        "funding_pnl": float(replay.diagnostics.funding_pnl),
        "borrow_cost": float(replay.diagnostics.borrow_cost),
        "n_trades": int(replay.diagnostics.n_trades),
        "rebalance_events": int(replay.diagnostics.rebalance_events),
        "final_portfolio_value": float(replay.book.portfolio_value),
        "terminal_quantities": replay.book.quantities.tolist(),
        "returns": values.tolist(),
    }


def run_smoke(source: Path, output: Path) -> dict[str, Any]:
    """Fit once and run the fixed development smoke against independent accounts."""
    protocol = _validate_protocol(source, output)
    if (output / "result.json").exists() or (output / "bundle").exists():
        raise FileExistsError("smoke economic output already exists")
    dataset = load_market_dataset_artifact(source / "dataset")
    if dataset.dataset_id != SOURCE_DATASET_ID:
        raise ValueError("source Dataset identity changed before smoke execution")
    if tuple(dataset.feature_names) != tuple(protocol["dataset_feature_names"]):
        raise ValueError("Dataset feature schema changed after smoke preparation")
    indices = tuple(protocol["feature_indices"])
    plan = inspect_study(source / "study").plan
    config = plan.baseline_config
    cutoff = int(np.searchsorted(dataset.timestamps, np.datetime64(config.fit_cutoff))) - 1
    strategy = fit_ppo_strategy(
        dataset,
        feature_indices=indices,
        fit_symbol_indices=tuple(config.fit_symbol_indices),
        start_index=0,
        stop_index=cutoff,
        gross_budget=GROSS_BUDGET,
        total_timesteps=REQUESTED_TIMESTEPS,
        seed=SEED,
        initial_capital=INITIAL_CAPITAL,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
        training_layout="sequential",
        risk_config=RISK_CONFIG,
        settle_terminal_position=True,
        normalize_features=False,
    )
    bundle_digest = save_ppo_inference_bundle(
        output / "bundle",
        strategy,
        feature_names=tuple(dataset.feature_names),
    )
    start = int(protocol["evaluation"]["start_index"])
    stop = int(protocol["evaluation"]["stop_index"])
    cells: dict[str, dict[str, dict[str, object]]] = {}
    for symbol_index, symbol in enumerate(SYMBOLS):
        per_symbol: dict[str, dict[str, object]] = {}
        for scenario, settings in SCENARIOS.items():
            per_symbol[scenario] = _cell_result(
                dataset,
                strategy,
                symbol_index=symbol_index,
                start=start,
                stop=stop,
                cost_multiplier=float(settings["cost_multiplier"]),
                latency_bars=int(settings["latency_bars"]),
            )
        cells[symbol] = per_symbol
    decision = promotion_decision(cells)
    result = {
        "schema": RESULT_SCHEMA,
        "protocol_digest": content_digest(protocol),
        "source_dataset_id": SOURCE_DATASET_ID,
        "seed": SEED,
        "requested_timesteps": REQUESTED_TIMESTEPS,
        "actual_timesteps": int(getattr(strategy.policy, "num_timesteps")),
        "bundle_digest": bundle_digest,
        "symbols": list(SYMBOLS),
        "feature_names": list(FEATURE_NAMES),
        "cells": cells,
        "promotion": decision,
        "development_only": True,
        "unused_data": False,
        "production_eligible": False,
    }
    raw = _write_once(output / "result.json", result)
    _write_once(
        output / "result.digest.json",
        {"digest": content_digest(result), "sha256": sha256(raw).hexdigest()},
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run"):
        child = sub.add_parser(command)
        child.add_argument("--source", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare_smoke(args.source, args.output)
    else:
        run_smoke(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
