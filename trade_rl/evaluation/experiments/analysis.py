"""Derived statistical analysis for immutable controlled experiment evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from statistics import median
from typing import cast

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.comparison.paired import compare_paired_returns
from trade_rl.evaluation.comparison.seed_robustness import (
    SeedEvaluation,
    summarize_seed_robustness,
)
from trade_rl.evaluation.experiments.contracts import StudyPlan
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.runs import LoadedCandidateRun
from trade_rl.evaluation.series import ReturnKind, ReturnSeries

_ANALYSIS_SCHEMA = "controlled_evidence_analysis_v1"
_LEGACY_COMPARISON_SCHEMA = "controlled_evidence_comparison_v1"
_COMPARISON_SCHEMA = "controlled_evidence_comparison_v2"
_SUPPORTED_COMPARISON_SCHEMAS = frozenset(
    {_LEGACY_COMPARISON_SCHEMA, _COMPARISON_SCHEMA}
)


@dataclass(frozen=True, slots=True)
class _Cell:
    symbol: str
    strategy: str
    returns: ReturnSeries
    metrics: dict[str, object]


@dataclass(frozen=True, slots=True)
class _CandidateMetrics:
    total_return: float
    max_drawdown: float
    turnover_total: float
    total_cost: float


def _paired_payload(
    candidate: ReturnSeries, baseline: ReturnSeries, *, n_bootstrap: int, seed: int
) -> dict[str, object]:
    paired = compare_paired_returns(
        candidate,
        baseline,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return {
        "excess_total_return": paired.excess_total_return,
        "excess_log_return": paired.excess_log_return,
        "mean_period_excess": paired.mean_period_excess,
        "mean_period_simple_excess": paired.mean_period_simple_excess,
        "p_value": paired.p_value,
        "lower_ci": paired.lower_ci,
        "upper_ci": paired.upper_ci,
        "block_size": paired.block_size,
    }


def _require_metric_number(metrics: Mapping[str, object], field: str) -> float:
    value = metrics.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactIntegrityError(f"candidate metric {field} is malformed")
    resolved = float(value)
    if not np.isfinite(resolved):
        raise ArtifactIntegrityError(f"candidate metric {field} is non-finite")
    return resolved


def _return_series(
    values: np.ndarray,
    metrics: Mapping[str, object],
) -> ReturnSeries:
    kind_raw = metrics.get("return_kind")
    periods_raw = metrics.get("periods_per_year")
    n_periods_raw = metrics.get("n_periods")
    if not isinstance(kind_raw, str):
        raise ArtifactIntegrityError("candidate return kind is malformed")
    try:
        kind = ReturnKind(kind_raw)
    except ValueError as error:
        raise ArtifactIntegrityError("candidate return kind is unsupported") from error
    if (
        isinstance(periods_raw, bool)
        or not isinstance(periods_raw, int)
        or periods_raw <= 0
    ):
        raise ArtifactIntegrityError("candidate periods_per_year is malformed")
    if (
        isinstance(n_periods_raw, bool)
        or not isinstance(n_periods_raw, int)
        or n_periods_raw != int(values.size)
    ):
        raise ArtifactIntegrityError("candidate n_periods does not match raw returns")
    return ReturnSeries(
        values=tuple(float(value) for value in values),
        kind=kind,
        periods_per_year=periods_raw,
    )


def _run_matrix(
    run: LoadedCandidateRun,
) -> tuple[tuple[str, ...], dict[tuple[str, str], _Cell]]:
    symbols_raw = run.summary.get("symbols")
    by_symbol = run.summary.get("by_symbol")
    if not isinstance(symbols_raw, list) or any(
        not isinstance(symbol, str) or not symbol for symbol in symbols_raw
    ):
        raise ArtifactIntegrityError("candidate symbol roster is malformed")
    symbols = tuple(cast(str, symbol) for symbol in symbols_raw)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ArtifactIntegrityError("candidate symbol roster must be unique")
    if not isinstance(by_symbol, list) or len(by_symbol) != len(symbols):
        raise ArtifactIntegrityError("candidate symbol × strategy matrix is incomplete")

    matrix: dict[tuple[str, str], _Cell] = {}
    used_return_keys: set[str] = set()
    for symbol_index, (expected_symbol, symbol_entry) in enumerate(
        zip(symbols, by_symbol, strict=True)
    ):
        if not isinstance(symbol_entry, dict):
            raise ArtifactIntegrityError("candidate symbol evidence is malformed")
        if symbol_entry.get("symbol") != expected_symbol:
            raise ArtifactIntegrityError("candidate symbol matrix ordering mismatch")
        if symbol_entry.get("symbol_index") not in (None, symbol_index):
            raise ArtifactIntegrityError("candidate symbol index mismatch")
        strategies = symbol_entry.get("strategies")
        if not isinstance(strategies, list) or len(strategies) != len(
            StudyPlan.STRATEGY_NAMES
        ):
            raise ArtifactIntegrityError(
                "candidate symbol × strategy matrix is incomplete"
            )
        names: list[str] = []
        for strategy in strategies:
            if not isinstance(strategy, dict):
                raise ArtifactIntegrityError("candidate strategy evidence is malformed")
            name = strategy.get("name")
            return_key = strategy.get("return_key")
            metrics_raw = strategy.get("metrics")
            if not isinstance(name, str) or not isinstance(return_key, str):
                raise ArtifactIntegrityError("candidate strategy evidence is malformed")
            if not isinstance(metrics_raw, dict) or any(
                not isinstance(key, str) for key in metrics_raw
            ):
                raise ArtifactIntegrityError("candidate strategy metrics are malformed")
            names.append(name)
            if return_key in used_return_keys:
                raise ArtifactIntegrityError("candidate return evidence is duplicated")
            used_return_keys.add(return_key)
            values = run.returns.get(return_key)
            if values is None:
                raise ArtifactIntegrityError("candidate return evidence is missing")
            key = (expected_symbol, name)
            if key in matrix:
                raise ArtifactIntegrityError(
                    "candidate symbol × strategy cell is duplicated"
                )
            metrics = cast(dict[str, object], metrics_raw)
            matrix[key] = _Cell(
                symbol=expected_symbol,
                strategy=name,
                returns=_return_series(values, metrics),
                metrics=metrics,
            )
        if tuple(names) != StudyPlan.STRATEGY_NAMES:
            raise ArtifactIntegrityError("candidate strategy roster/order mismatch")

    if set(run.returns) != used_return_keys:
        raise ArtifactIntegrityError("candidate return evidence has extra matrix cells")
    return symbols, matrix


def _validate_runs(
    loaded_runs: Mapping[int, LoadedCandidateRun],
) -> tuple[tuple[int, ...], tuple[str, ...], dict[int, dict[tuple[str, str], _Cell]]]:
    if len(loaded_runs) < 2:
        raise ArtifactIntegrityError("analysis requires at least two seed Runs")
    seeds = tuple(sorted(loaded_runs))
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in seeds
    ):
        raise ArtifactIntegrityError("analysis seed roster is malformed")
    matrices: dict[int, dict[tuple[str, str], _Cell]] = {}
    expected_symbols: tuple[str, ...] | None = None
    for seed in seeds:
        symbols, matrix = _run_matrix(loaded_runs[seed])
        if expected_symbols is None:
            expected_symbols = symbols
        elif symbols != expected_symbols:
            raise ArtifactIntegrityError("analysis symbol roster differs across seeds")
        matrices[seed] = matrix
    assert expected_symbols is not None
    return seeds, expected_symbols, matrices


def _with_digest(payload: dict[str, object]) -> dict[str, object]:
    resolved = dict(payload)
    resolved["analysis_digest"] = content_digest(payload)
    return resolved


def analyze_evidence_set(
    loaded_runs: Mapping[int, LoadedCandidateRun],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Derive within-suite evidence without changing the raw EvidenceSet identity."""

    seeds, symbols, matrices = _validate_runs(loaded_runs)
    first = matrices[seeds[0]]
    by_symbol: dict[str, object] = {}
    for symbol in symbols:
        cash = first[(symbol, "cash")].returns
        paired_vs_cash: dict[str, object] = {}
        for strategy in StudyPlan.PPO_SEED_INVARIANT_STRATEGY_NAMES:
            if strategy == "cash":
                continue
            paired_vs_cash[strategy] = _paired_payload(
                first[(symbol, strategy)].returns,
                cash,
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed,
            )
        paired_candidates = {
            "trend_vs_ridge24": _paired_payload(
                first[(symbol, "trend")].returns,
                first[(symbol, "ridge24")].returns,
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed,
            )
        }
        ppo_seed_evaluations = tuple(
            SeedEvaluation(
                seed=seed,
                returns=matrices[seed][(symbol, "ppo")].returns,
                turnover_total=_require_metric_number(
                    matrices[seed][(symbol, "ppo")].metrics,
                    "turnover_total",
                ),
                total_cost=_require_metric_number(
                    matrices[seed][(symbol, "ppo")].metrics,
                    "total_cost",
                ),
            )
            for seed in seeds
        )
        robustness = summarize_seed_robustness(
            evaluation_label=f"{symbol}:ppo_vs_ridge24",
            seeds=ppo_seed_evaluations,
            baseline=first[(symbol, "ridge24")].returns,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
        )
        by_symbol[symbol] = {
            "paired_vs_cash": paired_vs_cash,
            "paired_candidates": paired_candidates,
            "ppo_seed_robustness_vs_ridge24": asdict(robustness),
        }
    return _with_digest(
        {
            "schema_version": _ANALYSIS_SCHEMA,
            "seeds": list(seeds),
            "by_symbol": by_symbol,
        }
    )


def _candidate_metrics(cell: _Cell) -> _CandidateMetrics:
    return _CandidateMetrics(
        total_return=_require_metric_number(cell.metrics, "total_return"),
        max_drawdown=_require_metric_number(cell.metrics, "max_drawdown"),
        turnover_total=_require_metric_number(cell.metrics, "turnover_total"),
        total_cost=_require_metric_number(cell.metrics, "total_cost"),
    )


def _aggregate_ppo_candidate_metrics(cells: list[_Cell]) -> _CandidateMetrics:
    metrics = [_candidate_metrics(cell) for cell in cells]
    return _CandidateMetrics(
        total_return=float(median(item.total_return for item in metrics)),
        max_drawdown=max(item.max_drawdown for item in metrics),
        turnover_total=float(median(item.turnover_total for item in metrics)),
        total_cost=float(median(item.total_cost for item in metrics)),
    )


def _candidate_metrics_summary(
    metrics: list[_CandidateMetrics],
    *,
    excesses: list[float],
) -> dict[str, object]:
    return {
        "symbol_count": len(excesses),
        "positive_symbol_count": sum(value > 0.0 for value in excesses),
        "negative_symbol_count": sum(value < 0.0 for value in excesses),
        "zero_symbol_count": sum(value == 0.0 for value in excesses),
        "median_excess_total_return": float(median(excesses)),
        "worst_excess_total_return": min(excesses),
        "best_excess_total_return": max(excesses),
        "median_candidate_total_return": float(
            median(item.total_return for item in metrics)
        ),
        "worst_candidate_max_drawdown": max(item.max_drawdown for item in metrics),
        "median_candidate_turnover": float(
            median(item.turnover_total for item in metrics)
        ),
        "median_candidate_total_cost": float(
            median(item.total_cost for item in metrics)
        ),
    }


def compare_evidence_sets(
    baseline_runs: Mapping[int, LoadedCandidateRun],
    candidate_runs: Mapping[int, LoadedCandidateRun],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
    schema_version: str = _COMPARISON_SCHEMA,
) -> dict[str, object]:
    """Compare baseline/candidate EvidenceSets with matched PPO seeds."""

    if schema_version not in _SUPPORTED_COMPARISON_SCHEMAS:
        raise ArtifactIntegrityError("unsupported factor-effect comparison schema")

    baseline_seeds, baseline_symbols, baseline = _validate_runs(baseline_runs)
    candidate_seeds, candidate_symbols, candidate = _validate_runs(candidate_runs)
    if baseline_seeds != candidate_seeds:
        raise ArtifactIntegrityError("baseline/candidate PPO seed rosters must match")
    if baseline_symbols != candidate_symbols:
        raise ArtifactIntegrityError("baseline/candidate symbol rosters must match")

    first_seed = baseline_seeds[0]
    by_symbol: dict[str, object] = {}
    cross_inputs: dict[str, tuple[list[float], list[_CandidateMetrics]]] = {
        strategy: ([], []) for strategy in StudyPlan.STRATEGY_NAMES
    }
    for symbol in baseline_symbols:
        strategy_payloads: dict[str, object] = {}
        for strategy in StudyPlan.STRATEGY_NAMES:
            if strategy == "ppo":
                by_seed: dict[str, object] = {}
                excesses: list[float] = []
                candidate_cells: list[_Cell] = []
                for seed in baseline_seeds:
                    candidate_cell = candidate[seed][(symbol, strategy)]
                    paired = _paired_payload(
                        candidate_cell.returns,
                        baseline[seed][(symbol, strategy)].returns,
                        n_bootstrap=n_bootstrap,
                        seed=bootstrap_seed + seed,
                    )
                    by_seed[str(seed)] = paired
                    excesses.append(
                        _require_metric_number(paired, "excess_total_return")
                    )
                    candidate_cells.append(candidate_cell)
                aggregate = {
                    "positive_seed_count": sum(value > 0.0 for value in excesses),
                    "negative_seed_count": sum(value < 0.0 for value in excesses),
                    "zero_seed_count": sum(value == 0.0 for value in excesses),
                    "median_excess_total_return": float(median(excesses)),
                    "worst_excess_total_return": min(excesses),
                    "best_excess_total_return": max(excesses),
                }
                strategy_payloads[strategy] = {
                    "by_seed": by_seed,
                    "seed_aggregate": aggregate,
                }
                cross_inputs[strategy][0].append(float(median(excesses)))
                candidate_metrics = (
                    _candidate_metrics(candidate_cells[0])
                    if schema_version == _LEGACY_COMPARISON_SCHEMA
                    else _aggregate_ppo_candidate_metrics(candidate_cells)
                )
                cross_inputs[strategy][1].append(candidate_metrics)
            else:
                baseline_cell = baseline[first_seed][(symbol, strategy)]
                candidate_cell = candidate[first_seed][(symbol, strategy)]
                paired = _paired_payload(
                    candidate_cell.returns,
                    baseline_cell.returns,
                    n_bootstrap=n_bootstrap,
                    seed=bootstrap_seed,
                )
                strategy_payloads[strategy] = {
                    "baseline_metrics": baseline_cell.metrics,
                    "candidate_metrics": candidate_cell.metrics,
                    "paired": paired,
                }
                cross_inputs[strategy][0].append(
                    _require_metric_number(paired, "excess_total_return")
                )
                cross_inputs[strategy][1].append(_candidate_metrics(candidate_cell))
        by_symbol[symbol] = {"strategies": strategy_payloads}

    cross_symbol: dict[str, object] = {}
    for strategy, (excesses, metrics) in cross_inputs.items():
        cross_symbol[strategy] = _candidate_metrics_summary(
            metrics, excesses=excesses
        )

    return _with_digest(
        {
            "schema_version": schema_version,
            "seeds": list(baseline_seeds),
            "by_symbol": by_symbol,
            "cross_symbol": cross_symbol,
        }
    )


__all__ = ["analyze_evidence_set", "compare_evidence_sets"]
