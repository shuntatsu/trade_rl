from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from trade_rl.evaluation.comparison.paired import compare_paired_returns
from trade_rl.evaluation.comparison.seed_robustness import (
    SeedEvaluation,
    summarize_seed_robustness,
)
from trade_rl.evaluation.experiments.analysis import (
    analyze_evidence_set,
    compare_evidence_sets,
)
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.metrics import evaluate_performance
from trade_rl.evaluation.runs.artifact import LoadedCandidateRun
from trade_rl.evaluation.series import ReturnKind, ReturnSeries

STRATEGIES = (
    "cash",
    "constant_long",
    "constant_short",
    "trend",
    "mean_reversion",
    "ridge24",
    "lightgbm24",
    "ppo",
)


def _series(values: tuple[float, ...]) -> ReturnSeries:
    return ReturnSeries(
        values=values,
        kind=ReturnKind.BASE_BAR,
        periods_per_year=365,
    )


def _metrics(values: tuple[float, ...], *, turnover: float, cost: float) -> dict[str, object]:
    metrics = evaluate_performance(
        _series(values),
        turnover_total=turnover,
        total_cost=cost,
    )
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


def _run(seed: int, *, candidate_shift: float = 0.0) -> LoadedCandidateRun:
    symbols = ("BTCUSDT", "ETHUSDT")
    base = {
        "cash": (0.0, 0.0, 0.0, 0.0),
        "constant_long": (0.01, -0.005, 0.002, 0.004),
        "constant_short": (-0.01, 0.005, -0.002, -0.004),
        "trend": (0.012, -0.003, 0.004, 0.005),
        "mean_reversion": (-0.004, 0.009, -0.002, 0.006),
        "ridge24": (0.008, 0.002, -0.001, 0.004),
        "lightgbm24": (0.009, 0.001, 0.0, 0.003),
        "ppo": (
            0.006 + 0.001 * seed,
            0.002,
            -0.001,
            0.004 + candidate_shift,
        ),
    }
    returns: dict[str, np.ndarray] = {}
    by_symbol: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(symbols):
        strategies: list[dict[str, object]] = []
        symbol_scale = 1.0 if symbol_index == 0 else 0.5
        for strategy_index, name in enumerate(STRATEGIES):
            values = tuple(
                float(value * symbol_scale + (candidate_shift if name != "cash" else 0.0))
                for value in base[name]
            )
            key = f"symbol_{symbol_index}_strategy_{strategy_index}"
            returns[key] = np.asarray(values, dtype=np.float64)
            turnover = 0.0 if name == "cash" else 1.0 + strategy_index * 0.1
            cost = turnover * 0.001
            strategies.append(
                {
                    "name": name,
                    "return_key": key,
                    "metrics": _metrics(values, turnover=turnover, cost=cost),
                }
            )
        by_symbol.append(
            {
                "symbol_index": symbol_index,
                "symbol": symbol,
                "strategies": strategies,
            }
        )
    return LoadedCandidateRun(
        root=Path(f"/synthetic/seed-{seed}"),
        summary={
            "schema_version": "lean_candidate_result_v1",
            "symbols": list(symbols),
            "candidate_config": {"ppo_seed": seed},
            "by_symbol": by_symbol,
        },
        returns=returns,
        provenance={},
    )


def _paired_payload(result) -> dict[str, object]:
    return {
        "excess_total_return": result.excess_total_return,
        "excess_log_return": result.excess_log_return,
        "mean_period_excess": result.mean_period_excess,
        "mean_period_simple_excess": result.mean_period_simple_excess,
        "p_value": result.p_value,
        "lower_ci": result.lower_ci,
        "upper_ci": result.upper_ci,
        "block_size": result.block_size,
    }


def test_analysis_requires_exact_symbol_strategy_matrix() -> None:
    run = _run(0)
    broken_summary = dict(run.summary)
    by_symbol = [dict(item) for item in run.summary["by_symbol"]]  # type: ignore[index]
    first = dict(by_symbol[0])
    first["strategies"] = list(first["strategies"])[:-1]  # type: ignore[arg-type]
    by_symbol[0] = first
    broken_summary["by_symbol"] = by_symbol
    broken = LoadedCandidateRun(
        root=run.root,
        summary=broken_summary,
        returns=run.returns,
        provenance=run.provenance,
    )

    with pytest.raises(ArtifactIntegrityError, match="matrix|strategy"):
        analyze_evidence_set({0: broken, 1: _run(1)}, n_bootstrap=100, bootstrap_seed=7)


def test_analysis_reuses_exact_paired_comparison_for_trend_vs_cash() -> None:
    runs = {0: _run(0), 1: _run(1)}
    payload = analyze_evidence_set(runs, n_bootstrap=100, bootstrap_seed=7)

    btc = payload["by_symbol"]["BTCUSDT"]  # type: ignore[index]
    expected = compare_paired_returns(
        _series(tuple(runs[0].returns["symbol_0_strategy_3"])),
        _series(tuple(runs[0].returns["symbol_0_strategy_0"])),
        n_bootstrap=100,
        seed=7,
    )
    assert btc["paired_vs_cash"]["trend"] == _paired_payload(expected)


def test_analysis_reuses_seed_robustness_for_ppo_vs_ridge24() -> None:
    runs = {0: _run(0), 1: _run(1)}
    payload = analyze_evidence_set(runs, n_bootstrap=100, bootstrap_seed=7)

    expected = summarize_seed_robustness(
        evaluation_label="BTCUSDT:ppo_vs_ridge24",
        seeds=(
            SeedEvaluation(
                seed=seed,
                returns=_series(tuple(run.returns["symbol_0_strategy_7"])),
                turnover_total=float(
                    run.summary["by_symbol"][0]["strategies"][7]["metrics"]["turnover_total"]  # type: ignore[index]
                ),
                total_cost=float(
                    run.summary["by_symbol"][0]["strategies"][7]["metrics"]["total_cost"]  # type: ignore[index]
                ),
            )
            for seed, run in runs.items()
        ),
        baseline=_series(tuple(runs[0].returns["symbol_0_strategy_5"])),
        n_bootstrap=100,
        bootstrap_seed=7,
    )
    assert payload["by_symbol"]["BTCUSDT"]["ppo_seed_robustness_vs_ridge24"] == asdict(expected)  # type: ignore[index]


def test_compare_evidence_sets_matches_ppo_by_seed_without_combined_p_value() -> None:
    baseline = {0: _run(0), 1: _run(1)}
    candidate = {0: _run(0, candidate_shift=0.002), 1: _run(1, candidate_shift=-0.001)}
    payload = compare_evidence_sets(
        baseline,
        candidate,
        n_bootstrap=100,
        bootstrap_seed=11,
    )
    ppo = payload["by_symbol"]["BTCUSDT"]["strategies"]["ppo"]  # type: ignore[index]

    expected = {}
    for seed in (0, 1):
        paired = compare_paired_returns(
            _series(tuple(candidate[seed].returns["symbol_0_strategy_7"])),
            _series(tuple(baseline[seed].returns["symbol_0_strategy_7"])),
            n_bootstrap=100,
            seed=11 + seed,
        )
        expected[str(seed)] = _paired_payload(paired)
    assert ppo["by_seed"] == expected
    aggregate = ppo["seed_aggregate"]
    assert "p_value" not in aggregate
    values = [item["excess_total_return"] for item in expected.values()]
    assert aggregate == {
        "positive_seed_count": sum(value > 0.0 for value in values),
        "negative_seed_count": sum(value < 0.0 for value in values),
        "zero_seed_count": sum(value == 0.0 for value in values),
        "median_excess_total_return": float(np.median(values)),
        "worst_excess_total_return": min(values),
        "best_excess_total_return": max(values),
    }


def test_cross_symbol_summary_is_descriptive_only() -> None:
    baseline = {0: _run(0), 1: _run(1)}
    candidate = {0: _run(0, candidate_shift=0.001), 1: _run(1, candidate_shift=0.001)}
    payload = compare_evidence_sets(
        baseline,
        candidate,
        n_bootstrap=100,
        bootstrap_seed=3,
    )

    summary = payload["cross_symbol"]["trend"]  # type: ignore[index]
    assert "p_value" not in summary
    assert "confidence_interval" not in summary
    assert summary["positive_symbol_count"] == 2
    assert summary["negative_symbol_count"] == 0
    assert summary["symbol_count"] == 2
    assert summary["worst_excess_total_return"] <= summary["median_excess_total_return"]
    assert summary["median_excess_total_return"] <= summary["best_excess_total_return"]
