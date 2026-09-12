from __future__ import annotations

from copy import deepcopy
from statistics import median

import pytest

from tests.evaluation.experiments.test_analysis import _run
from trade_rl.evaluation.experiments.analysis import compare_evidence_sets
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.runs.artifact import LoadedCandidateRun

LEGACY_SCHEMA = "controlled_evidence_comparison_v1"
CURRENT_SCHEMA = "controlled_evidence_comparison_v2"


def _with_ppo_execution_metrics(
    run: LoadedCandidateRun, *, seed: int
) -> LoadedCandidateRun:
    summary = deepcopy(run.summary)
    by_symbol = summary.get("by_symbol")
    assert isinstance(by_symbol, list)
    turnover_by_symbol = (
        (0.0, 0.0, 100.0, 100.0),
        (1.0, 2.0, 3.0, 4.0),
    )
    for symbol_index, symbol_entry in enumerate(by_symbol):
        assert isinstance(symbol_entry, dict)
        strategies = symbol_entry.get("strategies")
        assert isinstance(strategies, list)
        ppo = next(
            strategy
            for strategy in strategies
            if isinstance(strategy, dict) and strategy.get("name") == "ppo"
        )
        metrics = ppo.get("metrics")
        assert isinstance(metrics, dict)
        turnover = turnover_by_symbol[symbol_index][seed]
        metrics["turnover_total"] = turnover
        metrics["total_cost"] = turnover * 0.125 + float(symbol_index)
    return LoadedCandidateRun(
        root=run.root,
        summary=summary,
        returns=run.returns,
        provenance=run.provenance,
    )


def _ppo_metric(run: LoadedCandidateRun, symbol_index: int, field: str) -> float:
    by_symbol = run.summary.get("by_symbol")
    assert isinstance(by_symbol, list)
    symbol_entry = by_symbol[symbol_index]
    assert isinstance(symbol_entry, dict)
    strategies = symbol_entry.get("strategies")
    assert isinstance(strategies, list)
    ppo = next(
        strategy
        for strategy in strategies
        if isinstance(strategy, dict) and strategy.get("name") == "ppo"
    )
    metrics = ppo.get("metrics")
    assert isinstance(metrics, dict)
    value = metrics.get(field)
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value)


def _two_stage_candidate_oracle(
    candidate: dict[int, LoadedCandidateRun],
) -> dict[str, float]:
    seeds = tuple(sorted(candidate))
    symbol_count = 2
    per_symbol_total_return: list[float] = []
    per_symbol_max_drawdown: list[float] = []
    per_symbol_turnover: list[float] = []
    per_symbol_total_cost: list[float] = []
    for symbol_index in range(symbol_count):
        per_symbol_total_return.append(
            float(
                median(
                    _ppo_metric(candidate[seed], symbol_index, "total_return")
                    for seed in seeds
                )
            )
        )
        per_symbol_max_drawdown.append(
            max(
                _ppo_metric(candidate[seed], symbol_index, "max_drawdown")
                for seed in seeds
            )
        )
        per_symbol_turnover.append(
            float(
                median(
                    _ppo_metric(candidate[seed], symbol_index, "turnover_total")
                    for seed in seeds
                )
            )
        )
        per_symbol_total_cost.append(
            float(
                median(
                    _ppo_metric(candidate[seed], symbol_index, "total_cost")
                    for seed in seeds
                )
            )
        )
    return {
        "median_candidate_total_return": float(median(per_symbol_total_return)),
        "worst_candidate_max_drawdown": max(per_symbol_max_drawdown),
        "median_candidate_turnover": float(median(per_symbol_turnover)),
        "median_candidate_total_cost": float(median(per_symbol_total_cost)),
    }


def _legacy_first_seed_oracle(
    candidate: dict[int, LoadedCandidateRun],
) -> dict[str, float]:
    first_seed = min(candidate)
    returns = [
        _ppo_metric(candidate[first_seed], index, "total_return") for index in range(2)
    ]
    drawdowns = [
        _ppo_metric(candidate[first_seed], index, "max_drawdown") for index in range(2)
    ]
    turnovers = [
        _ppo_metric(candidate[first_seed], index, "turnover_total")
        for index in range(2)
    ]
    costs = [
        _ppo_metric(candidate[first_seed], index, "total_cost") for index in range(2)
    ]
    return {
        "median_candidate_total_return": float(median(returns)),
        "worst_candidate_max_drawdown": max(drawdowns),
        "median_candidate_turnover": float(median(turnovers)),
        "median_candidate_total_cost": float(median(costs)),
    }


@pytest.mark.parametrize("seeds", ((0, 1, 2), (0, 1, 2, 3)))
def test_ppo_cross_symbol_candidate_metrics_use_equal_symbol_two_stage_seed_aggregation(
    seeds: tuple[int, ...],
) -> None:
    baseline = {seed: _run(seed) for seed in seeds}
    candidate = {
        seed: _with_ppo_execution_metrics(
            _run(seed, candidate_shift=0.001 * (seed + 1)),
            seed=seed,
        )
        for seed in seeds
    }

    payload = compare_evidence_sets(
        baseline,
        candidate,
        n_bootstrap=32,
        bootstrap_seed=13,
    )

    assert payload["schema_version"] == CURRENT_SCHEMA
    summary = payload["cross_symbol"]["ppo"]  # type: ignore[index]
    expected = _two_stage_candidate_oracle(candidate)
    for field, value in expected.items():
        assert summary[field] == value


def test_ppo_cross_symbol_aggregation_is_invariant_to_seed_mapping_insertion_order() -> (
    None
):
    seeds = (0, 1, 2, 3)
    baseline = {seed: _run(seed) for seed in seeds}
    candidate = {
        seed: _with_ppo_execution_metrics(
            _run(seed, candidate_shift=0.001 * (seed + 1)),
            seed=seed,
        )
        for seed in seeds
    }

    forward = compare_evidence_sets(
        baseline,
        candidate,
        n_bootstrap=32,
        bootstrap_seed=13,
    )
    reverse = compare_evidence_sets(
        dict(reversed(tuple(baseline.items()))),
        dict(reversed(tuple(candidate.items()))),
        n_bootstrap=32,
        bootstrap_seed=13,
    )

    assert reverse == forward


def test_explicit_legacy_v1_reproduces_first_seed_candidate_metric_semantics() -> None:
    seeds = (0, 1, 2)
    baseline = {seed: _run(seed) for seed in seeds}
    candidate = {
        seed: _with_ppo_execution_metrics(
            _run(seed, candidate_shift=0.001 * (seed + 1)),
            seed=seed,
        )
        for seed in seeds
    }

    payload = compare_evidence_sets(
        baseline,
        candidate,
        n_bootstrap=32,
        bootstrap_seed=13,
        schema_version=LEGACY_SCHEMA,
    )

    assert payload["schema_version"] == LEGACY_SCHEMA
    summary = payload["cross_symbol"]["ppo"]  # type: ignore[index]
    expected = _legacy_first_seed_oracle(candidate)
    for field, value in expected.items():
        assert summary[field] == value


def test_unknown_factor_effect_schema_is_rejected() -> None:
    baseline = {0: _run(0), 1: _run(1)}
    candidate = {
        0: _run(0, candidate_shift=0.001),
        1: _run(1, candidate_shift=0.001),
    }

    with pytest.raises(ArtifactIntegrityError, match="schema"):
        compare_evidence_sets(
            baseline,
            candidate,
            n_bootstrap=32,
            bootstrap_seed=13,
            schema_version="controlled_evidence_comparison_v999",
        )
