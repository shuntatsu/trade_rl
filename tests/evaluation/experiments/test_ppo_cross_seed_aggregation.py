from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from statistics import mean, median

import numpy as np
import pytest

from tests.evaluation.experiments.test_analysis import _run
from tests.evaluation.experiments.test_evidence import _config
from tests.evaluation.experiments.test_workflow import _with_baseline
from trade_rl.artifacts.canonical import to_json_value
from trade_rl.evaluation.experiments import (
    ControlledFactor,
    ControlledVerificationStatus,
    ExperimentComparison,
    define_experiment,
    inspect_study,
    run_experiment,
    verify_experiment,
)
from trade_rl.evaluation.experiments.analysis import compare_evidence_sets
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.inspection import (
    _experiment_dir,
    _experiment_state,
    _find_evidence,
    _reconstruct,
)
from trade_rl.evaluation.experiments.protocols import (
    ppo_holding_metrics,
    ppo_shared_cash_holding_metrics,
)
from trade_rl.evaluation.experiments.store import StudyStore
from trade_rl.evaluation.runs.artifact import LoadedCandidateRun

LEGACY_SCHEMA = "controlled_evidence_comparison_v1"
CURRENT_SCHEMA = "controlled_evidence_comparison_v2"
HOLDING_PROTOCOL_SCHEMA = "controlled_evidence_comparison_v3"
SHARED_CASH_HOLDING_PROTOCOL_SCHEMA = "controlled_evidence_comparison_v4"


def _with_shared_cash_portfolio(
    run: LoadedCandidateRun,
    values: tuple[float, ...],
    *,
    settled: bool = True,
) -> LoadedCandidateRun:
    summary = to_json_value(run.summary)
    assert isinstance(summary, dict)
    wealth = 1.0
    peak = 1.0
    maximum_drawdown = 0.0
    for value in values:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        maximum_drawdown = max(maximum_drawdown, 1.0 - wealth / peak)
    summary["shared_cash_ppo"] = {
        "name": "ppo",
        "return_key": "shared_cash_ppo",
        "metrics": {
            "total_return": wealth - 1.0,
            "max_drawdown": maximum_drawdown,
            "n_periods": len(values),
            "return_kind": "base_bar",
            "periods_per_year": 8_760,
        },
        "terminal_settlement_complete": settled,
        "ledger_evidence": {
            "payload": {
                "schema_version": "shared_cash_replay_ledger_v3",
                "final_max_drawdown": maximum_drawdown,
            }
        },
    }
    returns = dict(run.returns)
    returns["shared_cash_ppo"] = np.asarray(values, dtype=np.float64)
    return LoadedCandidateRun(
        root=run.root,
        summary=summary,
        returns=returns,
        provenance=run.provenance,
    )


def _with_ppo_execution_metrics(
    run: LoadedCandidateRun, *, seed: int
) -> LoadedCandidateRun:
    summary = to_json_value(run.summary)
    assert isinstance(summary, dict)
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


def _with_terminal_settlement(
    run: LoadedCandidateRun,
    *,
    complete: bool = True,
) -> LoadedCandidateRun:
    summary = to_json_value(run.summary)
    assert isinstance(summary, dict)
    summary["schema_version"] = "lean_candidate_result_v6"
    summary["evaluation"] = {"expected_periods": 4}
    by_symbol = summary.get("by_symbol")
    assert isinstance(by_symbol, list)
    for symbol_entry in by_symbol:
        assert isinstance(symbol_entry, dict)
        strategies = symbol_entry.get("strategies")
        assert isinstance(strategies, list)
        ppo = next(
            strategy
            for strategy in strategies
            if isinstance(strategy, dict) and strategy.get("name") == "ppo"
        )
        ppo["terminal_settlement_complete"] = complete
    return LoadedCandidateRun(
        root=run.root,
        summary=summary,
        returns=run.returns,
        provenance=run.provenance,
    )


def _ppo_metric(run: LoadedCandidateRun, symbol_index: int, field: str) -> float:
    by_symbol = run.summary.get("by_symbol")
    assert isinstance(by_symbol, Sequence)
    assert not isinstance(by_symbol, (str, bytes, bytearray))
    symbol_entry = by_symbol[symbol_index]
    assert isinstance(symbol_entry, Mapping)
    strategies = symbol_entry.get("strategies")
    assert isinstance(strategies, Sequence)
    assert not isinstance(strategies, (str, bytes, bytearray))
    ppo = next(
        strategy
        for strategy in strategies
        if isinstance(strategy, Mapping) and strategy.get("name") == "ppo"
    )
    metrics = ppo.get("metrics")
    assert isinstance(metrics, Mapping)
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


def test_holding_protocol_score_aggregates_symbols_within_each_seed_first() -> None:
    seeds = (0, 1, 2, 3, 4)
    candidate_shifts = (0.02, -0.03, 0.01, 0.04, -0.015)
    baseline = {seed: _with_terminal_settlement(_run(seed)) for seed in seeds}
    candidate = {
        seed: _with_terminal_settlement(
            _run(seed, candidate_shift=candidate_shifts[seed])
        )
        for seed in seeds
    }

    payload = compare_evidence_sets(
        baseline,
        candidate,
        n_bootstrap=32,
        bootstrap_seed=13,
        schema_version=HOLDING_PROTOCOL_SCHEMA,
    )

    symbols = tuple(candidate[seeds[0]].summary["symbols"])
    per_seed_candidate_means = {
        seed: mean(
            _ppo_metric(candidate[seed], symbol_index, "total_return")
            for symbol_index in range(len(symbols))
        )
        for seed in seeds
    }
    by_symbol = payload["by_symbol"]
    per_seed_excess_means = {
        seed: mean(
            by_symbol[symbol]["strategies"]["ppo"]["by_seed"][str(seed)][
                "excess_total_return"
            ]
            for symbol in symbols
        )
        for seed in seeds
    }
    expected_worst_drawdown = max(
        _ppo_metric(candidate[seed], symbol_index, "max_drawdown")
        for seed in seeds
        for symbol_index in range(len(symbols))
    )
    expected_worst_baseline_drawdown = max(
        _ppo_metric(baseline[seed], symbol_index, "max_drawdown")
        for seed in seeds
        for symbol_index in range(len(symbols))
    )

    summary = payload["cross_seed"]["ppo"]
    assert payload["schema_version"] == HOLDING_PROTOCOL_SCHEMA
    assert summary["median_candidate_total_return"] == median(
        per_seed_candidate_means.values()
    )
    assert summary["median_excess_total_return"] == median(
        per_seed_excess_means.values()
    )
    assert summary["worst_candidate_max_drawdown"] == expected_worst_drawdown
    assert summary["worst_baseline_max_drawdown"] == expected_worst_baseline_drawdown
    assert summary["worst_account_max_drawdown"] == max(
        expected_worst_drawdown,
        expected_worst_baseline_drawdown,
    )
    assert summary["symbol_count"] == len(symbols)
    assert summary["baseline_terminal_settlement_complete_account_count"] == len(
        seeds
    ) * len(symbols)
    assert summary["candidate_terminal_settlement_complete_account_count"] == len(
        seeds
    ) * len(symbols)
    assert summary["by_seed"] == {
        str(seed): {
            "mean_candidate_total_return": per_seed_candidate_means[seed],
            "mean_excess_total_return": per_seed_excess_means[seed],
            "worst_candidate_max_drawdown": max(
                _ppo_metric(candidate[seed], symbol_index, "max_drawdown")
                for symbol_index in range(len(symbols))
            ),
            "worst_baseline_max_drawdown": max(
                _ppo_metric(baseline[seed], symbol_index, "max_drawdown")
                for symbol_index in range(len(symbols))
            ),
            "worst_account_max_drawdown": max(
                max(
                    _ppo_metric(candidate[seed], symbol_index, "max_drawdown")
                    for symbol_index in range(len(symbols))
                ),
                max(
                    _ppo_metric(baseline[seed], symbol_index, "max_drawdown")
                    for symbol_index in range(len(symbols))
                ),
            ),
            "symbol_count": len(symbols),
            "baseline_terminal_settlement_complete_account_count": len(symbols),
            "candidate_terminal_settlement_complete_account_count": len(symbols),
        }
        for seed in seeds
    }


def test_shared_cash_protocol_compares_combined_portfolio_per_seed() -> None:
    seeds = (0, 1, 2, 3, 4)
    candidate_paths = (
        (0.08, -0.30, 0.20, 0.0),
        (0.03, 0.02, 0.0, 0.0),
        (0.01, 0.0, 0.0, 0.0),
        (0.10, -0.10, 0.0, 0.0),
        (0.02, 0.03, 0.0, 0.0),
    )
    baseline = {
        seed: _with_shared_cash_portfolio(_run(seed), (0.0, 0.0, 0.0, 0.0))
        for seed in seeds
    }
    candidate = {
        seed: _with_shared_cash_portfolio(
            _run(seed, candidate_shift=0.10),
            candidate_paths[seed],
        )
        for seed in seeds
    }

    payload = compare_evidence_sets(
        baseline,
        candidate,
        n_bootstrap=32,
        bootstrap_seed=13,
        schema_version=SHARED_CASH_HOLDING_PROTOCOL_SCHEMA,
    )

    portfolio = payload["cross_seed"]["shared_cash_ppo"]
    assert payload["schema_version"] == SHARED_CASH_HOLDING_PROTOCOL_SCHEMA
    assert portfolio["seed_count"] == 5
    assert portfolio["by_seed"]["0"]["candidate_max_drawdown"] == pytest.approx(0.30)
    assert portfolio["worst_candidate_max_drawdown"] == pytest.approx(0.30)
    assert portfolio["candidate_terminal_settlement_complete_seed_count"] == 5
    assert portfolio["baseline_terminal_settlement_complete_seed_count"] == 5
    assert portfolio["by_seed"]["0"]["candidate_total_return"] == pytest.approx(
        1.08 * 0.70 * 1.20 - 1.0
    )
    factor_digest = payload["analysis_digest"]
    comparison = ExperimentComparison(
        study_digest="a" * 64,
        experiment_digest="b" * 64,
        baseline_evidence_digest="c" * 64,
        candidate_evidence_digest="d" * 64,
        verification_digest="e" * 64,
        baseline_analysis_digest="f" * 64,
        candidate_analysis_digest="1" * 64,
        factor_effect_digest=factor_digest,
        factor_effect=payload,
    )
    metrics = ppo_shared_cash_holding_metrics(
        comparison,
        expected_seeds=seeds,
    )
    assert metrics.score == pytest.approx(0.01)
    assert metrics.median_excess_return == pytest.approx(0.01)
    assert metrics.worst_max_drawdown == pytest.approx(0.30)
    assert not metrics.eligible


def test_shared_cash_selector_accepts_exact_twenty_percent_drawdown() -> None:
    seeds = (0, 1, 2, 3, 4)
    baseline = {
        seed: _with_shared_cash_portfolio(_run(seed), (0.0, 0.0, 0.0, 0.0))
        for seed in seeds
    }
    candidate = {
        seed: _with_shared_cash_portfolio(
            _run(seed, candidate_shift=0.10),
            (0.20, -0.20, 0.40, 0.0),
        )
        for seed in seeds
    }
    payload = compare_evidence_sets(
        baseline,
        candidate,
        n_bootstrap=32,
        bootstrap_seed=13,
        schema_version=SHARED_CASH_HOLDING_PROTOCOL_SCHEMA,
    )
    comparison = ExperimentComparison(
        study_digest="a" * 64,
        experiment_digest="b" * 64,
        baseline_evidence_digest="c" * 64,
        candidate_evidence_digest="d" * 64,
        verification_digest="e" * 64,
        baseline_analysis_digest="f" * 64,
        candidate_analysis_digest="1" * 64,
        factor_effect_digest=payload["analysis_digest"],
        factor_effect=payload,
    )

    metrics = ppo_shared_cash_holding_metrics(comparison, expected_seeds=seeds)

    assert metrics.worst_max_drawdown == pytest.approx(0.20)
    assert metrics.median_excess_return > 0.0
    assert metrics.terminal_settlement_complete
    assert metrics.eligible


@pytest.mark.parametrize("candidate_portfolio_return", (-0.05, 0.0))
def test_shared_cash_selector_rejects_nonpositive_absolute_profit(
    candidate_portfolio_return: float,
) -> None:
    seeds = (0, 1, 2, 3, 4)
    baseline = {
        seed: _with_shared_cash_portfolio(_run(seed), (-0.10, 0.0, 0.0, 0.0))
        for seed in seeds
    }
    candidate = {
        seed: _with_shared_cash_portfolio(
            _run(seed, candidate_shift=0.10),
            (candidate_portfolio_return, 0.0, 0.0, 0.0),
        )
        for seed in seeds
    }
    payload = compare_evidence_sets(
        baseline,
        candidate,
        n_bootstrap=32,
        bootstrap_seed=13,
        schema_version=SHARED_CASH_HOLDING_PROTOCOL_SCHEMA,
    )
    comparison = ExperimentComparison(
        study_digest="a" * 64,
        experiment_digest="b" * 64,
        baseline_evidence_digest="c" * 64,
        candidate_evidence_digest="d" * 64,
        verification_digest="e" * 64,
        baseline_analysis_digest="f" * 64,
        candidate_analysis_digest="1" * 64,
        factor_effect_digest=payload["analysis_digest"],
        factor_effect=payload,
    )

    metrics = ppo_shared_cash_holding_metrics(comparison, expected_seeds=seeds)

    assert metrics.score == pytest.approx(candidate_portfolio_return)
    assert metrics.median_excess_return == pytest.approx(
        candidate_portfolio_return + 0.10
    )
    assert metrics.worst_max_drawdown <= 0.20
    assert metrics.terminal_settlement_complete
    assert not metrics.eligible


def test_holding_protocol_records_incomplete_terminal_accounts() -> None:
    seeds = (0, 1, 2, 3, 4)
    shifts = (0.02, -0.03, 0.01, 0.04, -0.015)
    baseline = {seed: _with_terminal_settlement(_run(seed)) for seed in seeds}
    candidate = {
        seed: _with_terminal_settlement(
            _run(seed, candidate_shift=shifts[seed]),
            complete=seed != 2,
        )
        for seed in seeds
    }

    payload = compare_evidence_sets(
        baseline,
        candidate,
        n_bootstrap=32,
        bootstrap_seed=13,
        schema_version=HOLDING_PROTOCOL_SCHEMA,
    )

    ppo = payload["cross_seed"]["ppo"]
    assert ppo["candidate_terminal_settlement_complete_account_count"] == 8
    assert (
        ppo["by_seed"]["2"]["candidate_terminal_settlement_complete_account_count"] == 0
    )
    factor_digest = payload["analysis_digest"]
    assert isinstance(factor_digest, str)
    comparison = ExperimentComparison(
        study_digest="a" * 64,
        experiment_digest="b" * 64,
        baseline_evidence_digest="c" * 64,
        candidate_evidence_digest="d" * 64,
        verification_digest="e" * 64,
        baseline_analysis_digest="f" * 64,
        candidate_analysis_digest="1" * 64,
        factor_effect_digest=factor_digest,
        factor_effect=payload,
    )
    metrics = ppo_holding_metrics(
        comparison,
        expected_symbols=("BTCUSDT", "ETHUSDT"),
        expected_seeds=seeds,
    )
    assert not metrics.terminal_settlement_complete
    assert not metrics.eligible


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


def test_inspection_replays_persisted_v1_factor_effect_with_legacy_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset_root, snapshot = _with_baseline(tmp_path, monkeypatch)
    assert snapshot.baseline is not None
    definition = define_experiment(
        root,
        dataset_root=dataset_root,
        hypothesis="More PPO training changes only PPO evidence.",
        factor=ControlledFactor.PPO_TRAINING_BUDGET,
        candidate_config=replace(_config(), ppo_total_timesteps=64),
        baseline_evidence_digest=snapshot.baseline.fingerprint,
    )
    run_experiment(root, 1, dataset_root=dataset_root)
    verification = verify_experiment(root, 1)
    assert verification.status is ControlledVerificationStatus.CONTROLLED

    store = StudyStore(root)
    state = _reconstruct(store)
    experiment = _experiment_state(state, 1)
    assert experiment.candidate is not None
    assert experiment.candidate_analysis is not None
    assert experiment.verification is not None
    baseline, baseline_analysis = _find_evidence(
        state, definition.baseline_evidence_digest
    )
    factor_effect = compare_evidence_sets(
        baseline.runs,
        experiment.candidate.runs,
        n_bootstrap=state.plan.n_bootstrap,
        bootstrap_seed=state.plan.bootstrap_seed,
        schema_version=LEGACY_SCHEMA,
    )
    factor_digest = factor_effect.get("analysis_digest")
    assert isinstance(factor_digest, str)
    comparison = ExperimentComparison(
        study_digest=state.plan.digest,
        experiment_digest=definition.digest,
        baseline_evidence_digest=baseline.evidence.fingerprint,
        candidate_evidence_digest=experiment.candidate.evidence.fingerprint,
        verification_digest=experiment.verification.digest,
        baseline_analysis_digest=baseline_analysis.analysis_digest,
        candidate_analysis_digest=experiment.candidate_analysis.analysis_digest,
        factor_effect_digest=factor_digest,
        factor_effect=factor_effect,
    )
    store.publish_json_once(
        _experiment_dir(1) / "comparison.json",
        comparison.to_payload(),
    )

    rebuilt = inspect_study(root)
    assert rebuilt.experiment_sequences == (1,)
