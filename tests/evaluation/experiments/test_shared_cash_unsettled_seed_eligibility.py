from __future__ import annotations

from dataclasses import replace

import numpy as np

from tests.evaluation.experiments.test_analysis import _run
from trade_rl.evaluation.experiments import ExperimentComparison
from trade_rl.evaluation.experiments.analysis import compare_evidence_sets
from trade_rl.evaluation.experiments.contracts import (
    PPO_HOLDING_DURATION_HORIZONS,
    ExperimentDecisionKind,
)
from trade_rl.evaluation.experiments.protocols import (
    ppo_holding_expected_decision,
    ppo_holding_winner_digest,
    ppo_shared_cash_holding_metrics,
)
from trade_rl.evaluation.runs.artifact import LoadedCandidateRun

_SEEDS = (2, 5, 9, 13, 17)
_SCHEMA = "controlled_evidence_comparison_v4"


def _with_shared_cash_portfolio(
    run: LoadedCandidateRun,
    *,
    values: tuple[float, ...],
    unsettled: bool = False,
) -> LoadedCandidateRun:
    summary = dict(run.summary)
    wealth = 1.0
    peak = 1.0
    maximum_drawdown = 0.0
    for value in values:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        maximum_drawdown = max(maximum_drawdown, 1.0 - wealth / peak)

    portfolio: dict[str, object] = {
        "name": "ppo",
        "return_key": "shared_cash_ppo",
        "metrics": {
            "total_return": wealth - 1.0,
            "max_drawdown": maximum_drawdown,
            "n_periods": len(values),
            "return_kind": "base_bar",
            "periods_per_year": 8_760,
        },
        "terminal_settlement_complete": not unsettled,
        "final_quantities": [0.25, 0.0] if unsettled else [0.0, 0.0],
        "active_order_remainders": [],
        "ledger_evidence": {
            "payload": {
                "schema_version": "shared_cash_replay_ledger_v3",
                "final_max_drawdown": maximum_drawdown,
            }
        },
    }
    summary["shared_cash_ppo"] = portfolio
    returns = dict(run.returns)
    returns["shared_cash_ppo"] = np.asarray(values, dtype=np.float64)
    return replace(run, summary=summary, returns=returns)


def _comparison(
    baseline: dict[int, LoadedCandidateRun],
    candidate: dict[int, LoadedCandidateRun],
) -> ExperimentComparison:
    payload = compare_evidence_sets(
        baseline,
        candidate,
        n_bootstrap=32,
        bootstrap_seed=13,
        schema_version=_SCHEMA,
    )
    return ExperimentComparison(
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


def test_one_unsettled_h0_seed_blocks_every_shared_cash_horizon() -> None:
    baseline = {
        seed: _with_shared_cash_portfolio(
            _run(seed),
            values=(0.0, 0.0, 0.0, 0.0),
            unsettled=(seed == 9),
        )
        for seed in _SEEDS
    }
    candidate = {
        seed: _with_shared_cash_portfolio(
            _run(seed, candidate_shift=0.01),
            values=(0.03, 0.01, 0.0, 0.0),
        )
        for seed in _SEEDS
    }

    eligible_arms: list[tuple[int, float, str]] = []
    for horizon in PPO_HOLDING_DURATION_HORIZONS:
        comparison = _comparison(baseline, candidate)
        metrics = ppo_shared_cash_holding_metrics(
            comparison,
            expected_seeds=_SEEDS,
        )

        assert not metrics.terminal_settlement_complete
        assert not metrics.eligible
        assert (
            ppo_holding_expected_decision(metrics)
            is ExperimentDecisionKind.KEEP_BASELINE
        )
        if metrics.eligible:
            eligible_arms.append((horizon, metrics.score, str(horizon)))

    assert ppo_holding_winner_digest(eligible_arms) is None
