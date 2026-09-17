from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.evaluation.experiments.ppo_interleaved_evaluation import (
    PPOReturnPathEvidence,
    PPOSeedEvidence,
    PPOSymbolEvidence,
    canonical_ppo_interleaved_evaluator_spec,
    return_path_sha256,
)
from trade_rl.evaluation.metrics import compound_return


def _path(name: str) -> PPOReturnPathEvidence:
    values = (0.0, 0.0)
    return PPOReturnPathEvidence(
        strategy_name=name,
        returns=values,
        return_sha256=return_path_sha256(values),
        total_return=0.0,
        total_cost=0.0,
        turnover_total=0.0,
        max_drawdown=0.0,
        termination_count=0,
        termination_reasons=(),
        n_periods=2,
        periods_per_year=8_760,
    )


def _seed_evidence(*, arm: str) -> PPOSeedEvidence:
    spec = canonical_ppo_interleaved_evaluator_spec()
    ppo = _path("ppo")
    cash = _path("cash")
    long = _path("constant_long")
    short = _path("constant_short")
    rows = tuple(
        PPOSymbolEvidence(
            symbol_index=index,
            symbol=symbol,
            ppo=ppo,
            cash=cash,
            constant_long=long,
            constant_short=short,
            ppo_returns_equal_cash=True,
            ppo_returns_equal_constant_long=True,
            ppo_returns_equal_constant_short=True,
        )
        for index, symbol in enumerate(spec.symbols)
    )
    baseline = arm == "baseline"
    return PPOSeedEvidence(
        spec_digest=spec.digest,
        protocol_head=spec.protocol_head,
        protocol_digest=spec.protocol_digest,
        implementation_head=spec.implementation_head,
        successor_bundle_run_id=spec.successor_bundle_run_id,
        successor_bundle_artifact_id=spec.successor_bundle_artifact_id,
        successor_bundle_artifact_digest=spec.successor_bundle_artifact_digest,
        dataset_id=spec.dataset_id,
        dataset_artifact_digest=spec.dataset_artifact_digest,
        study_digest=spec.study_digest,
        execution_overlay=spec.execution_overlay,
        symbols=spec.symbols,
        feature_names=spec.feature_names,
        feature_indices=spec.feature_indices,
        fit_symbol_names=spec.fit_symbol_names,
        fit_cutoff=spec.fit_cutoff,
        evaluation_start=spec.evaluation_start,
        evaluation_stop_exclusive=spec.evaluation_stop_exclusive,
        gross_budget=spec.gross_budget,
        initial_capital=spec.initial_capital,
        arm="baseline" if baseline else "candidate",
        seed=0,
        training_layout=(
            spec.baseline_training_layout if baseline else spec.candidate_training_layout
        ),
        rollout_steps_per_env=(
            spec.baseline_rollout_steps_per_env
            if baseline
            else spec.candidate_rollout_steps_per_env
        ),
        caller_total_timesteps=spec.ppo_total_timesteps,
        realized_num_timesteps=100_352 if baseline else 101_760,
        by_symbol=rows,
    )


def test_return_path_rejects_bool_or_string_aliases_before_normalization() -> None:
    with pytest.raises(ValueError, match="returns"):
        return_path_sha256((True, 0.0))
    with pytest.raises(ValueError, match="returns"):
        return_path_sha256(("0.0", 0.0))

    values = (True, 0.0)
    with pytest.raises(ValueError, match="returns"):
        PPOReturnPathEvidence(
            strategy_name="ppo",
            returns=values,  # type: ignore[arg-type]
            return_sha256=return_path_sha256((1.0, 0.0)),
            total_return=compound_return((1.0, 0.0)),
            total_cost=0.0,
            turnover_total=0.0,
            max_drawdown=0.0,
            termination_count=0,
            termination_reasons=(),
            n_periods=2,
            periods_per_year=8_760,
        )


def test_seed_evidence_rejects_float_aliases_for_integer_authorities() -> None:
    baseline = _seed_evidence(arm="baseline")
    with pytest.raises(ValueError, match="caller_total_timesteps"):
        replace(baseline, caller_total_timesteps=100_000.0)  # type: ignore[arg-type]

    candidate = _seed_evidence(arm="candidate")
    with pytest.raises(ValueError, match="rollout_steps_per_env"):
        replace(candidate, rollout_steps_per_env=384.0)  # type: ignore[arg-type]
