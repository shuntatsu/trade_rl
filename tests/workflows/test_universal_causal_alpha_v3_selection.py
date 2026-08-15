from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v3 import (
    CausalAlphaV3FitConfig,
    CausalAlphaV3TargetConfig,
)
from trade_rl.workflows.universal_causal_alpha_selection import (
    CausalAlphaSelectionThresholds,
)
from trade_rl.workflows.universal_causal_alpha_v3_contracts import (
    CausalAlphaV3CandidateConfig,
    CausalAlphaV3EpisodeMetric,
)
from trade_rl.workflows.universal_causal_alpha_v3_selection import (
    CausalAlphaV3SelectionRejected,
    causal_alpha_v3_grid_digest,
    default_causal_alpha_v3_candidate_grid,
    load_causal_alpha_v3_selection_checkpoint,
    rank_causal_alpha_v3_candidates,
    write_causal_alpha_v3_selection_checkpoint_metric,
)


def _candidate(name: str, *, cadence: int = 16) -> CausalAlphaV3CandidateConfig:
    return CausalAlphaV3CandidateConfig(
        name=name,
        fit=CausalAlphaV3FitConfig(ridge_strength=0.1),
        target=CausalAlphaV3TargetConfig(
            target_magnitudes=(0.0, 0.1),
            uncertainty_multiplier=1.0,
            execution_cost_multiplier=1.0,
            edge_margin=0.0,
            alpha_rebalance_decisions=cadence,
            strong_reversal_threshold=2.0,
            max_target_delta=0.1,
        ),
    )


def _metric(
    candidate: CausalAlphaV3CandidateConfig,
    *,
    symbol: str = "BTCUSDT",
    net: float = 0.01,
    gross: float = 0.02,
) -> CausalAlphaV3EpisodeMetric:
    return CausalAlphaV3EpisodeMetric(
        candidate_digest=candidate.digest,
        symbol=symbol,
        episode_index=0,
        contract_digest=content_digest(f"contract:{symbol}"),
        gross_return=gross,
        net_return=net,
        turnover_per_day=0.3,
        total_execution_cost=10.0,
        trade_count=5,
        hard_risk_violation=False,
        unexplained_execution_rejection_count=0,
    )


def test_v3_ranking_uses_lower_tail_then_mean_net() -> None:
    first = _candidate("first", cadence=16)
    second = _candidate("second", cadence=32)
    thresholds = CausalAlphaSelectionThresholds()

    selected = rank_causal_alpha_v3_candidates(
        candidates=(first, second),
        metrics={
            first.digest: (_metric(first, net=0.01),),
            second.digest: (_metric(second, net=0.02),),
        },
        thresholds=thresholds,
        generator_code_digest=content_digest("generator"),
        sample_scope_digest=content_digest("samples"),
        holdout_episode_digests={"BTCUSDT": content_digest("holdout")},
    )

    assert selected.selected_candidate_digest == second.digest
    assert selected.grid_digest == causal_alpha_v3_grid_digest(
        (first, second), thresholds
    )


def test_default_v3_grid_is_unique_and_slow_rebalancing() -> None:
    candidates = default_causal_alpha_v3_candidate_grid()

    assert len(candidates) >= 8
    assert len({item.digest for item in candidates}) == len(candidates)
    assert min(item.target.alpha_rebalance_decisions for item in candidates) >= 16
    assert max(item.target.max_target_delta for item in candidates) <= 0.1


def test_v3_ranking_rejects_complete_unprofitable_grid() -> None:
    candidate = _candidate("loser")

    with pytest.raises(CausalAlphaV3SelectionRejected) as caught:
        rank_causal_alpha_v3_candidates(
            candidates=(candidate,),
            metrics={candidate.digest: (_metric(candidate, net=-0.01),)},
            thresholds=CausalAlphaSelectionThresholds(),
            generator_code_digest=content_digest("generator"),
            sample_scope_digest=content_digest("samples"),
            holdout_episode_digests={"BTCUSDT": content_digest("holdout")},
        )

    assert caught.value.candidates[0].rejection_reasons == ("negative_mean_net_return",)


def test_v3_checkpoint_is_resumable_and_identity_bound(tmp_path: Path) -> None:
    candidate = _candidate("base")
    metric = _metric(candidate)
    thresholds = CausalAlphaSelectionThresholds()
    grid = causal_alpha_v3_grid_digest((candidate,), thresholds)
    generator = content_digest("generator")
    samples = content_digest("samples")
    checkpoint = tmp_path / "checkpoint.jsonl"
    write_causal_alpha_v3_selection_checkpoint_metric(
        checkpoint,
        metric,
        grid_digest=grid,
        generator_code_digest=generator,
        sample_scope_digest=samples,
    )

    loaded = load_causal_alpha_v3_selection_checkpoint(
        checkpoint,
        expected_grid_digest=grid,
        expected_generator_code_digest=generator,
        expected_sample_scope_digest=samples,
    )

    assert loaded == {candidate.digest: (metric,)}
    with pytest.raises(ValueError, match="grid digest mismatch"):
        load_causal_alpha_v3_selection_checkpoint(
            checkpoint,
            expected_grid_digest=content_digest("other-grid"),
            expected_generator_code_digest=generator,
            expected_sample_scope_digest=samples,
        )


def test_v3_checkpoint_rejects_duplicate_scope(tmp_path: Path) -> None:
    candidate = _candidate("base")
    metric = _metric(candidate)
    thresholds = CausalAlphaSelectionThresholds()
    grid = causal_alpha_v3_grid_digest((candidate,), thresholds)
    checkpoint = tmp_path / "checkpoint.jsonl"
    for _ in range(2):
        write_causal_alpha_v3_selection_checkpoint_metric(
            checkpoint,
            metric,
            grid_digest=grid,
            generator_code_digest=content_digest("generator"),
            sample_scope_digest=content_digest("samples"),
        )

    with pytest.raises(ValueError, match="duplicated"):
        load_causal_alpha_v3_selection_checkpoint(
            checkpoint,
            expected_grid_digest=grid,
            expected_generator_code_digest=content_digest("generator"),
            expected_sample_scope_digest=content_digest("samples"),
        )
