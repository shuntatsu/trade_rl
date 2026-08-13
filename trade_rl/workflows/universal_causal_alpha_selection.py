"""Train-only candidate grid and ranking for the Universal causal alpha teacher."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaControllerConfig,
    CausalAlphaHorizonMix,
    CausalAlphaRidgeConfig,
    causal_alpha_target_path,
    combine_causal_alpha_predictions,
)
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaCandidateConfig,
    CausalAlphaCandidateEpisodeMetrics,
    CausalAlphaCandidateEvidence,
    CausalAlphaSelectionEvidence,
    CausalAlphaSymbolSamples,
)
from trade_rl.workflows.universal_causal_alpha_fitting import (
    fit_expanding_causal_alpha_models,
)


def _candidate(
    *,
    name: str,
    ridge_strength: float,
    horizon_mix: CausalAlphaHorizonMix,
    score_scale: float,
    entry_threshold: float,
    exit_threshold: float,
    no_trade_band: float,
    max_target_delta: float,
) -> CausalAlphaCandidateConfig:
    return CausalAlphaCandidateConfig(
        name=name,
        ridge=CausalAlphaRidgeConfig(ridge_strength=ridge_strength),
        controller=CausalAlphaControllerConfig(
            horizon_mix=horizon_mix,
            score_scale=score_scale,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            no_trade_band=no_trade_band,
            max_target_delta=max_target_delta,
        ),
    )


def default_causal_alpha_candidate_grid(
    risk_config: PreTradeRiskConfig,
) -> tuple[CausalAlphaCandidateConfig, ...]:
    """Return the maintained bounded one-factor-at-a-time causal teacher grid."""

    if not isinstance(risk_config, PreTradeRiskConfig):
        raise TypeError("causal alpha default grid requires PreTradeRiskConfig")
    no_trade = float(risk_config.no_trade_band)
    max_delta = min(0.25, float(risk_config.max_abs_weight))
    if max_delta <= 0.0:
        raise ValueError("causal alpha max target delta cannot be resolved")
    base = dict(
        ridge_strength=0.01,
        horizon_mix=CausalAlphaHorizonMix.EQUAL,
        score_scale=50.0,
        entry_threshold=0.003,
        exit_threshold=0.001,
        no_trade_band=no_trade,
        max_target_delta=max_delta,
    )
    variants: tuple[tuple[str, dict[str, object]], ...] = (
        ("baseline", {}),
        ("ridge-strong", {"ridge_strength": 0.1}),
        ("horizon-24h", {"horizon_mix": CausalAlphaHorizonMix.H24}),
        ("horizon-72h", {"horizon_mix": CausalAlphaHorizonMix.H72}),
        ("scale-low", {"score_scale": 25.0}),
        ("scale-high", {"score_scale": 100.0}),
        (
            "threshold-low",
            {"entry_threshold": 0.0015, "exit_threshold": 0.0005},
        ),
        (
            "threshold-high",
            {"entry_threshold": 0.006, "exit_threshold": 0.002},
        ),
        ("no-trade-low", {"no_trade_band": no_trade * 0.5}),
        (
            "no-trade-high",
            {"no_trade_band": min(float(risk_config.max_abs_weight), no_trade * 2.0)},
        ),
        ("delta-low", {"max_target_delta": max_delta * 0.5}),
        (
            "delta-high",
            {
                "max_target_delta": min(
                    float(risk_config.max_abs_weight), max_delta * 2.0
                )
            },
        ),
    )
    result: list[CausalAlphaCandidateConfig] = []
    observed: set[str] = set()
    for name, overrides in variants:
        kwargs = {**base, **overrides, "name": name}
        candidate = _candidate(**kwargs)  # type: ignore[arg-type]
        if candidate.digest not in observed:
            observed.add(candidate.digest)
            result.append(candidate)
    if len(result) < 8:
        raise ValueError("causal alpha default grid collapsed unexpectedly")
    return tuple(result)


def _candidate_evidence(
    candidate: CausalAlphaCandidateConfig,
    metrics: tuple[CausalAlphaCandidateEpisodeMetrics, ...],
) -> CausalAlphaCandidateEvidence:
    if not metrics:
        raise ValueError("causal alpha candidate has no selection metrics")
    net_returns = np.asarray([item.net_return for item in metrics], dtype=np.float64)
    negative_gross = sum(item.gross_return < 0.0 for item in metrics)
    total_trades = sum(item.trade_count for item in metrics)
    risk_violation = any(item.risk_violation for item in metrics)
    reasons: list[str] = []
    if negative_gross > len(metrics) / 2.0:
        reasons.append("majority_negative_gross_return")
    if total_trades == 0:
        reasons.append("no_meaningful_trades")
    if risk_violation:
        reasons.append("risk_contract_violation")
    return CausalAlphaCandidateEvidence(
        candidate=candidate,
        episode_metrics=metrics,
        lower_tail_net_return=float(np.min(net_returns)),
        mean_net_return=float(np.mean(net_returns, dtype=np.float64)),
        turnover_per_day=float(
            np.mean([item.turnover_per_day for item in metrics], dtype=np.float64)
        ),
        total_execution_cost=float(
            np.sum([item.total_execution_cost for item in metrics], dtype=np.float64)
        ),
        negative_gross_episode_count=negative_gross,
        total_trade_count=total_trades,
        risk_violation=risk_violation,
        admissible=not reasons,
        rejection_reasons=tuple(reasons),
    )


def rank_causal_alpha_candidates(
    *,
    candidates: tuple[CausalAlphaCandidateConfig, ...],
    metrics: Mapping[str, tuple[CausalAlphaCandidateEpisodeMetrics, ...]],
    holdout_episode_digests: Mapping[str, str] | None = None,
) -> CausalAlphaSelectionEvidence:
    """Rank a complete candidate grid without consulting causal holdout metrics."""

    candidate_values = tuple(candidates)
    if not candidate_values:
        raise ValueError("causal alpha candidate grid must be non-empty")
    digests = tuple(candidate.digest for candidate in candidate_values)
    if len(set(digests)) != len(digests):
        raise ValueError("causal alpha candidate grid contains duplicate configs")
    if set(metrics) != set(digests):
        raise ValueError("causal alpha candidate metrics must cover the complete grid")
    evidence = tuple(
        _candidate_evidence(candidate, tuple(metrics[candidate.digest]))
        for candidate in candidate_values
    )
    admissible = tuple(item for item in evidence if item.admissible)
    if not admissible:
        raise RuntimeError("no admissible causal alpha candidate")
    selected = max(
        admissible,
        key=lambda item: (
            item.lower_tail_net_return,
            item.mean_net_return,
            -item.turnover_per_day,
            -item.total_execution_cost,
        ),
    )
    grid_digest = content_digest(
        {
            "candidate_digests": digests,
            "lower_tail_definition": "minimum_symbol_episode_net_return",
            "schema_version": "causal_alpha_selection_grid_v1",
        }
    )
    return CausalAlphaSelectionEvidence(
        candidates=evidence,
        selected_candidate_digest=selected.candidate.digest,
        grid_digest=grid_digest,
        holdout_episode_digests=(
            {} if holdout_episode_digests is None else dict(holdout_episode_digests)
        ),
    )


def _causal_alpha_target_for_contract(
    *,
    symbol: str,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    contract: OracleEpisodeContract,
    candidate: CausalAlphaCandidateConfig,
) -> np.ndarray:
    fitted = fit_expanding_causal_alpha_models(
        train_symbols=train_symbols,
        samples=samples,
        knowledge_cutoff=contract.start,
        ridge_config=candidate.ridge,
    )
    block = samples[symbol]
    if contract.dataset_id != block.dataset_id:
        raise ValueError("causal alpha selection contract dataset identity drifted")
    decisions = np.arange(contract.start, contract.stop - 1, dtype=np.int64)
    prediction_features, prediction_available, actionable = (
        block.prediction_inputs_for_decisions(decisions)
    )
    prediction_24h = fitted.model_24h.predict(
        prediction_features, feature_available=prediction_available
    )
    prediction_72h = fitted.model_72h.predict(
        prediction_features, feature_available=prediction_available
    )
    scores = combine_causal_alpha_predictions(
        prediction_24h,
        prediction_72h,
        candidate.controller.horizon_mix,
    )
    target_path = causal_alpha_target_path(
        scores,
        config=candidate.controller,
        initial_weight=float(contract.initial_weights[0]),
        actionable_mask=actionable,
    )
    return np.asarray(target_path.targets, dtype=np.float32).reshape(-1, 1)


__all__ = [
    "default_causal_alpha_candidate_grid",
    "rank_causal_alpha_candidates",
]
