"""Train-only candidate grid and ranking for the Universal causal alpha teacher."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_diagnostics import (
    CausalAlphaSignalDiagnostics,
    evaluate_causal_alpha_signal_diagnostics,
)
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaControllerConfig,
    CausalAlphaCostAwareConfig,
    CausalAlphaCostAwareTargetPath,
    CausalAlphaHorizonMix,
    CausalAlphaRidgeConfig,
    causal_alpha_cost_aware_target_path,
    causal_alpha_target_path,
    combine_causal_alpha_predictions,
)
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaCandidateConfig,
    CausalAlphaCandidateEpisodeMetrics,
    CausalAlphaCandidateEpisodeMetricsV2,
    CausalAlphaCandidateEvidence,
    CausalAlphaCandidateEvidenceV2,
    CausalAlphaExpandingFit,
    CausalAlphaSelectionEvidence,
    CausalAlphaSelectionEvidenceV2,
    CausalAlphaSymbolSamples,
)
from trade_rl.workflows.universal_causal_alpha_costs import (
    causal_alpha_liquidity_weight_caps,
    causal_alpha_one_way_cost_rates,
)
from trade_rl.workflows.universal_causal_alpha_fitting import (
    CausalAlphaExpandingFitCache,
    fit_expanding_causal_alpha_models,
)

_EXPLAINED_EXECUTION_NO_FILL_REASONS = frozenset(
    {"below_minimum_notional", "zero_quantity_after_rounding"}
)


def causal_alpha_unexplained_execution_rejection_count(
    reason_counts: tuple[tuple[str, int], ...],
) -> int:
    return sum(
        count
        for reason, count in reason_counts
        if reason not in _EXPLAINED_EXECUTION_NO_FILL_REASONS
    )


def _candidate_rejection_payload(
    evidence: CausalAlphaCandidateEvidence,
) -> dict[str, object]:
    return {
        "admissible": evidence.admissible,
        "candidate_digest": evidence.candidate.digest,
        "candidate_name": evidence.candidate.name,
        "episode_metrics": [
            {
                "artifact_digest": metric.digest,
                "episode_index": metric.episode_index,
                "gross_return": metric.gross_return,
                "net_return": metric.net_return,
                "risk_violation": metric.risk_violation,
                "symbol": metric.symbol,
                "total_execution_cost": metric.total_execution_cost,
                "trade_count": metric.trade_count,
                "turnover_per_day": metric.turnover_per_day,
            }
            for metric in evidence.episode_metrics
        ],
        "lower_tail_net_return": evidence.lower_tail_net_return,
        "mean_net_return": evidence.mean_net_return,
        "negative_gross_episode_count": evidence.negative_gross_episode_count,
        "rejection_reasons": list(evidence.rejection_reasons),
        "risk_violation": evidence.risk_violation,
        "total_execution_cost": evidence.total_execution_cost,
        "total_trade_count": evidence.total_trade_count,
        "turnover_per_day": evidence.turnover_per_day,
    }


class CausalAlphaSelectionRejected(RuntimeError):
    """Complete causal selection evidence when every candidate is rejected."""

    def __init__(self, candidates: tuple[CausalAlphaCandidateEvidence, ...]) -> None:
        self.candidates = tuple(candidates)
        payload = {
            "candidates": [
                _candidate_rejection_payload(item) for item in self.candidates
            ],
            "schema_version": "causal_alpha_selection_rejection_v1",
        }
        self.digest = content_digest(payload)
        super().__init__("no admissible causal alpha candidate")

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_digest": self.digest,
            "candidates": [
                _candidate_rejection_payload(item) for item in self.candidates
            ],
            "schema_version": "causal_alpha_selection_rejection_v1",
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaSelectionThresholds:
    minimum_mean_net_return: float = 0.0
    minimum_symbol_episode_net_return: float = -0.05
    maximum_mean_turnover_per_day: float = 1.0
    maximum_unexplained_execution_rejections: int = 0

    def __post_init__(self) -> None:
        for field in (
            "minimum_mean_net_return",
            "minimum_symbol_episode_net_return",
            "maximum_mean_turnover_per_day",
        ):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if self.maximum_mean_turnover_per_day < 0.0:
            raise ValueError("maximum_mean_turnover_per_day must be non-negative")
        value = self.maximum_unexplained_execution_rejections
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "maximum_unexplained_execution_rejections must be non-negative"
            )

    @property
    def digest(self) -> str:
        return content_digest(self)


class CausalAlphaSelectionRejectedV2(RuntimeError):
    def __init__(self, candidates: tuple[CausalAlphaCandidateEvidenceV2, ...]) -> None:
        self.candidates = tuple(candidates)
        self.digest = content_digest(
            {
                "candidate_evidence_digests": tuple(
                    item.digest for item in self.candidates
                ),
                "schema_version": "causal_alpha_selection_rejection_v2",
            }
        )
        super().__init__("no admissible cost-aware causal alpha candidate")

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_digest": self.digest,
            "candidates": tuple(
                {
                    "admissible": item.admissible,
                    "artifact_digest": item.digest,
                    "candidate_digest": item.candidate.digest,
                    "candidate_name": item.candidate.name,
                    "lower_tail_net_return": item.lower_tail_net_return,
                    "mean_net_return": item.mean_net_return,
                    "rejection_reasons": item.rejection_reasons,
                    "turnover_per_day": item.turnover_per_day,
                }
                for item in self.candidates
            ),
            "schema_version": "causal_alpha_selection_rejection_v2",
        }


class _CausalAlphaPredictionCache:
    """Reuse ridge predictions across controller-only candidate variants."""

    def __init__(self) -> None:
        self._cache: dict[
            tuple[str, str, str],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        self.prediction_count = 0
        self.hit_count = 0

    def resolve(
        self,
        *,
        symbol: str,
        block: CausalAlphaSymbolSamples,
        contract: OracleEpisodeContract,
        fitted: CausalAlphaExpandingFit,
        ridge_digest: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (symbol, contract.digest, ridge_digest)
        cached = self._cache.get(key)
        if cached is not None:
            self.hit_count += 1
            return cached
        decisions = np.arange(contract.start, contract.stop - 1, dtype=np.int64)
        prediction_features, prediction_available, actionable = (
            block.prediction_inputs_for_decisions(decisions)
        )
        resolved = (
            fitted.model_24h.predict(
                prediction_features,
                feature_available=prediction_available,
            ),
            fitted.model_72h.predict(
                prediction_features,
                feature_available=prediction_available,
            ),
            actionable,
        )
        self._cache[key] = resolved
        self.prediction_count += 1
        return resolved


class _CausalAlphaLiquidityCapCache:
    """Reuse causal liquidity estimates across controller-only candidates."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, str, float], np.ndarray] = {}
        self.calculation_count = 0
        self.hit_count = 0

    def resolve(
        self,
        *,
        symbol: str,
        dataset: Any,
        contract: OracleEpisodeContract,
        decision_indices: np.ndarray,
        reference_portfolio_value: float,
        economic: CausalAlphaCostAwareConfig,
    ) -> np.ndarray:
        if economic.max_position_to_market_notional is None:
            raise ValueError("causal alpha liquidity cache requires an enabled cap")
        key = (
            symbol,
            contract.digest,
            content_digest(
                {
                    "liquidity_lookback_decisions": (
                        economic.liquidity_lookback_decisions
                    ),
                    "liquidity_lower_quantile": economic.liquidity_lower_quantile,
                    "liquidity_safety_multiplier": (
                        economic.liquidity_safety_multiplier
                    ),
                    "max_position_to_market_notional": (
                        economic.max_position_to_market_notional
                    ),
                    "schema_version": "causal_alpha_liquidity_cap_v1",
                }
            ),
            float(reference_portfolio_value),
        )
        cached = self._cache.get(key)
        if cached is not None:
            self.hit_count += 1
            return cached
        resolved = causal_alpha_liquidity_weight_caps(
            dataset,
            decision_indices=decision_indices,
            reference_portfolio_value=reference_portfolio_value,
            max_position_to_market_notional=(economic.max_position_to_market_notional),
            lookback_decisions=economic.liquidity_lookback_decisions,
            lower_quantile=economic.liquidity_lower_quantile,
            safety_multiplier=economic.liquidity_safety_multiplier,
        )
        resolved.setflags(write=False)
        self._cache[key] = resolved
        self.calculation_count += 1
        return resolved


@dataclass(frozen=True, slots=True)
class CausalAlphaCostAwareContractTargets:
    actions: np.ndarray
    signal_24h: CausalAlphaSignalDiagnostics
    signal_72h: CausalAlphaSignalDiagnostics
    target_path: CausalAlphaCostAwareTargetPath

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32).copy(order="C")
        if actions.ndim != 2 or actions.shape != (self.target_path.targets.size, 1):
            raise ValueError("cost-aware causal alpha actions are not path aligned")
        if not np.isfinite(actions).all():
            raise ValueError("cost-aware causal alpha actions must be finite")
        actions.setflags(write=False)
        object.__setattr__(self, "actions", actions)


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


def default_cost_aware_causal_alpha_candidate_grid(
    *,
    risk_config: PreTradeRiskConfig,
    max_position_to_market_notional: float = 0.02,
) -> tuple[CausalAlphaCandidateConfig, ...]:
    if not isinstance(risk_config, PreTradeRiskConfig):
        raise TypeError("cost-aware causal alpha grid requires PreTradeRiskConfig")
    if risk_config.max_abs_weight < 0.5 or risk_config.max_gross < 0.5:
        raise ValueError(
            "cost-aware causal alpha baseline requires 0.5 exposure support"
        )
    if (
        not math.isfinite(max_position_to_market_notional)
        or max_position_to_market_notional <= 0.0
    ):
        raise ValueError("max_position_to_market_notional must be finite and positive")
    controller_base: dict[str, object] = {
        "horizon_mix": CausalAlphaHorizonMix.EQUAL,
        "score_scale": 25.0,
        "entry_threshold": 0.003,
        "exit_threshold": 0.001,
        "no_trade_band": 0.05,
        "max_target_delta": 0.125,
    }
    economic_base: dict[str, object] = {
        "execution_cost_multiplier": 1.5,
        "edge_margin": 0.001,
        "confirmation_count": 2,
        "strong_reversal_threshold": 0.02,
        "max_abs_target": 0.5,
        "max_position_to_market_notional": max_position_to_market_notional,
        "liquidity_lookback_decisions": 96,
        "liquidity_lower_quantile": 0.10,
        "liquidity_safety_multiplier": 0.80,
    }
    variants: tuple[tuple[str, dict[str, object], dict[str, object]], ...] = (
        ("cost-aware-baseline", {}, {}),
        ("horizon-24h", {"horizon_mix": CausalAlphaHorizonMix.H24}, {}),
        ("horizon-72h", {"horizon_mix": CausalAlphaHorizonMix.H72}, {}),
        ("cost-multiplier-high", {}, {"execution_cost_multiplier": 2.0}),
        ("edge-margin-high", {}, {"edge_margin": 0.002}),
        ("confirmation-one", {}, {"confirmation_count": 1}),
        ("confirmation-three", {}, {"confirmation_count": 3}),
        ("strong-reversal-low", {}, {"strong_reversal_threshold": 0.01}),
        ("scale-low", {"score_scale": 12.5}, {}),
        ("exposure-low", {}, {"max_abs_target": 0.25}),
        ("no-trade-high", {"no_trade_band": 0.10}, {}),
        ("delta-low", {"max_target_delta": 0.0625}, {}),
    )
    candidates = tuple(
        CausalAlphaCandidateConfig(
            name=name,
            ridge=CausalAlphaRidgeConfig(ridge_strength=0.01),
            controller=CausalAlphaControllerConfig(
                **{**controller_base, **controller_overrides}  # type: ignore[arg-type]
            ),
            economic_controller=CausalAlphaCostAwareConfig(
                **{**economic_base, **economic_overrides}  # type: ignore[arg-type]
            ),
        )
        for name, controller_overrides, economic_overrides in variants
    )
    if len({candidate.digest for candidate in candidates}) != len(candidates):
        raise ValueError("cost-aware causal alpha candidate grid contains duplicates")
    return candidates


def _cost_aware_candidate_evidence(
    candidate: CausalAlphaCandidateConfig,
    metrics: tuple[CausalAlphaCandidateEpisodeMetricsV2, ...],
    thresholds: CausalAlphaSelectionThresholds,
) -> CausalAlphaCandidateEvidenceV2:
    if not metrics:
        raise ValueError("cost-aware causal alpha candidate has no metrics")
    net_returns = np.asarray([item.net_return for item in metrics], dtype=np.float64)
    negative_gross = sum(item.gross_return < 0.0 for item in metrics)
    total_trades = sum(item.trade_count for item in metrics)
    hard_risk = any(item.hard_risk_violation for item in metrics)
    rejection_count = sum(
        causal_alpha_unexplained_execution_rejection_count(
            item.execution_rejection_reason_counts
        )
        for item in metrics
    )
    mean_net = float(np.mean(net_returns, dtype=np.float64))
    lower_tail = float(np.min(net_returns))
    mean_turnover = float(
        np.mean([item.turnover_per_day for item in metrics], dtype=np.float64)
    )
    reasons: list[str] = []
    if hard_risk:
        reasons.append("hard_risk_violation")
    if rejection_count > thresholds.maximum_unexplained_execution_rejections:
        reasons.append("unexplained_execution_rejection")
    if total_trades == 0:
        reasons.append("no_meaningful_trades")
    if mean_net < thresholds.minimum_mean_net_return:
        reasons.append("negative_mean_net_return")
    if lower_tail < thresholds.minimum_symbol_episode_net_return:
        reasons.append("lower_tail_net_return_below_floor")
    if mean_turnover > thresholds.maximum_mean_turnover_per_day:
        reasons.append("turnover_per_day_above_maximum")
    if negative_gross > len(metrics) / 2.0:
        reasons.append("majority_negative_gross_return")
    return CausalAlphaCandidateEvidenceV2(
        candidate=candidate,
        episode_metrics=metrics,
        lower_tail_net_return=lower_tail,
        mean_net_return=mean_net,
        turnover_per_day=mean_turnover,
        total_execution_cost=float(
            np.sum([item.total_execution_cost for item in metrics], dtype=np.float64)
        ),
        negative_gross_episode_count=negative_gross,
        total_trade_count=total_trades,
        unexplained_execution_rejection_count=rejection_count,
        hard_risk_violation=hard_risk,
        admissible=not reasons,
        rejection_reasons=tuple(reasons),
    )


def rank_cost_aware_causal_alpha_candidates(
    *,
    candidates: tuple[CausalAlphaCandidateConfig, ...],
    metrics: Mapping[str, tuple[CausalAlphaCandidateEpisodeMetricsV2, ...]],
    thresholds: CausalAlphaSelectionThresholds,
    holdout_episode_digests: Mapping[str, str] | None = None,
) -> CausalAlphaSelectionEvidenceV2:
    values = tuple(candidates)
    if not values or any(item.economic_controller is None for item in values):
        raise ValueError("cost-aware ranking requires v2 candidates")
    digests = tuple(item.digest for item in values)
    if len(set(digests)) != len(digests) or set(metrics) != set(digests):
        raise ValueError("cost-aware metrics must cover one unique complete grid")
    evidence = tuple(
        _cost_aware_candidate_evidence(
            candidate, tuple(metrics[candidate.digest]), thresholds
        )
        for candidate in values
    )
    admissible = tuple(item for item in evidence if item.admissible)
    if not admissible:
        raise CausalAlphaSelectionRejectedV2(evidence)
    selected = max(
        admissible,
        key=lambda item: (
            item.lower_tail_net_return,
            item.mean_net_return,
            -item.turnover_per_day,
            -item.total_execution_cost,
        ),
    )
    grid_digest = cost_aware_causal_alpha_grid_digest(values, thresholds)
    return CausalAlphaSelectionEvidenceV2(
        candidates=evidence,
        selected_candidate_digest=selected.candidate.digest,
        grid_digest=grid_digest,
        thresholds_digest=thresholds.digest,
        holdout_episode_digests=(
            {} if holdout_episode_digests is None else dict(holdout_episode_digests)
        ),
    )


def cost_aware_causal_alpha_grid_digest(
    candidates: tuple[CausalAlphaCandidateConfig, ...],
    thresholds: CausalAlphaSelectionThresholds,
) -> str:
    values = tuple(candidates)
    if not values or any(item.economic_controller is None for item in values):
        raise ValueError("cost-aware grid digest requires v2 candidates")
    digests = tuple(item.digest for item in values)
    if len(set(digests)) != len(digests):
        raise ValueError("cost-aware grid digest requires unique candidates")
    return content_digest(
        {
            "candidate_digests": digests,
            "schema_version": "causal_alpha_selection_grid_v2",
            "thresholds_digest": thresholds.digest,
        }
    )


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
        raise CausalAlphaSelectionRejected(evidence)
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
    fit_cache: CausalAlphaExpandingFitCache | None = None,
    prediction_cache: _CausalAlphaPredictionCache | None = None,
) -> np.ndarray:
    fitted = (
        fit_expanding_causal_alpha_models(
            train_symbols=train_symbols,
            samples=samples,
            knowledge_cutoff=contract.start,
            ridge_config=candidate.ridge,
        )
        if fit_cache is None
        else fit_cache.resolve(
            knowledge_cutoff=contract.start,
            ridge_config=candidate.ridge,
        )
    )
    block = samples[symbol]
    if contract.dataset_id != block.dataset_id:
        raise ValueError("causal alpha selection contract dataset identity drifted")
    if prediction_cache is None:
        prediction_cache = _CausalAlphaPredictionCache()
    prediction_24h, prediction_72h, actionable = prediction_cache.resolve(
        symbol=symbol,
        block=block,
        contract=contract,
        fitted=fitted,
        ridge_digest=candidate.ridge.digest,
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


def _cost_aware_causal_alpha_target_for_contract(
    *,
    symbol: str,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    contract: OracleEpisodeContract,
    candidate: CausalAlphaCandidateConfig,
    dataset: Any,
    execution_cost: ExecutionCostConfig,
    signal_delay_decisions: int,
    decision_bars: int,
    fit_cache: CausalAlphaExpandingFitCache | None = None,
    prediction_cache: _CausalAlphaPredictionCache | None = None,
    liquidity_cache: _CausalAlphaLiquidityCapCache | None = None,
) -> CausalAlphaCostAwareContractTargets:
    economic = candidate.economic_controller
    if economic is None:
        raise ValueError("cost-aware target generation requires a v2 candidate")
    fitted = (
        fit_expanding_causal_alpha_models(
            train_symbols=train_symbols,
            samples=samples,
            knowledge_cutoff=contract.start,
            ridge_config=candidate.ridge,
        )
        if fit_cache is None
        else fit_cache.resolve(
            knowledge_cutoff=contract.start,
            ridge_config=candidate.ridge,
        )
    )
    block = samples[symbol]
    if contract.dataset_id != block.dataset_id:
        raise ValueError("cost-aware selection contract dataset identity drifted")
    cache = (
        _CausalAlphaPredictionCache() if prediction_cache is None else prediction_cache
    )
    prediction_24h, prediction_72h, actionable = cache.resolve(
        symbol=symbol,
        block=block,
        contract=contract,
        fitted=fitted,
        ridge_digest=candidate.ridge.digest,
    )
    decisions = np.arange(contract.start, contract.stop - 1, dtype=np.int64)
    positions = np.searchsorted(block.decision_indices, decisions)
    present = positions < block.decision_indices.size
    matched = np.zeros(decisions.shape, dtype=np.bool_)
    if np.any(present):
        matched[present] = (
            block.decision_indices[positions[present]] == decisions[present]
        )
    labels_24h = np.full(decisions.shape, np.nan, dtype=np.float64)
    labels_72h = np.full(decisions.shape, np.nan, dtype=np.float64)
    ends_24h = np.full(decisions.shape, -1, dtype=np.int64)
    ends_72h = np.full(decisions.shape, -1, dtype=np.int64)
    if np.any(matched):
        source = positions[matched]
        labels_24h[matched] = block.labels_24h[source]
        labels_72h[matched] = block.labels_72h[source]
        ends_24h[matched] = block.label_end_indices_24h[source]
        ends_72h[matched] = block.label_end_indices_72h[source]
    diagnostic_mask = (
        actionable
        & matched
        & np.isfinite(labels_24h)
        & np.isfinite(labels_72h)
        & (ends_24h < contract.stop)
        & (ends_72h < contract.stop)
    )
    if int(np.count_nonzero(diagnostic_mask)) < 2:
        raise ValueError(
            "cost-aware selection episode has insufficient realized labels"
        )
    signal_24h = evaluate_causal_alpha_signal_diagnostics(
        prediction_24h[diagnostic_mask], labels_24h[diagnostic_mask]
    )
    signal_72h = evaluate_causal_alpha_signal_diagnostics(
        prediction_72h[diagnostic_mask], labels_72h[diagnostic_mask]
    )
    scores = combine_causal_alpha_predictions(
        prediction_24h, prediction_72h, candidate.controller.horizon_mix
    )
    cost_rates = causal_alpha_one_way_cost_rates(
        dataset,
        execution_cost,
        decision_indices=decisions,
        signal_delay_decisions=signal_delay_decisions,
        decision_bars=decision_bars,
    )
    liquidity_caps = (
        None
        if economic.max_position_to_market_notional is None
        else (
            causal_alpha_liquidity_weight_caps(
                dataset,
                decision_indices=decisions,
                reference_portfolio_value=block.reference_equity,
                max_position_to_market_notional=(
                    economic.max_position_to_market_notional
                ),
                lookback_decisions=economic.liquidity_lookback_decisions,
                lower_quantile=economic.liquidity_lower_quantile,
                safety_multiplier=economic.liquidity_safety_multiplier,
            )
            if liquidity_cache is None
            else liquidity_cache.resolve(
                symbol=symbol,
                dataset=dataset,
                contract=contract,
                decision_indices=decisions,
                reference_portfolio_value=block.reference_equity,
                economic=economic,
            )
        )
    )
    target_path = causal_alpha_cost_aware_target_path(
        scores,
        one_way_cost_rates=cost_rates,
        liquidity_weight_caps=liquidity_caps,
        controller=candidate.controller,
        economic=economic,
        initial_weight=float(contract.initial_weights[0]),
        actionable_mask=actionable,
    )
    return CausalAlphaCostAwareContractTargets(
        actions=np.asarray(target_path.targets, dtype=np.float32).reshape(-1, 1),
        signal_24h=signal_24h,
        signal_72h=signal_72h,
        target_path=target_path,
    )


__all__ = [
    "CausalAlphaSelectionRejected",
    "CausalAlphaSelectionRejectedV2",
    "CausalAlphaSelectionThresholds",
    "causal_alpha_one_way_cost_rates",
    "causal_alpha_unexplained_execution_rejection_count",
    "cost_aware_causal_alpha_grid_digest",
    "default_causal_alpha_candidate_grid",
    "default_cost_aware_causal_alpha_candidate_grid",
    "rank_causal_alpha_candidates",
    "rank_cost_aware_causal_alpha_candidates",
]
