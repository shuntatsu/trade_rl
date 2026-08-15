"""Production-environment adapters for the research-only causal alpha V3 teacher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_diagnostics import (
    evaluate_causal_alpha_signal_diagnostics,
)
from trade_rl.learning.causal_alpha_v3 import (
    CausalAlphaV3TargetPath,
    causal_alpha_v3_target_path,
)
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.universal_causal_alpha_contracts import CausalAlphaSymbolSamples
from trade_rl.workflows.universal_causal_alpha_costs import (
    causal_alpha_liquidity_weight_caps,
    causal_alpha_one_way_cost_rates,
)
from trade_rl.workflows.universal_causal_alpha_v3 import (
    CausalAlphaV3Fit,
    fit_causal_alpha_v3,
)
from trade_rl.workflows.universal_causal_alpha_v3_config import CausalAlphaV3Candidate
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3SignalScopeMetric,
    non_overlapping_causal_alpha_v3_rows,
)


class CausalAlphaV3SignalScopeUnavailable(ValueError):
    """One signal scope cannot provide an independently auditable metric."""


class CausalAlphaV3FitCache:
    """Share pooled V3 fits across symbols and target-only candidate variants."""

    def __init__(
        self,
        *,
        train_symbols: tuple[str, ...],
        samples: Mapping[str, CausalAlphaSymbolSamples],
    ) -> None:
        self.train_symbols = tuple(train_symbols)
        self.samples = dict(samples)
        self._cache: dict[tuple[str, int], CausalAlphaV3Fit] = {}
        self.fit_count = 0
        self.hit_count = 0

    def resolve(
        self, *, knowledge_cutoff: int, candidate: CausalAlphaV3Candidate
    ) -> CausalAlphaV3Fit:
        key = (candidate.fit.digest, knowledge_cutoff)
        cached = self._cache.get(key)
        if cached is not None:
            self.hit_count += 1
            return cached
        fitted = fit_causal_alpha_v3(
            train_symbols=self.train_symbols,
            samples=self.samples,
            knowledge_cutoff=knowledge_cutoff,
            config=candidate.fit,
        )
        self._cache[key] = fitted
        self.fit_count += 1
        return fitted


@dataclass(frozen=True, slots=True)
class CausalAlphaV3ContractTargets:
    actions: np.ndarray
    fit_digest: str
    forecast_digest: str
    target_path: Any

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32).copy(order="C")
        if actions.ndim != 2 or actions.shape[1] != 1 or actions.shape[0] == 0:
            raise ValueError(
                "V3 contract actions must be a non-empty scalar action path"
            )
        if not np.isfinite(actions).all():
            raise ValueError("V3 contract actions must be finite")
        require_sha256(self.fit_digest, field="V3 contract fit_digest")
        require_sha256(self.forecast_digest, field="V3 contract forecast_digest")
        target_digest = getattr(self.target_path, "digest", None)
        if not isinstance(target_digest, str):
            raise ValueError("V3 contract target_path digest is unavailable")
        require_sha256(target_digest, field="V3 contract target_path digest")
        reasons = tuple(getattr(self.target_path, "reasons", ()))
        if len(reasons) != actions.shape[0] or any(not reason for reason in reasons):
            raise ValueError("V3 contract target reasons must cover every action")
        actions.setflags(write=False)
        object.__setattr__(self, "actions", actions)


def _prediction_scope(
    *,
    symbol: str,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    contract: OracleEpisodeContract,
    candidate: CausalAlphaV3Candidate,
    fit_cache: CausalAlphaV3FitCache | None,
) -> tuple[
    CausalAlphaV3Fit, Any, CausalAlphaSymbolSamples, np.ndarray, np.ndarray, np.ndarray
]:
    if symbol not in samples:
        raise ValueError("V3 prediction symbol is outside sample scope")
    block = samples[symbol]
    if block.dataset_id != contract.dataset_id:
        raise ValueError("V3 prediction contract dataset identity drifted")
    fitted = (
        fit_causal_alpha_v3(
            train_symbols=train_symbols,
            samples=samples,
            knowledge_cutoff=contract.start,
            config=candidate.fit,
        )
        if fit_cache is None
        else fit_cache.resolve(knowledge_cutoff=contract.start, candidate=candidate)
    )
    decisions = np.arange(contract.start, contract.stop - 1, dtype=np.int64)
    features, available, actionable = block.prediction_inputs_for_decisions(decisions)
    forecast = fitted.predict(features, feature_available=available)
    return fitted, forecast, block, decisions, actionable, available


def _labels_for_decisions(
    block: CausalAlphaSymbolSamples,
    decisions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return matched, labels_24h, labels_72h, ends_24h, ends_72h


def build_causal_alpha_v3_signal_scope_metric(
    *,
    symbol: str,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    contract: OracleEpisodeContract,
    candidate: CausalAlphaV3Candidate,
    fit_cache: CausalAlphaV3FitCache | None = None,
) -> CausalAlphaV3SignalScopeMetric:
    """Fit at the scope start and measure independent realized 24h-equivalent alpha."""

    fitted, forecast, block, decisions, actionable, _ = _prediction_scope(
        symbol=symbol,
        train_symbols=train_symbols,
        samples=samples,
        contract=contract,
        candidate=candidate,
        fit_cache=fit_cache,
    )
    matched, labels_24h, labels_72h, ends_24h, ends_72h = _labels_for_decisions(
        block, decisions
    )
    eligible = (
        actionable
        & matched
        & np.isfinite(labels_24h)
        & np.isfinite(labels_72h)
        & (ends_24h >= decisions)
        & (ends_72h >= decisions)
        & (ends_24h < contract.stop)
        & (ends_72h < contract.stop)
    )
    cohort_rows = non_overlapping_causal_alpha_v3_rows(
        decision_indices=decisions,
        label_end_indices=ends_72h,
        eligible_mask=eligible,
    )
    if cohort_rows.size < 2:
        raise CausalAlphaV3SignalScopeUnavailable(
            "V3 signal scope has fewer than two non-overlapping realized labels"
        )
    prediction = forecast.expected_return_24h_equivalent[cohort_rows]
    realized = 0.5 * (labels_24h[cohort_rows] + labels_72h[cohort_rows] / 3.0)
    diagnostics = evaluate_causal_alpha_signal_diagnostics(prediction, realized)
    if diagnostics.rank_correlation is None:
        raise CausalAlphaV3SignalScopeUnavailable(
            "V3 signal scope rank correlation is undefined"
        )
    order = np.argsort(prediction, kind="mergesort")
    bucket = max(1, prediction.size // 5)
    bottom = order[:bucket]
    top = order[-bucket:]
    spread = float(
        np.mean(realized[top], dtype=np.float64)
        - np.mean(realized[bottom], dtype=np.float64)
    )
    return CausalAlphaV3SignalScopeMetric(
        fit_config_digest=candidate.fit.digest,
        symbol=symbol,
        episode_index=contract.episode_index,
        contract_digest=contract.digest,
        fit_digest=fitted.digest,
        forecast_digest=forecast.digest,
        sample_count=int(cohort_rows.size),
        rank_correlation=float(diagnostics.rank_correlation),
        direction_accuracy=diagnostics.direction_accuracy,
        top_bottom_realized_spread=spread,
        cohort_indices=tuple(int(decisions[row]) for row in cohort_rows),
    )


def build_causal_alpha_v3_contract_targets(
    *,
    symbol: str,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    contract: OracleEpisodeContract,
    candidate: CausalAlphaV3Candidate,
    dataset: Any,
    execution_cost: ExecutionCostConfig,
    signal_delay_decisions: int,
    decision_bars: int,
    max_position_to_market_notional: float,
    fit_cache: CausalAlphaV3FitCache | None = None,
    liquidity_lookback_decisions: int = 96,
    liquidity_lower_quantile: float = 0.10,
    liquidity_safety_multiplier: float = 0.80,
) -> CausalAlphaV3ContractTargets:
    """Compile one V3 contract using production cost and causal liquidity evidence."""

    fitted, forecast, block, decisions, actionable, _ = _prediction_scope(
        symbol=symbol,
        train_symbols=train_symbols,
        samples=samples,
        contract=contract,
        candidate=candidate,
        fit_cache=fit_cache,
    )
    cost_rates = causal_alpha_one_way_cost_rates(
        dataset,
        execution_cost,
        decision_indices=decisions,
        signal_delay_decisions=signal_delay_decisions,
        decision_bars=decision_bars,
    )
    liquidity_caps = causal_alpha_liquidity_weight_caps(
        dataset,
        decision_indices=decisions,
        reference_portfolio_value=block.reference_equity,
        max_position_to_market_notional=max_position_to_market_notional,
        lookback_decisions=liquidity_lookback_decisions,
        lower_quantile=liquidity_lower_quantile,
        safety_multiplier=liquidity_safety_multiplier,
    )
    target_path: CausalAlphaV3TargetPath = causal_alpha_v3_target_path(
        forecast.expected_return_24h_equivalent,
        uncertainties=forecast.uncertainty_24h_equivalent,
        one_way_cost_rates=cost_rates,
        liquidity_weight_caps=liquidity_caps,
        actionable_mask=actionable,
        config=candidate.target,
        initial_weight=float(contract.initial_weights[0]),
    )
    return CausalAlphaV3ContractTargets(
        actions=np.asarray(target_path.targets, dtype=np.float32).reshape(-1, 1),
        fit_digest=fitted.digest,
        forecast_digest=forecast.digest,
        target_path=target_path,
    )


def build_causal_alpha_v3_episode_batch(
    *,
    symbol: str,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    contracts: tuple[OracleEpisodeContract, ...],
    candidate: CausalAlphaV3Candidate,
    dataset: Any,
    execution_cost: ExecutionCostConfig,
    signal_delay_decisions: int,
    decision_bars: int,
    max_position_to_market_notional: float,
    teacher_config_digest: str,
    sampling_config_digest: str,
    fit_cache: CausalAlphaV3FitCache | None = None,
) -> EpisodeOracleBatch:
    require_sha256(teacher_config_digest, field="V3 teacher config digest")
    require_sha256(sampling_config_digest, field="V3 sampling config digest")
    values = tuple(contracts)
    if not values:
        raise ValueError("V3 episode batch requires contracts")
    targets = tuple(
        build_causal_alpha_v3_contract_targets(
            symbol=symbol,
            train_symbols=train_symbols,
            samples=samples,
            contract=contract,
            candidate=candidate,
            dataset=dataset,
            execution_cost=execution_cost,
            signal_delay_decisions=signal_delay_decisions,
            decision_bars=decision_bars,
            max_position_to_market_notional=max_position_to_market_notional,
            fit_cache=fit_cache,
        ).actions
        for contract in values
    )
    return EpisodeOracleBatch(
        dataset_id=samples[symbol].dataset_id,
        teacher_config_digest=teacher_config_digest,
        sampling_config_digest=sampling_config_digest,
        contracts=values,
        targets=targets,
        solver_provenance=None,
    )


__all__ = [
    "CausalAlphaV3ContractTargets",
    "CausalAlphaV3FitCache",
    "CausalAlphaV3SignalScopeUnavailable",
    "build_causal_alpha_v3_contract_targets",
    "build_causal_alpha_v3_episode_batch",
    "build_causal_alpha_v3_signal_scope_metric",
]
