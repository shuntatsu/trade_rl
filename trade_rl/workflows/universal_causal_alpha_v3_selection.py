"""Resumable production selection for the research-only Causal Alpha V3."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v3 import (
    CausalAlphaV3FitConfig,
    CausalAlphaV3TargetConfig,
)
from trade_rl.learning.episode_oracle_bc import (
    evaluate_episode_action_path_on_environment,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaEpisodePartition,
    CausalAlphaSymbolSamples,
)
from trade_rl.workflows.universal_causal_alpha_costs import (
    causal_alpha_liquidity_weight_caps,
    causal_alpha_one_way_cost_rates,
)
from trade_rl.workflows.universal_causal_alpha_fitting import (
    validate_universal_causal_alpha_partitions,
)
from trade_rl.workflows.universal_causal_alpha_selection import (
    CausalAlphaSelectionThresholds,
)
from trade_rl.workflows.universal_causal_alpha_v3 import (
    CausalAlphaV3FitCache,
    build_causal_alpha_v3_contract_targets,
)
from trade_rl.workflows.universal_causal_alpha_v3_contracts import (
    CausalAlphaV3CandidateConfig,
    CausalAlphaV3CandidateEvidence,
    CausalAlphaV3EpisodeMetric,
    CausalAlphaV3SelectionEvidence,
)


def default_causal_alpha_v3_candidate_grid() -> tuple[
    CausalAlphaV3CandidateConfig, ...
]:
    """Return a bounded one-factor-at-a-time V3 research grid."""

    fit = CausalAlphaV3FitConfig(ridge_strength=0.1)
    target = CausalAlphaV3TargetConfig(
        target_magnitudes=(0.0, 0.025, 0.05, 0.1),
        uncertainty_multiplier=1.0,
        execution_cost_multiplier=1.5,
        edge_margin=0.0005,
        alpha_rebalance_decisions=32,
        strong_reversal_threshold=2.0,
        max_target_delta=0.1,
    )
    variants = (
        ("v3-baseline", fit, target),
        ("ridge-strong", replace(fit, ridge_strength=1.0), target),
        ("uncertainty-low", fit, replace(target, uncertainty_multiplier=0.5)),
        ("uncertainty-high", fit, replace(target, uncertainty_multiplier=2.0)),
        ("cost-low", fit, replace(target, execution_cost_multiplier=1.0)),
        ("cost-high", fit, replace(target, execution_cost_multiplier=2.0)),
        ("edge-zero", fit, replace(target, edge_margin=0.0)),
        ("edge-high", fit, replace(target, edge_margin=0.001)),
        ("cadence-fast", fit, replace(target, alpha_rebalance_decisions=16)),
        ("cadence-slow", fit, replace(target, alpha_rebalance_decisions=96)),
        ("delta-low", fit, replace(target, max_target_delta=0.05)),
    )
    candidates = tuple(
        CausalAlphaV3CandidateConfig(
            name=name,
            fit=fit_config,
            target=target_config,
        )
        for name, fit_config, target_config in variants
    )
    if len({item.digest for item in candidates}) != len(candidates):
        raise RuntimeError("default V3 candidate grid unexpectedly collapsed")
    return candidates


def causal_alpha_v3_grid_digest(
    candidates: tuple[CausalAlphaV3CandidateConfig, ...],
    thresholds: CausalAlphaSelectionThresholds,
) -> str:
    values = tuple(candidates)
    digests = tuple(item.digest for item in values)
    if not values or len(set(digests)) != len(digests):
        raise ValueError("V3 grid requires unique candidates")
    return content_digest(
        {
            "candidate_digests": digests,
            "schema_version": "causal_alpha_v3_selection_grid_v1",
            "thresholds_digest": thresholds.digest,
        }
    )


def _candidate_evidence(
    candidate: CausalAlphaV3CandidateConfig,
    metrics: tuple[CausalAlphaV3EpisodeMetric, ...],
    thresholds: CausalAlphaSelectionThresholds,
) -> CausalAlphaV3CandidateEvidence:
    if not metrics:
        raise ValueError("V3 candidate has no selection metrics")
    mean_net = float(np.mean([item.net_return for item in metrics]))
    lower_tail = min(item.net_return for item in metrics)
    mean_turnover = float(np.mean([item.turnover_per_day for item in metrics]))
    negative_gross = sum(item.gross_return < 0.0 for item in metrics)
    trade_count = sum(item.trade_count for item in metrics)
    rejection_count = sum(
        item.unexplained_execution_rejection_count for item in metrics
    )
    hard_risk = any(item.hard_risk_violation for item in metrics)
    reasons: list[str] = []
    if hard_risk:
        reasons.append("hard_risk_violation")
    if rejection_count > thresholds.maximum_unexplained_execution_rejections:
        reasons.append("unexplained_execution_rejection")
    if trade_count == 0:
        reasons.append("no_meaningful_trades")
    if mean_net < thresholds.minimum_mean_net_return:
        reasons.append("negative_mean_net_return")
    if lower_tail < thresholds.minimum_symbol_episode_net_return:
        reasons.append("lower_tail_net_return_below_floor")
    if mean_turnover > thresholds.maximum_mean_turnover_per_day:
        reasons.append("turnover_per_day_above_maximum")
    if negative_gross > len(metrics) / 2.0:
        reasons.append("majority_negative_gross_return")
    return CausalAlphaV3CandidateEvidence.from_episode_metrics(
        candidate=candidate,
        episode_metrics=metrics,
        admissible=not reasons,
        rejection_reasons=tuple(reasons),
    )


class CausalAlphaV3SelectionRejected(RuntimeError):
    def __init__(
        self,
        candidates: tuple[CausalAlphaV3CandidateEvidence, ...],
        *,
        grid_digest: str,
        generator_code_digest: str,
        sample_scope_digest: str,
    ) -> None:
        self.candidates = tuple(candidates)
        self.grid_digest = grid_digest
        self.generator_code_digest = generator_code_digest
        self.sample_scope_digest = sample_scope_digest
        self.digest = content_digest(self.to_payload(include_digest=False))
        super().__init__("no admissible Causal Alpha V3 candidate")

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidate_evidence_digests": tuple(
                item.digest for item in self.candidates
            ),
            "candidates": tuple(item.to_payload() for item in self.candidates),
            "generator_code_digest": self.generator_code_digest,
            "grid_digest": self.grid_digest,
            "promotion_eligible": False,
            "sample_scope_digest": self.sample_scope_digest,
            "schema_version": "causal_alpha_v3_selection_rejection_v1",
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def rank_causal_alpha_v3_candidates(
    *,
    candidates: tuple[CausalAlphaV3CandidateConfig, ...],
    metrics: Mapping[str, tuple[CausalAlphaV3EpisodeMetric, ...]],
    thresholds: CausalAlphaSelectionThresholds,
    generator_code_digest: str,
    sample_scope_digest: str,
    holdout_episode_digests: Mapping[str, str],
) -> CausalAlphaV3SelectionEvidence:
    values = tuple(candidates)
    digests = tuple(item.digest for item in values)
    if not values or len(set(digests)) != len(digests) or set(metrics) != set(digests):
        raise ValueError("V3 metrics must cover one unique complete grid")
    for name, value in (
        ("generator_code_digest", generator_code_digest),
        ("sample_scope_digest", sample_scope_digest),
    ):
        if len(value) != 64:
            raise ValueError(f"V3 {name} is invalid")
    evidence = tuple(
        _candidate_evidence(candidate, tuple(metrics[candidate.digest]), thresholds)
        for candidate in values
    )
    grid_digest = causal_alpha_v3_grid_digest(values, thresholds)
    admissible = tuple(item for item in evidence if item.admissible)
    if not admissible:
        raise CausalAlphaV3SelectionRejected(
            evidence,
            grid_digest=grid_digest,
            generator_code_digest=generator_code_digest,
            sample_scope_digest=sample_scope_digest,
        )
    selected = max(
        admissible,
        key=lambda item: (
            item.lower_tail_net_return,
            item.mean_net_return,
            -item.turnover_per_day,
            -item.total_execution_cost,
        ),
    )
    return CausalAlphaV3SelectionEvidence(
        candidates=evidence,
        selected_candidate_digest=selected.candidate.digest,
        grid_digest=grid_digest,
        thresholds_digest=thresholds.digest,
        generator_code_digest=generator_code_digest,
        sample_scope_digest=sample_scope_digest,
        holdout_episode_digests=holdout_episode_digests,
    )


def write_causal_alpha_v3_selection_checkpoint_metric(
    path: Path,
    metric: CausalAlphaV3EpisodeMetric,
    *,
    grid_digest: str,
    generator_code_digest: str,
    sample_scope_digest: str,
) -> None:
    payload = {
        **metric.to_payload(),
        "generator_code_digest": generator_code_digest,
        "grid_digest": grid_digest,
        "sample_scope_digest": sample_scope_digest,
        "schema_version": "causal_alpha_v3_selection_checkpoint_metric_v1",
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("ab") as checkpoint:
        checkpoint.write(canonical_json_bytes(payload) + b"\n")
        checkpoint.flush()
        os.fsync(checkpoint.fileno())


def causal_alpha_v3_metric_from_payload(
    raw: Mapping[str, Any],
) -> CausalAlphaV3EpisodeMetric:
    return CausalAlphaV3EpisodeMetric(
        candidate_digest=str(raw["candidate_digest"]),
        symbol=str(raw["symbol"]),
        episode_index=int(raw["episode_index"]),
        contract_digest=str(raw["contract_digest"]),
        gross_return=float(raw["gross_return"]),
        net_return=float(raw["net_return"]),
        turnover_per_day=float(raw["turnover_per_day"]),
        total_execution_cost=float(raw["total_execution_cost"]),
        trade_count=int(raw["trade_count"]),
        hard_risk_violation=bool(raw["hard_risk_violation"]),
        unexplained_execution_rejection_count=int(
            raw["unexplained_execution_rejection_count"]
        ),
        digest=str(raw["artifact_digest"]),
    )


def load_causal_alpha_v3_selection_checkpoint(
    path: Path,
    *,
    expected_grid_digest: str,
    expected_generator_code_digest: str,
    expected_sample_scope_digest: str,
) -> dict[str, tuple[CausalAlphaV3EpisodeMetric, ...]]:
    source = Path(path)
    if not source.is_file():
        return {}
    result: dict[str, list[CausalAlphaV3EpisodeMetric]] = {}
    identities: set[tuple[str, str, int]] = set()
    with source.open("r", encoding="utf-8") as checkpoint:
        for line in checkpoint:
            raw = json.loads(line)
            if raw.get("schema_version") != (
                "causal_alpha_v3_selection_checkpoint_metric_v1"
            ):
                raise ValueError("V3 selection checkpoint schema mismatch")
            for field, expected in (
                ("grid_digest", expected_grid_digest),
                ("generator_code_digest", expected_generator_code_digest),
                ("sample_scope_digest", expected_sample_scope_digest),
            ):
                if raw.get(field) != expected:
                    raise ValueError(
                        f"V3 selection checkpoint {field.replace('_', ' ')} mismatch"
                    )
            metric = causal_alpha_v3_metric_from_payload(raw)
            identity = (metric.candidate_digest, metric.symbol, metric.episode_index)
            if identity in identities:
                raise ValueError("V3 selection checkpoint is duplicated")
            identities.add(identity)
            result.setdefault(metric.candidate_digest, []).append(metric)
    return {digest: tuple(values) for digest, values in result.items()}


def _unexplained_rejections(reason_counts: tuple[tuple[str, int], ...]) -> int:
    explained = {"below_minimum_notional", "zero_quantity_after_rounding"}
    return sum(count for reason, count in reason_counts if reason not in explained)


def evaluate_causal_alpha_v3_selection(
    *,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    partitions: Mapping[str, CausalAlphaEpisodePartition],
    candidates: tuple[CausalAlphaV3CandidateConfig, ...],
    environment_factories: Mapping[str, Any],
    episode_hours: float,
    thresholds: CausalAlphaSelectionThresholds,
    generator_code_digest: str,
    fit_cache: CausalAlphaV3FitCache,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    initial_metrics: Mapping[str, tuple[CausalAlphaV3EpisodeMetric, ...]] | None = None,
    max_position_to_market_notional: float = 0.02,
) -> CausalAlphaV3SelectionEvidence:
    symbols = tuple(train_symbols)
    partition_values = validate_universal_causal_alpha_partitions(
        train_symbols=symbols, partitions=partitions
    )
    if set(samples) != set(symbols) or set(environment_factories) != set(symbols):
        raise ValueError("V3 selection scope must exactly match train_symbols")
    if not math.isfinite(episode_hours) or episode_hours <= 0.0:
        raise ValueError("V3 episode_hours must be positive")
    if max_position_to_market_notional != 0.02:
        raise ValueError("V3 hard liquidity contract must remain 0.02")
    records = {
        item.digest: list((initial_metrics or {}).get(item.digest, ()))
        for item in candidates
    }
    expected_scopes = {
        (candidate.digest, symbol, contract.episode_index)
        for candidate in candidates
        for symbol in symbols
        for contract in partition_values[symbol].selection_contracts
    }
    completed = {
        (digest, metric.symbol, metric.episode_index)
        for digest, values in records.items()
        for metric in values
    }
    if not completed.issubset(expected_scopes):
        raise ValueError("V3 resumed metric identity is invalid")
    total = len(expected_scopes)
    episode_days = episode_hours / 24.0
    cost_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for symbol in symbols:
        environment = environment_factories[symbol]()
        close = getattr(environment, "close", None)
        if not callable(close):
            raise TypeError("V3 selection environment is not closable")
        try:
            config = getattr(environment, "config", None)
            execution_cost = getattr(config, "execution_cost", None)
            signal_delay = getattr(config, "signal_delay_decisions", None)
            decision_bars = getattr(environment, "decision_bars", None)
            if not isinstance(execution_cost, ExecutionCostConfig):
                raise TypeError("V3 execution cost config is unavailable")
            if not isinstance(signal_delay, int) or not isinstance(decision_bars, int):
                raise ValueError("V3 execution timing is unavailable")
            for contract in partition_values[symbol].selection_contracts:
                decisions = np.arange(contract.start, contract.stop - 1, dtype=np.int64)
                cost_key = (symbol, contract.digest)
                if cost_key not in cost_cache:
                    cost_cache[cost_key] = (
                        causal_alpha_one_way_cost_rates(
                            environment.dataset,
                            execution_cost,
                            decision_indices=decisions,
                            signal_delay_decisions=signal_delay,
                            decision_bars=decision_bars,
                        ),
                        causal_alpha_liquidity_weight_caps(
                            environment.dataset,
                            decision_indices=decisions,
                            reference_portfolio_value=samples[symbol].reference_equity,
                            max_position_to_market_notional=0.02,
                            lookback_decisions=96,
                            lower_quantile=0.10,
                            safety_multiplier=0.80,
                        ),
                    )
                costs, caps = cost_cache[cost_key]
                for candidate in candidates:
                    identity = (candidate.digest, symbol, contract.episode_index)
                    if identity in completed:
                        continue
                    target = build_causal_alpha_v3_contract_targets(
                        symbol=symbol,
                        samples=samples,
                        contract=contract,
                        candidate=candidate,
                        fit_cache=fit_cache,
                        one_way_cost_rates=costs,
                        liquidity_weight_caps=caps,
                    )
                    evaluation = evaluate_episode_action_path_on_environment(
                        environment, contract, actions=target.targets
                    )
                    performance = evaluation.performance
                    collapse = evaluation.collapse_evidence
                    metric = CausalAlphaV3EpisodeMetric(
                        candidate_digest=candidate.digest,
                        symbol=symbol,
                        episode_index=contract.episode_index,
                        contract_digest=contract.digest,
                        gross_return=float(performance.gross_return),
                        net_return=float(performance.net_return),
                        turnover_per_day=float(performance.turnover_total)
                        / episode_days,
                        total_execution_cost=float(performance.cost_total),
                        trade_count=int(performance.trade_count),
                        hard_risk_violation=bool(collapse.hard_risk_violation),
                        unexplained_execution_rejection_count=_unexplained_rejections(
                            collapse.execution_rejection_reason_counts
                        ),
                    )
                    records[candidate.digest].append(metric)
                    completed.add(identity)
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "candidate_digest": candidate.digest,
                                "completed_replays": len(completed),
                                "episode_index": contract.episode_index,
                                "episode_metric": metric.to_payload(),
                                "fit_cache_hits": fit_cache.fit_hit_count,
                                "fit_count": fit_cache.fit_count,
                                "phase": "causal_alpha_v3_selection",
                                "prediction_cache_hits": fit_cache.prediction_hit_count,
                                "prediction_count": fit_cache.prediction_count,
                                "symbol": symbol,
                                "total_replays": total,
                            }
                        )
        finally:
            close()
    return rank_causal_alpha_v3_candidates(
        candidates=candidates,
        metrics={digest: tuple(values) for digest, values in records.items()},
        thresholds=thresholds,
        generator_code_digest=generator_code_digest,
        sample_scope_digest=fit_cache.sample_scope_digest,
        holdout_episode_digests={
            symbol: partition_values[symbol].holdout_contract.digest
            for symbol in symbols
        },
    )


__all__ = [
    "CausalAlphaV3SelectionRejected",
    "causal_alpha_v3_grid_digest",
    "causal_alpha_v3_metric_from_payload",
    "default_causal_alpha_v3_candidate_grid",
    "evaluate_causal_alpha_v3_selection",
    "load_causal_alpha_v3_selection_checkpoint",
    "rank_causal_alpha_v3_candidates",
    "write_causal_alpha_v3_selection_checkpoint_metric",
]
