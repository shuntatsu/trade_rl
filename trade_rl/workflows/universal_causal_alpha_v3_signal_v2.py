"""Chronological clustered signal evidence for causal alpha V3."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any, Mapping

from trade_rl.evaluation.bootstrap import moving_block_mean_test
from trade_rl.workflows.universal_causal_alpha_v3_config import CausalAlphaV3SignalGate
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3BootstrapEvidence,
    CausalAlphaV3SignalGateEvidence,
    CausalAlphaV3SignalScopeMetric,
)

_SIGNAL_SCOPE_SCHEMA = "causal_alpha_v3_signal_scope_v2"


def signal_scope_metric_from_payload(
    raw: Mapping[str, Any],
) -> CausalAlphaV3SignalScopeMetric:
    fields = {
        "artifact_digest",
        "cohort_indices",
        "contract_digest",
        "contract_start",
        "contract_stop",
        "direction_accuracy",
        "episode_index",
        "fit_config_digest",
        "fit_digest",
        "forecast_digest",
        "rank_correlation",
        "run_manifest_digest",
        "sample_count",
        "schema_version",
        "symbol",
        "top_bottom_realized_spread",
    }
    values = dict(raw)
    if set(values) != fields:
        missing = sorted(fields - set(values))
        unknown = sorted(set(values) - fields)
        raise ValueError(
            f"V3 signal scope fields mismatch; missing={missing}, unknown={unknown}"
        )
    if values["schema_version"] != _SIGNAL_SCOPE_SCHEMA:
        raise ValueError("V3 signal scope schema is unsupported")
    rank_raw = values["rank_correlation"]
    rank = None if rank_raw is None else float(rank_raw)
    return CausalAlphaV3SignalScopeMetric(
        run_manifest_digest=str(values["run_manifest_digest"]),
        fit_config_digest=str(values["fit_config_digest"]),
        symbol=str(values["symbol"]),
        episode_index=int(values["episode_index"]),
        contract_start=int(values["contract_start"]),
        contract_stop=int(values["contract_stop"]),
        contract_digest=str(values["contract_digest"]),
        fit_digest=str(values["fit_digest"]),
        forecast_digest=str(values["forecast_digest"]),
        sample_count=int(values["sample_count"]),
        rank_correlation=rank,
        direction_accuracy=float(values["direction_accuracy"]),
        top_bottom_realized_spread=float(values["top_bottom_realized_spread"]),
        cohort_indices=tuple(int(item) for item in values["cohort_indices"]),
        digest=str(values["artifact_digest"]),
    )


def _bootstrap(
    values: tuple[float, ...], gate: CausalAlphaV3SignalGate
) -> CausalAlphaV3BootstrapEvidence:
    result = moving_block_mean_test(
        values,
        n_bootstrap=gate.bootstrap_resamples,
        seed=gate.bootstrap_seed,
        block_size=gate.bootstrap_block_size,
    )
    return CausalAlphaV3BootstrapEvidence(
        mean=float(fmean(values)),
        p_value=result.p_value,
        lower_ci=result.lower_ci,
        upper_ci=result.upper_ci,
        block_size=result.block_size,
    )


def _required_rank(metric: CausalAlphaV3SignalScopeMetric) -> float:
    value = metric.rank_correlation
    if value is None:
        raise ValueError("V3 signal rank correlation is unavailable")
    return value


def _episode_clusters(
    metrics: tuple[CausalAlphaV3SignalScopeMetric, ...],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    grouped: dict[tuple[int, int], list[CausalAlphaV3SignalScopeMetric]] = defaultdict(
        list
    )
    for metric in metrics:
        grouped[metric.cluster_identity].append(metric)
    ranks: list[float] = []
    spreads: list[float] = []
    directions: list[float] = []
    for interval in sorted(grouped):
        cluster = grouped[interval]
        symbols = tuple(item.symbol for item in cluster)
        if len(set(symbols)) != len(symbols):
            raise ValueError("V3 signal episode cluster contains duplicate symbols")
        ranks.append(float(fmean(_required_rank(item) for item in cluster)))
        spreads.append(
            float(fmean(item.top_bottom_realized_spread for item in cluster))
        )
        directions.append(
            float(fmean(item.direction_accuracy - 0.5 for item in cluster))
        )
    return tuple(ranks), tuple(spreads), tuple(directions)


def _positive_count(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def evaluate_causal_alpha_v3_signal_gate_clustered(
    metrics: tuple[CausalAlphaV3SignalScopeMetric, ...],
    *,
    expected_raw_scope_count: int,
    expected_independent_episode_count: int,
    gate: CausalAlphaV3SignalGate,
) -> CausalAlphaV3SignalGateEvidence:
    """Bootstrap independent chronological episodes, not correlated symbol copies."""

    values = tuple(metrics)
    if not values or any(
        not isinstance(item, CausalAlphaV3SignalScopeMetric) for item in values
    ):
        raise ValueError("V3 signal gate requires scope metrics")
    if len({item.identity for item in values}) != len(values):
        raise ValueError("V3 signal gate scope metrics are duplicated")
    run_digests = {item.run_manifest_digest for item in values}
    if len(run_digests) != 1:
        raise ValueError("V3 signal gate run manifest identity drifted")
    run_manifest_digest = next(iter(run_digests))
    raw_expected = _positive_count(
        expected_raw_scope_count, field="expected_raw_scope_count"
    )
    episode_expected = _positive_count(
        expected_independent_episode_count,
        field="expected_independent_episode_count",
    )
    if episode_expected > raw_expected:
        raise ValueError(
            "expected_independent_episode_count cannot exceed expected_raw_scope_count"
        )
    if len(values) > raw_expected:
        raise ValueError("V3 signal gate has more raw scopes than expected")
    if not isinstance(gate, CausalAlphaV3SignalGate):
        raise TypeError("V3 signal gate config is invalid")

    rank_values, spread_values, direction_values = _episode_clusters(values)
    independent_count = len(rank_values)
    if independent_count > episode_expected:
        raise ValueError("V3 signal gate has more independent episodes than expected")
    coverage = len(values) / float(raw_expected)
    rank = _bootstrap(rank_values, gate)
    spread = _bootstrap(spread_values, gate)
    direction = _bootstrap(direction_values, gate)
    reasons: list[str] = []
    if independent_count < gate.minimum_independent_episode_count:
        reasons.append("independent_episode_count")
    if coverage < gate.minimum_raw_scope_coverage:
        reasons.append("raw_scope_coverage")
    if rank.lower_ci < gate.minimum_rank_ic_lower_ci:
        reasons.append("rank_ic_lower_ci")
    if spread.lower_ci < gate.minimum_top_bottom_spread_lower_ci:
        reasons.append("top_bottom_spread_lower_ci")
    if direction.lower_ci < gate.minimum_direction_accuracy_excess_lower_ci:
        reasons.append("direction_accuracy_excess_lower_ci")
    return CausalAlphaV3SignalGateEvidence(
        metrics=values,
        run_manifest_digest=run_manifest_digest,
        raw_scope_count=len(values),
        expected_raw_scope_count=raw_expected,
        raw_scope_coverage=coverage,
        independent_episode_count=independent_count,
        expected_independent_episode_count=episode_expected,
        rank_ic=rank,
        top_bottom_spread=spread,
        direction_accuracy_excess=direction,
        gate_digest=gate.digest,
        passed=not reasons,
        rejection_reasons=tuple(reasons),
    )


__all__ = [
    "evaluate_causal_alpha_v3_signal_gate_clustered",
    "signal_scope_metric_from_payload",
]
