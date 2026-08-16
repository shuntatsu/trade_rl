"""Read-only forensic summaries for persisted Causal Alpha V3 Signal evidence."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence

from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_causal_alpha_v3_config import (
    CausalAlphaV3ResearchConfig,
)
from trade_rl.workflows.universal_causal_alpha_v3_identity import (
    CausalAlphaV3RunManifestV2,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3SignalScopeMetric,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_v2 import (
    signal_scope_metric_from_payload,
)

_SIGNAL_FORENSICS_SCHEMA = "causal_alpha_v3_signal_forensics_v1"
_SIGNAL_REJECTION_SCHEMA = "causal_alpha_v3_signal_rejection_v2"
_FIT_SIGNAL_RESULT_SCHEMA = "causal_alpha_v3_fit_signal_result_v2"
_SIGNAL_GATE_SCHEMA = "causal_alpha_v3_signal_gate_evidence_v2"
_SIGNAL_INDEPENDENCE_UNIT = "chronological_episode"
_SIGNAL_AGGREGATION_MODE = "cross_symbol_episode_mean"


@dataclass(frozen=True, slots=True)
class CausalAlphaV3NumericSummary:
    count: int
    mean: float
    std: float
    minimum: float
    maximum: float

    def to_payload(self) -> dict[str, object]:
        return {
            "count": self.count,
            "maximum": self.maximum,
            "mean": self.mean,
            "minimum": self.minimum,
            "std": self.std,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3SignedMetricSummary:
    count: int
    mean: float
    std: float
    minimum: float
    maximum: float
    negative_fraction: float

    def to_payload(self) -> dict[str, object]:
        return {
            "count": self.count,
            "maximum": self.maximum,
            "mean": self.mean,
            "minimum": self.minimum,
            "negative_fraction": self.negative_fraction,
            "std": self.std,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3TrendSummary:
    early_mean: float
    late_mean: float
    slope: float

    def to_payload(self) -> dict[str, object]:
        return {
            "early_mean": self.early_mean,
            "late_mean": self.late_mean,
            "slope": self.slope,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3FitSummary:
    fit_config_digest: str
    candidate_names: tuple[str, ...]
    ridge_strength: float
    raw_scope_count: int
    independent_episode_count: int
    symbol_count: int
    sample_count: CausalAlphaV3NumericSummary
    raw_rank_ic: CausalAlphaV3SignedMetricSummary
    raw_top_bottom_spread: CausalAlphaV3SignedMetricSummary
    raw_direction_accuracy_excess: CausalAlphaV3SignedMetricSummary
    episode_rank_ic: CausalAlphaV3SignedMetricSummary
    episode_top_bottom_spread: CausalAlphaV3SignedMetricSummary
    episode_direction_accuracy_excess: CausalAlphaV3SignedMetricSummary
    rank_ic_trend: CausalAlphaV3TrendSummary
    top_bottom_spread_trend: CausalAlphaV3TrendSummary
    direction_accuracy_excess_trend: CausalAlphaV3TrendSummary
    fit_digest_unique_count: int
    fit_digest_transition_count: int

    def to_payload(self) -> dict[str, object]:
        return {
            "candidate_names": self.candidate_names,
            "direction_accuracy_excess_trend": (
                self.direction_accuracy_excess_trend.to_payload()
            ),
            "episode_direction_accuracy_excess": (
                self.episode_direction_accuracy_excess.to_payload()
            ),
            "episode_rank_ic": self.episode_rank_ic.to_payload(),
            "episode_top_bottom_spread": self.episode_top_bottom_spread.to_payload(),
            "fit_config_digest": self.fit_config_digest,
            "fit_digest_transition_count": self.fit_digest_transition_count,
            "fit_digest_unique_count": self.fit_digest_unique_count,
            "independent_episode_count": self.independent_episode_count,
            "rank_ic_trend": self.rank_ic_trend.to_payload(),
            "raw_direction_accuracy_excess": (
                self.raw_direction_accuracy_excess.to_payload()
            ),
            "raw_rank_ic": self.raw_rank_ic.to_payload(),
            "raw_scope_count": self.raw_scope_count,
            "raw_top_bottom_spread": self.raw_top_bottom_spread.to_payload(),
            "ridge_strength": self.ridge_strength,
            "sample_count": self.sample_count.to_payload(),
            "symbol_count": self.symbol_count,
            "top_bottom_spread_trend": self.top_bottom_spread_trend.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3EpisodeSummary:
    fit_config_digest: str
    contract_start: int
    contract_stop: int
    episode_indices: tuple[int, ...]
    fit_digest: str
    symbol_count: int
    total_sample_count: int
    mean_sample_count: float
    rank_ic: float
    top_bottom_spread: float
    direction_accuracy: float
    direction_accuracy_excess: float
    negative_rank_symbol_count: int
    negative_spread_symbol_count: int
    negative_direction_excess_symbol_count: int

    @property
    def cluster_identity(self) -> tuple[int, int]:
        return (self.contract_start, self.contract_stop)

    def to_payload(self) -> dict[str, object]:
        return {
            "contract_start": self.contract_start,
            "contract_stop": self.contract_stop,
            "direction_accuracy": self.direction_accuracy,
            "direction_accuracy_excess": self.direction_accuracy_excess,
            "episode_indices": self.episode_indices,
            "fit_config_digest": self.fit_config_digest,
            "fit_digest": self.fit_digest,
            "mean_sample_count": self.mean_sample_count,
            "negative_direction_excess_symbol_count": (
                self.negative_direction_excess_symbol_count
            ),
            "negative_rank_symbol_count": self.negative_rank_symbol_count,
            "negative_spread_symbol_count": self.negative_spread_symbol_count,
            "rank_ic": self.rank_ic,
            "symbol_count": self.symbol_count,
            "top_bottom_spread": self.top_bottom_spread,
            "total_sample_count": self.total_sample_count,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3SymbolSummary:
    fit_config_digest: str
    symbol: str
    episode_count: int
    sample_count: CausalAlphaV3NumericSummary
    rank_ic: CausalAlphaV3SignedMetricSummary
    top_bottom_spread: CausalAlphaV3SignedMetricSummary
    direction_accuracy_excess: CausalAlphaV3SignedMetricSummary

    def to_payload(self) -> dict[str, object]:
        return {
            "direction_accuracy_excess": self.direction_accuracy_excess.to_payload(),
            "episode_count": self.episode_count,
            "fit_config_digest": self.fit_config_digest,
            "rank_ic": self.rank_ic.to_payload(),
            "sample_count": self.sample_count.to_payload(),
            "symbol": self.symbol,
            "top_bottom_spread": self.top_bottom_spread.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3FitComparison:
    left_fit_config_digest: str
    right_fit_config_digest: str
    common_episode_count: int
    mean_rank_ic_delta: float
    mean_top_bottom_spread_delta: float
    mean_direction_accuracy_excess_delta: float
    left_rank_ic_win_fraction: float
    left_top_bottom_spread_win_fraction: float
    left_direction_accuracy_excess_win_fraction: float

    def to_payload(self) -> dict[str, object]:
        return {
            "common_episode_count": self.common_episode_count,
            "left_direction_accuracy_excess_win_fraction": (
                self.left_direction_accuracy_excess_win_fraction
            ),
            "left_fit_config_digest": self.left_fit_config_digest,
            "left_rank_ic_win_fraction": self.left_rank_ic_win_fraction,
            "left_top_bottom_spread_win_fraction": (
                self.left_top_bottom_spread_win_fraction
            ),
            "mean_direction_accuracy_excess_delta": (
                self.mean_direction_accuracy_excess_delta
            ),
            "mean_rank_ic_delta": self.mean_rank_ic_delta,
            "mean_top_bottom_spread_delta": self.mean_top_bottom_spread_delta,
            "right_fit_config_digest": self.right_fit_config_digest,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV3UnavailableAnalysis:
    analysis: str
    reason: str

    def to_payload(self) -> dict[str, object]:
        return {"analysis": self.analysis, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class CausalAlphaV3SignalForensicsReport:
    run_manifest_digest: str
    config_digest: str
    train_symbols: tuple[str, ...]
    fit_config_digests: tuple[str, ...]
    raw_scope_count: int
    independent_episode_count: int
    source_signal_status: str
    source_rejection_digest: str | None
    fit_summaries: tuple[CausalAlphaV3FitSummary, ...]
    episode_summaries: tuple[CausalAlphaV3EpisodeSummary, ...]
    symbol_summaries: tuple[CausalAlphaV3SymbolSummary, ...]
    fit_comparisons: tuple[CausalAlphaV3FitComparison, ...]
    unavailable_analyses: tuple[CausalAlphaV3UnavailableAnalysis, ...]
    research_only: bool = True
    promotion_eligible: bool = False
    schema_version: str = _SIGNAL_FORENSICS_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != _SIGNAL_FORENSICS_SCHEMA:
            raise ValueError("unsupported V3 signal forensics schema")
        if not self.research_only or self.promotion_eligible:
            raise ValueError("V3 signal forensics must remain research-only")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 signal forensics digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "config_digest": self.config_digest,
            "episode_summaries": tuple(
                item.to_payload() for item in self.episode_summaries
            ),
            "fit_comparisons": tuple(
                item.to_payload() for item in self.fit_comparisons
            ),
            "fit_config_digests": self.fit_config_digests,
            "fit_summaries": tuple(item.to_payload() for item in self.fit_summaries),
            "independent_episode_count": self.independent_episode_count,
            "promotion_eligible": self.promotion_eligible,
            "raw_scope_count": self.raw_scope_count,
            "research_only": self.research_only,
            "run_manifest_digest": self.run_manifest_digest,
            "schema_version": self.schema_version,
            "source_rejection_digest": self.source_rejection_digest,
            "source_signal_status": self.source_signal_status,
            "symbol_summaries": tuple(
                item.to_payload() for item in self.symbol_summaries
            ),
            "train_symbols": self.train_symbols,
            "unavailable_analyses": tuple(
                item.to_payload() for item in self.unavailable_analyses
            ),
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _load_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is unreadable") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{field} must be a JSON object")
    return dict(raw)


def _numeric_summary(values: Sequence[int | float]) -> CausalAlphaV3NumericSummary:
    resolved = tuple(float(value) for value in values)
    if not resolved or not all(math.isfinite(value) for value in resolved):
        raise ValueError("V3 signal forensics numeric summary requires finite values")
    return CausalAlphaV3NumericSummary(
        count=len(resolved),
        mean=float(fmean(resolved)),
        std=0.0 if len(resolved) == 1 else float(pstdev(resolved)),
        minimum=float(min(resolved)),
        maximum=float(max(resolved)),
    )


def _signed_summary(values: Sequence[float]) -> CausalAlphaV3SignedMetricSummary:
    resolved = tuple(float(value) for value in values)
    base = _numeric_summary(resolved)
    return CausalAlphaV3SignedMetricSummary(
        count=base.count,
        mean=base.mean,
        std=base.std,
        minimum=base.minimum,
        maximum=base.maximum,
        negative_fraction=float(sum(value < 0.0 for value in resolved) / len(resolved)),
    )


def _trend(values: Sequence[float]) -> CausalAlphaV3TrendSummary:
    resolved = tuple(float(value) for value in values)
    if not resolved:
        raise ValueError("V3 signal forensics trend requires values")
    if len(resolved) == 1:
        return CausalAlphaV3TrendSummary(
            early_mean=resolved[0], late_mean=resolved[0], slope=0.0
        )
    split = len(resolved) // 2
    early = resolved[:split]
    late = resolved[split:]
    x_mean = (len(resolved) - 1) / 2.0
    y_mean = float(fmean(resolved))
    numerator = sum(
        (float(index) - x_mean) * (value - y_mean)
        for index, value in enumerate(resolved)
    )
    denominator = sum((float(index) - x_mean) ** 2 for index in range(len(resolved)))
    return CausalAlphaV3TrendSummary(
        early_mean=float(fmean(early)),
        late_mean=float(fmean(late)),
        slope=float(numerator / denominator),
    )


def _rank(metric: CausalAlphaV3SignalScopeMetric) -> float:
    value = metric.rank_correlation
    if value is None:
        raise ValueError("V3 signal forensics rank correlation is unavailable")
    return float(value)


def _direction_excess(metric: CausalAlphaV3SignalScopeMetric) -> float:
    return float(metric.direction_accuracy - 0.5)


def _fit_order(config: CausalAlphaV3ResearchConfig) -> tuple[str, ...]:
    result: list[str] = []
    for candidate in config.candidates:
        digest = candidate.fit.digest
        if digest not in result:
            result.append(digest)
    return tuple(result)


def _fit_metadata(
    config: CausalAlphaV3ResearchConfig,
) -> dict[str, tuple[tuple[str, ...], float]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    strengths: dict[str, set[float]] = defaultdict(set)
    for candidate in config.candidates:
        digest = candidate.fit.digest
        grouped[digest].append(candidate.name)
        strengths[digest].add(float(candidate.fit.ridge_strength))
    result: dict[str, tuple[tuple[str, ...], float]] = {}
    for digest in _fit_order(config):
        values = strengths[digest]
        if len(values) != 1:
            raise ValueError("V3 signal forensics fit metadata is inconsistent")
        result[digest] = (tuple(grouped[digest]), next(iter(values)))
    return result


def _load_metrics(
    root: Path,
    *,
    manifest: CausalAlphaV3RunManifestV2,
    config: CausalAlphaV3ResearchConfig,
) -> tuple[CausalAlphaV3SignalScopeMetric, ...]:
    records_root = root / "signal" / "records"
    if not records_root.is_dir():
        raise FileNotFoundError(records_root)
    paths = tuple(sorted(records_root.rglob("*.json")))
    if not paths:
        raise ValueError("V3 signal forensics found no persisted Signal records")
    allowed_symbols = set(manifest.train_symbols)
    allowed_fits = set(_fit_order(config))
    metrics: list[CausalAlphaV3SignalScopeMetric] = []
    identities: set[tuple[str, str, int]] = set()
    for path in paths:
        metric = signal_scope_metric_from_payload(
            _load_json(path, field="V3 signal record")
        )
        if metric.run_manifest_digest != manifest.digest:
            raise ValueError("V3 signal record run manifest identity drifted")
        if metric.symbol not in allowed_symbols:
            raise ValueError("V3 signal record symbol is outside the run manifest")
        if metric.fit_config_digest not in allowed_fits:
            raise ValueError("V3 signal record fit config is outside authored config")
        expected = (
            records_root
            / metric.fit_config_digest
            / metric.symbol
            / f"{metric.episode_index}.json"
        )
        if path != expected:
            raise ValueError("V3 signal record path identity drifted")
        if metric.identity in identities:
            raise ValueError("V3 signal record identity is duplicated")
        identities.add(metric.identity)
        metrics.append(metric)
    return tuple(metrics)


def _cluster_metrics(
    metrics: Sequence[CausalAlphaV3SignalScopeMetric],
    *,
    symbols: tuple[str, ...],
) -> tuple[tuple[CausalAlphaV3SignalScopeMetric, ...], ...]:
    grouped: dict[tuple[int, int], list[CausalAlphaV3SignalScopeMetric]] = defaultdict(
        list
    )
    for metric in metrics:
        grouped[metric.cluster_identity].append(metric)
    expected_symbols = set(symbols)
    symbol_order = {symbol: index for index, symbol in enumerate(symbols)}
    result: list[tuple[CausalAlphaV3SignalScopeMetric, ...]] = []
    for interval in sorted(grouped):
        cluster = grouped[interval]
        observed_symbols = tuple(item.symbol for item in cluster)
        if (
            len(observed_symbols) != len(set(observed_symbols))
            or set(observed_symbols) != expected_symbols
        ):
            raise ValueError(
                "V3 signal episode cluster does not cover the complete symbol scope"
            )
        if len({item.fit_digest for item in cluster}) != 1:
            raise ValueError("V3 signal episode cluster pooled fit identity drifted")
        result.append(
            tuple(sorted(cluster, key=lambda item: symbol_order[item.symbol]))
        )
    return tuple(result)


def _episode_summary(
    fit_config_digest: str,
    cluster: Sequence[CausalAlphaV3SignalScopeMetric],
) -> CausalAlphaV3EpisodeSummary:
    ranks = tuple(_rank(item) for item in cluster)
    spreads = tuple(float(item.top_bottom_realized_spread) for item in cluster)
    directions = tuple(float(item.direction_accuracy) for item in cluster)
    direction_excesses = tuple(value - 0.5 for value in directions)
    sample_counts = tuple(item.sample_count for item in cluster)
    fit_digests = {item.fit_digest for item in cluster}
    if len(fit_digests) != 1:
        raise ValueError("V3 signal episode cluster pooled fit identity drifted")
    first = cluster[0]
    return CausalAlphaV3EpisodeSummary(
        fit_config_digest=fit_config_digest,
        contract_start=first.contract_start,
        contract_stop=first.contract_stop,
        episode_indices=tuple(sorted({item.episode_index for item in cluster})),
        fit_digest=next(iter(fit_digests)),
        symbol_count=len(cluster),
        total_sample_count=sum(sample_counts),
        mean_sample_count=float(fmean(sample_counts)),
        rank_ic=float(fmean(ranks)),
        top_bottom_spread=float(fmean(spreads)),
        direction_accuracy=float(fmean(directions)),
        direction_accuracy_excess=float(fmean(direction_excesses)),
        negative_rank_symbol_count=sum(value < 0.0 for value in ranks),
        negative_spread_symbol_count=sum(value < 0.0 for value in spreads),
        negative_direction_excess_symbol_count=sum(
            value < 0.0 for value in direction_excesses
        ),
    )


def _fit_summary(
    fit_config_digest: str,
    metrics: Sequence[CausalAlphaV3SignalScopeMetric],
    episodes: Sequence[CausalAlphaV3EpisodeSummary],
    *,
    metadata: tuple[tuple[str, ...], float],
    symbol_count: int,
) -> CausalAlphaV3FitSummary:
    fit_digests = tuple(item.fit_digest for item in episodes)
    return CausalAlphaV3FitSummary(
        fit_config_digest=fit_config_digest,
        candidate_names=metadata[0],
        ridge_strength=metadata[1],
        raw_scope_count=len(metrics),
        independent_episode_count=len(episodes),
        symbol_count=symbol_count,
        sample_count=_numeric_summary(tuple(item.sample_count for item in metrics)),
        raw_rank_ic=_signed_summary(tuple(_rank(item) for item in metrics)),
        raw_top_bottom_spread=_signed_summary(
            tuple(float(item.top_bottom_realized_spread) for item in metrics)
        ),
        raw_direction_accuracy_excess=_signed_summary(
            tuple(_direction_excess(item) for item in metrics)
        ),
        episode_rank_ic=_signed_summary(tuple(item.rank_ic for item in episodes)),
        episode_top_bottom_spread=_signed_summary(
            tuple(item.top_bottom_spread for item in episodes)
        ),
        episode_direction_accuracy_excess=_signed_summary(
            tuple(item.direction_accuracy_excess for item in episodes)
        ),
        rank_ic_trend=_trend(tuple(item.rank_ic for item in episodes)),
        top_bottom_spread_trend=_trend(
            tuple(item.top_bottom_spread for item in episodes)
        ),
        direction_accuracy_excess_trend=_trend(
            tuple(item.direction_accuracy_excess for item in episodes)
        ),
        fit_digest_unique_count=len(set(fit_digests)),
        fit_digest_transition_count=sum(
            left != right for left, right in zip(fit_digests, fit_digests[1:])
        ),
    )


def _symbol_summary(
    fit_config_digest: str,
    symbol: str,
    metrics: Sequence[CausalAlphaV3SignalScopeMetric],
) -> CausalAlphaV3SymbolSummary:
    ordered = tuple(
        sorted(metrics, key=lambda item: (item.contract_start, item.contract_stop))
    )
    return CausalAlphaV3SymbolSummary(
        fit_config_digest=fit_config_digest,
        symbol=symbol,
        episode_count=len(ordered),
        sample_count=_numeric_summary(tuple(item.sample_count for item in ordered)),
        rank_ic=_signed_summary(tuple(_rank(item) for item in ordered)),
        top_bottom_spread=_signed_summary(
            tuple(float(item.top_bottom_realized_spread) for item in ordered)
        ),
        direction_accuracy_excess=_signed_summary(
            tuple(_direction_excess(item) for item in ordered)
        ),
    )


def _win_fraction(left: Sequence[float], right: Sequence[float]) -> float:
    pairs = tuple(zip(left, right, strict=True))
    if not pairs:
        raise ValueError("V3 signal fit comparison has no paired episodes")
    return float(sum(a > b for a, b in pairs) / len(pairs))


def _fit_comparison(
    left_fit: str,
    right_fit: str,
    episodes_by_fit: Mapping[
        str, Mapping[tuple[int, int], CausalAlphaV3EpisodeSummary]
    ],
) -> CausalAlphaV3FitComparison:
    left_map = episodes_by_fit[left_fit]
    right_map = episodes_by_fit[right_fit]
    common = tuple(sorted(set(left_map) & set(right_map)))
    if not common:
        raise ValueError("V3 signal fit comparison has no common episodes")
    left_rank = tuple(left_map[key].rank_ic for key in common)
    right_rank = tuple(right_map[key].rank_ic for key in common)
    left_spread = tuple(left_map[key].top_bottom_spread for key in common)
    right_spread = tuple(right_map[key].top_bottom_spread for key in common)
    left_direction = tuple(left_map[key].direction_accuracy_excess for key in common)
    right_direction = tuple(right_map[key].direction_accuracy_excess for key in common)
    return CausalAlphaV3FitComparison(
        left_fit_config_digest=left_fit,
        right_fit_config_digest=right_fit,
        common_episode_count=len(common),
        mean_rank_ic_delta=float(
            fmean(
                left - right for left, right in zip(left_rank, right_rank, strict=True)
            )
        ),
        mean_top_bottom_spread_delta=float(
            fmean(
                left - right
                for left, right in zip(left_spread, right_spread, strict=True)
            )
        ),
        mean_direction_accuracy_excess_delta=float(
            fmean(
                left - right
                for left, right in zip(left_direction, right_direction, strict=True)
            )
        ),
        left_rank_ic_win_fraction=_win_fraction(left_rank, right_rank),
        left_top_bottom_spread_win_fraction=_win_fraction(left_spread, right_spread),
        left_direction_accuracy_excess_win_fraction=_win_fraction(
            left_direction, right_direction
        ),
    )


def _require_exact_fields(
    raw: Mapping[str, Any], *, required: set[str], field: str
) -> dict[str, Any]:
    values = dict(raw)
    if set(values) != required:
        missing = sorted(required - set(values))
        unknown = sorted(set(values) - required)
        raise ValueError(
            f"{field} fields mismatch; missing={missing}, unknown={unknown}"
        )
    return values


def _validate_content_digest(raw: Mapping[str, Any], *, field: str) -> str:
    values = dict(raw)
    digest = values.pop("artifact_digest", None)
    expected = content_digest(values)
    if not isinstance(digest, str) or digest != expected:
        raise ValueError(f"{field} digest mismatch")
    return digest


def _validate_signal_rejection(
    path: Path,
    *,
    manifest: CausalAlphaV3RunManifestV2,
    config: CausalAlphaV3ResearchConfig,
    metrics_by_fit: Mapping[str, Sequence[CausalAlphaV3SignalScopeMetric]],
    episodes_by_fit: Mapping[str, Sequence[CausalAlphaV3EpisodeSummary]],
) -> str:
    raw = _require_exact_fields(
        _load_json(path, field="V3 signal rejection"),
        required={
            "artifact_digest",
            "fit_results",
            "promotion_eligible",
            "schema_version",
        },
        field="V3 signal rejection",
    )
    rejection_digest = _validate_content_digest(raw, field="V3 signal rejection")
    if raw["schema_version"] != _SIGNAL_REJECTION_SCHEMA:
        raise ValueError("V3 signal rejection schema is unsupported")
    if raw["promotion_eligible"] is not False:
        raise ValueError("V3 signal rejection cannot be promotion eligible")
    fit_results = raw["fit_results"]
    if not isinstance(fit_results, list | tuple):
        raise ValueError("V3 signal rejection fit_results are invalid")
    expected_fits = set(metrics_by_fit)
    seen_fits: set[str] = set()
    expected_raw_count = (
        len(manifest.train_symbols) * config.nested_selection.signal_contract_count
    )
    for raw_result in fit_results:
        if not isinstance(raw_result, Mapping):
            raise ValueError("V3 signal rejection fit result is invalid")
        result = _require_exact_fields(
            raw_result,
            required={
                "evidence",
                "fit_config_digest",
                "passed",
                "promotion_eligible",
                "schema_version",
                "unavailable_scope_contract_digests",
            },
            field="V3 signal rejection fit result",
        )
        fit_digest = str(result["fit_config_digest"])
        if fit_digest not in expected_fits or fit_digest in seen_fits:
            raise ValueError("V3 signal rejection fit config identity drifted")
        seen_fits.add(fit_digest)
        if result["schema_version"] != _FIT_SIGNAL_RESULT_SCHEMA:
            raise ValueError("V3 signal rejection fit result schema is unsupported")
        if result["promotion_eligible"] is not False:
            raise ValueError("V3 signal fit result cannot be promotion eligible")
        result_passed = result["passed"]
        if not isinstance(result_passed, bool):
            raise ValueError("V3 signal rejection fit pass state is invalid")
        if result_passed:
            raise ValueError("V3 signal rejection contains an invalid rejected fit pass state")
        unavailable = result["unavailable_scope_contract_digests"]
        if not isinstance(unavailable, list | tuple):
            raise ValueError("V3 signal unavailable scope evidence is invalid")
        evidence_raw = result["evidence"]
        if evidence_raw is None:
            raise ValueError("V3 completed signal rejection lacks fit evidence")
        if not isinstance(evidence_raw, Mapping):
            raise ValueError("V3 signal rejection evidence is invalid")
        evidence = dict(evidence_raw)
        _validate_content_digest(evidence, field="V3 signal rejection evidence")
        required_evidence_fields = {
            "aggregation_mode",
            "artifact_digest",
            "direction_accuracy_excess",
            "expected_independent_episode_count",
            "expected_raw_scope_count",
            "gate_digest",
            "independence_unit",
            "independent_episode_count",
            "metric_digests",
            "passed",
            "promotion_eligible",
            "rank_ic",
            "raw_scope_count",
            "raw_scope_coverage",
            "rejection_reasons",
            "run_manifest_digest",
            "schema_version",
            "top_bottom_spread",
        }
        _require_exact_fields(
            evidence,
            required=required_evidence_fields,
            field="V3 signal rejection evidence",
        )
        evidence_passed = evidence["passed"]
        if not isinstance(evidence_passed, bool):
            raise ValueError("V3 signal rejection evidence pass state is invalid")
        if evidence["schema_version"] != _SIGNAL_GATE_SCHEMA:
            raise ValueError("V3 signal rejection gate evidence schema is unsupported")
        if evidence["run_manifest_digest"] != manifest.digest:
            raise ValueError("V3 signal rejection run manifest identity drifted")
        if evidence["gate_digest"] != config.signal_gate.digest:
            raise ValueError("V3 signal rejection gate identity drifted")
        if evidence["independence_unit"] != _SIGNAL_INDEPENDENCE_UNIT:
            raise ValueError("V3 signal rejection independence unit drifted")
        if evidence["aggregation_mode"] != _SIGNAL_AGGREGATION_MODE:
            raise ValueError("V3 signal rejection aggregation mode drifted")
        actual_metrics = tuple(metrics_by_fit[fit_digest])
        actual_episodes = tuple(episodes_by_fit[fit_digest])
        metric_digests = evidence["metric_digests"]
        if not isinstance(metric_digests, list | tuple):
            raise ValueError("V3 signal rejection metric digests are invalid")
        expected_metric_digests = {item.digest for item in actual_metrics}
        if (
            len(metric_digests) != len(expected_metric_digests)
            or len(set(metric_digests)) != len(metric_digests)
            or set(metric_digests) != expected_metric_digests
        ):
            raise ValueError("V3 signal rejection metric digests drifted")
        if int(evidence["raw_scope_count"]) != len(actual_metrics):
            raise ValueError("V3 signal rejection raw scope count drifted")
        if int(evidence["expected_raw_scope_count"]) != expected_raw_count:
            raise ValueError("V3 signal rejection expected raw scope count drifted")
        if int(evidence["independent_episode_count"]) != len(actual_episodes):
            raise ValueError("V3 signal rejection independent episode count drifted")
        if int(evidence["expected_independent_episode_count"]) != (
            config.nested_selection.signal_contract_count
        ):
            raise ValueError(
                "V3 signal rejection expected independent episode count drifted"
            )
        expected_coverage = len(actual_metrics) / float(expected_raw_count)
        if not math.isclose(
            float(evidence["raw_scope_coverage"]),
            expected_coverage,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("V3 signal rejection raw scope coverage drifted")
        if evidence_passed != result_passed:
            raise ValueError("V3 signal rejection pass state drifted")
        if evidence["promotion_eligible"] is not False:
            raise ValueError("V3 signal gate evidence cannot be promotion eligible")
    if seen_fits != expected_fits:
        raise ValueError("V3 signal rejection fit result scope is incomplete")
    return rejection_digest


def _unavailable_analyses() -> tuple[CausalAlphaV3UnavailableAnalysis, ...]:
    return (
        CausalAlphaV3UnavailableAnalysis(
            analysis="horizon_24h_vs_72h",
            reason=(
                "Signal V2 leaves persist only the blended forecast diagnostics; "
                "separate 24h/72h predictions and realized outcomes are absent."
            ),
        ),
        CausalAlphaV3UnavailableAnalysis(
            analysis="coefficient_cosine_similarity",
            reason=(
                "Signal V2 leaves persist pooled fit digests but not ridge coefficient "
                "vectors."
            ),
        ),
        CausalAlphaV3UnavailableAnalysis(
            analysis="coefficient_sign_flip_rate",
            reason=(
                "Signal V2 leaves persist pooled fit digests but not ridge coefficient "
                "vectors."
            ),
        ),
        CausalAlphaV3UnavailableAnalysis(
            analysis="prediction_distribution",
            reason=(
                "Signal V2 leaves persist forecast digests, not prediction values or "
                "distribution summaries."
            ),
        ),
        CausalAlphaV3UnavailableAnalysis(
            analysis="residual_rmse_by_episode",
            reason=(
                "Signal V2 leaves persist pooled fit digests, not per-fit residual RMSE "
                "sidecars."
            ),
        ),
    )


def load_causal_alpha_v3_signal_forensics(
    root: Path,
) -> CausalAlphaV3SignalForensicsReport:
    """Describe persisted Signal V2 evidence without refitting or replaying anything."""

    source_root = Path(root)
    manifest = CausalAlphaV3RunManifestV2.from_payload(
        _load_json(source_root / "run-manifest.json", field="V3 run manifest")
    )
    config = CausalAlphaV3ResearchConfig.from_mapping(
        _load_json(source_root / "authored-config.json", field="V3 authored config")
    )
    if config.digest != manifest.config_digest:
        raise ValueError("V3 signal forensics authored config identity drifted")

    fit_order = _fit_order(config)
    metadata = _fit_metadata(config)
    metrics = _load_metrics(
        source_root,
        manifest=manifest,
        config=config,
    )
    metrics_by_fit: dict[str, tuple[CausalAlphaV3SignalScopeMetric, ...]] = {}
    episodes_by_fit_list: dict[str, tuple[CausalAlphaV3EpisodeSummary, ...]] = {}
    episode_maps: dict[str, dict[tuple[int, int], CausalAlphaV3EpisodeSummary]] = {}
    fit_summaries: list[CausalAlphaV3FitSummary] = []
    all_episode_summaries: list[CausalAlphaV3EpisodeSummary] = []
    symbol_summaries: list[CausalAlphaV3SymbolSummary] = []
    observed_fits = {item.fit_config_digest for item in metrics}
    if observed_fits != set(fit_order):
        raise ValueError("V3 signal forensics fit record scope is incomplete")

    common_cluster_scope: set[tuple[int, int]] | None = None
    for fit_digest in fit_order:
        fit_metrics = tuple(
            sorted(
                (item for item in metrics if item.fit_config_digest == fit_digest),
                key=lambda item: (
                    item.contract_start,
                    item.contract_stop,
                    manifest.train_symbols.index(item.symbol),
                ),
            )
        )
        metrics_by_fit[fit_digest] = fit_metrics
        clusters = _cluster_metrics(fit_metrics, symbols=manifest.train_symbols)
        episodes = tuple(_episode_summary(fit_digest, cluster) for cluster in clusters)
        cluster_scope = {item.cluster_identity for item in episodes}
        if common_cluster_scope is None:
            common_cluster_scope = cluster_scope
        elif cluster_scope != common_cluster_scope:
            raise ValueError("V3 signal fit chronological episode scope drifted")
        episodes_by_fit_list[fit_digest] = episodes
        episode_maps[fit_digest] = {item.cluster_identity: item for item in episodes}
        fit_summaries.append(
            _fit_summary(
                fit_digest,
                fit_metrics,
                episodes,
                metadata=metadata[fit_digest],
                symbol_count=len(manifest.train_symbols),
            )
        )
        all_episode_summaries.extend(episodes)
        for symbol in manifest.train_symbols:
            symbol_summaries.append(
                _symbol_summary(
                    fit_digest,
                    symbol,
                    tuple(item for item in fit_metrics if item.symbol == symbol),
                )
            )

    if common_cluster_scope is None or not common_cluster_scope:
        raise ValueError("V3 signal forensics found no chronological episodes")

    comparisons = tuple(
        _fit_comparison(left, right, episode_maps)
        for left, right in combinations(fit_order, 2)
    )
    rejection_path = source_root / "signal" / "rejection.json"
    rejection_digest: str | None = None
    source_status = "records_only"
    if rejection_path.exists():
        rejection_digest = _validate_signal_rejection(
            rejection_path,
            manifest=manifest,
            config=config,
            metrics_by_fit=metrics_by_fit,
            episodes_by_fit=episodes_by_fit_list,
        )
        source_status = "rejected"

    return CausalAlphaV3SignalForensicsReport(
        run_manifest_digest=manifest.digest,
        config_digest=config.digest,
        train_symbols=manifest.train_symbols,
        fit_config_digests=fit_order,
        raw_scope_count=len(metrics),
        independent_episode_count=len(common_cluster_scope),
        source_signal_status=source_status,
        source_rejection_digest=rejection_digest,
        fit_summaries=tuple(fit_summaries),
        episode_summaries=tuple(all_episode_summaries),
        symbol_summaries=tuple(symbol_summaries),
        fit_comparisons=comparisons,
        unavailable_analyses=_unavailable_analyses(),
    )


__all__ = [
    "CausalAlphaV3EpisodeSummary",
    "CausalAlphaV3FitComparison",
    "CausalAlphaV3FitSummary",
    "CausalAlphaV3NumericSummary",
    "CausalAlphaV3SignalForensicsReport",
    "CausalAlphaV3SignedMetricSummary",
    "CausalAlphaV3SymbolSummary",
    "CausalAlphaV3TrendSummary",
    "CausalAlphaV3UnavailableAnalysis",
    "load_causal_alpha_v3_signal_forensics",
]
