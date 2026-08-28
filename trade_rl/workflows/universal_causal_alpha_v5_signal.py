"""Fail-closed selective slow Signal gate for Causal Alpha V5."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.evaluation.bootstrap import moving_block_mean_test
from trade_rl.learning.causal_alpha_diagnostics import (
    evaluate_causal_alpha_signal_diagnostics,
)
from trade_rl.learning.causal_alpha_v5 import (
    CausalAlphaV5CalibrationConfig,
    CausalAlphaV5SelectiveForecast,
    V5SelectiveState,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    non_overlapping_causal_alpha_v3_rows,
)
from trade_rl.workflows.universal_causal_alpha_v4_signal import (
    CausalAlphaV4SignalBootstrapEvidence,
)

CAUSAL_ALPHA_V5_SIGNAL_SCOPE_SCHEMA: Final = "causal_alpha_v5_signal_scope_v1"
CAUSAL_ALPHA_V5_SELECTIVE_SLOW_SCHEMA: Final = (
    "causal_alpha_v5_selective_slow_evidence_v1"
)
CAUSAL_ALPHA_V5_SIGNAL_EVIDENCE_SCHEMA: Final = "causal_alpha_v5_signal_evidence_v1"
_BOOTSTRAP_RESAMPLES: Final = 10000
_BOOTSTRAP_SEED: Final = 20260823
_BOOTSTRAP_BLOCK_SIZE: Final = 2
_EXPECTED_SCOPE_COUNT: Final = 72
_EXPECTED_EPISODE_COUNT: Final = 8
_EXPECTED_SYMBOL_COUNT: Final = 9


@dataclass(frozen=True, slots=True)
class CausalAlphaV5SignalScopeMetric:
    run_manifest_digest: str
    calibration_config_digest: str
    symbol: str
    episode_index: int
    contract_start: int
    contract_stop: int
    contract_digest: str
    calibration_fit_digest: str
    selective_forecast_digest: str
    raw_sample_count: int
    raw_direction_sample_count: int
    active_sample_count: int
    active_direction_sample_count: int
    active_coverage: float
    unconditional_rank_correlation: float
    unconditional_direction_accuracy: float
    selective_direction_accuracy: float
    unconditional_top_bottom_realized_spread: float
    raw_cohort_indices: tuple[int, ...]
    active_cohort_indices: tuple[int, ...]
    inactive_reason_counts: tuple[tuple[str, int], ...]
    schema_version: str = CAUSAL_ALPHA_V5_SIGNAL_SCOPE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        for name in (
            "run_manifest_digest",
            "calibration_config_digest",
            "contract_digest",
            "calibration_fit_digest",
            "selective_forecast_digest",
        ):
            require_sha256(getattr(self, name), field=f"V5 signal {name}")
        if not self.symbol or self.episode_index < 0:
            raise ValueError("V5 signal scope identity is invalid")
        if self.contract_start < 0 or self.contract_stop <= self.contract_start:
            raise ValueError("V5 signal contract interval is invalid")
        counts = (
            self.raw_sample_count,
            self.raw_direction_sample_count,
            self.active_sample_count,
            self.active_direction_sample_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise ValueError("V5 signal support is invalid")
        if self.raw_sample_count < 2 or not (
            self.active_direction_sample_count
            <= self.active_sample_count
            <= self.raw_sample_count
            and self.active_direction_sample_count
            <= self.raw_direction_sample_count
            <= self.raw_sample_count
        ):
            raise ValueError("V5 signal support relationships are invalid")
        expected_coverage = self.active_sample_count / self.raw_sample_count
        if not math.isclose(
            self.active_coverage, expected_coverage, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("V5 signal active coverage is invalid")
        for value in (
            self.unconditional_rank_correlation,
            self.unconditional_direction_accuracy,
            self.selective_direction_accuracy,
        ):
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError("V5 signal bounded metric is invalid")
        if (
            not 0.0 <= self.unconditional_direction_accuracy <= 1.0
            or not 0.0 <= self.selective_direction_accuracy <= 1.0
        ):
            raise ValueError("V5 signal direction accuracy is invalid")
        if not math.isfinite(self.unconditional_top_bottom_realized_spread):
            raise ValueError("V5 signal spread is invalid")
        raw = tuple(self.raw_cohort_indices)
        active = tuple(self.active_cohort_indices)
        if len(raw) != self.raw_sample_count or len(active) != self.active_sample_count:
            raise ValueError("V5 signal cohorts do not match support")
        if (
            tuple(sorted(set(raw))) != raw
            or tuple(sorted(set(active))) != active
            or not set(active) <= set(raw)
        ):
            raise ValueError("V5 signal cohorts are invalid")
        reason_counts = tuple(self.inactive_reason_counts)
        if tuple(sorted(reason_counts)) != reason_counts or any(
            reason
            not in {
                state.value
                for state in V5SelectiveState
                if state is not V5SelectiveState.ACTIVE
            }
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for reason, count in reason_counts
        ):
            raise ValueError("V5 signal inactive reason evidence is invalid")
        if (
            sum(count for _, count in reason_counts)
            != self.raw_direction_sample_count - self.active_direction_sample_count
        ):
            raise ValueError(
                "V5 signal inactive direction rows are not fully accounted"
            )
        if self.schema_version != CAUSAL_ALPHA_V5_SIGNAL_SCOPE_SCHEMA:
            raise ValueError("unsupported V5 signal scope schema")
        object.__setattr__(self, "raw_cohort_indices", raw)
        object.__setattr__(self, "active_cohort_indices", active)
        object.__setattr__(self, "inactive_reason_counts", reason_counts)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V5 signal scope digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def identity(self) -> tuple[str, int]:
        return (self.symbol, self.episode_index)

    @property
    def cluster_identity(self) -> tuple[int, int]:
        return (self.contract_start, self.contract_stop)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "digest"
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV5SelectiveSlowEvidence:
    metrics: tuple[CausalAlphaV5SignalScopeMetric, ...]
    run_manifest_digest: str
    calibration_config_digest: str
    raw_scope_count: int
    independent_episode_count: int
    symbol_count: int
    overall_active_coverage: float
    unconditional_rank_ic: CausalAlphaV4SignalBootstrapEvidence
    unconditional_top_bottom_spread: CausalAlphaV4SignalBootstrapEvidence
    unconditional_direction_accuracy_excess: CausalAlphaV4SignalBootstrapEvidence
    selective_direction_accuracy_excess: CausalAlphaV4SignalBootstrapEvidence
    passed: bool
    rejection_reasons: tuple[str, ...]
    schema_version: str = CAUSAL_ALPHA_V5_SELECTIVE_SLOW_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        metrics = tuple(self.metrics)
        if not metrics or len({metric.identity for metric in metrics}) != len(metrics):
            raise ValueError("V5 slow evidence metrics are missing or duplicated")
        require_sha256(self.run_manifest_digest, field="V5 slow run manifest digest")
        require_sha256(self.calibration_config_digest, field="V5 slow config digest")
        if {metric.run_manifest_digest for metric in metrics} != {
            self.run_manifest_digest
        }:
            raise ValueError("V5 slow run identity drifted")
        if {metric.calibration_config_digest for metric in metrics} != {
            self.calibration_config_digest
        }:
            raise ValueError("V5 slow config identity drifted")
        if self.raw_scope_count != len(metrics):
            raise ValueError("V5 slow raw scope count is invalid")
        if not 0.0 <= self.overall_active_coverage <= 1.0:
            raise ValueError("V5 slow overall active coverage is invalid")
        reasons = tuple(self.rejection_reasons)
        if self.passed == bool(reasons):
            raise ValueError("V5 slow pass state and reasons disagree")
        if self.schema_version != CAUSAL_ALPHA_V5_SELECTIVE_SLOW_SCHEMA:
            raise ValueError("unsupported V5 selective slow schema")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V5 selective slow digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "calibration_config_digest": self.calibration_config_digest,
            "independent_episode_count": self.independent_episode_count,
            "metric_digests": tuple(metric.digest for metric in self.metrics),
            "overall_active_coverage": self.overall_active_coverage,
            "passed": self.passed,
            "raw_scope_count": self.raw_scope_count,
            "rejection_reasons": self.rejection_reasons,
            "run_manifest_digest": self.run_manifest_digest,
            "schema_version": self.schema_version,
            "selective_direction_accuracy_excess_digest": self.selective_direction_accuracy_excess.digest,
            "symbol_count": self.symbol_count,
            "unconditional_direction_accuracy_excess_digest": self.unconditional_direction_accuracy_excess.digest,
            "unconditional_rank_ic_digest": self.unconditional_rank_ic.digest,
            "unconditional_top_bottom_spread_digest": self.unconditional_top_bottom_spread.digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV5SignalEvidence:
    slow: CausalAlphaV5SelectiveSlowEvidence
    v4_fast_lane_digest: str
    v4_fast_lane_passed: bool
    passed: bool
    rejection_reasons: tuple[str, ...]
    schema_version: str = CAUSAL_ALPHA_V5_SIGNAL_EVIDENCE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.v4_fast_lane_digest, field="V5 fast lane digest")
        reasons = tuple(self.rejection_reasons)
        if self.passed == bool(reasons):
            raise ValueError("V5 signal pass state and reasons disagree")
        if self.passed != (self.slow.passed and self.v4_fast_lane_passed):
            raise ValueError("V5 signal combined state is invalid")
        if self.schema_version != CAUSAL_ALPHA_V5_SIGNAL_EVIDENCE_SCHEMA:
            raise ValueError("unsupported V5 signal evidence schema")
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V5 signal evidence digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "passed": self.passed,
            "rejection_reasons": self.rejection_reasons,
            "schema_version": self.schema_version,
            "slow_digest": self.slow.digest,
            "v4_fast_lane_digest": self.v4_fast_lane_digest,
            "v4_fast_lane_passed": self.v4_fast_lane_passed,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def causal_alpha_v5_signal_diagnostic_payload(
    evidence: CausalAlphaV5SignalEvidence,
) -> dict[str, object]:
    """Expose the scalar and per-scope evidence hidden behind core digests."""

    if not isinstance(evidence, CausalAlphaV5SignalEvidence):
        raise TypeError("V5 signal diagnostics require V5 signal evidence")
    slow = evidence.slow
    return {
        "schema_version": "causal_alpha_v5_signal_diagnostics_v1",
        "signal_evidence_digest": evidence.digest,
        "slow_evidence_digest": slow.digest,
        "passed": evidence.passed,
        "rejection_reasons": evidence.rejection_reasons,
        "v4_fast_lane_digest": evidence.v4_fast_lane_digest,
        "v4_fast_lane_passed": evidence.v4_fast_lane_passed,
        "run_manifest_digest": slow.run_manifest_digest,
        "calibration_config_digest": slow.calibration_config_digest,
        "raw_scope_count": slow.raw_scope_count,
        "independent_episode_count": slow.independent_episode_count,
        "symbol_count": slow.symbol_count,
        "overall_active_coverage": slow.overall_active_coverage,
        "unconditional_rank_ic": slow.unconditional_rank_ic.to_payload(),
        "unconditional_top_bottom_spread": (
            slow.unconditional_top_bottom_spread.to_payload()
        ),
        "unconditional_direction_accuracy_excess": (
            slow.unconditional_direction_accuracy_excess.to_payload()
        ),
        "selective_direction_accuracy_excess": (
            slow.selective_direction_accuracy_excess.to_payload()
        ),
        "metrics": tuple(metric.to_payload() for metric in slow.metrics),
    }


def _aligned(value: object, *, rows: int, dtype: Any, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.shape != (rows,):
        raise ValueError(f"V5 signal {field} is not decision aligned")
    return array


def build_causal_alpha_v5_signal_scope_metric(
    *,
    run_manifest_digest: str,
    calibration_config_digest: str,
    symbol: str,
    episode_index: int,
    contract_start: int,
    contract_stop: int,
    contract_digest: str,
    selective_forecast: CausalAlphaV5SelectiveForecast,
    labels_24h: object,
    label_end_indices_24h: object,
    labels_72h: object,
    label_end_indices_72h: object,
) -> CausalAlphaV5SignalScopeMetric:
    """Build unconditional and selective metrics from one canonical slow cohort."""

    if selective_forecast.symbol != symbol:
        raise ValueError("V5 signal forecast symbol drifted")
    decisions = np.asarray(selective_forecast.decision_indices, dtype=np.int64)
    rows = int(decisions.size)
    labels24 = _aligned(labels_24h, rows=rows, dtype=np.float64, field="labels_24h")
    labels72 = _aligned(labels_72h, rows=rows, dtype=np.float64, field="labels_72h")
    ends24 = _aligned(
        label_end_indices_24h, rows=rows, dtype=np.int64, field="ends_24h"
    )
    ends72 = _aligned(
        label_end_indices_72h, rows=rows, dtype=np.int64, field="ends_72h"
    )
    prediction = np.asarray(selective_forecast.slow_return_calibrated)
    realized = 0.5 * (labels24 + labels72 / 3.0)
    eligible = (
        np.isfinite(labels24)
        & np.isfinite(labels72)
        & np.isfinite(prediction)
        & (decisions >= contract_start)
        & (decisions < contract_stop)
        & (ends24 >= decisions)
        & (ends72 >= decisions)
        & (ends24 < contract_stop)
        & (ends72 < contract_stop)
    )
    raw_rows = non_overlapping_causal_alpha_v3_rows(
        decision_indices=decisions, label_end_indices=ends72, eligible_mask=eligible
    )
    if raw_rows.size < 2:
        raise ValueError("V5 signal scope has insufficient raw support")
    raw_prediction = prediction[raw_rows]
    raw_realized = realized[raw_rows]
    diagnostics = evaluate_causal_alpha_signal_diagnostics(raw_prediction, raw_realized)
    if diagnostics.rank_correlation is None:
        raise ValueError("V5 signal rank correlation is undefined")
    direction_mask = np.sign(raw_realized) != 0.0
    raw_direction_support = int(np.count_nonzero(direction_mask))
    if raw_direction_support == 0:
        raise ValueError("V5 signal scope has no direction support")
    active_mask = np.asarray(selective_forecast.active_mask)[raw_rows]
    active_direction_mask = active_mask & direction_mask
    active_direction_support = int(np.count_nonzero(active_direction_mask))
    selective_accuracy = (
        0.0
        if active_direction_support == 0
        else float(
            np.mean(
                np.sign(
                    np.asarray(selective_forecast.slow_direction_raw)[raw_rows][
                        active_direction_mask
                    ]
                )
                == np.sign(raw_realized[active_direction_mask])
            )
        )
    )
    unconditional_accuracy = float(
        np.mean(
            np.sign(
                np.asarray(selective_forecast.slow_direction_raw)[raw_rows][
                    direction_mask
                ]
            )
            == np.sign(raw_realized[direction_mask])
        )
    )
    order = np.argsort(raw_prediction, kind="mergesort")
    bucket = max(1, raw_rows.size // 5)
    spread = float(
        np.mean(raw_realized[order[-bucket:]]) - np.mean(raw_realized[order[:bucket]])
    )
    inactive: dict[str, int] = defaultdict(int)
    raw_states = tuple(selective_forecast.states[int(row)] for row in raw_rows)
    for state, has_direction, is_active in zip(
        raw_states, direction_mask, active_mask, strict=True
    ):
        if has_direction and not is_active:
            inactive[state.value] += 1
    active_rows = raw_rows[active_mask]
    return CausalAlphaV5SignalScopeMetric(
        run_manifest_digest=run_manifest_digest,
        calibration_config_digest=calibration_config_digest,
        symbol=symbol,
        episode_index=episode_index,
        contract_start=contract_start,
        contract_stop=contract_stop,
        contract_digest=contract_digest,
        calibration_fit_digest=selective_forecast.calibration_fit_digest,
        selective_forecast_digest=selective_forecast.digest,
        raw_sample_count=int(raw_rows.size),
        raw_direction_sample_count=raw_direction_support,
        active_sample_count=int(active_rows.size),
        active_direction_sample_count=active_direction_support,
        active_coverage=float(active_rows.size / raw_rows.size),
        unconditional_rank_correlation=float(diagnostics.rank_correlation),
        unconditional_direction_accuracy=unconditional_accuracy,
        selective_direction_accuracy=selective_accuracy,
        unconditional_top_bottom_realized_spread=spread,
        raw_cohort_indices=tuple(int(decisions[row]) for row in raw_rows),
        active_cohort_indices=tuple(int(decisions[row]) for row in active_rows),
        inactive_reason_counts=tuple(sorted(inactive.items())),
    )


def _bootstrap(values: tuple[float, ...]) -> CausalAlphaV4SignalBootstrapEvidence:
    result = moving_block_mean_test(
        values,
        n_bootstrap=_BOOTSTRAP_RESAMPLES,
        seed=_BOOTSTRAP_SEED,
        block_size=_BOOTSTRAP_BLOCK_SIZE,
    )
    return CausalAlphaV4SignalBootstrapEvidence(
        mean=float(fmean(values)),
        p_value=result.p_value,
        lower_ci=result.lower_ci,
        upper_ci=result.upper_ci,
        block_size=result.block_size,
    )


def evaluate_causal_alpha_v5_signal_gate(
    metrics: tuple[CausalAlphaV5SignalScopeMetric, ...],
    *,
    expected_symbols: tuple[str, ...],
    v4_fast_lane_digest: str,
    v4_fast_lane_passed: bool,
    config: CausalAlphaV5CalibrationConfig,
) -> CausalAlphaV5SignalEvidence:
    """Require unchanged V4 fast Signal and the predeclared V5 slow gate."""

    values = tuple(metrics)
    if not values or len({metric.identity for metric in values}) != len(values):
        raise ValueError("V5 signal requires unique scope metrics")
    if not isinstance(config, CausalAlphaV5CalibrationConfig):
        raise TypeError("V5 signal calibration config is invalid")
    expected = tuple(expected_symbols)
    if len(expected) != _EXPECTED_SYMBOL_COUNT or len(set(expected)) != len(expected):
        raise ValueError("V5 signal requires exactly nine expected symbols")
    run_digests = {metric.run_manifest_digest for metric in values}
    config_digests = {metric.calibration_config_digest for metric in values}
    if len(run_digests) != 1 or config_digests != {config.digest}:
        raise ValueError("V5 signal run/config identity drifted")
    grouped: dict[tuple[int, int], list[CausalAlphaV5SignalScopeMetric]] = defaultdict(
        list
    )
    for metric in values:
        grouped[metric.cluster_identity].append(metric)
    rank_values: list[float] = []
    spread_values: list[float] = []
    direction_values: list[float] = []
    selective_values: list[float] = []
    for interval in sorted(grouped):
        cluster = grouped[interval]
        rank_values.append(
            float(fmean(metric.unconditional_rank_correlation for metric in cluster))
        )
        spread_values.append(
            float(
                fmean(
                    metric.unconditional_top_bottom_realized_spread
                    for metric in cluster
                )
            )
        )
        direction_values.append(
            float(
                fmean(
                    metric.unconditional_direction_accuracy - 0.5 for metric in cluster
                )
            )
        )
        selective_values.append(
            float(
                fmean(metric.selective_direction_accuracy - 0.5 for metric in cluster)
            )
        )
    rank = _bootstrap(tuple(rank_values))
    spread = _bootstrap(tuple(spread_values))
    direction = _bootstrap(tuple(direction_values))
    selective = _bootstrap(tuple(selective_values))
    observed_symbols = {metric.symbol for metric in values}
    active_total = sum(metric.active_sample_count for metric in values)
    raw_total = sum(metric.raw_sample_count for metric in values)
    coverage = active_total / raw_total
    reasons: list[str] = []
    if len(values) != _EXPECTED_SCOPE_COUNT:
        reasons.append("raw_scope_count")
    if len(grouped) != _EXPECTED_EPISODE_COUNT:
        reasons.append("independent_episode_count")
    if observed_symbols != set(expected):
        reasons.append("symbol_coverage")
    if {metric.episode_index for metric in values} != set(
        range(_EXPECTED_EPISODE_COUNT)
    ):
        reasons.append("episode_coverage")
    if rank.lower_ci < 0.0:
        reasons.append("unconditional_rank_ic_lower_ci")
    if spread.lower_ci < 0.0:
        reasons.append("unconditional_top_bottom_spread_lower_ci")
    if direction.mean < 0.0:
        reasons.append("unconditional_direction_accuracy_excess_mean")
    if selective.lower_ci < 0.0:
        reasons.append("selective_direction_accuracy_excess_lower_ci")
    if coverage < config.minimum_active_coverage:
        reasons.append("active_coverage")
    if any(
        metric.active_direction_sample_count
        < max(
            config.minimum_scope_active_count,
            math.ceil(
                config.minimum_scope_active_fraction * metric.raw_direction_sample_count
            ),
        )
        for metric in values
    ):
        reasons.append("scope_active_support")
    slow = CausalAlphaV5SelectiveSlowEvidence(
        metrics=values,
        run_manifest_digest=next(iter(run_digests)),
        calibration_config_digest=config.digest,
        raw_scope_count=len(values),
        independent_episode_count=len(grouped),
        symbol_count=len(observed_symbols),
        overall_active_coverage=coverage,
        unconditional_rank_ic=rank,
        unconditional_top_bottom_spread=spread,
        unconditional_direction_accuracy_excess=direction,
        selective_direction_accuracy_excess=selective,
        passed=not reasons,
        rejection_reasons=tuple(reasons),
    )
    combined_reasons = list(reasons)
    if not v4_fast_lane_passed:
        combined_reasons.append("v4_fast_4h")
    return CausalAlphaV5SignalEvidence(
        slow=slow,
        v4_fast_lane_digest=v4_fast_lane_digest,
        v4_fast_lane_passed=v4_fast_lane_passed,
        passed=not combined_reasons,
        rejection_reasons=tuple(combined_reasons),
    )


__all__ = [
    "CAUSAL_ALPHA_V5_SELECTIVE_SLOW_SCHEMA",
    "CAUSAL_ALPHA_V5_SIGNAL_EVIDENCE_SCHEMA",
    "CAUSAL_ALPHA_V5_SIGNAL_SCOPE_SCHEMA",
    "CausalAlphaV5SelectiveSlowEvidence",
    "CausalAlphaV5SignalEvidence",
    "CausalAlphaV5SignalScopeMetric",
    "build_causal_alpha_v5_signal_scope_metric",
    "causal_alpha_v5_signal_diagnostic_payload",
    "evaluate_causal_alpha_v5_signal_gate",
]
