from __future__ import annotations

from pathlib import Path


path = Path("trade_rl/workflows/universal_causal_alpha_v4_signal.py")
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one match, got {count}: {old[:120]!r}")
    text = text.replace(old, new, 1)


replace_once(
    "import math\nfrom dataclasses import dataclass\nfrom typing import Final, Mapping\n",
    "import math\n"
    "from collections import defaultdict\n"
    "from dataclasses import dataclass\n"
    "from enum import Enum\n"
    "from statistics import fmean\n"
    "from typing import Final, Mapping\n",
)
replace_once(
    "from trade_rl.artifacts.hashing import content_digest\nfrom trade_rl.domain.common import require_sha256\n",
    "from trade_rl.artifacts.hashing import content_digest\n"
    "from trade_rl.domain.common import require_sha256\n"
    "from trade_rl.evaluation.bootstrap import moving_block_mean_test\n"
    "from trade_rl.learning.causal_alpha_diagnostics import (\n"
    "    evaluate_causal_alpha_signal_diagnostics,\n"
    ")\n"
    "from trade_rl.learning.causal_alpha_v4 import CausalAlphaV4Forecast\n"
    "from trade_rl.workflows.universal_causal_alpha_v3_signal import (\n"
    "    non_overlapping_causal_alpha_v3_rows,\n"
    ")\n",
)
replace_once(
    'CAUSAL_ALPHA_V4_LIVENESS_SCHEMA: Final = "causal_alpha_v4_liveness_evidence_v1"\n',
    'CAUSAL_ALPHA_V4_LIVENESS_SCHEMA: Final = "causal_alpha_v4_liveness_evidence_v1"\n'
    'CAUSAL_ALPHA_V4_SIGNAL_SCOPE_SCHEMA: Final = "causal_alpha_v4_signal_scope_v1"\n'
    'CAUSAL_ALPHA_V4_SIGNAL_LANE_SCHEMA: Final = "causal_alpha_v4_signal_lane_evidence_v1"\n'
    'CAUSAL_ALPHA_V4_SIGNAL_EVIDENCE_SCHEMA: Final = "causal_alpha_v4_signal_evidence_v1"\n'
    'CAUSAL_ALPHA_V4_SIGNAL_BOOTSTRAP_SCHEMA: Final = "causal_alpha_v4_signal_bootstrap_v1"\n',
)

addition = r'''

class CausalAlphaV4SignalLane(str, Enum):
    FAST_4H = "fast_4h"
    SLOW_FUSED = "slow_fused"


_V4_SIGNAL_LANES: Final = (
    CausalAlphaV4SignalLane.FAST_4H,
    CausalAlphaV4SignalLane.SLOW_FUSED,
)


@dataclass(frozen=True, slots=True)
class CausalAlphaV4SignalGateConfig:
    independent_episode_count: int = 8
    minimum_rank_ic_lower_ci: float = 0.0
    minimum_top_bottom_spread_lower_ci: float = 0.0
    minimum_direction_accuracy_excess_lower_ci: float = 0.0
    bootstrap_resamples: int = 10000
    bootstrap_seed: int = 20260823
    bootstrap_block_size: int = 2

    def __post_init__(self) -> None:
        if self.independent_episode_count != 8:
            raise ValueError("V4 signal independent episode count must remain 8")
        if self.minimum_rank_ic_lower_ci != 0.0:
            raise ValueError("V4 signal rank lower bound must remain zero")
        if self.minimum_top_bottom_spread_lower_ci != 0.0:
            raise ValueError("V4 signal spread lower bound must remain zero")
        if self.minimum_direction_accuracy_excess_lower_ci != 0.0:
            raise ValueError("V4 signal direction lower bound must remain zero")
        if self.bootstrap_resamples != 10000:
            raise ValueError("V4 signal bootstrap resamples must remain 10000")
        if self.bootstrap_seed != 20260823:
            raise ValueError("V4 signal bootstrap seed must remain frozen")
        if self.bootstrap_block_size != 2:
            raise ValueError("V4 signal bootstrap block size must remain 2")

    @classmethod
    def from_mapping(cls, raw: object) -> "CausalAlphaV4SignalGateConfig":
        if not isinstance(raw, Mapping):
            raise ValueError("V4 signal gate config must be an object")
        values = dict(raw)
        expected = {
            "independent_episode_count",
            "minimum_rank_ic_lower_ci",
            "minimum_top_bottom_spread_lower_ci",
            "minimum_direction_accuracy_excess_lower_ci",
            "bootstrap_resamples",
            "bootstrap_seed",
            "bootstrap_block_size",
        }
        if set(values) != expected:
            missing = sorted(expected - set(values))
            unknown = sorted(set(values) - expected)
            raise ValueError(
                f"V4 signal gate fields mismatch; missing={missing}, unknown={unknown}"
            )
        return cls(
            independent_episode_count=int(values["independent_episode_count"]),
            minimum_rank_ic_lower_ci=float(values["minimum_rank_ic_lower_ci"]),
            minimum_top_bottom_spread_lower_ci=float(
                values["minimum_top_bottom_spread_lower_ci"]
            ),
            minimum_direction_accuracy_excess_lower_ci=float(
                values["minimum_direction_accuracy_excess_lower_ci"]
            ),
            bootstrap_resamples=int(values["bootstrap_resamples"]),
            bootstrap_seed=int(values["bootstrap_seed"]),
            bootstrap_block_size=int(values["bootstrap_block_size"]),
        )

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "bootstrap_block_size": self.bootstrap_block_size,
                "bootstrap_resamples": self.bootstrap_resamples,
                "bootstrap_seed": self.bootstrap_seed,
                "independent_episode_count": self.independent_episode_count,
                "minimum_direction_accuracy_excess_lower_ci": (
                    self.minimum_direction_accuracy_excess_lower_ci
                ),
                "minimum_rank_ic_lower_ci": self.minimum_rank_ic_lower_ci,
                "minimum_top_bottom_spread_lower_ci": (
                    self.minimum_top_bottom_spread_lower_ci
                ),
                "schema_version": "causal_alpha_v4_signal_gate_config_v1",
            }
        )


@dataclass(frozen=True, slots=True)
class CausalAlphaV4SignalScopeMetric:
    run_manifest_digest: str
    fit_config_digest: str
    lane: CausalAlphaV4SignalLane
    symbol: str
    episode_index: int
    contract_start: int
    contract_stop: int
    contract_digest: str
    fit_digest: str
    forecast_digest: str
    liveness_digest: str
    sample_count: int
    direction_sample_count: int
    rank_correlation: float
    direction_accuracy: float
    top_bottom_realized_spread: float
    cohort_indices: tuple[int, ...]
    schema_version: str = CAUSAL_ALPHA_V4_SIGNAL_SCOPE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "run_manifest_digest",
            "fit_config_digest",
            "contract_digest",
            "fit_digest",
            "forecast_digest",
            "liveness_digest",
        ):
            require_sha256(getattr(self, field_name), field=f"V4 signal {field_name}")
        lane = CausalAlphaV4SignalLane(self.lane)
        if not self.symbol:
            raise ValueError("V4 signal symbol must be non-empty")
        if isinstance(self.episode_index, bool) or not isinstance(self.episode_index, int) or self.episode_index < 0:
            raise ValueError("V4 signal episode index must be non-negative")
        if (
            isinstance(self.contract_start, bool)
            or not isinstance(self.contract_start, int)
            or isinstance(self.contract_stop, bool)
            or not isinstance(self.contract_stop, int)
            or self.contract_start < 0
            or self.contract_stop <= self.contract_start
        ):
            raise ValueError("V4 signal contract interval is invalid")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count < 2:
            raise ValueError("V4 signal scope requires at least two samples")
        if (
            isinstance(self.direction_sample_count, bool)
            or not isinstance(self.direction_sample_count, int)
            or not 1 <= self.direction_sample_count <= self.sample_count
        ):
            raise ValueError("V4 signal direction support is invalid")
        if not math.isfinite(self.rank_correlation) or not -1.0 <= self.rank_correlation <= 1.0:
            raise ValueError("V4 signal rank correlation is invalid")
        if not math.isfinite(self.direction_accuracy) or not 0.0 <= self.direction_accuracy <= 1.0:
            raise ValueError("V4 signal direction accuracy is invalid")
        if not math.isfinite(self.top_bottom_realized_spread):
            raise ValueError("V4 signal top-bottom spread must be finite")
        cohort = tuple(self.cohort_indices)
        if (
            len(cohort) != self.sample_count
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in cohort)
            or tuple(sorted(set(cohort))) != cohort
        ):
            raise ValueError("V4 signal cohort indices are invalid")
        if self.schema_version != CAUSAL_ALPHA_V4_SIGNAL_SCOPE_SCHEMA:
            raise ValueError("unsupported V4 signal scope schema")
        object.__setattr__(self, "lane", lane)
        object.__setattr__(self, "cohort_indices", cohort)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V4 signal scope digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def identity(self) -> tuple[str, str, str, int]:
        return (
            self.lane.value,
            self.fit_config_digest,
            self.symbol,
            self.episode_index,
        )

    @property
    def cluster_identity(self) -> tuple[int, int]:
        return (self.contract_start, self.contract_stop)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "cohort_indices": self.cohort_indices,
            "contract_digest": self.contract_digest,
            "contract_start": self.contract_start,
            "contract_stop": self.contract_stop,
            "direction_accuracy": self.direction_accuracy,
            "direction_sample_count": self.direction_sample_count,
            "episode_index": self.episode_index,
            "fit_config_digest": self.fit_config_digest,
            "fit_digest": self.fit_digest,
            "forecast_digest": self.forecast_digest,
            "lane": self.lane.value,
            "liveness_digest": self.liveness_digest,
            "rank_correlation": self.rank_correlation,
            "run_manifest_digest": self.run_manifest_digest,
            "sample_count": self.sample_count,
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "top_bottom_realized_spread": self.top_bottom_realized_spread,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV4SignalBootstrapEvidence:
    mean: float
    p_value: float
    lower_ci: float
    upper_ci: float
    block_size: int
    schema_version: str = CAUSAL_ALPHA_V4_SIGNAL_BOOTSTRAP_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.mean, self.p_value, self.lower_ci, self.upper_ci)):
            raise ValueError("V4 signal bootstrap values must be finite")
        if not 0.0 <= self.p_value <= 1.0 or self.lower_ci > self.upper_ci:
            raise ValueError("V4 signal bootstrap interval/probability is invalid")
        if isinstance(self.block_size, bool) or not isinstance(self.block_size, int) or self.block_size <= 0:
            raise ValueError("V4 signal bootstrap block size must be positive")
        if self.schema_version != CAUSAL_ALPHA_V4_SIGNAL_BOOTSTRAP_SCHEMA:
            raise ValueError("unsupported V4 signal bootstrap schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V4 signal bootstrap digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "block_size": self.block_size,
            "lower_ci": self.lower_ci,
            "mean": self.mean,
            "p_value": self.p_value,
            "schema_version": self.schema_version,
            "upper_ci": self.upper_ci,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV4LaneSignalEvidence:
    lane: CausalAlphaV4SignalLane
    metrics: tuple[CausalAlphaV4SignalScopeMetric, ...]
    run_manifest_digest: str
    raw_scope_count: int
    expected_raw_scope_count: int
    independent_episode_count: int
    rank_ic: CausalAlphaV4SignalBootstrapEvidence
    top_bottom_spread: CausalAlphaV4SignalBootstrapEvidence
    direction_accuracy_excess: CausalAlphaV4SignalBootstrapEvidence
    gate_digest: str
    passed: bool
    rejection_reasons: tuple[str, ...]
    promotion_eligible: bool = False
    schema_version: str = CAUSAL_ALPHA_V4_SIGNAL_LANE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        lane = CausalAlphaV4SignalLane(self.lane)
        metrics = tuple(self.metrics)
        if not metrics or any(metric.lane is not lane for metric in metrics):
            raise ValueError("V4 lane evidence metric scope is invalid")
        if len({metric.identity for metric in metrics}) != len(metrics):
            raise ValueError("V4 lane evidence contains duplicate metrics")
        require_sha256(self.run_manifest_digest, field="V4 signal run manifest digest")
        require_sha256(self.gate_digest, field="V4 signal gate digest")
        if {metric.run_manifest_digest for metric in metrics} != {self.run_manifest_digest}:
            raise ValueError("V4 lane evidence run identity drifted")
        if len({metric.fit_config_digest for metric in metrics}) != 1:
            raise ValueError("V4 lane evidence fit config drifted")
        if self.raw_scope_count != len(metrics) or self.raw_scope_count <= 0:
            raise ValueError("V4 lane raw scope count is invalid")
        if self.expected_raw_scope_count <= 0 or self.raw_scope_count > self.expected_raw_scope_count:
            raise ValueError("V4 lane expected raw scope count is invalid")
        observed_clusters = len({metric.cluster_identity for metric in metrics})
        if self.independent_episode_count != observed_clusters or self.independent_episode_count <= 0:
            raise ValueError("V4 lane independent episode count is invalid")
        for field_name in ("rank_ic", "top_bottom_spread", "direction_accuracy_excess"):
            if not isinstance(getattr(self, field_name), CausalAlphaV4SignalBootstrapEvidence):
                raise TypeError(f"V4 lane {field_name} evidence is invalid")
        reasons = tuple(self.rejection_reasons)
        if self.passed == bool(reasons):
            raise ValueError("V4 lane pass state and rejection reasons disagree")
        if self.promotion_eligible:
            raise ValueError("V4 lane signal evidence cannot be promotion eligible")
        if self.schema_version != CAUSAL_ALPHA_V4_SIGNAL_LANE_SCHEMA:
            raise ValueError("unsupported V4 lane signal schema")
        object.__setattr__(self, "lane", lane)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V4 lane signal digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "direction_accuracy_excess_digest": self.direction_accuracy_excess.digest,
            "expected_raw_scope_count": self.expected_raw_scope_count,
            "gate_digest": self.gate_digest,
            "independent_episode_count": self.independent_episode_count,
            "lane": self.lane.value,
            "metric_digests": tuple(metric.digest for metric in self.metrics),
            "passed": self.passed,
            "promotion_eligible": self.promotion_eligible,
            "rank_ic_digest": self.rank_ic.digest,
            "raw_scope_count": self.raw_scope_count,
            "rejection_reasons": self.rejection_reasons,
            "run_manifest_digest": self.run_manifest_digest,
            "schema_version": self.schema_version,
            "top_bottom_spread_digest": self.top_bottom_spread.digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV4SignalEvidence:
    fast_4h: CausalAlphaV4LaneSignalEvidence
    slow_fused: CausalAlphaV4LaneSignalEvidence
    gate_digest: str
    passed: bool
    rejection_reasons: tuple[str, ...]
    promotion_eligible: bool = False
    schema_version: str = CAUSAL_ALPHA_V4_SIGNAL_EVIDENCE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.fast_4h.lane is not CausalAlphaV4SignalLane.FAST_4H:
            raise ValueError("V4 fast signal evidence lane is invalid")
        if self.slow_fused.lane is not CausalAlphaV4SignalLane.SLOW_FUSED:
            raise ValueError("V4 slow signal evidence lane is invalid")
        require_sha256(self.gate_digest, field="V4 signal evidence gate digest")
        if self.fast_4h.gate_digest != self.gate_digest or self.slow_fused.gate_digest != self.gate_digest:
            raise ValueError("V4 signal evidence gate identity drifted")
        reasons = tuple(self.rejection_reasons)
        if self.passed == bool(reasons):
            raise ValueError("V4 signal evidence pass state and reasons disagree")
        if self.promotion_eligible:
            raise ValueError("V4 signal evidence cannot be promotion eligible")
        if self.schema_version != CAUSAL_ALPHA_V4_SIGNAL_EVIDENCE_SCHEMA:
            raise ValueError("unsupported V4 signal evidence schema")
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V4 signal evidence digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "fast_4h_digest": self.fast_4h.digest,
            "gate_digest": self.gate_digest,
            "passed": self.passed,
            "promotion_eligible": self.promotion_eligible,
            "rejection_reasons": self.rejection_reasons,
            "schema_version": self.schema_version,
            "slow_fused_digest": self.slow_fused.digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _signal_array(value: object, *, rows: int, field_name: str, dtype: object) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.shape != (rows,):
        raise ValueError(f"V4 signal {field_name} is not decision aligned")
    return array


def _v4_scope_metric(
    *,
    run_manifest_digest: str,
    fit_config_digest: str,
    lane: CausalAlphaV4SignalLane,
    symbol: str,
    episode_index: int,
    contract_start: int,
    contract_stop: int,
    contract_digest: str,
    fit_digest: str,
    forecast_digest: str,
    liveness_digest: str,
    decisions: np.ndarray,
    cohort_rows: np.ndarray,
    prediction: np.ndarray,
    realized: np.ndarray,
    direction_score: np.ndarray,
) -> CausalAlphaV4SignalScopeMetric:
    selected_prediction = prediction[cohort_rows]
    selected_realized = realized[cohort_rows]
    selected_direction = direction_score[cohort_rows]
    diagnostics = evaluate_causal_alpha_signal_diagnostics(
        selected_prediction, selected_realized
    )
    if diagnostics.rank_correlation is None:
        raise ValueError("V4 signal scope rank correlation is undefined")
    direction_mask = np.sign(selected_realized) != 0.0
    direction_support = int(np.count_nonzero(direction_mask))
    if direction_support == 0:
        raise ValueError("V4 signal scope has no non-zero direction support")
    direction_accuracy = float(
        np.mean(
            np.sign(selected_direction[direction_mask])
            == np.sign(selected_realized[direction_mask])
        )
    )
    order = np.argsort(selected_prediction, kind="mergesort")
    bucket = max(1, selected_prediction.size // 5)
    bottom = order[:bucket]
    top = order[-bucket:]
    spread = float(
        np.mean(selected_realized[top], dtype=np.float64)
        - np.mean(selected_realized[bottom], dtype=np.float64)
    )
    return CausalAlphaV4SignalScopeMetric(
        run_manifest_digest=run_manifest_digest,
        fit_config_digest=fit_config_digest,
        lane=lane,
        symbol=symbol,
        episode_index=episode_index,
        contract_start=contract_start,
        contract_stop=contract_stop,
        contract_digest=contract_digest,
        fit_digest=fit_digest,
        forecast_digest=forecast_digest,
        liveness_digest=liveness_digest,
        sample_count=int(cohort_rows.size),
        direction_sample_count=direction_support,
        rank_correlation=float(diagnostics.rank_correlation),
        direction_accuracy=direction_accuracy,
        top_bottom_realized_spread=spread,
        cohort_indices=tuple(int(decisions[row]) for row in cohort_rows),
    )


def build_causal_alpha_v4_signal_scope_metrics(
    *,
    run_manifest_digest: str,
    fit_config_digest: str,
    symbol: str,
    episode_index: int,
    contract_start: int,
    contract_stop: int,
    contract_digest: str,
    fit_digest: str,
    forecast: CausalAlphaV4Forecast,
    liveness_digests: Mapping[str, str],
    actionable_mask: object,
    labels_4h: object,
    label_end_indices_4h: object,
    labels_24h: object,
    label_end_indices_24h: object,
    labels_72h: object,
    label_end_indices_72h: object,
) -> Mapping[CausalAlphaV4SignalLane, CausalAlphaV4SignalScopeMetric]:
    """Build independent 4h and slow-fused canonical V4 signal metrics."""

    for field_name, digest in (
        ("run_manifest_digest", run_manifest_digest),
        ("fit_config_digest", fit_config_digest),
        ("contract_digest", contract_digest),
        ("fit_digest", fit_digest),
    ):
        require_sha256(digest, field=f"V4 signal {field_name}")
    if forecast.symbol != symbol or forecast.fit_digest != fit_digest:
        raise ValueError("V4 signal forecast identity drifted")
    if tuple(liveness_digests) != tuple(lane.value for lane in _V4_SIGNAL_LANES):
        raise ValueError("V4 signal liveness digests must cover fast and slow lanes")
    for lane in _V4_SIGNAL_LANES:
        require_sha256(
            str(liveness_digests[lane.value]),
            field=f"V4 signal {lane.value} liveness digest",
        )
    decisions = np.asarray(forecast.decision_indices, dtype=np.int64).reshape(-1)
    rows = int(decisions.size)
    if rows < 2 or np.any(decisions < contract_start) or np.any(decisions >= contract_stop):
        raise ValueError("V4 signal forecast decisions are outside the contract")
    actionable = _signal_array(
        actionable_mask, rows=rows, field_name="actionable_mask", dtype=np.bool_
    ).astype(np.bool_, copy=False)
    labels4 = _signal_array(labels_4h, rows=rows, field_name="labels_4h", dtype=np.float64)
    ends4 = _signal_array(
        label_end_indices_4h,
        rows=rows,
        field_name="label_end_indices_4h",
        dtype=np.int64,
    )
    labels24 = _signal_array(labels_24h, rows=rows, field_name="labels_24h", dtype=np.float64)
    ends24 = _signal_array(
        label_end_indices_24h,
        rows=rows,
        field_name="label_end_indices_24h",
        dtype=np.int64,
    )
    labels72 = _signal_array(labels_72h, rows=rows, field_name="labels_72h", dtype=np.float64)
    ends72 = _signal_array(
        label_end_indices_72h,
        rows=rows,
        field_name="label_end_indices_72h",
        dtype=np.int64,
    )
    beta_available = np.asarray(forecast.beta_available, dtype=np.bool_).reshape(-1)
    fast_prediction = np.asarray(forecast.final_predictions["4h"], dtype=np.float64)
    slow_prediction = 0.5 * (
        np.asarray(forecast.final_predictions["24h"], dtype=np.float64)
        + np.asarray(forecast.final_predictions["72h"], dtype=np.float64) / 3.0
    )
    fast_direction = np.asarray(forecast.direction_scores["4h"], dtype=np.float64)
    slow_direction = 0.5 * (
        np.asarray(forecast.direction_scores["24h"], dtype=np.float64)
        + np.asarray(forecast.direction_scores["72h"], dtype=np.float64)
    )
    slow_realized = 0.5 * (labels24 + labels72 / 3.0)

    fast_eligible = (
        actionable
        & beta_available
        & np.isfinite(labels4)
        & np.isfinite(fast_prediction)
        & np.isfinite(fast_direction)
        & (ends4 >= decisions)
        & (ends4 < contract_stop)
    )
    slow_eligible = (
        actionable
        & beta_available
        & np.isfinite(labels24)
        & np.isfinite(labels72)
        & np.isfinite(slow_prediction)
        & np.isfinite(slow_direction)
        & (ends24 >= decisions)
        & (ends72 >= decisions)
        & (ends24 < contract_stop)
        & (ends72 < contract_stop)
    )
    fast_rows = non_overlapping_causal_alpha_v3_rows(
        decision_indices=decisions,
        label_end_indices=ends4,
        eligible_mask=fast_eligible,
    )
    slow_rows = non_overlapping_causal_alpha_v3_rows(
        decision_indices=decisions,
        label_end_indices=ends72,
        eligible_mask=slow_eligible,
    )
    if fast_rows.size < 2 or slow_rows.size < 2:
        raise ValueError("V4 signal scope has insufficient non-overlapping support")

    fast_metric = _v4_scope_metric(
        run_manifest_digest=run_manifest_digest,
        fit_config_digest=fit_config_digest,
        lane=CausalAlphaV4SignalLane.FAST_4H,
        symbol=symbol,
        episode_index=episode_index,
        contract_start=contract_start,
        contract_stop=contract_stop,
        contract_digest=contract_digest,
        fit_digest=fit_digest,
        forecast_digest=forecast.digest,
        liveness_digest=str(liveness_digests["fast_4h"]),
        decisions=decisions,
        cohort_rows=fast_rows,
        prediction=fast_prediction,
        realized=labels4,
        direction_score=fast_direction,
    )
    slow_metric = _v4_scope_metric(
        run_manifest_digest=run_manifest_digest,
        fit_config_digest=fit_config_digest,
        lane=CausalAlphaV4SignalLane.SLOW_FUSED,
        symbol=symbol,
        episode_index=episode_index,
        contract_start=contract_start,
        contract_stop=contract_stop,
        contract_digest=contract_digest,
        fit_digest=fit_digest,
        forecast_digest=forecast.digest,
        liveness_digest=str(liveness_digests["slow_fused"]),
        decisions=decisions,
        cohort_rows=slow_rows,
        prediction=slow_prediction,
        realized=slow_realized,
        direction_score=slow_direction,
    )
    return {
        CausalAlphaV4SignalLane.FAST_4H: fast_metric,
        CausalAlphaV4SignalLane.SLOW_FUSED: slow_metric,
    }


def _v4_bootstrap(
    values: tuple[float, ...], gate: CausalAlphaV4SignalGateConfig
) -> CausalAlphaV4SignalBootstrapEvidence:
    result = moving_block_mean_test(
        values,
        n_bootstrap=gate.bootstrap_resamples,
        seed=gate.bootstrap_seed,
        block_size=gate.bootstrap_block_size,
    )
    return CausalAlphaV4SignalBootstrapEvidence(
        mean=float(fmean(values)),
        p_value=result.p_value,
        lower_ci=result.lower_ci,
        upper_ci=result.upper_ci,
        block_size=result.block_size,
    )


def _v4_episode_clusters(
    metrics: tuple[CausalAlphaV4SignalScopeMetric, ...],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    grouped: dict[tuple[int, int], list[CausalAlphaV4SignalScopeMetric]] = defaultdict(list)
    for metric in metrics:
        grouped[metric.cluster_identity].append(metric)
    ranks: list[float] = []
    spreads: list[float] = []
    directions: list[float] = []
    for interval in sorted(grouped):
        cluster = grouped[interval]
        if len({metric.symbol for metric in cluster}) != len(cluster):
            raise ValueError("V4 signal episode cluster contains duplicate symbols")
        if len({metric.fit_digest for metric in cluster}) != 1:
            raise ValueError("V4 signal episode cluster fit digest drifted")
        ranks.append(float(fmean(metric.rank_correlation for metric in cluster)))
        spreads.append(float(fmean(metric.top_bottom_realized_spread for metric in cluster)))
        directions.append(float(fmean(metric.direction_accuracy - 0.5 for metric in cluster)))
    return tuple(ranks), tuple(spreads), tuple(directions)


def _evaluate_v4_lane(
    metrics: tuple[CausalAlphaV4SignalScopeMetric, ...],
    *,
    lane: CausalAlphaV4SignalLane,
    expected_raw_scope_count: int,
    gate: CausalAlphaV4SignalGateConfig,
) -> CausalAlphaV4LaneSignalEvidence:
    if not metrics or any(metric.lane is not lane for metric in metrics):
        raise ValueError("V4 signal lane metrics are unavailable or mixed")
    if expected_raw_scope_count <= 0 or len(metrics) > expected_raw_scope_count:
        raise ValueError("V4 signal expected raw scope count is invalid")
    run_digests = {metric.run_manifest_digest for metric in metrics}
    fit_config_digests = {metric.fit_config_digest for metric in metrics}
    if len(run_digests) != 1 or len(fit_config_digests) != 1:
        raise ValueError("V4 signal lane run/fit identity drifted")
    ranks, spreads, directions = _v4_episode_clusters(metrics)
    independent_count = len(ranks)
    rank = _v4_bootstrap(ranks, gate)
    spread = _v4_bootstrap(spreads, gate)
    direction = _v4_bootstrap(directions, gate)
    reasons: list[str] = []
    if len(metrics) != expected_raw_scope_count:
        reasons.append("raw_scope_count")
    if independent_count != gate.independent_episode_count:
        reasons.append("independent_episode_count")
    if rank.lower_ci < gate.minimum_rank_ic_lower_ci:
        reasons.append("rank_ic_lower_ci")
    if spread.lower_ci < gate.minimum_top_bottom_spread_lower_ci:
        reasons.append("top_bottom_spread_lower_ci")
    if direction.lower_ci < gate.minimum_direction_accuracy_excess_lower_ci:
        reasons.append("direction_accuracy_excess_lower_ci")
    return CausalAlphaV4LaneSignalEvidence(
        lane=lane,
        metrics=metrics,
        run_manifest_digest=next(iter(run_digests)),
        raw_scope_count=len(metrics),
        expected_raw_scope_count=expected_raw_scope_count,
        independent_episode_count=independent_count,
        rank_ic=rank,
        top_bottom_spread=spread,
        direction_accuracy_excess=direction,
        gate_digest=gate.digest,
        passed=not reasons,
        rejection_reasons=tuple(reasons),
    )


def evaluate_causal_alpha_v4_signal_gate(
    metrics: tuple[CausalAlphaV4SignalScopeMetric, ...],
    *,
    expected_raw_scope_count_per_lane: int,
    gate: CausalAlphaV4SignalGateConfig,
) -> CausalAlphaV4SignalEvidence:
    """Require both independent fast and slow signal lanes to clear the frozen gate."""

    values = tuple(metrics)
    if not values or len({metric.identity for metric in values}) != len(values):
        raise ValueError("V4 signal gate requires unique scope metrics")
    if not isinstance(gate, CausalAlphaV4SignalGateConfig):
        raise TypeError("V4 signal gate config is invalid")
    lane_metrics = {
        lane: tuple(metric for metric in values if metric.lane is lane)
        for lane in _V4_SIGNAL_LANES
    }
    fast = _evaluate_v4_lane(
        lane_metrics[CausalAlphaV4SignalLane.FAST_4H],
        lane=CausalAlphaV4SignalLane.FAST_4H,
        expected_raw_scope_count=expected_raw_scope_count_per_lane,
        gate=gate,
    )
    slow = _evaluate_v4_lane(
        lane_metrics[CausalAlphaV4SignalLane.SLOW_FUSED],
        lane=CausalAlphaV4SignalLane.SLOW_FUSED,
        expected_raw_scope_count=expected_raw_scope_count_per_lane,
        gate=gate,
    )
    reasons = tuple(
        f"{evidence.lane.value}:{reason}"
        for evidence in (fast, slow)
        for reason in evidence.rejection_reasons
    )
    return CausalAlphaV4SignalEvidence(
        fast_4h=fast,
        slow_fused=slow,
        gate_digest=gate.digest,
        passed=fast.passed and slow.passed,
        rejection_reasons=reasons,
    )
'''

replace_once("\n\n__all__ = [\n", addition + "\n\n__all__ = [\n")
replace_once(
    '    "CAUSAL_ALPHA_V4_LIVENESS_SCHEMA",\n',
    '    "CAUSAL_ALPHA_V4_LIVENESS_SCHEMA",\n'
    '    "CAUSAL_ALPHA_V4_SIGNAL_BOOTSTRAP_SCHEMA",\n'
    '    "CAUSAL_ALPHA_V4_SIGNAL_EVIDENCE_SCHEMA",\n'
    '    "CAUSAL_ALPHA_V4_SIGNAL_LANE_SCHEMA",\n'
    '    "CAUSAL_ALPHA_V4_SIGNAL_SCOPE_SCHEMA",\n',
)
replace_once(
    '    "CausalAlphaV4LivenessEvidence",\n',
    '    "CausalAlphaV4LivenessEvidence",\n'
    '    "CausalAlphaV4LaneSignalEvidence",\n'
    '    "CausalAlphaV4SignalBootstrapEvidence",\n'
    '    "CausalAlphaV4SignalEvidence",\n'
    '    "CausalAlphaV4SignalGateConfig",\n'
    '    "CausalAlphaV4SignalLane",\n'
    '    "CausalAlphaV4SignalScopeMetric",\n',
)
replace_once(
    '    "build_causal_alpha_v4_liveness_evidence",\n',
    '    "build_causal_alpha_v4_liveness_evidence",\n'
    '    "build_causal_alpha_v4_signal_scope_metrics",\n'
    '    "evaluate_causal_alpha_v4_signal_gate",\n',
)
path.write_text(text, encoding="utf-8")
