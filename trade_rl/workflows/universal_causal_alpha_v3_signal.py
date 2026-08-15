"""Chronological signal-gate contracts for the causal alpha V3 research runner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaEpisodePartition,
)

_NESTED_PARTITION_SCHEMA: Final = "causal_alpha_v3_nested_partition_v1"
_SIGNAL_SCOPE_SCHEMA: Final = "causal_alpha_v3_signal_scope_v2"
_SIGNAL_GATE_SCHEMA: Final = "causal_alpha_v3_signal_gate_evidence_v2"
_BOOTSTRAP_SCHEMA: Final = "causal_alpha_v3_bootstrap_evidence_v1"
_SIGNAL_INDEPENDENCE_UNIT: Final = "chronological_episode"
_SIGNAL_AGGREGATION_MODE: Final = "cross_symbol_episode_mean"


@dataclass(frozen=True, slots=True)
class CausalAlphaV3NestedPartition:
    signal_contracts: tuple[OracleEpisodeContract, ...]
    economic_contracts: tuple[OracleEpisodeContract, ...]
    holdout_contract: OracleEpisodeContract
    digest: str = ""

    def __post_init__(self) -> None:
        signal = tuple(self.signal_contracts)
        economic = tuple(self.economic_contracts)
        if not signal or not economic:
            raise ValueError(
                "V3 nested partition requires signal and economic contracts"
            )
        values = (*signal, *economic, self.holdout_contract)
        if any(not isinstance(item, OracleEpisodeContract) for item in values):
            raise TypeError("V3 nested partition contracts are invalid")
        if len({item.digest for item in values}) != len(values):
            raise ValueError("V3 nested partition contract scopes overlap")
        if any(left.stop > right.start for left, right in zip(values, values[1:])):
            raise ValueError("V3 nested partition contracts are not chronological")
        if len({item.dataset_id for item in values}) != 1:
            raise ValueError("V3 nested partition dataset identity drifted")
        object.__setattr__(self, "signal_contracts", signal)
        object.__setattr__(self, "economic_contracts", economic)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 nested partition digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def signal_contract_digests(self) -> tuple[str, ...]:
        return tuple(item.digest for item in self.signal_contracts)

    @property
    def economic_contract_digests(self) -> tuple[str, ...]:
        return tuple(item.digest for item in self.economic_contracts)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "economic_contract_digests": self.economic_contract_digests,
            "holdout_contract_digest": self.holdout_contract.digest,
            "schema_version": _NESTED_PARTITION_SCHEMA,
            "signal_contract_digests": self.signal_contract_digests,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def split_causal_alpha_v3_partitions(
    partitions: Mapping[str, CausalAlphaEpisodePartition],
    *,
    train_symbols: tuple[str, ...],
    signal_contract_count: int,
    minimum_economic_contract_count: int,
) -> dict[str, CausalAlphaV3NestedPartition]:
    symbols = tuple(train_symbols)
    if (
        not symbols
        or len(set(symbols)) != len(symbols)
        or any(not item for item in symbols)
    ):
        raise ValueError("V3 train_symbols must be non-empty and unique")
    if set(partitions) != set(symbols):
        raise ValueError("V3 partitions must exactly match train_symbols")
    for field, value in (
        ("signal_contract_count", signal_contract_count),
        ("minimum_economic_contract_count", minimum_economic_contract_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")

    result: dict[str, CausalAlphaV3NestedPartition] = {}
    for symbol in symbols:
        partition = partitions[symbol]
        if not isinstance(partition, CausalAlphaEpisodePartition):
            raise TypeError("V3 partition type is invalid")
        selection = tuple(partition.selection_contracts)
        if len(selection) - signal_contract_count < minimum_economic_contract_count:
            raise ValueError(
                f"V3 nested partition leaves insufficient economic scope for {symbol}"
            )
        result[symbol] = CausalAlphaV3NestedPartition(
            signal_contracts=selection[:signal_contract_count],
            economic_contracts=selection[signal_contract_count:],
            holdout_contract=partition.holdout_contract,
        )
    return result


def non_overlapping_causal_alpha_v3_rows(
    *,
    decision_indices: object,
    label_end_indices: object,
    eligible_mask: object,
) -> np.ndarray:
    decisions = np.asarray(decision_indices, dtype=np.int64).reshape(-1)
    label_ends = np.asarray(label_end_indices, dtype=np.int64).reshape(-1)
    eligible = np.asarray(eligible_mask, dtype=np.bool_).reshape(-1)
    if (
        decisions.size == 0
        or decisions.shape != label_ends.shape
        or decisions.shape != eligible.shape
    ):
        raise ValueError("V3 signal cohort inputs must be non-empty and sample aligned")
    if np.any(decisions < 0) or np.any(label_ends[eligible] < decisions[eligible]):
        raise ValueError("V3 signal cohort intervals are invalid")
    if np.any(np.diff(decisions) < 0):
        raise ValueError("V3 signal cohort decisions must be chronological")

    selected: list[int] = []
    previous_end = -1
    for row in np.flatnonzero(eligible):
        decision = int(decisions[row])
        if decision <= previous_end:
            continue
        selected.append(int(row))
        previous_end = int(label_ends[row])
    result = np.asarray(selected, dtype=np.int64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CausalAlphaV3SignalScopeMetric:
    fit_config_digest: str
    symbol: str
    episode_index: int
    contract_start: int
    contract_stop: int
    contract_digest: str
    fit_digest: str
    forecast_digest: str
    sample_count: int
    rank_correlation: float | None
    direction_accuracy: float
    top_bottom_realized_spread: float
    cohort_indices: tuple[int, ...]
    digest: str = ""

    def __post_init__(self) -> None:
        for field in (
            "fit_config_digest",
            "contract_digest",
            "fit_digest",
            "forecast_digest",
        ):
            require_sha256(getattr(self, field), field=f"V3 signal {field}")
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("V3 signal symbol must be non-empty")
        if (
            isinstance(self.episode_index, bool)
            or not isinstance(self.episode_index, int)
            or self.episode_index < 0
        ):
            raise ValueError("V3 signal episode_index must be non-negative")
        if (
            isinstance(self.contract_start, bool)
            or not isinstance(self.contract_start, int)
            or isinstance(self.contract_stop, bool)
            or not isinstance(self.contract_stop, int)
            or self.contract_start < 0
            or self.contract_stop <= self.contract_start
        ):
            raise ValueError("V3 signal contract interval is invalid")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 2
        ):
            raise ValueError("V3 signal scope requires at least two samples")
        if self.rank_correlation is None or not math.isfinite(self.rank_correlation):
            raise ValueError("V3 signal rank correlation must be finite")
        if not -1.0 <= self.rank_correlation <= 1.0:
            raise ValueError("V3 signal rank correlation must be within [-1, 1]")
        if (
            not math.isfinite(self.direction_accuracy)
            or not 0.0 <= self.direction_accuracy <= 1.0
        ):
            raise ValueError("V3 signal direction accuracy must be within [0, 1]")
        if not math.isfinite(self.top_bottom_realized_spread):
            raise ValueError("V3 signal top-bottom spread must be finite")
        cohort = tuple(self.cohort_indices)
        if (
            len(cohort) != self.sample_count
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in cohort
            )
            or tuple(sorted(set(cohort))) != cohort
        ):
            raise ValueError("V3 signal cohort indices are invalid")
        object.__setattr__(self, "cohort_indices", cohort)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 signal scope digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def identity(self) -> tuple[str, str, int]:
        return (self.fit_config_digest, self.symbol, self.episode_index)

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
            "episode_index": self.episode_index,
            "fit_config_digest": self.fit_config_digest,
            "fit_digest": self.fit_digest,
            "forecast_digest": self.forecast_digest,
            "rank_correlation": self.rank_correlation,
            "sample_count": self.sample_count,
            "schema_version": _SIGNAL_SCOPE_SCHEMA,
            "symbol": self.symbol,
            "top_bottom_realized_spread": self.top_bottom_realized_spread,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV3BootstrapEvidence:
    mean: float
    p_value: float
    lower_ci: float
    upper_ci: float
    block_size: int
    digest: str = ""

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.mean, self.p_value, self.lower_ci, self.upper_ci)
        ):
            raise ValueError("V3 bootstrap evidence must be finite")
        if not 0.0 <= self.p_value <= 1.0:
            raise ValueError("V3 bootstrap p_value must be within [0, 1]")
        if self.lower_ci > self.upper_ci:
            raise ValueError("V3 bootstrap confidence interval is reversed")
        if (
            isinstance(self.block_size, bool)
            or not isinstance(self.block_size, int)
            or self.block_size <= 0
        ):
            raise ValueError("V3 bootstrap block_size must be positive")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 bootstrap digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "block_size": self.block_size,
            "lower_ci": self.lower_ci,
            "mean": self.mean,
            "p_value": self.p_value,
            "schema_version": _BOOTSTRAP_SCHEMA,
            "upper_ci": self.upper_ci,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAlphaV3SignalGateEvidence:
    metrics: tuple[CausalAlphaV3SignalScopeMetric, ...]
    raw_scope_count: int
    expected_raw_scope_count: int
    raw_scope_coverage: float
    independent_episode_count: int
    expected_independent_episode_count: int
    rank_ic: CausalAlphaV3BootstrapEvidence
    top_bottom_spread: CausalAlphaV3BootstrapEvidence
    direction_accuracy_excess: CausalAlphaV3BootstrapEvidence
    gate_digest: str
    passed: bool
    rejection_reasons: tuple[str, ...]
    independence_unit: str = _SIGNAL_INDEPENDENCE_UNIT
    aggregation_mode: str = _SIGNAL_AGGREGATION_MODE
    promotion_eligible: bool = False
    digest: str = ""

    def __post_init__(self) -> None:
        values = tuple(self.metrics)
        if not values or len({item.identity for item in values}) != len(values):
            raise ValueError("V3 signal evidence requires unique scope metrics")
        if (
            isinstance(self.raw_scope_count, bool)
            or not isinstance(self.raw_scope_count, int)
            or self.raw_scope_count != len(values)
            or self.raw_scope_count <= 0
        ):
            raise ValueError("V3 signal raw scope count is invalid")
        if (
            isinstance(self.expected_raw_scope_count, bool)
            or not isinstance(self.expected_raw_scope_count, int)
            or self.expected_raw_scope_count < self.raw_scope_count
        ):
            raise ValueError("V3 signal expected raw scope count is invalid")
        expected_coverage = self.raw_scope_count / float(self.expected_raw_scope_count)
        if (
            not math.isfinite(self.raw_scope_coverage)
            or not 0.0 < self.raw_scope_coverage <= 1.0
            or abs(self.raw_scope_coverage - expected_coverage) > 1e-12
        ):
            raise ValueError("V3 signal raw scope coverage is invalid")
        observed_clusters = len({item.cluster_identity for item in values})
        if (
            isinstance(self.independent_episode_count, bool)
            or not isinstance(self.independent_episode_count, int)
            or self.independent_episode_count != observed_clusters
            or self.independent_episode_count <= 0
        ):
            raise ValueError("V3 signal independent episode count is invalid")
        if (
            isinstance(self.expected_independent_episode_count, bool)
            or not isinstance(self.expected_independent_episode_count, int)
            or self.expected_independent_episode_count < self.independent_episode_count
            or self.expected_independent_episode_count > self.expected_raw_scope_count
        ):
            raise ValueError("V3 signal expected independent episode count is invalid")
        if self.independence_unit != _SIGNAL_INDEPENDENCE_UNIT:
            raise ValueError("V3 signal independence unit is invalid")
        if self.aggregation_mode != _SIGNAL_AGGREGATION_MODE:
            raise ValueError("V3 signal aggregation mode is invalid")
        for field in ("rank_ic", "top_bottom_spread", "direction_accuracy_excess"):
            if not isinstance(getattr(self, field), CausalAlphaV3BootstrapEvidence):
                raise TypeError(f"V3 signal {field} evidence is invalid")
        require_sha256(self.gate_digest, field="V3 signal gate_digest")
        if self.promotion_eligible:
            raise ValueError("V3 signal gate evidence cannot be promotion eligible")
        reasons = tuple(self.rejection_reasons)
        if self.passed == bool(reasons):
            raise ValueError("V3 signal pass state and rejection reasons disagree")
        object.__setattr__(self, "metrics", values)
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 signal gate evidence digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "aggregation_mode": self.aggregation_mode,
            "direction_accuracy_excess": self.direction_accuracy_excess.to_payload(),
            "expected_independent_episode_count": self.expected_independent_episode_count,
            "expected_raw_scope_count": self.expected_raw_scope_count,
            "gate_digest": self.gate_digest,
            "independence_unit": self.independence_unit,
            "independent_episode_count": self.independent_episode_count,
            "metric_digests": tuple(item.digest for item in self.metrics),
            "passed": self.passed,
            "promotion_eligible": self.promotion_eligible,
            "rank_ic": self.rank_ic.to_payload(),
            "raw_scope_count": self.raw_scope_count,
            "raw_scope_coverage": self.raw_scope_coverage,
            "rejection_reasons": self.rejection_reasons,
            "schema_version": _SIGNAL_GATE_SCHEMA,
            "top_bottom_spread": self.top_bottom_spread.to_payload(),
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


__all__ = [
    "CausalAlphaV3BootstrapEvidence",
    "CausalAlphaV3NestedPartition",
    "CausalAlphaV3SignalGateEvidence",
    "CausalAlphaV3SignalScopeMetric",
    "non_overlapping_causal_alpha_v3_rows",
    "split_causal_alpha_v3_partitions",
]
