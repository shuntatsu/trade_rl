"""Closed causal-prefix feature preprocessing; no policy/runtime consumer."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.strategies.rl.ppo_normalization import PPOFeatureNormalizer

_CLOCKS = ("fit_last_event_time_ns", "fit_as_of_ns", "policy_start_time_ns")
_DIGESTS = (
    "feature_config_digest",
    "source_normalization_digest",
    "fit_consumption_digest",
)


def _clock(value: object) -> int:
    if type(value) is not int or not -(2**63) < value < 2**63:
        raise ValueError("preprocessing clocks must be native non-NaT nanoseconds")
    return value


def _native_json(value: object) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _native_json(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("preprocessing JSON keys must be native strings")
            _native_json(item)
        return
    raise ValueError("preprocessing payload must contain native finite JSON values")


@dataclass(frozen=True, slots=True, kw_only=True)
class AllocationFeaturePreprocessing:
    normalizer: PPOFeatureNormalizer
    feature_config_digest: str
    source_normalization_digest: str
    fit_last_event_time_ns: int
    fit_as_of_ns: int
    policy_start_index: int
    policy_start_time_ns: int
    admitted_row_indices: tuple[tuple[int, ...], ...]
    fit_consumption_digest: str

    def __post_init__(self) -> None:
        if type(self.normalizer) is not PPOFeatureNormalizer:
            raise ValueError("preprocessing requires the frozen numerical statistics")
        statistics = self.normalizer.to_payload()
        _native_json(statistics)
        object.__setattr__(
            self,
            "normalizer",
            PPOFeatureNormalizer.from_payload(
                json.loads(canonical_json_bytes(statistics))
            ),
        )
        if any(scale <= 1e-12 for scale in self.normalizer.scale):
            raise ValueError("scale at or below the floor must use fallback 1.0")
        for name in _DIGESTS:
            if type(getattr(self, name)) is not str:
                raise ValueError("preprocessing digests must be native strings")
            require_sha256(getattr(self, name), field=name)
        for name in _CLOCKS:
            _clock(getattr(self, name))
        if (
            not self.fit_last_event_time_ns
            < self.fit_as_of_ns
            <= self.policy_start_time_ns
        ):
            raise ValueError("prefix events must precede fit and fit precede policy")
        if (
            type(self.policy_start_index) is not int
            or self.policy_start_index < self.normalizer.stop_index
        ):
            raise ValueError("prefix fit must end before the policy decision window")
        rows = self.admitted_row_indices
        if type(rows) is not tuple or len(rows) != len(
            self.normalizer.fit_symbol_indices
        ):
            raise ValueError("admitted rows must match the immutable fit symbol roster")
        for row, counts in zip(rows, self.normalizer.usable_counts):
            if (
                type(row) is not tuple
                or not row
                or any(
                    type(i) is not int
                    or not self.normalizer.start_index <= i < self.normalizer.stop_index
                    for i in row
                )
                or row != tuple(sorted(set(row)))
                or any(count != len(row) for count in counts)
            ):
                raise ValueError(
                    "joint admitted rows must reconcile every feature count"
                )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "allocation_feature_preprocessing_v1",
            "statistics": self.normalizer.to_payload(),
            **{
                name: getattr(self, name)
                for name in (*_DIGESTS, *_CLOCKS, "policy_start_index")
            },
            "admitted_row_indices": [list(row) for row in self.admitted_row_indices],
            "excluded_row_counts": [
                self.normalizer.stop_index - self.normalizer.start_index - len(row)
                for row in self.admitted_row_indices
            ],
            "row_admission": "all_selected_finite_available_at_row_clock_v1",
            "weighting": "equal_symbol_population_variance_v1",
            "scale_floor": 1e-12,
            "scale_fallback": 1.0,
            "transform": "frozen_float64_standardization_guarded_float32_v1",
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload())

    @classmethod
    def from_payload(cls, value: object) -> AllocationFeaturePreprocessing:
        try:
            _native_json(value)
            if not isinstance(value, dict):
                raise ValueError("preprocessing payload must be a JSON mapping")
            declaration = cls(
                normalizer=PPOFeatureNormalizer.from_payload(value["statistics"]),
                admitted_row_indices=tuple(
                    tuple(row) for row in value["admitted_row_indices"]
                ),
                **{
                    name: value[name]
                    for name in (*_DIGESTS, *_CLOCKS, "policy_start_index")
                },
            )
            if canonical_json_bytes(value) != canonical_json_bytes(
                declaration.payload()
            ):
                raise ValueError("preprocessing differs from its closed declaration")
            return declaration
        except (KeyError, TypeError, OverflowError, RecursionError) as error:
            raise ValueError("invalid allocation preprocessing payload") from error

    def transform(
        self,
        feature_values: tuple[float, ...],
        *,
        feature_names: tuple[str, ...],
        feature_config_digest: str,
        source_normalization_digest: str,
        decision_time_ns: int,
    ) -> np.ndarray:
        """Only selected live feature values; never an account/terminal vector."""
        if (
            feature_names != self.normalizer.feature_names
            or feature_config_digest != self.feature_config_digest
            or source_normalization_digest != self.source_normalization_digest
            or _clock(decision_time_ns) < self.fit_as_of_ns
        ):
            raise ValueError("application features/build/clock differ from frozen fit")
        if type(feature_values) is not tuple or any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
            for v in feature_values
        ):
            raise ValueError("selected features must be a tuple of nonboolean numbers")
        try:
            result = self.normalizer.transform(
                np.asarray(feature_values, dtype=np.float64),
                np.ones(len(feature_values), dtype=bool),
            )
            return result.astype(np.float32)
        except (TypeError, OverflowError, FloatingPointError) as error:
            raise ValueError(
                "selected values exceed finite feature reporting"
            ) from error


__all__ = ["AllocationFeaturePreprocessing"]
