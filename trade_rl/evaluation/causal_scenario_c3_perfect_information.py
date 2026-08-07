"""Compatibility proof for C3 Perfect-Information comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.evaluation.causal_scenario_c3_contracts import (
    PerfectInformationComparison,
    PerfectInformationComparisonReason,
)

PERFECT_INFORMATION_COMPATIBILITY_SCHEMA: Final = (
    "causal_scenario_c3_perfect_information_compatibility_v1"
)


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _readonly_vector(
    name: str,
    value: object,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64).copy(order="C")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive")
    if non_negative and np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    array[array == 0.0] = 0.0
    array.setflags(write=False)
    return array


def _array_payload(value: np.ndarray) -> dict[str, object]:
    return {
        "dtype": value.dtype.str,
        "shape": tuple(int(size) for size in value.shape),
        "values": value.tolist(),
    }


@dataclass(frozen=True, slots=True)
class PerfectInformationCompatibilityEvidence:
    causal_period_digest: str
    bound_period_digest: str
    causal_return_matrix_digest: str
    bound_return_matrix_digest: str
    causal_initial_weights: np.ndarray
    bound_initial_weights: np.ndarray
    causal_aum: float
    bound_aum: float
    causal_max_abs_weight: np.ndarray
    bound_max_abs_weight: np.ndarray
    causal_max_gross: float
    bound_max_gross: float
    causal_max_net_exposure: float | None
    bound_max_net_exposure: float | None
    causal_transaction_cost_rate: np.ndarray
    bound_transaction_cost_rate: np.ndarray
    causal_liquidation_cost_rate: np.ndarray
    bound_liquidation_cost_rate: np.ndarray
    bound_result_digest: str
    bound_log_return: float
    causal_log_return: float
    tolerance: float = 1e-10
    schema_version: str = PERFECT_INFORMATION_COMPATIBILITY_SCHEMA

    def __post_init__(self) -> None:
        for field in (
            "causal_period_digest",
            "bound_period_digest",
            "causal_return_matrix_digest",
            "bound_return_matrix_digest",
            "bound_result_digest",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(str(getattr(self, field)), field=field),
            )
        causal_weights = _readonly_vector(
            "causal_initial_weights", self.causal_initial_weights
        )
        bound_weights = _readonly_vector(
            "bound_initial_weights", self.bound_initial_weights
        )
        if causal_weights.shape != bound_weights.shape:
            raise ValueError("initial weight dimensions must match")
        dimension = causal_weights.size
        causal_abs = _readonly_vector(
            "causal_max_abs_weight", self.causal_max_abs_weight, positive=True
        )
        bound_abs = _readonly_vector(
            "bound_max_abs_weight", self.bound_max_abs_weight, positive=True
        )
        causal_transaction = _readonly_vector(
            "causal_transaction_cost_rate",
            self.causal_transaction_cost_rate,
            non_negative=True,
        )
        bound_transaction = _readonly_vector(
            "bound_transaction_cost_rate",
            self.bound_transaction_cost_rate,
            non_negative=True,
        )
        causal_liquidation = _readonly_vector(
            "causal_liquidation_cost_rate",
            self.causal_liquidation_cost_rate,
            non_negative=True,
        )
        bound_liquidation = _readonly_vector(
            "bound_liquidation_cost_rate",
            self.bound_liquidation_cost_rate,
            non_negative=True,
        )
        for name, array in (
            ("causal_max_abs_weight", causal_abs),
            ("bound_max_abs_weight", bound_abs),
            ("causal_transaction_cost_rate", causal_transaction),
            ("bound_transaction_cost_rate", bound_transaction),
            ("causal_liquidation_cost_rate", causal_liquidation),
            ("bound_liquidation_cost_rate", bound_liquidation),
        ):
            if array.shape != (dimension,):
                raise ValueError(f"{name} dimension mismatch")
        for field in ("causal_aum", "bound_aum", "causal_max_gross", "bound_max_gross"):
            value = _finite_float(field, getattr(self, field))
            if value <= 0.0:
                raise ValueError(f"{field} must be positive")
            object.__setattr__(self, field, value)
        for field in ("causal_max_net_exposure", "bound_max_net_exposure"):
            value = getattr(self, field)
            if value is not None:
                normalized = _finite_float(field, value)
                if normalized < 0.0:
                    raise ValueError(f"{field} must be non-negative")
                object.__setattr__(self, field, normalized)
        object.__setattr__(
            self,
            "bound_log_return",
            _finite_float("bound_log_return", self.bound_log_return),
        )
        object.__setattr__(
            self,
            "causal_log_return",
            _finite_float("causal_log_return", self.causal_log_return),
        )
        tolerance = _finite_float("tolerance", self.tolerance)
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if self.schema_version != PERFECT_INFORMATION_COMPATIBILITY_SCHEMA:
            raise ValueError("unsupported Perfect Information compatibility schema")
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "causal_initial_weights", causal_weights)
        object.__setattr__(self, "bound_initial_weights", bound_weights)
        object.__setattr__(self, "causal_max_abs_weight", causal_abs)
        object.__setattr__(self, "bound_max_abs_weight", bound_abs)
        object.__setattr__(self, "causal_transaction_cost_rate", causal_transaction)
        object.__setattr__(self, "bound_transaction_cost_rate", bound_transaction)
        object.__setattr__(self, "causal_liquidation_cost_rate", causal_liquidation)
        object.__setattr__(self, "bound_liquidation_cost_rate", bound_liquidation)

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "bound_aum": self.bound_aum,
                "bound_initial_weights": _array_payload(self.bound_initial_weights),
                "bound_liquidation_cost_rate": _array_payload(
                    self.bound_liquidation_cost_rate
                ),
                "bound_log_return": self.bound_log_return,
                "bound_max_abs_weight": _array_payload(self.bound_max_abs_weight),
                "bound_max_gross": self.bound_max_gross,
                "bound_max_net_exposure": self.bound_max_net_exposure,
                "bound_period_digest": self.bound_period_digest,
                "bound_result_digest": self.bound_result_digest,
                "bound_return_matrix_digest": self.bound_return_matrix_digest,
                "bound_transaction_cost_rate": _array_payload(
                    self.bound_transaction_cost_rate
                ),
                "causal_aum": self.causal_aum,
                "causal_initial_weights": _array_payload(self.causal_initial_weights),
                "causal_liquidation_cost_rate": _array_payload(
                    self.causal_liquidation_cost_rate
                ),
                "causal_log_return": self.causal_log_return,
                "causal_max_abs_weight": _array_payload(self.causal_max_abs_weight),
                "causal_max_gross": self.causal_max_gross,
                "causal_max_net_exposure": self.causal_max_net_exposure,
                "causal_period_digest": self.causal_period_digest,
                "causal_return_matrix_digest": self.causal_return_matrix_digest,
                "causal_transaction_cost_rate": _array_payload(
                    self.causal_transaction_cost_rate
                ),
                "schema_version": self.schema_version,
                "tolerance": self.tolerance,
            }
        )


def _not_comparable(
    evidence: PerfectInformationCompatibilityEvidence,
    reason: PerfectInformationComparisonReason,
) -> PerfectInformationComparison:
    return PerfectInformationComparison.not_comparable(
        reason,
        compatibility_evidence_digest=evidence.digest,
    )


def evaluate_perfect_information_compatibility(
    evidence: PerfectInformationCompatibilityEvidence,
) -> PerfectInformationComparison:
    """Prove feasible-set dominance before asserting a Perfect-Information gap."""

    if not isinstance(evidence, PerfectInformationCompatibilityEvidence):
        raise TypeError("evidence must be PerfectInformationCompatibilityEvidence")
    tolerance = evidence.tolerance
    if evidence.causal_period_digest != evidence.bound_period_digest:
        return _not_comparable(
            evidence, PerfectInformationComparisonReason.PERIOD_MISMATCH
        )
    if evidence.causal_return_matrix_digest != evidence.bound_return_matrix_digest:
        return _not_comparable(
            evidence, PerfectInformationComparisonReason.RETURN_MATRIX_MISMATCH
        )
    if not np.allclose(
        evidence.causal_initial_weights,
        evidence.bound_initial_weights,
        rtol=0.0,
        atol=tolerance,
    ):
        return _not_comparable(
            evidence, PerfectInformationComparisonReason.INITIAL_WEIGHTS_MISMATCH
        )
    if not math.isclose(
        evidence.causal_aum, evidence.bound_aum, rel_tol=0.0, abs_tol=tolerance
    ):
        return _not_comparable(
            evidence, PerfectInformationComparisonReason.AUM_MISMATCH
        )
    exposure_relaxed = (
        np.all(
            evidence.bound_max_abs_weight >= evidence.causal_max_abs_weight - tolerance
        )
        and evidence.bound_max_gross >= evidence.causal_max_gross - tolerance
        and (
            evidence.bound_max_net_exposure is None
            or (
                evidence.causal_max_net_exposure is not None
                and evidence.bound_max_net_exposure
                >= evidence.causal_max_net_exposure - tolerance
            )
        )
    )
    if not exposure_relaxed:
        return _not_comparable(
            evidence, PerfectInformationComparisonReason.EXPOSURE_NOT_RELAXED
        )
    costs_relaxed = np.all(
        evidence.bound_transaction_cost_rate
        <= evidence.causal_transaction_cost_rate + tolerance
    ) and np.all(
        evidence.bound_liquidation_cost_rate
        <= evidence.causal_liquidation_cost_rate + tolerance
    )
    if not costs_relaxed:
        return _not_comparable(
            evidence, PerfectInformationComparisonReason.COST_NOT_RELAXED
        )
    if evidence.bound_log_return < evidence.causal_log_return - tolerance:
        return _not_comparable(
            evidence, PerfectInformationComparisonReason.BOUND_ORDER_VIOLATION
        )
    return PerfectInformationComparison.comparable(
        bound_log_return=evidence.bound_log_return,
        causal_log_return=evidence.causal_log_return,
        compatibility_evidence_digest=evidence.digest,
    )
