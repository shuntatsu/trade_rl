from __future__ import annotations

from dataclasses import replace

import numpy as np

from trade_rl.evaluation.causal_scenario_c3_contracts import (
    PerfectInformationComparisonReason,
    PerfectInformationComparisonStatus,
)
from trade_rl.evaluation.causal_scenario_c3_perfect_information import (
    PerfectInformationCompatibilityEvidence,
    evaluate_perfect_information_compatibility,
)


def _sha(char: str) -> str:
    return char * 64


def _evidence() -> PerfectInformationCompatibilityEvidence:
    return PerfectInformationCompatibilityEvidence(
        causal_period_digest=_sha("a"),
        bound_period_digest=_sha("a"),
        causal_return_matrix_digest=_sha("b"),
        bound_return_matrix_digest=_sha("b"),
        causal_initial_weights=np.asarray([0.1, -0.1]),
        bound_initial_weights=np.asarray([0.1, -0.1]),
        causal_aum=100_000.0,
        bound_aum=100_000.0,
        causal_max_abs_weight=np.asarray([0.45, 0.45]),
        bound_max_abs_weight=np.asarray([0.50, 0.50]),
        causal_max_gross=0.90,
        bound_max_gross=1.00,
        causal_max_net_exposure=0.40,
        bound_max_net_exposure=0.50,
        causal_transaction_cost_rate=np.asarray([0.0010, 0.0010]),
        bound_transaction_cost_rate=np.asarray([0.0005, 0.0005]),
        causal_liquidation_cost_rate=np.asarray([0.0010, 0.0010]),
        bound_liquidation_cost_rate=np.asarray([0.0005, 0.0005]),
        bound_result_digest=_sha("c"),
        bound_log_return=0.08,
        causal_log_return=0.05,
        tolerance=1e-10,
    )


def test_complete_dominance_evidence_is_comparable() -> None:
    evidence = _evidence()
    result = evaluate_perfect_information_compatibility(evidence)
    assert result.status is PerfectInformationComparisonStatus.COMPARABLE
    assert result.reason == PerfectInformationComparisonReason.DOMINANCE_VERIFIED.value
    assert result.gap == 0.03
    assert result.compatibility_evidence_digest == evidence.digest


def test_period_mismatch_is_not_comparable() -> None:
    result = evaluate_perfect_information_compatibility(
        replace(_evidence(), bound_period_digest=_sha("d"))
    )
    assert result.status is PerfectInformationComparisonStatus.NOT_COMPARABLE
    assert result.reason == PerfectInformationComparisonReason.PERIOD_MISMATCH.value
    assert result.gap is None


def test_initial_weights_mismatch_is_not_comparable() -> None:
    result = evaluate_perfect_information_compatibility(
        replace(_evidence(), bound_initial_weights=np.asarray([0.0, 0.0]))
    )
    assert result.reason == PerfectInformationComparisonReason.INITIAL_WEIGHTS_MISMATCH.value


def test_return_matrix_and_aum_must_match() -> None:
    matrix = evaluate_perfect_information_compatibility(
        replace(_evidence(), bound_return_matrix_digest=_sha("e"))
    )
    assert matrix.reason == PerfectInformationComparisonReason.RETURN_MATRIX_MISMATCH.value
    aum = evaluate_perfect_information_compatibility(
        replace(_evidence(), bound_aum=99_999.0)
    )
    assert aum.reason == PerfectInformationComparisonReason.AUM_MISMATCH.value


def test_bound_constraints_must_contain_causal_feasible_set() -> None:
    exposure = evaluate_perfect_information_compatibility(
        replace(_evidence(), bound_max_gross=0.80)
    )
    assert exposure.reason == PerfectInformationComparisonReason.EXPOSURE_NOT_RELAXED.value
    cost = evaluate_perfect_information_compatibility(
        replace(
            _evidence(),
            bound_transaction_cost_rate=np.asarray([0.0020, 0.0020]),
        )
    )
    assert cost.reason == PerfectInformationComparisonReason.COST_NOT_RELAXED.value


def test_asserted_bound_must_dominate_realized_causal_result() -> None:
    result = evaluate_perfect_information_compatibility(
        replace(_evidence(), bound_log_return=0.04)
    )
    assert result.reason == PerfectInformationComparisonReason.BOUND_ORDER_VIOLATION.value
    assert result.gap is None
