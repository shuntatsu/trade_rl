"""SD0 (#1146): the section-5 decision rule is a pure function with fixed thresholds.

Every input here is a hand-written synthetic number; no market data is read.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import pytest

from trade_rl.evaluation.signal_diagnosis.protocol import (
    SignalDiagnosisProtocolError,
    default_signal_diagnosis_protocol,
)
from trade_rl.evaluation.signal_diagnosis.verdict import (
    DecisionCellEvidence,
    ValidityFlags,
    VerdictResult,
    classify_signal_diagnosis,
    decision_cell_ids,
)

PROTOCOL = default_signal_diagnosis_protocol()
CELL_IDS = decision_cell_ids(PROTOCOL)
CLEAN = ValidityFlags(
    leakage_detected=False,
    dataset_identity_mismatch=False,
    protocol_digest_mismatch=False,
    available_at_violation=False,
    fs1_or_warmup_contact=False,
)


def _no_signal_cell(cell_id: str) -> DecisionCellEvidence:
    family_id, model_id = cell_id.split(":")
    return DecisionCellEvidence(
        family_id=family_id,
        model_id=model_id,
        rank_ic_vs_drift=0.001,
        rank_ic_bh_adjusted_p=0.60,
        beta_removed_rank_ic=0.001,
        beta_removed_bh_adjusted_p=0.70,
        years_same_sign=2,
        alpha_ci_lower_cost_x1=None,
        alpha_ci_upper_cost_x1=None,
        alpha_point_cost_x2=None,
    )


def _signal_cell(
    cell_id: str,
    *,
    lower: float | None = 0.002,
    upper: float | None = 0.010,
    point_x2: float | None = 0.001,
    **overrides: Any,
) -> DecisionCellEvidence:
    family_id, model_id = cell_id.split(":")
    base: dict[str, Any] = {
        "family_id": family_id,
        "model_id": model_id,
        "rank_ic_vs_drift": 0.02,
        "rank_ic_bh_adjusted_p": 0.04,
        "beta_removed_rank_ic": 0.015,
        "beta_removed_bh_adjusted_p": 0.05,
        "years_same_sign": 5,
        "alpha_ci_lower_cost_x1": lower,
        "alpha_ci_upper_cost_x1": upper,
        "alpha_point_cost_x2": point_x2,
    }
    base.update(overrides)
    return DecisionCellEvidence(**base)


def _cells(
    special: dict[str, DecisionCellEvidence],
) -> tuple[DecisionCellEvidence, ...]:
    return tuple(special.get(cid) or _no_signal_cell(cid) for cid in CELL_IDS)


def _classify(
    special: dict[str, DecisionCellEvidence], validity: ValidityFlags = CLEAN
) -> VerdictResult:
    return classify_signal_diagnosis(PROTOCOL, validity=validity, cells=_cells(special))


FIRST, SECOND, THIRD, FOURTH, FIFTH = CELL_IDS[:5]


def test_the_decision_cells_are_every_family_by_model_in_a_fixed_order() -> None:
    assert len(CELL_IDS) == 16
    assert CELL_IDS[0] == "return_momentum:ridge_inner_cv"
    assert CELL_IDS[1] == "return_momentum:lightgbm_fixed"
    assert CELL_IDS[-1] == "cross_asset:lightgbm_fixed"
    assert len(set(CELL_IDS)) == 16


# ----------------------------------------------------------------------- INVALID


@pytest.mark.parametrize(
    "flag",
    [
        "leakage_detected",
        "dataset_identity_mismatch",
        "protocol_digest_mismatch",
        "available_at_violation",
        "fs1_or_warmup_contact",
    ],
)
def test_any_validity_violation_is_invalid_even_with_a_candidate_inside(
    flag: str,
) -> None:
    validity = dataclasses.replace(CLEAN, **{flag: True})

    result = _classify({FIRST: _signal_cell(FIRST)}, validity)

    assert result.verdict == "INVALID"
    assert result.signal_cell_ids == () and result.candidate_cell_ids == ()
    assert any(flag in reason for reason in result.reasons)


def test_validity_flags_must_be_real_booleans() -> None:
    with pytest.raises(SignalDiagnosisProtocolError):
        ValidityFlags(
            leakage_detected=0,  # type: ignore[arg-type]
            dataset_identity_mismatch=False,
            protocol_digest_mismatch=False,
            available_at_violation=False,
            fs1_or_warmup_contact=False,
        )


# ---------------------------------------------------------------------- NO_SIGNAL


def test_all_cells_without_signal_is_no_signal() -> None:
    result = _classify({})

    assert result.verdict == "NO_SIGNAL"
    assert result.signal_cell_ids == () and result.candidate_cell_ids == ()
    assert result.claim_ceiling == "known_used_exploratory_screening_only"


def test_fdr_boundary_equal_to_q_is_significant_just_above_is_not() -> None:
    q = PROTOCOL.multiple_testing.bh_fdr_q
    assert q == 0.10

    at_q = _classify(
        {FIRST: _signal_cell(FIRST, rank_ic_bh_adjusted_p=q, lower=-0.001)}
    )
    above = _classify(
        {
            FIRST: _signal_cell(
                FIRST, rank_ic_bh_adjusted_p=math.nextafter(q, 1.0), lower=-0.001
            )
        }
    )

    assert at_q.verdict == "SIGNAL_NOT_TRADABLE"
    assert above.verdict == "NO_SIGNAL"


def test_a_negative_or_zero_rank_ic_increment_is_not_a_signal() -> None:
    for value in (0.0, -0.02):
        assert (
            _classify({FIRST: _signal_cell(FIRST, rank_ic_vs_drift=value)}).verdict
            == "NO_SIGNAL"
        )


def test_signal_that_vanishes_after_beta_removal_is_no_signal() -> None:
    above = math.nextafter(PROTOCOL.multiple_testing.bh_fdr_q, 1.0)
    vanished = _classify({FIRST: _signal_cell(FIRST, beta_removed_bh_adjusted_p=above)})
    wrong_sign = _classify({FIRST: _signal_cell(FIRST, beta_removed_rank_ic=-0.001)})
    zero = _classify({FIRST: _signal_cell(FIRST, beta_removed_rank_ic=0.0)})
    kept = _classify(
        {
            FIRST: _signal_cell(
                FIRST,
                beta_removed_bh_adjusted_p=PROTOCOL.multiple_testing.bh_fdr_q,
            )
        }
    )

    assert vanished.verdict == "NO_SIGNAL"
    assert wrong_sign.verdict == "NO_SIGNAL"
    assert zero.verdict == "NO_SIGNAL"
    assert kept.verdict == "SIGNAL_CANDIDATE"


def test_year_stability_boundary_is_four_of_five_full_years() -> None:
    three = _classify({FIRST: _signal_cell(FIRST, years_same_sign=3)})
    four = _classify({FIRST: _signal_cell(FIRST, years_same_sign=4)})
    five = _classify({FIRST: _signal_cell(FIRST, years_same_sign=5)})

    # The plan leaves "significant and beta-robust but year-unstable" unassigned;
    # the protocol resolves it to NO_SIGNAL (H0b: concentration in a few years).
    assert three.verdict == "NO_SIGNAL"
    assert three.signal_cell_ids == ()
    assert four.verdict == "SIGNAL_CANDIDATE"
    assert five.verdict == "SIGNAL_CANDIDATE"
    assert PROTOCOL.decision_rules.min_full_years_same_sign == 4


# ------------------------------------------------------------ SIGNAL_NOT_TRADABLE


def test_ci_containing_zero_at_cost_x1_is_not_tradable_even_with_positive_points() -> (
    None
):
    result = _classify(
        {FIRST: _signal_cell(FIRST, lower=-0.0005, upper=0.012, point_x2=0.004)}
    )

    assert result.verdict == "SIGNAL_NOT_TRADABLE"
    assert result.signal_cell_ids == (FIRST,)
    assert result.candidate_cell_ids == ()


def test_ci_lower_bound_exactly_zero_is_not_a_candidate_just_above_is() -> None:
    at_zero = _classify({FIRST: _signal_cell(FIRST, lower=0.0)})
    above = _classify({FIRST: _signal_cell(FIRST, lower=math.nextafter(0.0, 1.0))})

    assert at_zero.verdict == "SIGNAL_NOT_TRADABLE"
    assert above.verdict == "SIGNAL_CANDIDATE"


def test_cost_x2_point_estimate_must_be_strictly_positive() -> None:
    at_zero = _classify({FIRST: _signal_cell(FIRST, point_x2=0.0)})
    negative = _classify({FIRST: _signal_cell(FIRST, point_x2=-0.001)})
    positive = _classify(
        {FIRST: _signal_cell(FIRST, point_x2=math.nextafter(0.0, 1.0))}
    )

    # Excluding zero at cost x1 but failing doubled cost is not a candidate; the
    # protocol resolves this unassigned corner to SIGNAL_NOT_TRADABLE.
    assert at_zero.verdict == "SIGNAL_NOT_TRADABLE"
    assert negative.verdict == "SIGNAL_NOT_TRADABLE"
    assert positive.verdict == "SIGNAL_CANDIDATE"


def test_an_all_negative_interval_is_not_a_candidate() -> None:
    result = _classify({FIRST: _signal_cell(FIRST, lower=-0.01, upper=-0.002)})

    assert result.verdict == "SIGNAL_NOT_TRADABLE"


# --------------------------------------------------------------- SIGNAL_CANDIDATE


def test_a_signal_cell_with_a_clean_interval_and_positive_stress_is_a_candidate() -> (
    None
):
    result = _classify({SECOND: _signal_cell(SECOND)})

    assert result.verdict == "SIGNAL_CANDIDATE"
    assert result.signal_cell_ids == (SECOND,)
    assert result.candidate_cell_ids == (SECOND,)


def test_candidate_wins_over_not_tradable_and_no_signal_cells() -> None:
    result = _classify(
        {
            FIRST: _signal_cell(FIRST, lower=-0.001),  # signal, not tradable
            SECOND: _signal_cell(SECOND),  # candidate
            # all other cells: no signal
        }
    )

    assert result.verdict == "SIGNAL_CANDIDATE"
    assert result.signal_cell_ids == (FIRST, SECOND)
    assert result.candidate_cell_ids == (SECOND,)


def test_candidates_are_capped_at_three_ranked_by_ci_lower_then_cell_id() -> None:
    special = {
        FIRST: _signal_cell(FIRST, lower=0.002),
        SECOND: _signal_cell(SECOND, lower=0.005),
        THIRD: _signal_cell(THIRD, lower=0.005),
        FOURTH: _signal_cell(FOURTH, lower=0.001),
        FIFTH: _signal_cell(FIFTH, lower=0.004),
    }
    result = _classify(special)

    assert PROTOCOL.budget.max_candidates == 3
    assert result.verdict == "SIGNAL_CANDIDATE"
    assert len(result.signal_cell_ids) == 5
    # 0.005 (tie: cell id ascending), 0.005, then 0.004; 0.002 and 0.001 are cut.
    assert result.candidate_cell_ids == tuple(sorted((SECOND, THIRD))) + (FIFTH,)
    assert any("candidate cap" in reason for reason in result.reasons)


def test_exactly_three_qualifying_cells_are_all_kept_without_a_cap_note() -> None:
    result = _classify(
        {
            FIRST: _signal_cell(FIRST),
            SECOND: _signal_cell(SECOND),
            THIRD: _signal_cell(THIRD),
        }
    )

    assert len(result.candidate_cell_ids) == 3
    assert not any("candidate cap" in reason for reason in result.reasons)


# ----------------------------------------------------------- fail-closed inputs


def test_a_signal_cell_without_d4_evidence_cannot_be_classified() -> None:
    for kwargs in (
        {"lower": None},
        {"upper": None},
        {"point_x2": None},
    ):
        with pytest.raises(SignalDiagnosisProtocolError, match="D4"):
            _classify({FIRST: _signal_cell(FIRST, **kwargs)})


def test_d4_evidence_is_optional_only_for_cells_that_stopped_at_an_upper_layer() -> (
    None
):
    result = _classify({})
    assert result.verdict == "NO_SIGNAL"
    # Supplying D4 for a non-signal cell is allowed (all cells are saved).
    extra = _signal_cell(FIRST, rank_ic_bh_adjusted_p=0.9)
    assert _classify({FIRST: extra}).verdict == "NO_SIGNAL"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_are_rejected(bad: float) -> None:
    for field in (
        "rank_ic_vs_drift",
        "rank_ic_bh_adjusted_p",
        "beta_removed_rank_ic",
        "beta_removed_bh_adjusted_p",
    ):
        with pytest.raises(SignalDiagnosisProtocolError):
            _signal_cell(FIRST, **{field: bad})
    for field in ("lower", "upper", "point_x2"):
        with pytest.raises(SignalDiagnosisProtocolError):
            _signal_cell(FIRST, **{field: bad})


def test_probabilities_years_and_interval_order_are_validated() -> None:
    for kwargs in (
        {"rank_ic_bh_adjusted_p": -0.01},
        {"rank_ic_bh_adjusted_p": 1.01},
        {"beta_removed_bh_adjusted_p": 1.5},
        {"years_same_sign": -1},
        {"years_same_sign": 6},
        {"years_same_sign": 4.0},
        {"years_same_sign": True},
        {"lower": 0.01, "upper": 0.001},
    ):
        with pytest.raises(SignalDiagnosisProtocolError):
            _signal_cell(FIRST, **kwargs)


def test_cell_roster_must_match_the_protocol_exactly() -> None:
    cells = list(_cells({}))

    with pytest.raises(SignalDiagnosisProtocolError, match="missing"):
        classify_signal_diagnosis(PROTOCOL, validity=CLEAN, cells=tuple(cells[:-1]))
    with pytest.raises(SignalDiagnosisProtocolError, match="duplicate"):
        classify_signal_diagnosis(
            PROTOCOL, validity=CLEAN, cells=tuple(cells + [cells[0]])
        )
    unknown = dataclasses.replace(cells[0], family_id="not_a_family")
    with pytest.raises(SignalDiagnosisProtocolError, match="unknown"):
        classify_signal_diagnosis(
            PROTOCOL, validity=CLEAN, cells=tuple([unknown, *cells[1:]])
        )
    with pytest.raises(SignalDiagnosisProtocolError):
        classify_signal_diagnosis(PROTOCOL, validity=CLEAN, cells=())
    with pytest.raises(SignalDiagnosisProtocolError):
        classify_signal_diagnosis(PROTOCOL, validity="clean", cells=tuple(cells))  # type: ignore[arg-type]


def test_cell_order_does_not_change_the_verdict() -> None:
    forward = _cells(
        {FIRST: _signal_cell(FIRST), FIFTH: _signal_cell(FIFTH, lower=-0.1)}
    )
    reverse = tuple(reversed(forward))

    a = classify_signal_diagnosis(PROTOCOL, validity=CLEAN, cells=forward)
    b = classify_signal_diagnosis(PROTOCOL, validity=CLEAN, cells=reverse)

    assert a == b


def test_the_verdict_never_names_an_edge_or_authorization() -> None:
    result = _classify({FIRST: _signal_cell(FIRST)})

    fields = {field.name for field in dataclasses.fields(result)}
    assert fields == {
        "verdict",
        "signal_cell_ids",
        "candidate_cell_ids",
        "reasons",
        "claim_ceiling",
    }
    assert result.verdict in PROTOCOL.verdict_labels
    text = " ".join(result.reasons).lower()
    for word in ("edge", "profitab", "g4", "g5", "live"):
        assert word not in text
