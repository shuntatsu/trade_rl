"""Synthetic source-admission oracles for exact EWC cosine term counts (#1146).

This file uses no market data, calibration result or sealed/final evidence.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from trade_rl.evaluation.signal_diagnosis.protocol import (
    SignalDiagnosisProtocolError,
    default_signal_diagnosis_protocol,
)
from trade_rl.evaluation.signal_diagnosis.verdict import (
    DecisionCellEvidence,
    PValueProvenance,
    _check_provenance,
)
from trade_rl.evaluation.signal_measurement.significance import ewc_terms


@pytest.mark.parametrize("n_days", (0, 1, 3, 4, 60, 1460, 10**40, 10**500))
def test_ewc_term_roots_use_exact_integer_contract(n_days: int) -> None:
    """The two public paths must avoid float overflow and linear correction."""

    protocol = default_signal_diagnosis_protocol()
    measurement_terms = ewc_terms(n_days)
    protocol_terms = protocol.significance.cosine_terms(n_days)

    assert measurement_terms == protocol_terms
    assert 125 * measurement_terms**3 <= 8 * n_days**2
    assert 8 * n_days**2 < 125 * (measurement_terms + 1) ** 3
    if n_days == 1460:
        assert measurement_terms == 51


def _fake_cell(raw_days: int, demeaned_days: int) -> DecisionCellEvidence:
    protocol = default_signal_diagnosis_protocol()
    return cast(
        DecisionCellEvidence,
        SimpleNamespace(
            cell_id="synthetic",
            rank_ic_p_value=0.5,
            demeaned_p_value=0.5,
            p_value_provenance=PValueProvenance(
                procedure_id=protocol.significance.procedure_id,
                raw_n_days=raw_days,
                demeaned_n_days=demeaned_days,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("raw_days", "demeaned_days"),
    ((10**40, 1400), (1400, 10**500)),
)
def test_provenance_rejects_impossible_development_day_counts(
    raw_days: int,
    demeaned_days: int,
) -> None:
    """A claimant must not make a short frozen window contain aeons of days."""

    with pytest.raises(
        SignalDiagnosisProtocolError, match="eligible-day count exceeds"
    ):
        _check_provenance(
            (_fake_cell(raw_days, demeaned_days),),
            default_signal_diagnosis_protocol(),
        )


@pytest.mark.parametrize(
    ("raw_days", "demeaned_days"),
    ((2039, 2038), (2038, 2039)),
)
def test_half_open_development_window_excludes_its_terminal_day(
    raw_days: int,
    demeaned_days: int,
) -> None:
    """2026-08-01 is not another eligible day of [2021-01-01, 2026-08-01)."""

    protocol = default_signal_diagnosis_protocol()
    assert protocol.data.evaluation_start == "2021-01-01T00:00:00Z"
    assert protocol.data.evaluation_end == "2026-08-01T00:00:00Z"
    with pytest.raises(
        SignalDiagnosisProtocolError, match="eligible-day count exceeds"
    ):
        _check_provenance((_fake_cell(raw_days, demeaned_days),), protocol)


def test_full_2038_development_days_remain_admissible() -> None:
    protocol = default_signal_diagnosis_protocol()
    _check_provenance((_fake_cell(2038, 2038),), protocol)


def test_normal_provenance_day_counts_still_validate() -> None:
    _check_provenance(
        (_fake_cell(1400, 1400),),
        default_signal_diagnosis_protocol(),
    )
