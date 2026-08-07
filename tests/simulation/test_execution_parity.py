from __future__ import annotations

from trade_rl.simulation.execution_parity import (
    CanonicalExecutionRecord,
    compare_execution_traces,
    execution_trace_digest,
)


def _record(*, sequence: int = 1, position_lots: int = 10) -> CanonicalExecutionRecord:
    return CanonicalExecutionRecord(
        sequence=sequence,
        event_type="fill",
        timestamp_ns=1_000 + sequence,
        price_ticks=1_000,
        quantity_lots=10,
        fee_minor=4,
        funding_minor=0,
        position_lots=position_lots,
        equity_minor=99_996,
        terminal_reason=None,
    )


def test_equal_canonical_traces_have_exact_parity_and_same_digest() -> None:
    legacy = (_record(), _record(sequence=2, position_lots=0))
    nautilus = (_record(), _record(sequence=2, position_lots=0))

    report = compare_execution_traces(legacy=legacy, candidate=nautilus)

    assert report.matches is True
    assert report.mismatches == ()
    assert execution_trace_digest(legacy) == execution_trace_digest(nautilus)


def test_value_mismatch_reports_exact_field_and_sequence() -> None:
    legacy = (_record(),)
    nautilus = (
        CanonicalExecutionRecord(
            sequence=1,
            event_type="fill",
            timestamp_ns=1_001,
            price_ticks=1_001,
            quantity_lots=10,
            fee_minor=4,
            funding_minor=0,
            position_lots=10,
            equity_minor=99_996,
            terminal_reason=None,
        ),
    )

    report = compare_execution_traces(legacy=legacy, candidate=nautilus)

    assert report.matches is False
    assert len(report.mismatches) == 1
    mismatch = report.mismatches[0]
    assert mismatch.sequence == 1
    assert mismatch.field == "price_ticks"
    assert mismatch.legacy_value == 1_000
    assert mismatch.candidate_value == 1_001


def test_length_mismatch_is_explicit() -> None:
    report = compare_execution_traces(
        legacy=(_record(),),
        candidate=(_record(), _record(sequence=2)),
    )

    assert report.matches is False
    assert report.mismatches[-1].field == "trace_length"
    assert report.mismatches[-1].legacy_value == 1
    assert report.mismatches[-1].candidate_value == 2
