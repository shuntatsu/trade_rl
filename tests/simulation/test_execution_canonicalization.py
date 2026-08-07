from __future__ import annotations

from trade_rl.simulation.execution_canonicalization import (
    CanonicalFillSignature,
    CanonicalEconomicClosure,
    compare_dual_shadow_execution,
)


def _fill(
    *,
    sequence: int,
    timestamp_ns: int,
    price_ticks: int,
    quantity_lots: int,
    position_lots: int,
) -> CanonicalFillSignature:
    return CanonicalFillSignature(
        sequence=sequence,
        timestamp_ns=timestamp_ns,
        price_ticks=price_ticks,
        quantity_lots=quantity_lots,
        position_lots=position_lots,
    )


def _economics(*, fee_minor: int = 100, funding_minor: int = 0) -> CanonicalEconomicClosure:
    return CanonicalEconomicClosure(
        fee_minor=fee_minor,
        funding_minor=funding_minor,
        realized_pnl_minor=4_800_000_000,
        final_equity_minor=100_004_700_000_000,
        terminal_position_lots=0,
        terminal_open_orders=0,
    )


def test_dual_shadow_requires_both_fill_and_economic_equality() -> None:
    fills = (
        _fill(
            sequence=1,
            timestamp_ns=10,
            price_ticks=1001,
            quantity_lots=1000,
            position_lots=1000,
        ),
        _fill(
            sequence=2,
            timestamp_ns=200,
            price_ticks=1049,
            quantity_lots=-1000,
            position_lots=0,
        ),
    )

    report = compare_dual_shadow_execution(
        legacy_fills=fills,
        candidate_fills=fills,
        legacy_economics=_economics(),
        candidate_economics=_economics(),
    )

    assert report.fill_parity is True
    assert report.economic_parity is True
    assert report.exact_parity is True
    assert report.mismatches == ()


def test_economic_mismatch_blocks_exact_parity_even_when_fills_match() -> None:
    fills = (
        _fill(
            sequence=1,
            timestamp_ns=10,
            price_ticks=1001,
            quantity_lots=1000,
            position_lots=1000,
        ),
    )

    report = compare_dual_shadow_execution(
        legacy_fills=fills,
        candidate_fills=fills,
        legacy_economics=_economics(fee_minor=100),
        candidate_economics=_economics(fee_minor=101),
    )

    assert report.fill_parity is True
    assert report.economic_parity is False
    assert report.exact_parity is False
    assert report.mismatches == ("economics.fee_minor",)


def test_fill_mismatch_reports_field_without_hiding_economic_match() -> None:
    legacy = (
        _fill(
            sequence=1,
            timestamp_ns=10,
            price_ticks=1000,
            quantity_lots=1000,
            position_lots=1000,
        ),
    )
    candidate = (
        _fill(
            sequence=1,
            timestamp_ns=10,
            price_ticks=1001,
            quantity_lots=1000,
            position_lots=1000,
        ),
    )

    report = compare_dual_shadow_execution(
        legacy_fills=legacy,
        candidate_fills=candidate,
        legacy_economics=_economics(),
        candidate_economics=_economics(),
    )

    assert report.fill_parity is False
    assert report.economic_parity is True
    assert report.exact_parity is False
    assert report.mismatches == ("fills[0].price_ticks",)
