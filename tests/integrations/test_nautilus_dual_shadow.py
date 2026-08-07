from __future__ import annotations

from decimal import Decimal

import pytest

pytest.importorskip("nautilus_trader")

from trade_rl.integrations.nautilus.execution_probe import (
    run_flat_long_flat_execution_probe,
    run_flat_long_flat_short_flat_execution_probe,
)
from trade_rl.simulation.execution_canonicalization import compare_dual_shadow_execution
from trade_rl.simulation.legacy_execution_probe import (
    run_legacy_flat_long_flat_probe,
    run_legacy_flat_long_flat_short_flat_probe,
)


def _assert_exact_parity(*, legacy: object, candidate: object) -> None:
    legacy_fills = legacy.fills
    candidate_fills = candidate.fills
    legacy_economics = legacy.economics
    candidate_economics = candidate.economics
    report = compare_dual_shadow_execution(
        legacy_fills=legacy_fills,
        candidate_fills=candidate_fills,
        legacy_economics=legacy_economics,
        candidate_economics=candidate_economics,
    )

    assert report.fill_parity is True, (
        report.mismatches,
        legacy_fills,
        candidate_fills,
    )
    economics = (
        f"fee={legacy_economics.fee_minor}/{candidate_economics.fee_minor} "
        f"pnl={legacy_economics.realized_pnl_minor}/"
        f"{candidate_economics.realized_pnl_minor} "
        f"equity={legacy_economics.final_equity_minor}/"
        f"{candidate_economics.final_equity_minor}"
    )
    assert report.economic_parity is True, economics
    assert report.exact_parity is True, report.mismatches


@pytest.mark.nautilus
def test_flat_long_flat_conformance_fixture_has_exact_dual_shadow_parity() -> None:
    legacy = run_legacy_flat_long_flat_probe()
    candidate = run_flat_long_flat_execution_probe(
        starting_balance=Decimal("1000"),
    )

    _assert_exact_parity(legacy=legacy, candidate=candidate)


@pytest.mark.nautilus
def test_reduce_to_flat_sign_flip_has_exact_dual_shadow_parity() -> None:
    legacy = run_legacy_flat_long_flat_short_flat_probe()
    candidate = run_flat_long_flat_short_flat_execution_probe(
        starting_balance=Decimal("1000"),
    )

    assert [fill.position_lots for fill in candidate.fills] == [1000, 0, -1000, 0]
    _assert_exact_parity(legacy=legacy, candidate=candidate)
