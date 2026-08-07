from __future__ import annotations

from decimal import Decimal

import pytest

pytest.importorskip("nautilus_trader")

from trade_rl.integrations.nautilus.execution_probe import (
    run_flat_long_flat_execution_probe,
)
from trade_rl.simulation.execution_canonicalization import compare_dual_shadow_execution
from trade_rl.simulation.legacy_execution_probe import run_legacy_flat_long_flat_probe


@pytest.mark.nautilus
def test_flat_long_flat_conformance_fixture_has_exact_dual_shadow_parity() -> None:
    legacy = run_legacy_flat_long_flat_probe()
    candidate = run_flat_long_flat_execution_probe(
        starting_balance=Decimal("1000"),
    )

    report = compare_dual_shadow_execution(
        legacy_fills=legacy.fills,
        candidate_fills=candidate.fills,
        legacy_economics=legacy.economics,
        candidate_economics=candidate.economics,
    )

    assert report.fill_parity is True, (
        report.mismatches,
        legacy.fills,
        candidate.fills,
    )
    assert report.economic_parity is True, (
        report.mismatches,
        legacy.economics,
        candidate.economics,
    )
    assert report.exact_parity is True, report.mismatches
