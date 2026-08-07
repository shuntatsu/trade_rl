from __future__ import annotations

from decimal import Decimal
from typing import Protocol

import pytest

pytest.importorskip("nautilus_trader")

from trade_rl.integrations.nautilus.execution_probe import (
    run_flat_long_flat_execution_probe,
)
from trade_rl.simulation.execution_canonicalization import (
    CanonicalEconomicClosure,
    CanonicalFillSignature,
    compare_dual_shadow_execution,
)
from trade_rl.simulation.legacy_execution_probe import run_legacy_flat_long_flat_probe


class _CanonicalProbe(Protocol):
    @property
    def fills(self) -> tuple[CanonicalFillSignature, ...]: ...

    @property
    def economics(self) -> CanonicalEconomicClosure: ...


def _assert_exact_parity(
    *,
    legacy: _CanonicalProbe,
    candidate: _CanonicalProbe,
) -> None:
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
    economics = (
        f"fee={legacy.economics.fee_minor}/{candidate.economics.fee_minor} "
        f"pnl={legacy.economics.realized_pnl_minor}/"
        f"{candidate.economics.realized_pnl_minor} "
        f"equity={legacy.economics.final_equity_minor}/"
        f"{candidate.economics.final_equity_minor}"
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
