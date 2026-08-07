from __future__ import annotations

import pytest

pytest.importorskip("nautilus_trader")

from trade_rl.integrations.nautilus.capability import run_nautilus_capability_probe


def test_exact_nautilus_wheel_supports_required_backtest_primitives() -> None:
    report = run_nautilus_capability_probe()

    assert report.runtime.package_version == "1.230.0"
    assert report.engine_constructed is True
    assert report.binance_margin_venue_added is True
    assert report.engine_disposed is True
    assert report.errors == ()
