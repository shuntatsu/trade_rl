from __future__ import annotations

import json

import numpy as np

from tests.integrations.test_binance import FakeTransport
from trade_rl.data.build import ExecutionEconomicsProfile
from trade_rl.integrations.binance import build_binance_market_dataset


def _profile() -> ExecutionEconomicsProfile:
    return ExecutionEconomicsProfile(
        name="canonical_m2_research_assumption_v1",
        fee_rate=0.0005,
        spread_rate=0.0002,
        max_participation_rate=0.05,
        borrow_available=True,
        borrow_rate=0.0,
    )


def test_binance_build_passes_execution_economics_to_canonical_builder() -> None:
    transport = FakeTransport()
    common = {
        "market": "usds-m",
        "symbols": ("BTCUSDT",),
        "interval": "1h",
        "start_time": transport.start,
        "end_time": transport.start.replace(hour=3),
        "transport": transport,
    }

    legacy = build_binance_market_dataset(**common)
    priced = build_binance_market_dataset(
        **common,
        execution_economics=_profile(),
    )

    assert priced.dataset.dataset_id != legacy.dataset.dataset_id
    np.testing.assert_array_equal(
        priced.dataset.resolved_array("fee_rate"),
        np.full((3, 1), 0.0005),
    )
    np.testing.assert_array_equal(
        priced.dataset.resolved_array("spread_rate"),
        np.full((3, 1), 0.0002),
    )
    np.testing.assert_array_equal(
        priced.dataset.resolved_array("max_participation_rate"),
        np.full((3, 1), 0.05),
    )
    identity = json.loads(priced.dataset.identity_payload_json or "{}")
    assert identity["execution_economics"] == _profile().to_payload()
