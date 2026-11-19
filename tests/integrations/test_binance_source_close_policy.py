"""Synthetic contract oracles for opt-in Binance source close-time admission.

No Binance HTTP, archived market input, frozen Dataset or economic study is read.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from trade_rl.integrations.binance.dataset import (
    BinanceMarketDataSource,
    BinanceSourceClosePolicy,
    build_binance_market_dataset,
)
from trade_rl.integrations.binance.derived_klines import (
    BaseIntervalAggregatingTransport,
    aggregate_klines,
)

HOUR = 3_600_000
START = datetime(2026, 1, 1, tzinfo=UTC)
START_MS = int(START.timestamp() * 1000)


def _rows(
    *, count: int = 24, damaged_at: int | None = None, delta: int = 0
) -> list[list[object]]:
    result = []
    for i in range(count):
        t = START_MS + HOUR * i
        close_ms = t + HOUR - 1 + (delta if i == damaged_at else 0)
        result.append(
            [t, "100", "101", "99", "100", "10", close_ms, "1000", 1, "0", "0", "0"]
        )
    return result


class SyntheticTransport:
    def __init__(self, *, damage_at: int | None = None, delta: int = 0) -> None:
        self.damage_at = damage_at
        self.delta = delta

    def load_klines(self, **kw: Any) -> tuple[list[list[object]], str]:
        start_ms = kw["start_ms"]
        end_ms = kw["end_ms"]
        rows = _rows(
            count=(end_ms - start_ms) // HOUR,
            damaged_at=self.damage_at,
            delta=self.delta,
        )
        return rows, "synthetic"

    def load_funding_rates(self, **_kw: Any) -> tuple[list[tuple[int, float]], str]:
        return [], "synthetic"

    def load_exchange_information(self, **_kw: Any) -> None:
        raise AssertionError("metadata must be supplied explicitly")


@pytest.mark.parametrize("delta", (-1, 1, -HOUR))
@pytest.mark.parametrize("policy", ("strict", "non_tradable_rows"))
def test_opt_in_source_close_v2_rejects_malformed_row_under_both_gap_policies(
    delta: int, policy: str
) -> None:
    transport = SyntheticTransport(damage_at=6, delta=delta)
    source = BinanceMarketDataSource(
        market="usds-m",
        interval="1h",
        start_time=START,
        end_time=START + timedelta(hours=24),
        transport=transport,
        gap_policy=policy,
        source_close_policy=BinanceSourceClosePolicy.STRICT_SOURCE_CLOSE_V2,
    )
    with pytest.raises(ValueError, match="source close_time"):
        source.load("BTCUSDT")


def test_default_legacy_policy_retains_exact_prior_row_behavior() -> None:
    baseline = BinanceMarketDataSource(
        market="usds-m",
        interval="1h",
        start_time=START,
        end_time=START + timedelta(hours=24),
        transport=SyntheticTransport(),
    ).load("BTCUSDT")
    historically_admitted = BinanceMarketDataSource(
        market="usds-m",
        interval="1h",
        start_time=START,
        end_time=START + timedelta(hours=24),
        transport=SyntheticTransport(damage_at=6, delta=-HOUR),
    ).load("BTCUSDT")
    np.testing.assert_array_equal(baseline.timestamps, historically_admitted.timestamps)
    np.testing.assert_array_equal(baseline.close, historically_admitted.close)


@pytest.mark.parametrize("delta", (-1, 1, -HOUR))
def test_opt_in_aggregator_rejects_invalid_component_close_time(delta: int) -> None:
    bad = _rows(count=4, damaged_at=3, delta=delta)
    old, dropped = aggregate_klines(
        bad,
        base_interval="1h",
        interval="4h",
        start_ms=START_MS,
        end_ms=START_MS + 4 * HOUR,
    )
    assert len(old) == 1 and dropped == 0
    with pytest.raises(ValueError, match="source close_time"):
        aggregate_klines(
            bad,
            base_interval="1h",
            interval="4h",
            start_ms=START_MS,
            end_ms=START_MS + 4 * HOUR,
            source_close_policy=BinanceSourceClosePolicy.STRICT_SOURCE_CLOSE_V2,
        )


def test_v2_source_refuses_aggregator_that_did_not_validate_base_clock() -> None:
    source = BinanceMarketDataSource(
        market="usds-m",
        interval="1h",
        start_time=START,
        end_time=START + timedelta(hours=24),
        transport=BaseIntervalAggregatingTransport(SyntheticTransport()),
        source_close_policy=BinanceSourceClosePolicy.STRICT_SOURCE_CLOSE_V2,
    )
    with pytest.raises(ValueError, match="aggregate.*source close"):
        source.load_timeframe("BTCUSDT", "4h")


def _build(
    policy: str | BinanceSourceClosePolicy | None = None,
    *,
    metadata: dict[str, object] | None = None,
):
    kw: dict[str, object] = {}
    if policy is not None:
        kw["source_close_policy"] = policy
    return build_binance_market_dataset(
        market="usds-m",
        symbols=("BTCUSDT",),
        interval="1h",
        start_time=START,
        end_time=START + timedelta(days=20),
        transport_mode="vision",
        transport=SyntheticTransport(),
        tick_sizes=(0.01,),
        lot_sizes=(0.001,),
        minimum_notionals=(5.0,),
        listed_ats=(START - timedelta(days=100),),
        metadata_evidence=metadata,
        **kw,
    ).dataset


def test_versioned_source_close_policy_changes_new_dataset_identity_not_legacy() -> (
    None
):
    previous = _build(metadata={"provenance": "synthetic"})
    explicit = _build(
        BinanceSourceClosePolicy.RECONSTRUCTED_V1, metadata={"provenance": "synthetic"}
    )
    strict = _build(
        BinanceSourceClosePolicy.STRICT_SOURCE_CLOSE_V2,
        metadata={"provenance": "synthetic"},
    )
    assert previous.dataset_id == explicit.dataset_id
    assert previous.identity_payload_json == explicit.identity_payload_json
    assert strict.dataset_id != previous.dataset_id
    assert strict.recomputed_dataset_id() == strict.dataset_id
    metadata = json.loads(strict.identity_payload_json)["metadata_evidence"]
    assert metadata["binance_source_close_policy"] == "strict_source_close_v2"
    assert metadata["provenance"] == "synthetic"


def test_conflicting_version_in_caller_metadata_is_refused() -> None:
    with pytest.raises(ValueError, match="source close policy"):
        _build(
            BinanceSourceClosePolicy.STRICT_SOURCE_CLOSE_V2,
            metadata={"binance_source_close_policy": "reconstructed_v1"},
        )


def test_v2_aggregating_transport_validates_each_base_row_before_reporting_4h() -> None:
    transport = BaseIntervalAggregatingTransport(
        SyntheticTransport(damage_at=3, delta=-1),
        source_close_policy=BinanceSourceClosePolicy.STRICT_SOURCE_CLOSE_V2,
    )
    source = BinanceMarketDataSource(
        market="usds-m",
        interval="1h",
        start_time=START,
        end_time=START + timedelta(hours=24),
        transport=transport,
        source_close_policy=BinanceSourceClosePolicy.STRICT_SOURCE_CLOSE_V2,
    )
    with pytest.raises(ValueError, match="source close_time"):
        source.load_timeframe("BTCUSDT", "4h")
    assert transport.dropped_windows == {}


def test_v2_aggregating_transport_accepts_24h_of_valid_base_rows() -> None:
    transport = BaseIntervalAggregatingTransport(
        SyntheticTransport(),
        source_close_policy=BinanceSourceClosePolicy.STRICT_SOURCE_CLOSE_V2,
    )
    source = BinanceMarketDataSource(
        market="usds-m",
        interval="1h",
        start_time=START,
        end_time=START + timedelta(hours=24),
        transport=transport,
        source_close_policy=BinanceSourceClosePolicy.STRICT_SOURCE_CLOSE_V2,
    )
    series = source.load_timeframe("BTCUSDT", "4h")
    assert len(series.timestamps) == 6
    assert all(series.tradable)
    assert transport.dropped_windows[("BTCUSDT", "4h")] == 0


def test_unknown_source_close_policy_is_rejected_before_transport_reads() -> None:
    with pytest.raises(ValueError, match="not a valid"):
        BinanceMarketDataSource(
            market="usds-m",
            interval="1h",
            start_time=START,
            end_time=START + timedelta(hours=24),
            transport=SyntheticTransport(),
            source_close_policy="unknown",
        )
