from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("nautilus_trader")

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.integrations.nautilus.event_projection import SourceBar
from trade_rl.integrations.nautilus.historical_execution import (
    NautilusHistoricalTargetInterval,
)
from trade_rl.integrations.nautilus.historical_projection import (
    project_historical_interval_source_bars,
)
from trade_rl.integrations.nautilus.historical_subprocess import (
    run_historical_target_intervals_subprocess,
)
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor
from trade_rl.simulation.orders import OrderBookState
from trade_rl.simulation.target_execution import execute_target_statefully
from trade_rl.workflows.stage_a_nautilus_representative_batch import (
    run_and_persist_representative_nautilus_evidence,
)
from trade_rl.workflows.stage_a_nautilus_representative_evidence import (
    load_representative_nautilus_evidence,
)
from trade_rl.workflows.stage_a_nautilus_representative_runner import (
    run_representative_nautilus_window,
)

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "nautilus"
_FIXTURE = _FIXTURE_ROOT / "btcusdt-usdsm-representative-15m.json"
_VOLUME_FIXTURE = _FIXTURE_ROOT / "btcusdt-usdsm-representative-15m-quote-volume.json"
_BAR_SPAN_MS = 15 * 60 * 1000
_REPRESENTATIVE_TIME_QUANTILES = (0.1, 0.5, 0.9)


def _read_payload(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _payload() -> dict[str, Any]:
    return _read_payload(_FIXTURE)


def _source_bars(rows: list[list[object]]) -> tuple[SourceBar, ...]:
    bars: list[SourceBar] = []
    for row in rows:
        assert len(row) == 7
        open_time_ms, open_price, high_price, low_price, close_price, mark, index = row
        assert isinstance(open_time_ms, int)
        bars.append(
            SourceBar(
                open_ns=open_time_ms * 1_000_000,
                close_ns=(open_time_ms + _BAR_SPAN_MS - 1) * 1_000_000,
                open_price=float(open_price),
                high_price=float(high_price),
                low_price=float(low_price),
                close_price=float(close_price),
                mark_price=float(mark),
                index_price=float(index),
            )
        )
    return tuple(bars)


def _round_trip_intervals(
    rows: list[list[object]],
) -> tuple[NautilusHistoricalTargetInterval, ...]:
    bars = _source_bars(rows)
    assert len(bars) == 16
    return (
        NautilusHistoricalTargetInterval(
            sequence=1,
            target_exposure=0.10,
            allocated_equity=100_000.0,
            source_bars=bars[:8],
        ),
        NautilusHistoricalTargetInterval(
            sequence=2,
            target_exposure=0.0,
            allocated_equity=100_000.0,
            source_bars=bars[8:],
        ),
    )


def _real_dataset_for_quantile(time_quantile: float) -> MarketDataset:
    payload = _payload()
    volume_payload = _read_payload(_VOLUME_FIXTURE)
    price_windows = {
        float(window["time_quantile"]): window for window in payload["windows"]
    }
    volume_windows = {
        float(window["time_quantile"]): window for window in volume_payload["windows"]
    }
    price_window = price_windows[time_quantile]
    volume_window = volume_windows[time_quantile]
    rows = price_window["rows"]
    volume_rows = volume_window["rows"]
    assert len(rows) == len(volume_rows) == 16
    assert [row[0] for row in rows] == [row[0] for row in volume_rows]

    open_times_ms = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    close_times_ms = open_times_ms + _BAR_SPAN_MS
    timestamps = close_times_ms.astype("datetime64[ms]").astype("datetime64[ns]")
    shape = (len(rows), 1)

    funding_rate = np.zeros(shape, dtype=np.float64)
    funding_event_count = np.zeros(shape, dtype=np.int32)
    funding_due = np.zeros(shape, dtype=np.bool_)
    funding = price_window["funding"]
    for funding_row in funding:
        funding_time_ms = int(funding_row[0])
        funding_index = int(
            np.searchsorted(close_times_ms, funding_time_ms, side="left")
        )
        assert 0 <= funding_index < len(rows)
        funding_rate[funding_index, 0] = float(funding_row[1])
        funding_event_count[funding_index, 0] += 1
        funding_due[funding_index, 0] = True

    def column(index: int) -> np.ndarray:
        return np.asarray([[float(row[index])] for row in rows], dtype=np.float64)

    dataset_id = content_digest(
        {
            "interval": payload["interval"],
            "price_window": price_window,
            "schema_version": "btc_usdsm_representative_market_dataset_v1",
            "symbol": payload["symbol"],
            "volume_window": volume_window,
        }
    )
    return MarketDataset(
        dataset_id=dataset_id,
        symbols=("BTCUSDT",),
        timestamps=timestamps,
        features=np.zeros((len(rows), 1, 1), dtype=np.float32),
        global_features=np.zeros((len(rows), 1), dtype=np.float32),
        open=column(1),
        high=column(2),
        low=column(3),
        close=column(4),
        volume=np.asarray([[float(row[1])] for row in volume_rows], dtype=np.float64),
        funding_rate=funding_rate,
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((len(rows), 1, 1), dtype=np.bool_),
        feature_names=("probe",),
        global_feature_names=("probe",),
        periods_per_year=35_040,
        funding_event_count=funding_event_count,
        funding_due=funding_due,
        minimum_notional=np.full(shape, 5.0, dtype=np.float64),
        lot_size=np.full(shape, 0.001, dtype=np.float64),
        tick_size=np.full(shape, 0.1, dtype=np.float64),
        mark_price=column(5),
        index_price=column(6),
        volume_units=(VolumeUnit.QUOTE_NOTIONAL,),
        contract_multipliers=np.array([1.0], dtype=np.float64),
    )


def _real_funding_dataset() -> MarketDataset:
    dataset = _real_dataset_for_quantile(0.9)
    assert dataset.funding_rate is not None
    assert dataset.funding_event_count is not None
    assert dataset.funding_due is not None
    assert dataset.funding_rate[4, 0] == pytest.approx(0.00004698)
    assert dataset.funding_event_count[4, 0] == 1
    assert bool(dataset.funding_due[4, 0]) is True
    return dataset


def _representative_source_digest(markets: dict[float, MarketDataset]) -> str:
    return content_digest(
        {
            "schema_version": "stage_a_nautilus_representative_source_v1",
            "windows": [
                {
                    "dataset_id": markets[time_quantile].dataset_id,
                    "time_quantile": time_quantile,
                }
                for time_quantile in _REPRESENTATIVE_TIME_QUANTILES
            ],
        }
    )


@pytest.mark.nautilus
def test_representative_fixture_is_time_selected_real_binance_data() -> None:
    payload = _payload()

    assert payload["schema_version"] == "btc_usdsm_representative_windows_v1"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["interval"] == "15m"
    assert payload["canonical_range"] == [
        "2021-01-01T00:00:00Z",
        "2026-07-01T00:00:00Z",
        192672,
    ]
    assert payload["selection"]["method"] == "time_quantiles_floor_15m"
    assert payload["selection"]["quantiles"] == [0.1, 0.5, 0.9]
    assert payload["selection"]["window_bars"] == 16
    assert payload["bar_columns"] == [
        "open_time_ms",
        "open",
        "high",
        "low",
        "close",
        "mark_close",
        "index_close",
    ]
    assert [window["time_quantile"] for window in payload["windows"]] == [
        0.1,
        0.5,
        0.9,
    ]
    assert all(len(window["rows"]) == 16 for window in payload["windows"])

    for window in payload["windows"]:
        assert all(len(row) == 7 for row in window["rows"])
        open_times = [row[0] for row in window["rows"]]
        assert all(isinstance(value, int) for value in open_times)
        assert all(
            right - left == _BAR_SPAN_MS
            for left, right in zip(open_times, open_times[1:])
        )

    assert payload["windows"][2]["funding"] == [
        ["1765526400007", "0.00004698", "92392.37302174", "Regular"]
    ]


@pytest.mark.nautilus
def test_quote_volume_sidecar_matches_representative_price_windows() -> None:
    payload = _payload()
    volume = _read_payload(_VOLUME_FIXTURE)

    assert volume["schema_version"] == "btc_usdsm_representative_quote_volume_v1"
    assert volume["symbol"] == payload["symbol"]
    assert volume["interval"] == payload["interval"]
    assert volume["volume_unit"] == "quote_notional"
    assert [window["time_quantile"] for window in volume["windows"]] == [
        0.1,
        0.5,
        0.9,
    ]

    for price_window, volume_window in zip(
        payload["windows"], volume["windows"], strict=True
    ):
        assert volume_window["time_quantile"] == price_window["time_quantile"]
        assert len(volume_window["rows"]) == len(price_window["rows"]) == 16
        assert [row[0] for row in volume_window["rows"]] == [
            row[0] for row in price_window["rows"]
        ]
        assert all(float(row[1]) > 0.0 for row in volume_window["rows"])


@pytest.mark.nautilus
def test_real_funding_boundary_matches_legacy_and_nautilus_actual_position() -> None:
    dataset = _real_funding_dataset()
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    book = BookState.zero(
        dataset.n_symbols,
        100_000.0,
        initial_prices=dataset.close[0],
        contract_multipliers=dataset.resolved_array("contract_multipliers"),
    )

    legacy = execute_target_statefully(
        executor,
        book,
        OrderBookState.empty(),
        np.array([0.10], dtype=np.float64),
        start_index=0,
        bars=4,
        target_identity="representative-real-funding",
    )

    assert len(legacy.funding_evidence) == 1
    boundary = legacy.funding_evidence[0]
    assert boundary.processing_index == 4
    assert boundary.timestamp_ns == int(dataset.timestamps[4].astype(np.int64))
    assert boundary.mark_prices == pytest.approx((92_374.51554348,))
    assert boundary.funding_rates == pytest.approx((0.00004698,))
    assert boundary.signed_quantities == pytest.approx((0.108,))
    assert boundary.funding_amount == pytest.approx(-0.4686935119451306)

    candidate = run_historical_target_intervals_subprocess(
        (
            NautilusHistoricalTargetInterval(
                sequence=1,
                target_exposure=0.10,
                allocated_equity=100_000.0,
                source_bars=project_historical_interval_source_bars(
                    dataset,
                    start_index=0,
                    end_index=4,
                ),
            ),
        ),
        snapshot_timestamps_ns=(boundary.timestamp_ns,),
        starting_balance=Decimal("100000"),
        no_trade_band=0.0,
    )

    assert candidate.execution.runtime_version == "1.230.0"
    assert len(candidate.execution.position_snapshots) == 1
    snapshot = candidate.execution.position_snapshots[0]
    assert snapshot.timestamp_ns == boundary.timestamp_ns
    assert float(snapshot.signed_quantity) == pytest.approx(
        boundary.signed_quantities[0]
    )


@pytest.mark.nautilus
def test_real_window_persists_stage_a_structural_and_economic_evidence(
    tmp_path: Path,
) -> None:
    window = run_representative_nautilus_window(
        market=_real_funding_dataset(),
        time_quantile=0.9,
        store_root=tmp_path / "stage-a",
        target_exposure=0.10,
    )

    assert window.structural.candidate_runtime_version == "1.230.0"
    assert window.structural.terminal_position_matches is True
    assert window.structural.terminal_open_orders_passed is True
    assert window.structural.funding_matches is True
    assert window.structural.structural_passed is True
    assert window.economic.replay_digest == window.structural.replay_digest
    assert isinstance(window.economic.normalized_equity_delta_minor, int)


@pytest.mark.nautilus
def test_real_representative_windows_persist_one_bound_aggregate_evidence(
    tmp_path: Path,
) -> None:
    markets = {
        time_quantile: _real_dataset_for_quantile(time_quantile)
        for time_quantile in _REPRESENTATIVE_TIME_QUANTILES
    }
    output_path = tmp_path / "representative-evidence.json"
    source_digest = _representative_source_digest(markets)

    evidence = run_and_persist_representative_nautilus_evidence(
        markets=markets,
        source_digest=source_digest,
        store_root=tmp_path / "stage-a",
        output_path=output_path,
        target_exposure=0.10,
    )

    assert evidence.source_digest == source_digest
    assert evidence.time_quantiles == _REPRESENTATIVE_TIME_QUANTILES
    assert [window.structural.dataset_id for window in evidence.windows] == [
        markets[time_quantile].dataset_id
        for time_quantile in _REPRESENTATIVE_TIME_QUANTILES
    ]
    assert all(
        window.structural.candidate_runtime_version == "1.230.0"
        for window in evidence.windows
    )
    assert load_representative_nautilus_evidence(output_path) == evidence


@pytest.mark.nautilus
def test_representative_real_windows_replay_deterministically_and_finish_flat() -> None:
    payload = _payload()

    for window in payload["windows"]:
        intervals = _round_trip_intervals(window["rows"])
        first = run_historical_target_intervals_subprocess(
            intervals,
            starting_balance=Decimal("100000"),
            no_trade_band=0.0,
        )
        second = run_historical_target_intervals_subprocess(
            intervals,
            starting_balance=Decimal("100000"),
            no_trade_band=0.0,
        )

        assert first.worker_pid != second.worker_pid
        assert first.execution == second.execution
        assert first.execution.runtime_version == "1.230.0"
        assert first.execution.fills
        assert first.execution.fills[0].position_lots > 0
        assert first.execution.fills[-1].position_lots == 0
        assert first.execution.terminal_position_lots == 0
        assert first.execution.terminal_open_orders == 0