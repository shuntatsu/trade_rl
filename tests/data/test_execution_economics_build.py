from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data.artifacts.publication import (
    load_market_dataset_artifact,
    publish_market_dataset_artifact,
)
from trade_rl.data.build.builder import MarketDatasetBuilder
from trade_rl.data.build.config import load_market_build_request
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    NormalizationMode,
    VolumeUnit,
)
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.evaluation.replay import run_single_symbol_replay
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent

_GOLDEN_DATASET_ID = "4410d7d30853230b467c34717e2488120e9cbc4bb311c2d3a8ea0f486bc539a5"
_GOLDEN_FEATURE_CONFIG_DIGEST = (
    "3bea1eacf43bd1010466c602e64bed52414884d5625842b495730b3d7647e0a9"
)
_GOLDEN_NORMALIZATION_DIGEST = (
    "4f712886ea3495c2ef089bfa1eee8294f757eb9070174d78de907864288163b1"
)


def _economics_payload() -> dict[str, object]:
    return {
        "schema_version": "execution_economics_v1",
        "fee_rate": 0.0,
        "maker_fee_rate": 0.0002,
        "taker_fee_rate": 0.0005,
        "spread_rate": 0.0002,
        "max_participation_rate": 0.05,
        "borrow_available": True,
        "borrow_rate": 0.0,
    }


def _request_payload(*, economics: dict[str, object] | None) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_root": ".",
        "base_timeframe": "1h",
        "features": [{"name": "ret", "kind": "log_return"}],
        "instruments": [
            {
                "symbol": "BTCUSDT",
                "listed_at": "2026-01-01T00:00:00Z",
                "volume_unit": "base_asset",
                "contract_multiplier": 1.0,
            }
        ],
    }
    if economics is not None:
        payload["execution_economics"] = economics
    return payload


def _load_request(
    tmp_path: Path,
    *,
    economics: dict[str, object] | None,
):
    path = tmp_path / "build.json"
    path.write_text(
        json.dumps(_request_payload(economics=economics)),
        encoding="utf-8",
    )
    return load_market_build_request(path)


def _raw_series(n_bars: int = 40) -> RawMarketSeries:
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = np.exp(np.arange(n_bars, dtype=np.float64) * 0.002)
    open_price = np.concatenate([close[:1], close[:-1]])
    return RawMarketSeries(
        timestamps=timestamps,
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=100.0 + np.arange(n_bars, dtype=np.float64),
        funding_rate=np.where(np.arange(n_bars) % 8 == 0, 0.0001, 0.0),
        tradable=np.ones(n_bars, dtype=np.bool_),
    )


def _config() -> MarketBuildConfig:
    return MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(
                name="ret_1",
                kind=FeatureKind.LOG_RETURN,
                lookback=1,
                normalization=NormalizationMode.NONE,
            ),
            FeatureSpec(
                name="vol_12",
                kind=FeatureKind.REALIZED_VOLATILITY,
                lookback=12,
            ),
        ),
    )


def _instrument() -> InstrumentContract:
    return InstrumentContract(
        symbol="BTCUSDT",
        listed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        volume_unit=VolumeUnit.BASE_ASSET,
        tick_size=0.1,
        lot_size=0.001,
        minimum_notional=5.0,
    )


def _build_dataset(*, execution_economics=None):
    return MarketDatasetBuilder(_config()).build(
        InMemoryMarketDataSource({"BTCUSDT": _raw_series()}),
        (_instrument(),),
        execution_economics=execution_economics,
    )


class _AlwaysLong:
    def decide(self, observation: StrategyObservation) -> PositionIntent:
        del observation
        return PositionIntent.LONG


def test_market_build_request_parses_strict_execution_economics(
    tmp_path: Path,
) -> None:
    request = _load_request(tmp_path, economics=_economics_payload())

    assert request.execution_economics is not None
    assert request.execution_economics.canonical_payload() == _economics_payload()


def test_market_build_request_rejects_unknown_execution_economics_field(
    tmp_path: Path,
) -> None:
    economics = _economics_payload()
    economics["unexpected"] = 1

    with pytest.raises(
        ValueError,
        match="execution_economics contains unknown fields",
    ):
        _load_request(tmp_path, economics=economics)


def test_market_build_request_rejects_generic_and_venue_fee_double_count(
    tmp_path: Path,
) -> None:
    economics = _economics_payload()
    economics["fee_rate"] = 0.0001

    with pytest.raises(
        ValueError,
        match="fee_rate cannot be combined with maker_fee_rate or taker_fee_rate",
    ):
        _load_request(tmp_path, economics=economics)


def test_execution_economics_change_dataset_identity_not_feature_digests(
    tmp_path: Path,
) -> None:
    request = _load_request(tmp_path, economics=_economics_payload())
    baseline = _build_dataset()
    with_economics = _build_dataset(
        execution_economics=request.execution_economics,
    )

    assert baseline.dataset_id == _GOLDEN_DATASET_ID
    assert baseline.feature_config_digest == _GOLDEN_FEATURE_CONFIG_DIGEST
    assert baseline.normalization_digest == _GOLDEN_NORMALIZATION_DIGEST
    assert with_economics.dataset_id != baseline.dataset_id
    assert with_economics.feature_config_digest == baseline.feature_config_digest
    assert with_economics.normalization_digest == baseline.normalization_digest
    np.testing.assert_allclose(with_economics.resolved_array("fee_rate"), 0.0)
    np.testing.assert_allclose(with_economics.resolved_array("maker_fee_rate"), 0.0002)
    np.testing.assert_allclose(with_economics.resolved_array("taker_fee_rate"), 0.0005)
    np.testing.assert_allclose(with_economics.resolved_array("spread_rate"), 0.0002)
    np.testing.assert_allclose(
        with_economics.resolved_array("max_participation_rate"),
        0.05,
    )
    np.testing.assert_array_equal(
        with_economics.resolved_array("borrow_available"),
        True,
    )
    np.testing.assert_allclose(with_economics.resolved_array("borrow_rate"), 0.0)

    identity = json.loads(with_economics.identity_payload_json or "{}")
    assert identity["execution_economics"] == _economics_payload()


def test_execution_economics_survive_artifact_roundtrip(tmp_path: Path) -> None:
    request = _load_request(tmp_path, economics=_economics_payload())
    dataset = _build_dataset(execution_economics=request.execution_economics)
    artifact_root = tmp_path / "artifact"

    publish_market_dataset_artifact(artifact_root, dataset)
    reloaded = load_market_dataset_artifact(artifact_root)

    assert reloaded.dataset_id == dataset.dataset_id
    for name in (
        "fee_rate",
        "maker_fee_rate",
        "taker_fee_rate",
        "spread_rate",
        "max_participation_rate",
        "borrow_available",
        "borrow_rate",
    ):
        np.testing.assert_array_equal(
            reloaded.resolved_array(name),
            dataset.resolved_array(name),
        )


def test_zero_overlay_replay_charges_dataset_authoritative_costs(
    tmp_path: Path,
) -> None:
    request = _load_request(tmp_path, economics=_economics_payload())
    dataset = _build_dataset(execution_economics=request.execution_economics)

    result = run_single_symbol_replay(
        dataset,
        _AlwaysLong(),
        start_index=1,
        stop_index=10,
        gross_budget=0.25,
        initial_capital=1_000.0,
    )

    assert result.diagnostics.n_trades > 0
    assert result.diagnostics.turnover_total > 0.0
    assert result.diagnostics.total_cost > 0.0
