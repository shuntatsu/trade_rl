from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from trade_rl.data.build.config import (
    MarketDatasetBuildRequest,
    load_market_build_request,
)
from trade_rl.data.contracts import MarketBuildConfig


def test_market_build_config_owns_explicit_raw_json_field_roster() -> None:
    assert MarketBuildConfig.JSON_FIELDS == (
        "base_timeframe",
        "features",
        "calendar_kind",
        "session_periods_per_year",
        "cross_asset_reference_symbol",
    )
    dataclass_fields = {field.name for field in fields(MarketBuildConfig)}
    assert dataclass_fields == {*MarketBuildConfig.JSON_FIELDS, "schema_version"}


def test_market_dataset_build_request_composes_root_json_fields() -> None:
    assert MarketDatasetBuildRequest.JSON_FIELDS == (
        "source_root",
        *MarketBuildConfig.JSON_FIELDS,
        "instruments",
        "execution_economics",
    )


def test_market_build_loader_consumes_request_json_field_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MarketDatasetBuildRequest,
        "JSON_FIELDS",
        tuple(
            field
            for field in MarketDatasetBuildRequest.JSON_FIELDS
            if field != "cross_asset_reference_symbol"
        ),
    )
    path = tmp_path / "build.json"
    path.write_text(
        json.dumps(
            {
                "source_root": ".",
                "base_timeframe": "1h",
                "cross_asset_reference_symbol": "BTCUSDT",
                "features": [{"name": "ret", "kind": "log_return"}],
                "instruments": [
                    {
                        "symbol": "BTCUSDT",
                        "listed_at": "2020-01-01T00:00:00Z",
                        "volume_unit": "base_asset",
                        "contract_multiplier": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="market build config contains unknown fields.*cross_asset_reference_symbol",
    ):
        load_market_build_request(path)
