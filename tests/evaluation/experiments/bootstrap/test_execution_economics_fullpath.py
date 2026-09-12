from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from tests.evaluation.experiments.bootstrap.test_binance import _config
from tests.evaluation.experiments.bootstrap.test_workflow import _install_fakes
from trade_rl.data.build import ExecutionEconomicsProfile, MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.evaluation.experiments.bootstrap.workflow import (
    bootstrap_canonical_m2_study,
    inspect_canonical_m2_bootstrap,
)
from trade_rl.integrations.binance import BinanceDatasetBuildResult


def _profile() -> ExecutionEconomicsProfile:
    return ExecutionEconomicsProfile(
        name="canonical_m2_research_assumption_v1",
        fee_rate=0.0005,
        spread_rate=0.0002,
        max_participation_rate=0.05,
        borrow_available=True,
        borrow_rate=0.0,
    )


def _priced_synthetic_dataset(
    metadata_evidence: object,
    profile: ExecutionEconomicsProfile,
):
    periods = 60 * 24
    timestamps = np.datetime64("2024-01-01T01:00:00", "ns") + np.arange(
        periods
    ) * np.timedelta64(1, "h")
    close = 100.0 + np.arange(periods, dtype=np.float64) * 0.01
    raw_by_symbol = {}
    for offset, symbol in enumerate(("BTCUSDT", "ETHUSDT")):
        shifted = close + offset * 10.0
        open_price = np.concatenate([shifted[:1], shifted[:-1]])
        raw_by_symbol[symbol] = RawMarketSeries(
            timestamps=timestamps,
            open=open_price,
            high=np.maximum(open_price, shifted) + 1.0,
            low=np.minimum(open_price, shifted) - 1.0,
            close=shifted,
            volume=np.full(periods, 1_000_000.0),
            funding_rate=np.zeros(periods),
            tradable=np.ones(periods, dtype=np.bool_),
        )
    contracts = tuple(
        InstrumentContract(symbol=symbol, listed_at=datetime(2024, 1, 1, tzinfo=UTC))
        for symbol in ("BTCUSDT", "ETHUSDT")
    )
    return MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="1h",
            features=(
                FeatureSpec(
                    name="1h__log_return_24bar",
                    kind=FeatureKind.LOG_RETURN,
                    lookback=1,
                ),
            ),
        )
    ).build(
        InMemoryMarketDataSource(raw_by_symbol),
        contracts,
        identity_provenance=metadata_evidence,
        execution_economics=profile,
    )


def test_v2_bootstrap_and_inspection_accept_matching_execution_economics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.experiments.bootstrap import workflow as workflow_module

    _install_fakes(monkeypatch)
    profile = _profile()
    config = replace(
        _config(),
        schema_version="canonical_m2_bootstrap_config_v2",
        execution_economics=profile,
    )
    config_path = tmp_path / "bootstrap.json"
    config_path.write_text(json.dumps(config.to_payload()), encoding="utf-8")

    def build(**kwargs: object) -> BinanceDatasetBuildResult:
        observed = kwargs.get("execution_economics")
        assert observed == profile
        dataset = _priced_synthetic_dataset(kwargs["metadata_evidence"], profile)
        return BinanceDatasetBuildResult(
            dataset=dataset,
            metadata=(),
            sources_used=("frozen:exchange-info", "vision"),
            feature_timeframes=("1h",),
        )

    monkeypatch.setattr(workflow_module, "build_binance_market_dataset", build)
    output = tmp_path / "canonical-m2-v2"

    result = bootstrap_canonical_m2_study(config_path, output)

    assert result.config_digest == config.digest
    assert inspect_canonical_m2_bootstrap(output) == result
