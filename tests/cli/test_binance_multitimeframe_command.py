from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from trade_rl.cli import app


def test_data_binance_passes_ordered_feature_timeframes(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    dataset = SimpleNamespace(
        dataset_id="a" * 64,
        n_bars=13_104,
        n_features=10,
        n_symbols=3,
        symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
        timestamps=[],
    )

    def fake_build(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(
            dataset=dataset,
            sources_used=("vision",),
            metadata=(),
            feature_timeframes=("15m", "1h", "4h", "1d"),
        )

    monkeypatch.setattr(app, "build_binance_market_dataset", fake_build)
    monkeypatch.setattr(
        app,
        "publish_market_dataset_artifact",
        lambda *_: SimpleNamespace(artifact_digest="b" * 64),
    )
    stdout = io.StringIO()

    exit_code = app.main(
        [
            "data",
            "binance",
            "--market",
            "usds-m",
            "--symbol",
            "BTCUSDT",
            "--symbol",
            "ETHUSDT",
            "--symbol",
            "BNBUSDT",
            "--interval",
            "1h",
            "--feature-timeframe",
            "15m",
            "--feature-timeframe",
            "4h",
            "--feature-timeframe",
            "1d",
            "--start-time",
            "2024-12-01T00:00:00Z",
            "--end-time",
            "2026-06-01T00:00:00Z",
            "--tick-size",
            "0.1",
            "--tick-size",
            "0.01",
            "--tick-size",
            "0.01",
            "--lot-size",
            "0.001",
            "--lot-size",
            "0.001",
            "--lot-size",
            "0.01",
            "--minimum-notional",
            "5",
            "--minimum-notional",
            "5",
            "--minimum-notional",
            "5",
            "--listed-at",
            "2019-09-08T00:00:00Z",
            "--listed-at",
            "2019-11-27T00:00:00Z",
            "--listed-at",
            "2020-02-10T00:00:00Z",
            "--output",
            str(tmp_path / "dataset"),
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    assert captured["feature_timeframes"] == ("15m", "4h", "1d")
    payload = json.loads(stdout.getvalue())
    assert payload["feature_timeframes"] == ["15m", "1h", "4h", "1d"]
    assert payload["n_symbols"] == 3
    assert payload["n_features"] == 10


def test_data_binance_rejects_duplicate_feature_timeframes(tmp_path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        app.main(
            [
                "data",
                "binance",
                "--market",
                "usds-m",
                "--symbol",
                "BTCUSDT",
                "--interval",
                "1h",
                "--feature-timeframe",
                "4h",
                "--feature-timeframe",
                "4h",
                "--start-time",
                "2026-01-01T00:00:00Z",
                "--end-time",
                "2026-02-01T00:00:00Z",
                "--output",
                str(tmp_path / "dataset"),
            ],
            stdout=io.StringIO(),
        )
