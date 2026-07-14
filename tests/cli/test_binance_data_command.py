from __future__ import annotations

import io
import json
from types import SimpleNamespace

from trade_rl.cli import app


def test_data_binance_builds_and_publishes_dataset(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    dataset = SimpleNamespace(
        dataset_id="a" * 64,
        n_bars=168,
        n_features=5,
        n_symbols=1,
        symbols=("BTCUSDT",),
        timestamps=[],
    )

    def fake_build(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(
            dataset=dataset,
            sources_used=("vision",),
            metadata=(),
        )

    def fake_publish(output: object, observed: object) -> object:
        captured["output"] = output
        captured["published_dataset"] = observed
        return SimpleNamespace(artifact_digest="b" * 64)

    monkeypatch.setattr(app, "build_binance_market_dataset", fake_build)
    monkeypatch.setattr(app, "publish_market_dataset_artifact", fake_publish)
    stdout = io.StringIO()

    exit_code = app.main(
        [
            "data",
            "binance",
            "--market",
            "usds-m",
            "--symbol",
            "BTCUSDT",
            "--interval",
            "1h",
            "--start-time",
            "2026-06-01T00:00:00Z",
            "--end-time",
            "2026-06-08T00:00:00Z",
            "--transport",
            "vision",
            "--tick-size",
            "0.1",
            "--lot-size",
            "0.001",
            "--minimum-notional",
            "5",
            "--output",
            str(tmp_path / "dataset"),
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    payload = json.loads(stdout.getvalue())
    assert payload == {
        "artifact_digest": "b" * 64,
        "dataset_id": "a" * 64,
        "end_time": "2026-06-08T00:00:00+00:00",
        "interval": "1h",
        "market": "usds-m",
        "n_bars": 168,
        "n_features": 5,
        "n_symbols": 1,
        "production_status": "NO-GO",
        "schema": "binance_dataset_build_result_v1",
        "sources_used": ["vision"],
        "start_time": "2026-06-01T00:00:00+00:00",
        "symbols": ["BTCUSDT"],
        "transport": "vision",
    }
    assert captured["market"] == "usds-m"
    assert captured["symbols"] == ("BTCUSDT",)
    assert captured["tick_sizes"] == (0.1,)
    assert captured["lot_sizes"] == (0.001,)
    assert captured["minimum_notionals"] == (5.0,)
    assert captured["published_dataset"] is dataset


def test_data_binance_requires_metadata_cardinality_per_symbol(tmp_path) -> None:
    stdout = io.StringIO()

    try:
        app.main(
            [
                "data",
                "binance",
                "--market",
                "usds-m",
                "--symbol",
                "BTCUSDT",
                "--symbol",
                "ETHUSDT",
                "--interval",
                "1h",
                "--start-time",
                "2026-06-01T00:00:00Z",
                "--end-time",
                "2026-06-08T00:00:00Z",
                "--tick-size",
                "0.1",
                "--output",
                str(tmp_path / "dataset"),
            ],
            stdout=stdout,
        )
    except ValueError as error:
        assert "tick-size" in str(error)
    else:
        raise AssertionError("metadata cardinality mismatch was not rejected")
