from __future__ import annotations

from pathlib import Path


def test_training_compose_separates_market_data_ownership() -> None:
    compose = (
        Path(__file__).resolve().parents[1] / "compose.training.yaml"
    ).read_text(encoding="utf-8")

    assert "market-data-sync:" in compose
    assert "trade-rl-market-archives:/workspace/market-data/binance-vision" in compose
    assert (
        "trade-rl-market-archives:/workspace/market-data/binance-vision:ro"
        in compose
    )
    assert "trade-rl-training-runs:/workspace/var/runs" in compose
    assert "trade-rl-teacher-cache:/workspace/var/cache/teacher-artifacts" in compose
    assert "trade-rl-training-data:/workspace/var" not in compose
