from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.data.metadata_promotion import (
    load_metadata_promotion_evidence,
    metadata_promotion_from_dataset,
    write_metadata_promotion_evidence,
)


def _dataset(metadata_evidence: dict[str, object]) -> MarketDataset:
    timestamps = np.datetime64("2026-01-01", "ns") + np.arange(3) * np.timedelta64(
        1, "h"
    )
    close = np.asarray([[100.0], [101.0], [102.0]])
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=("BTCUSDT",),
        timestamps=timestamps,
        features=np.ones((3, 1, 1), dtype=np.float32),
        global_features=np.ones((3, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((3, 1), 1_000_000.0),
        funding_rate=np.zeros((3, 1)),
        tradable=np.ones((3, 1), dtype=np.bool_),
        feature_available=np.ones((3, 1, 1), dtype=np.bool_),
        feature_names=("return",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )
    return dataset.with_content_identity({"metadata_evidence": metadata_evidence})


def _evidence(*, historical: bool) -> dict[str, object]:
    return {
        "authentication": "ed25519" if historical else "none",
        "coverage": {
            "application": (
                "effective-dated-full-interval"
                if historical
                else "static-full-interval"
            ),
            "end_time": "2026-01-04T00:00:00+00:00",
            "start_time": "2026-01-01T00:00:00+00:00",
        },
        "limitations": [] if historical else ["static rules"],
        "mode": "historical_signed" if historical else "frozen_snapshot",
        "point_in_time": historical,
        "source_payload_digest": "1" * 64,
    }


def test_historical_metadata_promotion_round_trips(tmp_path: Path) -> None:
    promotion = metadata_promotion_from_dataset(_dataset(_evidence(historical=True)))
    promotion.require_promotable()
    path = tmp_path / "metadata-promotion.json"
    write_metadata_promotion_evidence(path, promotion)
    assert load_metadata_promotion_evidence(path) == promotion


def test_static_metadata_cannot_be_promoted() -> None:
    promotion = metadata_promotion_from_dataset(_dataset(_evidence(historical=False)))
    assert promotion.promotable is False
    with pytest.raises(ValueError, match="historical_signed"):
        promotion.require_promotable()
