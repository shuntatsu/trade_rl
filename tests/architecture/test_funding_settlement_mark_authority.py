from __future__ import annotations

from trade_rl.data.artifacts.codec import DATASET_ARTIFACT_SCHEMA
from trade_rl.data.identity import (
    DATASET_ID_ARRAY_FIELDS,
    MARKET_DATASET_IDENTITY_SCHEMA,
)


def test_settlement_mark_notional_is_identity_and_artifact_bound() -> None:
    assert MARKET_DATASET_IDENTITY_SCHEMA == "market_dataset_identity_v7"
    assert "funding_price_rate" in DATASET_ID_ARRAY_FIELDS
    assert DATASET_ARTIFACT_SCHEMA == "market_dataset_artifact_v4"
