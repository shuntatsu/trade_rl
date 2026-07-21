from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from trade_rl.catalog.contracts import (
    ArtifactKind,
    ArtifactQuery,
    ArtifactRegistration,
    ArtifactStatus,
    cache_key_digest,
)


def test_cache_key_digest_is_canonical_and_order_independent() -> None:
    left = cache_key_digest({"dataset_id": "a" * 64, "fold": {"end": 20, "start": 0}})
    right = cache_key_digest({"fold": {"start": 0, "end": 20}, "dataset_id": "a" * 64})

    assert left == right
    assert len(left) == 64


def test_registration_validates_and_freezes_json_payloads() -> None:
    registration = ArtifactRegistration(
        artifact_digest="a" * 64,
        artifact_kind=ArtifactKind.MARKET_DATASET,
        schema_version="market_dataset_artifact_v3",
        cache_key={"symbols": ["BTCUSDT", "ETHUSDT"], "range": {"start": 0, "end": 20}},
        metadata={"rows": 20},
        location="/tmp/dataset",
        size_bytes=100,
        dataset_id="b" * 64,
    )

    assert registration.cache_key_digest == cache_key_digest(registration.cache_key)
    assert registration.cache_key["symbols"] == ("BTCUSDT", "ETHUSDT")
    with pytest.raises(TypeError):
        registration.cache_key["other"] = 1  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        registration.size_bytes = 101  # type: ignore[misc]


@pytest.mark.parametrize("digest", ["", "x" * 64, "a" * 63])
def test_registration_rejects_invalid_digest(digest: str) -> None:
    with pytest.raises(ValueError, match="artifact_digest"):
        ArtifactRegistration(
            artifact_digest=digest,
            artifact_kind=ArtifactKind.MODEL,
            schema_version="model_v1",
            cache_key={"key": "value"},
            metadata={},
            location="/tmp/model",
            size_bytes=0,
        )


def test_registration_rejects_non_json_and_negative_size() -> None:
    with pytest.raises(ValueError, match="JSON"):
        ArtifactRegistration(
            artifact_digest="a" * 64,
            artifact_kind=ArtifactKind.MODEL,
            schema_version="model_v1",
            cache_key={"bad": object()},
            metadata={},
            location="/tmp/model",
            size_bytes=0,
        )
    with pytest.raises(ValueError, match="size_bytes"):
        ArtifactRegistration(
            artifact_digest="a" * 64,
            artifact_kind=ArtifactKind.MODEL,
            schema_version="model_v1",
            cache_key={"key": "value"},
            metadata={},
            location="/tmp/model",
            size_bytes=-1,
        )


def test_query_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        ArtifactQuery(status=ArtifactStatus.READY, limit=0)
