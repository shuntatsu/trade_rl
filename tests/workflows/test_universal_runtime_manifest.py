from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trade_rl.integrations.postgres_market_tables import (
    UNIVERSAL_202411_202607_CACHE_ID,
    UNIVERSAL_202411_202607_TABLES,
)
from trade_rl.workflows.universal_runtime_manifest import (
    UniversalRuntimeManifest,
    load_universal_runtime_manifest,
    write_universal_runtime_manifest,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def runtime_manifest_fixture() -> UniversalRuntimeManifest:
    symbols = tuple(f"ASSET{index:02d}USDT" for index in range(15))
    return UniversalRuntimeManifest(
        cache_id=UNIVERSAL_202411_202607_CACHE_ID,
        tables=UNIVERSAL_202411_202607_TABLES,
        research_start=datetime(2024, 11, 13, tzinfo=UTC),
        research_end=datetime(2026, 7, 5, tzinfo=UTC),
        instrument_artifact_relpath=Path("instruments"),
        dataset_artifact_relpath=Path("datasets"),
        normalizer_artifact_relpath=Path("normalizer"),
        train_symbols=symbols[:9],
        validation_symbols=symbols[9:12],
        test_symbols=symbols[12:],
        fold_train_range=(0, 50_000),
        shared_complete_row_count=50_000,
        catalog_digest=_digest("catalog"),
        partition_digest=_digest("partition"),
        split_manifest_digest=_digest("split"),
        feature_schema_digest=_digest("features"),
        statistics_digest=_digest("statistics"),
        metadata_evidence_digest=_digest("metadata"),
        source_manifest_digest=_digest("source"),
        dataset_digests=tuple(
            (symbol, _digest(f"dataset:{symbol}")) for symbol in symbols[:9]
        ),
    )


def test_runtime_manifest_round_trip_closes_all_static_identities(
    tmp_path: Path,
) -> None:
    expected = runtime_manifest_fixture()

    path = write_universal_runtime_manifest(
        tmp_path / "runtime-manifest.json", expected
    )
    actual = load_universal_runtime_manifest(path)

    assert actual == expected
    assert actual.train_symbols == expected.train_symbols
    assert actual.fold_train_range == (0, expected.shared_complete_row_count)
    assert len(actual.manifest_digest) == 64


@pytest.mark.parametrize(
    "secret", ("postgresql://user:password@db/x", "password", "token")
)
def test_runtime_manifest_rejects_secret_or_unknown_material(secret: str) -> None:
    payload = runtime_manifest_fixture().to_json_dict()
    payload["unexpected"] = secret

    with pytest.raises(ValueError, match="unknown|secret"):
        UniversalRuntimeManifest.from_json_dict(payload)


def test_runtime_manifest_rejects_nonportable_artifact_path() -> None:
    with pytest.raises(ValueError, match="relative"):
        UniversalRuntimeManifest.from_json_dict(
            {
                **runtime_manifest_fixture().to_json_dict(),
                "dataset_artifact_relpath": "../datasets",
            }
        )
