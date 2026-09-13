from __future__ import annotations

import json

import numpy as np
import pytest

from trade_rl.data import (
    MarketDataset,
    load_market_dataset_artifact,
    publish_market_dataset_artifact,
)
from trade_rl.evaluation.experiments.bootstrap.capacity_materialization import (
    build_aggtrades_calibrated_capacity_dataset,
    canonical_aggtrades_capacity_authority,
)

_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")


def _source_dataset() -> MarketDataset:
    n = 4
    close = np.full((n, len(_SYMBOLS)), 100.0, dtype=np.float64)
    base = MarketDataset(
        dataset_id="0" * 64,
        symbols=_SYMBOLS,
        timestamps=np.datetime64("2022-01-01", "ns")
        + (np.arange(n) + 1) * np.timedelta64(1, "h"),
        features=np.zeros((n, len(_SYMBOLS), 1), dtype=np.float32),
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n, len(_SYMBOLS)), 1_000.0, dtype=np.float64),
        funding_rate=np.zeros((n, len(_SYMBOLS)), dtype=np.float64),
        tradable=np.ones((n, len(_SYMBOLS)), dtype=np.bool_),
        feature_available=np.ones((n, len(_SYMBOLS), 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        max_participation_rate=np.full(
            (n, len(_SYMBOLS)),
            0.05,
            dtype=np.float64,
        ),
    )
    return base.with_content_identity(
        {
            "raw_source": "synthetic-source-fixture",
            "execution_economics": {
                "schema_version": "execution_economics_profile_v1",
                "fee_rate": 0.0005,
                "maker_fee_rate": 0.0,
                "taker_fee_rate": 0.0,
                "spread_rate": 0.0002,
                "max_participation_rate": 0.05,
                "borrow_available": True,
                "borrow_rate": 0.0,
            },
        }
    )


def test_calibrated_dataset_changes_only_participation_array_and_identity(
    tmp_path,
) -> None:
    source = _source_dataset()
    source_publication = publish_market_dataset_artifact(tmp_path / "source", source)
    authority = canonical_aggtrades_capacity_authority()

    successor = build_aggtrades_calibrated_capacity_dataset(
        source,
        source_artifact_schema="market_dataset_artifact_v3",
        source_artifact_digest=source_publication.artifact_digest,
        authority=authority,
    )

    assert successor.dataset_id != source.dataset_id
    for name, source_array in source.identity_arrays().items():
        successor_array = successor.identity_arrays()[name]
        assert successor_array.dtype == source_array.dtype
        assert successor_array.shape == source_array.shape
        if name == "max_participation_rate":
            expected = np.broadcast_to(
                np.asarray(authority.symbol_caps, dtype=np.float64),
                source_array.shape,
            )
            assert np.array_equal(successor_array, expected)
        else:
            assert source_array.tobytes() == successor_array.tobytes(), name

    identity = json.loads(successor.identity_payload_json or "{}")
    assert "execution_economics" not in identity
    assert identity["source_dataset"]["dataset_id"] == source.dataset_id
    assert (
        identity["source_dataset"]["artifact_digest"]
        == source_publication.artifact_digest
    )
    assert (
        identity["source_dataset"]["identity_payload"]["execution_economics"][
            "max_participation_rate"
        ]
        == 0.05
    )
    assert identity["execution_capacity_calibration"] == authority.to_payload()

    successor_publication = publish_market_dataset_artifact(
        tmp_path / "successor",
        successor,
    )
    assert successor_publication.artifact_digest != source_publication.artifact_digest
    reloaded = load_market_dataset_artifact(tmp_path / "successor")
    assert reloaded.dataset_id == successor.dataset_id
    assert np.array_equal(
        reloaded.resolved_array("max_participation_rate"),
        successor.resolved_array("max_participation_rate"),
    )


def test_calibration_authority_preserves_exact_symbol_cap_mapping_without_pnl_inputs() -> (
    None
):
    authority = canonical_aggtrades_capacity_authority()

    assert authority.symbols == _SYMBOLS
    assert authority.symbol_caps == (
        0.0021629560553901974,
        0.002044685341258238,
        0.002184898995567895,
        0.0020480213652913385,
        0.002346308308284808,
    )
    assert authority.protocol_digest == (
        "5fb013fb0a3d717846a701b23d2f4bfca8e742ebaef0e6f564a331053f2ed071"
    )
    assert authority.result_digest == (
        "a5490f9209b16a8ba94f11506b83aa45ce44c7d61d13234480e2362f6df89809"
    )
    assert authority.result_json_sha256 == (
        "397004e6a1a9c3b30b32212066ac41366f7af74edc0ae23372f847e50c6c5a4e"
    )
    forbidden = {
        "returns",
        "pnl",
        "strategy",
        "candidate_result",
        "experiment_result",
    }
    assert forbidden.isdisjoint(authority.to_payload())


def test_calibrated_dataset_rejects_symbol_order_drift(tmp_path) -> None:
    source = _source_dataset()
    source_publication = publish_market_dataset_artifact(tmp_path / "source", source)
    malformed = MarketDataset(
        **{
            **{
                field: getattr(source, field)
                for field in source.__dataclass_fields__
                if source.__dataclass_fields__[field].init
            },
            "symbols": tuple(reversed(source.symbols)),
            "dataset_id": "1" * 64,
            "identity_payload_json": None,
        }
    )

    with pytest.raises(ValueError, match="symbol roster"):
        build_aggtrades_calibrated_capacity_dataset(
            malformed,
            source_artifact_schema="market_dataset_artifact_v3",
            source_artifact_digest=source_publication.artifact_digest,
            authority=canonical_aggtrades_capacity_authority(),
        )
