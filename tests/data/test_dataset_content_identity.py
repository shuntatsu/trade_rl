from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data import load_market_dataset_artifact, write_market_dataset_files
from trade_rl.data.contracts import MarketCalendarKind, VolumeUnit
from trade_rl.data.market import MarketDataset


def dataset(*, fee_rate: float = 0.0) -> MarketDataset:
    n_bars = 8
    close = np.full((n_bars, 1), 100.0)
    return MarketDataset(
        dataset_id="0" * 64,
        symbols=("ASSET",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n_bars, 1), 1_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("feature",),
        global_feature_names=("global",),
        periods_per_year=8_760,
        fee_rate=np.full((n_bars, 1), fee_rate),
    )


def test_execution_semantics_are_part_of_dataset_identity() -> None:
    zero_fee = dataset(fee_rate=0.0).with_content_identity()
    charged = dataset(fee_rate=0.001).with_content_identity()
    assert zero_fee.dataset_id != charged.dataset_id


def test_formal_artifact_rejects_unidentified_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="content identity"):
        write_market_dataset_files(tmp_path, dataset())


def test_content_identity_round_trip_recomputes_all_arrays(tmp_path: Path) -> None:
    original = dataset(fee_rate=0.001).with_content_identity({"source": "unit-test"})
    write_market_dataset_files(tmp_path, original)
    restored = load_market_dataset_artifact(tmp_path)
    assert restored.dataset_id == original.dataset_id
    assert restored.identity_payload_json == original.identity_payload_json

    with pytest.raises(ValueError, match="dataset_id"):
        replace(
            restored,
            dataset_id="f" * 64,
            fee_rate=np.full((restored.n_bars, restored.n_symbols), 0.002),
        )


def test_legacy_identity_rejects_changed_funding_price_rate() -> None:
    legacy = load_market_dataset_artifact(
        Path(__file__).parent / "fixtures" / "market_build_v2_artifact"
    )
    assert legacy.identity_payload_json is not None
    assert '"schema":"market_dataset_identity_v6"' in legacy.identity_payload_json
    changed = legacy.resolved_array("funding_price_rate").copy()
    changed[0, 0] += 1.0

    with pytest.raises(ValueError, match="legacy dataset identity.*funding"):
        replace(legacy, funding_price_rate=changed)


def test_content_identified_dataset_arrays_cannot_be_reenabled_for_write() -> None:
    market = dataset(fee_rate=0.001).with_content_identity({"source": "unit-test"})

    arrays = (
        market.timestamps,
        market.features,
        market.global_features,
        market.close,
        market.tradable,
        market.feature_available,
        market.fee_rate,
        market.information_available,
        market.contract_multipliers,
    )
    for array in arrays:
        assert array.flags.writeable is False
        with pytest.raises(ValueError, match="WRITEABLE|writeable|writable"):
            array.setflags(write=True)


def test_market_dataset_internal_caches_are_deeply_immutable() -> None:
    market = dataset().with_content_identity({"source": "unit-test"})

    for array in (
        market._timestamp_ns,
        market._eligibility_invalid_prefix,
        market._feature_invalid_prefix,
    ):
        assert array.flags.writeable is False
        with pytest.raises(ValueError, match="WRITEABLE|writeable|writable"):
            array.setflags(write=True)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("symbols", ("OTHER",)),
        ("feature_names", ("other_feature",)),
        ("global_feature_names", ("other_global",)),
        ("volume_units", (VolumeUnit.QUOTE_NOTIONAL,)),
        ("feature_config_digest", "a" * 64),
        ("normalization_digest", "b" * 64),
        ("calendar_kind", MarketCalendarKind.CONTINUOUS),
        ("nominal_bar_hours", 2.0),
        ("periods_per_year", 252),
    ],
)
def test_content_identity_rejects_changed_scalar_meaning(
    field_name: str, replacement: object
) -> None:
    original = replace(
        dataset(), calendar_kind=MarketCalendarKind.SESSION
    ).with_content_identity()

    with pytest.raises(ValueError, match="dataset_id.*contract"):
        replace(original, **{field_name: replacement})


def test_recomputed_identity_detects_live_feature_label_drift() -> None:
    original = dataset().with_content_identity()
    object.__setattr__(original, "feature_names", ("other_feature",))

    assert original.recomputed_dataset_id() != original.dataset_id
    assert original.identity_verified is False


def test_explicitly_reidentified_scalar_change_has_a_new_identity() -> None:
    original = dataset().with_content_identity()
    successor = replace(
        original, feature_names=("other_feature",), identity_payload_json=None
    ).with_content_identity()

    assert successor.dataset_id != original.dataset_id
    assert successor.identity_verified is True


@pytest.mark.parametrize("change", ["missing", "malformed", "changed"])
def test_stored_descriptor_tampering_is_not_verified(change: str) -> None:
    original = dataset().with_content_identity()
    payload = json.loads(original.identity_payload_json)
    if change == "missing":
        del payload["dataset_contract"]
    elif change == "malformed":
        payload["dataset_contract"] = "invalid"
    else:
        payload["dataset_contract"]["feature_names"] = ["other_feature"]
    object.__setattr__(original, "identity_payload_json", json.dumps(payload))

    assert original.identity_verified is False
