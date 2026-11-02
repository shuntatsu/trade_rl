from __future__ import annotations

import json

import numpy as np
import pytest

from tools.issue810_joint_fee_only_fixture import build_joint_fee_only_fixture


def test_fixture_identity_and_all_identity_arrays_are_reproducible() -> None:
    dataset, recipe = build_joint_fee_only_fixture()
    repeated, repeated_recipe = build_joint_fee_only_fixture()

    assert dataset.dataset_id != "0" * 64
    assert dataset.identity_verified
    assert repeated.identity_verified
    assert dataset.dataset_id == repeated.dataset_id
    assert dataset.identity_payload_json == repeated.identity_payload_json
    assert dict(recipe) == dict(repeated_recipe)

    payload = json.loads(dataset.identity_payload_json or "{}")
    serialized_recipe = json.loads(json.dumps(dict(recipe), sort_keys=True))
    assert payload["schema"] == "market_dataset_identity_v6"
    assert payload["dataset_contract"] == {
        "calendar_kind": "continuous_24_7",
        "feature_config_digest": "0" * 64,
        "feature_names": ["synthetic_zero"],
        "global_feature_names": ["synthetic_global_zero"],
        "nominal_bar_hours": 1.0,
        "normalization_digest": "0" * 64,
        "periods_per_year": 8_760,
        "symbols": ["AAA", "BBB"],
        "volume_units": ["base_asset", "base_asset"],
    }
    assert payload["source_declared_synthetic_recipe"] == serialized_recipe
    assert set(payload) == {
        "schema",
        "dataset_contract",
        "source_declared_synthetic_recipe",
    }

    first_arrays = dataset.identity_arrays()
    repeated_arrays = repeated.identity_arrays()
    assert first_arrays.keys() == repeated_arrays.keys()
    for name, array in first_arrays.items():
        assert not array.flags.writeable, name
        assert not repeated_arrays[name].flags.writeable, name
        np.testing.assert_array_equal(array, repeated_arrays[name], err_msg=name)


def test_fixture_uses_exact_utc_hour_rows_shapes_and_available_zero_features() -> None:
    dataset, recipe = build_joint_fee_only_fixture()
    expected = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        37, dtype=np.int64
    ) * np.timedelta64(1, "h")

    assert dataset.symbols == ("AAA", "BBB")
    assert dataset.timestamps.shape == (37,)
    np.testing.assert_array_equal(dataset.timestamps, expected)
    assert dataset.timestamps[0] == np.datetime64("2026-01-01T00:00:00", "ns")
    assert dataset.timestamps[-1] == np.datetime64("2026-01-02T12:00:00", "ns")
    assert dataset.regular_cadence
    assert dataset.bar_hours == 1.0
    assert dataset.periods_per_year == 8_760
    assert dataset.features.shape == (37, 2, 1)
    assert dataset.features.dtype == np.dtype(np.float32)
    assert dataset.global_features.shape == (37, 1)
    assert dataset.global_features.dtype == np.dtype(np.float32)
    assert dataset.feature_names == ("synthetic_zero",)
    assert dataset.global_feature_names == ("synthetic_global_zero",)
    np.testing.assert_array_equal(dataset.features, np.zeros((37, 2, 1), np.float32))
    np.testing.assert_array_equal(
        dataset.global_features, np.zeros((37, 1), np.float32)
    )
    np.testing.assert_array_equal(dataset.feature_available, np.ones((37, 2, 1), bool))
    np.testing.assert_array_equal(
        dataset.global_feature_available, np.ones((37, 1), bool)
    )
    np.testing.assert_array_equal(
        dataset.identity_arrays()["information_available"], np.ones((37, 2), bool)
    )
    np.testing.assert_array_equal(
        dataset.identity_arrays()["available_at"],
        np.broadcast_to(expected[:, None], (37, 2)),
    )
    np.testing.assert_array_equal(
        dataset.feature_staleness, np.zeros((37, 2, 1), np.float32)
    )
    np.testing.assert_array_equal(
        dataset.feature_staleness_hours, np.zeros((37, 2, 1), np.float32)
    )
    np.testing.assert_array_equal(
        dataset.global_feature_staleness_hours, np.zeros((37, 1), np.float32)
    )
    assert recipe["timestamp_start_utc"] == "2026-01-01T00:00:00Z"
    assert recipe["row_count"] == 37
    assert recipe["bar_interval_seconds"] == 3_600
    assert recipe["feature_units"] == ("dimensionless",)
    assert recipe["global_feature_units"] == ("dimensionless",)


def test_fixture_declares_zero_dataset_costs_and_fixed_capacity_boundary() -> None:
    dataset, recipe = build_joint_fee_only_fixture()
    arrays = dataset.identity_arrays()

    for name in (
        "funding_rate",
        "fee_rate",
        "maker_fee_rate",
        "taker_fee_rate",
        "spread_rate",
        "borrow_rate",
        "cash_rate",
        "dividend",
        "feature_missing_reason",
        "global_feature_missing_reason",
    ):
        np.testing.assert_array_equal(
            arrays[name], np.zeros_like(arrays[name]), err_msg=name
        )
    np.testing.assert_array_equal(
        arrays["funding_event_count"], np.zeros((37, 2), np.int32)
    )
    np.testing.assert_array_equal(arrays["funding_due"], np.zeros((37, 2), bool))
    np.testing.assert_array_equal(arrays["tradable"], np.ones((37, 2), bool))
    np.testing.assert_array_equal(arrays["symbol_active"], np.ones((37, 2), bool))
    np.testing.assert_array_equal(arrays["borrow_available"], np.ones((37, 2), bool))
    np.testing.assert_array_equal(arrays["buy_allowed"], np.ones((37, 2), bool))
    np.testing.assert_array_equal(arrays["sell_allowed"], np.ones((37, 2), bool))
    np.testing.assert_array_equal(arrays["open"], np.full((37, 2), 100.0))
    np.testing.assert_array_equal(arrays["high"], np.full((37, 2), 100.0))
    np.testing.assert_array_equal(arrays["low"], np.full((37, 2), 100.0))
    np.testing.assert_array_equal(arrays["close"], np.full((37, 2), 100.0))
    np.testing.assert_array_equal(arrays["mark_price"], np.full((37, 2), 100.0))
    np.testing.assert_array_equal(arrays["volume"], np.full((37, 2), 1_000.0))
    np.testing.assert_array_equal(arrays["index_price"], np.full((37, 2), 100.0))
    np.testing.assert_array_equal(arrays["split_factor"], np.ones((37, 2)))
    np.testing.assert_array_equal(arrays["delisting_recovery"], np.ones((37, 2)))
    np.testing.assert_array_equal(arrays["max_participation_rate"], np.ones((37, 2)))
    np.testing.assert_array_equal(arrays["minimum_notional"], np.zeros((37, 2)))
    np.testing.assert_array_equal(arrays["lot_size"], np.zeros((37, 2)))
    np.testing.assert_array_equal(arrays["tick_size"], np.zeros((37, 2)))
    np.testing.assert_array_equal(arrays["contract_multipliers"], np.ones(2))
    assert dataset.volume_units == ("base_asset", "base_asset")
    assert recipe["dataset_fee_rate"] == 0.0
    assert recipe["dataset_spread_rate"] == 0.0
    assert recipe["dataset_funding_rate"] == 0.0
    assert recipe["dataset_borrow_rate"] == 0.0
    assert recipe["volume"] == 1_000.0
    assert recipe["dataset_max_participation_rate"] == 1.0
    assert recipe["minimum_notional"] == 0.0
    assert recipe["lot_size"] == 0.0
    assert recipe["tick_size"] == 0.0
    assert recipe["contract_multipliers"] == (1.0, 1.0)
    assert recipe["dataset_cash_rate"] == 0.0
    assert recipe["dataset_dividend"] == 0.0
    assert recipe["split_factor"] == 1.0
    assert recipe["delisting_recovery"] == 1.0
    assert recipe["feature_config_digest"] == "0" * 64
    assert recipe["normalization_digest"] == "0" * 64
    assert recipe["execution_cost_configuration_included"] is False
    assert recipe["schema"] == "issue810_joint_fee_only_synthetic_dataset_v1"
    assert recipe["source"] == "tools/issue810_joint_fee_only_fixture.py"
    assert recipe["construction"] == "deterministic_in_memory_synthetic_only"
    assert recipe["timestamp_stop_utc_inclusive"] == "2026-01-02T12:00:00Z"
    assert recipe["index_price_value"] == 100.0
    assert recipe["all_global_feature_staleness_hours"] == 0.0
    with pytest.raises(TypeError):
        recipe["dataset_fee_rate"] = 1.0  # type: ignore[index]
