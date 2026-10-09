"""G2 causal/mathematical counterexamples for opt-in path signatures."""

from dataclasses import replace
from math import log

import numpy as np
import pytest

from tests.evaluation.test_shared_cash_replay import _market
from trade_rl.data.features.signature import with_path_signatures


def test_same_endpoints_opposite_paths_have_opposite_signed_area() -> None:
    source = _market(np.array([[100.0, 100.0], [102.0, 98.0], [100.0, 100.0]]))
    result = with_path_signatures(source, window_bars=3, depth=2)
    prefix = "path_sig_v1_w3_d2_tp"
    assert not result.feature_available[:2, :, -6:].any()
    assert result.feature_available[2, :, -6:].all()
    expected_a = log(1.02)
    expected_b = log(0.98)
    tp = result.feature_names.index(f"{prefix}_tp")
    pt = result.feature_names.index(f"{prefix}_pt")
    np.testing.assert_allclose(
        result.features[2, :, tp], [-expected_a / 2, -expected_b / 2], atol=1e-7
    )
    np.testing.assert_allclose(
        result.features[2, :, pt], [expected_a / 2, expected_b / 2], atol=1e-7
    )
    np.testing.assert_allclose(
        result.features[2, :, result.feature_names.index(f"{prefix}_p")],
        [0, 0],
        atol=1e-7,
    )


def test_constant_path_has_expected_tensor_factorials() -> None:
    source = _market(np.full((5, 1), 100.0))
    result = with_path_signatures(source, window_bars=3, depth=3)
    prefix = "path_sig_v1_w3_d3_tp"
    assert result.features[2, 0, result.feature_names.index(f"{prefix}_t")] == 1
    np.testing.assert_allclose(
        [
            result.features[2, 0, result.feature_names.index(f"{prefix}_tt")],
            result.features[2, 0, result.feature_names.index(f"{prefix}_ttt")],
        ],
        [0.5, 1.0 / 6.0],
        atol=1e-7,
    )
    for word in ("p", "tp", "pt", "pp", "tpp", "ptp", "ppt"):
        assert (
            result.features[2, 0, result.feature_names.index(f"{prefix}_{word}")] == 0
        )


def test_chen_identity_independent_closed_form_two_segments() -> None:
    source = _market(np.array([[100.0], [105.0], [101.0]]))
    result = with_path_signatures(source, window_bars=3, depth=2)
    a = np.array([0.5, log(105.0) - log(100.0)])
    b = np.array([0.5, log(101.0) - log(105.0)])
    level_two = 0.5 * np.outer(a, a) + np.outer(a, b) + 0.5 * np.outer(b, b)
    prefix = "path_sig_v1_w3_d2_tp"
    for i, word in enumerate(("tt", "tp", "pt", "pp")):
        actual = result.features[2, 0, result.feature_names.index(f"{prefix}_{word}")]
        np.testing.assert_allclose(
            actual, level_two.reshape(-1)[i], rtol=1e-6, atol=1e-7
        )


def test_future_modifications_do_not_change_prefix_features() -> None:
    source = _market(
        np.array([[100.0], [105.0], [99.0], [102.0], [103.0], [104.0], [105.0]])
    )
    changed = source.close.copy()
    changed[5:] *= 3.0
    future = replace(source, close=changed, high=np.maximum(source.high, changed))
    first = with_path_signatures(source, window_bars=3, depth=3)
    second = with_path_signatures(future, window_bars=3, depth=3)
    np.testing.assert_array_equal(first.features[:5], second.features[:5])
    np.testing.assert_array_equal(
        first.feature_available[:5], second.feature_available[:5]
    )


def test_unavailable_and_late_rows_invalidate_full_window_without_backfill() -> None:
    source = _market(np.arange(100.0, 108.0).reshape(-1, 1))
    information = source.resolved_array("information_available").copy()
    information[3, 0] = False
    late = source.resolved_array("available_at").copy()
    late[5, 0] = source.timestamps[6]
    modified = replace(source, information_available=information, available_at=late)
    result = with_path_signatures(modified, window_bars=3)
    assert result.feature_available[2, 0, -6:].all()
    assert not result.feature_available[3:8, 0, -6:].any()
    assert (result.features[3:8, 0, -6:] == 0).all()
    assert (result.feature_staleness[3:8, 0, -6:] == 1).all()
    assert (result.feature_missing_reason[3:8, 0, -6:] != 0).all()


def test_volume_is_optional_and_zero_volume_blocks_only_volume_mode() -> None:
    volume = np.array([[100.0], [0.0], [250.0], [500.0], [600.0]])
    source = _market(
        np.array([[100.0], [101.0], [102.0], [103.0], [104.0]]),
        volume=volume,
    )
    with_volume = with_path_signatures(
        source, window_bars=3, depth=2, include_volume=True
    )
    without_volume = with_path_signatures(source, window_bars=3, depth=2)
    assert with_volume.n_features - source.n_features == 12
    assert not with_volume.feature_available[2, 0, -12:].any()
    assert with_volume.feature_available[4, 0, -12:].all()
    assert without_volume.feature_available[2, 0, -6:].all()
    np.testing.assert_array_equal(
        source.features, with_volume.features[:, :, : source.n_features]
    )


def test_scale_invariance_and_no_ohlc_intrabar_order_lookups() -> None:
    source = _market(np.array([[100.0], [105.0], [99.0], [110.0], [107.0]]))
    scaled_close = source.close * 10.0
    scaled = replace(
        source, close=scaled_close, high=np.maximum(source.high, scaled_close)
    )
    one = with_path_signatures(source, window_bars=3)
    two = with_path_signatures(scaled, window_bars=3)
    np.testing.assert_allclose(
        one.features[:, :, -6:], two.features[:, :, -6:], atol=1e-6
    )
    ohlc = replace(source, high=source.high * 1.5, low=source.low * 0.5)
    with_different_ohlc = with_path_signatures(ohlc, window_bars=3)
    np.testing.assert_array_equal(one.features, with_different_ohlc.features)


def test_deterministic_dataset_identity_and_existing_economics() -> None:
    source = _market(np.array([[100.0], [102.0], [99.0], [103.0], [105.0]]))
    a = with_path_signatures(source, window_bars=3)
    b = with_path_signatures(source, window_bars=3)
    c = with_path_signatures(source, window_bars=4)
    assert a.identity_verified
    assert a.dataset_id == b.dataset_id
    assert a.dataset_id != source.dataset_id
    assert a.dataset_id != c.dataset_id
    np.testing.assert_array_equal(a.close, source.close)
    np.testing.assert_array_equal(a.fee_rate, source.fee_rate)
    np.testing.assert_array_equal(a.funding_rate, source.funding_rate)
    np.testing.assert_array_equal(
        a.feature_available[:, :, : source.n_features], source.feature_available
    )
    with pytest.raises(ValueError, match="already exist"):
        with_path_signatures(a, window_bars=3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_bars": 2}, "window_bars"),
        ({"window_bars": True}, "window_bars"),
        ({"window_bars": 1000}, "window_bars"),
        ({"depth": 0}, "depth"),
        ({"depth": 4}, "depth"),
        ({"depth": True}, "depth"),
        ({"include_volume": 1}, "include_volume"),
    ],
)
def test_invalid_budgets_fail_closed(kwargs: dict[str, object], message: str) -> None:
    source = _market(np.full((6, 1), 100.0))
    with pytest.raises(ValueError, match=message):
        with_path_signatures(source, **({"window_bars": 3} | kwargs))
