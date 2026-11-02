"""Train-window standardize-and-clip input transform and shift report (#1025 TD8)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.evaluation.decision_value_td8_support import signal_market
from trade_rl.evaluation.decision_value_feature_transform import (
    apply_feature_transform,
    covariate_shift_report,
    fit_train_feature_transform,
)


def test_transform_statistics_come_from_the_train_window_only() -> None:
    market = signal_market(300)
    base = fit_train_feature_transform(
        market, train_start_index=10, train_stop_index=150, clip=3.0
    )
    # Violently change everything strictly after the train window.
    features = market.features.copy()
    features[151:] = features[151:] * 50.0 + 7.0
    glob = market.global_features.copy()
    glob[151:] += 9.0
    mutated = replace(
        market, features=features, global_features=glob, identity_payload_json=None
    )
    again = fit_train_feature_transform(
        mutated, train_start_index=10, train_stop_index=150, clip=3.0
    )
    assert np.array_equal(base.feature_mean, again.feature_mean)
    assert np.array_equal(base.feature_scale, again.feature_scale)
    assert np.array_equal(base.global_mean, again.global_mean)
    assert base.digest == again.digest

    # Statistics equal the plain train-window mean / population std.
    window = market.features[10:151].astype(np.float64)
    assert np.allclose(base.feature_mean, window.mean(axis=0))
    assert np.allclose(base.feature_scale, window.std(axis=0))

    # Train rows are unchanged by later data; later rows are clipped, not refit.
    transformed = apply_feature_transform(mutated, again)
    reference = apply_feature_transform(market, base)
    assert np.array_equal(transformed.features[:151], reference.features[:151])
    assert float(np.max(np.abs(transformed.features))) <= 3.0 + 1e-6
    assert float(np.max(np.abs(transformed.global_features))) <= 3.0 + 1e-6


def test_apply_transform_clips_standardizes_and_rebinds_identity() -> None:
    market = signal_market(200)
    transform = fit_train_feature_transform(
        market, train_start_index=5, train_stop_index=120, clip=3.0
    )
    transformed = apply_feature_transform(market, transform)
    train = transformed.features[5:121].astype(np.float64)
    # Standardization (up to the clip, which only trims the tails).
    assert np.allclose(train.mean(axis=0), 0.0, atol=0.1)
    assert np.all(np.abs(transformed.features) <= 3.0 + 1e-6)
    assert transformed.dataset_id == market.dataset_id
    assert transformed.feature_config_digest == market.feature_config_digest
    # The extra transform is part of the normalization identity.
    assert transformed.normalization_digest != market.normalization_digest
    assert transformed.normalization_digest != "0" * 64
    # Prices/volumes are untouched.
    assert np.array_equal(transformed.close, market.close)
    assert transformed.feature_names == market.feature_names


def test_constant_feature_is_centered_not_divided_by_zero() -> None:
    market = signal_market(120)
    features = market.features.copy()
    features[:, :, 2] = 4.0
    market = replace(market, features=features, identity_payload_json=None)
    transform = fit_train_feature_transform(
        market, train_start_index=0, train_stop_index=100, clip=3.0
    )
    transformed = apply_feature_transform(market, transform)
    assert np.all(transformed.features[:, :, 2] == 0.0)
    assert np.isfinite(transformed.features).all()


def test_unavailable_cells_do_not_enter_statistics() -> None:
    market = signal_market(150)
    avail = market.feature_available.copy()
    features = market.features.copy()
    avail[:20, :, 1] = False
    features[:20, :, 1] = 1.0e6
    market = replace(
        market,
        features=features,
        feature_available=avail,
        feature_staleness=None,
        identity_payload_json=None,
    )
    transform = fit_train_feature_transform(
        market, train_start_index=0, train_stop_index=100, clip=3.0
    )
    assert np.all(np.abs(transform.feature_mean[:, 1]) < 5.0)


def test_covariate_shift_report_is_diagnostic_and_matches_hand_values() -> None:
    market = signal_market(300)
    features = market.features.copy()
    features[200:260, :, 1] += 10.0 * float(features[:150, :, 1].std())
    shifted = replace(market, features=features, identity_payload_json=None)
    transform = fit_train_feature_transform(
        shifted, train_start_index=0, train_stop_index=149, clip=3.0
    )
    report = covariate_shift_report(shifted, transform, start_index=200, stop_index=259)
    by_name = {row["name"]: row for row in report if row["scope"] == "symbol"}
    row = by_name["f_1"]
    window = shifted.features[200:260, :, 1].astype(np.float64)
    z = (window - transform.feature_mean[:, 1]) / transform.feature_scale[:, 1]
    assert row["mean_z"] == pytest.approx(float(z.mean()))
    assert row["max_abs_z"] == pytest.approx(float(np.abs(z).max()))
    assert row["mean_z"] > 5.0
    assert row["fraction_clipped"] == pytest.approx(float((np.abs(z) > 3.0).mean()))
    assert by_name["f_0"]["max_abs_z"] < 6.0
    # Report is purely diagnostic: nothing is dropped or flagged for removal.
    assert {r["name"] for r in report if r["scope"] == "symbol"} == set(
        shifted.feature_names
    )
    assert all("drop" not in r for r in report)
    assert any(r["scope"] == "global" for r in report)


def test_invalid_arguments_fail_closed() -> None:
    market = signal_market(60)
    with pytest.raises(ValueError, match="clip"):
        fit_train_feature_transform(
            market, train_start_index=0, train_stop_index=30, clip=0.0
        )
    with pytest.raises(ValueError, match="train"):
        fit_train_feature_transform(
            market, train_start_index=30, train_stop_index=30, clip=3.0
        )
    with pytest.raises(ValueError, match="train"):
        fit_train_feature_transform(
            market, train_start_index=0, train_stop_index=60, clip=3.0
        )


def test_historical_normalizer_excludes_records_arriving_after_its_cutoff() -> None:
    market = signal_market(170)
    delayed = market.available_at.copy()
    delayed[80, 0] = market.timestamps[140]
    f = market.features.copy()
    f[80, 0, 1] = 1_000_000.0
    information_available = market.resolved_array("information_available").copy()
    information_available[80, 0] = False
    source = replace(
        market,
        features=f,
        available_at=delayed,
        information_available=information_available,
        identity_payload_json=None,
    )
    as_of = fit_train_feature_transform(
        source,
        train_start_index=10,
        train_stop_index=100,
        as_of_index=120,
        clip=3.0,
    )
    without_late_record = source.features.copy()
    without_late_record[80, 0, 1] = -1_000_000.0
    changed = replace(source, features=without_late_record, identity_payload_json=None)
    again = fit_train_feature_transform(
        changed,
        train_start_index=10,
        train_stop_index=100,
        as_of_index=120,
        clip=3.0,
    )
    assert np.array_equal(as_of.feature_mean, again.feature_mean)
    assert np.array_equal(as_of.feature_scale, again.feature_scale)
    assert as_of.digest == again.digest
    without_boundary = fit_train_feature_transform(
        source, train_start_index=10, train_stop_index=100, clip=3.0
    )
    assert without_boundary.feature_mean[0, 1] != as_of.feature_mean[0, 1]
    with pytest.raises(ValueError, match="as_of_index"):
        fit_train_feature_transform(
            source,
            train_start_index=10,
            train_stop_index=100,
            as_of_index=100,
            clip=3.0,
        )


def test_fit_cutoff_excludes_late_training_record_from_statistics() -> None:
    """A record published after fit cutoff cannot alter its normalizer."""
    market = signal_market(420, seed=13)
    delayed = market.available_at.copy()
    delayed[21, 0] = market.timestamps[360]
    info = market.resolved_array("information_available").copy()
    info[21, 0] = False
    left = market.features.copy()
    right = market.features.copy()
    left[21, 0, 1], right[21, 0, 1] = 1_000_000.0, -1_000_000.0
    a = replace(
        market,
        features=left,
        available_at=delayed,
        information_available=info,
        identity_payload_json=None,
    )
    b = replace(
        market,
        features=right,
        available_at=delayed,
        information_available=info,
        identity_payload_json=None,
    )
    fit_a = fit_train_feature_transform(
        a, train_start_index=20, train_stop_index=240, clip=3.0, as_of_index=330
    )
    fit_b = fit_train_feature_transform(
        b, train_start_index=20, train_stop_index=240, clip=3.0, as_of_index=330
    )
    assert fit_a.digest == fit_b.digest
    assert np.array_equal(fit_a.feature_mean, fit_b.feature_mean)


def test_block_pipeline_passes_fit_cutoff_to_normalizer(monkeypatch) -> None:
    """The real runner must use the as-of code path, not the unbounded default."""
    from trade_rl.evaluation import decision_value_dev_study as study

    market = signal_market(500, seed=37)
    proto = study.default_2024q1_decision_value_dev_protocol()
    block = replace(
        proto.blocks[0],
        train_start_index=24,
        train_stop_index=300,
        fit_cutoff_index=321,
        eval_start_index=322,
        eval_stop_index=450,
    )
    real_fit = study.fit_train_feature_transform
    called = []

    def fit_guard(*args, **kwargs):
        called.append(kwargs.get("as_of_index"))
        assert kwargs.get("as_of_index") == block.fit_cutoff_index
        return real_fit(*args, **kwargs)

    class StopHere(Exception):
        pass

    with monkeypatch.context() as patch:
        patch.setattr(study, "fit_train_feature_transform", fit_guard)
        patch.setattr(
            study, "fit_ridge_forecast", lambda *a, **k: (_ for _ in ()).throw(StopHere)
        )
        with pytest.raises(StopHere):
            study._train_block(market, proto, block, 0.0008)
    assert called == [block.fit_cutoff_index]
