from types import SimpleNamespace

import numpy as np
import pytest

from mars_lite.serving.feature_provider import CsvFeatureProvider


class _Runtime:
    def __init__(self, bundle):
        self.bundle = bundle

    def active_bundle(self):
        return self.bundle


def _bundle(base_timeframe: str = "4h"):
    return SimpleNamespace(
        bundle_digest="digest-v1",
        metadata={
            "symbols": ["BTCUSDT"],
            "run_config": {"base_timeframe": base_timeframe},
            "post_processor": {"vol_lookback": 1},
        },
        preprocessing={"rank_window": 2},
    )


def _feature_set(feature_value: float = 0.2):
    return SimpleNamespace(
        n_bars=2,
        symbols=["BTCUSDT"],
        timestamps=np.asarray(
            ["2026-07-11T00:00:00", "2026-07-11T04:00:00"],
            dtype="datetime64[ns]",
        ),
        features=np.asarray([[[0.1]], [[feature_value]]], dtype=np.float64),
        global_features=np.asarray([[0.0], [0.1]], dtype=np.float64),
        close=np.asarray([[100.0], [101.0]], dtype=np.float64),
        feature_names=["ret"],
        global_feature_names=["hour_sin"],
    )


def test_feature_provider_uses_bundled_base_timeframe(tmp_path, monkeypatch) -> None:
    import mars_lite.data.sources as sources
    import mars_lite.features.feature_pipeline as feature_pipeline

    captured = {}

    class FakePipeline:
        def __init__(self, symbols, base_timeframe="1h"):
            captured["symbols"] = list(symbols)
            captured["base_timeframe"] = base_timeframe

        def build(self, source):
            captured["source"] = source
            return _feature_set()

    sentinel_source = object()
    monkeypatch.setattr(
        sources, "create_source", lambda *args, **kwargs: sentinel_source
    )
    monkeypatch.setattr(feature_pipeline, "FeaturePipeline", FakePipeline)

    provider = CsvFeatureProvider(
        runtime=_Runtime(_bundle("4h")),
        data_dir=tmp_path,
        cache_ttl_seconds=0,
    )
    snapshot = provider.get_snapshot()

    assert captured["symbols"] == ["BTCUSDT"]
    assert captured["base_timeframe"] == "4h"
    assert captured["source"] is sentinel_source
    assert snapshot.symbols == ("BTCUSDT",)


def test_feature_provider_rejects_unknown_bundled_timeframe(tmp_path) -> None:
    provider = CsvFeatureProvider(
        runtime=_Runtime(_bundle("2h")),
        data_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="unsupported bundled base_timeframe"):
        provider.get_snapshot()


def test_feature_provider_snapshot_id_changes_when_content_changes(
    tmp_path, monkeypatch
) -> None:
    import mars_lite.data.sources as sources
    import mars_lite.features.feature_pipeline as feature_pipeline

    values = iter([0.2, 0.3])

    class FakePipeline:
        def __init__(self, symbols, base_timeframe="1h"):
            pass

        def build(self, source):
            return _feature_set(next(values))

    monkeypatch.setattr(sources, "create_source", lambda *args, **kwargs: object())
    monkeypatch.setattr(feature_pipeline, "FeaturePipeline", FakePipeline)

    first = CsvFeatureProvider(
        runtime=_Runtime(_bundle("4h")),
        data_dir=tmp_path,
        cache_ttl_seconds=0,
    ).get_snapshot()
    second = CsvFeatureProvider(
        runtime=_Runtime(_bundle("4h")),
        data_dir=tmp_path,
        cache_ttl_seconds=0,
    ).get_snapshot()

    assert first.snapshot_id != second.snapshot_id


def test_feature_provider_excludes_incomplete_latest_4h_bar(
    tmp_path, monkeypatch
) -> None:
    import mars_lite.data.sources as sources
    import mars_lite.features.feature_pipeline as feature_pipeline

    monkeypatch.setattr(sources, "create_source", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        feature_pipeline,
        "FeaturePipeline",
        lambda *args, **kwargs: SimpleNamespace(build=lambda source: _feature_set()),
    )
    provider = CsvFeatureProvider(
        runtime=_Runtime(_bundle("4h")),
        data_dir=tmp_path,
        cache_ttl_seconds=0,
        clock=lambda: np.datetime64("2026-07-11T07:00:00", "ns"),
    )

    snapshot = provider.get_snapshot()

    assert snapshot.feature_history.shape[0] == 1
    assert snapshot.feature_history[-1, 0, 0] == pytest.approx(0.1)
    assert snapshot.data_age_hours == pytest.approx(3.0)


def test_feature_provider_ages_daily_data_from_bar_close(tmp_path, monkeypatch) -> None:
    import mars_lite.data.sources as sources
    import mars_lite.features.feature_pipeline as feature_pipeline

    feature_set = _feature_set()
    feature_set.timestamps = np.asarray(
        ["2026-07-10T00:00:00", "2026-07-11T00:00:00"],
        dtype="datetime64[ns]",
    )
    monkeypatch.setattr(sources, "create_source", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        feature_pipeline,
        "FeaturePipeline",
        lambda *args, **kwargs: SimpleNamespace(build=lambda source: feature_set),
    )
    provider = CsvFeatureProvider(
        runtime=_Runtime(_bundle("1d")),
        data_dir=tmp_path,
        cache_ttl_seconds=0,
        clock=lambda: np.datetime64("2026-07-12T12:00:00", "ns"),
    )

    snapshot = provider.get_snapshot()

    assert snapshot.data_age_hours == pytest.approx(12.0)


def test_feature_provider_rejects_when_no_bar_is_complete(
    tmp_path, monkeypatch
) -> None:
    import mars_lite.data.sources as sources
    import mars_lite.features.feature_pipeline as feature_pipeline

    feature_set = _feature_set()
    feature_set.timestamps = np.asarray(
        ["2026-07-12T08:00:00", "2026-07-12T09:00:00"],
        dtype="datetime64[ns]",
    )
    monkeypatch.setattr(sources, "create_source", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        feature_pipeline,
        "FeaturePipeline",
        lambda *args, **kwargs: SimpleNamespace(build=lambda source: feature_set),
    )
    provider = CsvFeatureProvider(
        runtime=_Runtime(_bundle("1h")),
        data_dir=tmp_path,
        cache_ttl_seconds=0,
        clock=lambda: np.datetime64("2026-07-12T08:30:00", "ns"),
    )

    with pytest.raises(ValueError, match="no completed bar"):
        provider.get_snapshot()


def test_incomplete_bar_mutation_does_not_change_snapshot_id(
    tmp_path, monkeypatch
) -> None:
    import mars_lite.data.sources as sources
    import mars_lite.features.feature_pipeline as feature_pipeline

    values = iter([0.2, 0.9])

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            pass

        def build(self, source):
            return _feature_set(next(values))

    monkeypatch.setattr(sources, "create_source", lambda *args, **kwargs: object())
    monkeypatch.setattr(feature_pipeline, "FeaturePipeline", FakePipeline)
    kwargs = {
        "runtime": _Runtime(_bundle("4h")),
        "data_dir": tmp_path,
        "cache_ttl_seconds": 0,
        "clock": lambda: np.datetime64("2026-07-11T07:00:00", "ns"),
    }

    first = CsvFeatureProvider(**kwargs).get_snapshot()
    second = CsvFeatureProvider(**kwargs).get_snapshot()

    assert first.snapshot_id == second.snapshot_id
