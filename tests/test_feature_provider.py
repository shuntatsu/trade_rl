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


def _feature_set():
    return SimpleNamespace(
        n_bars=2,
        symbols=["BTCUSDT"],
        timestamps=np.asarray(
            ["2026-07-11T00:00:00", "2026-07-11T04:00:00"],
            dtype="datetime64[ns]",
        ),
        features=np.asarray([[[0.1]], [[0.2]]], dtype=np.float64),
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
    monkeypatch.setattr(sources, "create_source", lambda *args, **kwargs: sentinel_source)
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
