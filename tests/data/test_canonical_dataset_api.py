from __future__ import annotations

import trade_rl.data as data
from trade_rl.data.artifacts import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
    publish_market_dataset_artifact,
    write_market_dataset_files,
)


def test_canonical_dataset_api_uses_data_package_exports() -> None:
    assert data.inspect_published_market_dataset_artifact is (
        inspect_published_market_dataset_artifact
    )
    assert data.load_market_dataset_artifact is load_market_dataset_artifact
    assert data.publish_market_dataset_artifact is publish_market_dataset_artifact
    assert data.write_market_dataset_files is write_market_dataset_files


def test_deprecated_direct_dataset_writer_is_not_exported() -> None:
    assert not hasattr(data, "write_market_dataset_artifact")
    assert "write_market_dataset_artifact" not in data.__all__
