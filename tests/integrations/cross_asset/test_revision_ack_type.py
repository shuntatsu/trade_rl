"""Explicit revision-hindsight acknowledgement must not coerce truthy inputs."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integrations.cross_asset.support import seed_store
from tests.integrations.cross_asset.test_dataset_overlay import _dataset
from trade_rl.integrations.cross_asset.dataset import with_cross_asset_global_features
from trade_rl.integrations.cross_asset.loader import load_cross_asset_features


@pytest.mark.parametrize("invalid_ack", ["false", "true", 0, 1, None, [], {}])
def test_revision_hindsight_ack_requires_exact_boolean(
    tmp_path: Path, invalid_ack: object
) -> None:
    store = seed_store(tmp_path)
    source = _dataset()
    frame = load_cross_asset_features(
        store.root, source.timestamps, opt_in_series=("SP500",)
    )
    with pytest.raises(ValueError, match="accept_revision_hindsight.*bool"):
        with_cross_asset_global_features(
            source, frame, feature_names=("SP500__ret1",),
            accept_revision_hindsight=invalid_ack,  # type: ignore[arg-type]
        )
    assert source.identity_verified
