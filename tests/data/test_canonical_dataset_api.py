from __future__ import annotations

import trade_rl.data.artifacts as legacy_artifacts


def test_legacy_dataset_module_does_not_export_ambiguous_writer() -> None:
    assert not hasattr(legacy_artifacts, "write_market_dataset_artifact")
    assert "write_market_dataset_artifact" not in legacy_artifacts.__all__
