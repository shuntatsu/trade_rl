from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace


def test_residual_walk_forward_config_is_immutable_and_records_effective_values() -> (
    None
):
    module_path = Path("mars_lite/pipeline/residual_wf_config.py")
    assert module_path.is_file()
    module = importlib.import_module("mars_lite.pipeline.residual_wf_config")
    config_type = module.ResidualWalkForwardConfig

    args = SimpleNamespace(
        decision_every=1,
        scan_horizons=True,
        horizon=12,
        ensemble=1,
        n_seeds=3,
        folds=3,
        purge_bars=8,
        base_timeframe="4h",
        run_tier="research",
        signal_model="gbm",
        fee_profile="taker",
        git_sha="abc123",
    )
    original = vars(args).copy()

    config = config_type.from_args(args, dataset_identity="dataset-sha")

    assert vars(args) == original
    assert config.requested_decision_every == 1
    assert config.effective_decision_every == 6
    assert config.requested_ensemble_size == 1
    assert config.effective_ensemble_size == 3
    assert config.requested_folds == 3
    assert config.effective_purge_bars == 24
    assert config.base_timeframe == "4h"
    assert config.bars_per_year == 2_190
    assert config.horizon == 12
    assert config.dataset_identity == "dataset-sha"
    assert config.git_sha == "abc123"
    assert config.to_dict()["effective_decision_every"] == 6
