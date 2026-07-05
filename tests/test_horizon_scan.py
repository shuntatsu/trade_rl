"""
ホライズンスキャンのテスト
"""

import numpy as np
import pytest

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.features.horizon_scan import (
    run_horizon_scan, default_feature_groups, DEFAULT_HORIZONS,
)


@pytest.fixture(scope="module")
def fs_alpha():
    src = SyntheticSource(n_days=60, alpha="cross", seed=5)
    return FeaturePipeline(src.symbols).build(src)


@pytest.fixture(scope="module")
def fs_noise():
    src = SyntheticSource(n_days=60, alpha="none", seed=6)
    return FeaturePipeline(src.symbols).build(src)


class TestHorizonScan:

    def test_default_feature_groups_cover_all_features(self, fs_alpha):
        groups = default_feature_groups(fs_alpha)
        covered = sorted(n for names in groups.values() for n in names)
        assert covered == sorted(fs_alpha.feature_names)

    def test_scan_alpha_data_finds_positive_horizon(self, fs_alpha):
        report = run_horizon_scan(fs_alpha, horizons=(1, 4, 24), n_folds=4)
        assert report.best_horizon in (1, 4, 24)
        best = max(r.mean_oos_ic for r in report.results)
        assert best > 0.05
        assert "derivatives" in report.results[0].group_ic

    def test_scan_noise_data_stays_low(self, fs_noise):
        report = run_horizon_scan(fs_noise, horizons=(1, 4, 24), n_folds=4)
        for r in report.results:
            assert abs(r.mean_oos_ic) < 0.15

    def test_summary_and_to_dict_run(self, fs_alpha):
        report = run_horizon_scan(fs_alpha, horizons=(4, 24), n_folds=4)
        assert "Best horizon" in report.summary()
        d = report.to_dict()
        assert d["best_horizon"] in (4, 24)
        assert len(d["results"]) == 2


class TestDecisionEvery:

    def test_decision_every_holds_between_decisions(self, fs_alpha):
        from mars_lite.env.portfolio_env import PortfolioTradingEnv
        env = PortfolioTradingEnv(fs_alpha, episode_bars=20, decision_every=4)
        env.reset(seed=0, options={"start_idx": 0})
        rng = np.random.default_rng(1)
        for i in range(8):
            action = rng.uniform(-1, 1, env.n_symbols)
            _, _, term, trunc, info = env.step(action)
            if i % 4 != 0:
                assert info["turnover"] == 0.0
            if term or trunc:
                break
