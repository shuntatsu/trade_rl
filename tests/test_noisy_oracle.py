"""
ノイズ入りオラクル（現実的な天井）のテスト
"""

import numpy as np
import pytest

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.features.signal_check import _rank_ic
from mars_lite.learning.baselines import (
    calibrate_noise_to_ic, noisy_oracle_strategy, oracle_dp_strategy,
    simulate_strategy, flat_strategy,
)


def _synthetic_fs(alpha="cross", days=40, seed=3):
    src = SyntheticSource(n_days=days, alpha=alpha, seed=seed)
    return FeaturePipeline(src.symbols).build(src)


class TestCalibrateNoise:

    def test_calibrated_ic_close_to_target(self):
        rng = np.random.default_rng(0)
        fwd = rng.normal(0, 0.01, size=(500, 5))
        for target in [0.05, 0.2, 0.5]:
            sigma = calibrate_noise_to_ic(fwd, target, seed=1)
            noisy = fwd + np.random.default_rng(2).normal(0, sigma, size=fwd.shape)
            ic = _rank_ic(noisy.flatten(), fwd.flatten())
            assert abs(ic - target) < 0.08

    def test_zero_target_gives_large_noise(self):
        rng = np.random.default_rng(0)
        fwd = rng.normal(0, 0.01, size=(200, 3))
        sigma = calibrate_noise_to_ic(fwd, 0.0, seed=1)
        assert sigma > np.std(fwd) * 10


class TestNoisyOracle:

    def test_monotonic_in_target_ic(self):
        """目標ICが高いほど（真の未来知識に近いほど）収益が上がる"""
        fs = _synthetic_fs(alpha="cross", days=60)
        r_low = noisy_oracle_strategy(fs, target_ic=0.02, seed=0, n_draws=2)
        r_mid = noisy_oracle_strategy(fs, target_ic=0.2, seed=0, n_draws=2)
        perfect = oracle_dp_strategy(fs)
        assert r_low.total_return <= r_mid.total_return + 1e-6
        assert r_mid.total_return <= perfect.total_return + 1e-6

    def test_low_ic_far_below_perfect_oracle(self):
        """IC≈0の劣化オラクルは完全オラクルに遠く及ばない

        （DPは自分の劣化シグナルを「真実」として扱うため、低ICでも
        取引自体は行い、コストで実際にはマイナスになりうる。ここでは
        「フラットと同等」ではなく「完全オラクルとは大差」を検証する）
        """
        fs = _synthetic_fs(alpha="cross", days=60, seed=9)
        r_noisy = noisy_oracle_strategy(fs, target_ic=0.01, seed=0, n_draws=2)
        perfect = oracle_dp_strategy(fs)
        assert r_noisy.total_return < perfect.total_return * 0.1

    def test_returns_strategy_result_with_valid_equity(self):
        fs = _synthetic_fs(alpha="cross", days=30)
        r = noisy_oracle_strategy(fs, target_ic=0.1, seed=1, n_draws=2)
        assert np.isfinite(r.equity_curve).all()
        assert r.equity_curve[0] == pytest.approx(1.0)
        assert r.name == "oracle_ic0.10"


class TestDecisionEveryOracle:

    def test_lower_frequency_reduces_turnover(self):
        """decision_everyを上げると回転が減る（同一signal・同一コスト）"""
        fs = _synthetic_fs(alpha="cross", days=60, seed=4)
        r1 = noisy_oracle_strategy(fs, target_ic=0.05, seed=0, n_draws=2, decision_every=1)
        r8 = noisy_oracle_strategy(fs, target_ic=0.05, seed=0, n_draws=2, decision_every=8)
        assert r8.turnover_total < r1.turnover_total

    def test_low_frequency_can_reduce_cost_drag(self):
        """低ICで発生する過剰な回転コストは、意思決定頻度を下げると緩和される"""
        fs = _synthetic_fs(alpha="cross", days=60, seed=9)
        r1 = noisy_oracle_strategy(fs, target_ic=0.01, seed=0, n_draws=2, decision_every=1)
        r8 = noisy_oracle_strategy(fs, target_ic=0.01, seed=0, n_draws=2, decision_every=8)
        assert r8.total_return >= r1.total_return - 1e-6

    def test_default_decision_every_matches_legacy_behavior(self):
        """decision_every省略時は従来と同じ挙動（回帰防止）"""
        fs = _synthetic_fs(alpha="cross", days=30, seed=2)
        r_default = noisy_oracle_strategy(fs, target_ic=0.1, seed=1, n_draws=2)
        r_explicit = noisy_oracle_strategy(fs, target_ic=0.1, seed=1, n_draws=2, decision_every=1)
        np.testing.assert_allclose(r_default.equity_curve, r_explicit.equity_curve)
