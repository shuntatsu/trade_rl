"""
combine_weight_fns（複数WeightFnの固定比率合成）のテスト
"""

import numpy as np

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.learning.baselines import combine_weight_fns


def _fs(days=60, alpha="cross", seed=1):
    src = SyntheticSource(n_days=days, alpha=alpha, seed=seed)
    return FeaturePipeline(src.symbols).build(src)


class TestCombine:
    def test_single_component_weight_one_is_identity(self):
        fs = _fs()

        def fn_a(fs, t, prev):
            w = np.zeros(fs.n_symbols)
            w[0] = 0.5
            return w

        combo = combine_weight_fns([(1.0, fn_a)])
        w = combo(fs, 10, np.zeros(fs.n_symbols))
        np.testing.assert_allclose(w, fn_a(fs, 10, np.zeros(fs.n_symbols)))

    def test_weights_are_linearly_combined(self):
        fs = _fs()

        def fn_a(fs, t, prev):
            w = np.zeros(fs.n_symbols)
            w[0] = 1.0
            return w

        def fn_b(fs, t, prev):
            w = np.zeros(fs.n_symbols)
            w[1] = 1.0
            return w

        combo = combine_weight_fns([(0.75, fn_a), (0.25, fn_b)])
        w = combo(fs, 10, np.zeros(fs.n_symbols))
        assert abs(w[0] - 0.75) < 1e-9
        assert abs(w[1] - 0.25) < 1e-9

    def test_gross_over_one_is_projected(self):
        fs = _fs()

        def fn_a(fs, t, prev):
            w = np.zeros(fs.n_symbols)
            w[0], w[1] = 1.0, -1.0
            return w

        combo = combine_weight_fns([(1.0, fn_a), (1.0, fn_a)])  # gross=4
        w = combo(fs, 10, np.zeros(fs.n_symbols))
        assert float(np.abs(w).sum()) <= 1.0 + 1e-9

    def test_prev_is_passed_through_unmodified(self):
        fs = _fs()
        seen_prev = []

        def fn_a(fs, t, prev):
            seen_prev.append(prev.copy())
            return np.zeros(fs.n_symbols)

        combo = combine_weight_fns([(1.0, fn_a)])
        prev = np.array([0.1] + [0.0] * (fs.n_symbols - 1))
        combo(fs, 10, prev)
        np.testing.assert_allclose(seen_prev[0], prev)
