"""
DPオラクル蒸留教師（clairvoyant teacher distillation）のテスト
"""

import numpy as np

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.learning.bc_warmstart import dp_oracle_teacher, generate_teacher_dataset


def _synthetic_fs(alpha="cross", days=40, seed=3, alpha_strength=0.002):
    src = SyntheticSource(
        n_days=days, alpha=alpha, alpha_strength=alpha_strength, seed=seed
    )
    return FeaturePipeline(src.symbols).build(src)


class TestDPOracleTeacher:
    def test_weights_gross_bounded(self):
        fs = _synthetic_fs(alpha="cross", days=30)
        teacher = dp_oracle_teacher(fs)
        prev = np.zeros(fs.n_symbols)
        for t in [0, 10, fs.n_bars - 2]:
            w = teacher(fs, t, prev)
            assert w.shape == (fs.n_symbols,)
            assert np.abs(w).sum() <= 1.0 + 1e-9

    def test_perfect_foresight_net_long_in_sustained_uptrend(self):
        """持続的な上昇相場では完全予知教師は平均してネットロングに寄る"""
        fs = _synthetic_fs(alpha="bull", alpha_strength=0.001, days=60, seed=1)
        teacher = dp_oracle_teacher(fs, allow_short=True)
        prev = np.zeros(fs.n_symbols)
        sums = []
        for t in range(0, fs.n_bars - 2, 5):
            w = teacher(fs, t, prev)
            sums.append(w.sum())
        assert np.mean(sums) > 0

    def test_noisy_variant_differs_from_perfect(self):
        fs = _synthetic_fs(alpha="cross", days=40, seed=2)
        perfect = dp_oracle_teacher(fs)
        noisy = dp_oracle_teacher(fs, noisy_ic=0.1, seed=5)
        prev = np.zeros(fs.n_symbols)
        diffs = 0
        for t in range(0, fs.n_bars - 2, 3):
            wp = perfect(fs, t, prev)
            wn = noisy(fs, t, prev)
            if not np.allclose(wp, wn):
                diffs += 1
        assert diffs > 0

    def test_generate_teacher_dataset_runs_with_oracle_teacher(self):
        fs = _synthetic_fs(alpha="cross", days=20, seed=4)
        teacher = dp_oracle_teacher(fs)
        X, A = generate_teacher_dataset(fs, teacher, {})
        assert X.shape[0] == A.shape[0]
        assert A.shape[1] == fs.n_symbols
        assert np.isfinite(X).all()
        assert np.abs(A).sum(axis=1).max() <= 1.0 + 1e-6
