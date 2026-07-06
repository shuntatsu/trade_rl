"""
階層マルチタイムフレーム（HTF）ゲートのテスト（項目5）
"""

import numpy as np
import pytest

from mars_lite.data.sources import SyntheticSource
from mars_lite.env.portfolio_env import PortfolioTradingEnv
from mars_lite.features.feature_pipeline import FeaturePipeline


@pytest.fixture(scope="module")
def fs():
    src = SyntheticSource(n_days=40, alpha="cross", seed=7)
    return FeaturePipeline(src.symbols).build(src)


class TestHTFGate:
    def test_gate_disabled_by_default(self, fs):
        env = PortfolioTradingEnv(fs)
        assert env._htf_idx is None
        w = np.array([0.5, -0.5] + [0.0] * (fs.n_symbols - 2))
        np.testing.assert_array_equal(env.apply_htf_gate(w), w)

    def test_gate_resolves_feature_index(self, fs):
        env = PortfolioTradingEnv(fs, htf_gate=True)
        assert env._htf_idx == fs.feature_names.index("4h_ret_z20")

    def test_gate_blocks_counter_trend(self, fs):
        env = PortfolioTradingEnv(
            fs, htf_gate=True, htf_threshold=0.3, htf_neutral_scale=0.5
        )
        env.reset(options={"start_idx": 100})
        idx = env._htf_idx
        # HTFトレンドを人工的に設定
        htf = np.zeros(fs.n_symbols)
        htf[0] = 1.0  # 上昇 → ショート禁止
        htf[1] = -1.0  # 下降 → ロング禁止
        # neutral（0）はスケール縮小
        env.fs.features[env.t][:, idx] = htf.astype(np.float32)

        w = np.array([-0.4, 0.4] + [0.2] * (fs.n_symbols - 2))
        gated = env.apply_htf_gate(w)
        assert gated[0] == 0.0  # 上昇HTFでショート → 0
        assert gated[1] == 0.0  # 下降HTFでロング → 0
        assert np.allclose(gated[2:], 0.2 * 0.5)  # neutral縮小

    def test_gate_preserves_aligned_sizing(self, fs):
        """HTF方向と整合する成分は大きさをそのまま通す（サイジングは1hに委ねる）"""
        env = PortfolioTradingEnv(fs, htf_gate=True, htf_threshold=0.3)
        env.reset(options={"start_idx": 100})
        idx = env._htf_idx
        htf = np.zeros(fs.n_symbols)
        htf[0] = 1.0  # 上昇
        htf[1] = -1.0  # 下降
        env.fs.features[env.t][:, idx] = htf.astype(np.float32)

        w = np.array([0.6, -0.3] + [0.0] * (fs.n_symbols - 2))
        gated = env.apply_htf_gate(w)
        assert gated[0] == 0.6  # 上昇HTFでロング → 維持
        assert gated[1] == -0.3  # 下降HTFでショート → 維持

    def test_gate_never_increases_gross(self, fs):
        env = PortfolioTradingEnv(fs, htf_gate=True)
        env.reset(options={"start_idx": 50})
        rng = np.random.default_rng(0)
        for _ in range(20):
            w = env.project_weights(rng.uniform(-1, 1, fs.n_symbols))
            gated = env.apply_htf_gate(w)
            assert np.abs(gated).sum() <= np.abs(w).sum() + 1e-9

    def test_full_step_runs_with_gate(self, fs):
        env = PortfolioTradingEnv(fs, htf_gate=True)
        obs, _ = env.reset(options={"start_idx": 0})
        for _ in range(30):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            assert np.isfinite(r)
            if term or trunc:
                break
