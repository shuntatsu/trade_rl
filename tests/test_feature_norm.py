"""入力分布正規化（ローリング・ガウスランク）のテスト"""

import numpy as np

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import (
    FeaturePipeline,
    _gaussian_rank_transform,
)
from mars_lite.utils.metrics import _norm_ppf, norm_ppf_array


class TestNormPpfArray:
    def test_matches_scalar(self):
        """ベクトル化逆正規CDFがスカラー版と一致"""
        ps = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        v = norm_ppf_array(ps)
        s = np.array([_norm_ppf(p) for p in ps])
        assert np.max(np.abs(v - s)) < 1e-9

    def test_clipped_at_extremes(self):
        """端点(0/1)近傍でも±8にクリップされ無限大にならない"""
        v = norm_ppf_array(np.array([1e-12, 1.0 - 1e-12]))
        assert np.all(np.isfinite(v))
        assert np.all(np.abs(v) <= 8.0)


class TestGaussianRankTransform:
    def _fs(self, days=60):
        src = SyntheticSource(n_days=days, alpha="cross", seed=0)
        return FeaturePipeline(src.symbols).build(src)

    def test_shape_preserved(self):
        fs = self._fs()
        fsn = fs.gaussian_rank_normalized()
        assert fsn.features.shape == fs.features.shape

    def test_marginal_is_standard_normal(self):
        """変換後の各チャネルはウォームアップ後おおむね N(0,1)"""
        fs = self._fs()
        fsn = fs.gaussian_rank_normalized()
        col = fsn.features[100:, 0, 0]
        assert abs(float(col.mean())) < 0.2
        assert 0.7 < float(col.std()) < 1.3

    def test_causal_no_future_leak(self):
        """先頭プレフィックスのみで正規化しても、全系列正規化の同区間と一致
        （trailing窓のみ参照＝未来リーク無しの証明）"""
        fs = self._fs()
        fsn_full = fs.gaussian_rank_normalized()
        fsn_pre = fs.slice(0, 400).gaussian_rank_normalized()
        assert np.allclose(fsn_full.features[:400], fsn_pre.features, atol=1e-9)

    def test_constant_channel_maps_to_zero(self):
        """定数/ゼロ埋めチャネルは変換後も0のまま（分散ゼロで縮退）"""
        feats = np.zeros((300, 3, 2))
        feats[:, :, 0] = np.random.default_rng(0).normal(size=(300, 3))  # 変動あり
        # channel 1 は全ゼロ（定数）
        out = _gaussian_rank_transform(feats, window=100, min_periods=20)
        assert np.allclose(out[:, :, 1], 0.0)
