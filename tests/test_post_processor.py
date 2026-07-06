"""
後処理器のテスト
"""

import numpy as np
import pytest

from mars_lite.trading.post_processor import (
    PortfolioPostProcessor,
    PostProcessConfig,
    make_default_processor,
    make_legacy_processor,
)


class TestPostProcessor:
    def test_leverage_respected(self):
        pp = PortfolioPostProcessor(PostProcessConfig(ema_alpha=1.0, no_trade_band=0.0))
        raw = np.array([0.5, 0.5, 0.5, -0.5])
        w, _ = pp.process(raw, np.zeros(4))
        assert np.abs(w).sum() <= 1.0 + 1e-9

    def test_concentration_cap(self):
        pp = PortfolioPostProcessor(
            PostProcessConfig(ema_alpha=1.0, max_weight=0.3, no_trade_band=0.0)
        )
        raw = np.array([1.0, 0.0, 0.0, 0.0])
        w, _ = pp.process(raw, np.zeros(4))
        assert np.abs(w).max() <= 0.3 + 1e-9

    def test_no_trade_band(self):
        pp = PortfolioPostProcessor(
            PostProcessConfig(ema_alpha=1.0, no_trade_band=0.1, max_weight=1.0)
        )
        prev = np.array([0.2, 0.2, 0.0, 0.0])
        raw = np.array([0.22, 0.2, 0.05, 0.0])  # 変化は0.02,0,0.05 いずれも<0.1
        w, _ = pp.process(raw, prev)
        np.testing.assert_allclose(w, prev)  # すべて据え置き

    def test_ema_smoothing(self):
        pp = PortfolioPostProcessor(
            PostProcessConfig(ema_alpha=0.5, no_trade_band=0.0, max_weight=1.0)
        )
        prev = np.array([0.0, 0.0])
        raw = np.array([0.8, 0.0])
        w, _ = pp.process(raw, prev)
        # 0.5*0.8 + 0.5*0 = 0.4（射影後も0.4のまま）
        assert w[0] == pytest.approx(0.4, abs=1e-6)

    def test_vol_targeting_reduces_gross(self):
        cfg = PostProcessConfig(
            ema_alpha=1.0, no_trade_band=0.0, max_weight=1.0, target_vol=0.10
        )  # 低い目標
        pp = PortfolioPostProcessor(cfg)
        raw = np.array([0.5, 0.5])
        # 高ボラの直近リターン
        rng = np.random.default_rng(0)
        recent = rng.normal(0, 0.05, (48, 2))
        w, info = pp.process(raw, np.zeros(2), recent_returns=recent)
        assert info.vol_scale < 1.0
        assert np.abs(w).sum() < np.abs(raw).sum()

    def test_drawdown_derisk(self):
        cfg = PostProcessConfig(
            ema_alpha=1.0,
            no_trade_band=0.0,
            max_weight=1.0,
            dd_derisk_start=0.1,
            dd_derisk_floor=0.3,
        )
        pp = PortfolioPostProcessor(cfg)
        raw = np.array([0.5, 0.5])
        w_normal, _ = pp.process(raw, np.zeros(2), drawdown=0.0)
        w_dd, info = pp.process(raw, np.zeros(2), drawdown=0.5)
        assert info.dd_scale < 1.0
        assert np.abs(w_dd).sum() < np.abs(w_normal).sum()

    def test_disagreement_scaling(self):
        cfg = PostProcessConfig(
            ema_alpha=1.0, no_trade_band=0.0, max_weight=1.0, disagreement_penalty=1.0
        )
        pp = PortfolioPostProcessor(cfg)
        raw = np.array([0.5, 0.5])
        w, info = pp.process(raw, np.zeros(2), disagreement=0.5)
        assert info.disagreement_scale == pytest.approx(0.5)
        assert np.abs(w).sum() == pytest.approx(0.5, abs=1e-6)

    def test_legacy_equivalence(self):
        """legacy処理は射影＋バンドのみ（平滑/上限/ボラ無効）"""
        pp = make_legacy_processor(min_trade_delta=0.02)
        raw = np.array([0.6, 0.4, 0.0])
        w, _ = pp.process(raw, np.zeros(3))
        # 射影のみ（gross=1.0なのでそのまま）、集中上限は効かない
        np.testing.assert_allclose(w, raw, atol=1e-9)

    def test_default_config_roundtrip(self):
        """to_dict → PostProcessConfig(**d) で復元できる（serving一致に必要）"""
        pp = make_default_processor()
        d = pp.cfg.to_dict()
        pp2 = PortfolioPostProcessor(PostProcessConfig(**d))
        assert pp2.cfg.ema_alpha == pp.cfg.ema_alpha
        assert pp2.cfg.target_vol == pp.cfg.target_vol
