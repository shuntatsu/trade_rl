"""
リスクオーバーレイ（mars_lite.trading.risk_overlay）のテスト

RuleRiskOverlayが既存のインライン実装（post_processor.pyの既定パス）と
数値的に同一であることをゴールデンテストで保証する。
"""

import numpy as np
import pytest

from mars_lite.trading.post_processor import PortfolioPostProcessor, PostProcessConfig
from mars_lite.trading.risk_overlay import RuleRiskOverlay, RuleRiskOverlayConfig


def _cfg_pair(target_vol=0.5, dd_derisk_start=0.10, dd_derisk_floor=0.3,
              disagreement_penalty=1.0, vol_lookback=48):
    pp_cfg = PostProcessConfig(
        ema_alpha=0.5, max_weight=0.4, no_trade_band=0.04,
        target_vol=target_vol, vol_lookback=vol_lookback,
        dd_derisk_start=dd_derisk_start, dd_derisk_floor=dd_derisk_floor,
        disagreement_penalty=disagreement_penalty,
    )
    overlay_cfg = RuleRiskOverlayConfig(
        target_vol=target_vol, vol_lookback=vol_lookback,
        dd_derisk_start=dd_derisk_start, dd_derisk_floor=dd_derisk_floor,
        disagreement_penalty=disagreement_penalty,
    )
    return pp_cfg, overlay_cfg


class TestRuleRiskOverlayParity:
    def test_matches_legacy_inline_no_triggers(self):
        """ボラ/DD/不一致いずれも発火しない通常ケース"""
        pp_cfg, overlay_cfg = _cfg_pair()
        legacy = PortfolioPostProcessor(pp_cfg)
        via_overlay = PortfolioPostProcessor(pp_cfg, risk_overlay=RuleRiskOverlay(overlay_cfg))

        rng = np.random.default_rng(0)
        raw = rng.uniform(-0.3, 0.3, 7)
        prev = rng.uniform(-0.1, 0.1, 7)
        recent = rng.normal(0, 0.001, (48, 7))

        w1, info1 = legacy.process(raw.copy(), prev.copy(), recent_returns=recent.copy())
        w2, info2 = via_overlay.process(raw.copy(), prev.copy(), recent_returns=recent.copy())

        np.testing.assert_allclose(w1, w2, atol=1e-12)
        assert info1.vol_scale == pytest.approx(info2.vol_scale)
        assert info1.dd_scale == pytest.approx(info2.dd_scale)
        assert info1.disagreement_scale == pytest.approx(info2.disagreement_scale)
        assert info1.est_port_vol == pytest.approx(info2.est_port_vol)

    def test_matches_legacy_inline_vol_target_triggers(self):
        """大きなリターンでボラターゲティングが発火するケース"""
        pp_cfg, overlay_cfg = _cfg_pair(target_vol=0.1)
        legacy = PortfolioPostProcessor(pp_cfg)
        via_overlay = PortfolioPostProcessor(pp_cfg, risk_overlay=RuleRiskOverlay(overlay_cfg))

        rng = np.random.default_rng(1)
        raw = np.full(7, 0.5) * rng.choice([-1, 1], 7)
        prev = np.zeros(7)
        recent = rng.normal(0, 0.05, (48, 7))  # 大きめのボラでtarget_vol超過を誘発

        w1, info1 = legacy.process(raw.copy(), prev.copy(), recent_returns=recent.copy())
        w2, info2 = via_overlay.process(raw.copy(), prev.copy(), recent_returns=recent.copy())

        assert info1.vol_scale < 1.0  # 発火の確認（テストの前提）
        np.testing.assert_allclose(w1, w2, atol=1e-12)
        assert info1.vol_scale == pytest.approx(info2.vol_scale)

    def test_matches_legacy_inline_dd_derisk_triggers(self):
        """DD閾値超過でデリスクが発火するケース"""
        pp_cfg, overlay_cfg = _cfg_pair()
        legacy = PortfolioPostProcessor(pp_cfg)
        via_overlay = PortfolioPostProcessor(pp_cfg, risk_overlay=RuleRiskOverlay(overlay_cfg))

        raw = np.full(7, 0.3)
        prev = np.zeros(7)

        w1, info1 = legacy.process(raw.copy(), prev.copy(), drawdown=0.25)
        w2, info2 = via_overlay.process(raw.copy(), prev.copy(), drawdown=0.25)

        assert info1.dd_scale < 1.0
        np.testing.assert_allclose(w1, w2, atol=1e-12)
        assert info1.dd_scale == pytest.approx(info2.dd_scale)

    def test_matches_legacy_inline_disagreement_triggers(self):
        """不一致度縮小が発火するケース"""
        pp_cfg, overlay_cfg = _cfg_pair()
        legacy = PortfolioPostProcessor(pp_cfg)
        via_overlay = PortfolioPostProcessor(pp_cfg, risk_overlay=RuleRiskOverlay(overlay_cfg))

        raw = np.full(7, 0.3)
        prev = np.zeros(7)

        w1, info1 = legacy.process(raw.copy(), prev.copy(), disagreement=0.4)
        w2, info2 = via_overlay.process(raw.copy(), prev.copy(), disagreement=0.4)

        assert info1.disagreement_scale < 1.0
        np.testing.assert_allclose(w1, w2, atol=1e-12)
        assert info1.disagreement_scale == pytest.approx(info2.disagreement_scale)

    def test_gross_never_increases(self):
        """オーバーレイはグロスを増やさない（不変条件）"""
        overlay = RuleRiskOverlay(RuleRiskOverlayConfig(target_vol=0.1, dd_derisk_start=0.05))
        rng = np.random.default_rng(2)
        w = rng.uniform(-0.5, 0.5, 7)
        gross_before = float(np.abs(w).sum())
        recent = rng.normal(0, 0.05, (48, 7))

        scaled, _ = overlay.scale(w, drawdown=0.3, disagreement=0.5, recent_returns=recent)
        gross_after = float(np.abs(scaled).sum())
        assert gross_after <= gross_before + 1e-12


class TestRLRiskOverlay:
    def test_scale_respects_gross_multiplier_bounds(self):
        from mars_lite.trading.risk_overlay import RLRiskOverlay

        class _FixedAgent:
            def predict(self, obs, deterministic=True):
                return np.array([0.3]), None

        overlay = RLRiskOverlay(agent=_FixedAgent(), target_vol=0.5)
        w = np.array([0.4, -0.3, 0.2, 0.0, -0.1, 0.3, -0.2])
        scaled, info = overlay.scale(w, drawdown=0.1, disagreement=0.0, recent_returns=None)

        np.testing.assert_allclose(scaled, w * 0.3, atol=1e-12)
        # 単一のグロス乗数が④⑤⑥全てを代替するため3項目とも同じ値になる
        assert info["vol_scale"] == pytest.approx(0.3)
        assert info["dd_scale"] == pytest.approx(0.3)
        assert info["disagreement_scale"] == pytest.approx(0.3)

    def test_scale_clips_out_of_range_actions(self):
        from mars_lite.trading.risk_overlay import RLRiskOverlay

        class _ExtremeAgent:
            def predict(self, obs, deterministic=True):
                return np.array([5.0]), None  # 範囲外（policyの出力異常を想定）

        overlay = RLRiskOverlay(agent=_ExtremeAgent())
        w = np.array([0.5, -0.5])
        scaled, _ = overlay.scale(w, drawdown=0.0, disagreement=0.0, recent_returns=None)
        # gross_multiplierは[0,1]にクリップされ、グロスを増やさない
        assert np.abs(scaled).sum() <= np.abs(w).sum() + 1e-12
