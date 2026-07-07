"""
mars_lite.pipeline.phases の純粋関数（build_post_processor/build_env_kwargs）
のテスト。CLI引数からの後処理器・env構築引数の組み立てロジックを検証する
（学習を伴うphase_*本体の統合テストは別途・重いため対象外）。
"""

from types import SimpleNamespace

import pytest

from mars_lite.pipeline.phases import build_env_kwargs, build_post_processor
from mars_lite.trading.post_processor import PostProcessConfig


def _args(**overrides):
    defaults = dict(
        postproc="full",
        target_vol=0.5,
        beta_neutral=False,
        min_trade_delta=0.04,
        lambda_turnover=0.04,
        htf_gate=False,
        obs_risk_state=False,
        disagreement_dr=0.0,
        decision_every=1,
        scan_horizons=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestBuildPostProcessor:
    def test_default_horizon4_matches_make_default_processor(self):
        """horizon=4は後方互換値（ema_alpha=0.5, no_trade_band=0.04）と一致するはず"""
        pp = build_post_processor(_args(), horizon=4)
        assert pp.cfg.ema_alpha == pytest.approx(0.5)
        assert pp.cfg.no_trade_band == pytest.approx(0.04)
        assert pp.cfg.target_vol == pytest.approx(0.5)

    def test_legacy_mode_ignores_horizon_scaling(self):
        pp = build_post_processor(_args(postproc="legacy"), horizon=8)
        assert pp.cfg.target_vol is None
        assert pp.cfg.dd_derisk_start == 1.0  # make_legacy_processorの既定

    def test_larger_horizon_smooths_more_and_widens_band(self):
        """horizonが大きい(低頻度アルファ)ほどema_alphaは下がり(平滑強め)no_trade_bandは広がる"""
        pp4 = build_post_processor(_args(), horizon=4)
        pp8 = build_post_processor(_args(), horizon=8)
        assert pp8.cfg.ema_alpha < pp4.cfg.ema_alpha
        assert pp8.cfg.no_trade_band > pp4.cfg.no_trade_band

    def test_target_vol_zero_disables_vol_targeting(self):
        pp = build_post_processor(_args(target_vol=0.0), horizon=4)
        assert pp.cfg.target_vol is None

    def test_beta_neutral_flag_propagates(self):
        pp = build_post_processor(_args(beta_neutral=True), horizon=4)
        assert pp.cfg.beta_neutral is True


class TestBuildEnvKwargs:
    def test_minimal_kwargs_by_default(self):
        pp = PostProcessConfig()
        from mars_lite.trading.post_processor import PortfolioPostProcessor

        ekw = build_env_kwargs(_args(), PortfolioPostProcessor(pp), horizon=4)
        assert ekw["post_processor"] is not None
        assert ekw["min_trade_delta"] == 0.04
        assert ekw["lambda_turnover"] == 0.04
        assert "htf_gate" not in ekw
        assert "obs_risk_state" not in ekw
        assert "disagreement_dr_max" not in ekw
        assert "decision_every" not in ekw

    def test_htf_gate_flag_propagates(self):
        from mars_lite.trading.post_processor import make_default_processor

        ekw = build_env_kwargs(
            _args(htf_gate=True), make_default_processor(), horizon=4
        )
        assert ekw["htf_gate"] is True

    def test_obs_risk_state_flag_propagates(self):
        from mars_lite.trading.post_processor import make_default_processor

        ekw = build_env_kwargs(
            _args(obs_risk_state=True), make_default_processor(), horizon=4
        )
        assert ekw["obs_risk_state"] is True

    def test_disagreement_dr_propagates_only_when_positive(self):
        from mars_lite.trading.post_processor import make_default_processor

        pp = make_default_processor()
        ekw_off = build_env_kwargs(_args(disagreement_dr=0.0), pp, horizon=4)
        assert "disagreement_dr_max" not in ekw_off
        ekw_on = build_env_kwargs(_args(disagreement_dr=0.3), pp, horizon=4)
        assert ekw_on["disagreement_dr_max"] == 0.3

    def test_explicit_decision_every_wins_over_auto(self):
        from mars_lite.trading.post_processor import make_default_processor

        pp = make_default_processor()
        ekw = build_env_kwargs(
            _args(decision_every=3, scan_horizons=True),
            pp,
            horizon=8,
        )
        assert ekw["decision_every"] == 3

    def test_auto_decision_every_from_scan_horizons(self):
        from mars_lite.trading.post_processor import make_default_processor

        pp = make_default_processor()
        ekw = build_env_kwargs(
            _args(decision_every=1, scan_horizons=True),
            pp,
            horizon=8,
        )
        assert ekw["decision_every"] == 4  # max(1, horizon // 2)

    def test_no_auto_decision_every_without_scan_horizons(self):
        from mars_lite.trading.post_processor import make_default_processor

        pp = make_default_processor()
        ekw = build_env_kwargs(
            _args(decision_every=1, scan_horizons=False),
            pp,
            horizon=8,
        )
        assert "decision_every" not in ekw
