"""
ガードレール・品質ゲート・リーク自己検査のテスト
"""

import numpy as np

from mars_lite.data.quality import check_symbol, run_quality_gate
from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.features.signal_check import run_leak_self_test
from mars_lite.trading.guardrails import (
    GuardrailState,
    apply_guardrails,
    evaluate_guardrails,
)


class TestGuardrails:
    def _w(self):
        return np.array([0.3, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0])

    def test_stale_data_flattens(self):
        r = evaluate_guardrails(self._w(), 1.0, 0.5, data_age_hours=5.0)
        assert r.action == "flatten"
        assert apply_guardrails(self._w(), r).sum() == 0.0

    def test_nan_features_flatten(self):
        feats = np.array([1.0, np.nan, 2.0])
        r = evaluate_guardrails(self._w(), 1.0, 0.5, data_age_hours=0.5, features=feats)
        assert r.action == "flatten"

    def test_allzero_features_flatten(self):
        feats = np.zeros(10)
        r = evaluate_guardrails(self._w(), 1.0, 0.5, data_age_hours=0.5, features=feats)
        assert r.action == "flatten"

    def test_daily_loss_flattens(self):
        st = GuardrailState(day_start_value=1.0, peak_value=1.0)
        r = evaluate_guardrails(self._w(), 0.90, 0.5, data_age_hours=0.5, state=st)
        assert r.action == "flatten"

    def test_drawdown_flattens(self):
        st = GuardrailState(day_start_value=1.5, peak_value=1.5)
        r = evaluate_guardrails(self._w(), 1.0, 0.5, data_age_hours=0.5, state=st)
        assert r.action == "flatten"  # DD = 1 - 1.0/1.5 = 33% > 20%

    def test_consecutive_losses_scales(self):
        st = GuardrailState(day_start_value=1.0, peak_value=1.0, consecutive_losses=20)
        r = evaluate_guardrails(self._w(), 1.0, 0.1, data_age_hours=0.5, state=st)
        assert r.action == "scale"
        assert r.scale == 0.5

    def test_turnover_anomaly_scales(self):
        st = GuardrailState(turnover_mean=0.2, turnover_std=0.1)
        r = evaluate_guardrails(
            self._w(), 1.0, turnover=1.0, data_age_hours=0.5, state=st
        )
        assert r.action == "scale"

    def test_proceed_when_healthy(self):
        st = GuardrailState(turnover_mean=0.3, turnover_std=0.1)
        r = evaluate_guardrails(
            self._w(), 1.0, turnover=0.3, data_age_hours=0.5, state=st
        )
        assert r.action == "proceed"
        np.testing.assert_array_equal(apply_guardrails(self._w(), r), self._w())


class TestQualityGate:
    def test_clean_synthetic_passes(self):
        src = SyntheticSource(n_days=20, alpha="cross", seed=2)
        rep = run_quality_gate(src, src.symbols, base_timeframe="1h")
        assert len(rep.passing_symbols) == len(src.symbols)

    def test_too_few_bars_fails(self):
        src = SyntheticSource(n_days=3, alpha="none", seed=2)
        q = check_symbol(src, src.symbols[0], "1h", min_bars=200)
        assert not q.passed


class TestLeakSelfTest:
    def test_detector_healthy(self):
        src = SyntheticSource(n_days=30, alpha="cross", seed=3)
        fs = FeaturePipeline(src.symbols).build(src)
        lt = run_leak_self_test(fs)
        assert lt["shuffle_ic"] < 0.05  # シャッフルで相関消失
        assert lt["future_shift_ic"] > lt["base_ic"]  # 未来シフトでIC増大
        assert lt["healthy"]
