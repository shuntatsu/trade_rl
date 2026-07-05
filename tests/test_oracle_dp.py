"""
手数料込みDPオラクル則のテスト（山と谷を取る最適経路）
"""

import numpy as np
import pytest

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.learning.baselines import (
    oracle_dp_strategy, run_all_baselines, simulate_strategy,
    equal_weight_strategy,
)


def _synthetic_fs(alpha="cross", days=40, seed=3):
    src = SyntheticSource(n_days=days, alpha=alpha, seed=seed)
    return FeaturePipeline(src.symbols).build(src)


class TestOracleDP:

    def test_oracle_is_upper_bound(self):
        """オラクルは同一コストで全ベースラインを上回る（理論上限）"""
        fs = _synthetic_fs(alpha="cross")
        results = run_all_baselines(fs)
        oracle = results["oracle_dp"]
        for name, r in results.items():
            if name == "oracle_dp":
                continue
            assert oracle.total_return >= r.total_return - 1e-9, \
                f"oracle {oracle.total_return:.3f} < {name} {r.total_return:.3f}"

    def test_oracle_positive_on_any_moving_market(self):
        """価格が動けば（山谷があれば）オラクルは黒字"""
        fs = _synthetic_fs(alpha="none", seed=1)
        oracle = oracle_dp_strategy(fs)
        assert oracle.total_return > 0.0

    def test_high_fee_suppresses_trading(self):
        """手数料を極端に上げると回転が減る（微小な山谷は取らない）"""
        fs = _synthetic_fs(alpha="cross")
        lo = oracle_dp_strategy(fs, fee_rate=0.0001, spread_rate=0.0, impact_rate=0.0)
        hi = oracle_dp_strategy(fs, fee_rate=0.05, spread_rate=0.0, impact_rate=0.0)
        assert hi.turnover_total < lo.turnover_total
        # 高手数料でも損はしない（フラットが常に選択肢）
        assert hi.total_return >= -1e-9

    def test_captures_peaks_and_valleys(self):
        """明確な上げ→下げの1銘柄で、B&Hより高いリターン（谷で買い山で売る）"""
        # 1銘柄・鋸波状の価格を直接構成
        from mars_lite.features.feature_pipeline import FeatureSet
        n = 400
        t = np.arange(n)
        price = 100.0 * (1.0 + 0.2 * np.sin(2 * np.pi * t / 50))  # 山谷が明確
        close = price.reshape(-1, 1)

        fs = FeatureSet(
            symbols=["X"], timestamps=t.astype("datetime64[s]"),
            features=np.zeros((n, 1, 1), dtype=np.float32),
            global_features=np.zeros((n, 1), dtype=np.float32),
            close=close, open_next=close.copy(),
            funding_rate=np.zeros((n, 1), dtype=np.float32),
            feature_names=["dummy"], global_feature_names=["g"],
        )
        oracle = oracle_dp_strategy(fs, fee_rate=0.0005, spread_rate=0.0, impact_rate=0.0)
        bh = simulate_strategy(fs, equal_weight_strategy, fee_rate=0.0005,
                               spread_rate=0.0, impact_rate=0.0, min_trade_delta=0.0)
        # 山谷を取るので横ばい資産のB&H（~0%）を大きく上回る
        assert oracle.total_return > 0.5
        assert oracle.total_return > bh.total_return

    def test_no_short_variant(self):
        """allow_short=Falseではロング/フラットのみ（下げ相場でショートしない）"""
        fs = _synthetic_fs(alpha="cross")
        long_only = oracle_dp_strategy(fs, allow_short=False)
        both = oracle_dp_strategy(fs, allow_short=True)
        # ショート可の方が上限は高い（下げも取れる）
        assert both.total_return >= long_only.total_return - 1e-9
