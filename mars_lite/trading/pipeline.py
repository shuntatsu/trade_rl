"""
方策の生ウェイトから執行ウェイトを導く決定パイプライン。

学習時とServing時が同じproposal constraint / post-processing経路を通る。
HTFは現在保有ではなく今回のdesired proposalへ適用し、その後にEMAや
no-trade bandなどのstateful処理を行う。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from mars_lite.trading.htf_constraint import HTFProposalConstraint
from mars_lite.trading.post_processor import _project_leverage, make_legacy_processor


@dataclass
class PortfolioState:
    """パイプライン呼び出し時点のポートフォリオ状態"""

    weights: np.ndarray
    portfolio_value: float = 1.0
    peak_value: float = 1.0
    disagreement: float = 0.0

    @property
    def drawdown(self) -> float:
        return 1.0 - self.portfolio_value / max(self.peak_value, 1e-9)


@dataclass
class MarketView:
    """パイプラインが必要とする市場データの切り出し"""

    recent_returns: Optional[np.ndarray] = None
    htf_trend: Optional[np.ndarray] = None

    @classmethod
    def from_feature_set(
        cls, fs, t: int, vol_lookback: int = 0, htf_idx: Optional[int] = None
    ) -> "MarketView":
        recent_returns = None
        if vol_lookback > 0:
            start = max(0, t - vol_lookback)
            if t > start:
                recent_returns = (
                    np.diff(fs.close[start : t + 1], axis=0) / fs.close[start:t, :]
                )
        htf_trend = None
        if htf_idx is not None:
            htf_trend = fs.features[t][:, htf_idx]
        return cls(recent_returns=recent_returns, htf_trend=htf_trend)


class DecisionPipeline:
    """Proposal constraint -> stateful post-process の共有実装。"""

    def __init__(
        self,
        post_processor=None,
        min_trade_delta: float = 0.04,
        htf_threshold: float = 0.3,
        htf_neutral_scale: float = 0.5,
        max_leverage: float = 1.0,
    ):
        self.post_processor = post_processor or make_legacy_processor(min_trade_delta)
        self.min_trade_delta = min_trade_delta
        self.htf_threshold = htf_threshold
        self.htf_neutral_scale = htf_neutral_scale
        self.max_leverage = max_leverage
        self.htf_constraint = HTFProposalConstraint(
            threshold=htf_threshold, neutral_scale=htf_neutral_scale
        )

    def project(self, raw_action: np.ndarray) -> np.ndarray:
        return _project_leverage(
            np.asarray(raw_action, dtype=np.float64).flatten(), self.max_leverage
        )

    def process_proposal(
        self, proposal: np.ndarray, state: PortfolioState, market: MarketView
    ):
        """Apply HTF to desired proposal, then run stateful post-processing."""

        prev = np.asarray(state.weights, dtype=np.float64)
        desired = np.asarray(proposal, dtype=np.float64)
        htf_result = None
        if market.htf_trend is not None:
            htf_result = self.htf_constraint.apply(desired, market.htf_trend)
            desired = htf_result.weights

        target, pp_info = self.post_processor.process(
            desired,
            prev,
            recent_returns=market.recent_returns,
            drawdown=state.drawdown,
            disagreement=state.disagreement,
        )
        if htf_result is not None:
            pp_info.extra["htf_zeroed_fraction"] = htf_result.zeroed_fraction
            pp_info.extra["htf_neutral_scaled_fraction"] = (
                htf_result.neutral_scaled_fraction
            )
            pp_info.extra["htf_constrained_weights"] = desired.copy()
        return target, pp_info

    def target_weights(
        self, proj: np.ndarray, state: PortfolioState, market: MarketView
    ):
        """Compatibility wrapper for existing direct-weight callers."""

        return self.process_proposal(proj, state, market)

    def apply_htf_gate(self, w: np.ndarray, htf_trend: np.ndarray) -> np.ndarray:
        """Compatibility helper; new code should call process_proposal."""

        return self.htf_constraint.apply(w, htf_trend).weights
