"""
リスクオーバーレイ（ボラターゲティング・DDデリスク・不一致縮小の交換可能な実装）

PortfolioPostProcessor.process() の④⑤⑥段（グロス量を決めるリスク制御）を
差し替え可能にする抽象。既定は現行ロジックを忠実に抽出した RuleRiskOverlay
（post_processorに risk_overlay を渡さない場合の既存インライン実装と
数値的に同一、tests/test_risk_overlay.py::test_rule_overlay_matches_legacy_inline
で保証）。

RLRiskOverlay は「配分（銘柄間の相対ウェイト）は既存の配分エージェントに
任せ、グロスのスケール（どれだけリスクを取るか）だけを学習するRL」という
オプトインの第2エージェント。docs/ARCHITECTURE.md §2.8と同じ理由（原則1）
で、既定はRuleRiskOverlayのまま。RLRiskOverlayはP0＋汎用性スイートで
ルール比の優位性を示すまで昇格しない。
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import numpy as np

BARS_PER_YEAR_1H = 24 * 365


class RiskOverlay(Protocol):
    """方策の提案ウェイトのグロスをリスク状況に応じてスケールする"""

    def scale(
        self,
        w: np.ndarray,
        drawdown: float,
        disagreement: float,
        recent_returns: Optional[np.ndarray],
    ) -> "tuple[np.ndarray, Dict[str, Any]]":
        """グロスを増やさないスケーリングを適用し、(scaled_w, info) を返す"""
        ...

    def reset(self) -> None:
        """エピソード境界での内部状態リセット（ルールベースは通常no-op）"""
        ...


@dataclass
class RuleRiskOverlayConfig:
    target_vol: Optional[float] = None
    vol_lookback: int = 48
    dd_derisk_start: float = 0.10
    dd_derisk_floor: float = 0.3
    disagreement_penalty: float = 1.0
    bars_per_year: int = BARS_PER_YEAR_1H


class RuleRiskOverlay:
    """PortfolioPostProcessorの④⑤⑥（ボラ目標・DDデリスク・不一致縮小）を
    抽出した既定実装。post_processor.pyのインライン実装と数値的に同一。
    """

    def __init__(self, config: Optional[RuleRiskOverlayConfig] = None):
        self.cfg = config or RuleRiskOverlayConfig()

    def reset(self) -> None:
        pass

    def scale(self, w, drawdown, disagreement, recent_returns):
        cfg = self.cfg
        w = np.asarray(w, dtype=np.float64).copy()
        info: Dict[str, Any] = {
            "vol_scale": 1.0,
            "dd_scale": 1.0,
            "disagreement_scale": 1.0,
            "est_port_vol": 0.0,
        }

        # ④ ボラターゲティング
        if (
            cfg.target_vol is not None
            and recent_returns is not None
            and len(recent_returns) >= 5
        ):
            port_ret = recent_returns @ w
            est_vol = float(np.std(port_ret) * np.sqrt(cfg.bars_per_year))
            info["est_port_vol"] = est_vol
            if est_vol > cfg.target_vol and est_vol > 1e-9:
                vol_scale = cfg.target_vol / est_vol
                info["vol_scale"] = vol_scale
                w = w * vol_scale

        # ⑤ ドローダウン応答
        if drawdown > cfg.dd_derisk_start:
            over = (drawdown - cfg.dd_derisk_start) / max(
                1.0 - cfg.dd_derisk_start, 1e-9
            )
            dd_scale = float(max(cfg.dd_derisk_floor, 1.0 - over))
            info["dd_scale"] = dd_scale
            w = w * dd_scale

        # ⑥ アンサンブル不一致縮小
        if disagreement > 0:
            disagreement_scale = float(
                max(0.0, 1.0 - cfg.disagreement_penalty * disagreement)
            )
            info["disagreement_scale"] = disagreement_scale
            w = w * disagreement_scale

        return w, info


class RLRiskOverlay:
    """学習済みPPOでグロス乗数(0〜1)を決めるリスクオーバーレイ（opt-in）

    観測は [gross_before, drawdown, disagreement, est_vol_ratio,
    recent_return_mean, recent_return_std] の6次元固定ベクトル。
    learning/overlay_trainer.train_risk_overlay で学習した agent を渡す。
    """

    def __init__(
        self,
        agent,
        target_vol: Optional[float] = 0.5,
        bars_per_year: int = BARS_PER_YEAR_1H,
    ):
        self.agent = agent
        self.target_vol = target_vol
        self.bars_per_year = bars_per_year

    def reset(self) -> None:
        pass

    def build_obs(self, w, drawdown, disagreement, recent_returns) -> np.ndarray:
        gross = float(np.abs(w).sum())
        est_vol = 0.0
        if recent_returns is not None and len(recent_returns) >= 5:
            port_ret = recent_returns @ w
            est_vol = float(np.std(port_ret) * np.sqrt(self.bars_per_year))
        vol_ratio = est_vol / self.target_vol if self.target_vol else 0.0
        ret_mean = (
            float(recent_returns.mean())
            if recent_returns is not None and len(recent_returns)
            else 0.0
        )
        ret_std = (
            float(recent_returns.std())
            if recent_returns is not None and len(recent_returns)
            else 0.0
        )
        return np.array(
            [gross, drawdown, disagreement, vol_ratio, ret_mean, ret_std],
            dtype=np.float32,
        )

    def scale(self, w, drawdown, disagreement, recent_returns):
        obs = self.build_obs(w, drawdown, disagreement, recent_returns)
        action, _ = self.agent.predict(obs, deterministic=True)
        gross_mult = float(np.clip(np.asarray(action).flatten()[0], 0.0, 1.0))
        scaled = np.asarray(w, dtype=np.float64) * gross_mult
        # 単一の学習済みスカラーが④⑤⑥(ボラ目標/DDデリスク/不一致縮小)を
        # まとめて代替するため、どの要因がどれだけ効いたかは分解できない。
        # dd_scaleだけに詰め込むとPortfolioTradingEnv.obs_risk_state経由で
        # 方策に「ドローダウン応答だけが動いた」という偽の信号を与えてしまう
        # ため、3つとも同じ値にして「オーバーレイの合成スケール」として扱う。
        info = {
            "vol_scale": gross_mult,
            "dd_scale": gross_mult,
            "disagreement_scale": gross_mult,
            "est_port_vol": float(obs[3] * (self.target_vol or 0.0)),
        }
        return scaled, info
