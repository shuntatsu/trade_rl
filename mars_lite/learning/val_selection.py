"""
検証ベースモデル選択モジュール

学習中に検証スライス（OOS）で定期評価し、最良時点のパラメータを保持する。
小さいデータセットにPPOを長時間回すと訓練軌道に過学習するため、
「検証成績が最良のスナップショット」を最終モデルとして採用する。
（P0健全性試験で実際に過学習による失敗を観測したことへの対策）
"""

import io
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.trading.post_processor import BARS_PER_YEAR_1H


def quick_evaluate(agent, fs: FeatureSet, **env_kwargs) -> float:
    """検証スライス全体を1エピソード決定的に走らせ、リスク調整後スコアを返す"""
    from mars_lite.env.portfolio_env import PortfolioTradingEnv

    # 検証はスライス全体を1エピソードで走査。学習側の episode_bars /
    # regime_start_pool（レジーム専門家用）はここでは無視する。
    env_kwargs = {
        k: v
        for k, v in env_kwargs.items()
        if k not in ("episode_bars", "regime_start_pool")
    }
    env = PortfolioTradingEnv(fs, episode_bars=fs.n_bars - 2, **env_kwargs)
    obs, _ = env.reset(options={"start_idx": 0})
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc

    return sortino_score(getattr(env, "_returns_history", []))


def sortino_score(returns_history) -> float:
    """検証スライスの1バーリターン系列 → Sortino比（下方偏差調整の年率リターン）。

    旧指標 total_return − 0.5·max_dd は、弱αでリターンがノイズに埋もれる実データ
    では max_dd 項が支配し「取引しない（flat）」を常に最良と判定して PBT/val選択を
    保守崩壊させていた（実データ3本すべての遠因）。Sortinoは
      - flat（無取引・全ゼロ）→ 0
      - 弱くても正のリスク調整エッジ → 正（＝flatより高評価。ここが是正点）
      - 損失を出す方策 → 負（正しく棄却）
    と振る舞い、下方偏差でリスクも見るため崩壊しない。
    """
    rets = np.asarray(returns_history, dtype=np.float64)
    if rets.size < 5 or not np.any(np.abs(rets) > 1e-12):
        return 0.0
    mean = float(rets.mean())
    downside = np.minimum(rets, 0.0)
    dstd = float(np.sqrt(np.mean(downside**2)))  # 目標0の下方偏差
    sortino = mean / (dstd + 1e-9) * np.sqrt(BARS_PER_YEAR_1H)
    return float(np.clip(sortino, -50.0, 50.0))


class ValSelectionCallback(BaseCallback):
    """
    定期的に検証スライスで評価し、最良パラメータを記憶するコールバック

    学習終了後に restore_best(agent) を呼ぶと最良時点に巻き戻す。
    """

    def __init__(
        self,
        val_fs: FeatureSet,
        eval_freq: int = 20_000,
        env_kwargs: Optional[dict] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.val_fs = val_fs
        self.eval_freq = eval_freq
        self.env_kwargs = env_kwargs or {}
        self.best_score = -np.inf
        self.best_params: Optional[bytes] = None
        self.history = []

    def _evaluate_and_maybe_save(self) -> None:
        score = quick_evaluate(self.model, self.val_fs, **self.env_kwargs)
        self.history.append({"step": self.num_timesteps, "val_score": score})
        if self.verbose >= 1:
            print(
                f"[ValSelection] step={self.num_timesteps:,} val_score={score:+.4f} "
                f"(best={self.best_score:+.4f})"
            )
        if score > self.best_score:
            self.best_score = score
            buf = io.BytesIO()
            self.model.save(buf)
            self.best_params = buf.getvalue()

    def _on_training_start(self) -> None:
        # 初期方策（mean≈0 = ほぼ無取引）も選択候補に含める。
        # 予測力のないデータではこれが最良となり「取引しない」が選ばれる。
        self._evaluate_and_maybe_save()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq < self.training_env.num_envs:
            self._evaluate_and_maybe_save()
        return True

    def _on_training_end(self) -> None:
        # 最終時点も候補に含める
        self._evaluate_and_maybe_save()

    def restore_best(self, agent):
        """最良スナップショットのパラメータをagentに書き戻す"""
        if self.best_params is None:
            return agent
        from stable_baselines3 import PPO

        buf = io.BytesIO(self.best_params)
        best = PPO.load(buf, device=agent.device)
        agent.set_parameters(best.get_parameters())
        return agent
