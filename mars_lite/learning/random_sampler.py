"""
ランダムエピソードサンプラー

訓練データからランダムにエピソード開始位置をサンプリングする。
train/val/testモードに応じて適切な範囲からサンプリング。
"""

from typing import List, Literal, Optional

import numpy as np
import pandas as pd


class RandomEpisodeSampler:
    """
    ランダムエピソードサンプラー

    指定されたデータ範囲からランダムにエピソード開始位置をサンプリング。
    """

    def __init__(
        self,
        data: pd.DataFrame,
        episode_length: int,
        lookback: int = 10,
        seed: Optional[int] = None,
    ):
        """
        Args:
            data: サンプリング対象のDataFrame
            episode_length: エピソード長（ステップ数）
            lookback: ルックバック期間（開始位置の下限に使用）
            seed: 乱数シード
        """
        self.data = data
        self.episode_length = episode_length
        self.lookback = lookback

        self._rng = np.random.default_rng(seed)

        # 有効範囲を計算
        self.min_start = lookback
        self.max_start = len(data) - episode_length - 1

        if self.max_start <= self.min_start:
            raise ValueError(
                f"データ長が不足しています。"
                f"必要: {lookback + episode_length + 1}, 実際: {len(data)}"
            )

    def sample_start_idx(self) -> int:
        """
        ランダムにエピソード開始位置をサンプリング

        Returns:
            開始インデックス
        """
        return int(self._rng.integers(self.min_start, self.max_start + 1))

    def sample_batch(self, n: int) -> List[int]:
        """
        複数のエピソード開始位置をサンプリング

        Args:
            n: サンプル数

        Returns:
            開始インデックスのリスト
        """
        return [self.sample_start_idx() for _ in range(n)]

    def reset_seed(self, seed: int):
        """乱数シードをリセット"""
        self._rng = np.random.default_rng(seed)

    @property
    def n_valid_starts(self) -> int:
        """有効な開始位置の数"""
        return max(0, self.max_start - self.min_start + 1)


class MultiModeEpisodeSampler:
    """
    train/val/testモード別エピソードサンプラー

    各モードに応じたデータ範囲からサンプリング。
    """

    def __init__(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        episode_length: int,
        lookback: int = 10,
        seed: Optional[int] = None,
    ):
        """
        Args:
            train_data: 訓練データ
            val_data: 検証データ
            test_data: テストデータ
            episode_length: エピソード長
            lookback: ルックバック期間
            seed: 乱数シード
        """
        self.episode_length = episode_length
        self.lookback = lookback

        self._samplers = {}
        self._data = {}

        for mode, data in [
            ("train", train_data),
            ("val", val_data),
            ("test", test_data),
        ]:
            if len(data) >= lookback + episode_length + 1:
                self._data[mode] = data
                self._samplers[mode] = RandomEpisodeSampler(
                    data=data,
                    episode_length=episode_length,
                    lookback=lookback,
                    seed=seed,
                )
            else:
                # データ不足の場合はNone
                self._data[mode] = data
                self._samplers[mode] = None

    def sample_start_idx(self, mode: Literal["train", "val", "test"] = "train") -> int:
        """
        指定モードのデータからランダムにサンプリング

        Args:
            mode: サンプリングモード

        Returns:
            開始インデックス
        """
        sampler = self._samplers.get(mode)
        if sampler is None:
            raise ValueError(f"{mode}データが不足しています")
        return sampler.sample_start_idx()

    def get_data(self, mode: Literal["train", "val", "test"]) -> pd.DataFrame:
        """指定モードのデータを取得"""
        return self._data.get(mode)

    def is_mode_available(self, mode: Literal["train", "val", "test"]) -> bool:
        """指定モードが利用可能か"""
        return self._samplers.get(mode) is not None


class SequentialEpisodeSampler:
    """
    シーケンシャルエピソードサンプラー

    評価時に使用。データを順番に走査。
    """

    def __init__(
        self,
        data: pd.DataFrame,
        episode_length: int,
        lookback: int = 10,
        stride: int = None,
    ):
        """
        Args:
            data: サンプリング対象のDataFrame
            episode_length: エピソード長
            lookback: ルックバック期間
            stride: エピソード間のストライド（Noneならepisode_length）
        """
        self.data = data
        self.episode_length = episode_length
        self.lookback = lookback
        self.stride = stride if stride is not None else episode_length

        # 有効範囲
        self.min_start = lookback
        self.max_start = len(data) - episode_length - 1

        # 全開始位置を計算
        self._start_indices = list(
            range(self.min_start, self.max_start + 1, self.stride)
        )
        self._current_idx = 0

    def sample_start_idx(self) -> int:
        """次のエピソード開始位置を返す"""
        if self._current_idx >= len(self._start_indices):
            self.reset()

        idx = self._start_indices[self._current_idx]
        self._current_idx += 1
        return idx

    def reset(self):
        """イテレーションをリセット"""
        self._current_idx = 0

    @property
    def n_episodes(self) -> int:
        """総エピソード数"""
        return len(self._start_indices)

    def __iter__(self):
        self.reset()
        return self

    def __next__(self) -> int:
        if self._current_idx >= len(self._start_indices):
            raise StopIteration
        return self.sample_start_idx()

    def __len__(self) -> int:
        return len(self._start_indices)
