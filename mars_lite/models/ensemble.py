from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from stable_baselines3 import PPO


class EnsemblePredictor:
    """
    複数モデルによるアンサンブル推論クラス。

    機能:
    1. 複数のPPOモデルをロード
    2. 入力に対して全モデルの予測を実行
    3. 結果を統合（平均、多数決など）して返す
    """

    def __init__(self, model_paths: List[Union[str, Path]], device: str = "auto"):
        """
        Args:
            model_paths: ロードするモデルファイルのパスリスト
            device: 推論デバイス ("cpu", "cuda", "auto")
        """
        self.models = []
        self.model_names = []
        self.device = device

        self._load_models(model_paths)

    def _load_models(self, paths: List[Union[str, Path]]) -> None:
        """モデルを一括ロード"""
        print(f"[Ensemble] Loading {len(paths)} models...")
        for p in paths:
            path_obj = Path(p)
            if not path_obj.exists():
                print(f"[Ensemble] Warning: Model not found at {p}. Skipping.")
                continue

            try:
                model = PPO.load(path_obj, device=self.device)
                self.models.append(model)
                self.model_names.append(path_obj.stem)
                print(f"[Ensemble] Loaded: {path_obj.stem}")
            except Exception as e:
                print(f"[Ensemble] Error loading {p}: {e}")

        if not self.models:
            raise RuntimeError("No models loaded successfully for Ensemble.")
        print(f"[Ensemble] Successfully loaded {len(self.models)} models.")

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
        method: str = "mean",
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        アンサンブル予測を実行

        Args:
            method: 統合方法 ("mean", "vote")

        Returns:
            (aggregated_action, states)
            states is currently None (recurrent support requires detailed handling)
        """
        actions = []

        # 各モデルで推論
        for model in self.models:
            # PPO.predict returns (action, state)
            action, _ = model.predict(
                observation,
                state=None,  # Tuple state handling for ensemble is complex, ignore for now
                episode_start=episode_start,
                deterministic=deterministic,
            )
            actions.append(action)

        # (models, batch_size, action_dim)
        stacked_actions = np.stack(actions, axis=0)

        final_action = None

        if method == "mean":
            # 単純平均
            final_action = np.mean(stacked_actions, axis=0)

        elif method == "vote":
            # 多数決（ドテン判定）
            # 正負（Long/Short）の方向で多数決を取り、採用された方向の平均値を使う
            # または単純に符号の合計を見る

            # (models, batch, dim) -> sign -> mean
            signs = np.sign(stacked_actions)
            avg_sign = np.mean(signs, axis=0)  # -1.0 ~ 1.0

            # しきい値: 例えば絶対値が0.5以上（過半数以上が同じ方向）なら採用
            # そうでなければ 0 (No Action)
            mask = np.abs(avg_sign) >= 0.5

            # 平均アクション（大きさ用）
            mean_act = np.mean(stacked_actions, axis=0)

            # マスク適用
            final_action = mean_act * mask.astype(np.float32)

        elif method == "median":
            final_action = np.median(stacked_actions, axis=0)

        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        return final_action, None

    def get_individual_predictions(self, observation: np.ndarray) -> Dict[str, float]:
        """デバッグ用: 個別モデルの予測値を取得（バッチサイズ1前提）"""
        results = {}
        for name, model in zip(self.model_names, self.models):
            action, _ = model.predict(observation, deterministic=True)
            results[name] = float(action[0])
        return results
