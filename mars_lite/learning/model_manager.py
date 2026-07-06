"""
モデルマネージャーモジュール

学習済みモデルの保存・読み込み・一覧管理。
メタデータ（学習パラメータ、成績、日時）も一緒に保存。
"""

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.base_class import BaseAlgorithm

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    PPO = None
    BaseAlgorithm = object

import gymnasium as gym


@dataclass
class ModelMetadata:
    """
    モデルメタデータ

    保存時に記録する情報。
    """

    # 基本情報
    model_id: str
    created_at: float
    updated_at: float

    # 学習パラメータ
    total_timesteps: int = 0
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    gamma: float = 0.99

    # 環境情報
    env_name: str = ""
    observation_shape: str = ""
    action_shape: str = ""

    # 成績
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_episode_length: float = 0.0
    n_episodes: int = 0

    # その他
    notes: str = ""
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """辞書から生成"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ModelManager:
    """
    モデル管理クラス

    モデルの保存・読み込み・一覧表示・削除を管理。
    """

    def __init__(self, models_dir: str = "./output/models"):
        """
        Args:
            models_dir: モデル保存ディレクトリ
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        agent: "BaseAlgorithm",
        model_id: str,
        metadata: Optional[ModelMetadata] = None,
        overwrite: bool = False,
    ) -> Path:
        """
        モデルを保存

        Args:
            agent: SB3エージェント
            model_id: モデルID（ファイル名）
            metadata: メタデータ（オプション）
            overwrite: 上書き許可

        Returns:
            保存先パス
        """
        if not HAS_SB3:
            raise ImportError("stable-baselines3 is required.")

        model_path = self.models_dir / f"{model_id}.zip"
        meta_path = self.models_dir / f"{model_id}.json"

        if model_path.exists() and not overwrite:
            raise FileExistsError(f"Model already exists: {model_path}")

        # モデル保存
        agent.save(str(model_path))

        # メタデータ保存
        if metadata is None:
            metadata = ModelMetadata(
                model_id=model_id,
                created_at=time.time(),
                updated_at=time.time(),
            )
        else:
            metadata.updated_at = time.time()

        # エージェントから追加情報を取得
        if hasattr(agent, "num_timesteps"):
            metadata.total_timesteps = agent.num_timesteps
        if hasattr(agent, "learning_rate"):
            lr = agent.learning_rate
            if callable(lr):
                try:
                    lr = lr(1.0)
                except:
                    lr = 0.0
            metadata.learning_rate = float(lr)
        if hasattr(agent, "n_steps"):
            metadata.n_steps = agent.n_steps
        if hasattr(agent, "batch_size"):
            metadata.batch_size = agent.batch_size
        if hasattr(agent, "gamma"):
            metadata.gamma = agent.gamma

        # 環境情報
        if hasattr(agent, "observation_space"):
            metadata.observation_shape = str(agent.observation_space.shape)
        if hasattr(agent, "action_space"):
            metadata.action_shape = str(agent.action_space.shape)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

        return model_path

    def load(
        self,
        model_id: str,
        env: Optional[gym.Env] = None,
    ) -> tuple:
        """
        モデルを読み込み

        Args:
            model_id: モデルID
            env: 環境（オプション）

        Returns:
            (agent, metadata)
        """
        if not HAS_SB3:
            raise ImportError("stable-baselines3 is required.")

        model_path = self.models_dir / f"{model_id}.zip"
        meta_path = self.models_dir / f"{model_id}.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # モデル読み込み
        agent = PPO.load(str(model_path), env=env)

        # メタデータ読み込み
        metadata = None
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = ModelMetadata.from_dict(json.load(f))

        return agent, metadata

    def list_models(self) -> List[Dict[str, Any]]:
        """
        モデル一覧を取得

        Returns:
            モデル情報のリスト
        """
        models = []

        for model_path in self.models_dir.glob("*.zip"):
            model_id = model_path.stem
            meta_path = model_path.with_suffix(".json")

            info = {
                "id": model_id,
                "path": str(model_path),
                "size_bytes": model_path.stat().st_size,
                "modified_at": model_path.stat().st_mtime,
            }

            # メタデータがあれば追加
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        info["metadata"] = json.load(f)
                except Exception:
                    pass

            models.append(info)

        # 更新日時でソート（新しい順）
        models.sort(key=lambda x: x["modified_at"], reverse=True)

        return models

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        モデル情報を取得

        Args:
            model_id: モデルID

        Returns:
            モデル情報（見つからなければNone）
        """
        model_path = self.models_dir / f"{model_id}.zip"

        if not model_path.exists():
            return None

        meta_path = model_path.with_suffix(".json")

        info = {
            "id": model_id,
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "modified_at": model_path.stat().st_mtime,
        }

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                info["metadata"] = json.load(f)

        return info

    def delete(self, model_id: str) -> bool:
        """
        モデルを削除

        Args:
            model_id: モデルID

        Returns:
            削除成功フラグ
        """
        model_path = self.models_dir / f"{model_id}.zip"
        meta_path = self.models_dir / f"{model_id}.json"

        if not model_path.exists():
            return False

        model_path.unlink()

        if meta_path.exists():
            meta_path.unlink()

        return True

    def copy(self, src_id: str, dst_id: str) -> Path:
        """
        モデルをコピー

        Args:
            src_id: コピー元ID
            dst_id: コピー先ID

        Returns:
            コピー先パス
        """
        src_model = self.models_dir / f"{src_id}.zip"
        src_meta = self.models_dir / f"{src_id}.json"
        dst_model = self.models_dir / f"{dst_id}.zip"
        dst_meta = self.models_dir / f"{dst_id}.json"

        if not src_model.exists():
            raise FileNotFoundError(f"Source model not found: {src_model}")

        if dst_model.exists():
            raise FileExistsError(f"Destination model already exists: {dst_model}")

        shutil.copy2(src_model, dst_model)

        if src_meta.exists():
            with open(src_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["model_id"] = dst_id
            meta["created_at"] = time.time()
            meta["updated_at"] = time.time()
            with open(dst_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        return dst_model

    def update_metadata(
        self,
        model_id: str,
        updates: Dict[str, Any],
    ) -> Optional[ModelMetadata]:
        """
        メタデータを更新

        Args:
            model_id: モデルID
            updates: 更新内容

        Returns:
            更新後のメタデータ
        """
        meta_path = self.models_dir / f"{model_id}.json"

        if not meta_path.exists():
            return None

        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data.update(updates)
        data["updated_at"] = time.time()

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return ModelMetadata.from_dict(data)


# グローバルマネージャーインスタンス
_global_model_manager: Optional[ModelManager] = None


def get_model_manager(models_dir: str = "./output/models") -> ModelManager:
    """グローバルモデルマネージャーを取得"""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = ModelManager(models_dir)
    return _global_model_manager
