"""
モデル永続化・バージョン管理（削除したlearning/model_manager.pyの後継）

旧model_manager.pyはレガシー単一銘柄エージェント用で、既に削除済みの
env/data モジュールに依存していたため使用不能だった。ここではポートフォリオ
エージェント向けに、既存の保存規約（{name}.zip = SB3形式 + {name}.json = メタ
データ）と互換な形で書き直す。

保存規約: {model_dir}/{name}.zip（SB3 save） + {model_dir}/{name}.json（メタデータ）。
メタデータには銘柄リスト・後処理設定・特徴マスク・学習時RunConfig・評価指標を含め、
/api/signal/latest と週次再学習のシャドー比較の両方がこれを読む
（ARCHITECTURE.md §3「再学習ループ」）。

昇格(promote)は current.json ポインタの更新のみで、モデル本体は移動しない
（ロールバックが即座にできるように）。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time


@dataclass
class ModelMetadata:
    """モデルに紐づくメタデータ（train/serve一致に必要な情報一式）"""
    symbols: List[str] = field(default_factory=list)
    post_processor: Dict[str, Any] = field(default_factory=dict)
    feature_mask: Optional[List[bool]] = None
    run_config: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    git_sha: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbols": self.symbols,
            "post_processor": self.post_processor,
            "feature_mask": self.feature_mask,
            "run_config": self.run_config,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "git_sha": self.git_sha,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelMetadata":
        return cls(
            symbols=list(d.get("symbols") or []),
            post_processor=dict(d.get("post_processor") or {}),
            feature_mask=d.get("feature_mask"),
            run_config=d.get("run_config"),
            metrics=dict(d.get("metrics") or {}),
            timestamp=float(d.get("timestamp", time.time())),
            git_sha=d.get("git_sha"),
        )


def _model_path(model_dir: Path, name: str) -> Path:
    return Path(model_dir) / f"{name}.zip"


def _meta_path(model_dir: Path, name: str) -> Path:
    return Path(model_dir) / f"{name}.json"


def save_bundle(model_dir: Path, name: str, agent: Any, metadata: ModelMetadata) -> Path:
    """agent（SB3モデルまたは.save()を持つオブジェクト）とメタデータを保存する"""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(model_dir / name))
    _meta_path(model_dir, name).write_text(
        json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return _model_path(model_dir, name)


def load_metadata(model_dir: Path, name: str) -> Optional[ModelMetadata]:
    """メタデータのみ読み込む（agentのロードはPPO.load等呼び出し側が行う）"""
    p = _meta_path(model_dir, name)
    if not p.exists():
        return None
    return ModelMetadata.from_dict(json.loads(p.read_text(encoding="utf-8")))


def model_exists(model_dir: Path, name: str) -> bool:
    return _model_path(model_dir, name).exists()


def list_models(model_dir: Path) -> List[str]:
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return []
    return sorted(p.stem for p in model_dir.glob("*.zip"))


def _pointer_path(model_dir: Path) -> Path:
    return Path(model_dir) / "current.json"


def _history_path(model_dir: Path) -> Path:
    return Path(model_dir) / "promotions.json"


def promote(model_dir: Path, name: str) -> None:
    """指定モデルを「現行」に昇格する（ポインタ更新のみ、本体は移動しない）"""
    model_dir = Path(model_dir)
    if not model_exists(model_dir, name):
        raise FileNotFoundError(f"model not found: {name} (in {model_dir})")

    history_path = _history_path(model_dir)
    history = []
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))

    current = get_current(model_dir)
    if current is not None:
        history.append({"name": current, "demoted_at": time.time()})
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    _pointer_path(model_dir).write_text(
        json.dumps({"name": name, "promoted_at": time.time()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_current(model_dir: Path) -> Optional[str]:
    p = _pointer_path(model_dir)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("name")


def rollback(model_dir: Path) -> Optional[str]:
    """直前に昇格していたモデルへポインタを戻す。履歴が無ければNone"""
    model_dir = Path(model_dir)
    history_path = _history_path(model_dir)
    if not history_path.exists():
        return None
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not history:
        return None
    previous = history.pop()["name"]
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    _pointer_path(model_dir).write_text(
        json.dumps({"name": previous, "promoted_at": time.time()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return previous
