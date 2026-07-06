"""
統一設定ツリー（mars_lite.config.RunConfig）

学習・評価・運用で共有するハイパーパラメータの単一の正。
これまで scripts/train_portfolio.py のargparseデフォルトや各モジュールの
関数デフォルトに散在していた値をここに集約する。

既定値は docs/ARCHITECTURE.md の実測ベンチマークに基づく（§2「確定仕様と根拠」）。
既定値を変更する場合は同じベンチマークで根拠を示し、
tests/test_config.py の「憲法テスト」を合わせて更新すること。
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

from mars_lite.trading.post_processor import PostProcessConfig
from mars_lite.trading.guardrails import GuardrailConfig

BARS_PER_YEAR_1H = 24 * 365


@dataclass
class DataConfig:
    """データソース設定"""
    source: str = "synthetic"  # "synthetic" / "csv" / "postgres" / "hyperliquid"
    symbols: Optional[List[str]] = None
    data_dir: str = "./data"
    days: int = 240  # syntheticの生成日数
    alpha: str = "cross"  # synthetic専用: none/cross/meanrev/multi/bull
    alpha_strength: float = 0.002
    seed: int = 0
    warmup_days: float = 0.0
    pg_dsn: Optional[str] = None
    pg_source: str = "binance"
    pg_derivatives_source: Optional[str] = None


@dataclass
class FeatureConfig:
    """特徴量パイプライン設定"""
    horizon: int = 4  # 予測ホライズン（バー数）。ICゲート/BC教師/特徴マスクに使う
    scan_horizons: bool = False
    horizons: Tuple[int, ...] = (1, 2, 4, 8, 24, 48, 72)
    feature_norm: str = "none"  # "none" / "rank_gauss"
    feature_mask: bool = False


@dataclass
class EnvConfig:
    """PortfolioTradingEnv設定（実測値はARCHITECTURE.md §2.2）"""
    episode_bars: int = 200
    fee_rate: float = 0.0005
    spread_rate: float = 0.0002
    impact_rate: float = 0.0001
    cost_multiplier: float = 1.0
    lambda_turnover: float = 0.04
    reward_scale: float = 100.0
    decision_every: int = 1
    htf_gate: bool = False
    htf_threshold: float = 0.3
    htf_neutral_scale: float = 0.5
    min_trade_delta: float = 0.04
    bars_per_year: int = BARS_PER_YEAR_1H


@dataclass
class TrainConfig:
    """PPO学習設定（実測値はARCHITECTURE.md §2.1「学習レシピ」）"""
    timesteps: int = 300_000
    seed: int = 0
    n_envs: int = 8
    learning_rate: float = 3e-4
    ent_coef: float = 0.002
    gamma: float = 0.5  # 0.995では-26%→0.5で+46%に反転（実測）
    gae_lambda: float = 0.9
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 6
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_size: str = "small"  # small=実証済み既定、large=要再ベンチ
    dropout: float = 0.0
    extractor: str = "tfgated"
    bc_warmstart: bool = True
    bc_teacher: str = "auto"
    bc_epochs: int = 15
    val_eval_freq: int = 20_000
    ensemble: int = 1  # 実データでは3推奨


@dataclass
class ServeConfig:
    """運用（/api/signal/latest）設定"""
    model_dir: str = "./output/portfolio/models"
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)


@dataclass
class RunConfig:
    """学習1回分の設定一式。JSON化してモデルメタデータに丸ごと保存する。"""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    postproc: PostProcessConfig = field(default_factory=PostProcessConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        def _filtered(dc_cls, d: Dict[str, Any]):
            valid = {f.name for f in dc_cls.__dataclass_fields__.values()}
            return dc_cls(**{k: v for k, v in d.items() if k in valid})

        d = data or {}
        serve_data = dict(d.get("serve", {}) or {})
        guardrails = serve_data.pop("guardrails", None)
        serve_kwargs = {k: v for k, v in serve_data.items()
                        if k in ServeConfig.__dataclass_fields__}
        if guardrails is not None:
            serve_kwargs["guardrails"] = _filtered(GuardrailConfig, guardrails)

        return cls(
            data=_filtered(DataConfig, d.get("data", {}) or {}),
            features=_filtered(FeatureConfig, d.get("features", {}) or {}),
            env=_filtered(EnvConfig, d.get("env", {}) or {}),
            postproc=_filtered(PostProcessConfig, d.get("postproc", {}) or {}),
            train=_filtered(TrainConfig, d.get("train", {}) or {}),
            serve=ServeConfig(**serve_kwargs),
        )

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
