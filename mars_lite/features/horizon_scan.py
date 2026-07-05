"""
ホライズンスキャンモジュール

手元のデータに実際にアルファがあるホライズンを見つける。ゲート1
(signal_check.run_signal_check) は既定でhorizon=4（1h足で4時間先）
固定だが、シグナルの効く時間軸はデータ・特徴群によって異なる。
複数ホライズンでウォークフォワードOOS ICを測り、最良ホライズンと
特徴グループ別のIC内訳を返す。

注意: train/val/testの分離原則に従い、ホライズン選択はtrainスライス
のみで行い、選ばれたホライズンをtestで最終評価する（本モジュール自体は
ウォークフォワード分割のみでtest分離は呼び出し側の責務）。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from mars_lite.features.feature_pipeline import FeatureSet, TF_BLOCK_FEATURES
from mars_lite.features.signal_check import run_signal_check, SignalReport

DEFAULT_HORIZONS: Tuple[int, ...] = (1, 2, 4, 8, 24, 48, 72)


def default_feature_groups(fs: FeatureSet) -> Dict[str, List[str]]:
    """特徴名から標準グループを導出（TFブロック別 + オーダーフロー + デリバティブ + funding + クロスセクション + その他）"""
    groups: Dict[str, List[str]] = {}
    tf_feature_set = set(TF_BLOCK_FEATURES)
    for name in fs.feature_names:
        if "_" in name:
            prefix, rest = name.split("_", 1)
            if prefix in ("15m", "30m", "1h", "4h", "1d") and rest in tf_feature_set:
                groups.setdefault(f"tf_{prefix}", []).append(name)
                continue
        if name.startswith("of_"):
            groups.setdefault("orderflow", []).append(name)
        elif name in ("oi_z", "oi_change", "ls_ratio_z", "liq_z"):
            groups.setdefault("derivatives", []).append(name)
        elif name.startswith("funding_") or name == "time_to_funding":
            groups.setdefault("funding", []).append(name)
        elif name in ("ret_rank", "btc_rel_z"):
            groups.setdefault("cross_sectional", []).append(name)
        else:
            groups.setdefault("other", []).append(name)
    return groups


@dataclass
class HorizonResult:
    horizon: int
    mean_oos_ic: float
    positive_fold_ratio: float
    group_ic: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "horizon": self.horizon,
            "mean_oos_ic": self.mean_oos_ic,
            "positive_fold_ratio": self.positive_fold_ratio,
            "group_ic": self.group_ic,
        }


@dataclass
class HorizonScanReport:
    results: List[HorizonResult]

    @property
    def best_horizon(self) -> int:
        return max(self.results, key=lambda r: r.mean_oos_ic).horizon

    def to_dict(self) -> Dict:
        return {
            "best_horizon": self.best_horizon,
            "results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        lines = ["[Horizon Scan]"]
        for r in self.results:
            top_group = max(r.group_ic.items(), key=lambda kv: abs(kv[1])) if r.group_ic else ("-", 0.0)
            lines.append(
                f"  horizon={r.horizon:<4} OOS_IC={r.mean_oos_ic:+.4f} "
                f"(+folds={r.positive_fold_ratio:.0%})  top_group={top_group[0]}({top_group[1]:+.3f})"
            )
        lines.append(f"  Best horizon: {self.best_horizon}")
        return "\n".join(lines)


def run_horizon_scan(
    fs: FeatureSet,
    horizons: Tuple[int, ...] = DEFAULT_HORIZONS,
    n_folds: int = 5,
    min_positive_ratio: float = 0.6,
    threshold: float = 0.0,
    feature_groups: Optional[Dict[str, List[str]]] = None,
    target: str = "raw",
) -> HorizonScanReport:
    """
    複数ホライズンでウォークフォワードOOS ICをスキャンする

    Args:
        fs: 学習スライスのFeatureSet（リーク防止のためtestは渡さないこと）
        horizons: 走査するホライズン（バー数）
        n_folds: 各ホライズンでのウォークフォワード分割数
        threshold: run_signal_checkに渡す合否閾値（本関数はpassed判定を
                   使わず比較のみに使うため実質的に無視してよい）
        feature_groups: name -> [feature_names] のグループ定義
                        （省略時は default_feature_groups で自動導出）
        target: "raw"|"cs_demean"|"vol_norm"（signal_check._pool参照）
    """
    groups = feature_groups or default_feature_groups(fs)
    name_to_idx = {n: i for i, n in enumerate(fs.feature_names)}

    results: List[HorizonResult] = []
    for h in horizons:
        purge = max(24, h)
        report: SignalReport = run_signal_check(
            fs, horizon=h, n_folds=n_folds, purge_bars=purge,
            threshold=threshold, min_positive_ratio=min_positive_ratio,
            target=target,
        )
        group_ic = {}
        for gname, fnames in groups.items():
            ics = [report.per_feature_ic[n] for n in fnames if n in name_to_idx]
            group_ic[gname] = float(np.mean(np.abs(ics))) if ics else 0.0

        results.append(HorizonResult(
            horizon=h, mean_oos_ic=report.mean_oos_ic,
            positive_fold_ratio=report.positive_fold_ratio,
            group_ic=group_ic,
        ))

    return HorizonScanReport(results=results)


def compute_breakeven_ic(
    fs: FeatureSet,
    horizon: int,
    candidate_ics: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3),
    decision_every: Optional[int] = None,
    seed: int = 0,
    n_draws: int = 2,
    **cost_kwargs,
) -> Optional[float]:
    """
    そのホライズン・意思決定頻度で「コスト後に黒字化する最小の目標IC」を推定する

    ノイズ入りオラクル（baselines.noisy_oracle_strategy）を候補IC群で
    昇順に評価し、total_return > 0 になる最初のICを返す。全て黒字化
    しなければNone（=このコスト構造ではどのIC水準でも割に合わない）。

    decision_every省略時は Phase B の既定連動則
    （max(1, horizon // 2)）を使う。
    """
    from mars_lite.learning.baselines import noisy_oracle_strategy

    de = decision_every if decision_every is not None else max(1, horizon // 2)
    for ic in sorted(candidate_ics):
        result = noisy_oracle_strategy(
            fs, target_ic=ic, seed=seed, n_draws=n_draws,
            decision_every=de, **cost_kwargs,
        )
        if result.total_return > 0:
            return float(ic)
    return None
