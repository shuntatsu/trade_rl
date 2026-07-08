"""
レジーム・ハイブリッドルーター

単一戦略（trend_following等）は局面によって明確に勝敗が分かれる。実データ検証
（本モジュールの元になった調査）では trend_following がテスト区間で
trend_up_early(+20.0%)・range_highvol(+10.8%) を稼ぐ一方、trend_down_early(-2.2%)・
trend_up_mature(-5.2%)・range_lowvol(-1.3%) で負けることが確認された。

本モジュールは 6分類レジーム（regime_taxonomy.FINE_REGIMES）ごとに
{trend_following / flat / specialist(RL専門家)} を割り当てるルーターを提供する。
割当表（RouterTable）は「そのレジームで基準戦略が持続的に負けているか」を
過去データのみから判定する固定規則で導出する（閾値のチューニングはしない）。

過学習対策: 表の導出(derive_router_table)は常に「ある時点 end より前のデータ
のみ」を参照する設計にしてある。呼び出し側（run_router.py）が
walk-forward の各foldで「そのfoldのテスト区間より前」を end に渡すことで、
表自体が未来を覗き見ないようにする。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.features.regime_taxonomy import FINE_REGIMES, label_fine_regimes
from mars_lite.learning.baselines import (
    WeightFn,
    flat_strategy,
    simulate_strategy,
    trend_following_strategy,
)

# 固定パラメータ（実データで再チューニングしない）
MIN_REGIME_BARS = 300  # これ未満のレジームは判断せずデフォルト"tf"
SIGN_AGREEMENT_THRESHOLD = 2.0 / 3.0
N_SUBFOLDS = 3  # 表導出時の符号一致判定に使う内部分割数
CONFIRM_BARS = 2  # レジーム切替の確認に要する連続バー数（チャタリング対策）

# label_fine_regimesの既定値と一致させる（regime_taxonomy.label_fine_regimesの
# シグネチャ参照）。表をJSON保存する際に明示的に記録し、train/serve一致を保つ。
DEFAULT_LABELER_PARAMS = {
    "trend_threshold": 0.5,
    "vol_threshold": 0.0,
    "age_bars": 24,
}

# 合否基準（事前登録。実行結果を見てから変更しない）
ROUTER_GATE_CRITERIA = {
    "phase1": {
        "min_folds_beat_tf": 3,
        "min_folds_total": 4,
        "min_median_uplift_pt": 0.5,
        "require_positive_median_return": True,
    },
    "phase2": {
        "min_folds_beat_phase1": 3,
        "min_folds_total": 4,
        "min_median_uplift_pt": 0.5,
        "min_folds_regime_contribution_positive": 3,
    },
    "phase3": {
        "require_beat_tf": True,
        "require_positive_return": True,
        "require_positive_cost2x": True,
    },
}


@dataclass
class RouterTable:
    """レジーム→戦略の割当表（train/serve一致のためJSON化できる）"""

    assignments: Dict[str, str]  # regime -> "tf" | "flat" | "specialist"
    labeler_params: Dict[str, float]  # trend_threshold, vol_threshold, age_bars
    derivation: Dict[str, dict] = field(default_factory=dict)
    confirm_bars: int = CONFIRM_BARS

    def to_dict(self) -> Dict:
        return {
            "assignments": dict(self.assignments),
            "labeler_params": dict(self.labeler_params),
            "derivation": self.derivation,
            "confirm_bars": self.confirm_bars,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "RouterTable":
        return cls(
            assignments=dict(d["assignments"]),
            labeler_params=dict(d["labeler_params"]),
            derivation=d.get("derivation", {}),
            confirm_bars=int(d.get("confirm_bars", CONFIRM_BARS)),
        )

    def save(self, path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path) -> "RouterTable":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def regime_contributions(
    fs: FeatureSet,
    weight_fn: WeightFn,
    labels: np.ndarray,
    start: int,
    end: int,
) -> Dict[str, Dict[str, float]]:
    """
    weight_fn を [start, end) 区間でバックテストし、レジーム別の複利寄与を
    集計する（simulate_strategyのコストモデルをそのまま使う）。

    simulate_strategy の rets[i] は t=start+i での意思決定の結果なので、
    labels[start+i] と1対1対応する。

    simulate_strategy は内部で fs.close[t+1] を参照するため end_idx は
    fs.n_bars-1 が上限。end がそれを超える場合はクランプする。
    """
    end = min(end, fs.n_bars - 1)
    if end - start < 2:
        return {r: {"n_bars": 0, "growth": 0.0, "mean_ret": 0.0} for r in FINE_REGIMES}

    res = simulate_strategy(fs, weight_fn, start_idx=start, end_idx=end)
    equity = res.equity_curve
    rets = np.diff(equity) / equity[:-1]
    n = min(len(rets), end - start)
    seg_labels = labels[start : start + n]

    out: Dict[str, Dict[str, float]] = {}
    for r in FINE_REGIMES:
        mask = seg_labels == r
        cnt = int(mask.sum())
        if cnt == 0:
            out[r] = {"n_bars": 0, "growth": 0.0, "mean_ret": 0.0}
            continue
        growth = float(np.prod(1.0 + rets[:n][mask]) - 1.0)
        out[r] = {
            "n_bars": cnt,
            "growth": growth,
            "mean_ret": float(rets[:n][mask].mean()),
        }
    return out


def _subfold_growths(
    fs: FeatureSet,
    weight_fn: WeightFn,
    labels: np.ndarray,
    end: int,
    regime: str,
    n_subfolds: int = N_SUBFOLDS,
) -> List[float]:
    """[0, end) を n_subfolds分割し、各分割内でのregime寄与growthを返す"""
    edges = np.linspace(0, end, n_subfolds + 1).astype(int)
    growths = []
    for i in range(n_subfolds):
        s, e = int(edges[i]), int(edges[i + 1])
        if e - s < 50:
            continue
        contrib = regime_contributions(fs, weight_fn, labels, s, e)
        growths.append(contrib[regime]["growth"])
    return growths


def derive_router_table(
    fs: FeatureSet,
    labels: np.ndarray,
    end: int,
    weight_fn: WeightFn = trend_following_strategy,
    labeler_params: Optional[Dict[str, float]] = None,
    min_regime_bars: int = MIN_REGIME_BARS,
    sign_agreement_threshold: float = SIGN_AGREEMENT_THRESHOLD,
    n_subfolds: int = N_SUBFOLDS,
    confirm_bars: int = CONFIRM_BARS,
) -> RouterTable:
    """
    [0, end) のデータのみを使ってレジーム別の割当表を導出する（固定決定規則）。

    規則:
      - そのレジームのバー数 < min_regime_bars → 既定"tf"（小サンプルは判断しない）
      - サブフォールド寄与のsign_agreement >= threshold かつ 全体寄与 < 0
        → "flat"（基準戦略が持続的に負けている）
      - それ以外 → "tf"
    """
    contrib = regime_contributions(fs, weight_fn, labels, 0, end)
    assignments: Dict[str, str] = {}
    derivation: Dict[str, dict] = {}

    for r in FINE_REGIMES:
        n_bars = contrib[r]["n_bars"]
        if n_bars < min_regime_bars:
            assignments[r] = "tf"
            derivation[r] = {
                "reason": "insufficient_sample",
                "n_bars": n_bars,
                "total_growth": contrib[r]["growth"],
            }
            continue

        growths = _subfold_growths(fs, weight_fn, labels, end, r, n_subfolds)
        total_growth = contrib[r]["growth"]
        if len(growths) < 2:
            assignments[r] = "tf"
            derivation[r] = {
                "reason": "insufficient_subfolds",
                "n_bars": n_bars,
                "total_growth": total_growth,
            }
            continue

        dom_sign = float(np.sign(sum(growths)))
        agree = (
            float(np.mean([np.sign(g) == dom_sign for g in growths]))
            if dom_sign != 0
            else 0.0
        )
        if agree >= sign_agreement_threshold and total_growth < 0:
            assignments[r] = "flat"
            derivation[r] = {
                "reason": "persistent_negative",
                "n_bars": n_bars,
                "total_growth": total_growth,
                "subfold_growths": growths,
                "sign_agreement": agree,
            }
        else:
            assignments[r] = "tf"
            derivation[r] = {
                "reason": "default_or_positive",
                "n_bars": n_bars,
                "total_growth": total_growth,
                "subfold_growths": growths,
                "sign_agreement": agree,
            }

    return RouterTable(
        assignments=assignments,
        labeler_params=labeler_params or dict(DEFAULT_LABELER_PARAMS),
        derivation=derivation,
        confirm_bars=confirm_bars,
    )


def _confirm_series(raw_labels: np.ndarray, confirm_bars: int) -> np.ndarray:
    """
    生ラベル系列にヒステリシスをかけた「確定レジーム」系列を返す（因果的）。

    confirmed[i] は raw_labels[0..i] のみに依存するため、事前一括計算しても
    バーごとの逐次呼び出しと同じ結果になる（未来を見ない）。
    候補レジームが confirm_bars 本連続で観測されて初めて切り替える。
    """
    n = len(raw_labels)
    confirmed = np.empty(n, dtype=object)
    if n == 0:
        return confirmed

    current = raw_labels[0]
    candidate = raw_labels[0]
    candidate_count = 1
    confirmed[0] = current
    for i in range(1, n):
        if raw_labels[i] == candidate:
            candidate_count += 1
        else:
            candidate = raw_labels[i]
            candidate_count = 1
        if candidate_count >= confirm_bars and candidate != current:
            current = candidate
        confirmed[i] = current
    return confirmed


def make_router_weight_fn(
    fs: FeatureSet,
    table: RouterTable,
    specialists: Optional[Dict[str, WeightFn]] = None,
) -> WeightFn:
    """
    RouterTable から WeightFn を組み立てる。

    **下位戦略が実際に変わったバー**（例: flat→tf, tf→specialist）では、
    下位戦略を w=zeros で呼び強制リバランスする。trend_following_strategy は
    `t%24!=0 and w.any()` で前回ウェイトを素通しするため、そうしないと
    specialist→tf切替時に最大23本の遅延が発生するため。

    注意: 判定は「生ラベル/確定レジームが変わったか」ではなく「割り当てられた
    戦略オブジェクトが変わったか」で行う。例えば range_lowvol→range_highvol は
    レジームとしては切替だが、両方とも"tf"割当なら実際にはtrend_following
    自身の内部ロジック（24本毎リバランス・prevウェイト素通し）に任せるべきで、
    ここで毎回w=zerosを強制すると同一戦略なのに不要な往復コストを払う
    （実データ検証で発見: 全レジームがtf割当のfoldでもrouterの回転率が
    trend_following単体の約1.9倍になっていた）。
    """
    specialists = specialists or {}
    raw_labels = label_fine_regimes(fs, **table.labeler_params)
    confirmed = _confirm_series(raw_labels, table.confirm_bars)

    def strategy_for(regime) -> WeightFn:
        assignment = table.assignments.get(regime, "tf")
        if assignment == "flat":
            return flat_strategy
        if assignment == "specialist" and regime in specialists:
            return specialists[regime]
        return trend_following_strategy

    def weight_fn(fs_: FeatureSet, t: int, w: np.ndarray) -> np.ndarray:
        idx = min(t, len(confirmed) - 1)
        regime = confirmed[idx]
        strat = strategy_for(regime)
        prev_regime = confirmed[idx - 1] if idx > 0 else regime
        strategy_changed = idx > 0 and strategy_for(prev_regime) is not strat
        if strategy_changed:
            return strat(fs_, t, np.zeros(fs_.n_symbols))
        return strat(fs_, t, w)

    return weight_fn
