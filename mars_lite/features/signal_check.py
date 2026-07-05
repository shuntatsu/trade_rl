"""
シグナル検証モジュール（ゲート1）

特徴量に「次kバーのリターン」への予測力（アルファ）があるかを
RL学習の前に検証する。予測力が無いデータでRLを学習させても
「取引しない」が最適解になるため、ここが不合格なら先に進まない。

手法:
- 単一特徴量のランクIC（Spearman相関）: 特徴量ごとの素の予測力
- Ridge回帰のウォークフォワードOOSランクIC: 特徴量を合成した予測力

判定基準（デフォルト）:
- 合成予測のOOSランクICの平均 >= 0.02 かつ 期間の6割以上で正
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mars_lite.features.feature_pipeline import FeatureSet


@dataclass
class SignalReport:
    """ゲート1の判定レポート"""
    per_feature_ic: Dict[str, float]
    oos_ic_by_fold: List[float]
    mean_oos_ic: float
    positive_fold_ratio: float
    n_folds: int
    horizon: int
    passed: bool
    threshold: float
    t_stat: float = 0.0
    min_t_stat: float = 1.0
    stability_passed: bool = True
    target: str = "raw"

    def to_dict(self) -> Dict:
        return {
            "mean_oos_ic": float(self.mean_oos_ic),
            "positive_fold_ratio": float(self.positive_fold_ratio),
            "n_folds": int(self.n_folds),
            "horizon": int(self.horizon),
            "passed": bool(self.passed),
            "threshold": float(self.threshold),
            "t_stat": float(self.t_stat),
            "min_t_stat": float(self.min_t_stat),
            "stability_passed": bool(self.stability_passed),
            "target": str(self.target),
            "oos_ic_by_fold": [float(x) for x in self.oos_ic_by_fold],
            "top_features": dict(sorted(
                {k: float(v) for k, v in self.per_feature_ic.items()}.items(), key=lambda x: -abs(x[1])
            )[:10]),
        }

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        top = sorted(self.per_feature_ic.items(), key=lambda x: -abs(x[1]))[:5]
        lines = [
            f"[Gate 1: Signal Check] {status} (target={self.target})",
            f"  OOS rank IC: mean={self.mean_oos_ic:.4f} "
            f"(threshold={self.threshold}), positive folds="
            f"{self.positive_fold_ratio:.0%} ({self.n_folds} folds, horizon={self.horizon} bars)",
            f"  Stability: t_stat={self.t_stat:.2f} (min={self.min_t_stat}) "
            f"{'OK' if self.stability_passed else 'UNSTABLE (fold IC sign flips too much)'}",
            "  Top features by |IC|:",
        ]
        lines += [f"    {name:<24} IC={ic:+.4f}" for name, ic in top]
        return "\n".join(lines)


def _rank(x: np.ndarray) -> np.ndarray:
    """1次元配列を[0,1]ランクに変換"""
    order = np.argsort(np.argsort(x))
    return order / max(len(x) - 1, 1)


def _rank_ic(pred: np.ndarray, target: np.ndarray) -> float:
    """SpearmanランクIC（scipy不使用）"""
    if len(pred) < 10 or np.std(pred) == 0 or np.std(target) == 0:
        return 0.0
    return float(np.corrcoef(_rank(pred), _rank(target))[0, 1])


def _fold_ic_stats(
    oos_ics: List[float], min_t_stat: float = 1.0,
) -> Tuple[float, float, float, bool]:
    """
    fold別OOS ICのリストから (mean_ic, positive_ratio, t_stat, stability_passed) を計算

    t_stat = mean/std*sqrt(n) は「fold間ICの平均が0と有意に異なるか」の
    片標本t検定統計量。境界ぎりぎりの平均IC点推定だけでは、fold間で
    符号が頻繁に反転する不安定な信号でもゲートを通ってしまうため、
    この安定性判定を独立に検証できるよう関数として切り出す。
    """
    if not oos_ics:
        return 0.0, 0.0, 0.0, False

    mean_ic = float(np.mean(oos_ics))
    pos_ratio = float(np.mean([ic > 0 for ic in oos_ics]))

    n = len(oos_ics)
    ic_std = float(np.std(oos_ics, ddof=1)) if n > 1 else 0.0
    if n > 1 and ic_std > 1e-12:
        t_stat = mean_ic / ic_std * np.sqrt(n)
    else:
        # foldが1つ以下、またはfold間で全く同一の値（分散ゼロ）の場合は
        # 安定性を主張できないため、非負のICなら中立的にt_stat=min_t_statと
        # みなす（ゼロ除算での偽の合格/不合格を避ける）
        t_stat = min_t_stat if mean_ic > 0 else 0.0
    stability_passed = abs(t_stat) >= min_t_stat
    return mean_ic, pos_ratio, float(t_stat), stability_passed


def _forward_returns(fs: FeatureSet, horizon: int) -> np.ndarray:
    """次horizonバーの累積ログリターン (n_bars, n_symbols)、末尾はNaN"""
    log_close = np.log(fs.close)
    fwd = np.full_like(log_close, np.nan)
    fwd[:-horizon] = log_close[horizon:] - log_close[:-horizon]
    return fwd


def _realized_vol(fs: FeatureSet, window: int = 20) -> np.ndarray:
    """直近windowバーの実現ボラ（1バーlogリターンのローリング標準偏差、per symbol）"""
    log_ret = np.diff(np.log(fs.close), axis=0)
    log_ret = np.vstack([np.zeros((1, fs.n_symbols)), log_ret])
    vol = np.full_like(log_ret, np.nan)
    for i in range(fs.n_symbols):
        s = pd.Series(log_ret[:, i])
        vol[:, i] = s.rolling(window, min_periods=max(5, window // 4)).std().to_numpy()
    return vol


def _transform_target(fwd: np.ndarray, target: str, fs: FeatureSet) -> np.ndarray:
    """
    前方リターンのターゲット変換

    raw       : そのまま（絶対リターン予測。方向性ベータを含む）
    cs_demean : バー毎に銘柄間平均を引く（市場中立の相対アルファのみ抽出）
    vol_norm  : 直近実現ボラで正規化（外れ値バーの支配を防ぐ）
    """
    if target == "raw":
        return fwd
    if target == "cs_demean":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            row_mean = np.nanmean(fwd, axis=1, keepdims=True)
        return fwd - row_mean
    if target == "vol_norm":
        vol = _realized_vol(fs)
        vol = np.where(vol > 1e-9, vol, np.nan)
        return fwd / vol
    raise ValueError(f"unknown target: {target}")


def _pool(fs: FeatureSet, horizon: int, target: str = "raw"):
    """全銘柄をプールした (X, y, bar_idx) を作成（末尾horizonバー除外）

    target: "raw"（既定・絶対リターン）| "cs_demean"（市場中立の相対アルファ）
            | "vol_norm"（ボラ正規化）。signal_check・ridge教師・特徴マスクの
            すべてで同じ意味論を共有する。
    """
    n_bars, n_sym, n_feat = fs.features.shape
    fwd = _forward_returns(fs, horizon)
    fwd = _transform_target(fwd, target, fs)
    valid = n_bars - horizon
    X = fs.features[:valid].reshape(valid * n_sym, n_feat)
    y = fwd[:valid].reshape(valid * n_sym)
    bar_idx = np.repeat(np.arange(valid), n_sym)
    mask = np.isfinite(y)
    return X[mask], y[mask], bar_idx[mask]


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 10.0) -> np.ndarray:
    """Ridge回帰の閉形式解（バイアス項付き）"""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ y)


def _ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def compute_feature_mask(
    fs: FeatureSet,
    horizon: int = 4,
    n_folds: int = 4,
    min_abs_ic: float = 0.01,
    min_sign_agreement: float = 0.75,
    target: str = "raw",
) -> Dict[str, object]:
    """
    IC安定性による特徴マスクを計算（時間軸の自動取捨選択）

    学習スライスを時間順にn_folds分割し、特徴ごとにfold別ランクICを計算。
    「平均|IC|がしきい値以上」かつ「ICの符号がfold間で一致」する特徴のみ残す。
    予測力のないTFブロックはここで自動的に落ちるため、多時間軸観測を
    「シグナルがあるTFだけ使う」形に縮退させられる。

    target: "raw"|"cs_demean"|"vol_norm"（_pool参照）。ridge教師をcs_demean
    で学習する場合は、特徴マスクも同じtargetで計算するのが整合的。

    注意: fsには**学習スライスのみ**を渡すこと（選択リーク防止）。
    マスクは推論時にも同一適用が必要（モデルメタデータに保存する）。

    Returns:
        {mask: np.ndarray(bool, n_features), kept: [names], dropped: [names],
         per_feature: {name: {mean_abs_ic, sign_agreement}}}
    """
    X, y, bar_idx = _pool(fs, horizon, target=target)
    n_bars = int(bar_idx.max()) + 1
    edges = np.linspace(0, n_bars, n_folds + 1).astype(int)

    n_feat = X.shape[1]
    fold_ics = np.zeros((n_folds, n_feat))
    for k in range(n_folds):
        m = (bar_idx >= edges[k]) & (bar_idx < edges[k + 1])
        if m.sum() < 50:
            continue
        for j in range(n_feat):
            fold_ics[k, j] = _rank_ic(X[m][:, j], y[m])

    mean_abs = np.abs(fold_ics.mean(axis=0))
    signs = np.sign(fold_ics)
    dominant = np.sign(fold_ics.mean(axis=0))[None, :]
    agreement = (signs == dominant).mean(axis=0)

    mask = (mean_abs >= min_abs_ic) & (agreement >= min_sign_agreement)
    names = fs.feature_names
    return {
        "mask": mask,
        "kept": [n for n, m in zip(names, mask) if m],
        "dropped": [n for n, m in zip(names, mask) if not m],
        "per_feature": {
            n: {"mean_abs_ic": float(a), "sign_agreement": float(g)}
            for n, a, g in zip(names, mean_abs, agreement)
        },
    }


def run_trend_gate(
    fs: FeatureSet, horizon: int = 4, t_threshold: float = 3.0,
    n_folds: int = 4, min_sign_agreement: float = 1.0,
) -> Dict[str, object]:
    """
    方向性トレンドゲート（**持続的な**方向性ベータの有無を検出）

    全銘柄プールの前方リターン平均が有意に非ゼロ（t検定）**かつ**、
    時系列fold間でドリフトの符号が一致することを要求する。

    後者が重要: ランダムウォークは有限窓で実現ドリフトを持つため単純な
    t検定は偽陽性を出す（1つの実現に過剰適合）。真の持続ドリフトは全fold
    で同符号だが、ランダムウォークのドリフトはfold間でばらつく。ICマスクと
    同じ符号一致原理でこれを弾く。

    Returns:
        {mean_fwd, t_stat, fold_sign_agreement, has_trend, direction}
    """
    _, y, bar_idx = _pool(fs, horizon)
    if len(y) < 40:
        return {"mean_fwd": 0.0, "t_stat": 0.0, "fold_sign_agreement": 0.0,
                "has_trend": False, "direction": 0}

    mean = float(np.mean(y))
    se = float(np.std(y) / np.sqrt(len(y)))
    t = mean / se if se > 1e-12 else 0.0
    direction = int(np.sign(mean))

    # fold別のドリフト符号一致
    n_bars = int(bar_idx.max()) + 1
    edges = np.linspace(0, n_bars, n_folds + 1).astype(int)
    fold_means = []
    for k in range(n_folds):
        m = (bar_idx >= edges[k]) & (bar_idx < edges[k + 1])
        if m.sum() >= 10:
            fold_means.append(float(np.mean(y[m])))
    agreement = (float(np.mean([np.sign(fm) == direction for fm in fold_means]))
                 if fold_means else 0.0)

    has_trend = abs(t) >= t_threshold and agreement >= min_sign_agreement
    return {
        "mean_fwd": mean,
        "t_stat": float(t),
        "fold_sign_agreement": agreement,
        "has_trend": bool(has_trend),
        "direction": direction,
    }


def run_leak_self_test(fs: FeatureSet, horizon: int = 4) -> Dict[str, object]:
    """
    リーク検出器自体の健全性を検査する自己テスト

    - shuffle検査: ターゲットを時間シャッフルするとIC≈0 になるべき
      （ならなければ評価コード側のバグ）
    - future-shift検査: 特徴を意図的に1バー未来へずらすとICが跳ね上がるべき
      （＝リーク検出器が実際にリークを捕捉できる証明）

    Returns:
        {shuffle_ic, base_ic, future_shift_ic, healthy: bool}
    """
    X, y, bar_idx = _pool(fs, horizon)

    # ベースIC（全特徴平均の素のIC）
    base_ic = float(np.mean([abs(_rank_ic(X[:, j], y)) for j in range(X.shape[1])]))

    # shuffle: yをシャッフル → 相関消失を期待
    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(y)
    shuffle_ic = float(np.mean([abs(_rank_ic(X[:, j], y_shuf)) for j in range(X.shape[1])]))

    # future-shift: 特徴を未来方向へずらす（バー単位）→ リーク混入 → IC増大を期待
    n_bars = int(bar_idx.max()) + 1
    n_sym = fs.n_symbols
    fwd = _forward_returns(fs, horizon)
    valid = n_bars
    # 特徴を1バー未来にシフト（先読み）
    feat_shift = np.roll(fs.features, -1, axis=0)
    Xs = feat_shift[:valid].reshape(valid * n_sym, -1)
    ys = fwd[:valid].reshape(valid * n_sym)
    mask = np.isfinite(ys)
    Xs, ys = Xs[mask], ys[mask]
    future_ic = float(np.mean([abs(_rank_ic(Xs[:, j], ys)) for j in range(Xs.shape[1])]))

    healthy = shuffle_ic < 0.02 and future_ic > base_ic
    return {
        "shuffle_ic": shuffle_ic,
        "base_ic": base_ic,
        "future_shift_ic": future_ic,
        "healthy": bool(healthy),
    }


def run_signal_check(
    fs: FeatureSet,
    horizon: int = 4,
    n_folds: int = 5,
    purge_bars: int = 24,
    threshold: float = 0.02,
    min_positive_ratio: float = 0.6,
    min_t_stat: float = 1.0,
    target: str = "raw",
) -> SignalReport:
    """
    ウォークフォワードICゲートを実行

    Args:
        fs: FeatureSet
        horizon: 予測対象の先読みバー数（1h足なら4=4時間先）
        n_folds: ウォークフォワード分割数
        purge_bars: 学習と検証の間に挟むギャップ（リーク防止）
        threshold: 合格に必要な平均OOSランクIC
        min_positive_ratio: 正のICが必要なfoldの割合
        min_t_stat: fold間IC安定性のt値下限（mean/std*sqrt(n_folds)）。
            平均ICが閾値を超えても、fold間で符号が頻繁に反転する
            （t値が低い）場合は不合格にする。境界ぎりぎりの点推定
            だけでRL学習にGOを出す事故を防ぐ。
        target: "raw"（既定・絶対リターン、方向性ベータ込み）|
            "cs_demean"（バー毎に銘柄間平均を引いた市場中立の相対アルファ）|
            "vol_norm"（直近実現ボラで正規化）。7銘柄程度の狭いユニバースで
            は市場全体の方向がICを汚染しうるため、cs_demeanで測ると
            「本当の相対アルファ」だけを見られる。
    """
    X, y, bar_idx = _pool(fs, horizon, target=target)
    n_bars = int(bar_idx.max()) + 1

    # 特徴量ごとの素のIC（全期間）
    per_feature = {
        name: _rank_ic(X[:, j], y) for j, name in enumerate(fs.feature_names)
    }

    # ウォークフォワード: 前半で学習→purge→次区間で検証、をスライド
    fold_edges = np.linspace(0, n_bars, n_folds + 2).astype(int)
    oos_ics: List[float] = []
    for k in range(1, n_folds + 1):
        train_end = fold_edges[k]
        test_start = train_end + purge_bars
        test_end = fold_edges[k + 1]
        if test_start >= test_end or train_end < 50:
            continue

        train_mask = bar_idx < train_end
        test_mask = (bar_idx >= test_start) & (bar_idx < test_end)
        if train_mask.sum() < 100 or test_mask.sum() < 50:
            continue

        w = _ridge_fit(X[train_mask], y[train_mask])
        pred = _ridge_predict(X[test_mask], w)
        oos_ics.append(_rank_ic(pred, y[test_mask]))

    mean_ic, pos_ratio, t_stat, stability_passed = _fold_ic_stats(oos_ics, min_t_stat)
    passed = mean_ic >= threshold and pos_ratio >= min_positive_ratio and stability_passed

    return SignalReport(
        per_feature_ic=per_feature,
        oos_ic_by_fold=[float(x) for x in oos_ics],
        mean_oos_ic=mean_ic,
        positive_fold_ratio=pos_ratio,
        n_folds=len(oos_ics),
        horizon=horizon,
        passed=passed,
        threshold=threshold,
        t_stat=float(t_stat),
        min_t_stat=min_t_stat,
        stability_passed=stability_passed,
        target=target,
    )
