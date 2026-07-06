"""
ゲート1診断: horizon×targetのネストウォークフォワード検証

用途: gate1(IC>=0.02)が通らない場合に、「パラメータ選択(horizon/target/
Ridge正則化)が悪いのか」と「特徴量セット自体の限界なのか」を切り分ける。

設計上守っていること:
- holdout分離: 本モジュールに渡すFeatureSetは呼び出し側が既にdev区間へ
  絞り込んだものである前提。horizon選択・target選択・lambda選択・
  特徴選択のいずれにもholdoutを使ってはならない（呼び出し側の責務）。
- ネストウォークフォワード: 各outer foldの外側テスト区間はlambda選択に
  一切使わない。外側学習区間の末尾側をinner検証として切り出してlambdaを選び、
  外側学習区間全体で選んだlambdaにより再学習してから外側テストで評価する。
- 標準化: 各outer foldの学習区間の平均・標準偏差だけを使い、その値を
  検証・テスト区間に適用する（未来情報の混入を防ぐ）。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.features.horizon_scan import default_feature_groups
from mars_lite.features.signal_check import (
    _fold_ic_stats,
    _forward_returns,
    _rank_ic,
    _transform_target,
)

DEFAULT_LAMBDAS: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)


def _pool_with_symbol(fs: FeatureSet, horizon: int, target: str):
    """signal_check._pool と同じだが、行ごとの銘柄indexも返す"""
    n_bars, n_sym, n_feat = fs.features.shape
    fwd = _forward_returns(fs, horizon)
    fwd = _transform_target(fwd, target, fs)
    valid = n_bars - horizon
    X = fs.features[:valid].reshape(valid * n_sym, n_feat)
    y = fwd[:valid].reshape(valid * n_sym)
    bar_idx = np.repeat(np.arange(valid), n_sym)
    sym_idx = np.tile(np.arange(n_sym), valid)
    mask = np.isfinite(y)
    return X[mask], y[mask], bar_idx[mask], sym_idx[mask]


def _standardize(train: np.ndarray, other: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """trainの平均・標準偏差だけで両方を標準化（未来情報の混入防止）"""
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (train - mean) / std, (other - mean) / std


def _fit_ridge(Xb: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ y)


@dataclass
class FoldDetail:
    fold: int
    lam: float
    oos_ic: float
    symbol_ic: Dict[int, float] = field(default_factory=dict)


@dataclass
class MatrixCellResult:
    horizon: int
    target: str
    mean_oos_ic: float
    positive_fold_ratio: float
    t_stat: float
    stability_passed: bool
    n_folds: int
    fold_details: List[FoldDetail]
    per_feature_ic: Dict[str, float]
    group_ic: Dict[str, float]
    symbol_ic_mean: Dict[int, float]

    def passed(self, threshold: float = 0.02, min_positive_ratio: float = 0.6) -> bool:
        return (
            self.mean_oos_ic >= threshold
            and self.positive_fold_ratio >= min_positive_ratio
            and self.stability_passed
        )

    def summary_line(self) -> str:
        status = "PASS" if self.passed() else "fail"
        top_group = (
            max(self.group_ic.items(), key=lambda kv: kv[1])
            if self.group_ic
            else ("-", 0.0)
        )
        return (
            f"  h={self.horizon:<3} target={self.target:<10} "
            f"IC={self.mean_oos_ic:+.4f} (+folds={self.positive_fold_ratio:.0%}) "
            f"t={self.t_stat:+.2f} {status:<4} top_group={top_group[0]}({top_group[1]:.3f})"
        )


def nested_walk_forward_ic(
    fs: FeatureSet,
    horizon: int,
    target: str = "raw",
    n_folds: int = 5,
    purge_bars: Optional[int] = None,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
    inner_val_frac: float = 0.2,
    min_t_stat: float = 1.0,
    feature_mask: Optional[np.ndarray] = None,
) -> MatrixCellResult:
    """
    horizon×target1件についてネストウォークフォワードOOS ICを計算する。

    feature_mask: Noneなら全特徴。boolマスクを渡すとその特徴だけに次元を
    落として学習・評価する（比較実験用。ゼロ埋めではなく実際に列を落とす）。
    """
    purge_bars = purge_bars if purge_bars is not None else max(24, horizon)
    X_full, y, bar_idx, sym_idx = _pool_with_symbol(fs, horizon, target)
    names = fs.feature_names
    if feature_mask is not None:
        X_full = X_full[:, feature_mask]
        names = [n for n, m in zip(names, feature_mask) if m]

    n_bars = int(bar_idx.max()) + 1
    fold_edges = np.linspace(0, n_bars, n_folds + 2).astype(int)

    oos_ics: List[float] = []
    fold_details: List[FoldDetail] = []
    symbol_ic_lists: Dict[int, List[float]] = {}

    for k in range(1, n_folds + 1):
        train_end = fold_edges[k]
        test_start = train_end + purge_bars
        test_end = fold_edges[k + 1]
        if test_start >= test_end or train_end < 200:
            continue
        train_mask = bar_idx < train_end
        test_mask = (bar_idx >= test_start) & (bar_idx < test_end)
        if train_mask.sum() < 200 or test_mask.sum() < 50:
            continue

        # --- inner split: lambda選択は外側テストに一切触れない ---
        inner_split_bar = int(train_end * (1.0 - inner_val_frac))
        inner_val_start = inner_split_bar + purge_bars
        inner_train_mask = bar_idx < inner_split_bar
        inner_val_mask = (bar_idx >= inner_val_start) & (bar_idx < train_end)

        best_lam = lambdas[len(lambdas) // 2]
        if inner_train_mask.sum() >= 150 and inner_val_mask.sum() >= 30:
            Xtr_raw, Xval_raw = X_full[inner_train_mask], X_full[inner_val_mask]
            ytr, yval = y[inner_train_mask], y[inner_val_mask]
            Xtr, Xval = _standardize(Xtr_raw, Xval_raw)
            Xtrb = np.hstack([Xtr, np.ones((len(Xtr), 1))])
            Xvalb = np.hstack([Xval, np.ones((len(Xval), 1))])
            XtX = Xtrb.T @ Xtrb
            Xty = Xtrb.T @ ytr
            best_ic = -np.inf
            for lam in lambdas:
                w = np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), Xty)
                ic = _rank_ic(Xvalb @ w, yval)
                if ic > best_ic:
                    best_ic, best_lam = ic, lam

        # --- 外側学習区間全体で選ばれたlambdaにより再学習し、外側テストで評価 ---
        Xtr_raw, Xte_raw = X_full[train_mask], X_full[test_mask]
        ytr = y[train_mask]
        Xtr, Xte = _standardize(Xtr_raw, Xte_raw)
        Xtrb = np.hstack([Xtr, np.ones((len(Xtr), 1))])
        Xteb = np.hstack([Xte, np.ones((len(Xte), 1))])
        w = _fit_ridge(Xtrb, ytr, best_lam)
        pred = Xteb @ w
        y_test = y[test_mask]
        ic = _rank_ic(pred, y_test)
        oos_ics.append(ic)

        sym_test = sym_idx[test_mask]
        fd = FoldDetail(fold=k, lam=best_lam, oos_ic=ic)
        for s in np.unique(sym_test):
            m = sym_test == s
            if m.sum() >= 20:
                sic = _rank_ic(pred[m], y_test[m])
                fd.symbol_ic[int(s)] = sic
                symbol_ic_lists.setdefault(int(s), []).append(sic)
        fold_details.append(fd)

    mean_ic, pos_ratio, t_stat, stability = _fold_ic_stats(oos_ics, min_t_stat)

    per_feature_ic = {name: _rank_ic(X_full[:, j], y) for j, name in enumerate(names)}
    groups = default_feature_groups(fs)
    group_ic: Dict[str, float] = {}
    for gname, fnames in groups.items():
        ics = [per_feature_ic[n] for n in fnames if n in per_feature_ic]
        if ics:
            group_ic[gname] = float(np.mean(np.abs(ics)))

    symbol_ic_mean = {s: float(np.mean(v)) for s, v in symbol_ic_lists.items()}

    return MatrixCellResult(
        horizon=horizon,
        target=target,
        mean_oos_ic=mean_ic,
        positive_fold_ratio=pos_ratio,
        t_stat=t_stat,
        stability_passed=stability,
        n_folds=len(oos_ics),
        fold_details=fold_details,
        per_feature_ic=per_feature_ic,
        group_ic=group_ic,
        symbol_ic_mean=symbol_ic_mean,
    )


def run_matrix(
    fs: FeatureSet,
    horizons: Sequence[int] = (1, 2, 4, 8, 12, 24, 48, 72),
    targets: Sequence[str] = ("raw", "cs_demean", "vol_norm"),
    **kwargs,
) -> List[MatrixCellResult]:
    """horizon×targetの全組み合わせでnested_walk_forward_icを実行する"""
    results = []
    for target in targets:
        for horizon in horizons:
            results.append(nested_walk_forward_ic(fs, horizon, target=target, **kwargs))
    return results
