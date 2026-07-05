"""
ボラティリティ推定量

OHLCベースのレンジ推定量（クローズのみのstdより情報効率が高い）。
"""

import numpy as np
import pandas as pd


def calc_garman_klass(df: pd.DataFrame) -> pd.Series:
    """
    Garman-Klass推定量（バーごとのボラティリティ、対数レンジベース）

    sigma^2 = 0.5*ln(H/L)^2 - (2*ln2 - 1)*ln(C/O)^2
    """
    hl = np.log(df["high"] / df["low"].replace(0, np.nan)) ** 2
    co = np.log(df["close"] / df["open"].replace(0, np.nan)) ** 2
    var = 0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co
    return np.sqrt(var.clip(lower=0)).fillna(0.0)


def calc_parkinson(df: pd.DataFrame) -> pd.Series:
    """Parkinson推定量（高値安値レンジのみ使用）"""
    hl = np.log(df["high"] / df["low"].replace(0, np.nan)) ** 2
    var = hl / (4.0 * np.log(2.0))
    return np.sqrt(var.clip(lower=0)).fillna(0.0)
