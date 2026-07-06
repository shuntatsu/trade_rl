"""
ユーティリティモジュール

設定管理・評価指標
"""

from .config import MarsLiteConfig, default_config
from .metrics import (
    calc_differential_sharpe_ratio,
    calc_execution_metrics,
    calc_implementation_shortfall,
)

__all__ = [
    "MarsLiteConfig",
    "default_config",
    "calc_implementation_shortfall",
    "calc_differential_sharpe_ratio",
    "calc_execution_metrics",
]
