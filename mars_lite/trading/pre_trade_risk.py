"""
注文前リスク制御モジュール
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


class PreTradeRejection(Exception):
    """
    注文前リスクチェックに引っかかった場合にスローされる例外
    """

    def __init__(self, reason: str, details: dict):
        super().__init__(f"PreTradeRejection: {reason} - {details}")
        self.reason = reason
        self.details = details


@dataclass
class PreTradeRiskConfig:
    """
    注文前リスク制御の設定パラメータを保持するデータクラス
    """

    max_leverage: float | None = None
    max_single_weight: float | None = None
    max_position_pct: float | None = None
    max_notional: float | None = None
    forbidden_symbols: set[str] | None = None


class PreTradeRiskVerifier:
    """
    目標ウェイトおよびポートフォリオ価値を検証し、制限超過時に PreTradeRejection をスローするクラス
    """

    def __init__(self, config: PreTradeRiskConfig):
        self.config = config

    def validate(
        self,
        target_weights: np.ndarray,
        portfolio_value: float,
        symbols: Iterable[str] | None = None,
    ) -> None:
        w = np.asarray(target_weights, dtype=np.float64)
        gross_leverage = float(np.abs(w).sum())

        if self.config.forbidden_symbols and symbols is not None:
            self._validate_forbidden_symbols(w, list(symbols))

        if self.config.max_leverage is not None:
            if gross_leverage > self.config.max_leverage:
                raise PreTradeRejection(
                    reason="leverage_limit_exceeded",
                    details={
                        "gross_leverage": gross_leverage,
                        "max_leverage": self.config.max_leverage,
                    },
                )

        if self.config.max_single_weight is not None:
            max_single = float(np.abs(w).max()) if len(w) > 0 else 0.0
            if max_single > self.config.max_single_weight:
                raise PreTradeRejection(
                    reason="single_weight_limit_exceeded",
                    details={
                        "max_single_weight_found": max_single,
                        "max_single_weight_allowed": self.config.max_single_weight,
                    },
                )

        if self.config.max_position_pct is not None:
            max_position_pct = float(np.abs(w).max()) if len(w) > 0 else 0.0
            if max_position_pct > self.config.max_position_pct:
                raise PreTradeRejection(
                    reason="position_pct_limit_exceeded",
                    details={
                        "max_position_pct_found": max_position_pct,
                        "max_position_pct_allowed": self.config.max_position_pct,
                    },
                )

        if self.config.max_notional is not None:
            total_notional = portfolio_value * gross_leverage
            if total_notional > self.config.max_notional:
                raise PreTradeRejection(
                    reason="notional_limit_exceeded",
                    details={
                        "total_notional": total_notional,
                        "max_notional": self.config.max_notional,
                    },
                )

    def _validate_forbidden_symbols(
        self, weights: np.ndarray, symbols: list[str]
    ) -> None:
        if len(symbols) != len(weights):
            raise ValueError("symbols length must match target_weights length")
        forbidden = self.config.forbidden_symbols or set()
        for symbol, weight in zip(symbols, weights):
            if symbol in forbidden and abs(float(weight)) > 0:
                raise PreTradeRejection(
                    reason="forbidden_symbol",
                    details={"symbol": symbol, "target_weight": float(weight)},
                )
