"""Action-space ablation contracts used by sealed OOS comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from trade_rl.rl.actions import ActionSpec


class ActionAblation(str, Enum):
    BASELINE_ONLY = "A0_baseline_only"
    TREND_LEGACY = "A1_trend_legacy"
    TREND_ALPHA_LEGACY = "A2_trend_alpha_legacy"
    FACTORIZED = "A3_factorized_four_control"
    FACTORIZED_4 = "A4_factorized_plus_4"
    FACTORIZED_8 = "A5_factorized_plus_8"
    DIRECT_SYMBOL = "A6_direct_symbol_residual"


@dataclass(frozen=True, slots=True)
class ActionExperimentSpec:
    ablation: ActionAblation | str
    n_symbols: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "ablation", ActionAblation(self.ablation))
        if self.n_symbols <= 0:
            raise ValueError("n_symbols must be positive")

    @property
    def policy_enabled(self) -> bool:
        return self.ablation is not ActionAblation.BASELINE_ONLY

    @property
    def accept_legacy_actions(self) -> bool:
        return self.ablation in {
            ActionAblation.TREND_LEGACY,
            ActionAblation.TREND_ALPHA_LEGACY,
        }

    @property
    def action_spec(self) -> ActionSpec:
        alpha = self.ablation not in {
            ActionAblation.BASELINE_ONLY,
            ActionAblation.TREND_LEGACY,
        }
        factors = {
            ActionAblation.FACTORIZED_4: 4,
            ActionAblation.FACTORIZED_8: 8,
            ActionAblation.DIRECT_SYMBOL: self.n_symbols,
        }.get(self.ablation, 0)
        return ActionSpec(alpha_enabled=alpha, n_factors=factors)

    def direct_symbol_basis(self) -> np.ndarray | None:
        if self.ablation is not ActionAblation.DIRECT_SYMBOL:
            return None
        return np.eye(self.n_symbols, dtype=np.float64)
