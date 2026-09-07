from __future__ import annotations

import math
from enum import IntEnum


class PositionIntent(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


def target_weight_for_intent(
    intent: PositionIntent,
    *,
    gross_budget: float,
) -> float:
    if not isinstance(intent, PositionIntent):
        raise TypeError("intent must be PositionIntent")
    if not math.isfinite(gross_budget) or gross_budget <= 0.0 or gross_budget > 1.0:
        raise ValueError("gross_budget must be finite and within (0, 1]")
    return float(intent) * gross_budget
