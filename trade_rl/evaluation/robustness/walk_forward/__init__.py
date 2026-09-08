"""Pure fold construction and outer-OOS aggregation."""

from trade_rl.evaluation.robustness.walk_forward.folds import (
    IndexRange,
    WalkForwardFold,
    build_folds,
)
from trade_rl.evaluation.robustness.walk_forward.stitching import (
    FoldOOSResult,
    StitchedOOS,
    StitchMode,
    stitch_oos,
)

__all__ = [
    "FoldOOSResult",
    "IndexRange",
    "StitchedOOS",
    "StitchMode",
    "WalkForwardFold",
    "build_folds",
    "stitch_oos",
]
