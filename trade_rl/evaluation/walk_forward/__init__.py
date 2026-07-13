"""Pure fold construction and outer-OOS aggregation."""

from trade_rl.evaluation.walk_forward.folds import (
    IndexRange,
    WalkForwardFold,
    build_folds,
)
from trade_rl.evaluation.walk_forward.stitching import (
    FoldOOSResult,
    StitchedOOS,
    stitch_oos,
)

__all__ = [
    "FoldOOSResult",
    "IndexRange",
    "StitchedOOS",
    "WalkForwardFold",
    "build_folds",
    "stitch_oos",
]
