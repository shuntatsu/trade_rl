"""Pure measurement tools of the signal diagnosis (Issue #1146, card SD1).

Rank and linear IC, cross-sectional demeaning and a causal beta residual, Holm and
Benjamini-Hochberg corrections, a block-bootstrap interval and a feature
non-stationarity profile. The package reads no dataset or file, fits no model and
owns no runner; it is verified on seeded synthetic markets only and says nothing
about real data.
"""

from trade_rl.evaluation.signal_measurement.ic import (
    DailyIc,
    SignalMeasurementError,
    average_ranks,
    daily_ic,
    daily_rank_ic_increment,
    linear_ic,
    rank_ic,
    utc_day_ids,
)
from trade_rl.evaluation.signal_measurement.multitest import (
    BlockMeanResult,
    bh_adjusted,
    bh_rejections,
    block_mean_interval,
    holm_adjusted,
    holm_rejections,
)
from trade_rl.evaluation.signal_measurement.profile import (
    NonstationarityProfile,
    PeriodProfile,
    SkippedProfile,
    nonstationarity_profile,
)
from trade_rl.evaluation.signal_measurement.targets import (
    causal_beta_residual,
    cross_sectional_demean,
    forward_log_returns,
    market_mean,
)

__all__ = [
    "BlockMeanResult",
    "DailyIc",
    "NonstationarityProfile",
    "PeriodProfile",
    "SignalMeasurementError",
    "SkippedProfile",
    "average_ranks",
    "bh_adjusted",
    "bh_rejections",
    "block_mean_interval",
    "causal_beta_residual",
    "cross_sectional_demean",
    "daily_ic",
    "daily_rank_ic_increment",
    "forward_log_returns",
    "holm_adjusted",
    "holm_rejections",
    "linear_ic",
    "market_mean",
    "nonstationarity_profile",
    "rank_ic",
    "utc_day_ids",
]
