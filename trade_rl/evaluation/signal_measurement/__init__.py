"""Pure measurement tools of the signal diagnosis (Issue #1146, card SD1).

Rank and linear IC, cross-sectional demeaning and a causal beta residual, Holm and
Benjamini-Hochberg corrections, a block-bootstrap interval, a one-sided
equal-weighted cosine test of a positive mean, a feature non-stationarity profile, the signed after-cost and drift-excess payoffs and a
future-perturbation leakage probe. The package reads no dataset or file, fits no model and
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
from trade_rl.evaluation.signal_measurement.leakage import (
    LeakageProbeResult,
    affine_label_copy,
    future_perturbation_probe,
)
from trade_rl.evaluation.signal_measurement.multitest import (
    BlockMeanResult,
    bh_adjusted,
    bh_rejections,
    block_mean_interval,
    holm_adjusted,
    holm_rejections,
)
from trade_rl.evaluation.signal_measurement.payoff import (
    DailyMean,
    after_cost_payoff,
    daily_mean,
    drift_excess_payoff,
    drift_sign_payoff,
)
from trade_rl.evaluation.signal_measurement.profile import (
    NonstationarityProfile,
    PeriodProfile,
    SkippedProfile,
    nonstationarity_profile,
)
from trade_rl.evaluation.signal_measurement.significance import (
    EwcMeanTest,
    ewc_mean_test,
    ewc_terms,
    student_t_sf,
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
    "DailyMean",
    "EwcMeanTest",
    "LeakageProbeResult",
    "NonstationarityProfile",
    "PeriodProfile",
    "SignalMeasurementError",
    "SkippedProfile",
    "affine_label_copy",
    "after_cost_payoff",
    "average_ranks",
    "bh_adjusted",
    "bh_rejections",
    "block_mean_interval",
    "causal_beta_residual",
    "cross_sectional_demean",
    "daily_ic",
    "daily_mean",
    "daily_rank_ic_increment",
    "drift_excess_payoff",
    "drift_sign_payoff",
    "ewc_mean_test",
    "ewc_terms",
    "forward_log_returns",
    "future_perturbation_probe",
    "holm_adjusted",
    "holm_rejections",
    "linear_ic",
    "market_mean",
    "nonstationarity_profile",
    "rank_ic",
    "student_t_sf",
    "utc_day_ids",
]
