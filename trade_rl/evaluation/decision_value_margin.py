"""Pre-registered, train-internal decision-margin selection (#1025 TD8).

The decision margin multiplier ``k`` scales the round-trip cost fraction of a
candidate. It is chosen from a fixed grid using *only* the train window:

1. Order the matured branch labels by decision time; the first 75% (by branch
   time) fit a temporary value model.
2. The last 25% form a calibration slice. A closed-loop native replay of the
   temporary model with each grid multiplier runs on that slice, and the
   multiplier with the best after-cost net return wins. Exact ties go to the
   larger multiplier.

When supplied with the raw source, the temporary fit has its own fit-cutoff
normalizer and the calibration replay uses that frozen vintage. The calibration
replay is also checked by the native source-backed ledger oracle. No evaluation-
block data enters this margin choice.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TypeVar

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.decision_value_feature_transform import (
    apply_feature_transform,
    fit_train_feature_transform,
)
from trade_rl.evaluation.decision_value_ledger_oracle import (
    verify_paired_economic_ledger,
)
from trade_rl.evaluation.paired_portfolio_label import (
    SyntheticPairedLabel,
    SyntheticPairedLabelBatch,
    build_synthetic_paired_label_batch,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.portfolio import PortfolioPolicy

_ModelT = TypeVar("_ModelT")


@dataclass(frozen=True, slots=True)
class MarginSelection:
    """Chosen multiplier and the train-internal evidence behind it."""

    multiplier: float
    net_returns: tuple[tuple[float, float], ...]
    n_fit_branches: int
    n_calibration_branches: int
    calibration_start_index: int
    calibration_stop_index: int
    fit_normalizer_digest: str | None = None

    def to_mapping(self) -> dict[str, object]:
        values = {
            "multiplier": self.multiplier,
            "net_returns": [[k, r] for k, r in self.net_returns],
            "n_fit_branches": self.n_fit_branches,
            "n_calibration_branches": self.n_calibration_branches,
            "calibration_start_index": self.calibration_start_index,
            "calibration_stop_index": self.calibration_stop_index,
        }
        if self.fit_normalizer_digest is not None:
            values["fit_normalizer_digest"] = self.fit_normalizer_digest
        return values


def _validate_grid(grid: tuple[float, ...]) -> tuple[float, ...]:
    if type(grid) is not tuple or not grid:
        raise ValueError("margin grid must be a non-empty tuple")
    values: list[float] = []
    for item in grid:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("margin grid must contain real numbers")
        value = float(item)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("margin grid must contain finite non-negative values")
        values.append(value)
    if any(b <= a for a, b in zip(values, values[1:], strict=False)):
        raise ValueError("margin grid must be strictly increasing")
    return tuple(values)


def select_margin_multiplier(
    *,
    dataset: MarketDataset,
    samples: tuple[SyntheticPairedLabel, ...],
    fit_model: Callable[[SyntheticPairedLabelBatch], _ModelT],
    make_policy: Callable[[_ModelT, float, MarketDataset], PortfolioPolicy],
    grid: tuple[float, ...],
    calibration_fraction: float,
    horizon_bars: int,
    initial_capital: float,
    execution_cost: ExecutionCostConfig,
    risk_config: PreTradeRiskConfig,
    reference_policy_digest: str,
    raw_dataset: MarketDataset | None = None,
    train_start_index: int | None = None,
    feature_clip: float | None = None,
    require_v4_evidence: bool = False,
) -> MarginSelection:
    """Select the margin multiplier using time-ordered train-internal data only."""
    multipliers = _validate_grid(grid)
    if (
        isinstance(calibration_fraction, bool)
        or not isinstance(calibration_fraction, (int, float))
        or not 0.0 < float(calibration_fraction) < 1.0
    ):
        raise ValueError("calibration fraction must lie strictly inside (0, 1)")
    if type(horizon_bars) is not int or horizon_bars <= 0:
        raise ValueError("horizon_bars must be a positive native integer")
    if type(samples) is not tuple or not samples:
        raise ValueError("margin selection requires a non-empty set of labels")

    times = sorted({s.decision_time_ns for s in samples})
    n_fit = int(math.floor((1.0 - float(calibration_fraction)) * len(times)))
    if n_fit < 1 or len(times) - n_fit < 1:
        raise ValueError("labels cannot be split into fit and calibration branches")
    fit_times = set(times[:n_fit])
    timestamps_ns = dataset.timestamps.astype(np.int64)
    last_fit_index = int(np.searchsorted(timestamps_ns, times[n_fit - 1]))
    last_cal_index = int(np.searchsorted(timestamps_ns, times[-1]))
    # The last fit label matures at ``last_fit_index + horizon``; one embargo bar
    # keeps its maturity strictly before the temporary fit cutoff.
    cutoff_index = last_fit_index + horizon_bars + 1
    cal_start_index = cutoff_index + 1
    cal_stop_index = last_cal_index + horizon_bars
    if cal_start_index >= cal_stop_index or cal_stop_index >= dataset.n_bars:
        raise ValueError("calibration slice is empty or exceeds the dataset")

    fit_cutoff = dataset.timestamps[cutoff_index]
    fit_cutoff_ns = int(timestamps_ns[cutoff_index])
    # The temporary fit has an earlier cutoff than the final block fit. It
    # must have its own normalizer; fitting on the full block would leak the
    # held-out calibration features into every training row.
    calibration_view = dataset
    transform_digest: str | None = None
    if raw_dataset is not None:
        if (
            not isinstance(raw_dataset, MarketDataset)
            or type(train_start_index) is not int
            or feature_clip is None
            or raw_dataset.dataset_id != dataset.dataset_id
            or raw_dataset.feature_config_digest != dataset.feature_config_digest
            or raw_dataset.symbols != dataset.symbols
            or raw_dataset.feature_names != dataset.feature_names
            or raw_dataset.global_feature_names != dataset.global_feature_names
            or not np.array_equal(raw_dataset.timestamps, dataset.timestamps)
        ):
            raise ValueError("raw calibration source/clock/feature context mismatch")
        transform = fit_train_feature_transform(
            raw_dataset,
            train_start_index=train_start_index,
            train_stop_index=cutoff_index - 1,
            as_of_index=cutoff_index,
            clip=feature_clip,
        )
        calibration_view = apply_feature_transform(raw_dataset, transform)
        transform_digest = transform.digest
    elif train_start_index is not None or feature_clip is not None:
        raise ValueError("calibration vintage requires a raw dataset")

    subset = []
    for sample in samples:
        if sample.decision_time_ns not in fit_times:
            continue
        if raw_dataset is None:
            subset.append(replace(sample, fit_cutoff_ns=fit_cutoff_ns))
            continue
        index = int(np.searchsorted(timestamps_ns, sample.decision_time_ns))
        if (
            index >= len(timestamps_ns)
            or timestamps_ns[index] != sample.decision_time_ns
        ):
            raise ValueError("fit sample does not belong to calibration dataset clock")
        observation = replace(
            sample.observation,
            features=tuple(
                tuple(float(v) for v in row) for row in calibration_view.features[index]
            ),
            global_features=tuple(
                float(v) for v in calibration_view.global_features[index]
            ),
            feature_available=tuple(
                tuple(bool(v) for v in row)
                for row in calibration_view.feature_available[index]
            ),
            global_feature_available=tuple(
                bool(v)
                for v in calibration_view.resolved_array("global_feature_available")[
                    index
                ]
            ),
        )
        subset.append(
            replace(
                sample,
                observation=observation,
                normalization_digest=calibration_view.normalization_digest,
                fit_cutoff_ns=fit_cutoff_ns,
            )
        )
    batch = build_synthetic_paired_label_batch(
        tuple(subset),
        fit_cutoff=fit_cutoff,
        evaluation_start=dataset.timestamps[cal_start_index],
        expected_reference_policy_digest=reference_policy_digest,
        require_v4_evidence=require_v4_evidence,
    )
    model = fit_model(batch)

    results: list[tuple[float, float]] = []
    best_multiplier = multipliers[0]
    best_return = -math.inf
    for k in multipliers:
        replay = run_shared_portfolio_replay(
            calibration_view,
            make_policy(model, k, calibration_view),
            start_index=cal_start_index,
            stop_index=cal_stop_index,
            initial_capital=initial_capital,
            execution_cost=execution_cost,
            risk=PreTradeRisk(risk_config),
            settle_terminal_position=True,
            capture_ledger_evidence=True,
        )
        if not verify_paired_economic_ledger(
            replay,
            calibration_view,
            expected_intervals=cal_stop_index - cal_start_index,
            initial_capital=initial_capital,
            expected_start_index=cal_start_index,
            execution_cost=execution_cost,
        ):
            raise ValueError("calibration replay failed the independent ledger oracle")
        net = float(replay.book.portfolio_value) / float(initial_capital) - 1.0
        results.append((k, net))
        if net >= best_return:  # exact ties go to the larger multiplier
            best_return = net
            best_multiplier = k
    return MarginSelection(
        multiplier=best_multiplier,
        net_returns=tuple(results),
        n_fit_branches=n_fit,
        n_calibration_branches=len(times) - n_fit,
        calibration_start_index=cal_start_index,
        calibration_stop_index=cal_stop_index,
        fit_normalizer_digest=transform_digest,
    )


__all__ = ["MarginSelection", "select_margin_multiplier"]
