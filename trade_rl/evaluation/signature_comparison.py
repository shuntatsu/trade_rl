"""Development-only, result-blind Ridge controls for opt-in Signature features.

This is not a study selector, frozen evidence writer, independent approval,
or authorization to inspect unused/final data. The caller must preregister
the Dataset, exact split, feature names, budget and financial assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from trade_rl.artifacts import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import (
    SingleSymbolReplayResult,
    run_single_symbol_replay,
)
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.forecasts.ridge import (
    RidgeForecastStrategy,
    fit_ridge_forecast,
)
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent


@dataclass(frozen=True, slots=True)
class SignatureRidgeComparison:
    """Paired diagnostic replays with shared capital/costs/decision range."""

    source_dataset_id: str
    augmented_dataset_id: str
    baseline: SingleSymbolReplayResult
    signature: SingleSymbolReplayResult
    cash: SingleSymbolReplayResult


class _CashStrategy:
    def decide(self, observation: StrategyObservation) -> PositionIntent:
        return PositionIntent.FLAT


def validate_signature_pair(
    original: MarketDataset,
    augmented: MarketDataset,
    *,
    baseline_feature_names: tuple[str, ...],
    signature_feature_names: tuple[str, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Never silently compare different price, fee, risk, time or symbol scopes."""
    if (
        not original.identity_verified
        or not augmented.identity_verified
        or original.dataset_id == augmented.dataset_id
        or original.symbols != augmented.symbols
        or not np.array_equal(original.timestamps, augmented.timestamps)
        or augmented.feature_names[: original.n_features] != original.feature_names
        or not np.array_equal(
            original.features, augmented.features[:, :, : original.n_features]
        )
        or not np.array_equal(
            original.feature_available,
            augmented.feature_available[:, :, : original.n_features],
        )
    ):
        raise ValueError(
            "Signature pair must share an identical validated base Dataset"
        )
    for field in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "funding_rate",
        "tradable",
        "fee_rate",
        "maker_fee_rate",
        "taker_fee_rate",
        "spread_rate",
        "max_participation_rate",
        "minimum_notional",
        "lot_size",
        "tick_size",
        "borrow_rate",
        "funding_due",
        "mark_price",
        "index_price",
        "available_at",
        "information_available",
        "symbol_active",
    ):
        if not np.array_equal(
            original.resolved_array(field), augmented.resolved_array(field)
        ):
            raise ValueError(f"Signature augmentation changed economic input: {field}")
    if not baseline_feature_names or not signature_feature_names:
        raise ValueError("baseline and Signature feature rosters must be nonempty")
    if (
        len(set(baseline_feature_names)) != len(baseline_feature_names)
        or len(set(signature_feature_names)) != len(signature_feature_names)
        or set(baseline_feature_names) & set(signature_feature_names)
    ):
        raise ValueError(
            "Signature comparison feature rosters must be unique and disjoint"
        )
    if any(
        not (name.startswith("mt_path_sig_v1_") or name.startswith("path_sig_v1_"))
        for name in signature_feature_names
    ):
        raise ValueError(
            "Signature features must come from an explicit Signature schema"
        )
    try:
        baseline = tuple(
            original.feature_names.index(name) for name in baseline_feature_names
        )
        extended = tuple(
            augmented.feature_names.index(name)
            for name in (*baseline_feature_names, *signature_feature_names)
        )
    except ValueError as error:
        raise ValueError(
            "Signature comparison feature is missing from Dataset"
        ) from error
    return baseline, extended


def _matched_baseline_fit_dataset(
    original: MarketDataset,
    augmented: MarketDataset,
    signature_indices: tuple[int, ...],
) -> MarketDataset:
    """Give both regressions the same eligible training rows.

    A long Signature warmup or a missing native source must not silently change
    the baseline's effective training sample. This is a *transient fit-only*
    dataset; evaluation uses the unmodified original source Dataset.
    """
    signature_ready = np.all(
        augmented.feature_available[:, :, list(signature_indices)], axis=2
    )
    fit_mask = signature_ready[:, :, None]
    available = original.feature_available & fit_mask
    staleness = np.where(
        fit_mask, original.resolved_array("feature_staleness"), 1.0
    ).astype(np.float32)
    original_reasons = original.resolved_array("feature_missing_reason")
    missing_reason = np.where(
        fit_mask, original_reasons, np.ones_like(original_reasons)
    )
    definition: dict[str, object] = {
        "schema": "signature_comparison_matched_fit_v1",
        "base_dataset_id": original.dataset_id,
        "augmented_dataset_id": augmented.dataset_id,
        "signature_feature_names": [
            augmented.feature_names[index] for index in signature_indices
        ],
        "selection": "all_selected_signature_features_available",
    }
    return replace(
        original,
        identity_payload_json=None,
        feature_available=available,
        feature_staleness=staleness,
        feature_missing_reason=missing_reason,
        feature_config_digest=content_digest(definition),
    ).with_content_identity(definition)


def run_ridge_signature_comparison(
    original: MarketDataset,
    augmented: MarketDataset,
    *,
    baseline_feature_names: tuple[str, ...],
    signature_feature_names: tuple[str, ...],
    fit_cutoff: np.datetime64,
    evaluation_start_index: int,
    evaluation_stop_index: int,
    symbol_index: int,
    horizon_hours: int,
    alpha: float,
    entry_threshold: float,
    exit_threshold: float,
    gross_budget: float,
    initial_capital: float,
    execution_cost: ExecutionCostConfig,
    risk_config: PreTradeRiskConfig | None = None,
    one_way_switch_cost: float | None = None,
) -> SignatureRidgeComparison:
    """Execute paired Ridge/Cash diagnostics; never select a winner.

    The research owner, not this helper, owns unused data authorization and
    significance/selection across many hypotheses or PPO seeds.
    """
    baseline_indices, extended_indices = validate_signature_pair(
        original,
        augmented,
        baseline_feature_names=baseline_feature_names,
        signature_feature_names=signature_feature_names,
    )
    if (
        isinstance(evaluation_start_index, bool)
        or isinstance(evaluation_stop_index, bool)
        or not 0 <= evaluation_start_index < evaluation_stop_index < original.n_bars
    ):
        raise ValueError("Signature comparison evaluation range is invalid")
    if not 0 <= symbol_index < original.n_symbols:
        raise ValueError("Signature comparison symbol is invalid")
    cutoff = np.datetime64(fit_cutoff, "ns")
    if cutoff > original.timestamps[evaluation_start_index]:
        raise ValueError("Signature comparison training reaches into evaluation")
    fit_baseline = _matched_baseline_fit_dataset(
        original, augmented, extended_indices[len(baseline_indices) :]
    )
    baseline_model = fit_ridge_forecast(
        fit_baseline,
        feature_indices=baseline_indices,
        fit_cutoff=cutoff,
        horizon_hours=horizon_hours,
        alpha=alpha,
    )
    signature_model = fit_ridge_forecast(
        augmented,
        feature_indices=extended_indices,
        fit_cutoff=cutoff,
        horizon_hours=horizon_hours,
        alpha=alpha,
    )
    arms = (
        (
            original,
            RidgeForecastStrategy(
                baseline_model,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                one_way_switch_cost=one_way_switch_cost,
            ),
        ),
        (
            augmented,
            RidgeForecastStrategy(
                signature_model,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                one_way_switch_cost=one_way_switch_cost,
            ),
        ),
        (original, _CashStrategy()),
    )
    replays: list[SingleSymbolReplayResult] = []
    for dataset, strategy in arms:
        replays.append(
            run_single_symbol_replay(
                dataset,
                strategy,
                symbol_index=symbol_index,
                start_index=evaluation_start_index,
                stop_index=evaluation_stop_index,
                gross_budget=gross_budget,
                initial_capital=initial_capital,
                execution_cost=execution_cost,
                risk=None if risk_config is None else PreTradeRisk(risk_config),
                settle_terminal_position=True,
            )
        )
    return SignatureRidgeComparison(
        source_dataset_id=original.dataset_id,
        augmented_dataset_id=augmented.dataset_id,
        baseline=replays[0],
        signature=replays[1],
        cash=replays[2],
    )


__all__ = [
    "SignatureRidgeComparison",
    "run_ridge_signature_comparison",
    "validate_signature_pair",
]
