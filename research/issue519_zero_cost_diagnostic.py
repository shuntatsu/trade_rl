"""Read-only zero-cost diagnostic for verified Issue #519 development evidence.

This module is intentionally outside the Study state machine. It does not define,
run, verify, compare, or decide a Controlled Experiment. It loads an already
verified result artifact, constructs a new in-memory diagnostic Dataset identity
with only execution/financing cost arrays zeroed, and replays the frozen
deterministic mean-reversion rules. The output is development-only diagnostic
evidence and must not be presented as canonical Study evidence, final-test
evidence, or production profitability.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np

from trade_rl.data import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)
from trade_rl.data.market import MarketDataset
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import BookState, MarketExecutor
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.interface import SingleSymbolStrategy, StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent, target_weight_for_intent
from trade_rl.strategies.rules.mean_reversion import (
    MeanReversionIntentConfig,
    MeanReversionIntentStrategy,
)

EXPECTED_SOURCE_DATASET_ID = (
    "d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f"
)
EXPECTED_SOURCE_DATASET_ARTIFACT_DIGEST = (
    "77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8"
)
EXPECTED_EXP002_CANDIDATE_FINGERPRINT = (
    "f5ec53197a2f487ba11cc01a78edd4c93df8a2f172095e5903ee079ce5e9d0d0"
)
ZEROED_ECONOMIC_ARRAYS = (
    "fee_rate",
    "maker_fee_rate",
    "taker_fee_rate",
    "spread_rate",
    "funding_rate",
    "borrow_rate",
    "cash_rate",
)
PRESERVED_EXECUTION_ARRAYS = (
    "max_participation_rate",
    "minimum_notional",
    "lot_size",
    "tick_size",
    "borrow_available",
    "funding_due",
    "buy_allowed",
    "sell_allowed",
    "mark_price",
    "index_price",
    "contract_multipliers",
)
VARIANTS = (
    ("baseline", "study/baseline/evidence/runs/seed-0/summary.json"),
    (
        "exp1_entry_threshold_0_02",
        "study/experiments/0001/candidate/evidence/runs/seed-0/summary.json",
    ),
    (
        "exp2_daily_signal",
        "study/experiments/0002/candidate/evidence/runs/seed-0/summary.json",
    ),
)
_TOLERANCE = 1e-12


def make_zero_cost_diagnostic_dataset(source: MarketDataset) -> MarketDataset:
    """Return a new identity whose cost/financing arrays are exactly zero.

    Market prices, features, causal availability, tradability, liquidity capacity,
    order constraints, contract multipliers, and all other arrays remain identical.
    """

    price_shape = (source.n_bars, source.n_symbols)
    zero_price = np.zeros(price_shape, dtype=np.float64)
    zero_cash = np.zeros(source.n_bars, dtype=np.float64)
    diagnostic = replace(
        source,
        dataset_id="0" * 64,
        identity_payload_json=None,
        fee_rate=zero_price,
        maker_fee_rate=zero_price,
        taker_fee_rate=zero_price,
        spread_rate=zero_price,
        funding_rate=zero_price,
        borrow_rate=zero_price,
        cash_rate=zero_cash,
    ).with_content_identity(
        {
            "diagnostic_only": True,
            "diagnostic_schema": "zero_cost_counterfactual_v2",
            "source_dataset_id": source.dataset_id,
            "zeroed_arrays": list(ZEROED_ECONOMIC_ARRAYS),
        }
    )
    _validate_diagnostic_dataset(source, diagnostic)
    return diagnostic


def _validate_diagnostic_dataset(
    source: MarketDataset,
    diagnostic: MarketDataset,
) -> None:
    if diagnostic.dataset_id == source.dataset_id:
        raise RuntimeError("zero-cost diagnostic Dataset identity did not change")
    if not diagnostic.identity_verified:
        raise RuntimeError("zero-cost diagnostic Dataset identity is not verified")
    for field in ZEROED_ECONOMIC_ARRAYS:
        if np.any(diagnostic.resolved_array(field) != 0.0):
            raise RuntimeError(f"zero-cost diagnostic did not zero {field}")
    diagnostic_arrays = diagnostic.identity_arrays()
    for field, source_array in source.identity_arrays().items():
        if field in ZEROED_ECONOMIC_ARRAYS:
            continue
        if not np.array_equal(diagnostic_arrays[field], source_array):
            raise RuntimeError(f"zero-cost diagnostic changed non-cost array: {field}")
    for field in PRESERVED_EXECUTION_ARRAYS:
        if not np.array_equal(
            diagnostic.resolved_array(field),
            source.resolved_array(field),
        ):
            raise RuntimeError(f"zero-cost diagnostic changed execution constraint: {field}")


def _intent_label(value: object) -> str:
    if isinstance(value, PositionIntent):
        return value.name
    label = str(value)
    if label not in {"LONG", "SHORT", "FLAT"}:
        raise ValueError(f"unsupported intent label: {label}")
    return label


def direction_log_return_attribution(
    returns: np.ndarray,
    intents: Iterable[object],
) -> dict[str, object]:
    """Partition aligned log returns by supplied intent labels.

    This generic helper is exact only when every supplied return is already aligned
    to the supplied intent. Canonical next-open replay uses the segmented tracer
    below because an interval contains a prior-intent overnight gap plus a
    post-open segment after the new target is processed.
    """

    values = np.asarray(returns, dtype=np.float64).reshape(-1)
    labels = tuple(_intent_label(value) for value in intents)
    if values.size != len(labels):
        raise ValueError("returns and intents must have the same length")
    if not np.isfinite(values).all() or np.any(values <= -1.0):
        raise ValueError("returns must be finite and greater than -1")
    logs = np.log1p(values)
    output: dict[str, object] = {}
    total = 0.0
    for label, prefix in (("LONG", "long"), ("SHORT", "short"), ("FLAT", "flat")):
        mask = np.asarray([item == label for item in labels], dtype=np.bool_)
        contribution = float(logs[mask].sum())
        output[f"{prefix}_log_return_contribution"] = contribution
        output[f"{prefix}_intervals"] = int(mask.sum())
        total += contribution
    output["total_log_return"] = float(logs.sum())
    if not math.isclose(total, float(output["total_log_return"]), rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("direction attribution does not reconstruct total log return")
    return output


def _desired_quantity_from_weight(
    book: BookState,
    target_weight: float,
    *,
    symbol_index: int,
) -> float:
    multiplier = float(book.contract_multipliers[symbol_index])
    denominator = float(book.mark_prices[symbol_index]) * multiplier
    return float(target_weight * book.portfolio_value / denominator)


def _weight_for_desired_quantity(
    book: BookState,
    desired_quantity: float,
    *,
    symbol_index: int,
) -> float:
    if book.portfolio_value <= 0.0:
        return 0.0
    multiplier = float(book.contract_multipliers[symbol_index])
    return float(
        desired_quantity
        * float(book.mark_prices[symbol_index])
        * multiplier
        / book.portfolio_value
    )


def _diagnostic_risk(executor: MarketExecutor) -> PreTradeRisk:
    hard_limit = min(1.0, float(executor.cost.max_leverage))
    return PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=hard_limit,
            max_abs_weight=hard_limit,
            max_turnover=None,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        )
    )


def _strategy_observation(
    dataset: MarketDataset,
    *,
    index: int,
    symbol_index: int,
    book: BookState,
    current_intent: PositionIntent,
) -> StrategyObservation:
    return StrategyObservation(
        index=index,
        timestamp=dataset.timestamps[index],
        symbol=dataset.symbols[symbol_index],
        features=dataset.features[index, symbol_index],
        feature_available=dataset.feature_available[index, symbol_index],
        feature_staleness=dataset.resolved_array("feature_staleness")[
            index, symbol_index
        ],
        global_features=dataset.global_features[index],
        global_feature_available=dataset.resolved_array("global_feature_available")[
            index
        ],
        current_intent=current_intent,
        current_weight=float(book.weights[symbol_index]),
    )


def _add_intent_log(
    totals: dict[str, float],
    intent: PositionIntent,
    value: float,
) -> None:
    totals[_intent_label(intent)] += value


def run_segmented_zero_cost_replay(
    dataset: MarketDataset,
    strategy: SingleSymbolStrategy,
    *,
    symbol_index: int,
    start_index: int,
    stop_index: int,
    gross_budget: float,
    initial_capital: float,
) -> dict[str, object]:
    """Replay zero-cost next-open execution and attribute temporal return segments.

    The close-to-next-open gap is carried by the pre-decision holding and is
    therefore attributed to the prior intent. The residual interval log return is
    attributed to the newly requested intent after next-open processing. The two
    segments exactly reconstruct the canonical interval net log return. Corporate
    actions and financing are rejected here rather than silently misattributed.
    """

    if not 0 <= symbol_index < dataset.n_symbols:
        raise ValueError("symbol_index is outside the Dataset")
    if not 0 <= start_index < stop_index < dataset.n_bars:
        raise ValueError("replay range must satisfy 0 <= start < stop < n_bars")
    if not math.isfinite(initial_capital) or initial_capital <= 0.0:
        raise ValueError("initial_capital must be finite and positive")
    target_weight_for_intent(PositionIntent.LONG, gross_budget=gross_budget)
    for field in ZEROED_ECONOMIC_ARRAYS:
        if np.any(dataset.resolved_array(field) != 0.0):
            raise RuntimeError(f"segmented replay requires zeroed {field}")

    initial_prices = dataset.resolved_array("mark_price")[start_index]
    book = BookState.zero(
        dataset.n_symbols,
        initial_capital,
        initial_prices,
        contract_multipliers=dataset.contract_multipliers,
    )
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    risk = _diagnostic_risk(executor)
    current_intent = PositionIntent.FLAT
    desired_quantity = 0.0
    returns: list[float] = []
    contributions = {"LONG": 0.0, "SHORT": 0.0, "FLAT": 0.0}
    gap_segments = 0
    post_open_segments = 0
    partial_fill_intervals = 0
    index = start_index

    while index < stop_index:
        observation = _strategy_observation(
            dataset,
            index=index,
            symbol_index=symbol_index,
            book=book,
            current_intent=current_intent,
        )
        intent = strategy.decide(observation)
        if not isinstance(intent, PositionIntent):
            raise TypeError("strategy.decide must return PositionIntent")
        changed_intent = intent is not current_intent
        if changed_intent:
            proposal_weight = target_weight_for_intent(
                intent,
                gross_budget=gross_budget,
            )
            desired_quantity = _desired_quantity_from_weight(
                book,
                proposal_weight,
                symbol_index=symbol_index,
            )
        proposal_weight = _weight_for_desired_quantity(
            book,
            desired_quantity,
            symbol_index=symbol_index,
        )
        proposal_weights = np.zeros(dataset.n_symbols, dtype=np.float64)
        proposal_weights[symbol_index] = proposal_weight
        constrained = risk.constrain(
            proposal_weights,
            current=book.weights,
            drawdown=book.max_drawdown,
        )
        target_weight = float(constrained.weights[symbol_index])
        if constrained.was_constrained and any(
            reason != "max_turnover" for reason in constrained.reasons
        ):
            desired_quantity = _desired_quantity_from_weight(
                book,
                target_weight,
                symbol_index=symbol_index,
            )

        processing_index = index + 1
        if float(dataset.resolved_array("split_factor")[processing_index, symbol_index]) != 1.0:
            raise RuntimeError("segmented diagnostic does not attribute split bars")
        if not bool(dataset.resolved_array("asset_active")[processing_index, symbol_index]):
            raise RuntimeError("segmented diagnostic does not attribute inactive-asset bars")
        if float(dataset.resolved_array("dividend")[processing_index, symbol_index]) != 0.0:
            raise RuntimeError("segmented diagnostic does not attribute dividend bars")

        period_start_value = max(book.portfolio_value, _TOLERANCE)
        gap_book = book.clone()
        gap_book.revalue(dataset.open[processing_index])
        gap_return = max(gap_book.portfolio_value, 0.0) / period_start_value - 1.0
        if not math.isfinite(gap_return) or gap_return <= -1.0:
            raise RuntimeError("segmented diagnostic gap return is invalid")

        execution = executor.execute_interval(
            book,
            constrained.weights,
            start_index=index,
            bars=1,
        )
        if execution.next_index != processing_index:
            raise RuntimeError("segmented execution did not advance exactly one bar")
        if execution.interval_cost != 0.0:
            raise RuntimeError("segmented zero-cost replay produced transaction cost")
        if execution.interval_funding != 0.0 or execution.interval_borrow_cost != 0.0:
            raise RuntimeError("segmented zero-cost replay produced financing cost")
        if execution.interval_cash_interest != 0.0 or execution.interval_dividend != 0.0:
            raise RuntimeError("segmented zero-cost replay produced carry outside attribution")
        if execution.termination_reason is not None:
            raise RuntimeError("segmented diagnostic does not attribute terminated intervals")

        interval_return = float(execution.interval_net_return)
        if not math.isfinite(interval_return) or interval_return <= -1.0:
            raise RuntimeError("segmented diagnostic interval return is invalid")
        gap_log = math.log1p(gap_return)
        post_open_log = math.log1p(interval_return) - gap_log
        _add_intent_log(contributions, current_intent, gap_log)
        _add_intent_log(contributions, intent, post_open_log)
        gap_segments += 1
        post_open_segments += 1
        if execution.fill_ratio < 1.0 - _TOLERANCE:
            partial_fill_intervals += 1

        returns.append(interval_return)
        book = execution.book
        current_intent = intent
        index = execution.next_index

    return_values = np.asarray(returns, dtype=np.float64)
    total_log = float(np.log1p(return_values).sum())
    attributed_log = sum(contributions.values())
    if not math.isclose(total_log, attributed_log, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("segmented intent attribution does not reconstruct total log return")
    final_log = math.log(max(book.portfolio_value, _TOLERANCE) / initial_capital)
    if not math.isclose(total_log, final_log, rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError("segmented replay return path differs from final book value")

    return {
        "returns": returns,
        "final_portfolio_value": book.portfolio_value,
        "turnover_total": book.turnover_total,
        "n_trades": book.n_trades,
        "rebalance_events": book.rebalance_events,
        "partial_fill_intervals": partial_fill_intervals,
        "exact_intent_log_attribution": {
            "long_log_return_contribution": contributions["LONG"],
            "short_log_return_contribution": contributions["SHORT"],
            "flat_log_return_contribution": contributions["FLAT"],
            "total_log_return": total_log,
            "gap_segments": gap_segments,
            "post_open_segments": post_open_segments,
            "partial_fill_intervals": partial_fill_intervals,
            "timing_semantics": (
                "close_to_next_open_gap_is_prior_intent; post_open_residual_is_new_intent"
            ),
        },
    }


def _load_json(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"JSON object expected: {path}")
    return raw


def _mean_reversion_entry(summary: dict[str, object], symbol_index: int) -> dict[str, object]:
    by_symbol = summary.get("by_symbol")
    if not isinstance(by_symbol, list) or not 0 <= symbol_index < len(by_symbol):
        raise RuntimeError("candidate summary symbol evidence malformed")
    symbol = by_symbol[symbol_index]
    if not isinstance(symbol, dict):
        raise RuntimeError("candidate summary symbol entry malformed")
    strategies = symbol.get("strategies")
    if not isinstance(strategies, list):
        raise RuntimeError("candidate summary strategy roster malformed")
    matches = [
        entry
        for entry in strategies
        if isinstance(entry, dict) and entry.get("name") == "mean_reversion"
    ]
    if len(matches) != 1:
        raise RuntimeError("candidate summary mean_reversion entry missing or duplicated")
    return matches[0]


def _number(mapping: object, field: str) -> float:
    if not isinstance(mapping, dict):
        raise RuntimeError(f"mapping malformed for {field}")
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"numeric field malformed: {field}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"numeric field non-finite: {field}")
    return result


def _integer(mapping: object, field: str) -> int:
    if not isinstance(mapping, dict):
        raise RuntimeError(f"mapping malformed for {field}")
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"integer field malformed: {field}")
    return value


def _evaluation_indices(
    dataset: MarketDataset,
    summary: dict[str, object],
) -> tuple[int, int, float, float]:
    evaluation = summary.get("evaluation")
    if not isinstance(evaluation, dict):
        raise RuntimeError("candidate evaluation metadata malformed")
    start_raw = evaluation.get("start")
    stop_raw = evaluation.get("stop_exclusive")
    if not isinstance(start_raw, str) or not isinstance(stop_raw, str):
        raise RuntimeError("candidate evaluation timestamps malformed")
    start_time = np.datetime64(start_raw, "ns")
    stop_time = np.datetime64(stop_raw, "ns")
    start = int(np.searchsorted(dataset.timestamps, start_time, side="left"))
    stop = int(np.searchsorted(dataset.timestamps, stop_time, side="left"))
    if not 0 <= start < stop < dataset.n_bars:
        raise RuntimeError("candidate evaluation range does not map to source Dataset")
    if dataset.timestamps[start] != start_time or dataset.timestamps[stop] != stop_time:
        raise RuntimeError("candidate evaluation timestamps are absent from source Dataset")
    gross_budget = _number(evaluation, "gross_budget")
    initial_capital = _number(evaluation, "initial_capital")
    return start, stop, gross_budget, initial_capital


def _strategy_from_summary(summary: dict[str, object]) -> MeanReversionIntentStrategy:
    config = summary.get("candidate_config")
    if not isinstance(config, dict):
        raise RuntimeError("candidate_config malformed")
    signal_index = config.get("signal_index")
    if isinstance(signal_index, bool) or not isinstance(signal_index, int) or signal_index < 0:
        raise RuntimeError("candidate signal_index malformed")
    return MeanReversionIntentStrategy(
        MeanReversionIntentConfig(
            signal_index=signal_index,
            entry_threshold=_number(config, "rule_entry_threshold"),
            exit_threshold=_number(config, "rule_exit_threshold"),
        )
    )


def _zero_cost_variant(
    dataset: MarketDataset,
    summary: dict[str, object],
) -> dict[str, object]:
    start, stop, gross_budget, initial_capital = _evaluation_indices(dataset, summary)
    symbols = summary.get("symbols")
    if symbols != list(dataset.symbols):
        raise RuntimeError("candidate summary symbol roster differs from Dataset")
    results: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(dataset.symbols):
        nominal = _mean_reversion_entry(summary, symbol_index)
        nominal_metrics = nominal.get("metrics")
        strategy = _strategy_from_summary(summary)
        replay = run_segmented_zero_cost_replay(
            dataset,
            strategy,
            symbol_index=symbol_index,
            start_index=start,
            stop_index=stop,
            gross_budget=gross_budget,
            initial_capital=initial_capital,
        )
        returns = np.asarray(replay["returns"], dtype=np.float64)
        total_return = float(replay["final_portfolio_value"]) / initial_capital - 1.0
        compounded = float(np.prod(1.0 + returns, dtype=np.float64) - 1.0)
        if not math.isclose(total_return, compounded, rel_tol=1e-10, abs_tol=1e-10):
            raise RuntimeError("zero-cost replay book return differs from interval compounding")
        results.append(
            {
                "symbol": symbol,
                "zero_cost_total_return": total_return,
                "zero_cost_final_portfolio_value": replay["final_portfolio_value"],
                "zero_cost_turnover_total": replay["turnover_total"],
                "zero_cost_n_trades": replay["n_trades"],
                "zero_cost_rebalance_events": replay["rebalance_events"],
                "zero_cost_total_cost": 0.0,
                "zero_cost_funding_pnl": 0.0,
                "zero_cost_borrow_cost": 0.0,
                "exact_next_open_intent_attribution": replay[
                    "exact_intent_log_attribution"
                ],
                "nominal_total_return": _number(nominal_metrics, "total_return"),
                "nominal_turnover_total": _number(nominal_metrics, "turnover_total"),
                "nominal_total_cost": _number(nominal_metrics, "total_cost"),
                "nominal_n_trades": _integer(nominal_metrics, "n_trades"),
                "zero_cost_minus_nominal_total_return": total_return
                - _number(nominal_metrics, "total_return"),
            }
        )
    totals = [float(entry["zero_cost_total_return"]) for entry in results]
    positive = sum(value > 0.0 for value in totals)
    median_return = float(median(totals))
    if positive >= 4 and median_return > 0.0:
        classification = "GROSS_EDGE_PRESENT_COSTS_MATERIAL"
    elif positive <= 2 or median_return <= 0.0:
        classification = "GROSS_EDGE_INSUFFICIENT"
    else:
        classification = "MIXED_GROSS_EDGE"
    long_positive = sum(
        float(
            entry["exact_next_open_intent_attribution"][  # type: ignore[index]
                "long_log_return_contribution"
            ]
        )
        > 0.0
        for entry in results
    )
    short_positive = sum(
        float(
            entry["exact_next_open_intent_attribution"][  # type: ignore[index]
                "short_log_return_contribution"
            ]
        )
        > 0.0
        for entry in results
    )
    return {
        "symbols": results,
        "aggregate": {
            "positive_zero_cost_symbol_count": positive,
            "negative_zero_cost_symbol_count": sum(value < 0.0 for value in totals),
            "zero_zero_cost_symbol_count": sum(value == 0.0 for value in totals),
            "median_zero_cost_total_return": median_return,
            "worst_zero_cost_total_return": min(totals),
            "best_zero_cost_total_return": max(totals),
            "long_positive_exact_intent_contribution_symbol_count": long_positive,
            "short_positive_exact_intent_contribution_symbol_count": short_positive,
            "root_cause_classification": classification,
        },
    }


def run_diagnostic(result_root: Path, output_root: Path) -> dict[str, object]:
    """Run the read-only zero-cost diagnostic on one verified recovered result."""

    result_index = _load_json(result_root / "portable-exp002-result-index.json")
    if result_index.get("candidate_evidence_fingerprint") != EXPECTED_EXP002_CANDIDATE_FINGERPRINT:
        raise RuntimeError("source Experiment 0002 candidate fingerprint drift")
    if result_index.get("candidate_execution_count") != 1 or result_index.get("candidate_rerun") is not False:
        raise RuntimeError("source Experiment 0002 execution-count contract drift")

    published = inspect_published_market_dataset_artifact(result_root / "dataset")
    source = load_market_dataset_artifact(result_root / "dataset")
    if source.dataset_id != EXPECTED_SOURCE_DATASET_ID:
        raise RuntimeError("source Dataset ID drift")
    if published.artifact_digest != EXPECTED_SOURCE_DATASET_ARTIFACT_DIGEST:
        raise RuntimeError("source Dataset artifact digest drift")
    if not source.identity_verified:
        raise RuntimeError("source Dataset identity is not verified")
    diagnostic = make_zero_cost_diagnostic_dataset(source)

    summaries: dict[str, dict[str, object]] = {}
    for name, relative in VARIANTS:
        summaries[name] = _load_json(result_root / relative)
    evaluation_contract = [
        _evaluation_indices(source, summaries[name]) for name, _ in VARIANTS
    ]
    if len(set(evaluation_contract)) != 1:
        raise RuntimeError("baseline/Exp1/Exp2 evaluation contracts differ")

    variants = {
        name: _zero_cost_variant(diagnostic, summaries[name]) for name, _ in VARIANTS
    }
    exp2_aggregate = variants["exp2_daily_signal"]["aggregate"]
    if not isinstance(exp2_aggregate, dict):
        raise RuntimeError("Exp2 diagnostic aggregate malformed")
    report: dict[str, object] = {
        "schema_version": "canonical_m2_zero_cost_root_cause_diagnostic_v2",
        "diagnostic_only": True,
        "not_study_evidence": True,
        "not_final_test_evidence": True,
        "not_production_authorization": True,
        "source_result_candidate_fingerprint": EXPECTED_EXP002_CANDIDATE_FINGERPRINT,
        "source_dataset_id": source.dataset_id,
        "source_dataset_artifact_digest": published.artifact_digest,
        "diagnostic_dataset_id": diagnostic.dataset_id,
        "zeroed_economic_arrays": list(ZEROED_ECONOMIC_ARRAYS),
        "preserved_execution_arrays": list(PRESERVED_EXECUTION_ARRAYS),
        "variants": variants,
        "exp2_root_cause_classification": exp2_aggregate.get("root_cause_classification"),
        "intent_attribution_semantics": (
            "next-open exact temporal split: close-to-next-open gap is prior intent; "
            "remaining interval log return after next-open processing is new intent"
        ),
        "interpretation_note": (
            "Zero-cost is a counterfactual development diagnostic. It removes dataset-authoritative "
            "transaction/financing costs while preserving market path, causal features, liquidity "
            "capacity, order constraints, hard risk, and strategy parameters."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "zero-cost-diagnostic.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    report = run_diagnostic(args.result_root, args.output_root)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "direction_log_return_attribution",
    "make_zero_cost_diagnostic_dataset",
    "run_diagnostic",
    "run_segmented_zero_cost_replay",
]
