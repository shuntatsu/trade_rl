"""One-shot research evaluation for the sealed mean-reversion economic gate."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields, replace
from statistics import median

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.experiments.bootstrap.mean_reversion_economic_gate_prereg import (
    canonical_mean_reversion_economic_gate_protocol,
)
from trade_rl.evaluation.metrics import evaluate_performance
from trade_rl.evaluation.replay import run_single_symbol_replay
from trade_rl.evaluation.runs.config import (
    CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.rules.mean_reversion import (
    MeanReversionIntentConfig,
    MeanReversionIntentStrategy,
)
from trade_rl.strategies.rules.mean_reversion_economic_gate import (
    MeanReversionEconomicGateConfig,
    MeanReversionEconomicGateStrategy,
)

_SCHEMA_VERSION = "mean_reversion_economic_gate_evaluation_v1"
_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_CAPACITY_CAPS = (
    0.0021629560553901974,
    0.002044685341258238,
    0.002184898995567895,
    0.0020480213652913385,
    0.002346308308284808,
)
_SPEC_VALUES: dict[str, object] = {
    "schema_version": _SCHEMA_VERSION,
    "issue_number": 549,
    "dataset_id": "6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518",
    "dataset_artifact_digest": "af481dd978db7d84cd3aa8ff4f5a35d8608ac44c755dd74f61e934105c02b6b7",
    "study_digest": "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820",
    "protocol_digest": "c4d6b6f4627dcc58160454f706afc176a37c150e4b2ed731fca929450a76328f",
    "calibration_result_digest": "df341ed67e87453a887a0889a82a1831a75843a58c7b25f0e98046af03e0814f",
    "calibration_artifact_id": 10338681607,
    "calibration_artifact_digest": "4bb76085456a2bcf4235979995798ce2b68c9373588e394b7b2adbc09a023a1e",
    "calibration_verifier_artifact_id": 10338981011,
    "calibration_verifier_artifact_digest": "1a1d8c234ad09b7d81265201b4f9942660f7def9212cfee7cb969d75b89929b6",
    "cost_gate_artifact_id": 10341675711,
    "cost_gate_artifact_digest": "249c507f7ea4c84f2af3fe6c37d2b1205a6e8bda8425bbeb2866329788ae87c0",
    "plan_metadata_artifact_id": 10341661866,
    "execution_overlay": CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY,
    "symbols": _SYMBOLS,
    "capacity_caps": _CAPACITY_CAPS,
    "signal_name": "1h__log_return_24bar",
    "signal_index": 2,
    "evaluation_start": "2023-01-01T00:00:00.000000000",
    "evaluation_stop_exclusive": "2025-01-01T00:00:00.000000000",
    "rule_entry_threshold": 0.01,
    "rule_exit_threshold": 0.0025,
    "beta_gate": -0.026993016001905932,
    "fee_rate": 0.0005,
    "taker_fee_rate": 0.0,
    "spread_rate": 0.0002,
    "one_way_explicit_cost": 0.0007,
    "gross_budget": 0.5,
    "initial_capital": 100_000.0,
    "promote_min_positive_effect_symbols": 4,
    "promote_min_cost_reduction_symbols": 4,
    "promote_min_turnover_reduction_symbols": 4,
    "promote_min_drawdown_nonworse_symbols": 4,
    "reject_max_positive_effect_symbols": 2,
    "reject_max_cost_reduction_symbols": 2,
    "reject_max_turnover_reduction_symbols": 2,
    "reject_max_drawdown_nonworse_symbols": 2,
    "production_eligible": False,
    "final_test_authorized": False,
    "shared_cash_profitability_established": False,
    "live_trading_authorized": False,
}


@dataclass(frozen=True, slots=True)
class MeanReversionEconomicGateEvaluationSpec:
    """Exact result-blind authorities for the one-shot research evaluation."""

    issue_number: int
    dataset_id: str
    dataset_artifact_digest: str
    study_digest: str
    protocol_digest: str
    calibration_result_digest: str
    calibration_artifact_id: int
    calibration_artifact_digest: str
    calibration_verifier_artifact_id: int
    calibration_verifier_artifact_digest: str
    cost_gate_artifact_id: int
    cost_gate_artifact_digest: str
    plan_metadata_artifact_id: int
    execution_overlay: str
    symbols: tuple[str, ...]
    capacity_caps: tuple[float, ...]
    signal_name: str
    signal_index: int
    evaluation_start: str
    evaluation_stop_exclusive: str
    rule_entry_threshold: float
    rule_exit_threshold: float
    beta_gate: float
    fee_rate: float
    taker_fee_rate: float
    spread_rate: float
    one_way_explicit_cost: float
    gross_budget: float
    initial_capital: float
    promote_min_positive_effect_symbols: int
    promote_min_cost_reduction_symbols: int
    promote_min_turnover_reduction_symbols: int
    promote_min_drawdown_nonworse_symbols: int
    reject_max_positive_effect_symbols: int
    reject_max_cost_reduction_symbols: int
    reject_max_turnover_reduction_symbols: int
    reject_max_drawdown_nonworse_symbols: int
    production_eligible: bool
    final_test_authorized: bool
    shared_cash_profitability_established: bool
    live_trading_authorized: bool
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        actual = {item.name: getattr(self, item.name) for item in fields(self)}
        if actual != _SPEC_VALUES:
            raise ValueError("frozen evaluation specification drifted")

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            payload[item.name] = list(value) if isinstance(value, tuple) else value
        return payload

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class MeanReversionEconomicGateSymbolResult:
    symbol: str
    baseline_total_return: float
    candidate_total_return: float
    excess_total_return: float
    baseline_total_cost: float
    candidate_total_cost: float
    baseline_turnover_total: float
    candidate_turnover_total: float
    baseline_max_drawdown: float
    candidate_max_drawdown: float
    baseline_termination_count: int
    candidate_termination_count: int
    baseline_n_periods: int
    candidate_n_periods: int
    baseline_return_sha256: str
    candidate_return_sha256: str

    def to_payload(self) -> dict[str, object]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(frozen=True, slots=True)
class MeanReversionEconomicGateEvaluation:
    spec_digest: str
    dataset_id: str
    by_symbol: tuple[MeanReversionEconomicGateSymbolResult, ...]
    positive_effect_symbols: int
    median_excess_total_return: float
    cost_reduction_symbols: int
    turnover_reduction_symbols: int
    drawdown_nonworse_symbols: int
    new_termination_symbols: int
    candidate_positive_total_return_symbols: int
    research_status: str
    production_eligible: bool = False
    final_test_authorized: bool = False
    shared_cash_profitability_established: bool = False
    live_trading_authorized: bool = False
    schema_version: str = _SCHEMA_VERSION

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "spec_digest": self.spec_digest,
            "dataset_id": self.dataset_id,
            "by_symbol": [item.to_payload() for item in self.by_symbol],
            "positive_effect_symbols": self.positive_effect_symbols,
            "median_excess_total_return": self.median_excess_total_return,
            "cost_reduction_symbols": self.cost_reduction_symbols,
            "turnover_reduction_symbols": self.turnover_reduction_symbols,
            "drawdown_nonworse_symbols": self.drawdown_nonworse_symbols,
            "new_termination_symbols": self.new_termination_symbols,
            "candidate_positive_total_return_symbols": (
                self.candidate_positive_total_return_symbols
            ),
            "research_status": self.research_status,
            "production_eligible": self.production_eligible,
            "final_test_authorized": self.final_test_authorized,
            "shared_cash_profitability_established": (
                self.shared_cash_profitability_established
            ),
            "live_trading_authorized": self.live_trading_authorized,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def canonical_mean_reversion_economic_gate_evaluation_spec() -> (
    MeanReversionEconomicGateEvaluationSpec
):
    protocol = canonical_mean_reversion_economic_gate_protocol()
    if protocol.digest != _SPEC_VALUES["protocol_digest"]:
        raise ValueError("sealed protocol digest differs from evaluation authority")
    return MeanReversionEconomicGateEvaluationSpec(**_SPEC_VALUES)  # type: ignore[arg-type]


def research_status_from_counts(
    *,
    positive_effect_symbols: int,
    median_excess_total_return: float,
    cost_reduction_symbols: int,
    turnover_reduction_symbols: int,
    drawdown_nonworse_symbols: int,
    new_termination_symbols: int,
) -> str:
    """Replay the preregistered research-only decision rule."""

    spec = canonical_mean_reversion_economic_gate_evaluation_spec()
    if not math.isfinite(median_excess_total_return):
        raise ValueError("median_excess_total_return must be finite")
    counts = (
        positive_effect_symbols,
        cost_reduction_symbols,
        turnover_reduction_symbols,
        drawdown_nonworse_symbols,
        new_termination_symbols,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts
    ):
        raise ValueError("research decision counts must be non-negative integers")

    promote = (
        positive_effect_symbols >= spec.promote_min_positive_effect_symbols
        and median_excess_total_return > 0.0
        and cost_reduction_symbols >= spec.promote_min_cost_reduction_symbols
        and turnover_reduction_symbols >= spec.promote_min_turnover_reduction_symbols
        and drawdown_nonworse_symbols >= spec.promote_min_drawdown_nonworse_symbols
        and new_termination_symbols == 0
    )
    if promote:
        return "PROMOTE_RESEARCH_REFERENCE"
    reject = (
        positive_effect_symbols <= spec.reject_max_positive_effect_symbols
        or median_excess_total_return <= 0.0
        or cost_reduction_symbols <= spec.reject_max_cost_reduction_symbols
        or turnover_reduction_symbols <= spec.reject_max_turnover_reduction_symbols
        or drawdown_nonworse_symbols <= spec.reject_max_drawdown_nonworse_symbols
        or new_termination_symbols > 0
    )
    return "REJECT_MECHANISM" if reject else "INCONCLUSIVE"


def _exact_index(dataset: MarketDataset, timestamp: str, *, field: str) -> int:
    target = np.datetime64(timestamp, "ns")
    values = np.asarray(dataset.timestamps, dtype="datetime64[ns]")
    matches = np.flatnonzero(values == target)
    if matches.size != 1:
        raise ValueError(f"{field} must exactly match one Dataset timestamp")
    return int(matches[0])


def _validate_dataset(
    dataset: MarketDataset, spec: MeanReversionEconomicGateEvaluationSpec
) -> tuple[int, int]:
    if dataset.dataset_id != spec.dataset_id:
        raise ValueError("dataset identity differs from frozen evaluation")
    if tuple(dataset.symbols) != spec.symbols:
        raise ValueError("dataset symbol roster differs from frozen evaluation")
    if not 0 <= spec.signal_index < len(dataset.feature_names):
        raise ValueError("signal index is outside Dataset feature roster")
    if dataset.feature_names[spec.signal_index] != spec.signal_name:
        raise ValueError("signal identity differs from frozen evaluation")

    start = _exact_index(dataset, spec.evaluation_start, field="evaluation_start")
    stop = _exact_index(
        dataset,
        spec.evaluation_stop_exclusive,
        field="evaluation_stop_exclusive",
    )
    if not 0 <= start < stop < dataset.n_bars:
        raise ValueError("evaluation range differs from frozen evaluation")

    execution_slice = slice(start, stop + 1)
    expected_shape = (stop + 1 - start, len(spec.symbols))
    for field, expected in (
        ("fee_rate", spec.fee_rate),
        ("taker_fee_rate", spec.taker_fee_rate),
        ("spread_rate", spec.spread_rate),
    ):
        values = np.asarray(
            dataset.resolved_array(field)[execution_slice], dtype=np.float64
        )
        if values.shape != expected_shape or not np.all(values == expected):
            raise ValueError(f"evaluation cost drift: {field}")

    participation = np.asarray(
        dataset.resolved_array("max_participation_rate")[execution_slice],
        dtype=np.float64,
    )
    expected_caps = np.broadcast_to(np.asarray(spec.capacity_caps), expected_shape)
    if participation.shape != expected_shape or not np.array_equal(
        participation, expected_caps
    ):
        raise ValueError("evaluation capacity drift")
    return start, stop


def _return_sha256(values: tuple[float, ...]) -> str:
    array = np.asarray(values, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def evaluate_mean_reversion_economic_gate(
    dataset: MarketDataset,
    spec: MeanReversionEconomicGateEvaluationSpec,
) -> MeanReversionEconomicGateEvaluation:
    """Evaluate baseline and gated mean-reversion on one identical replay contract."""

    canonical = canonical_mean_reversion_economic_gate_evaluation_spec()
    if spec != canonical or spec.digest != canonical.digest:
        raise ValueError("evaluation spec differs from frozen evaluation")
    start, stop = _validate_dataset(dataset, spec)

    proposal = MeanReversionIntentConfig(
        signal_index=spec.signal_index,
        entry_threshold=spec.rule_entry_threshold,
        exit_threshold=spec.rule_exit_threshold,
    )
    gate_config = MeanReversionEconomicGateConfig(
        proposal=proposal,
        beta_gate=spec.beta_gate,
        one_way_explicit_cost=spec.one_way_explicit_cost,
    )
    execution_cost = replace(
        ExecutionCostConfig.zero(),
        processing_bar_volume_capacity=False,
    )

    results: list[MeanReversionEconomicGateSymbolResult] = []
    for symbol_index, symbol in enumerate(spec.symbols):
        baseline_replay = run_single_symbol_replay(
            dataset,
            MeanReversionIntentStrategy(proposal),
            symbol_index=symbol_index,
            start_index=start,
            stop_index=stop,
            gross_budget=spec.gross_budget,
            initial_capital=spec.initial_capital,
            execution_cost=execution_cost,
            risk=None,
        )
        candidate_replay = run_single_symbol_replay(
            dataset,
            MeanReversionEconomicGateStrategy(gate_config),
            symbol_index=symbol_index,
            start_index=start,
            stop_index=stop,
            gross_budget=spec.gross_budget,
            initial_capital=spec.initial_capital,
            execution_cost=execution_cost,
            risk=None,
        )
        baseline_diagnostics = baseline_replay.diagnostics
        candidate_diagnostics = candidate_replay.diagnostics
        baseline = evaluate_performance(
            baseline_replay.returns,
            turnover_total=baseline_diagnostics.turnover_total,
            total_cost=baseline_diagnostics.total_cost,
            funding_pnl=baseline_diagnostics.funding_pnl,
            borrow_cost=baseline_diagnostics.borrow_cost,
            n_trades=baseline_diagnostics.n_trades,
            rebalance_events=baseline_diagnostics.rebalance_events,
            termination_count=len(baseline_diagnostics.termination_reasons),
        )
        candidate = evaluate_performance(
            candidate_replay.returns,
            turnover_total=candidate_diagnostics.turnover_total,
            total_cost=candidate_diagnostics.total_cost,
            funding_pnl=candidate_diagnostics.funding_pnl,
            borrow_cost=candidate_diagnostics.borrow_cost,
            n_trades=candidate_diagnostics.n_trades,
            rebalance_events=candidate_diagnostics.rebalance_events,
            termination_count=len(candidate_diagnostics.termination_reasons),
        )
        results.append(
            MeanReversionEconomicGateSymbolResult(
                symbol=symbol,
                baseline_total_return=baseline.total_return,
                candidate_total_return=candidate.total_return,
                excess_total_return=candidate.total_return - baseline.total_return,
                baseline_total_cost=baseline.total_cost,
                candidate_total_cost=candidate.total_cost,
                baseline_turnover_total=baseline.turnover_total,
                candidate_turnover_total=candidate.turnover_total,
                baseline_max_drawdown=baseline.max_drawdown,
                candidate_max_drawdown=candidate.max_drawdown,
                baseline_termination_count=baseline.termination_count,
                candidate_termination_count=candidate.termination_count,
                baseline_n_periods=baseline.n_periods,
                candidate_n_periods=candidate.n_periods,
                baseline_return_sha256=_return_sha256(baseline_replay.returns.values),
                candidate_return_sha256=_return_sha256(candidate_replay.returns.values),
            )
        )

    if tuple(item.symbol for item in results) != spec.symbols:
        raise RuntimeError("paired evaluation output symbol roster drifted")
    if any(item.baseline_n_periods != item.candidate_n_periods for item in results):
        raise RuntimeError("paired evaluation return horizon drifted")

    positive = sum(item.excess_total_return > 0.0 for item in results)
    excess_median = float(median(item.excess_total_return for item in results))
    cost_reduction = sum(
        item.candidate_total_cost < item.baseline_total_cost for item in results
    )
    turnover_reduction = sum(
        item.candidate_turnover_total < item.baseline_turnover_total for item in results
    )
    drawdown_nonworse = sum(
        item.candidate_max_drawdown <= item.baseline_max_drawdown for item in results
    )
    new_termination = sum(
        item.candidate_termination_count > item.baseline_termination_count
        for item in results
    )
    candidate_positive = sum(item.candidate_total_return > 0.0 for item in results)
    status = research_status_from_counts(
        positive_effect_symbols=positive,
        median_excess_total_return=excess_median,
        cost_reduction_symbols=cost_reduction,
        turnover_reduction_symbols=turnover_reduction,
        drawdown_nonworse_symbols=drawdown_nonworse,
        new_termination_symbols=new_termination,
    )
    return MeanReversionEconomicGateEvaluation(
        spec_digest=spec.digest,
        dataset_id=dataset.dataset_id,
        by_symbol=tuple(results),
        positive_effect_symbols=positive,
        median_excess_total_return=excess_median,
        cost_reduction_symbols=cost_reduction,
        turnover_reduction_symbols=turnover_reduction,
        drawdown_nonworse_symbols=drawdown_nonworse,
        new_termination_symbols=new_termination,
        candidate_positive_total_return_symbols=candidate_positive,
        research_status=status,
    )


__all__ = [
    "MeanReversionEconomicGateEvaluation",
    "MeanReversionEconomicGateEvaluationSpec",
    "MeanReversionEconomicGateSymbolResult",
    "canonical_mean_reversion_economic_gate_evaluation_spec",
    "evaluate_mean_reversion_economic_gate",
    "research_status_from_counts",
]
