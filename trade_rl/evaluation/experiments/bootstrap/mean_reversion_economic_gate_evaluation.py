"""One-shot research evaluation for the sealed mean-reversion economic gate."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, fields
from statistics import median

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.comparison.strategies import compare_strategies_by_symbol
from trade_rl.evaluation.experiments.bootstrap.mean_reversion_economic_gate_prereg import (
    canonical_mean_reversion_economic_gate_protocol,
)
from trade_rl.evaluation.runs import execution_cost_for_overlay
from trade_rl.strategies.rules.mean_reversion import (
    MeanReversionIntentConfig,
    MeanReversionIntentStrategy,
)
from trade_rl.strategies.rules.mean_reversion_economic_gate import (
    MeanReversionEconomicGateConfig,
    MeanReversionEconomicGateStrategy,
)

_SCHEMA_VERSION = "mean_reversion_economic_gate_evaluation_v1"
_RESULT_SCHEMA_VERSION = "mean_reversion_economic_gate_evaluation_result_v1"
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
    "protocol_digest": "c4d6b6f4627dcc58160454f706afc176a37c150e4b2ed731fca929450a76328f",
    "prereg_seal_digest": "ce0e56e57366e12b7a0b89327083b4acf90d91f28254767fedd2c61142591dbd",
    "prereg_seal_artifact_id": 10337873013,
    "prereg_seal_artifact_digest": "c92a9bf230a7a348d3461fa713809b2aa7487b2058f90cd520e405b6b4a64eed",
    "calibration_result_digest": "df341ed67e87453a887a0889a82a1831a75843a58c7b25f0e98046af03e0814f",
    "calibration_artifact_id": 10338681607,
    "calibration_artifact_digest": "4bb76085456a2bcf4235979995798ce2b68c9373588e394b7b2adbc09a023a1e",
    "calibration_verifier_artifact_id": 10338981011,
    "calibration_verifier_artifact_digest": "1a1d8c234ad09b7d81265201b4f9942660f7def9212cfee7cb969d75b89929b6",
    "cost_gate_run_id": 34828815819,
    "cost_gate_artifact_id": 10341675711,
    "cost_gate_artifact_digest": "249c507f7ea4c84f2af3fe6c37d2b1205a6e8bda8425bbeb2866329788ae87c0",
    "plan_metadata_run_id": 34829589372,
    "plan_metadata_artifact_id": 10341661866,
    "plan_metadata_artifact_digest": "0734edf3283b84848f63de3fe0e16153564cfc9731d620d76ba9f762b90945db",
    "successor_bundle_run_id": 34803217815,
    "successor_bundle_artifact_id": 10331899302,
    "successor_bundle_artifact_digest": "89e899427f23fa46929c8be1e71fd49abe0d1d465c7a7f796a0874426b885bce",
    "dataset_id": "6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518",
    "dataset_artifact_digest": "af481dd978db7d84cd3aa8ff4f5a35d8608ac44c755dd74f61e934105c02b6b7",
    "study_digest": "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820",
    "execution_overlay": "zero_overlay_dataset_fields_authoritative_previous_completed_bar_capacity",
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


def _strict_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _json_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class MeanReversionEconomicGateEvaluationSpec:
    """Exact result-blind authorities for the one-shot research evaluation."""

    schema_version: str
    issue_number: int
    protocol_digest: str
    prereg_seal_digest: str
    prereg_seal_artifact_id: int
    prereg_seal_artifact_digest: str
    calibration_result_digest: str
    calibration_artifact_id: int
    calibration_artifact_digest: str
    calibration_verifier_artifact_id: int
    calibration_verifier_artifact_digest: str
    cost_gate_run_id: int
    cost_gate_artifact_id: int
    cost_gate_artifact_digest: str
    plan_metadata_run_id: int
    plan_metadata_artifact_id: int
    plan_metadata_artifact_digest: str
    successor_bundle_run_id: int
    successor_bundle_artifact_id: int
    successor_bundle_artifact_digest: str
    dataset_id: str
    dataset_artifact_digest: str
    study_digest: str
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

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            expected = _SPEC_VALUES[item.name]
            if not _strict_equal(value, expected):
                raise ValueError(
                    f"frozen evaluation field {item.name} does not match preregistration"
                )

    def to_payload(self) -> dict[str, object]:
        return {
            item.name: _json_value(getattr(self, item.name)) for item in fields(self)
        }

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
    baseline_termination_reasons: tuple[str, ...]
    candidate_termination_reasons: tuple[str, ...]
    new_termination: bool
    baseline_n_periods: int
    candidate_n_periods: int
    baseline_return_sha256: str
    candidate_return_sha256: str

    def __post_init__(self) -> None:
        numeric = (
            self.baseline_total_return,
            self.candidate_total_return,
            self.excess_total_return,
            self.baseline_total_cost,
            self.candidate_total_cost,
            self.baseline_turnover_total,
            self.candidate_turnover_total,
            self.baseline_max_drawdown,
            self.candidate_max_drawdown,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("symbol evaluation metrics must be finite")
        for value in (
            self.baseline_termination_count,
            self.candidate_termination_count,
            self.baseline_n_periods,
            self.candidate_n_periods,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "symbol evaluation counts must be non-negative integers"
                )
        for digest in (self.baseline_return_sha256, self.candidate_return_sha256):
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError("return digest must be lowercase SHA-256")
        if type(self.new_termination) is not bool:
            raise ValueError("new_termination must be a bool")

    def to_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "baseline_total_return": self.baseline_total_return,
            "candidate_total_return": self.candidate_total_return,
            "excess_total_return": self.excess_total_return,
            "baseline_total_cost": self.baseline_total_cost,
            "candidate_total_cost": self.candidate_total_cost,
            "baseline_turnover_total": self.baseline_turnover_total,
            "candidate_turnover_total": self.candidate_turnover_total,
            "baseline_max_drawdown": self.baseline_max_drawdown,
            "candidate_max_drawdown": self.candidate_max_drawdown,
            "baseline_termination_count": self.baseline_termination_count,
            "candidate_termination_count": self.candidate_termination_count,
            "baseline_termination_reasons": list(self.baseline_termination_reasons),
            "candidate_termination_reasons": list(self.candidate_termination_reasons),
            "new_termination": self.new_termination,
            "baseline_n_periods": self.baseline_n_periods,
            "candidate_n_periods": self.candidate_n_periods,
            "baseline_return_sha256": self.baseline_return_sha256,
            "candidate_return_sha256": self.candidate_return_sha256,
        }


@dataclass(frozen=True, slots=True)
class MeanReversionEconomicGateEvaluation:
    spec_digest: str
    dataset_id: str
    symbols: tuple[str, ...]
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
    schema_version: str = _RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported economic-gate evaluation result schema")
        if (
            self.symbols != _SYMBOLS
            or tuple(item.symbol for item in self.by_symbol) != self.symbols
        ):
            raise ValueError("economic-gate evaluation result symbol roster mismatch")
        if self.research_status not in {
            "PROMOTE_RESEARCH_REFERENCE",
            "REJECT_MECHANISM",
            "INCONCLUSIVE",
        }:
            raise ValueError("unsupported economic-gate research status")
        if not math.isfinite(self.median_excess_total_return):
            raise ValueError("median_excess_total_return must be finite")
        for value in (
            self.positive_effect_symbols,
            self.cost_reduction_symbols,
            self.turnover_reduction_symbols,
            self.drawdown_nonworse_symbols,
            self.new_termination_symbols,
            self.candidate_positive_total_return_symbols,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 5
            ):
                raise ValueError(
                    "evaluation symbol counts must be integers within [0, 5]"
                )
        for value in (
            self.production_eligible,
            self.final_test_authorized,
            self.shared_cash_profitability_established,
            self.live_trading_authorized,
        ):
            if type(value) is not bool or value:
                raise ValueError(
                    "research evaluation cannot authorize production or final test"
                )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "spec_digest": self.spec_digest,
            "dataset_id": self.dataset_id,
            "symbols": list(self.symbols),
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
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= len(spec.symbols)
        for value in counts
    ):
        raise ValueError("research decision counts must be integers within [0, 5]")

    reject = (
        positive_effect_symbols <= spec.reject_max_positive_effect_symbols
        or median_excess_total_return <= 0.0
        or cost_reduction_symbols <= spec.reject_max_cost_reduction_symbols
        or turnover_reduction_symbols <= spec.reject_max_turnover_reduction_symbols
        or drawdown_nonworse_symbols <= spec.reject_max_drawdown_nonworse_symbols
        or new_termination_symbols > 0
    )
    if reject:
        return "REJECT_MECHANISM"
    promote = (
        positive_effect_symbols >= spec.promote_min_positive_effect_symbols
        and median_excess_total_return > 0.0
        and cost_reduction_symbols >= spec.promote_min_cost_reduction_symbols
        and turnover_reduction_symbols >= spec.promote_min_turnover_reduction_symbols
        and drawdown_nonworse_symbols >= spec.promote_min_drawdown_nonworse_symbols
        and new_termination_symbols == 0
    )
    return "PROMOTE_RESEARCH_REFERENCE" if promote else "INCONCLUSIVE"


def _exact_index(dataset: MarketDataset, timestamp: str, *, field: str) -> int:
    target = np.datetime64(timestamp, "ns")
    values = np.asarray(dataset.timestamps, dtype="datetime64[ns]")
    matches = np.flatnonzero(values == target)
    if matches.size != 1:
        raise ValueError(f"{field} must exactly match one Dataset timestamp")
    return int(matches[0])


def _validate_dataset(
    dataset: MarketDataset,
    spec: MeanReversionEconomicGateEvaluationSpec,
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
    for field_name, expected in (
        ("fee_rate", spec.fee_rate),
        ("taker_fee_rate", spec.taker_fee_rate),
        ("spread_rate", spec.spread_rate),
    ):
        values = np.asarray(
            dataset.resolved_array(field_name)[execution_slice], dtype=np.float64
        )
        if (
            values.shape != expected_shape
            or not np.isfinite(values).all()
            or np.any(values < 0.0)
            or not np.all(values == expected)
        ):
            raise ValueError(f"evaluation cost drift: {field_name}")

    participation = np.asarray(
        dataset.resolved_array("max_participation_rate")[execution_slice],
        dtype=np.float64,
    )
    expected_caps = np.broadcast_to(
        np.asarray(spec.capacity_caps, dtype=np.float64), expected_shape
    )
    if (
        participation.shape != expected_shape
        or not np.isfinite(participation).all()
        or not np.array_equal(participation, expected_caps)
    ):
        raise ValueError("evaluation capacity drift")
    return start, stop


def evaluation_return_sha256(values: object) -> str:
    """Stable semantic digest for one one-dimensional float64 return series."""

    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("evaluation returns must be finite and one-dimensional")
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _has_new_termination(
    baseline_reasons: tuple[str, ...],
    candidate_reasons: tuple[str, ...],
    *,
    baseline_count: int,
    candidate_count: int,
) -> bool:
    if candidate_count > baseline_count:
        return True
    baseline_counter = Counter(baseline_reasons)
    candidate_counter = Counter(candidate_reasons)
    return any(
        candidate_counter[reason] > baseline_counter[reason]
        for reason in candidate_counter
    )


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
    strategies = {
        "baseline": MeanReversionIntentStrategy(proposal),
        "candidate": MeanReversionEconomicGateStrategy(
            MeanReversionEconomicGateConfig(
                proposal=proposal,
                beta_gate=spec.beta_gate,
                one_way_explicit_cost=spec.one_way_explicit_cost,
            )
        ),
    }
    comparison = compare_strategies_by_symbol(
        dataset,
        strategies,
        start_index=start,
        stop_index=stop,
        gross_budget=spec.gross_budget,
        initial_capital=spec.initial_capital,
        execution_cost=execution_cost_for_overlay(spec.execution_overlay),
        risk=None,
    )

    results: list[MeanReversionEconomicGateSymbolResult] = []
    for symbol_result in comparison.by_symbol:
        entries = {entry.name: entry for entry in symbol_result.comparison.entries}
        if set(entries) != {"baseline", "candidate"}:
            raise RuntimeError("paired evaluation strategy roster drifted")
        baseline = entries["baseline"]
        candidate = entries["candidate"]
        baseline_reasons = tuple(baseline.replay.diagnostics.termination_reasons)
        candidate_reasons = tuple(candidate.replay.diagnostics.termination_reasons)
        is_new_termination = _has_new_termination(
            baseline_reasons,
            candidate_reasons,
            baseline_count=baseline.metrics.termination_count,
            candidate_count=candidate.metrics.termination_count,
        )
        results.append(
            MeanReversionEconomicGateSymbolResult(
                symbol=symbol_result.symbol,
                baseline_total_return=baseline.metrics.total_return,
                candidate_total_return=candidate.metrics.total_return,
                excess_total_return=(
                    candidate.metrics.total_return - baseline.metrics.total_return
                ),
                baseline_total_cost=baseline.metrics.total_cost,
                candidate_total_cost=candidate.metrics.total_cost,
                baseline_turnover_total=baseline.metrics.turnover_total,
                candidate_turnover_total=candidate.metrics.turnover_total,
                baseline_max_drawdown=baseline.metrics.max_drawdown,
                candidate_max_drawdown=candidate.metrics.max_drawdown,
                baseline_termination_count=baseline.metrics.termination_count,
                candidate_termination_count=candidate.metrics.termination_count,
                baseline_termination_reasons=baseline_reasons,
                candidate_termination_reasons=candidate_reasons,
                new_termination=is_new_termination,
                baseline_n_periods=baseline.metrics.n_periods,
                candidate_n_periods=candidate.metrics.n_periods,
                baseline_return_sha256=evaluation_return_sha256(
                    baseline.replay.returns.values
                ),
                candidate_return_sha256=evaluation_return_sha256(
                    candidate.replay.returns.values
                ),
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
    new_termination = sum(item.new_termination for item in results)
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
        symbols=spec.symbols,
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
    "evaluation_return_sha256",
    "research_status_from_counts",
]
