"""Run one persisted real-window Stage A legacy-versus-Nautilus probe."""

from __future__ import annotations

import math
import tempfile
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.walk_forward.folds import IndexRange
from trade_rl.integrations.nautilus.instrument import MAINTAINED_BTCUSDT_PERPETUAL
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor
from trade_rl.simulation.execution_promotion import (
    execution_evidence_from_cost,
    write_execution_evidence,
)
from trade_rl.simulation.execution_replay import (
    build_execution_event_artifact,
    write_execution_event_artifact,
)
from trade_rl.simulation.funding_evidence import build_funding_evidence_artifact
from trade_rl.simulation.order_event_batches import merge_order_event_batches
from trade_rl.simulation.orders import OrderBookState
from trade_rl.simulation.target_execution import execute_target_statefully
from trade_rl.workflows.stage_a_execution_store import StageAExecutionPromotionStore
from trade_rl.workflows.stage_a_nautilus_economic_comparison import (
    StageANautilusHistoricalEconomicClosure,
    compare_stage_a_nautilus_historical_economics,
)
from trade_rl.workflows.stage_a_nautilus_historical_differential import (
    build_stage_a_nautilus_historical_differential_evidence,
)
from trade_rl.workflows.stage_a_nautilus_historical_replay import (
    execute_stage_a_nautilus_historical_replay,
)
from trade_rl.workflows.stage_a_nautilus_representative_evidence import (
    RepresentativeNautilusWindowEvidence,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
)

_INITIAL_CAPITAL = 100_000.0
_OPEN_TRANSITION_END = 8
_TERMINAL_TRANSITION_END = 15
_SETTLEMENT_CURRENCY_PRECISION = 8
_PROBE_SCHEMA = "stage_a_nautilus_representative_probe_v1"


def run_representative_nautilus_window(
    *,
    market: MarketDataset,
    time_quantile: float,
    store_root: str | Path,
    target_exposure: float = 0.10,
) -> RepresentativeNautilusWindowEvidence:
    """Persist and compare one fixed, price-independent 4-hour probe window."""

    _validate_market(market)
    target = _validate_target_exposure(target_exposure)
    cost = ExecutionCostConfig.zero()
    request, candidate_config_digest = _request_for_probe(
        market=market,
        time_quantile=time_quantile,
        target_exposure=target,
        execution_identity=cost.execution_policy_digest,
    )

    executor = MarketExecutor(market, cost)
    initial_book = BookState.zero(
        market.n_symbols,
        _INITIAL_CAPITAL,
        initial_prices=market.close[0],
        contract_multipliers=market.resolved_array("contract_multipliers"),
    )
    first = execute_target_statefully(
        executor,
        initial_book,
        OrderBookState.empty(),
        np.array([target], dtype=np.float64),
        start_index=0,
        bars=_OPEN_TRANSITION_END,
        target_identity=f"representative-{time_quantile:.1f}-open",
    )
    second = execute_target_statefully(
        executor,
        first.book,
        first.order_book,
        np.array([0.0], dtype=np.float64),
        start_index=_OPEN_TRANSITION_END,
        bars=_TERMINAL_TRANSITION_END - _OPEN_TRANSITION_END,
        target_identity=f"representative-{time_quantile:.1f}-flat",
    )

    actions = ((target,), (0.0,))
    transition_end_indices = (_OPEN_TRANSITION_END, _TERMINAL_TRANSITION_END)
    equity_curve = (
        _INITIAL_CAPITAL,
        float(first.book.portfolio_value),
        float(second.book.portfolio_value),
    )
    observation_digests = tuple(
        _decision_boundary_digest(market, index=index)
        for index in (0, *transition_end_indices)
    )
    order_events = merge_order_event_batches(
        (first.order_events, second.order_events)
    )
    funding_boundaries = (*first.funding_evidence, *second.funding_evidence)

    with tempfile.TemporaryDirectory(
        prefix="trade-rl-representative-nautilus-"
    ) as temporary:
        source_root = Path(temporary)
        event_artifact = build_execution_event_artifact(
            candidate_config_digest=candidate_config_digest,
            evaluation_run_digest=request.digest,
            fold=request.fold,
            seed=request.seed,
            dataset_id=market.dataset_id,
            execution_policy_digest=cost.execution_policy_digest,
            actions=actions,
            observation_digests=observation_digests,
            equity_curve=equity_curve,
            order_events=order_events,
            terminal_book=second.book,
            terminal_order_book=second.order_book,
        )
        event_path = write_execution_event_artifact(
            source_root / "order-events.json", event_artifact
        )
        execution_evidence = execution_evidence_from_cost(
            dataset_id=market.dataset_id,
            cost=cost,
            sensitivity_path_modes=(cost.path_mode,),
            order_event_artifact_path=event_path,
        )
        evidence_path = source_root / "execution-evidence.json"
        write_execution_evidence(evidence_path, execution_evidence)
        funding_artifact = build_funding_evidence_artifact(
            dataset_id=market.dataset_id,
            execution_policy_digest=cost.execution_policy_digest,
            symbol_count=1,
            boundaries=funding_boundaries,
        )
        funding_path = source_root / "funding-evidence.json"
        funding_path.write_bytes(funding_artifact.raw_bytes)

        stored = StageAExecutionPromotionStore(Path(store_root)).publish(
            request=request,
            candidate_config_digest=candidate_config_digest,
            actions=actions,
            observation_digests=observation_digests,
            equity_curve=equity_curve,
            transition_end_indices=transition_end_indices,
            event_artifact_path=event_path,
            execution_evidence_path=evidence_path,
            funding_evidence_path=funding_path,
        )

    candidate = execute_stage_a_nautilus_historical_replay(
        stored.artifact,
        market,
        funding_evidence=funding_boundaries,
        no_trade_band=0.0,
    )
    structural = build_stage_a_nautilus_historical_differential_evidence(
        stored,
        candidate,
    )
    candidate_funding_minor = sum(
        record.funding_minor for record in candidate.funding_records
    )
    economic = compare_stage_a_nautilus_historical_economics(
        structural=structural,
        legacy=StageANautilusHistoricalEconomicClosure(
            final_equity_minor=_minor_units(second.book.portfolio_value),
            execution_cost_minor=_minor_units(second.book.total_cost),
        ),
        candidate=StageANautilusHistoricalEconomicClosure(
            final_equity_minor=(
                candidate.execution.final_balance_minor + candidate_funding_minor
            ),
            execution_cost_minor=candidate.execution.fee_minor,
        ),
    )
    return RepresentativeNautilusWindowEvidence(
        time_quantile=time_quantile,
        structural=structural,
        economic=economic,
    )


def _validate_market(market: MarketDataset) -> None:
    if market.symbols != ("BTCUSDT",):
        raise ValueError("representative Nautilus probe requires BTCUSDT only")
    if market.n_bars != 16:
        raise ValueError("representative Nautilus probe requires exactly 16 bars")
    for field, actual, expected in (
        (
            "lot_size",
            market.resolved_array("lot_size")[:, 0],
            float(MAINTAINED_BTCUSDT_PERPETUAL.size_increment),
        ),
        (
            "tick_size",
            market.resolved_array("tick_size")[:, 0],
            float(MAINTAINED_BTCUSDT_PERPETUAL.price_increment),
        ),
        (
            "contract_multiplier",
            np.broadcast_to(
                market.resolved_array("contract_multipliers"),
                (market.n_bars, market.n_symbols),
            )[:, 0],
            1.0,
        ),
    ):
        if not np.allclose(actual, expected, rtol=0.0, atol=0.0):
            raise ValueError(f"representative Nautilus probe {field} mismatch")


def _validate_target_exposure(value: float) -> float:
    target = float(value)
    if not math.isfinite(target) or not 0.0 < target <= 1.0:
        raise ValueError("representative target exposure must be within (0, 1]")
    return target


def _request_for_probe(
    *,
    market: MarketDataset,
    time_quantile: float,
    target_exposure: float,
    execution_identity: str,
) -> tuple[StageAEvaluationCellRequest, str]:
    probe_identity = {
        "dataset_id": market.dataset_id,
        "initial_capital": _INITIAL_CAPITAL,
        "open_transition_end": _OPEN_TRANSITION_END,
        "schema_version": _PROBE_SCHEMA,
        "target_exposure": target_exposure,
        "terminal_transition_end": _TERMINAL_TRANSITION_END,
        "time_quantile": time_quantile,
    }
    candidate_config_digest = content_digest(probe_identity)
    request = StageAEvaluationCellRequest(
        plan_digest=content_digest({"probe": probe_identity, "role": "plan"}),
        evaluation_dataset_manifest_digest=content_digest(
            {"probe": probe_identity, "role": "dataset_manifest"}
        ),
        split="validation",
        triplet_id=content_digest({"probe": probe_identity, "role": "triplet"}),
        fold=0,
        seed=0,
        candidate_id="representative-fixed-target-probe",
        checkpoint_digest=content_digest(
            {"probe": probe_identity, "role": "checkpoint"}
        ),
        dataset_id=market.dataset_id,
        evaluation_range=IndexRange(0, _TERMINAL_TRANSITION_END),
        feature_identity=market.feature_config_digest,
        execution_identity=execution_identity,
        evaluation_identity=content_digest(
            {"probe": probe_identity, "role": "evaluation"}
        ),
    )
    return request, candidate_config_digest


def _decision_boundary_digest(market: MarketDataset, *, index: int) -> str:
    return content_digest(
        {
            "close": float(market.close[index, 0]),
            "dataset_id": market.dataset_id,
            "index": index,
            "index_price": float(market.resolved_array("index_price")[index, 0]),
            "mark_price": float(market.resolved_array("mark_price")[index, 0]),
            "schema_version": "representative_probe_decision_boundary_v1",
            "timestamp_ns": int(market.timestamps[index].astype(np.int64)),
        }
    )


def _minor_units(value: float) -> int:
    decimal_value = Decimal(str(float(value)))
    if not decimal_value.is_finite() or decimal_value < 0:
        raise ValueError("representative economic value must be finite and non-negative")
    scaled = decimal_value * (Decimal(10) ** _SETTLEMENT_CURRENCY_PRECISION)
    return int(scaled.quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))


__all__ = ["run_representative_nautilus_window"]
