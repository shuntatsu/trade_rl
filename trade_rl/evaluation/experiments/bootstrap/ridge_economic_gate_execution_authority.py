"""Result-blind PRE-P&L authority and execution provenance for Issue #616.

The pairwise evaluator in :mod:`ridge_economic_gate_evaluation` is the synthetic
computation core. Real economic execution must cross the boundary in this
module so Dataset-only cost constancy is proved and content-addressed before
model fitting/replay, and the published result is bound to the exact economic
implementation HEAD and workflow run.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.experiments.bootstrap.ridge_economic_gate_evaluation import (
    RidgeEconomicGateEvaluation,
    RidgeEconomicGateEvaluationSpec,
    canonical_ridge_economic_gate_evaluation_spec,
    evaluate_ridge_economic_gate,
)

_COST_AUTHORITY_SCHEMA = "ridge_economic_gate_pre_pnl_cost_authority_v1"
_EXECUTION_PUBLICATION_SCHEMA = "ridge_economic_gate_execution_publication_v1"


def _require_hex(value: object, *, field: str, length: int = 64) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{field} must be a {length}-character lowercase hex value")
    if value.lower() != value or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a {length}-character lowercase hex value")
    return value


def _require_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _require_finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite non-negative number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return resolved


def _array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8")
    hasher = hashlib.sha256()
    hasher.update(array.ndim.to_bytes(8, "big", signed=False))
    for size in array.shape:
        hasher.update(int(size).to_bytes(8, "big", signed=False))
    hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


def _exact_timestamp_index(dataset: MarketDataset, timestamp: str, *, field: str) -> int:
    target = np.datetime64(timestamp, "ns")
    matches = np.flatnonzero(dataset.timestamps == target)
    if matches.size != 1:
        raise ValueError(f"{field} is not uniquely present in Dataset timestamps")
    return int(matches[0])


@dataclass(frozen=True, slots=True)
class RidgeEconomicGatePrePnlCostAuthority:
    """Immutable Dataset-only proof required before any Ridge gate fit/replay."""

    schema_version: str
    protocol_digest: str
    spec_digest: str
    dataset_id: str
    dataset_artifact_digest: str
    evaluation_start: str
    evaluation_stop_exclusive: str
    n_evaluation_rows: int
    market_order_fee_rate: float
    market_order_taker_fee_rate: float
    market_order_spread_rate: float
    one_way_explicit_cost: float
    capacity_caps: tuple[float, ...]
    fee_rate_sha256: str
    taker_fee_rate_sha256: str
    spread_rate_sha256: str
    capacity_sha256: str
    authority_implementation_head: str
    authority_run_id: int
    result_blind: bool = True
    evaluation_pnl_inspected: bool = False

    def __post_init__(self) -> None:
        spec = canonical_ridge_economic_gate_evaluation_spec()
        expected = {
            "schema_version": _COST_AUTHORITY_SCHEMA,
            "protocol_digest": spec.protocol_digest,
            "spec_digest": spec.digest,
            "dataset_id": spec.dataset_id,
            "dataset_artifact_digest": spec.dataset_artifact_digest,
            "evaluation_start": spec.evaluation_start,
            "evaluation_stop_exclusive": spec.evaluation_stop_exclusive,
            "market_order_fee_rate": spec.market_order_fee_rate,
            "market_order_taker_fee_rate": spec.market_order_taker_fee_rate,
            "market_order_spread_rate": spec.market_order_spread_rate,
            "one_way_explicit_cost": spec.one_way_explicit_cost,
            "capacity_caps": spec.capacity_caps,
            "result_blind": True,
            "evaluation_pnl_inspected": False,
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(f"{field_name} differs from sealed Issue #616 authority")
        _require_positive_int(self.n_evaluation_rows, field="n_evaluation_rows")
        _require_hex(
            self.authority_implementation_head,
            field="authority_implementation_head",
            length=40,
        )
        _require_positive_int(self.authority_run_id, field="authority_run_id")
        for field_name in (
            "protocol_digest",
            "spec_digest",
            "dataset_id",
            "dataset_artifact_digest",
            "fee_rate_sha256",
            "taker_fee_rate_sha256",
            "spread_rate_sha256",
            "capacity_sha256",
        ):
            _require_hex(getattr(self, field_name), field=field_name)
        for field_name in (
            "market_order_fee_rate",
            "market_order_taker_fee_rate",
            "market_order_spread_rate",
            "one_way_explicit_cost",
        ):
            _require_finite_nonnegative(getattr(self, field_name), field=field_name)
        if type(self.result_blind) is not bool or type(self.evaluation_pnl_inspected) is not bool:
            raise ValueError("authority research-boundary flags must be booleans")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_digest": self.protocol_digest,
            "spec_digest": self.spec_digest,
            "dataset_id": self.dataset_id,
            "dataset_artifact_digest": self.dataset_artifact_digest,
            "evaluation_start": self.evaluation_start,
            "evaluation_stop_exclusive": self.evaluation_stop_exclusive,
            "n_evaluation_rows": self.n_evaluation_rows,
            "market_order_fee_rate": self.market_order_fee_rate,
            "market_order_taker_fee_rate": self.market_order_taker_fee_rate,
            "market_order_spread_rate": self.market_order_spread_rate,
            "one_way_explicit_cost": self.one_way_explicit_cost,
            "capacity_caps": list(self.capacity_caps),
            "fee_rate_sha256": self.fee_rate_sha256,
            "taker_fee_rate_sha256": self.taker_fee_rate_sha256,
            "spread_rate_sha256": self.spread_rate_sha256,
            "capacity_sha256": self.capacity_sha256,
            "authority_implementation_head": self.authority_implementation_head,
            "authority_run_id": self.authority_run_id,
            "result_blind": self.result_blind,
            "evaluation_pnl_inspected": self.evaluation_pnl_inspected,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def build_ridge_economic_gate_pre_pnl_cost_authority(
    dataset: MarketDataset,
    *,
    authority_implementation_head: str,
    authority_run_id: int,
    spec: RidgeEconomicGateEvaluationSpec | None = None,
) -> RidgeEconomicGatePrePnlCostAuthority:
    """Prove the sealed evaluation-window economics without fitting or replaying."""

    if not isinstance(dataset, MarketDataset):
        raise TypeError("dataset must be a MarketDataset")
    resolved_spec = spec or canonical_ridge_economic_gate_evaluation_spec()
    if not isinstance(resolved_spec, RidgeEconomicGateEvaluationSpec):
        raise TypeError("spec must be a RidgeEconomicGateEvaluationSpec")
    if dataset.dataset_id != resolved_spec.dataset_id:
        raise ValueError("Dataset id differs from sealed Issue #616 authority")
    if tuple(dataset.symbols) != resolved_spec.symbols:
        raise ValueError("Dataset symbols differ from sealed Issue #616 authority")

    start_index = _exact_timestamp_index(
        dataset, resolved_spec.evaluation_start, field="evaluation_start"
    )
    stop_index = _exact_timestamp_index(
        dataset,
        resolved_spec.evaluation_stop_exclusive,
        field="evaluation_stop_exclusive",
    )
    if not 0 <= start_index < stop_index <= dataset.n_bars:
        raise ValueError("sealed evaluation range is invalid for Dataset")
    scope = slice(start_index, stop_index)
    n_evaluation_rows = stop_index - start_index

    resolved_arrays: dict[str, np.ndarray] = {}
    for field_name, expected in (
        ("fee_rate", resolved_spec.market_order_fee_rate),
        ("taker_fee_rate", resolved_spec.market_order_taker_fee_rate),
        ("spread_rate", resolved_spec.market_order_spread_rate),
    ):
        values = np.asarray(dataset.resolved_array(field_name)[scope], dtype=np.float64)
        if values.shape != (n_evaluation_rows, len(resolved_spec.symbols)):
            raise ValueError(f"PRE-P&L cost authority: {field_name} shape is invalid")
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError(f"PRE-P&L cost authority: {field_name} is invalid")
        if not bool(np.all(values == expected)):
            raise ValueError(
                f"PRE-P&L cost authority: {field_name} differs from sealed value"
            )
        resolved_arrays[field_name] = values

    capacity = np.asarray(
        dataset.resolved_array("max_participation_rate")[scope], dtype=np.float64
    )
    expected_capacity = np.broadcast_to(
        np.asarray(resolved_spec.capacity_caps, dtype=np.float64), capacity.shape
    )
    if capacity.shape != (n_evaluation_rows, len(resolved_spec.symbols)):
        raise ValueError("PRE-P&L cost authority: capacity shape is invalid")
    if not np.isfinite(capacity).all() or not np.array_equal(capacity, expected_capacity):
        raise ValueError(
            "PRE-P&L cost authority: capacity differs from sealed Issue #616 authority"
        )

    return RidgeEconomicGatePrePnlCostAuthority(
        schema_version=_COST_AUTHORITY_SCHEMA,
        protocol_digest=resolved_spec.protocol_digest,
        spec_digest=resolved_spec.digest,
        dataset_id=resolved_spec.dataset_id,
        dataset_artifact_digest=resolved_spec.dataset_artifact_digest,
        evaluation_start=resolved_spec.evaluation_start,
        evaluation_stop_exclusive=resolved_spec.evaluation_stop_exclusive,
        n_evaluation_rows=n_evaluation_rows,
        market_order_fee_rate=resolved_spec.market_order_fee_rate,
        market_order_taker_fee_rate=resolved_spec.market_order_taker_fee_rate,
        market_order_spread_rate=resolved_spec.market_order_spread_rate,
        one_way_explicit_cost=resolved_spec.one_way_explicit_cost,
        capacity_caps=resolved_spec.capacity_caps,
        fee_rate_sha256=_array_sha256(resolved_arrays["fee_rate"]),
        taker_fee_rate_sha256=_array_sha256(resolved_arrays["taker_fee_rate"]),
        spread_rate_sha256=_array_sha256(resolved_arrays["spread_rate"]),
        capacity_sha256=_array_sha256(capacity),
        authority_implementation_head=_require_hex(
            authority_implementation_head,
            field="authority_implementation_head",
            length=40,
        ),
        authority_run_id=_require_positive_int(authority_run_id, field="authority_run_id"),
    )


def _validated_pre_pnl_authority(
    dataset: MarketDataset,
    authority: RidgeEconomicGatePrePnlCostAuthority,
    spec: RidgeEconomicGateEvaluationSpec,
) -> RidgeEconomicGatePrePnlCostAuthority:
    if not isinstance(authority, RidgeEconomicGatePrePnlCostAuthority):
        raise TypeError(
            "pre_pnl_authority must be a RidgeEconomicGatePrePnlCostAuthority"
        )
    reconstructed = build_ridge_economic_gate_pre_pnl_cost_authority(
        dataset,
        authority_implementation_head=authority.authority_implementation_head,
        authority_run_id=authority.authority_run_id,
        spec=spec,
    )
    if reconstructed.to_payload() != authority.to_payload():
        raise ValueError("PRE-P&L cost authority does not match current Dataset evidence")
    return reconstructed


@dataclass(frozen=True, slots=True)
class RidgeEconomicGateExecutionPublication:
    """Canonical publication envelope binding P&L to exact execution provenance."""

    pre_pnl_cost_authority_digest: str
    economic_implementation_head: str
    economic_run_id: int
    result: RidgeEconomicGateEvaluation

    def __post_init__(self) -> None:
        _require_hex(
            self.pre_pnl_cost_authority_digest,
            field="pre_pnl_cost_authority_digest",
        )
        _require_hex(
            self.economic_implementation_head,
            field="economic_implementation_head",
            length=40,
        )
        _require_positive_int(self.economic_run_id, field="economic_run_id")
        if not isinstance(self.result, RidgeEconomicGateEvaluation):
            raise TypeError("result must be a RidgeEconomicGateEvaluation")
        spec = canonical_ridge_economic_gate_evaluation_spec()
        if self.result.spec_digest != spec.digest or self.result.dataset_id != spec.dataset_id:
            raise ValueError("execution result differs from sealed Issue #616 authority")

    def to_payload(self) -> dict[str, object]:
        spec = canonical_ridge_economic_gate_evaluation_spec()
        return {
            "schema_version": _EXECUTION_PUBLICATION_SCHEMA,
            "protocol_digest": spec.protocol_digest,
            "spec_digest": spec.digest,
            "dataset_id": self.result.dataset_id,
            "dataset_artifact_digest": spec.dataset_artifact_digest,
            "pre_pnl_cost_authority_digest": self.pre_pnl_cost_authority_digest,
            "economic_implementation_head": self.economic_implementation_head,
            "economic_run_id": self.economic_run_id,
            "evaluation_result_digest": content_digest(self.result.to_payload()),
            "result": self.result.to_payload(),
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def canonical_ridge_economic_gate_execution_publication_bytes(
    publication: RidgeEconomicGateExecutionPublication,
) -> bytes:
    """Encode the execution envelope in deterministic content-addressed JSON."""

    if not isinstance(publication, RidgeEconomicGateExecutionPublication):
        raise TypeError("publication must be a RidgeEconomicGateExecutionPublication")
    document = {
        "schema_version": _EXECUTION_PUBLICATION_SCHEMA,
        "content_digest": publication.digest,
        "publication": publication.to_payload(),
    }
    return (
        json.dumps(document, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def execute_ridge_economic_gate_with_authority(
    dataset: MarketDataset,
    *,
    pre_pnl_authority: RidgeEconomicGatePrePnlCostAuthority,
    economic_implementation_head: str,
    economic_run_id: int,
    spec: RidgeEconomicGateEvaluationSpec | None = None,
) -> RidgeEconomicGateExecutionPublication:
    """Run P&L only after PRE-P&L authority and execution identity validate."""

    resolved_spec = spec or canonical_ridge_economic_gate_evaluation_spec()
    if not isinstance(resolved_spec, RidgeEconomicGateEvaluationSpec):
        raise TypeError("spec must be a RidgeEconomicGateEvaluationSpec")
    validated_authority = _validated_pre_pnl_authority(
        dataset, pre_pnl_authority, resolved_spec
    )
    implementation_head = _require_hex(
        economic_implementation_head,
        field="economic_implementation_head",
        length=40,
    )
    run_id = _require_positive_int(economic_run_id, field="economic_run_id")

    result = evaluate_ridge_economic_gate(dataset, resolved_spec)
    return RidgeEconomicGateExecutionPublication(
        pre_pnl_cost_authority_digest=validated_authority.digest,
        economic_implementation_head=implementation_head,
        economic_run_id=run_id,
        result=result,
    )


__all__ = [
    "RidgeEconomicGateExecutionPublication",
    "RidgeEconomicGatePrePnlCostAuthority",
    "build_ridge_economic_gate_pre_pnl_cost_authority",
    "canonical_ridge_economic_gate_execution_publication_bytes",
    "execute_ridge_economic_gate_with_authority",
]
