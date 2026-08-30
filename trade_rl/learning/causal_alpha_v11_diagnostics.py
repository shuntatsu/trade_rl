"""Trade-level D1 decomposition for exact Causal Alpha V9 control replays."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.rollout_evaluation import (
    ActionPathLifecycleTrace,
    ActionPathStepTrace,
)

CAUSAL_ALPHA_V11_DIAGNOSTIC_SCHEMA: Final = "causal_alpha_v11_diagnostic_v1"
_HORIZON_DECISIONS: Final = 16
_HOURS_PER_DECISION: Final = 0.25
_EPSILON: Final = 1e-12


@dataclass(frozen=True, slots=True)
class CausalAlphaV11EntryEvidence:
    entry_index: int
    direction: int
    directional_label_4h: float
    round_trip_cost: float
    entry_edge: float


@dataclass(frozen=True, slots=True)
class CausalAlphaV11TradeDecomposition:
    entry_index: int
    first_neutral_index: int | None
    exit_index: int | None
    direction: int
    right_censored: bool
    entry_to_neutral_gross_log_return: float
    entry_to_neutral_net_log_return: float
    entry_to_neutral_cost: float
    neutral_to_exit_gross_log_return: float
    neutral_to_exit_net_log_return: float
    neutral_to_exit_cost: float
    total_gross_log_return: float
    total_net_log_return: float
    total_cost: float
    turnover: float
    exposure_hours: float
    maximum_adverse_excursion: float
    maximum_favorable_excursion: float


@dataclass(frozen=True, slots=True)
class CausalAlphaV11DiagnosticSummary:
    trade_count: int
    neutral_observed_count: int
    right_censored_count: int
    mean_net_log_return: float
    median_net_log_return: float
    positive_fraction: float
    cvar10_net_log_return: float
    mean_entry_to_neutral_net_log_return: float
    mean_neutral_to_exit_net_log_return: float
    net_log_return_per_exposure_hour: float
    net_log_return_per_turnover: float
    mean_maximum_adverse_excursion: float
    mean_maximum_favorable_excursion: float


@dataclass(frozen=True, slots=True)
class CausalAlphaV11DiagnosticEvidence:
    symbol: str
    episode_id: str
    expected_target_digest: str
    regenerated_target_digest: str
    policy_input_digest: str
    step_trace_digest: str
    lifecycle_trace_digest: str
    entries: tuple[CausalAlphaV11EntryEvidence, ...]
    trades: tuple[CausalAlphaV11TradeDecomposition, ...]
    pooled_summary: CausalAlphaV11DiagnosticSummary
    long_summary: CausalAlphaV11DiagnosticSummary
    short_summary: CausalAlphaV11DiagnosticSummary
    reconciliation_error: float
    schema_version: str = CAUSAL_ALPHA_V11_DIAGNOSTIC_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.symbol or not self.episode_id:
            raise ValueError("V11 diagnostic scope identity is invalid")
        for field_name in (
            "expected_target_digest",
            "regenerated_target_digest",
            "policy_input_digest",
            "step_trace_digest",
            "lifecycle_trace_digest",
        ):
            require_sha256(getattr(self, field_name), field=f"V11 {field_name}")
        if self.expected_target_digest != self.regenerated_target_digest:
            raise ValueError("V11 target digest mismatch")
        if (
            not math.isfinite(self.reconciliation_error)
            or self.reconciliation_error > 1e-10
        ):
            raise ValueError("V11 diagnostic return reconciliation failed")
        if self.schema_version != CAUSAL_ALPHA_V11_DIAGNOSTIC_SCHEMA:
            raise ValueError("unsupported V11 diagnostic schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V11 diagnostic digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "entries": tuple(asdict(item) for item in self.entries),
            "episode_id": self.episode_id,
            "expected_target_digest": self.expected_target_digest,
            "lifecycle_trace_digest": self.lifecycle_trace_digest,
            "long_summary": asdict(self.long_summary),
            "policy_input_digest": self.policy_input_digest,
            "pooled_summary": asdict(self.pooled_summary),
            "reconciliation_error": self.reconciliation_error,
            "regenerated_target_digest": self.regenerated_target_digest,
            "schema_version": self.schema_version,
            "short_summary": asdict(self.short_summary),
            "step_trace_digest": self.step_trace_digest,
            "symbol": self.symbol,
            "trades": tuple(asdict(item) for item in self.trades),
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _aligned(
    value: object,
    *,
    rows: int,
    dtype: type[np.float64] | type[np.int8] | type[np.bool_],
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.shape != (rows,):
        raise ValueError("V11 diagnostic arrays must align with the step trace")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError("V11 diagnostic arrays must be finite")
    return array


def _log_sum(values: np.ndarray) -> float:
    return float(np.sum(np.log1p(values), dtype=np.float64))


def _summary(
    trades: tuple[CausalAlphaV11TradeDecomposition, ...], *, direction: int | None
) -> CausalAlphaV11DiagnosticSummary:
    selected = tuple(
        trade for trade in trades if direction is None or trade.direction == direction
    )
    if not selected:
        return CausalAlphaV11DiagnosticSummary(
            trade_count=0,
            neutral_observed_count=0,
            right_censored_count=0,
            mean_net_log_return=0.0,
            median_net_log_return=0.0,
            positive_fraction=0.0,
            cvar10_net_log_return=0.0,
            mean_entry_to_neutral_net_log_return=0.0,
            mean_neutral_to_exit_net_log_return=0.0,
            net_log_return_per_exposure_hour=0.0,
            net_log_return_per_turnover=0.0,
            mean_maximum_adverse_excursion=0.0,
            mean_maximum_favorable_excursion=0.0,
        )
    net = np.asarray([trade.total_net_log_return for trade in selected])
    before = np.asarray([trade.entry_to_neutral_net_log_return for trade in selected])
    after = np.asarray([trade.neutral_to_exit_net_log_return for trade in selected])
    exposure = sum(trade.exposure_hours for trade in selected)
    turnover = sum(trade.turnover for trade in selected)
    tail_count = max(1, int(math.ceil(0.1 * len(selected))))
    return CausalAlphaV11DiagnosticSummary(
        trade_count=len(selected),
        neutral_observed_count=sum(
            trade.first_neutral_index is not None for trade in selected
        ),
        right_censored_count=sum(trade.right_censored for trade in selected),
        mean_net_log_return=float(np.mean(net)),
        median_net_log_return=float(np.median(net)),
        positive_fraction=float(np.mean(net > 0.0)),
        cvar10_net_log_return=float(np.mean(np.sort(net)[:tail_count])),
        mean_entry_to_neutral_net_log_return=float(np.mean(before)),
        mean_neutral_to_exit_net_log_return=float(np.mean(after)),
        net_log_return_per_exposure_hour=float(np.sum(net) / max(exposure, _EPSILON)),
        net_log_return_per_turnover=float(np.sum(net) / max(turnover, _EPSILON)),
        mean_maximum_adverse_excursion=float(
            np.mean([trade.maximum_adverse_excursion for trade in selected])
        ),
        mean_maximum_favorable_excursion=float(
            np.mean([trade.maximum_favorable_excursion for trade in selected])
        ),
    )


def build_causal_alpha_v11_diagnostics(
    *,
    symbol: str,
    episode_id: str,
    step_trace: ActionPathStepTrace,
    lifecycle_trace: ActionPathLifecycleTrace,
    qualified_directions: object,
    actionable_mask: object,
    labels_4h: object,
    one_way_cost_rates: object,
    expected_target_digest: str,
    regenerated_target_digest: str,
    policy_input_digest: str,
) -> CausalAlphaV11DiagnosticEvidence:
    """Join regenerated V9 signals to authoritative r21 lifecycle economics."""

    if expected_target_digest != regenerated_target_digest:
        raise ValueError("V11 target digest mismatch")
    rows = step_trace.decision_count
    if lifecycle_trace.decision_count != rows:
        raise ValueError("V11 step and lifecycle traces are not aligned")
    if step_trace.realized_weights.shape[1] != 1:
        raise ValueError("V11 D1 requires one independently replayed symbol")
    qualified = _aligned(qualified_directions, rows=rows, dtype=np.int8)
    actionable = _aligned(actionable_mask, rows=rows, dtype=np.bool_)
    labels = _aligned(labels_4h, rows=rows, dtype=np.float64)
    one_way_costs = _aligned(one_way_cost_rates, rows=rows, dtype=np.float64)
    if np.any(~np.isin(qualified, (-1, 0, 1))) or np.any(one_way_costs < 0.0):
        raise ValueError("V11 signal or cost evidence is invalid")

    trades: list[CausalAlphaV11TradeDecomposition] = []
    entries: list[CausalAlphaV11EntryEvidence] = []
    entry_offsets = [
        index
        for index, transition in enumerate(lifecycle_trace.transition_classes)
        if transition == "entry"
    ]
    for entry_offset in entry_offsets:
        realized = float(step_trace.realized_weights[entry_offset, 0])
        direction = int(np.sign(realized))
        if direction == 0:
            raise ValueError("V11 lifecycle entry did not create realized exposure")
        exit_offset = next(
            (
                index
                for index in range(entry_offset + 1, rows)
                if lifecycle_trace.transition_classes[index] == "exit"
            ),
            None,
        )
        terminal = rows - 1 if exit_offset is None else exit_offset
        neutral_offset = next(
            (
                index
                for index in range(entry_offset + 1, terminal + 1)
                if index % _HORIZON_DECISIONS == 0
                and bool(actionable[index])
                and int(qualified[index]) == 0
            ),
            None,
        )
        split = terminal + 1 if neutral_offset is None else neutral_offset
        total_stop = terminal + 1
        gross_before = _log_sum(step_trace.gross_returns[entry_offset:split])
        net_before = _log_sum(step_trace.net_returns[entry_offset:split])
        gross_after = _log_sum(step_trace.gross_returns[split:total_stop])
        net_after = _log_sum(step_trace.net_returns[split:total_stop])
        total_gross = _log_sum(step_trace.gross_returns[entry_offset:total_stop])
        total_net = _log_sum(step_trace.net_returns[entry_offset:total_stop])
        cumulative_net = np.cumsum(
            np.log1p(step_trace.net_returns[entry_offset:total_stop]),
            dtype=np.float64,
        )
        exposure_hours = float(
            np.count_nonzero(
                np.abs(step_trace.realized_weights[entry_offset:total_stop, 0])
                > _EPSILON
            )
            * _HOURS_PER_DECISION
        )
        trades.append(
            CausalAlphaV11TradeDecomposition(
                entry_index=int(step_trace.decision_indices[entry_offset]),
                first_neutral_index=(
                    None
                    if neutral_offset is None
                    else int(step_trace.decision_indices[neutral_offset])
                ),
                exit_index=(
                    None
                    if exit_offset is None
                    else int(step_trace.decision_indices[exit_offset])
                ),
                direction=direction,
                right_censored=exit_offset is None,
                entry_to_neutral_gross_log_return=gross_before,
                entry_to_neutral_net_log_return=net_before,
                entry_to_neutral_cost=float(
                    np.sum(step_trace.costs[entry_offset:split])
                ),
                neutral_to_exit_gross_log_return=gross_after,
                neutral_to_exit_net_log_return=net_after,
                neutral_to_exit_cost=float(np.sum(step_trace.costs[split:total_stop])),
                total_gross_log_return=total_gross,
                total_net_log_return=total_net,
                total_cost=float(np.sum(step_trace.costs[entry_offset:total_stop])),
                turnover=float(np.sum(step_trace.turnovers[entry_offset:total_stop])),
                exposure_hours=exposure_hours,
                maximum_adverse_excursion=float(
                    np.minimum(0.0, np.min(cumulative_net))
                ),
                maximum_favorable_excursion=float(
                    np.maximum(0.0, np.max(cumulative_net))
                ),
            )
        )
        entries.append(
            CausalAlphaV11EntryEvidence(
                entry_index=int(step_trace.decision_indices[entry_offset]),
                direction=direction,
                directional_label_4h=float(direction * labels[entry_offset]),
                round_trip_cost=float(2.0 * one_way_costs[entry_offset]),
                entry_edge=float(
                    direction * labels[entry_offset] - 2.0 * one_way_costs[entry_offset]
                ),
            )
        )
    trade_tuple = tuple(trades)
    reconciliation_error = max(
        (
            abs(
                trade.total_net_log_return
                - trade.entry_to_neutral_net_log_return
                - trade.neutral_to_exit_net_log_return
            )
            for trade in trade_tuple
        ),
        default=0.0,
    )
    return CausalAlphaV11DiagnosticEvidence(
        symbol=symbol,
        episode_id=episode_id,
        expected_target_digest=expected_target_digest,
        regenerated_target_digest=regenerated_target_digest,
        policy_input_digest=policy_input_digest,
        step_trace_digest=step_trace.digest,
        lifecycle_trace_digest=lifecycle_trace.digest,
        entries=tuple(entries),
        trades=trade_tuple,
        pooled_summary=_summary(trade_tuple, direction=None),
        long_summary=_summary(trade_tuple, direction=1),
        short_summary=_summary(trade_tuple, direction=-1),
        reconciliation_error=reconciliation_error,
    )


__all__ = [
    "CAUSAL_ALPHA_V11_DIAGNOSTIC_SCHEMA",
    "CausalAlphaV11DiagnosticEvidence",
    "CausalAlphaV11DiagnosticSummary",
    "CausalAlphaV11EntryEvidence",
    "CausalAlphaV11TradeDecomposition",
    "build_causal_alpha_v11_diagnostics",
]
