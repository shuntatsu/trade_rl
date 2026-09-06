"""Canonical candidate/cash replay pairing evidence for U2 Development Selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.workflows.universal_trade_rl_u2_contract import (
    U2_TRAINING_SEEDS,
    UniversalTradeRLU2Contract,
)
from trade_rl.workflows.universal_trade_rl_u2_development_closure import (
    UniversalTradeRLU2FinalCheckpointClosure,
)
from trade_rl.workflows.universal_trade_rl_u2_replay import (
    UniversalTradeRLU2ReplayEvidence,
    UniversalTradeRLU2ReplayVariant,
)
from trade_rl.workflows.universal_trade_rl_u2_time_partition import (
    U2_DECISION_STEP_NS,
    UniversalTradeRLU2TimePartition,
)

U2_PAIRED_REPLAY_SCOPE_EVIDENCE_SCHEMA: Final = (
    "universal_trade_rl_u2_paired_replay_scope_evidence_v1"
)


def _non_negative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    return resolved


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2PairedReplayScopeEvidence:
    """Content-addressed period-aligned candidate/cash replay pair."""

    training_seed: int
    source_window: str
    cell: str
    concrete_symbol: str
    scope_digest: str
    evaluation_dataset_digest: str
    paired_candidate_checkpoint_digest: str
    candidate_replay_evidence_digest: str
    cash_replay_evidence_digest: str
    decision_timestamps_ns: tuple[int, ...]
    candidate_minus_cash_net_log_excess: tuple[float, ...]
    schema_version: str = U2_PAIRED_REPLAY_SCOPE_EVIDENCE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_PAIRED_REPLAY_SCOPE_EVIDENCE_SCHEMA:
            raise ValueError("unsupported U2 paired replay scope evidence schema")
        if (
            isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, int)
            or self.training_seed not in U2_TRAINING_SEEDS
        ):
            raise ValueError("U2 paired replay training seed is not preregistered")
        for field_name, value in (
            ("source window", self.source_window),
            ("cell", self.cell),
            ("concrete symbol", self.concrete_symbol),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"U2 paired replay {field_name} must be non-empty")
        for field_name, digest in (
            ("scope", self.scope_digest),
            ("evaluation dataset", self.evaluation_dataset_digest),
            ("candidate checkpoint", self.paired_candidate_checkpoint_digest),
            ("candidate replay", self.candidate_replay_evidence_digest),
            ("cash replay", self.cash_replay_evidence_digest),
        ):
            require_sha256(digest, field=f"U2 paired replay {field_name} digest")
        if self.candidate_replay_evidence_digest == self.cash_replay_evidence_digest:
            raise ValueError("U2 paired replay candidate/cash evidence must be distinct")

        timestamps = tuple(self.decision_timestamps_ns)
        values = tuple(
            _finite(value, field="U2 paired replay net-log excess")
            for value in self.candidate_minus_cash_net_log_excess
        )
        if not timestamps or len(timestamps) != len(values):
            raise ValueError(
                "U2 paired replay timestamps and net-log excess must be aligned"
            )
        for timestamp in timestamps:
            _non_negative_integer(
                timestamp,
                field="U2 paired replay decision timestamp",
            )
        if timestamps != tuple(sorted(set(timestamps))):
            raise ValueError("U2 paired replay timestamps must be sorted and unique")
        if len(timestamps) > 1 and any(
            later - earlier != U2_DECISION_STEP_NS
            for earlier, later in zip(timestamps, timestamps[1:])
        ):
            raise ValueError("U2 paired replay timestamps must follow the 15m grid")
        object.__setattr__(self, "decision_timestamps_ns", timestamps)
        object.__setattr__(self, "candidate_minus_cash_net_log_excess", values)

        expected_digest = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 paired replay scope evidence digest")
            if self.digest != expected_digest:
                raise ValueError("U2 paired replay scope evidence digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "training_seed": self.training_seed,
            "source_window": self.source_window,
            "cell": self.cell,
            "concrete_symbol": self.concrete_symbol,
            "scope_digest": self.scope_digest,
            "evaluation_dataset_digest": self.evaluation_dataset_digest,
            "paired_candidate_checkpoint_digest": self.paired_candidate_checkpoint_digest,
            "candidate_replay_evidence_digest": self.candidate_replay_evidence_digest,
            "cash_replay_evidence_digest": self.cash_replay_evidence_digest,
            "decision_timestamps_ns": self.decision_timestamps_ns,
            "candidate_minus_cash_net_log_excess": (
                self.candidate_minus_cash_net_log_excess
            ),
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def build_universal_trade_rl_u2_paired_replay_scope_evidence(
    *,
    candidate_replay: UniversalTradeRLU2ReplayEvidence,
    cash_replay: UniversalTradeRLU2ReplayEvidence,
    u2_contract: UniversalTradeRLU2Contract,
    time_partition: UniversalTradeRLU2TimePartition,
    checkpoint_closure: UniversalTradeRLU2FinalCheckpointClosure,
) -> UniversalTradeRLU2PairedReplayScopeEvidence:
    """Derive paired net-log excess only from exact replay evidence."""

    if not isinstance(candidate_replay, UniversalTradeRLU2ReplayEvidence):
        raise TypeError("U2 paired replay requires candidate replay evidence")
    if not isinstance(cash_replay, UniversalTradeRLU2ReplayEvidence):
        raise TypeError("U2 paired replay requires cash replay evidence")
    if not isinstance(u2_contract, UniversalTradeRLU2Contract):
        raise TypeError("U2 paired replay requires a U2 contract")
    if not isinstance(time_partition, UniversalTradeRLU2TimePartition):
        raise TypeError("U2 paired replay requires a U2 time partition")
    if not isinstance(checkpoint_closure, UniversalTradeRLU2FinalCheckpointClosure):
        raise TypeError("U2 paired replay requires final checkpoint closure")
    if candidate_replay.policy_variant != UniversalTradeRLU2ReplayVariant.CANDIDATE.value:
        raise ValueError("U2 paired replay candidate evidence has the wrong variant")
    if cash_replay.policy_variant != UniversalTradeRLU2ReplayVariant.CASH.value:
        raise ValueError("U2 paired replay cash evidence has the wrong variant")

    if time_partition.digest != u2_contract.time_partition_digest:
        raise ValueError("U2 paired replay time-partition identity mismatch")
    if checkpoint_closure.u2_contract_digest != u2_contract.digest:
        raise ValueError("U2 paired replay checkpoint/U2 identity mismatch")
    if checkpoint_closure.time_partition_digest != time_partition.digest:
        raise ValueError("U2 paired replay checkpoint/time-partition identity mismatch")
    if checkpoint_closure.universe_manifest_digest != u2_contract.universe_manifest_digest:
        raise ValueError("U2 paired replay checkpoint/universe identity mismatch")
    if checkpoint_closure.u1_contract_digest != u2_contract.u1_contract_digest:
        raise ValueError("U2 paired replay checkpoint/U1 identity mismatch")
    if checkpoint_closure.normalizer_digest != u2_contract.u1_normalizer_digest:
        raise ValueError("U2 paired replay checkpoint/normalizer identity mismatch")

    identity_fields = (
        "scope_closure_digest",
        "scope_digest",
        "universe_manifest_digest",
        "u1_contract_digest",
        "u2_contract_digest",
        "source_dataset_digest",
        "evaluation_dataset_digest",
        "concrete_symbol",
        "symbol_role",
        "cell",
        "source_window",
        "tile_index",
        "evaluation_seed",
        "paired_candidate_checkpoint_digest",
        "outcome_start_bar_index",
        "outcome_stop_bar_index_exclusive",
        "evaluation_start_bar_index",
        "evaluation_stop_bar_index",
        "runtime_start_bar_index",
        "runtime_end_bar_index",
        "observed_decision_count",
    )
    if any(
        getattr(candidate_replay, field_name) != getattr(cash_replay, field_name)
        for field_name in identity_fields
    ):
        raise ValueError("U2 candidate/cash replay pair identity or dataset drifted")
    if candidate_replay.u2_contract_digest != u2_contract.digest:
        raise ValueError("U2 paired replay evidence U2 identity mismatch")
    if candidate_replay.universe_manifest_digest != u2_contract.universe_manifest_digest:
        raise ValueError("U2 paired replay evidence universe identity mismatch")
    if candidate_replay.u1_contract_digest != u2_contract.u1_contract_digest:
        raise ValueError("U2 paired replay evidence U1 identity mismatch")

    checkpoint_digest = candidate_replay.paired_candidate_checkpoint_digest
    matching_seeds = tuple(
        seed
        for seed, digest in checkpoint_closure.checkpoint_digests
        if digest == checkpoint_digest
    )
    if len(matching_seeds) != 1:
        raise ValueError("U2 paired replay candidate checkpoint is outside exact closure")
    training_seed = matching_seeds[0]

    candidate_bars = tuple(
        step.decision_bar_index for step in candidate_replay.step_evidence
    )
    cash_bars = tuple(step.decision_bar_index for step in cash_replay.step_evidence)
    if candidate_bars != cash_bars:
        raise ValueError("U2 candidate/cash replay period alignment drifted")
    if len(candidate_replay.net_simple_returns) != len(cash_replay.net_simple_returns):
        raise ValueError("U2 candidate/cash replay return length drifted")

    decision_timestamps_ns = tuple(
        time_partition.common_first_timestamp_ns
        + decision_bar_index * U2_DECISION_STEP_NS
        for decision_bar_index in candidate_bars
    )
    net_log_excess = tuple(
        math.log1p(candidate_return) - math.log1p(cash_return)
        for candidate_return, cash_return in zip(
            candidate_replay.net_simple_returns,
            cash_replay.net_simple_returns,
            strict=True,
        )
    )
    return UniversalTradeRLU2PairedReplayScopeEvidence(
        training_seed=training_seed,
        source_window=candidate_replay.source_window,
        cell=candidate_replay.cell,
        concrete_symbol=candidate_replay.concrete_symbol,
        scope_digest=candidate_replay.scope_digest,
        evaluation_dataset_digest=candidate_replay.evaluation_dataset_digest,
        paired_candidate_checkpoint_digest=checkpoint_digest,
        candidate_replay_evidence_digest=candidate_replay.digest,
        cash_replay_evidence_digest=cash_replay.digest,
        decision_timestamps_ns=decision_timestamps_ns,
        candidate_minus_cash_net_log_excess=net_log_excess,
    )


__all__ = [
    "U2_PAIRED_REPLAY_SCOPE_EVIDENCE_SCHEMA",
    "UniversalTradeRLU2PairedReplayScopeEvidence",
    "build_universal_trade_rl_u2_paired_replay_scope_evidence",
]
