from pathlib import Path

path = Path("trade_rl/workflows/universal_trade_rl_u2_selection.py")
text = path.read_text(encoding="utf-8")

contract_import_old = '''from trade_rl.workflows.universal_trade_rl_u2_contract import U2_TRAINING_SEEDS
'''
contract_import_new = '''from trade_rl.workflows.universal_trade_rl_u2_contract import (
    U2_TRAINING_SEEDS,
    UniversalTradeRLU2Contract,
)
'''
if contract_import_old not in text:
    raise SystemExit("Task5 contract import anchor missing")
text = text.replace(contract_import_old, contract_import_new, 1)

replay_import_anchor = '''from trade_rl.workflows.universal_trade_rl_u2_replay import (
    UniversalTradeRLU2ReplayEvidence,
    UniversalTradeRLU2ReplayVariant,
)
'''
time_partition_import = '''from trade_rl.workflows.universal_trade_rl_u2_time_partition import (
    UniversalTradeRLU2TimePartition,
)
'''
if replay_import_anchor not in text:
    raise SystemExit("Task5 replay import anchor missing")
text = text.replace(
    replay_import_anchor,
    replay_import_anchor + time_partition_import,
    1,
)

pair_insert_anchor = '''@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2PairedExcessPoint:
'''
pair_block = '''U2_PAIRED_REPLAY_SCOPE_SCHEMA: Final = (
    "universal_trade_rl_u2_paired_replay_scope_evidence_v1"
)
U2_PAIRED_BOOTSTRAP_SEGMENT_SCHEMA: Final = (
    "universal_trade_rl_u2_paired_bootstrap_segment_v1"
)


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2PairedReplayScopeEvidence:
    """Digest-bound exact candidate/cash pairing for one replay scope."""

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
    schema_version: str = U2_PAIRED_REPLAY_SCOPE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_PAIRED_REPLAY_SCOPE_SCHEMA:
            raise ValueError("unsupported U2 paired replay scope schema")
        if (
            isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, int)
            or self.training_seed not in U2_TRAINING_SEEDS
        ):
            raise ValueError("U2 paired replay training seed is not preregistered")
        for field_name, value in (
            ("source_window", self.source_window),
            ("cell", self.cell),
            ("concrete_symbol", self.concrete_symbol),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"U2 paired replay {field_name} must be non-empty")
        for field_name, value in (
            ("scope_digest", self.scope_digest),
            ("evaluation_dataset_digest", self.evaluation_dataset_digest),
            (
                "paired_candidate_checkpoint_digest",
                self.paired_candidate_checkpoint_digest,
            ),
            (
                "candidate_replay_evidence_digest",
                self.candidate_replay_evidence_digest,
            ),
            ("cash_replay_evidence_digest", self.cash_replay_evidence_digest),
        ):
            require_sha256(value, field=f"U2 paired replay {field_name}")
        if self.candidate_replay_evidence_digest == self.cash_replay_evidence_digest:
            raise ValueError("U2 paired replay candidate/cash evidence must be distinct")

        timestamps = tuple(self.decision_timestamps_ns)
        values = tuple(
            _finite(value, field="U2 paired replay net-log excess")
            for value in self.candidate_minus_cash_net_log_excess
        )
        if not timestamps or len(timestamps) != len(values):
            raise ValueError("U2 paired replay timestamps/excess must be non-empty and aligned")
        for timestamp in timestamps:
            _non_negative_int(timestamp, field="U2 paired replay decision timestamp")
        if timestamps != tuple(sorted(set(timestamps))):
            raise ValueError("U2 paired replay timestamps must be sorted and unique")
        object.__setattr__(self, "decision_timestamps_ns", timestamps)
        object.__setattr__(self, "candidate_minus_cash_net_log_excess", values)

        expected_digest = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 paired replay evidence digest")
            if self.digest != expected_digest:
                raise ValueError("U2 paired replay evidence digest mismatch")
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
    """Pair exact candidate/cash replay children and derive net-log excess."""

    if not isinstance(candidate_replay, UniversalTradeRLU2ReplayEvidence):
        raise TypeError("U2 pairing requires candidate replay evidence")
    if not isinstance(cash_replay, UniversalTradeRLU2ReplayEvidence):
        raise TypeError("U2 pairing requires cash replay evidence")
    if not isinstance(u2_contract, UniversalTradeRLU2Contract):
        raise TypeError("U2 pairing requires a U2 contract")
    if not isinstance(time_partition, UniversalTradeRLU2TimePartition):
        raise TypeError("U2 pairing requires the U2 time partition")
    if not isinstance(checkpoint_closure, UniversalTradeRLU2FinalCheckpointClosure):
        raise TypeError("U2 pairing requires final checkpoint closure")
    for label, evidence in (
        ("candidate", candidate_replay),
        ("cash", cash_replay),
    ):
        if evidence.digest != content_digest(evidence.to_payload(include_digest=False)):
            raise ValueError(f"U2 pairing {label} replay evidence digest drifted")
    if candidate_replay.policy_variant != UniversalTradeRLU2ReplayVariant.CANDIDATE.value:
        raise ValueError("U2 pairing candidate replay variant is invalid")
    if cash_replay.policy_variant != UniversalTradeRLU2ReplayVariant.CASH.value:
        raise ValueError("U2 pairing cash replay variant is invalid")

    paired_identity_fields = (
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
        "final_current_bar_index",
        "observed_decision_count",
    )
    for field_name in paired_identity_fields:
        if getattr(candidate_replay, field_name) != getattr(cash_replay, field_name):
            raise ValueError(f"U2 candidate/cash pair {field_name} identity mismatch")

    if time_partition.digest != u2_contract.time_partition_digest:
        raise ValueError("U2 pairing time-partition contract identity mismatch")
    if time_partition.universe_manifest_digest != u2_contract.universe_manifest_digest:
        raise ValueError("U2 pairing time-partition universe identity mismatch")
    if checkpoint_closure.u2_contract_digest != u2_contract.digest:
        raise ValueError("U2 pairing checkpoint U2 identity mismatch")
    if checkpoint_closure.universe_manifest_digest != u2_contract.universe_manifest_digest:
        raise ValueError("U2 pairing checkpoint universe identity mismatch")
    if checkpoint_closure.u1_contract_digest != u2_contract.u1_contract_digest:
        raise ValueError("U2 pairing checkpoint U1 identity mismatch")
    if checkpoint_closure.time_partition_digest != time_partition.digest:
        raise ValueError("U2 pairing checkpoint time-partition identity mismatch")
    if candidate_replay.u2_contract_digest != u2_contract.digest:
        raise ValueError("U2 pairing replay U2 identity mismatch")
    if candidate_replay.u1_contract_digest != u2_contract.u1_contract_digest:
        raise ValueError("U2 pairing replay U1 identity mismatch")
    if candidate_replay.universe_manifest_digest != u2_contract.universe_manifest_digest:
        raise ValueError("U2 pairing replay universe identity mismatch")

    checkpoints = dict(checkpoint_closure.checkpoint_digests)
    seed = candidate_replay.evaluation_seed
    expected_checkpoint = checkpoints.get(seed)
    if expected_checkpoint is None:
        raise ValueError("U2 pairing checkpoint seed closure is incomplete")
    if candidate_replay.paired_candidate_checkpoint_digest != expected_checkpoint:
        raise ValueError("U2 pairing replay checkpoint identity mismatch")
    if not candidate_replay.normal_completion or not cash_replay.normal_completion:
        raise ValueError("U2 pairing requires normally completed candidate/cash replay")

    candidate_bars = tuple(
        step.decision_bar_index for step in candidate_replay.step_evidence
    )
    cash_bars = tuple(step.decision_bar_index for step in cash_replay.step_evidence)
    if candidate_bars != cash_bars:
        raise ValueError("U2 candidate/cash pair decision-bar alignment mismatch")
    timestamps = tuple(
        time_partition.common_first_timestamp_ns
        + bar_index * time_partition.decision_step_ns
        for bar_index in candidate_bars
    )
    excess = tuple(
        math.log1p(candidate_return) - math.log1p(cash_return)
        for candidate_return, cash_return in zip(
            candidate_replay.net_simple_returns,
            cash_replay.net_simple_returns,
            strict=True,
        )
    )
    return UniversalTradeRLU2PairedReplayScopeEvidence(
        training_seed=seed,
        source_window=candidate_replay.source_window,
        cell=candidate_replay.cell,
        concrete_symbol=candidate_replay.concrete_symbol,
        scope_digest=candidate_replay.scope_digest,
        evaluation_dataset_digest=candidate_replay.evaluation_dataset_digest,
        paired_candidate_checkpoint_digest=expected_checkpoint,
        candidate_replay_evidence_digest=candidate_replay.digest,
        cash_replay_evidence_digest=cash_replay.digest,
        decision_timestamps_ns=timestamps,
        candidate_minus_cash_net_log_excess=excess,
    )


'''
if pair_insert_anchor not in text:
    raise SystemExit("Task5 paired replay insertion anchor missing")
text = text.replace(pair_insert_anchor, pair_block + pair_insert_anchor, 1)

segment_anchor = '''@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2DevelopmentBootstrapResult:
'''
paired_segment_block = '''@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2PairedBootstrapSegment(
    UniversalTradeRLU2ReducedBootstrapSegment
):
    """Reduced D1/D2 bootstrap segment with exact candidate/cash provenance."""

    paired_scope_evidence_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        UniversalTradeRLU2ReducedBootstrapSegment.__post_init__(self)
        provenance = tuple(self.paired_scope_evidence_digests)
        if not provenance:
            raise ValueError("U2 paired bootstrap segment requires cash pairing provenance")
        for digest in provenance:
            require_sha256(digest, field="U2 paired bootstrap scope evidence digest")
        if len(set(provenance)) != len(provenance):
            raise ValueError("U2 paired bootstrap provenance must be unique")
        object.__setattr__(self, "paired_scope_evidence_digests", provenance)

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "schema_version": U2_PAIRED_BOOTSTRAP_SEGMENT_SCHEMA,
                "source_window": self.source_window,
                "decision_timestamps_ns": self.decision_timestamps_ns,
                "net_log_excess": self.net_log_excess,
                "paired_scope_evidence_digests": self.paired_scope_evidence_digests,
            }
        )


def reduce_universal_trade_rl_u2_paired_replay_evidence(
    *,
    pairs: tuple[UniversalTradeRLU2PairedReplayScopeEvidence, ...],
    expected_symbols: tuple[str, ...],
) -> tuple[UniversalTradeRLU2PairedBootstrapSegment, ...]:
    """Reduce exact D1/D2 candidate-cash pairs and preserve their provenance."""

    resolved = tuple(pairs)
    symbols = tuple(expected_symbols)
    if not resolved:
        raise ValueError("U2 pairing reducer requires paired replay evidence")
    if not symbols or symbols != tuple(sorted(set(symbols))):
        raise ValueError("U2 pairing reducer symbols must be sorted and unique")
    if any(
        not isinstance(pair, UniversalTradeRLU2PairedReplayScopeEvidence)
        for pair in resolved
    ):
        raise TypeError("U2 pairing reducer contains invalid paired evidence")

    window_cells = (
        ("development_future_1", "D1"),
        ("development_future_2", "D2"),
    )
    expected_identities = tuple(
        (seed, window, cell, symbol)
        for seed in U2_TRAINING_SEEDS
        for window, cell in window_cells
        for symbol in symbols
    )
    identities = tuple(
        (pair.training_seed, pair.source_window, pair.cell, pair.concrete_symbol)
        for pair in resolved
    )
    if len(set(identities)) != len(identities):
        raise ValueError("U2 pairing reducer contains duplicate seed/window/symbol identity")
    if set(identities) != set(expected_identities) or len(identities) != len(
        expected_identities
    ):
        raise ValueError(
            "U2 pairing closure must cover every seed/window/symbol exactly once"
        )
    by_identity = {
        (pair.training_seed, pair.source_window, pair.cell, pair.concrete_symbol): pair
        for pair in resolved
    }

    segments: list[UniversalTradeRLU2PairedBootstrapSegment] = []
    for window, cell in window_cells:
        scoped = tuple(
            by_identity[(seed, window, cell, symbol)]
            for seed in U2_TRAINING_SEEDS
            for symbol in symbols
        )
        timestamp_closure = {pair.decision_timestamps_ns for pair in scoped}
        if len(timestamp_closure) != 1:
            raise ValueError("U2 pairing window decision timestamp closure drifted")
        timestamps = next(iter(timestamp_closure))
        reduced: list[float] = []
        for offset in range(len(timestamps)):
            seed_means = tuple(
                fmean(
                    by_identity[(seed, window, cell, symbol)]
                    .candidate_minus_cash_net_log_excess[offset]
                    for symbol in symbols
                )
                for seed in U2_TRAINING_SEEDS
            )
            reduced.append(float(median(seed_means)))
        segments.append(
            UniversalTradeRLU2PairedBootstrapSegment(
                source_window=window,
                decision_timestamps_ns=timestamps,
                net_log_excess=tuple(reduced),
                paired_scope_evidence_digests=tuple(pair.digest for pair in scoped),
            )
        )
    return tuple(segments)


'''
if segment_anchor not in text:
    raise SystemExit("Task5 paired segment insertion anchor missing")
text = text.replace(segment_anchor, paired_segment_block + segment_anchor, 1)

bootstrap_field_old = '''    passed: bool
    quantile_method: str = _U2_BOOTSTRAP_QUANTILE_METHOD
'''
bootstrap_field_new = '''    passed: bool
    paired_scope_evidence_digests: tuple[str, ...] = ()
    quantile_method: str = _U2_BOOTSTRAP_QUANTILE_METHOD
'''
if bootstrap_field_old not in text:
    raise SystemExit("Task5 bootstrap provenance field anchor missing")
text = text.replace(bootstrap_field_old, bootstrap_field_new, 1)

bootstrap_validation_anchor = '''        object.__setattr__(self, "source_windows", source_windows)
        object.__setattr__(self, "segment_digests", segment_digests)
        object.__setattr__(self, "block_lengths", block_lengths)
        for field_name in (
'''
bootstrap_validation_new = '''        object.__setattr__(self, "source_windows", source_windows)
        object.__setattr__(self, "segment_digests", segment_digests)
        object.__setattr__(self, "block_lengths", block_lengths)
        pairing_provenance = tuple(self.paired_scope_evidence_digests)
        for digest in pairing_provenance:
            require_sha256(digest, field="U2 seed-robustness cash-pair provenance")
        if len(set(pairing_provenance)) != len(pairing_provenance):
            raise ValueError("U2 seed-robustness cash-pair provenance must be unique")
        object.__setattr__(
            self,
            "paired_scope_evidence_digests",
            pairing_provenance,
        )
        for field_name in (
'''
if bootstrap_validation_anchor not in text:
    raise SystemExit("Task5 bootstrap validation anchor missing")
text = text.replace(bootstrap_validation_anchor, bootstrap_validation_new, 1)

bootstrap_payload_old = '''            "block_lengths": self.block_lengths,
            "observed_mean": self.observed_mean,
'''
bootstrap_payload_new = '''            "block_lengths": self.block_lengths,
            "paired_scope_evidence_digests": self.paired_scope_evidence_digests,
            "observed_mean": self.observed_mean,
'''
if bootstrap_payload_old not in text:
    raise SystemExit("Task5 bootstrap payload anchor missing")
text = text.replace(bootstrap_payload_old, bootstrap_payload_new, 1)

bootstrap_return_old = '''        confidence_level=confidence,
        passed=lower_ci > lower_ci_min,
    )
'''
bootstrap_return_new = '''        confidence_level=confidence,
        passed=lower_ci > lower_ci_min,
        paired_scope_evidence_digests=tuple(
            digest
            for segment in resolved
            for digest in getattr(segment, "paired_scope_evidence_digests", ())
        ),
    )
'''
if bootstrap_return_old not in text:
    raise SystemExit("Task5 bootstrap result construction anchor missing")
text = text.replace(bootstrap_return_old, bootstrap_return_new, 1)

final_provenance_anchor = '''        object.__setattr__(self, "seed_robustness_gates", robustness)

        primary_by_cell = {gate.cell: gate for gate in primary}
'''
final_provenance_new = '''        object.__setattr__(self, "seed_robustness_gates", robustness)
        if any(
            not gate.bootstrap_result.paired_scope_evidence_digests
            for gate in robustness
        ):
            raise ValueError(
                "U2 final Selection requires candidate/cash pairing provenance in every bootstrap"
            )

        primary_by_cell = {gate.cell: gate for gate in primary}
'''
if final_provenance_anchor not in text:
    raise SystemExit("Task5 final provenance gate anchor missing")
text = text.replace(final_provenance_anchor, final_provenance_new, 1)

aggregate_anchor = '''        if (
            aggregate.bootstrap_result.source_windows != expected_windows
            or aggregate.bootstrap_result.segment_digests != expected_segment_digests
            or aggregate.bootstrap_result.block_lengths != expected_block_lengths
        ):
            raise ValueError(
                "U2 final Selection D1+D2 aggregate bootstrap identity mismatch"
            )

        reasons = tuple(
'''
aggregate_new = '''        if (
            aggregate.bootstrap_result.source_windows != expected_windows
            or aggregate.bootstrap_result.segment_digests != expected_segment_digests
            or aggregate.bootstrap_result.block_lengths != expected_block_lengths
        ):
            raise ValueError(
                "U2 final Selection D1+D2 aggregate bootstrap identity mismatch"
            )
        expected_pairing_provenance = (
            d1.bootstrap_result.paired_scope_evidence_digests
            + d2.bootstrap_result.paired_scope_evidence_digests
        )
        if (
            aggregate.bootstrap_result.paired_scope_evidence_digests
            != expected_pairing_provenance
        ):
            raise ValueError(
                "U2 final Selection D1+D2 candidate/cash pairing provenance mismatch"
            )

        reasons = tuple(
'''
if aggregate_anchor not in text:
    raise SystemExit("Task5 aggregate provenance anchor missing")
text = text.replace(aggregate_anchor, aggregate_new, 1)

all_anchor = '''    "U2_DEVELOPMENT_BOOTSTRAP_SCHEMA",
    "U2_DEVELOPMENT_PANEL_SCHEMA",
    "U2_DEVELOPMENT_WINDOWS",
'''
all_new = '''    "U2_DEVELOPMENT_BOOTSTRAP_SCHEMA",
    "U2_DEVELOPMENT_PANEL_SCHEMA",
    "U2_DEVELOPMENT_WINDOWS",
    "U2_PAIRED_REPLAY_SCOPE_SCHEMA",
    "U2_PAIRED_BOOTSTRAP_SEGMENT_SCHEMA",
'''
if all_anchor not in text:
    raise SystemExit("Task5 __all__ constant anchor missing")
text = text.replace(all_anchor, all_new, 1)

all_type_anchor = '''    "UniversalTradeRLU2DevelopmentBootstrapResult",
    "UniversalTradeRLU2PairedExcessPoint",
    "UniversalTradeRLU2ReducedBootstrapSegment",
'''
all_type_new = '''    "UniversalTradeRLU2DevelopmentBootstrapResult",
    "UniversalTradeRLU2PairedReplayScopeEvidence",
    "UniversalTradeRLU2PairedBootstrapSegment",
    "UniversalTradeRLU2PairedExcessPoint",
    "UniversalTradeRLU2ReducedBootstrapSegment",
'''
if all_type_anchor not in text:
    raise SystemExit("Task5 __all__ type anchor missing")
text = text.replace(all_type_anchor, all_type_new, 1)

all_function_anchor = '''    "bootstrap_universal_trade_rl_u2_development_panel",
    "build_universal_trade_rl_u2_selection_leaf_metrics",
    "reduce_universal_trade_rl_u2_development_panel",
'''
all_function_new = '''    "bootstrap_universal_trade_rl_u2_development_panel",
    "build_universal_trade_rl_u2_paired_replay_scope_evidence",
    "build_universal_trade_rl_u2_selection_leaf_metrics",
    "reduce_universal_trade_rl_u2_development_panel",
    "reduce_universal_trade_rl_u2_paired_replay_evidence",
'''
if all_function_anchor not in text:
    raise SystemExit("Task5 __all__ function anchor missing")
text = text.replace(all_function_anchor, all_function_new, 1)

path.write_text(text, encoding="utf-8")
