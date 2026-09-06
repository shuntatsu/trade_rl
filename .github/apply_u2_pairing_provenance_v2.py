from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, got {count}")
    return text.replace(old, new, 1)


source = Path("trade_rl/workflows/universal_trade_rl_u2_selection.py")
text = source.read_text(encoding="utf-8")

replay_import = '''from trade_rl.workflows.universal_trade_rl_u2_replay import (
    UniversalTradeRLU2ReplayEvidence,
    UniversalTradeRLU2ReplayVariant,
)
'''
time_import = replay_import + '''from trade_rl.workflows.universal_trade_rl_u2_time_partition import (
    U2_DECISION_STEP_NS,
    UniversalTradeRLU2TimePartition,
)
'''
text = replace_once(text, replay_import, time_import, "time-partition import")

pair_block = '''U2_PAIRED_REPLAY_SCOPE_SCHEMA: Final = "universal_trade_rl_u2_paired_replay_scope_v1"
_U2_PAIRED_REPLAY_WINDOW_CELLS: Final = {
    "development_future_1": "D1",
    "development_future_2": "D2",
}


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2PairedReplayScopeEvidence:
    """Exact same-scope candidate/cash replay provenance for one leaf."""

    training_seed: int
    source_window: str
    cell: str
    concrete_symbol: str
    scope_digest: str
    evaluation_dataset_digest: str
    u2_contract_digest: str
    time_partition_digest: str
    checkpoint_closure_digest: str
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
        expected_cell = _U2_PAIRED_REPLAY_WINDOW_CELLS.get(self.source_window)
        if expected_cell is None or self.cell != expected_cell:
            raise ValueError("U2 paired replay window/cell identity is invalid")
        if not isinstance(self.concrete_symbol, str) or not self.concrete_symbol:
            raise ValueError("U2 paired replay symbol must be non-empty")
        for field_name, value in (
            ("scope digest", self.scope_digest),
            ("evaluation dataset digest", self.evaluation_dataset_digest),
            ("U2 contract digest", self.u2_contract_digest),
            ("time partition digest", self.time_partition_digest),
            ("checkpoint closure digest", self.checkpoint_closure_digest),
            ("paired candidate checkpoint digest", self.paired_candidate_checkpoint_digest),
            ("candidate replay evidence digest", self.candidate_replay_evidence_digest),
            ("cash replay evidence digest", self.cash_replay_evidence_digest),
        ):
            require_sha256(value, field=f"U2 paired replay {field_name}")

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

        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 paired replay scope evidence digest")
            if self.digest != expected:
                raise ValueError("U2 paired replay scope evidence digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def identity(self) -> tuple[int, str, str, str]:
        return (
            self.training_seed,
            self.source_window,
            self.concrete_symbol,
            self.scope_digest,
        )

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "training_seed": self.training_seed,
            "source_window": self.source_window,
            "cell": self.cell,
            "concrete_symbol": self.concrete_symbol,
            "scope_digest": self.scope_digest,
            "evaluation_dataset_digest": self.evaluation_dataset_digest,
            "u2_contract_digest": self.u2_contract_digest,
            "time_partition_digest": self.time_partition_digest,
            "checkpoint_closure_digest": self.checkpoint_closure_digest,
            "paired_candidate_checkpoint_digest": self.paired_candidate_checkpoint_digest,
            "candidate_replay_evidence_digest": self.candidate_replay_evidence_digest,
            "cash_replay_evidence_digest": self.cash_replay_evidence_digest,
            "decision_timestamps_ns": self.decision_timestamps_ns,
            "candidate_minus_cash_net_log_excess": self.candidate_minus_cash_net_log_excess,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def build_universal_trade_rl_u2_paired_replay_scope_evidence(
    *,
    candidate_replay: UniversalTradeRLU2ReplayEvidence,
    cash_replay: UniversalTradeRLU2ReplayEvidence,
    u2_contract: u2_contract.UniversalTradeRLU2Contract,
    time_partition: UniversalTradeRLU2TimePartition,
    checkpoint_closure: UniversalTradeRLU2FinalCheckpointClosure,
) -> UniversalTradeRLU2PairedReplayScopeEvidence:
    """Derive exact candidate-minus-cash log excess from same-scope replay evidence."""

    if not isinstance(candidate_replay, UniversalTradeRLU2ReplayEvidence):
        raise TypeError("U2 paired replay candidate evidence is invalid")
    if not isinstance(cash_replay, UniversalTradeRLU2ReplayEvidence):
        raise TypeError("U2 paired replay cash evidence is invalid")
    if candidate_replay.policy_variant != UniversalTradeRLU2ReplayVariant.CANDIDATE.value:
        raise ValueError("U2 paired replay requires candidate replay evidence")
    if cash_replay.policy_variant != UniversalTradeRLU2ReplayVariant.CASH.value:
        raise ValueError("U2 paired replay requires cash replay evidence")
    if candidate_replay.evaluation_seed != cash_replay.evaluation_seed:
        raise ValueError("U2 paired replay seed identity mismatch")

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
        "outcome_start_bar_index",
        "outcome_stop_bar_index_exclusive",
        "evaluation_start_bar_index",
        "evaluation_stop_bar_index",
        "runtime_start_bar_index",
        "runtime_end_bar_index",
        "observed_decision_count",
    )
    for field_name in identity_fields:
        if getattr(candidate_replay, field_name) != getattr(cash_replay, field_name):
            raise ValueError(f"U2 paired replay {field_name} identity mismatch")

    if candidate_replay.u2_contract_digest != u2_contract.digest:
        raise ValueError("U2 paired replay contract identity mismatch")
    if not isinstance(time_partition, UniversalTradeRLU2TimePartition):
        raise TypeError("U2 paired replay time partition is invalid")
    if time_partition.digest != u2_contract.time_partition_digest:
        raise ValueError("U2 paired replay time partition identity mismatch")
    if time_partition.universe_manifest_digest != candidate_replay.universe_manifest_digest:
        raise ValueError("U2 paired replay universe/time identity mismatch")
    if not isinstance(checkpoint_closure, UniversalTradeRLU2FinalCheckpointClosure):
        raise TypeError("U2 paired replay checkpoint closure is invalid")
    if checkpoint_closure.u2_contract_digest != u2_contract.digest:
        raise ValueError("U2 paired replay checkpoint closure U2 identity mismatch")
    if checkpoint_closure.time_partition_digest != time_partition.digest:
        raise ValueError("U2 paired replay checkpoint/time identity mismatch")

    seed = candidate_replay.evaluation_seed
    checkpoints = dict(checkpoint_closure.checkpoint_digests)
    if seed not in checkpoints:
        raise ValueError("U2 paired replay checkpoint seed closure is incomplete")
    expected_checkpoint = checkpoints[seed]
    if (
        candidate_replay.paired_candidate_checkpoint_digest != expected_checkpoint
        or cash_replay.paired_candidate_checkpoint_digest != expected_checkpoint
    ):
        raise ValueError("U2 paired replay candidate checkpoint identity mismatch")

    candidate_steps = tuple(candidate_replay.step_evidence)
    cash_steps = tuple(cash_replay.step_evidence)
    if (
        len(candidate_steps) != candidate_replay.observed_decision_count
        or len(cash_steps) != cash_replay.observed_decision_count
    ):
        raise ValueError("U2 paired replay step evidence closure is incomplete")
    candidate_indices = tuple(step.decision_bar_index for step in candidate_steps)
    cash_indices = tuple(step.decision_bar_index for step in cash_steps)
    if candidate_indices != cash_indices:
        raise ValueError("U2 paired replay decision path identity mismatch")
    if (
        len(candidate_replay.net_simple_returns) != len(candidate_indices)
        or len(cash_replay.net_simple_returns) != len(candidate_indices)
    ):
        raise ValueError("U2 paired replay return path closure is incomplete")

    timestamps = tuple(
        time_partition.common_first_timestamp_ns + index * U2_DECISION_STEP_NS
        for index in candidate_indices
    )
    if timestamps and timestamps[-1] > time_partition.common_last_timestamp_ns:
        raise ValueError("U2 paired replay decision timestamp exceeds partition")
    excess: list[float] = []
    for candidate_return, cash_return in zip(
        candidate_replay.net_simple_returns,
        cash_replay.net_simple_returns,
        strict=True,
    ):
        if candidate_return <= -1.0 or cash_return <= -1.0:
            raise ValueError("U2 paired replay simple return must be greater than -1")
        excess.append(math.log1p(candidate_return) - math.log1p(cash_return))

    return UniversalTradeRLU2PairedReplayScopeEvidence(
        training_seed=seed,
        source_window=candidate_replay.source_window,
        cell=candidate_replay.cell,
        concrete_symbol=candidate_replay.concrete_symbol,
        scope_digest=candidate_replay.scope_digest,
        evaluation_dataset_digest=candidate_replay.evaluation_dataset_digest,
        u2_contract_digest=u2_contract.digest,
        time_partition_digest=time_partition.digest,
        checkpoint_closure_digest=checkpoint_closure.digest,
        paired_candidate_checkpoint_digest=expected_checkpoint,
        candidate_replay_evidence_digest=candidate_replay.digest,
        cash_replay_evidence_digest=cash_replay.digest,
        decision_timestamps_ns=timestamps,
        candidate_minus_cash_net_log_excess=tuple(excess),
    )


'''
insert_anchor = '@dataclass(frozen=True, slots=True)\nclass UniversalTradeRLU2PairedExcessPoint:'
text = replace_once(text, insert_anchor, pair_block + insert_anchor, "paired evidence insertion")

segment_fields = '''    source_window: str
    decision_timestamps_ns: tuple[int, ...]
    net_log_excess: tuple[float, ...]
'''
segment_fields_new = segment_fields + '''    paired_scope_evidence: tuple[UniversalTradeRLU2PairedReplayScopeEvidence, ...] = ()
'''
text = replace_once(text, segment_fields, segment_fields_new, "reduced segment fields")

segment_values = '''        values = tuple(
            _finite(value, field="U2 bootstrap segment net-log excess")
            for value in self.net_log_excess
        )
'''
segment_values_new = segment_values + '''        paired_scope_evidence = tuple(self.paired_scope_evidence)
        if any(
            not isinstance(pair, UniversalTradeRLU2PairedReplayScopeEvidence)
            for pair in paired_scope_evidence
        ):
            raise TypeError("U2 bootstrap segment paired replay evidence is invalid")
        paired_scope_evidence = tuple(
            sorted(paired_scope_evidence, key=lambda pair: pair.identity)
        )
        if len({pair.identity for pair in paired_scope_evidence}) != len(
            paired_scope_evidence
        ):
            raise ValueError("U2 bootstrap segment paired replay identity is duplicated")
        if any(
            pair.source_window != self.source_window for pair in paired_scope_evidence
        ):
            raise ValueError("U2 bootstrap segment paired replay window mismatch")
'''
text = replace_once(text, segment_values, segment_values_new, "segment paired validation")
text = replace_once(
    text,
    '        object.__setattr__(self, "net_log_excess", values)\n',
    '        object.__setattr__(self, "net_log_excess", values)\n'
    '        object.__setattr__(self, "paired_scope_evidence", paired_scope_evidence)\n',
    "segment paired assignment",
)

segment_digest = '''    @property
    def digest(self) -> str:
        return content_digest(
            {
                "schema_version": U2_DEVELOPMENT_PANEL_SCHEMA,
                "source_window": self.source_window,
                "decision_timestamps_ns": self.decision_timestamps_ns,
                "net_log_excess": self.net_log_excess,
            }
        )
'''
segment_digest_new = '''    @property
    def paired_scope_evidence_digests(self) -> tuple[str, ...]:
        return tuple(pair.digest for pair in self.paired_scope_evidence)

    @property
    def digest(self) -> str:
        payload: dict[str, object] = {
            "schema_version": U2_DEVELOPMENT_PANEL_SCHEMA,
            "source_window": self.source_window,
            "decision_timestamps_ns": self.decision_timestamps_ns,
            "net_log_excess": self.net_log_excess,
        }
        if self.paired_scope_evidence:
            payload["paired_scope_evidence_digests"] = self.paired_scope_evidence_digests
        return content_digest(payload)
'''
text = replace_once(text, segment_digest, segment_digest_new, "segment digest")

paired_reducer = '''def _reduce_u2_paired_replay_evidence_for_windows(
    *,
    pairs: tuple[UniversalTradeRLU2PairedReplayScopeEvidence, ...],
    expected_symbols: tuple[str, ...],
    expected_windows: tuple[str, ...],
) -> tuple[UniversalTradeRLU2ReducedBootstrapSegment, ...]:
    resolved = tuple(pairs)
    symbols = tuple(expected_symbols)
    windows = tuple(expected_windows)
    if not resolved:
        raise ValueError("U2 paired replay reduction requires evidence")
    if any(
        not isinstance(pair, UniversalTradeRLU2PairedReplayScopeEvidence)
        for pair in resolved
    ):
        raise TypeError("U2 paired replay reduction evidence is invalid")
    if not symbols or symbols != tuple(sorted(set(symbols))):
        raise ValueError("U2 paired replay expected symbols must be sorted and unique")
    if windows != tuple(window for window in U2_DEVELOPMENT_WINDOWS if window in windows):
        raise ValueError("U2 paired replay expected windows are not canonical")
    if {pair.source_window for pair in resolved} != set(windows):
        raise ValueError("U2 paired replay window closure is incomplete")
    if {pair.training_seed for pair in resolved} != set(U2_TRAINING_SEEDS):
        raise ValueError("U2 paired replay seed closure is incomplete")
    if {pair.concrete_symbol for pair in resolved} != set(symbols):
        raise ValueError("U2 paired replay symbol closure is incomplete")
    identities = tuple(pair.identity for pair in resolved)
    if len(set(identities)) != len(identities):
        raise ValueError("U2 paired replay contains duplicate scope identity")
    for field_name in (
        "u2_contract_digest",
        "time_partition_digest",
        "checkpoint_closure_digest",
    ):
        if len({getattr(pair, field_name) for pair in resolved}) != 1:
            raise ValueError(f"U2 paired replay {field_name} identity drifted")

    closure_by_seed = {
        seed: {
            (pair.cell, pair.concrete_symbol, pair.scope_digest)
            for pair in resolved
            if pair.training_seed == seed
        }
        for seed in U2_TRAINING_SEEDS
    }
    canonical_closure = closure_by_seed[U2_TRAINING_SEEDS[0]]
    if not canonical_closure or any(
        closure_by_seed[seed] != canonical_closure for seed in U2_TRAINING_SEEDS[1:]
    ):
        raise ValueError("U2 paired replay scope closure drifted across seeds")

    by_window_timestamp: dict[
        tuple[str, int], dict[int, dict[str, float]]
    ] = defaultdict(lambda: defaultdict(dict))
    for pair in resolved:
        for timestamp, value in zip(
            pair.decision_timestamps_ns,
            pair.candidate_minus_cash_net_log_excess,
            strict=True,
        ):
            symbol_map = by_window_timestamp[(pair.source_window, timestamp)][
                pair.training_seed
            ]
            if pair.concrete_symbol in symbol_map:
                raise ValueError("U2 paired replay timestamp identity is duplicated")
            symbol_map[pair.concrete_symbol] = value

    segments: list[UniversalTradeRLU2ReducedBootstrapSegment] = []
    for source_window in windows:
        timestamps = tuple(
            sorted(
                timestamp
                for window, timestamp in by_window_timestamp
                if window == source_window
            )
        )
        if not timestamps:
            raise ValueError("U2 paired replay window has no timestamps")
        reduced: list[float] = []
        for timestamp in timestamps:
            seed_map = by_window_timestamp[(source_window, timestamp)]
            if set(seed_map) != set(U2_TRAINING_SEEDS):
                raise ValueError("U2 paired replay timestamp seed closure is incomplete")
            seed_means: list[float] = []
            for seed in U2_TRAINING_SEEDS:
                symbol_map = seed_map[seed]
                if set(symbol_map) != set(symbols):
                    raise ValueError("U2 paired replay timestamp symbol closure is incomplete")
                seed_means.append(fmean(symbol_map[symbol] for symbol in symbols))
            reduced.append(float(median(seed_means)))
        window_pairs = tuple(
            sorted(
                (pair for pair in resolved if pair.source_window == source_window),
                key=lambda pair: pair.identity,
            )
        )
        segments.append(
            UniversalTradeRLU2ReducedBootstrapSegment(
                source_window=source_window,
                decision_timestamps_ns=timestamps,
                net_log_excess=tuple(reduced),
                paired_scope_evidence=window_pairs,
            )
        )
    return tuple(segments)


def reduce_universal_trade_rl_u2_paired_replay_evidence(
    *,
    pairs: tuple[UniversalTradeRLU2PairedReplayScopeEvidence, ...],
    expected_symbols: tuple[str, ...],
) -> tuple[UniversalTradeRLU2ReducedBootstrapSegment, ...]:
    """Reduce exact candidate/cash pair evidence without losing provenance."""

    return _reduce_u2_paired_replay_evidence_for_windows(
        pairs=tuple(pairs),
        expected_symbols=tuple(expected_symbols),
        expected_windows=U2_DEVELOPMENT_WINDOWS,
    )


'''
bootstrap_anchor = 'def bootstrap_universal_trade_rl_u2_development_panel(\n'
text = replace_once(text, bootstrap_anchor, paired_reducer + bootstrap_anchor, "paired reducer insertion")

bootstrap_fields = '''    confidence_level: float
    passed: bool
    quantile_method: str = _U2_BOOTSTRAP_QUANTILE_METHOD
'''
bootstrap_fields_new = '''    confidence_level: float
    passed: bool
    paired_scope_evidence_digests: tuple[str, ...] = ()
    quantile_method: str = _U2_BOOTSTRAP_QUANTILE_METHOD
'''
text = replace_once(text, bootstrap_fields, bootstrap_fields_new, "robustness bootstrap fields")

bootstrap_assign = '''        object.__setattr__(self, "block_lengths", block_lengths)
'''
bootstrap_assign_new = bootstrap_assign + '''        paired_digests = tuple(self.paired_scope_evidence_digests)
        for digest in paired_digests:
            require_sha256(digest, field="U2 seed-robustness paired replay digest")
        if len(set(paired_digests)) != len(paired_digests):
            raise ValueError("U2 seed-robustness paired replay digests must be unique")
        object.__setattr__(self, "paired_scope_evidence_digests", paired_digests)
'''
text = replace_once(text, bootstrap_assign, bootstrap_assign_new, "bootstrap paired validation")

bootstrap_payload_tail = '''            "passed": self.passed,
        }
        if include_digest:
'''
bootstrap_payload_tail_new = '''            "passed": self.passed,
        }
        if self.paired_scope_evidence_digests:
            payload["paired_scope_evidence_digests"] = self.paired_scope_evidence_digests
        if include_digest:
'''
bootstrap_class_start = text.index("class UniversalTradeRLU2SeedRobustnessBootstrapResult:")
bootstrap_class_end = text.index("\ndef _bootstrap_universal_trade_rl_u2_seed_robustness(", bootstrap_class_start)
bootstrap_section = text[bootstrap_class_start:bootstrap_class_end]
bootstrap_section = replace_once(
    bootstrap_section,
    bootstrap_payload_tail,
    bootstrap_payload_tail_new,
    "bootstrap payload",
)
text = text[:bootstrap_class_start] + bootstrap_section + text[bootstrap_class_end:]

bootstrap_return = '''        confidence_level=confidence,
        passed=lower_ci > lower_ci_min,
    )
'''
bootstrap_return_new = '''        confidence_level=confidence,
        passed=lower_ci > lower_ci_min,
        paired_scope_evidence_digests=tuple(
            digest
            for segment in resolved
            for digest in segment.paired_scope_evidence_digests
        ),
    )
'''
text = replace_once(text, bootstrap_return, bootstrap_return_new, "bootstrap paired propagation")

robust_fields = '''    rejection_reasons: tuple[str, ...]
    passed: bool
    scope_closure_digest: str = field(init=False)
'''
robust_fields_new = '''    rejection_reasons: tuple[str, ...]
    passed: bool
    paired_scope_evidence: tuple[UniversalTradeRLU2PairedReplayScopeEvidence, ...] = ()
    scope_closure_digest: str = field(init=False)
'''
text = replace_once(text, robust_fields, robust_fields_new, "robustness fields")

robust_scope_check = '''        if self.bootstrap_result.source_windows != expected_windows:
            raise ValueError("U2 seed-robustness bootstrap scope is inconsistent")
'''
robust_scope_check_new = robust_scope_check + '''        paired_scope_evidence = tuple(self.paired_scope_evidence)
        if any(
            not isinstance(pair, UniversalTradeRLU2PairedReplayScopeEvidence)
            for pair in paired_scope_evidence
        ):
            raise TypeError("U2 seed-robustness paired replay evidence is invalid")
        if paired_scope_evidence:
            paired_scope_evidence = tuple(
                sorted(paired_scope_evidence, key=lambda pair: pair.identity)
            )
            paired_digests = tuple(pair.digest for pair in paired_scope_evidence)
            if paired_digests != self.bootstrap_result.paired_scope_evidence_digests:
                raise ValueError("U2 seed-robustness bootstrap pairing provenance mismatch")
            paired_closure = tuple(
                sorted(
                    {
                        (pair.cell, pair.concrete_symbol, pair.scope_digest)
                        for pair in paired_scope_evidence
                    }
                )
            )
            if paired_closure != scope_closure:
                raise ValueError("U2 seed-robustness paired replay scope closure mismatch")
            if {pair.training_seed for pair in paired_scope_evidence} != set(U2_TRAINING_SEEDS):
                raise ValueError("U2 seed-robustness paired replay seed closure is incomplete")
            if {pair.source_window for pair in paired_scope_evidence} != set(expected_windows):
                raise ValueError("U2 seed-robustness paired replay window closure is incomplete")
        elif self.bootstrap_result.paired_scope_evidence_digests:
            raise ValueError("U2 seed-robustness paired replay evidence is missing")
        object.__setattr__(self, "paired_scope_evidence", paired_scope_evidence)
'''
text = replace_once(text, robust_scope_check, robust_scope_check_new, "robustness paired validation")

robust_payload_tail = '''            "passed": self.passed,
        }
        if include_digest:
'''
robust_payload_tail_new = '''            "passed": self.passed,
        }
        if self.paired_scope_evidence:
            payload["paired_scope_evidence_digests"] = tuple(
                pair.digest for pair in self.paired_scope_evidence
            )
        if include_digest:
'''
robust_class_start = text.index("class UniversalTradeRLU2SeedRobustnessEvidence:")
robust_class_end = text.index("\ndef evaluate_universal_trade_rl_u2_seed_robustness(", robust_class_start)
robust_section = text[robust_class_start:robust_class_end]
robust_section = replace_once(
    robust_section,
    robust_payload_tail,
    robust_payload_tail_new,
    "robustness payload",
)
text = text[:robust_class_start] + robust_section + text[robust_class_end:]

robust_return = '''        bootstrap_result=bootstrap_result,
        rejection_reasons=reasons,
        passed=not reasons,
    )
'''
robust_return_new = '''        bootstrap_result=bootstrap_result,
        rejection_reasons=reasons,
        passed=not reasons,
        paired_scope_evidence=tuple(
            pair
            for segment in selected_segments
            for pair in segment.paired_scope_evidence
        ),
    )
'''
text = replace_once(text, robust_return, robust_return_new, "robustness paired propagation")

final_set = '''        object.__setattr__(self, "seed_robustness_gates", robustness)

        primary_by_cell = {gate.cell: gate for gate in primary}
'''
final_set_new = '''        object.__setattr__(self, "seed_robustness_gates", robustness)

        checkpoints = dict(self.checkpoint_closure.checkpoint_digests)
        for gate in robustness:
            if not gate.paired_scope_evidence:
                raise ValueError(
                    "U2 final Selection requires exact candidate/cash pairing provenance"
                )
            for pair in gate.paired_scope_evidence:
                if pair.u2_contract_digest != self.u2_contract.digest:
                    raise ValueError("U2 final Selection paired replay U2 identity mismatch")
                if pair.time_partition_digest != self.u2_contract.time_partition_digest:
                    raise ValueError("U2 final Selection paired replay time identity mismatch")
                if pair.checkpoint_closure_digest != self.checkpoint_closure.digest:
                    raise ValueError(
                        "U2 final Selection paired replay checkpoint closure mismatch"
                    )
                if pair.paired_candidate_checkpoint_digest != checkpoints[pair.training_seed]:
                    raise ValueError(
                        "U2 final Selection paired replay checkpoint identity mismatch"
                    )

        primary_by_cell = {gate.cell: gate for gate in primary}
'''
text = replace_once(text, final_set, final_set_new, "final pairing requirement")

aggregate_check = '''        if (
            aggregate.bootstrap_result.source_windows != expected_windows
            or aggregate.bootstrap_result.segment_digests != expected_segment_digests
            or aggregate.bootstrap_result.block_lengths != expected_block_lengths
        ):
            raise ValueError(
                "U2 final Selection D1+D2 aggregate bootstrap identity mismatch"
            )
'''
aggregate_check_new = aggregate_check + '''        standalone_pairing_digests = tuple(
            pair.digest
            for gate in (d1, d2)
            for pair in gate.paired_scope_evidence
        )
        aggregate_pairing_digests = tuple(
            pair.digest for pair in aggregate.paired_scope_evidence
        )
        if aggregate_pairing_digests != standalone_pairing_digests:
            raise ValueError(
                "U2 final Selection D1+D2 aggregate cash-pairing provenance mismatch"
            )
'''
text = replace_once(text, aggregate_check, aggregate_check_new, "aggregate pairing check")

exports = '''    "UniversalTradeRLU2PairedExcessPoint",
    "UniversalTradeRLU2ReducedBootstrapSegment",
'''
exports_new = '''    "U2_PAIRED_REPLAY_SCOPE_SCHEMA",
    "UniversalTradeRLU2PairedReplayScopeEvidence",
    "build_universal_trade_rl_u2_paired_replay_scope_evidence",
    "reduce_universal_trade_rl_u2_paired_replay_evidence",
    "UniversalTradeRLU2PairedExcessPoint",
    "UniversalTradeRLU2ReducedBootstrapSegment",
'''
text = replace_once(text, exports, exports_new, "pairing exports")
source.write_text(text, encoding="utf-8")

pairing_test = Path("tests/integrations/test_universal_trade_rl_u2_selection_pairing.py")
ptext = pairing_test.read_text(encoding="utf-8")
pair_direct = '''        evaluation_dataset_digest=content_digest(
            {"fixture": "pairing-dataset", "symbol": symbol}
        ),
        paired_candidate_checkpoint_digest=_checkpoint_digest(seed),
'''
pair_direct_new = '''        evaluation_dataset_digest=content_digest(
            {"fixture": "pairing-dataset", "symbol": symbol}
        ),
        u2_contract_digest=content_digest({"fixture": "pairing-u2-contract"}),
        time_partition_digest=content_digest({"fixture": "pairing-time-partition"}),
        checkpoint_closure_digest=content_digest({"fixture": "pairing-checkpoint-closure"}),
        paired_candidate_checkpoint_digest=_checkpoint_digest(seed),
'''
ptext = replace_once(ptext, pair_direct, pair_direct_new, "pairing synthetic identities")
pairing_test.write_text(ptext, encoding="utf-8")

final_test = Path("tests/workflows/test_universal_trade_rl_u2_selection_final.py")
ftext = final_test.read_text(encoding="utf-8")
summary_anchor = 'def _summary_for_seed(*, leaves, training_seed: int):\n'
paired_helper = '''def _paired_segments(*, u2_contract, checkpoint_closure):
    module = _module()
    checkpoints = dict(checkpoint_closure.checkpoint_digests)
    symbols = ("DEV_A", "DEV_B")
    pairs = tuple(
        module.UniversalTradeRLU2PairedReplayScopeEvidence(
            training_seed=seed,
            source_window=window,
            cell=cell,
            concrete_symbol=symbol,
            scope_digest=content_digest(
                {
                    "fixture": "u2-final-selection-tile",
                    "cell": cell,
                    "symbol": symbol,
                }
            ),
            evaluation_dataset_digest=content_digest(
                {"fixture": "u2-final-pairing-dataset", "symbol": symbol}
            ),
            u2_contract_digest=u2_contract.digest,
            time_partition_digest=u2_contract.time_partition_digest,
            checkpoint_closure_digest=checkpoint_closure.digest,
            paired_candidate_checkpoint_digest=checkpoints[seed],
            candidate_replay_evidence_digest=content_digest(
                {
                    "fixture": "u2-final-pairing-candidate",
                    "seed": seed,
                    "window": window,
                    "symbol": symbol,
                }
            ),
            cash_replay_evidence_digest=content_digest(
                {
                    "fixture": "u2-final-pairing-cash",
                    "seed": seed,
                    "window": window,
                    "symbol": symbol,
                }
            ),
            decision_timestamps_ns=timestamps,
            candidate_minus_cash_net_log_excess=(0.02, 0.02, 0.02, 0.02),
        )
        for seed in (0, 1, 2)
        for window, cell, timestamps in (
            ("development_future_1", "D1", (100, 200, 300, 400)),
            ("development_future_2", "D2", (500, 600, 700, 800)),
        )
        for symbol in symbols
    )
    return module.reduce_universal_trade_rl_u2_paired_replay_evidence(
        pairs=pairs,
        expected_symbols=symbols,
    )


'''
ftext = replace_once(ftext, summary_anchor, paired_helper + summary_anchor, "final paired helper")
ftext = replace_once(
    ftext,
    'def _passing_selection_children():\n',
    'def _passing_selection_children(*, u2_contract, checkpoint_closure):\n',
    "passing fixture signature",
)
ftext = replace_once(
    ftext,
    '    segments = _segments()\n',
    '    segments = _paired_segments(\n'
    '        u2_contract=u2_contract,\n'
    '        checkpoint_closure=checkpoint_closure,\n'
    '    )\n',
    "passing fixture segments",
)
ftext = replace_once(
    ftext,
    '    primary, robustness = _passing_selection_children()\n',
    '    primary, robustness = _passing_selection_children(\n'
    '        u2_contract=u2_contract,\n'
    '        checkpoint_closure=checkpoint_closure,\n'
    '    )\n',
    "final fixture paired children",
)

aggregate_old = '''        _u2_contract,
        _base_lock,
        _development_lock,
        _checkpoint_closure,
        primary,
        robustness,
    ) = _final_fixture()
    d1, d2, _d12 = robustness
'''
aggregate_new = '''        u2_contract,
        _base_lock,
        _development_lock,
        checkpoint_closure,
        primary,
        robustness,
    ) = _final_fixture()
    d1, d2, _d12 = robustness
'''
ftext = replace_once(ftext, aggregate_old, aggregate_new, "aggregate fixture identities")
ftext = replace_once(
    ftext,
    '        segments=_segments(),\n    )\n    assert altered_d12.passed is True\n',
    '        segments=_paired_segments(\n'
    '            u2_contract=u2_contract,\n'
    '            checkpoint_closure=checkpoint_closure,\n'
    '        ),\n'
    '    )\n'
    '    assert altered_d12.passed is True\n',
    "aggregate paired segments",
)

negative_old = '''    (
        _u2_contract,
        _base_lock,
        _development_lock,
        _checkpoint_closure,
        primary,
        robustness,
    ) = _final_fixture()

    with pytest.raises(ValueError, match="cash|pair|provenance|bootstrap"):
        _build_final(primary=primary, robustness=robustness)
'''
negative_new = '''    module = _module()
    (
        _u2_contract,
        _base_lock,
        _development_lock,
        _checkpoint_closure,
        primary,
        _robustness,
    ) = _final_fixture()
    d1_leaves = _cell_leaves("D1")
    d2_leaves = _cell_leaves("D2")
    legacy_segments = _segments()
    legacy_robustness = (
        module.evaluate_universal_trade_rl_u2_seed_robustness(
            scope="D1",
            leaves=d1_leaves,
            segments=legacy_segments,
        ),
        module.evaluate_universal_trade_rl_u2_seed_robustness(
            scope="D2",
            leaves=d2_leaves,
            segments=legacy_segments,
        ),
        module.evaluate_universal_trade_rl_u2_seed_robustness(
            scope="D1+D2",
            leaves=d1_leaves + d2_leaves,
            segments=legacy_segments,
        ),
    )
    assert all(gate.passed for gate in legacy_robustness)
    assert all(
        not gate.bootstrap_result.paired_scope_evidence_digests
        for gate in legacy_robustness
    )

    with pytest.raises(ValueError, match="cash|pair|provenance|bootstrap"):
        _build_final(primary=primary, robustness=legacy_robustness)
'''
ftext = replace_once(ftext, negative_old, negative_new, "final unpaired negative")
final_test.write_text(ftext, encoding="utf-8")
