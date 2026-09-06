from pathlib import Path


path = Path("trade_rl/workflows/universal_trade_rl_u2_selection.py")
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str, *, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, got {count}")
    text = text.replace(old, new, 1)


helper_anchor = '''def _positive_wealth(log_growth: float, *, field: str) -> float:
    try:
        wealth = math.exp(log_growth)
    except OverflowError as error:
        raise ValueError(f"{field} wealth overflowed") from error
    if not math.isfinite(wealth) or wealth <= 0.0:
        raise ValueError(f"{field} wealth must be finite and positive")
    return wealth
'''
helper_new = helper_anchor + '''

def _require_current_artifact_digest(artifact: object, *, field: str) -> None:
    """Reject a content-addressed artifact whose stored digest is stale."""

    digest = getattr(artifact, "digest", None)
    to_payload = getattr(artifact, "to_payload", None)
    if not isinstance(digest, str) or not callable(to_payload):
        raise TypeError(f"{field} must be a digest-bound artifact")
    require_sha256(digest, field=f"{field} digest")
    payload = to_payload(include_digest=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{field} payload must be an object")
    if digest != content_digest(payload):
        raise ValueError(f"{field} artifact digest mismatch")
'''
replace_once(helper_anchor, helper_new, label="artifact digest helper")

leaf_builder_anchor = '''    if not isinstance(replay_evidence, UniversalTradeRLU2ReplayEvidence):
        raise TypeError("U2 Selection leaf requires replay evidence")
    if (
        replay_evidence.policy_variant
'''
leaf_builder_new = '''    if not isinstance(replay_evidence, UniversalTradeRLU2ReplayEvidence):
        raise TypeError("U2 Selection leaf requires replay evidence")
    _require_current_artifact_digest(
        replay_evidence,
        field="U2 Selection leaf replay evidence",
    )
    if (
        replay_evidence.policy_variant
'''
replace_once(leaf_builder_anchor, leaf_builder_new, label="leaf replay integrity")

summary_anchor = '''    if any(
        not isinstance(leaf, UniversalTradeRLU2SelectionLeafMetrics)
        for leaf in resolved
    ):
        raise TypeError("U2 Selection summary contains an invalid leaf")
    identities = tuple(leaf.identity for leaf in resolved)
'''
summary_new = '''    if any(
        not isinstance(leaf, UniversalTradeRLU2SelectionLeafMetrics)
        for leaf in resolved
    ):
        raise TypeError("U2 Selection summary contains an invalid leaf")
    for leaf in resolved:
        _require_current_artifact_digest(
            leaf,
            field="U2 Selection summary leaf",
        )
    identities = tuple(leaf.identity for leaf in resolved)
'''
replace_once(summary_anchor, summary_new, label="summary leaf integrity")

pair_grid_anchor = '''        if timestamps != tuple(sorted(set(timestamps))):
            raise ValueError("U2 paired replay timestamps must be sorted and unique")
        object.__setattr__(self, "decision_timestamps_ns", timestamps)
'''
pair_grid_new = '''        if timestamps != tuple(sorted(set(timestamps))):
            raise ValueError("U2 paired replay timestamps must be sorted and unique")
        if len(timestamps) > 1 and any(
            later - earlier != U2_DECISION_STEP_NS
            for earlier, later in zip(timestamps, timestamps[1:])
        ):
            raise ValueError("U2 paired replay timestamps must follow the 15m grid")
        object.__setattr__(self, "decision_timestamps_ns", timestamps)
'''
replace_once(pair_grid_anchor, pair_grid_new, label="pair timestamp grid")

pair_builder_anchor = '''    if not isinstance(candidate_replay, UniversalTradeRLU2ReplayEvidence):
        raise TypeError("U2 paired replay candidate evidence is invalid")
    if not isinstance(cash_replay, UniversalTradeRLU2ReplayEvidence):
        raise TypeError("U2 paired replay cash evidence is invalid")
    if (
        candidate_replay.policy_variant
'''
pair_builder_new = '''    if not isinstance(candidate_replay, UniversalTradeRLU2ReplayEvidence):
        raise TypeError("U2 paired replay candidate evidence is invalid")
    if not isinstance(cash_replay, UniversalTradeRLU2ReplayEvidence):
        raise TypeError("U2 paired replay cash evidence is invalid")
    _require_current_artifact_digest(
        candidate_replay,
        field="U2 paired replay candidate evidence",
    )
    _require_current_artifact_digest(
        cash_replay,
        field="U2 paired replay cash evidence",
    )
    if (
        candidate_replay.policy_variant
'''
replace_once(pair_builder_anchor, pair_builder_new, label="pair source integrity")

seed_anchor = '''    seed = candidate_replay.evaluation_seed
    checkpoints = dict(checkpoint_closure.checkpoint_digests)
    if seed not in checkpoints:
        raise ValueError("U2 paired replay checkpoint seed closure is incomplete")
    expected_checkpoint = checkpoints[seed]
    if (
        candidate_replay.paired_candidate_checkpoint_digest != expected_checkpoint
        or cash_replay.paired_candidate_checkpoint_digest != expected_checkpoint
    ):
        raise ValueError("U2 paired replay candidate checkpoint identity mismatch")
'''
seed_new = '''    if (
        candidate_replay.paired_candidate_checkpoint_digest
        != cash_replay.paired_candidate_checkpoint_digest
    ):
        raise ValueError("U2 paired replay candidate checkpoint pair identity mismatch")
    expected_checkpoint = candidate_replay.paired_candidate_checkpoint_digest
    matching_seeds = tuple(
        seed
        for seed, digest in checkpoint_closure.checkpoint_digests
        if digest == expected_checkpoint
    )
    if len(matching_seeds) != 1:
        raise ValueError("U2 paired replay candidate checkpoint is outside exact closure")
    seed = matching_seeds[0]
'''
replace_once(seed_anchor, seed_new, label="training seed derivation")

reducer_anchor = '''    if any(
        not isinstance(pair, UniversalTradeRLU2PairedReplayScopeEvidence)
        for pair in resolved
    ):
        raise TypeError("U2 paired replay reduction evidence is invalid")
    if not symbols or symbols != tuple(sorted(set(symbols))):
'''
reducer_new = '''    if any(
        not isinstance(pair, UniversalTradeRLU2PairedReplayScopeEvidence)
        for pair in resolved
    ):
        raise TypeError("U2 paired replay reduction evidence is invalid")
    for pair in resolved:
        _require_current_artifact_digest(
            pair,
            field="U2 paired replay reduction evidence",
        )
    if not symbols or symbols != tuple(sorted(set(symbols))):
'''
replace_once(reducer_anchor, reducer_new, label="reducer pair integrity")

robust_leaf_anchor = '''    if any(
        not isinstance(leaf, UniversalTradeRLU2SelectionLeafMetrics)
        for leaf in resolved_leaves
    ):
        raise TypeError("U2 seed-robustness leaf is invalid")
    identities = tuple(leaf.identity for leaf in resolved_leaves)
'''
robust_leaf_new = '''    if any(
        not isinstance(leaf, UniversalTradeRLU2SelectionLeafMetrics)
        for leaf in resolved_leaves
    ):
        raise TypeError("U2 seed-robustness leaf is invalid")
    for leaf in resolved_leaves:
        _require_current_artifact_digest(
            leaf,
            field="U2 seed-robustness Selection leaf",
        )
    identities = tuple(leaf.identity for leaf in resolved_leaves)
'''
replace_once(robust_leaf_anchor, robust_leaf_new, label="robustness leaf integrity")

robust_pair_anchor = '''    selected_segments = tuple(by_window[window] for window in expected_windows)
    if scope == "D1+D2" and set(observed_windows) != set(expected_windows):
        raise ValueError("U2 seed-robustness aggregate bootstrap scope is invalid")

    thresholds = _u2_cross_seed_robustness_thresholds()
'''
robust_pair_new = '''    selected_segments = tuple(by_window[window] for window in expected_windows)
    if scope == "D1+D2" and set(observed_windows) != set(expected_windows):
        raise ValueError("U2 seed-robustness aggregate bootstrap scope is invalid")

    paired_scope_evidence = tuple(
        pair
        for segment in selected_segments
        for pair in segment.paired_scope_evidence
    )
    if paired_scope_evidence:
        pair_by_leaf_identity = {
            (
                pair.training_seed,
                pair.cell,
                pair.concrete_symbol,
                pair.scope_digest,
            ): pair
            for pair in paired_scope_evidence
        }
        if len(pair_by_leaf_identity) != len(paired_scope_evidence):
            raise ValueError("U2 seed-robustness paired replay identity is duplicated")
        for pair in paired_scope_evidence:
            _require_current_artifact_digest(
                pair,
                field="U2 seed-robustness paired replay evidence",
            )
        for leaf in resolved_leaves:
            pair = pair_by_leaf_identity.get(
                (
                    leaf.training_seed,
                    leaf.cell,
                    leaf.concrete_symbol,
                    leaf.tile_identity,
                )
            )
            if (
                pair is None
                or leaf.replay_evidence_digest
                != pair.candidate_replay_evidence_digest
            ):
                raise ValueError(
                    "U2 seed-robustness leaf candidate replay provenance mismatch"
                )

    thresholds = _u2_cross_seed_robustness_thresholds()
'''
replace_once(robust_pair_anchor, robust_pair_new, label="robustness pair binding")

robust_return_anchor = '''        passed=not reasons,
        paired_scope_evidence=tuple(
            pair
            for segment in selected_segments
            for pair in segment.paired_scope_evidence
        ),
    )
'''
robust_return_new = '''        passed=not reasons,
        paired_scope_evidence=paired_scope_evidence,
    )
'''
replace_once(robust_return_anchor, robust_return_new, label="robustness pair propagation")

final_top_anchor = '''        if not isinstance(
            self.checkpoint_closure,
            UniversalTradeRLU2FinalCheckpointClosure,
        ):
            raise TypeError("U2 final Selection requires final checkpoint closure")

        if self.base_lock.u2_contract_digest != self.u2_contract.digest:
'''
final_top_new = '''        if not isinstance(
            self.checkpoint_closure,
            UniversalTradeRLU2FinalCheckpointClosure,
        ):
            raise TypeError("U2 final Selection requires final checkpoint closure")

        for artifact, field_name in (
            (self.u2_contract, "U2 final Selection U2 contract"),
            (self.base_lock, "U2 final Selection base lock"),
            (self.development_lock, "U2 final Selection Development lock"),
            (self.checkpoint_closure, "U2 final Selection checkpoint closure"),
        ):
            _require_current_artifact_digest(artifact, field=field_name)

        if self.base_lock.u2_contract_digest != self.u2_contract.digest:
'''
replace_once(final_top_anchor, final_top_new, label="final top artifact integrity")

primary_anchor = '''        if any(
            not isinstance(
                gate,
                UniversalTradeRLU2PrimarySelectionCellGateEvidence,
            )
            for gate in primary
        ):
            raise TypeError("U2 final Selection primary gate is invalid")
        if tuple(gate.cell for gate in primary) != _U2_FINAL_PRIMARY_CELLS:
'''
primary_new = '''        if any(
            not isinstance(
                gate,
                UniversalTradeRLU2PrimarySelectionCellGateEvidence,
            )
            for gate in primary
        ):
            raise TypeError("U2 final Selection primary gate is invalid")
        for gate in primary:
            _require_current_artifact_digest(
                gate,
                field="U2 final Selection primary gate",
            )
            _require_current_artifact_digest(
                gate.summary,
                field="U2 final Selection primary summary",
            )
            for symbol_metrics in gate.summary.symbol_metrics:
                _require_current_artifact_digest(
                    symbol_metrics,
                    field="U2 final Selection primary symbol metrics",
                )
        if tuple(gate.cell for gate in primary) != _U2_FINAL_PRIMARY_CELLS:
'''
replace_once(primary_anchor, primary_new, label="final primary artifact integrity")

robust_final_anchor = '''        if any(
            not isinstance(gate, UniversalTradeRLU2SeedRobustnessEvidence)
            for gate in robustness
        ):
            raise TypeError("U2 final Selection robustness gate is invalid")
        if tuple(gate.scope for gate in robustness) != _U2_FINAL_ROBUSTNESS_SCOPES:
'''
robust_final_new = '''        if any(
            not isinstance(gate, UniversalTradeRLU2SeedRobustnessEvidence)
            for gate in robustness
        ):
            raise TypeError("U2 final Selection robustness gate is invalid")
        for gate in robustness:
            _require_current_artifact_digest(
                gate,
                field="U2 final Selection robustness gate",
            )
            _require_current_artifact_digest(
                gate.bootstrap_result,
                field="U2 final Selection robustness bootstrap",
            )
            for summary in gate.summaries:
                _require_current_artifact_digest(
                    summary,
                    field="U2 final Selection robustness summary",
                )
                for symbol_metrics in summary.symbol_metrics:
                    _require_current_artifact_digest(
                        symbol_metrics,
                        field="U2 final Selection robustness symbol metrics",
                    )
            for pair in gate.paired_scope_evidence:
                _require_current_artifact_digest(
                    pair,
                    field="U2 final Selection paired replay evidence",
                )
        if tuple(gate.scope for gate in robustness) != _U2_FINAL_ROBUSTNESS_SCOPES:
'''
replace_once(robust_final_anchor, robust_final_new, label="final robustness artifact integrity")

path.write_text(text, encoding="utf-8")
