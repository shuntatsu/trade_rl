from pathlib import Path


path = Path("trade_rl/workflows/universal_trade_rl_u2_selection.py")
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str, *, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, got {count}")
    text = text.replace(old, new, 1)


replace_once(
    '''        for leaf in resolved_leaves:
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
''',
    '''        for leaf in resolved_leaves:
            matched_pair = pair_by_leaf_identity.get(
                (
                    leaf.training_seed,
                    leaf.cell,
                    leaf.concrete_symbol,
                    leaf.tile_identity,
                )
            )
            if (
                matched_pair is None
                or leaf.replay_evidence_digest
                != matched_pair.candidate_replay_evidence_digest
            ):
''',
    label="robustness matched pair variable",
)

replace_once(
    '''        for gate in primary:
            _require_current_artifact_digest(
                gate,
                field="U2 final Selection primary gate",
            )
            _require_current_artifact_digest(
                gate.summary,
                field="U2 final Selection primary summary",
            )
            for symbol_metrics in gate.summary.symbol_metrics:
''',
    '''        for primary_gate in primary:
            _require_current_artifact_digest(
                primary_gate,
                field="U2 final Selection primary gate",
            )
            _require_current_artifact_digest(
                primary_gate.summary,
                field="U2 final Selection primary summary",
            )
            for symbol_metrics in primary_gate.summary.symbol_metrics:
''',
    label="primary gate variable",
)

replace_once(
    '''        for gate in robustness:
            _require_current_artifact_digest(
                gate,
                field="U2 final Selection robustness gate",
            )
            _require_current_artifact_digest(
                gate.bootstrap_result,
                field="U2 final Selection robustness bootstrap",
            )
            for summary in gate.summaries:
''',
    '''        for robustness_gate in robustness:
            _require_current_artifact_digest(
                robustness_gate,
                field="U2 final Selection robustness gate",
            )
            _require_current_artifact_digest(
                robustness_gate.bootstrap_result,
                field="U2 final Selection robustness bootstrap",
            )
            for summary in robustness_gate.summaries:
''',
    label="robustness gate variable",
)

replace_once(
    '''            for pair in gate.paired_scope_evidence:
                _require_current_artifact_digest(
                    pair,
                    field="U2 final Selection paired replay evidence",
                )
        if tuple(gate.scope for gate in robustness) != _U2_FINAL_ROBUSTNESS_SCOPES:
''',
    '''            for pair in robustness_gate.paired_scope_evidence:
                _require_current_artifact_digest(
                    pair,
                    field="U2 final Selection paired replay evidence",
                )
        if tuple(gate.scope for gate in robustness) != _U2_FINAL_ROBUSTNESS_SCOPES:
''',
    label="robustness pair variable",
)

path.write_text(text, encoding="utf-8")
