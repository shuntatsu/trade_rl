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
    '''from trade_rl.workflows.universal_trade_rl_u2_predevelopment import (
    UniversalTradeRLU2DevelopmentLock,
)
''',
    '''from trade_rl.workflows.universal_trade_rl_u2_predevelopment import (
    UniversalTradeRLU2DevelopmentLock,
    universal_trade_rl_u2_evaluation_seed,
)
''',
    label="evaluation seed import",
)

replace_once(
    '''def build_universal_trade_rl_u2_selection_leaf_metrics(
    *,
    training_seed: int,
    replay_evidence: UniversalTradeRLU2ReplayEvidence,
) -> UniversalTradeRLU2SelectionLeafMetrics:
''',
    '''def build_universal_trade_rl_u2_selection_leaf_metrics(
    *,
    training_seed: int,
    replay_evidence: UniversalTradeRLU2ReplayEvidence,
    checkpoint_closure: UniversalTradeRLU2FinalCheckpointClosure,
) -> UniversalTradeRLU2SelectionLeafMetrics:
''',
    label="leaf builder signature",
)

replace_once(
    '''    if not isinstance(replay_evidence, UniversalTradeRLU2ReplayEvidence):
        raise TypeError("U2 Selection leaf requires replay evidence")
    if (
        replay_evidence.policy_variant
''',
    '''    if not isinstance(replay_evidence, UniversalTradeRLU2ReplayEvidence):
        raise TypeError("U2 Selection leaf requires replay evidence")
    if not isinstance(checkpoint_closure, UniversalTradeRLU2FinalCheckpointClosure):
        raise TypeError("U2 Selection leaf requires final checkpoint closure")
    _require_current_artifact_digest(
        replay_evidence,
        field="U2 Selection leaf replay evidence",
    )
    _require_current_artifact_digest(
        checkpoint_closure,
        field="U2 Selection leaf checkpoint closure",
    )
    if checkpoint_closure.u2_contract_digest != replay_evidence.u2_contract_digest:
        raise ValueError("U2 Selection leaf checkpoint/U2 identity mismatch")
    if (
        checkpoint_closure.universe_manifest_digest
        != replay_evidence.universe_manifest_digest
    ):
        raise ValueError("U2 Selection leaf checkpoint/universe identity mismatch")
    if checkpoint_closure.u1_contract_digest != replay_evidence.u1_contract_digest:
        raise ValueError("U2 Selection leaf checkpoint/U1 identity mismatch")
    expected_evaluation_seed = universal_trade_rl_u2_evaluation_seed(
        u2_contract_digest=replay_evidence.u2_contract_digest,
        scope_digest=replay_evidence.scope_digest,
    )
    if replay_evidence.evaluation_seed != expected_evaluation_seed:
        raise ValueError(
            "U2 Selection leaf requires the scope-common evaluation CRN seed"
        )
    expected_checkpoint = dict(checkpoint_closure.checkpoint_digests)[training_seed]
    if replay_evidence.paired_candidate_checkpoint_digest != expected_checkpoint:
        raise ValueError(
            "U2 Selection leaf training-seed checkpoint identity mismatch"
        )
    if (
        replay_evidence.policy_variant
''',
    label="leaf identity validation",
)

replace_once(
    '''    if candidate_replay.evaluation_seed != cash_replay.evaluation_seed:
        raise ValueError("U2 paired replay seed identity mismatch")

    identity_fields = (
''',
    '''    if candidate_replay.evaluation_seed != cash_replay.evaluation_seed:
        raise ValueError("U2 paired replay seed identity mismatch")
    expected_evaluation_seed = universal_trade_rl_u2_evaluation_seed(
        u2_contract_digest=candidate_replay.u2_contract_digest,
        scope_digest=candidate_replay.scope_digest,
    )
    if candidate_replay.evaluation_seed != expected_evaluation_seed:
        raise ValueError(
            "U2 paired replay requires the scope-common evaluation CRN seed"
        )

    identity_fields = (
''',
    label="paired replay CRN validation",
)

path.write_text(text, encoding="utf-8")
