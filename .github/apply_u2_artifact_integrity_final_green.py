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

pair_anchor = '''    if not isinstance(candidate_replay, UniversalTradeRLU2ReplayEvidence):
        raise TypeError("U2 paired replay candidate evidence is invalid")
    if not isinstance(cash_replay, UniversalTradeRLU2ReplayEvidence):
        raise TypeError("U2 paired replay cash evidence is invalid")
    if (
        candidate_replay.policy_variant
'''
pair_new = '''    if not isinstance(candidate_replay, UniversalTradeRLU2ReplayEvidence):
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
replace_once(pair_anchor, pair_new, label="pairing replay integrity")

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
replace_once(final_top_anchor, final_top_new, label="final top-level integrity")

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
        for primary_gate in primary:
            _require_current_artifact_digest(
                primary_gate,
                field="U2 final Selection primary gate",
            )
            _require_current_artifact_digest(
                primary_gate.summary,
                field="U2 final Selection primary summary",
            )
            for symbol_metrics in primary_gate.summary.symbol_metrics:
                _require_current_artifact_digest(
                    symbol_metrics,
                    field="U2 final Selection primary symbol metrics",
                )
        if tuple(gate.cell for gate in primary) != _U2_FINAL_PRIMARY_CELLS:
'''
replace_once(primary_anchor, primary_new, label="final primary integrity")

robustness_anchor = '''        if any(
            not isinstance(gate, UniversalTradeRLU2SeedRobustnessEvidence)
            for gate in robustness
        ):
            raise TypeError("U2 final Selection robustness gate is invalid")
        if tuple(gate.scope for gate in robustness) != _U2_FINAL_ROBUSTNESS_SCOPES:
'''
robustness_new = '''        if any(
            not isinstance(gate, UniversalTradeRLU2SeedRobustnessEvidence)
            for gate in robustness
        ):
            raise TypeError("U2 final Selection robustness gate is invalid")
        for robustness_gate in robustness:
            _require_current_artifact_digest(
                robustness_gate,
                field="U2 final Selection robustness gate",
            )
            _require_current_artifact_digest(
                robustness_gate.bootstrap_result,
                field="U2 final Selection robustness bootstrap",
            )
            for summary in robustness_gate.summaries:
                _require_current_artifact_digest(
                    summary,
                    field="U2 final Selection robustness summary",
                )
                for symbol_metrics in summary.symbol_metrics:
                    _require_current_artifact_digest(
                        symbol_metrics,
                        field="U2 final Selection robustness symbol metrics",
                    )
            for pair in robustness_gate.paired_scope_evidence:
                _require_current_artifact_digest(
                    pair,
                    field="U2 final Selection paired replay evidence",
                )
        if tuple(gate.scope for gate in robustness) != _U2_FINAL_ROBUSTNESS_SCOPES:
'''
replace_once(robustness_anchor, robustness_new, label="final robustness integrity")

path.write_text(text, encoding="utf-8")
