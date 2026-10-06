from dataclasses import FrozenInstanceError, replace

import pytest

from tests.evaluation.test_allocation_comparison import (
    complete_matrix,
    contract,
    sha,
)
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.allocation_comparison import (
    AllocationCandidateKind,
    AllocationValidity,
)


def capability():
    from trade_rl.evaluation.allocation_selection import (
        AllocationCashReference,
        AllocationSelectionOutcome,
        AllocationSelectionRule,
        select_allocation_candidate,
    )

    return (
        AllocationCashReference,
        AllocationSelectionOutcome,
        AllocationSelectionRule,
        select_allocation_candidate,
    )


def rule(declaration, *, nonrl=0.0, cash=0.0):
    _, _, Rule, _ = capability()
    return Rule(
        comparison_contract_digest=declaration.digest,
        minimum_incremental_vs_nonrl_base=nonrl,
        minimum_incremental_vs_cash_base=cash,
    )


def cash_references(declaration, *, base_profit=0.005):
    Cash, *_ = capability()
    values = []
    for scenario in declaration.scenarios:
        values.append(
            Cash(
                contract_digest=declaration.digest,
                scenario=scenario.name,
                control_evidence_digest=sha(f"cash-control-{scenario.name}"),
                oos_source_digest=sha(f"oos-{scenario.name}"),
                opening_state_digest=sha(f"opening-{scenario.name}"),
                terminal_profit_rate=(
                    base_profit if scenario.name == "base" else base_profit / 2.0
                ),
                max_drawdown=0.0,
                validity=AllocationValidity.VALID,
                coverage_complete=True,
                termination_reason=None,
            )
        )
    return tuple(values)


def select(
    declaration,
    rows=None,
    cash=None,
    selection_rule=None,
):
    *_, selector = capability()
    return selector(
        declaration,
        tuple(complete_matrix(declaration) if rows is None else rows),
        cash_references(declaration) if cash is None else cash,
        rule(declaration) if selection_rule is None else selection_rule,
    )


def test_rule_is_closed_immutable_and_roundtrips_canonical_bytes():
    _, _, Rule, _ = capability()
    declaration = contract()
    value = rule(declaration, nonrl=0.001, cash=0.002)
    payload = value.payload()
    assert payload["schema"] == "allocation_selection_rule_v1"
    assert payload["selection_id"] == "after_cost_terminal_profit_v1"
    assert payload["tie_break_order"] == [
        "nonrl",
        "residual_ppo",
        "direct_ppo",
    ]
    assert payload["strict_thresholds"] is True
    assert payload["threshold_comparison"] == "strict_gt_isclose_1e-12_v1"
    restored = Rule.from_payload(payload)
    assert canonical_json_bytes(restored.payload()) == canonical_json_bytes(payload)
    assert restored.digest == value.digest
    with pytest.raises(FrozenInstanceError):
        value.minimum_incremental_vs_cash_base = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"comparison_contract_digest": "bad"},
        {"minimum_incremental_vs_nonrl_base": -0.1},
        {"minimum_incremental_vs_cash_base": -0.1},
    ],
)
def test_rule_rejects_malformed_digest_or_negative_thresholds(changes):
    declaration = contract()
    values = {
        "comparison_contract_digest": declaration.digest,
        "minimum_incremental_vs_nonrl_base": 0.0,
        "minimum_incremental_vs_cash_base": 0.0,
    }
    values.update(changes)
    _, _, Rule, _ = capability()
    with pytest.raises(ValueError):
        Rule(**values)


def test_default_rule_selects_residual_ppo_as_highest_eligible_profit():
    _, Outcome, _, _ = capability()
    declaration = contract()
    decision = select(declaration)
    assert decision.outcome is Outcome.WINNER
    assert decision.selected_candidate is AllocationCandidateKind.RESIDUAL_PPO
    assert decision.eligible_candidates == (
        AllocationCandidateKind.NONRL,
        AllocationCandidateKind.RESIDUAL_PPO,
        AllocationCandidateKind.DIRECT_PPO,
    )
    assert decision.payload()["winner_score"] == pytest.approx(0.035)
    assert "p_value" not in decision.payload()
    assert "final_test" not in decision.payload()


def test_cash_floor_can_produce_no_winner_even_when_candidates_have_positive_profit():
    _, Outcome, _, _ = capability()
    declaration = contract()
    decision = select(
        declaration,
        cash=cash_references(declaration, base_profit=0.06),
    )
    assert decision.outcome is Outcome.NO_WINNER
    assert decision.selected_candidate is None
    assert decision.eligible_candidates == ()
    assert all("cash" in row.reasons for row in decision.candidates)


def test_rl_must_beat_nonrl_not_only_cash():
    _, Outcome, _, _ = capability()
    declaration = contract()
    rows = list(complete_matrix(declaration))
    for index, row in enumerate(rows):
        if (
            row.candidate is AllocationCandidateKind.RESIDUAL_PPO
            and row.scenario == "base"
        ):
            rows[index] = replace(row, terminal_profit_rate=0.019)
        if (
            row.candidate is AllocationCandidateKind.DIRECT_PPO
            and row.scenario == "base"
        ):
            rows[index] = replace(row, terminal_profit_rate=0.018)
    decision = select(declaration, rows=rows)
    assert decision.outcome is Outcome.WINNER
    assert decision.selected_candidate is AllocationCandidateKind.NONRL
    residual = next(
        row
        for row in decision.candidates
        if row.candidate is AllocationCandidateKind.RESIDUAL_PPO
    )
    direct = next(
        row
        for row in decision.candidates
        if row.candidate is AllocationCandidateKind.DIRECT_PPO
    )
    assert "nonrl" in residual.reasons
    assert "nonrl" in direct.reasons


def test_required_invalid_candidate_evidence_makes_whole_comparison_invalid():
    _, Outcome, _, _ = capability()
    declaration = contract()
    rows = list(complete_matrix(declaration))
    index = next(
        i
        for i, row in enumerate(rows)
        if row.candidate is AllocationCandidateKind.DIRECT_PPO
        and row.seed == 0
        and row.scenario == "base"
    )
    rows[index] = replace(rows[index], validity=AllocationValidity.INVALID)
    decision = select(declaration, rows=rows)
    assert decision.outcome is Outcome.INVALID
    assert decision.selected_candidate is None
    assert "required_candidate_evidence_invalid" in decision.invalid_reasons


def test_required_invalid_cash_reference_makes_comparison_invalid():
    _, Outcome, _, _ = capability()
    declaration = contract()
    refs = list(cash_references(declaration))
    base = next(i for i, row in enumerate(refs) if row.scenario == "base")
    refs[base] = replace(refs[base], coverage_complete=False)
    decision = select(declaration, cash=tuple(refs))
    assert decision.outcome is Outcome.INVALID
    assert "required_cash_reference_invalid" in decision.invalid_reasons


def test_required_risk_failure_disqualifies_candidate_but_does_not_invalidate_study():
    _, Outcome, _, _ = capability()
    declaration = contract()
    rows = list(complete_matrix(declaration))
    residual_cost = next(
        i
        for i, row in enumerate(rows)
        if row.candidate is AllocationCandidateKind.RESIDUAL_PPO
        and row.seed == 0
        and row.scenario == "cost_2x"
    )
    rows[residual_cost] = replace(rows[residual_cost], max_drawdown=0.50)
    decision = select(declaration, rows=rows)
    assert decision.outcome is Outcome.WINNER
    assert decision.selected_candidate is AllocationCandidateKind.DIRECT_PPO
    residual = next(
        row
        for row in decision.candidates
        if row.candidate is AllocationCandidateKind.RESIDUAL_PPO
    )
    assert "risk_execution" in residual.reasons


def test_optional_diagnostic_failure_does_not_invalidate_or_disqualify_candidate():
    _, Outcome, _, _ = capability()
    declaration = contract()
    rows = list(complete_matrix(declaration))
    diagnostic = next(
        i
        for i, row in enumerate(rows)
        if row.candidate is AllocationCandidateKind.RESIDUAL_PPO
        and row.seed == 0
        and row.scenario == "diagnostic_gap"
    )
    rows[diagnostic] = replace(
        rows[diagnostic],
        validity=AllocationValidity.INVALID,
        coverage_complete=False,
        max_drawdown=0.9,
    )
    decision = select(declaration, rows=rows)
    assert decision.outcome is Outcome.WINNER
    assert decision.selected_candidate is AllocationCandidateKind.RESIDUAL_PPO
    assert (
        "diagnostic_gap"
        in next(
            row
            for row in decision.candidates
            if row.candidate is AllocationCandidateKind.RESIDUAL_PPO
        ).diagnostic_failures
    )


def test_exact_score_tie_between_rl_candidates_prefers_residual_ppo():
    _, Outcome, _, _ = capability()
    declaration = contract()
    rows = list(complete_matrix(declaration))
    for index, row in enumerate(rows):
        if (
            row.candidate is AllocationCandidateKind.DIRECT_PPO
            and row.scenario == "base"
        ):
            rows[index] = replace(row, terminal_profit_rate=0.035)
    decision = select(declaration, rows=rows)
    assert decision.outcome is Outcome.WINNER
    assert decision.selected_candidate is AllocationCandidateKind.RESIDUAL_PPO


def test_threshold_is_strict_not_greater_or_equal():
    _, Outcome, _, _ = capability()
    declaration = contract()
    selection_rule = rule(declaration, nonrl=0.015, cash=0.0)
    decision = select(declaration, selection_rule=selection_rule)
    # Residual median increment is mathematically 0.015 and must not pass
    # merely because binary floating arithmetic produces 0.015000000000000003.
    # Direct PPO improves only 0.01, so nonRL remains the only eligible candidate.
    assert decision.outcome is Outcome.WINNER
    assert decision.selected_candidate is AllocationCandidateKind.NONRL
    residual = next(
        row
        for row in decision.candidates
        if row.candidate is AllocationCandidateKind.RESIDUAL_PPO
    )
    assert "nonrl" in residual.reasons


@pytest.mark.parametrize(
    "mode", ["missing", "duplicate", "wrong_contract", "wrong_source"]
)
def test_cash_reference_matrix_fails_closed_on_structural_mismatch(mode):
    declaration = contract()
    refs = list(cash_references(declaration))
    if mode == "missing":
        refs.pop()
    elif mode == "duplicate":
        refs.append(refs[-1])
    elif mode == "wrong_contract":
        refs[0] = replace(refs[0], contract_digest=sha("wrong-contract"))
    else:
        refs[0] = replace(refs[0], oos_source_digest=sha("wrong-source"))
    with pytest.raises(ValueError):
        select(declaration, cash=tuple(refs))


def test_decision_digest_binds_rule_summary_cash_and_outcome():
    declaration = contract()
    decision = select(declaration)
    payload = decision.payload()
    assert decision.digest == content_digest(payload)
    changed = select(
        declaration,
        selection_rule=rule(declaration, cash=0.001),
    )
    assert changed.digest != decision.digest


def test_actual_cash_control_evidence_projects_to_selector_reference():
    from tests.evaluation.test_allocation_cash_control import cash_fixture
    from tests.evaluation.test_allocation_comparison_evidence import contract_for
    from trade_rl.evaluation.allocation_cash_control import (
        allocation_cash_reference,
        build_allocation_cash_control_evidence,
        run_continuous_allocation_cash_control,
    )

    first, second, folds = cash_fixture()
    declaration = contract_for(first, folds)
    result = run_continuous_allocation_cash_control(folds, (first, second))
    evidence = build_allocation_cash_control_evidence(
        declaration,
        scenario="base",
        folds=folds,
        environments=(first, second),
        result=result,
        validity_evidence_digest=sha("cash-validity"),
    )
    reference = allocation_cash_reference(evidence)
    assert reference.contract_digest == declaration.digest
    assert reference.control_evidence_digest == evidence.digest
    assert reference.terminal_profit_rate == evidence.terminal_profit_rate
    assert reference.oos_source_digest == evidence.oos_source_digest
    assert reference.opening_state_digest == evidence.opening_state_digest
