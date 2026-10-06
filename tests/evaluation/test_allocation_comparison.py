from dataclasses import FrozenInstanceError, replace

import pytest

from trade_rl.artifacts import canonical_json_bytes, content_digest


def capability():
    from trade_rl.evaluation.allocation_comparison import (
        AllocationCandidateKind,
        AllocationComparisonContract,
        AllocationComparisonEvidence,
        AllocationComparisonScenario,
        AllocationValidity,
        summarize_allocation_comparison,
        validate_allocation_comparison_evidence,
    )

    return (
        AllocationCandidateKind,
        AllocationComparisonContract,
        AllocationComparisonEvidence,
        AllocationComparisonScenario,
        AllocationValidity,
        summarize_allocation_comparison,
        validate_allocation_comparison_evidence,
    )


def sha(label):
    return content_digest({"label": label})


def contract():
    _, Contract, _, Scenario, _, _, _ = capability()
    return Contract(
        dataset_id=sha("dataset"),
        objective_digest=sha("objective"),
        clock_digest=sha("clock"),
        forecast_context_digest=sha("forecast-context"),
        economics_digest=sha("economics"),
        risk_digest=sha("risk"),
        fold_plan_digest=sha("fold-plan"),
        account_mode="independent_symbol",
        initial_capital=1_000.0,
        scenarios=(
            Scenario("base", sha("scenario-base"), required=True),
            Scenario("cost_2x", sha("scenario-cost"), required=True),
            Scenario("diagnostic_gap", sha("scenario-gap"), required=False),
        ),
        rl_seeds=(0, 7),
        maximum_drawdown=0.20,
    )


def evidence(
    declaration,
    candidate,
    scenario,
    *,
    seed=None,
    profit=0.02,
    drawdown=0.08,
    validity="valid",
    coverage=True,
    termination=None,
    opening=None,
    source=None,
):
    Candidate, _, Evidence, _, Validity, _, _ = capability()
    kind = candidate if isinstance(candidate, Candidate) else Candidate(candidate)
    status = validity if isinstance(validity, Validity) else Validity(validity)
    if kind is Candidate.NONRL:
        policy = None
        seed = None
    else:
        policy = sha(f"policy-{kind.value}-{seed}")
    return Evidence(
        contract_digest=declaration.digest,
        candidate=kind,
        scenario=scenario,
        seed=seed,
        policy_digest=policy,
        oos_source_digest=source or sha(f"oos-{scenario}"),
        ledger_digest=sha(f"ledger-{kind.value}-{scenario}-{seed}"),
        execution_digest=sha(f"execution-{kind.value}-{scenario}-{seed}"),
        validity_evidence_digest=sha(f"validity-{kind.value}-{scenario}-{seed}"),
        opening_state_digest=opening or sha(f"opening-{scenario}"),
        closing_state_digest=sha(f"closing-{kind.value}-{scenario}-{seed}"),
        terminal_profit_rate=profit,
        max_drawdown=drawdown,
        validity=status,
        coverage_complete=coverage,
        termination_reason=termination,
    )


def complete_matrix(declaration):
    Candidate, *_ = capability()
    rows = []
    nonrl = {"base": 0.02, "cost_2x": 0.01, "diagnostic_gap": -0.01}
    residual = {
        0: {"base": 0.03, "cost_2x": 0.015, "diagnostic_gap": -0.02},
        7: {"base": 0.04, "cost_2x": 0.005, "diagnostic_gap": -0.03},
    }
    direct = {
        0: {"base": 0.01, "cost_2x": 0.02, "diagnostic_gap": -0.04},
        7: {"base": 0.05, "cost_2x": -0.01, "diagnostic_gap": -0.05},
    }
    for scenario, profit in nonrl.items():
        rows.append(evidence(declaration, Candidate.NONRL, scenario, profit=profit))
    for seed, values in residual.items():
        for scenario, profit in values.items():
            rows.append(
                evidence(
                    declaration,
                    Candidate.RESIDUAL_PPO,
                    scenario,
                    seed=seed,
                    profit=profit,
                )
            )
    for seed, values in direct.items():
        for scenario, profit in values.items():
            rows.append(
                evidence(
                    declaration,
                    Candidate.DIRECT_PPO,
                    scenario,
                    seed=seed,
                    profit=profit,
                )
            )
    return tuple(rows)


def test_contract_is_closed_immutable_and_roundtrips_canonical_bytes():
    _, Contract, _, _, _, _, _ = capability()
    declaration = contract()
    payload = declaration.payload()
    assert payload["schema"] == "allocation_comparison_contract_v1"
    assert payload["candidate_roster"] == [
        "nonrl",
        "residual_ppo",
        "direct_ppo",
    ]
    assert payload["scenarios"][0]["name"] == "base"
    assert payload["scenarios"][0]["required"] is True
    assert declaration.digest == content_digest(payload)
    restored = Contract.from_payload(payload)
    assert canonical_json_bytes(restored.payload()) == canonical_json_bytes(payload)
    assert restored.digest == declaration.digest
    with pytest.raises(FrozenInstanceError):
        declaration.initial_capital = 2_000.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset_id": "bad"},
        {"account_mode": "unknown"},
        {"initial_capital": 0.0},
        {"rl_seeds": ()},
        {"rl_seeds": (0, 0)},
        {"rl_seeds": (7, 0)},
        {"maximum_drawdown": 0.0},
    ],
)
def test_contract_rejects_malformed_identity_capital_seed_and_risk(changes):
    declaration = contract()
    values = {
        field: getattr(declaration, field)
        for field in (
            "dataset_id",
            "objective_digest",
            "clock_digest",
            "forecast_context_digest",
            "economics_digest",
            "risk_digest",
            "fold_plan_digest",
            "account_mode",
            "initial_capital",
            "scenarios",
            "rl_seeds",
            "maximum_drawdown",
        )
    }
    values.update(changes)
    _, Contract, *_ = capability()
    with pytest.raises(ValueError):
        Contract(**values)


def test_evidence_enforces_nonrl_and_rl_identity_semantics():
    Candidate, _, Evidence, _, Validity, _, _ = capability()
    declaration = contract()
    base = evidence(declaration, Candidate.NONRL, "base")
    assert base.seed is None and base.policy_digest is None

    values = base.payload() | {
        "candidate": "residual_ppo",
        "seed": None,
        "policy_digest": None,
    }
    with pytest.raises(ValueError):
        Evidence.from_payload(values)

    rl = evidence(declaration, Candidate.RESIDUAL_PPO, "base", seed=0)
    assert rl.seed == 0 and rl.policy_digest is not None
    assert rl.validity is Validity.VALID


def test_complete_matrix_validates_and_summary_reports_incremental_profit_without_winner():
    Candidate, _, _, _, _, summarize, validate = capability()
    declaration = contract()
    rows = complete_matrix(declaration)
    normalized = validate(declaration, rows)
    assert len(normalized) == 15

    summary = summarize(declaration, rows)
    payload = summary.payload()
    assert payload["schema"] == "allocation_comparison_summary_v1"
    assert "winner" not in payload
    assert "selected_candidate" not in payload

    by_candidate = {row["candidate"]: row for row in payload["candidates"]}
    assert by_candidate["nonrl"]["median_base_profit_rate"] == pytest.approx(0.02)
    assert by_candidate["nonrl"]["median_incremental_vs_nonrl_base"] == 0.0
    assert by_candidate["residual_ppo"]["median_base_profit_rate"] == pytest.approx(
        0.035
    )
    assert by_candidate["residual_ppo"][
        "median_incremental_vs_nonrl_base"
    ] == pytest.approx(0.015)
    assert by_candidate["direct_ppo"]["median_base_profit_rate"] == pytest.approx(0.03)
    assert by_candidate["direct_ppo"][
        "median_incremental_vs_nonrl_base"
    ] == pytest.approx(0.01)
    assert all(row["validity_passed"] for row in by_candidate.values())
    assert all(row["risk_execution_passed"] for row in by_candidate.values())


def test_high_profit_invalid_evidence_never_becomes_valid_in_summary():
    Candidate, _, _, _, Validity, summarize, _ = capability()
    declaration = contract()
    rows = list(complete_matrix(declaration))
    index = next(
        i
        for i, row in enumerate(rows)
        if row.candidate is Candidate.DIRECT_PPO
        and row.seed == 7
        and row.scenario == "base"
    )
    rows[index] = replace(
        rows[index],
        terminal_profit_rate=9.0,
        validity=Validity.INVALID,
    )
    summary = summarize(declaration, tuple(rows))
    direct = next(
        value
        for value in summary.payload()["candidates"]
        if value["candidate"] == "direct_ppo"
    )
    assert direct["median_base_profit_rate"] > 1.0
    assert direct["validity_passed"] is False


def test_required_risk_failure_is_distinct_from_diagnostic_scenario_failure():
    Candidate, _, _, _, _, summarize, _ = capability()
    declaration = contract()
    rows = list(complete_matrix(declaration))

    diagnostic = next(
        i
        for i, row in enumerate(rows)
        if row.candidate is Candidate.RESIDUAL_PPO
        and row.seed == 0
        and row.scenario == "diagnostic_gap"
    )
    rows[diagnostic] = replace(rows[diagnostic], max_drawdown=0.50)
    summary = summarize(declaration, tuple(rows))
    residual = next(
        value
        for value in summary.payload()["candidates"]
        if value["candidate"] == "residual_ppo"
    )
    assert residual["risk_execution_passed"] is True
    assert residual["diagnostic_failures"] == ["diagnostic_gap"]

    required = next(
        i
        for i, row in enumerate(rows)
        if row.candidate is Candidate.RESIDUAL_PPO
        and row.seed == 7
        and row.scenario == "cost_2x"
    )
    rows[required] = replace(rows[required], max_drawdown=0.50)
    summary = summarize(declaration, tuple(rows))
    residual = next(
        value
        for value in summary.payload()["candidates"]
        if value["candidate"] == "residual_ppo"
    )
    assert residual["risk_execution_passed"] is False


@pytest.mark.parametrize("mode", ["missing", "duplicate", "wrong_contract"])
def test_matrix_rejects_missing_duplicate_or_cross_contract_rows(mode):
    _, _, _, _, _, _, validate = capability()
    declaration = contract()
    rows = list(complete_matrix(declaration))
    if mode == "missing":
        rows.pop()
    elif mode == "duplicate":
        rows.append(rows[-1])
    else:
        rows[-1] = replace(rows[-1], contract_digest=sha("another-contract"))
    with pytest.raises(ValueError):
        validate(declaration, tuple(rows))


@pytest.mark.parametrize("field", ["oos_source_digest", "opening_state_digest"])
def test_matrix_rejects_candidate_context_mismatch_within_same_scenario(field):
    Candidate, _, _, _, _, _, validate = capability()
    declaration = contract()
    rows = list(complete_matrix(declaration))
    index = next(
        i
        for i, row in enumerate(rows)
        if row.candidate is Candidate.DIRECT_PPO
        and row.seed == 0
        and row.scenario == "base"
    )
    rows[index] = replace(rows[index], **{field: sha(f"wrong-{field}")})
    with pytest.raises(ValueError, match="common|opening|source"):
        validate(declaration, tuple(rows))


def test_coverage_and_termination_affect_separate_validity_and_risk_execution_flags():
    Candidate, _, _, _, _, summarize, _ = capability()
    declaration = contract()
    rows = list(complete_matrix(declaration))
    coverage = next(
        i
        for i, row in enumerate(rows)
        if row.candidate is Candidate.RESIDUAL_PPO
        and row.seed == 0
        and row.scenario == "base"
    )
    rows[coverage] = replace(rows[coverage], coverage_complete=False)
    termination = next(
        i
        for i, row in enumerate(rows)
        if row.candidate is Candidate.DIRECT_PPO
        and row.seed == 0
        and row.scenario == "cost_2x"
    )
    rows[termination] = replace(rows[termination], termination_reason="drawdown_stop")

    summary = summarize(declaration, tuple(rows)).payload()
    by_candidate = {row["candidate"]: row for row in summary["candidates"]}
    assert by_candidate["residual_ppo"]["validity_passed"] is False
    assert by_candidate["direct_ppo"]["risk_execution_passed"] is False


def test_rl_policy_identity_cannot_change_between_base_and_stress_scenarios():
    Candidate, _, _, _, _, _, validate = capability()
    declaration = contract()
    rows = list(complete_matrix(declaration))
    index = next(
        i
        for i, row in enumerate(rows)
        if row.candidate is Candidate.RESIDUAL_PPO
        and row.seed == 0
        and row.scenario == "cost_2x"
    )
    rows[index] = replace(rows[index], policy_digest=sha("stress-specific-policy"))
    with pytest.raises(ValueError, match="one policy|across scenarios"):
        validate(declaration, tuple(rows))
