from __future__ import annotations

from copy import deepcopy

from tools.agent_repo.research_assurance import assurance_digest, evaluate_assurance


def _record() -> dict[str, object]:
    return {
        "schema_version": "research_assurance_v1",
        "identity": {
            "protocol_head": "a" * 40,
            "implementation_head": "b" * 40,
        },
        "thesis": {
            "final_decision": "whether to authorize one development-only factor test",
            "hypothesis": "the factor improves net performance through the stated mechanism",
            "economic_mechanism": "causal state changes position choice before execution",
            "counter_hypothesis": "the apparent effect is only exposure beta or accounting drift",
            "information_gain": "the experiment separates the factor from the frozen control",
            "cheapest_falsifier": "synthetic invariance and control-path equivalence",
            "stop_rule": "stop the line if the factor cannot beat the control under the frozen gate",
            "stage": "development",
            "metric_proxy_rationale": "net return plus risk and cost diagnostics approximates the development objective",
            "limitations": "development evidence cannot establish unused-data or live profitability",
        },
        "mechanism": {
            "chain": [
                "source",
                "availability",
                "feature_state",
                "model_strategy",
                "intent",
                "order",
                "fill",
                "accounting",
                "evidence",
                "decision",
            ],
            "authorities": {
                "unit": "canonical quantity and lot contract",
                "time": "point-in-time availability contract",
                "state": "realized fill and BookState",
                "sign": "canonical cash-flow convention",
                "risk": "hard-risk owner",
                "execution": "MarketExecutor",
                "accounting": "BookState",
                "evidence": "content-addressed result artifact",
            },
            "independent_oracles": [
                "metamorphic future-row mutation invariance",
                "independent account arithmetic",
            ],
        },
        "evidence": {
            "point_in_time": True,
            "common_accounting": True,
            "realistic_costs": True,
            "hard_risk": True,
            "terminal_state": True,
            "fit_development_unused_separated": True,
            "multi_symbol": True,
            "multi_period": True,
            "controls": ["cash", "constant_long", "constant_short"],
            "robustness": ["double_cost", "latency"],
            "independent_reconstruction": "recompute account arithmetic without trusting decision booleans",
            "no_development_rescue": True,
            "evidence_level": "development_profitability",
        },
        "claims": {
            "claim_level": "development_profitability",
            "permitted": ["development-only profitability under the frozen assumptions"],
            "forbidden": ["unused-data validity", "production eligibility", "live trading"],
            "next_authorized_action": "separate unused-data protocol only if the frozen gate passes",
            "production_eligible": False,
            "live_trading_authorized": False,
        },
        "review": {
            "status": "unreviewed",
            "reviewed_record_digest": None,
            "protocol_head": None,
            "implementation_head": None,
            "reviewer": None,
            "rationale": "awaiting adversarial review",
        },
    }


def _reviewed(record: dict[str, object]) -> dict[str, object]:
    reviewed = deepcopy(record)
    digest = assurance_digest(reviewed)
    identity = reviewed["identity"]
    assert isinstance(identity, dict)
    reviewed["review"] = {
        "status": "pass",
        "reviewed_record_digest": digest,
        "protocol_head": identity["protocol_head"],
        "implementation_head": identity["implementation_head"],
        "reviewer": "independent-adversarial-review",
        "rationale": "thesis, mechanism, evidence and bounded claims were challenged",
    }
    return reviewed


def test_complete_but_unreviewed_record_cannot_authorize_execution() -> None:
    result = evaluate_assurance(_record())
    assert result.status == "UNREVIEWED"
    assert result.economic_execution_authorized is False
    assert result.errors == ()


def test_exact_reviewed_record_with_bounded_claim_passes() -> None:
    result = evaluate_assurance(_reviewed(_record()))
    assert result.status == "PASS"
    assert result.economic_execution_authorized is True
    assert result.errors == ()


def test_review_is_stale_after_record_or_head_changes() -> None:
    record = _reviewed(_record())
    thesis = record["thesis"]
    assert isinstance(thesis, dict)
    thesis["hypothesis"] = "changed after review"
    result = evaluate_assurance(record)
    assert result.status == "BLOCKED"
    assert result.economic_execution_authorized is False
    assert any("reviewed_record_digest" in error for error in result.errors)


def test_missing_counter_hypothesis_or_falsifier_fails_closed() -> None:
    record = _record()
    thesis = record["thesis"]
    assert isinstance(thesis, dict)
    del thesis["counter_hypothesis"]
    result = evaluate_assurance(record)
    assert result.status == "BLOCKED"
    assert any("counter_hypothesis" in error for error in result.errors)

    record = _record()
    thesis = record["thesis"]
    assert isinstance(thesis, dict)
    thesis["cheapest_falsifier"] = ""
    result = evaluate_assurance(record)
    assert result.status == "BLOCKED"
    assert any("cheapest_falsifier" in error for error in result.errors)


def test_evidence_level_cannot_be_promoted_into_stronger_claim() -> None:
    record = _record()
    evidence = record["evidence"]
    claims = record["claims"]
    assert isinstance(evidence, dict)
    assert isinstance(claims, dict)
    evidence["evidence_level"] = "relative_improvement"
    claims["claim_level"] = "development_profitability"
    result = evaluate_assurance(record)
    assert result.status == "BLOCKED"
    assert any("claim_level exceeds evidence_level" in error for error in result.errors)


def test_software_evidence_cannot_be_represented_as_economic_evidence() -> None:
    record = _record()
    evidence = record["evidence"]
    claims = record["claims"]
    assert isinstance(evidence, dict)
    assert isinstance(claims, dict)
    evidence["evidence_level"] = "software_validity"
    claims["claim_level"] = "relative_improvement"
    result = evaluate_assurance(record)
    assert result.status == "BLOCKED"
    assert any("claim_level exceeds evidence_level" in error for error in result.errors)


def test_development_stage_cannot_authorize_production_or_live() -> None:
    record = _record()
    claims = record["claims"]
    assert isinstance(claims, dict)
    claims["claim_level"] = "production"
    claims["production_eligible"] = True
    claims["live_trading_authorized"] = True
    result = evaluate_assurance(record)
    assert result.status == "BLOCKED"
    assert any("development stage" in error for error in result.errors)


def test_structurally_complete_wrong_question_stays_blocked_by_adversarial_review() -> None:
    record = _record()
    digest = assurance_digest(record)
    identity = record["identity"]
    assert isinstance(identity, dict)
    record["review"] = {
        "status": "blocked",
        "reviewed_record_digest": digest,
        "protocol_head": identity["protocol_head"],
        "implementation_head": identity["implementation_head"],
        "reviewer": "independent-adversarial-review",
        "rationale": "the metric can pass under trivial exposure beta and does not identify the claimed mechanism",
    }
    result = evaluate_assurance(record)
    assert result.status == "BLOCKED"
    assert result.economic_execution_authorized is False
    assert result.errors == ()


def test_changed_mechanism_head_invalidates_prior_review() -> None:
    record = _reviewed(_record())
    identity = record["identity"]
    assert isinstance(identity, dict)
    identity["implementation_head"] = "c" * 40
    result = evaluate_assurance(record)
    assert result.status == "BLOCKED"
    assert any("implementation_head" in error for error in result.errors)
