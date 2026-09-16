from __future__ import annotations


def _protocol():
    module = __import__(
        "trade_rl.evaluation.experiments.ppo_global_btc_regime_evaluation_prereg",
        fromlist=["*"],
    )
    return module.canonical_ppo_global_btc_regime_evaluation_protocol()


def test_development_acceptance_is_not_an_operational_profitability_gate() -> None:
    protocol = _protocol()

    assert protocol.development_acceptance_scope == "robust_factor_improvement"
    assert protocol.development_acceptance_gate_metrics == (
        "positive_factor_effect_symbols",
        "cross_symbol_median_factor_effect_gt_zero",
        "positive_cross_symbol_median_seeds",
        "no_new_termination",
        "unaffected_raw_returns_equal",
    )
    assert protocol.candidate_profitability_statistics_are_diagnostic_only is True
    assert protocol.development_acceptance_establishes_profitability is False
    assert protocol.operational_eligibility_established is False

    assert not hasattr(protocol, "accept_requires_positive_candidate_return_symbols")
    assert not hasattr(
        protocol,
        "accept_requires_cross_symbol_median_candidate_return_gt_zero",
    )


def test_development_acceptance_keeps_robust_improvement_requirements() -> None:
    protocol = _protocol()

    assert protocol.accept_requires_positive_factor_effect_symbols == 5
    assert protocol.accept_requires_cross_symbol_median_factor_effect_gt_zero is True
    assert protocol.accept_requires_positive_cross_symbol_median_seeds == 5
    assert protocol.accept_requires_no_new_termination is True
    assert protocol.accept_requires_all_unaffected_raw_returns_equal is True
    assert protocol.valid_non_accept_decision == "KEEP_BASELINE"
    assert protocol.invalid_decision == "INVALID"


def test_profitability_diagnostics_do_not_open_final_or_live_boundaries() -> None:
    protocol = _protocol()

    assert protocol.candidate_return_statistic == (
        "per_symbol_median_candidate_total_return_across_matched_seeds"
    )
    assert protocol.economic_execution_authorized is False
    assert protocol.economic_result_inspected is False
    assert protocol.final_test_access_authorized is False
    assert protocol.production_eligible is False
    assert protocol.live_trading_authorized is False
    assert protocol.merge_authorized is False
