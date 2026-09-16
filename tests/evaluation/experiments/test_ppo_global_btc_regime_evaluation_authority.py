from __future__ import annotations


def _module():
    return __import__(
        "trade_rl.evaluation.experiments.ppo_global_btc_regime_evaluation_prereg",
        fromlist=["*"],
    )


def test_evaluation_authority_binds_factor_and_implementation_seals() -> None:
    protocol = _module().canonical_ppo_global_btc_regime_evaluation_protocol()

    assert protocol.factor_prereg_seal_run_id == 35_078_049_585
    assert protocol.factor_prereg_seal_artifact_id == 10_439_281_380
    assert protocol.factor_prereg_seal_artifact_digest == (
        "sha256:b030ecc638c440fafcbee27cc45fe0eb0b242309c5bb6ddaaaf32980894e2859"
    )
    assert protocol.factor_prereg_fresh_artifact_id == 10_438_184_487
    assert protocol.factor_prereg_fresh_artifact_digest == (
        "sha256:1a405428a709d8994a0f70d16697be249a664f57afaa74732102c2a8ea32fc69"
    )
    assert protocol.implementation_seal_run_id == 35_084_553_267
    assert protocol.implementation_seal_artifact_id == 10_441_133_239
    assert protocol.implementation_seal_artifact_digest == (
        "sha256:8f8a3319f9213bf999b3ce2d911882fd9276daf575a7cde99ec33de9e8208b37"
    )
    assert protocol.implementation_fresh_artifact_id == 10_441_194_709
    assert protocol.implementation_fresh_artifact_digest == (
        "sha256:39f578321699d3deba1c1cc67ee9d5401c7a45573088d63b2c346f09721c7f52"
    )


def test_evaluation_authority_defines_unambiguous_robust_profit_statistics() -> None:
    protocol = _module().canonical_ppo_global_btc_regime_evaluation_protocol()

    assert protocol.factor_effect_statistic == (
        "per_symbol_median_matched_seed_excess_total_return"
    )
    assert protocol.candidate_return_statistic == (
        "per_symbol_median_candidate_total_return_across_matched_seeds"
    )
    assert protocol.seed_robustness_statistic == (
        "per_seed_cross_symbol_median_excess_total_return"
    )
    assert protocol.positive_threshold == 0.0
    assert protocol.positive_comparison == "strictly_greater_than"
    assert protocol.termination_rule == (
        "candidate_must_not_introduce_new_hard_or_economic_termination"
    )
    assert protocol.acceptance_rule_conjunction == "all_conditions_required"
    assert protocol.no_post_result_threshold_change is True
    assert protocol.no_post_result_metric_substitution is True
