from __future__ import annotations

from dataclasses import replace

import numpy as np

from tools.tmp_issue610_ppo_global_btc_regime_evaluation import (
    ACCEPT_CANDIDATE,
    EXPECTED_SEMANTIC_CHANGED_FIELDS,
    INVALID,
    KEEP_BASELINE,
    build_candidate_carrier_plan,
    evaluate_frozen_gate,
    validate_candidate_ppo_returns,
    validate_controlled_semantic_delta,
    validate_unaffected_raw_returns,
)
from trade_rl.evaluation.experiments.contracts import (
    ControlledFactor,
    ResolvedRunConfig,
    StudyPlan,
)
from trade_rl.strategies.rl.ppo import (
    PPO_GLOBAL_BTC_REGIME_CONTEXT,
    PPO_GLOBAL_BTC_REGIME_OBSERVATION_SCHEMA,
    PPO_GLOBAL_FEATURE_NAMES,
    PPO_OBSERVATION_SCHEMA,
)

_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
_SEEDS = (0, 1, 2, 3, 4)
_UNAFFECTED = (
    "trend",
    "mean_reversion",
    "ridge24",
    "lightgbm24",
    "cash",
    "constant_long",
    "constant_short",
)


def _baseline_config() -> ResolvedRunConfig:
    return ResolvedRunConfig(
        signal_name="log_return_24bar",
        signal_index=0,
        feature_names=("log_return_24bar", "funding_rate"),
        feature_indices=(0, 1),
        fit_symbol_names=_SYMBOLS,
        fit_symbol_indices=(0, 1, 2, 3, 4),
        fit_cutoff="2021-01-01T00:00:00.000000000",
        rule_entry_threshold=0.01,
        rule_exit_threshold=0.001,
        forecast_entry_threshold=0.01,
        forecast_exit_threshold=0.001,
        ppo_total_timesteps=100_000,
        ppo_seed=0,
        evaluation_start="2021-01-01T00:00:00.000000000",
        evaluation_stop_exclusive="2023-01-01T00:00:00.000000000",
        gross_budget=1.0,
        initial_capital=10_000.0,
        execution_overlay="zero_overlay_dataset_fields_authoritative",
        ppo_observation_schema=PPO_OBSERVATION_SCHEMA,
        ppo_global_feature_names=PPO_GLOBAL_FEATURE_NAMES,
        schema_version="resolved_run_config_v2",
    )


def _candidate_config() -> ResolvedRunConfig:
    return replace(
        _baseline_config(),
        ppo_observation_schema=PPO_GLOBAL_BTC_REGIME_OBSERVATION_SCHEMA,
        ppo_global_context=PPO_GLOBAL_BTC_REGIME_CONTEXT,
        schema_version="resolved_run_config_v3",
    )


def _source_plan() -> StudyPlan:
    return StudyPlan(
        research_question="baseline",
        dataset_id="1" * 64,
        dataset_artifact_schema="market_dataset_artifact_v1",
        dataset_artifact_digest="2" * 64,
        symbols=_SYMBOLS,
        baseline_config=_baseline_config(),
        ppo_seeds=_SEEDS,
        allowed_factors=(ControlledFactor.FEATURE_SET,),
        max_experiments=4,
        n_bootstrap=2_000,
        bootstrap_seed=1_729,
        implementation_digest="3" * 64,
        runtime_environment_digest="4" * 64,
    )


def _seedless(config: ResolvedRunConfig) -> dict[str, object]:
    payload = config.to_payload()
    payload.pop("ppo_seed")
    return payload


def _comparison() -> dict[str, object]:
    return {
        "schema_version": "controlled_evidence_comparison_v2",
        "seeds": list(_SEEDS),
        "by_symbol": {
            symbol: {
                "strategies": {
                    "ppo": {
                        "by_seed": {
                            str(seed): {"excess_total_return": 0.01 + seed * 0.001}
                            for seed in _SEEDS
                        },
                        "seed_aggregate": {"median_excess_total_return": 0.012},
                    }
                }
            }
            for symbol in _SYMBOLS
        },
        "cross_symbol": {"ppo": {"median_excess_total_return": 0.012}},
    }


def _return_matrix() -> dict[int, dict[tuple[str, str], np.ndarray]]:
    return {
        seed: {
            (symbol, strategy): np.array([0.0, 0.01, -0.005], dtype=np.float64)
            for symbol in _SYMBOLS
            for strategy in (*_UNAFFECTED, "ppo")
        }
        for seed in _SEEDS
    }


def test_carrier_plan_is_separate_but_preserves_frozen_source_authorities() -> None:
    source = _source_plan()
    candidate = _candidate_config()

    carrier = build_candidate_carrier_plan(
        source_plan=source,
        candidate_config=candidate,
        implementation_digest="a" * 64,
        runtime_environment_digest="b" * 64,
    )

    assert carrier.digest != source.digest
    assert carrier.dataset_id == source.dataset_id
    assert carrier.dataset_artifact_digest == source.dataset_artifact_digest
    assert carrier.symbols == source.symbols
    assert carrier.ppo_seeds == source.ppo_seeds
    assert carrier.n_bootstrap == source.n_bootstrap
    assert carrier.bootstrap_seed == source.bootstrap_seed
    assert carrier.max_experiments == source.max_experiments
    assert carrier.baseline_config == candidate
    assert carrier.implementation_digest == "a" * 64
    assert carrier.runtime_environment_digest == "b" * 64
    assert source.baseline_config == _baseline_config()


def test_semantic_delta_is_exactly_the_preregistered_global_btc_contract() -> None:
    baseline = _seedless(_baseline_config())
    candidate = _seedless(_candidate_config())

    changed, violations = validate_controlled_semantic_delta(baseline, candidate)

    assert changed == EXPECTED_SEMANTIC_CHANGED_FIELDS
    assert violations == ()

    drifted = dict(candidate)
    drifted["gross_budget"] = 2.0
    changed, violations = validate_controlled_semantic_delta(baseline, drifted)
    assert ("gross_budget",) in changed
    assert violations


def test_semantic_delta_rejects_noop_and_wrong_global_context() -> None:
    baseline = _seedless(_baseline_config())
    candidate = _seedless(_candidate_config())

    changed, violations = validate_controlled_semantic_delta(baseline, baseline)
    assert changed == ()
    assert violations

    candidate["ppo_global_context"] = "wrong"
    changed, violations = validate_controlled_semantic_delta(baseline, candidate)
    assert changed == EXPECTED_SEMANTIC_CHANGED_FIELDS
    assert violations


def test_unaffected_raw_returns_require_exact_equality_for_all_cells() -> None:
    baseline = _return_matrix()
    candidate = _return_matrix()

    assert (
        validate_unaffected_raw_returns(
            baseline,
            candidate,
            seeds=_SEEDS,
            symbols=_SYMBOLS,
            unaffected_strategies=_UNAFFECTED,
        )
        == ()
    )

    candidate[3][("XRPUSDT", "ridge24")] = np.array(
        [0.0, 0.01, -0.004], dtype=np.float64
    )
    violations = validate_unaffected_raw_returns(
        baseline,
        candidate,
        seeds=_SEEDS,
        symbols=_SYMBOLS,
        unaffected_strategies=_UNAFFECTED,
    )
    assert violations == ("unaffected raw returns changed: seed=3 XRPUSDT/ridge24",)


def test_candidate_ppo_returns_require_complete_finite_nonempty_cells() -> None:
    candidate = _return_matrix()
    assert (
        validate_candidate_ppo_returns(candidate, seeds=_SEEDS, symbols=_SYMBOLS) == ()
    )

    del candidate[4][("ADAUSDT", "ppo")]
    violations = validate_candidate_ppo_returns(
        candidate, seeds=_SEEDS, symbols=_SYMBOLS
    )
    assert violations == ("candidate PPO returns missing: seed=4 ADAUSDT",)

    candidate = _return_matrix()
    candidate[2][("BNBUSDT", "ppo")] = np.array([0.0, np.nan])
    violations = validate_candidate_ppo_returns(
        candidate, seeds=_SEEDS, symbols=_SYMBOLS
    )
    assert violations == ("candidate PPO returns non-finite: seed=2 BNBUSDT",)


def test_candidate_ppo_returns_reject_extra_seed_and_symbol_roster() -> None:
    candidate = _return_matrix()
    candidate[99] = {key: values.copy() for key, values in candidate[0].items()}
    violations = validate_candidate_ppo_returns(
        candidate, seeds=_SEEDS, symbols=_SYMBOLS
    )
    assert violations == ("candidate seed roster/order mismatch",)

    candidate = _return_matrix()
    candidate[0][("SOLUSDT", "ppo")] = np.array([0.0, 0.01, -0.005], dtype=np.float64)
    violations = validate_candidate_ppo_returns(
        candidate, seeds=_SEEDS, symbols=_SYMBOLS
    )
    assert violations == ("candidate symbol roster/order mismatch: seed=0",)


def test_frozen_gate_accepts_only_when_all_five_preregistered_conditions_hold() -> None:
    accepted = evaluate_frozen_gate(
        _comparison(),
        seeds=_SEEDS,
        symbols=_SYMBOLS,
        no_new_termination=True,
        unaffected_raw_returns_equal=True,
        validity_violations=(),
    )
    assert accepted["decision"] == ACCEPT_CANDIDATE
    assert accepted["positive_factor_effect_symbols"] == 5
    assert accepted["positive_cross_symbol_median_seeds"] == 5
    assert accepted["all_gates_pass"] is True

    comparison = _comparison()
    comparison["by_symbol"]["ETHUSDT"]["strategies"]["ppo"]["seed_aggregate"][
        "median_excess_total_return"
    ] = 0.0
    invalid = evaluate_frozen_gate(
        comparison,
        seeds=_SEEDS,
        symbols=_SYMBOLS,
        no_new_termination=True,
        unaffected_raw_returns_equal=True,
        validity_violations=(),
    )
    assert invalid["decision"] == INVALID
    assert invalid["evidence_valid"] is False


def test_frozen_gate_rejects_inconsistent_cross_symbol_aggregate() -> None:
    comparison = _comparison()
    comparison["cross_symbol"]["ppo"]["median_excess_total_return"] = 0.5

    result = evaluate_frozen_gate(
        comparison,
        seeds=_SEEDS,
        symbols=_SYMBOLS,
        no_new_termination=True,
        unaffected_raw_returns_equal=True,
        validity_violations=(),
    )

    assert result["decision"] == INVALID
    assert result["evidence_valid"] is False


def test_frozen_gate_requires_every_seed_cross_symbol_median_strictly_positive() -> (
    None
):
    comparison = _comparison()
    for symbol in _SYMBOLS:
        comparison["by_symbol"][symbol]["strategies"]["ppo"]["by_seed"]["3"][
            "excess_total_return"
        ] = -0.01
    for symbol in _SYMBOLS:
        comparison["by_symbol"][symbol]["strategies"]["ppo"]["seed_aggregate"][
            "median_excess_total_return"
        ] = 0.011
    comparison["cross_symbol"]["ppo"]["median_excess_total_return"] = 0.011

    result = evaluate_frozen_gate(
        comparison,
        seeds=_SEEDS,
        symbols=_SYMBOLS,
        no_new_termination=True,
        unaffected_raw_returns_equal=True,
        validity_violations=(),
    )
    assert result["decision"] == KEEP_BASELINE
    assert result["positive_cross_symbol_median_seeds"] == 4


def test_invalid_evidence_cannot_be_downgraded_to_keep_baseline() -> None:
    result = evaluate_frozen_gate(
        _comparison(),
        seeds=_SEEDS,
        symbols=_SYMBOLS,
        no_new_termination=True,
        unaffected_raw_returns_equal=True,
        validity_violations=("authority mismatch",),
    )
    assert result["decision"] == INVALID
    assert result["all_gates_pass"] is False

    result = evaluate_frozen_gate(
        _comparison(),
        seeds=_SEEDS,
        symbols=_SYMBOLS,
        no_new_termination=True,
        unaffected_raw_returns_equal=False,
        validity_violations=(),
    )
    assert result["decision"] == INVALID


def test_new_termination_is_valid_non_accept_not_invalid() -> None:
    result = evaluate_frozen_gate(
        _comparison(),
        seeds=_SEEDS,
        symbols=_SYMBOLS,
        no_new_termination=False,
        unaffected_raw_returns_equal=True,
        validity_violations=(),
    )
    assert result["decision"] == KEEP_BASELINE
    assert result["all_gates_pass"] is False
