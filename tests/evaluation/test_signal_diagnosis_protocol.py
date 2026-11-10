"""SD0 (#1146): the typed signal-diagnosis protocol is frozen, strict and result-blind."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import trade_rl.strategies.forecasts.lightgbm as lightgbm_module
from trade_rl.evaluation.comparison.bootstrap import moving_block_mean_test
from trade_rl.evaluation.dev_realdata_study.dataset_build import (
    STUDY_ARTIFACT_DIGEST,
    STUDY_DATASET_ID,
)
from trade_rl.evaluation.dev_realdata_study.protocol import PROTOCOL as DEV_PROTOCOL
from trade_rl.evaluation.forward_study.registry import Fs1Registry
from trade_rl.evaluation.signal_diagnosis.protocol import (
    PROTOCOL_SCHEMA,
    ForbiddenEvaluationWindowError,
    SignalDiagnosisProtocolError,
    SignalDiagnosisRunRequest,
    admit_run,
    default_signal_diagnosis_protocol,
    load_signal_diagnosis_protocol,
    protocol_canonical_bytes,
    protocol_digest,
    protocol_from_payload,
    reject_forbidden_window,
)
from trade_rl.integrations.binance.dataset import binance_multitimeframe_feature_specs
from trade_rl.integrations.market_store.embargo import (
    FS1_WARMUP_START,
    SEALED_WINDOW_START,
)
from trade_rl.strategies.forecasts.lightgbm import (
    fit_lightgbm_forecast_from_training_set,
)
from trade_rl.strategies.forecasts.prequential import PrequentialSchedule
from trade_rl.strategies.forecasts.ridge import fit_ridge_forecast_from_training_set
from trade_rl.strategies.forecasts.supervised import CausalForecastTrainingSet

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "report" / "issue1146-sd0-signal-diagnosis-protocol-v2-20261110.json"
V1_BLOCKED_DIGEST = "7c353bf45feea49aa421172f6a8df4d9ed39789585d438c7a4b57d1409e5dfc9"
PINNED_PROTOCOL_DIGEST = (
    "7a550e789ce76fb2bfb598a9e1bc087bfc4ba30747ff0f845b2b79f51ff8f570"
)
SHA_A = "a" * 64


def _payload() -> dict[str, Any]:
    decoded = json.loads(protocol_canonical_bytes(default_signal_diagnosis_protocol()))
    assert isinstance(decoded, dict)
    return decoded


def _raw(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _dict_paths(value: object, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Every path to a JSON object (including the root) inside a payload."""

    paths: list[tuple[str, ...]] = []
    if isinstance(value, dict):
        paths.append(prefix)
        for key, child in value.items():
            paths.extend(_dict_paths(child, (*prefix, key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_dict_paths(child, (*prefix, str(index))))
    return paths


def _walk(payload: Any, path: tuple[str, ...]) -> Any:
    node = payload
    for step in path:
        node = node[int(step)] if isinstance(node, list) else node[step]
    return node


_OBJECT_PATHS = _dict_paths(_payload())
_FIELD_PATHS = [
    (path, key) for path in _OBJECT_PATHS for key in _walk(_payload(), path)
]


def _utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


# ---------------------------------------------------------------- identity / digest


def test_default_protocol_is_frozen_and_its_digest_is_pinned() -> None:
    protocol = default_signal_diagnosis_protocol()
    raw = protocol_canonical_bytes(protocol)

    assert protocol.schema_version == PROTOCOL_SCHEMA == "signal_diagnosis_protocol_v2"
    assert protocol_digest(protocol) != V1_BLOCKED_DIGEST
    assert protocol_digest(protocol) == hashlib.sha256(raw).hexdigest()
    assert protocol_digest(protocol) == PINNED_PROTOCOL_DIGEST
    assert default_signal_diagnosis_protocol() == protocol
    assert b"\r" not in raw and b"\n" not in raw
    with pytest.raises(FrozenInstanceError):
        protocol.card = "SD9"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        protocol.data.dataset_id = "0" * 64  # type: ignore[misc]


def test_committed_protocol_report_is_the_canonical_bytes_of_the_default() -> None:
    protocol = default_signal_diagnosis_protocol()
    committed = REPORT.read_bytes()

    assert committed == protocol_canonical_bytes(protocol)
    loaded = load_signal_diagnosis_protocol(
        committed, expected_digest=PINNED_PROTOCOL_DIGEST
    )
    assert loaded == protocol


def test_status_and_claim_scope_say_known_used_exploratory_and_planned() -> None:
    protocol = default_signal_diagnosis_protocol()

    assert protocol.issue == 1146
    assert protocol.parent_issue == 810
    assert protocol.follows_issue == 1025
    assert protocol.card == "SD0"
    assert protocol.execution_status == "planned_not_executed"
    assert protocol.claim_ceiling == "known_used_exploratory_screening_only"
    assert protocol.data.known_used is True
    assert protocol.data.evidence_scope == "known_used_post_hoc_exploratory"
    assert set(protocol.forbidden_claims) >= {
        "edge",
        "profitability",
        "g4_economic_evidence",
        "g5_eligibility",
        "unused_data_validity",
        "final_evaluation",
        "live_trading",
    }
    # SD0 owns no runner: nothing in the payload names an executable.
    text = protocol_canonical_bytes(protocol).decode("utf-8")
    for forbidden in ("allow_network", "run_id", "output_root", "economic_result"):
        assert forbidden not in text


# ---------------------------------------------------------------- data scope / pins


def test_dataset_pins_equal_the_existing_development_dataset_identity() -> None:
    data = default_signal_diagnosis_protocol().data

    assert data.dataset_id == STUDY_DATASET_ID == DEV_PROTOCOL.dataset_id
    assert data.dataset_id.startswith("0b955471") and data.dataset_id.endswith("e174")
    assert data.artifact_digest == STUDY_ARTIFACT_DIGEST
    assert data.symbols == ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
    assert data.base_interval == "1h"
    assert data.feature_timeframes == ("4h", "1d")
    assert data.n_features == 167 and data.n_global_features == 4
    assert data.reference_block == "bot-development-2024q1"
    assert data.reference_block_role == "reference_only_not_evidence"
    for digest in (
        data.dataset_id,
        data.artifact_digest,
        data.arrays_digest,
        data.feature_config_digest,
    ):
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")


def test_window_end_and_forbidden_spans_come_from_the_fs1_registry_and_embargo() -> (
    None
):
    protocol = default_signal_diagnosis_protocol()
    registry = Fs1Registry()

    assert protocol.data.evaluation_end == registry.warmup_only_start + "Z"
    assert protocol.data.forbidden_spans[0].start == registry.warmup_only_start + "Z"
    assert protocol.data.forbidden_spans[0].end == registry.scoring_start + "Z"
    assert protocol.data.forbidden_spans[1].start == registry.scoring_start + "Z"
    assert protocol.data.forbidden_spans[1].end is None
    # The market-data store pins the same instants (it may not be imported by the
    # evaluation package, so the equality is asserted here instead).
    assert protocol.data.evaluation_end == FS1_WARMUP_START.isoformat().replace(
        "+00:00", "Z"
    )
    assert protocol.data.forbidden_spans[
        1
    ].start == SEALED_WINDOW_START.isoformat().replace("+00:00", "Z")
    assert protocol.data.evaluation_start == "2021-01-01T00:00:00Z"
    assert protocol.data.window_semantics == "half_open_bar_period_interval"
    assert protocol.data.labels_must_mature_inside_window is True


# ---------------------------------------------------------------- fixed grids


def test_feature_families_partition_the_maintained_167_feature_contract() -> None:
    protocol = default_signal_diagnosis_protocol()
    specs = binance_multitimeframe_feature_specs(
        base_timeframe="1h", feature_timeframes=("4h", "1d")
    )
    family_of = {
        name: family.family_id
        for family in protocol.families
        for name in family.feature_names
    }

    assert [family.family_id for family in protocol.families] == [
        "return_momentum",
        "oscillator",
        "trend",
        "volatility_range",
        "candle_shape",
        "volume_flow",
        "funding",
        "cross_asset",
    ]
    assert sum(len(family.feature_names) for family in protocol.families) == 167
    assert len(family_of) == 167  # no feature belongs to two families
    assert len(specs) == 167
    assert all(spec.name in family_of for spec in specs)  # every feature is assigned
    for spec in specs:
        family = next(
            f for f in protocol.families if f.family_id == family_of[spec.name]
        )
        assert spec.kind.value in family.kinds, spec.name
    assert {kind for f in protocol.families for kind in f.kinds} == {
        spec.kind.value for spec in specs
    }


def test_grids_equal_the_values_fixed_in_issue_1146() -> None:
    protocol = default_signal_diagnosis_protocol()

    assert protocol.horizons.primary_bars == 24
    assert protocol.horizons.secondary_bars == (4, 72)
    assert protocol.horizons.secondary_role == "reported_not_decisive"
    assert protocol.measurement.rank_ic_estimator.startswith("spearman")
    assert protocol.measurement.linear_ic_estimator.startswith("pearson")
    assert protocol.stability.reported_slices == (
        "year",
        "symbol",
        "regime",
        "extreme_day_exclusion",
    )
    assert protocol.stability.decisive_slices == ("year",)
    assert [t.target_id for t in protocol.targets] == [
        "raw_forward_log_return",
        "cross_sectionally_demeaned_forward_log_return",
        "cost_adjusted_sign_payoff",
        "drift_excess_payoff",
    ]
    assert [t.transform_id for t in protocol.transforms] == [
        "raw",
        "train_window_standardise_clip3",
    ]
    assert protocol.transforms[1].standardise is True
    assert protocol.transforms[1].clip_abs == 3.0
    assert protocol.models.model_ids == ("ridge_inner_cv", "lightgbm_fixed")
    assert protocol.models.ridge.penalty_grid == (0.1, 1.0, 10.0, 100.0, 1000.0)
    assert protocol.models.ridge.penalty_scope == "train_window_only"
    assert protocol.models.ridge.inner_cv_scheme == "forward_chaining_expanding"
    assert protocol.prequential.refit_scheme == "monthly_expanding_refit"
    assert protocol.prequential.refit_interval_days == 30
    assert protocol.prequential.activation_delay_hours == 0
    assert protocol.costs.fee_bps_per_side == 5
    assert protocol.costs.spread_bps_per_side == 2
    assert protocol.costs.round_trip_cost_bps == 14
    assert protocol.costs.k_grid == (1, 2)
    assert protocol.costs.cost_multipliers == (1, 2)
    assert protocol.bootstrap.method == "moving_block_mean_test"
    assert protocol.bootstrap.block_days == 20
    assert protocol.bootstrap.resamples == 2000
    assert protocol.bootstrap.seed == 0
    assert protocol.bootstrap.confidence_level == 0.95
    assert protocol.multiple_testing.reported == ("holm_fwer", "bh_fdr")
    assert protocol.multiple_testing.holm_fwer_alpha == 0.05
    assert protocol.multiple_testing.bh_fdr_q == 0.10
    assert protocol.multiple_testing.decision_procedure == "bh_fdr"
    assert protocol.stability.full_years == (2022, 2023, 2024, 2025)
    assert protocol.stability.partial_years_reported_only == (2021, 2026)
    assert protocol.stability.regime_lookback_days == 30
    assert protocol.stability.extreme_day_trim_fraction_per_tail == 0.01
    assert [b.baseline_id for b in protocol.baselines] == [
        "zero_forecast",
        "drift_expanding_mean_monthly",
        "drift_sign_policy_zero_cost",
        "beta_regression_on_market_mean",
    ]


def test_decision_rules_and_budget_fix_the_section_5_and_6_thresholds() -> None:
    protocol = default_signal_diagnosis_protocol()
    rules = protocol.decision_rules
    budget = protocol.budget

    assert protocol.verdict_labels == (
        "INVALID",
        "NO_SIGNAL",
        "SIGNAL_NOT_TRADABLE",
        "SIGNAL_CANDIDATE",
    )
    assert rules.min_full_years_same_sign == 4
    assert rules.candidate_ci_lower_floor == 0.0
    assert rules.candidate_stress_point_floor == 0.0
    assert rules.tradability_cost_multiplier == 1
    assert rules.stress_cost_multiplier == 2
    assert rules.primary_horizon_bars == 24
    assert rules.primary_transform_id == "train_window_standardise_clip3"
    assert rules.signal_target_id == "raw_forward_log_return"
    assert (
        rules.persistence_target_id == "cross_sectionally_demeaned_forward_log_return"
    )
    assert rules.tradability_target_id == "drift_excess_payoff"
    assert (
        rules.candidate_ranking == "drift_excess_ci_lower_cost_x1_desc_then_cell_id_asc"
    )
    assert budget.single_run is True
    assert budget.save_all_cells is True
    assert budget.post_result_additions_allowed is False
    assert budget.retune_prior_learner_on_this_data is False
    assert budget.max_candidates == 3
    assert budget.native_replay_verification_cap == 3


def test_lightgbm_spec_equals_the_constructor_the_repo_fitter_uses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeRegressor:
        def __init__(self, **params: object) -> None:
            captured.update(params)

        def fit(self, *args: object, **kwargs: object) -> FakeRegressor:
            return self

        def predict(self, features: object) -> object:
            return np.zeros(1)

    monkeypatch.setattr(
        lightgbm_module, "_lightgbm_regressor_factory", lambda: FakeRegressor
    )
    training = CausalForecastTrainingSet(
        feature_indices=(0, 1),
        features=np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]]),
        labels=np.array([0.1, -0.1, 0.2]),
        label_end_times=np.array(
            ["2020-01-02", "2020-01-03", "2020-01-04"], dtype="datetime64[ns]"
        ),
        sample_weights=np.ones(3),
        fit_cutoff=np.datetime64("2020-02-01T00:00:00", "ns"),
        horizon_hours=24,
    )
    fit_lightgbm_forecast_from_training_set(training, random_state=0)

    spec = default_signal_diagnosis_protocol().models.lightgbm
    assert captured == {
        "objective": spec.objective,
        "n_estimators": spec.n_estimators,
        "learning_rate": spec.learning_rate,
        "num_leaves": spec.num_leaves,
        "max_depth": spec.max_depth,
        "min_child_samples": spec.min_child_samples,
        "subsample": spec.subsample,
        "colsample_bytree": spec.colsample_bytree,
        "reg_lambda": spec.reg_lambda,
        "random_state": spec.random_state,
        "n_jobs": spec.n_jobs,
        "verbosity": spec.verbosity,
    }


def test_existing_components_accept_the_protocol_parameters() -> None:
    protocol = default_signal_diagnosis_protocol()
    # Synthetic numbers only: this checks signatures, not market behaviour.
    result = moving_block_mean_test(
        tuple(math.sin(i) for i in range(120)),
        n_bootstrap=protocol.bootstrap.resamples,
        seed=protocol.bootstrap.seed,
        block_size=protocol.bootstrap.block_days,
    )
    assert result.block_size == protocol.bootstrap.block_days
    assert result.lower_ci <= result.upper_ci

    schedule = PrequentialSchedule(
        first_fit_time=np.datetime64(protocol.prequential.first_fit_time[:-1], "ns"),
        refit_interval_days=protocol.prequential.refit_interval_days,
        activation_delay_hours=protocol.prequential.activation_delay_hours,
    )
    cutoffs = schedule.cutoffs(np.datetime64("2021-08-01T00:00:00", "ns"))
    assert len(cutoffs) >= 4
    assert cutoffs[1] - cutoffs[0] == np.timedelta64(30, "D")

    ridge_training = CausalForecastTrainingSet(
        feature_indices=(0,),
        features=np.array([[0.0], [1.0], [2.0], [3.0]]),
        labels=np.array([0.0, 0.1, 0.2, 0.3]),
        label_end_times=np.array(
            ["2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
            dtype="datetime64[ns]",
        ),
        sample_weights=np.ones(4),
        fit_cutoff=np.datetime64("2020-02-01T00:00:00", "ns"),
        horizon_hours=24,
    )
    for penalty in protocol.models.ridge.penalty_grid:
        fit_ridge_forecast_from_training_set(ridge_training, alpha=penalty)


def test_round_trip_cost_is_the_development_protocol_fee_plus_spread_both_sides() -> (
    None
):
    costs = default_signal_diagnosis_protocol().costs

    assert costs.fee_bps_per_side == round(DEV_PROTOCOL.fee_rate * 10_000)
    assert costs.spread_bps_per_side == round(DEV_PROTOCOL.spread_rate * 10_000)
    assert costs.round_trip_cost_bps == 2 * (
        costs.fee_bps_per_side + costs.spread_bps_per_side
    )
    assert DEV_PROTOCOL.bootstrap_block_days == 20
    assert DEV_PROTOCOL.bootstrap_resamples == 2000
    assert DEV_PROTOCOL.bootstrap_seed == 0
    assert DEV_PROTOCOL.refit_interval_days == 30


# ---------------------------------------------------------------- strict structure


@pytest.mark.parametrize(
    ("path", "key"), _FIELD_PATHS, ids=["/".join((*p, k)) for p, k in _FIELD_PATHS]
)
def test_a_missing_field_at_any_level_fails_closed(
    path: tuple[str, ...], key: str
) -> None:
    payload = _payload()
    del _walk(payload, path)[key]

    with pytest.raises(SignalDiagnosisProtocolError, match="missing"):
        protocol_from_payload(payload)


@pytest.mark.parametrize(
    "path", _OBJECT_PATHS, ids=["/".join(p) or "root" for p in _OBJECT_PATHS]
)
def test_an_extra_field_at_any_level_fails_closed(path: tuple[str, ...]) -> None:
    payload = _payload()
    _walk(payload, path)["unexpected_extra_field"] = 1

    with pytest.raises(SignalDiagnosisProtocolError, match="unexpected"):
        protocol_from_payload(payload)


@pytest.mark.parametrize(
    "claim_field",
    [
        "edge",
        "edge_claim",
        "g4_authorized",
        "g5_eligible",
        "profitability",
        "production_eligible",
        "live_trading_authorized",
        "economic_execution_authorized",
        "winner",
    ],
)
def test_claim_fields_are_refused_even_when_false(claim_field: str) -> None:
    payload = _payload()
    payload[claim_field] = False

    with pytest.raises(SignalDiagnosisProtocolError, match="claim field"):
        protocol_from_payload(payload)
    nested = _payload()
    nested["budget"][claim_field] = True
    with pytest.raises(SignalDiagnosisProtocolError, match="claim field"):
        protocol_from_payload(nested)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["horizons"].__setitem__("secondary_bars", []),
        lambda p: p.__setitem__("families", []),
        lambda p: p["families"][0].__setitem__("feature_names", []),
        lambda p: p.__setitem__("targets", []),
        lambda p: p.__setitem__("transforms", []),
        lambda p: p["models"]["ridge"].__setitem__("penalty_grid", []),
        lambda p: p["models"].__setitem__("model_ids", []),
        lambda p: p["costs"].__setitem__("k_grid", []),
        lambda p: p["costs"].__setitem__("cost_multipliers", []),
        lambda p: p["multiple_testing"].__setitem__("reported", []),
        lambda p: p["stability"].__setitem__("full_years", []),
        lambda p: p.__setitem__("baselines", []),
        lambda p: p.__setitem__("verdict_labels", []),
        lambda p: p["data"].__setitem__("symbols", []),
        lambda p: p["data"].__setitem__("forbidden_spans", []),
    ],
    ids=[
        "secondary_horizons",
        "families",
        "family_features",
        "targets",
        "transforms",
        "penalty_grid",
        "model_ids",
        "k_grid",
        "cost_multipliers",
        "reported_corrections",
        "full_years",
        "baselines",
        "verdict_labels",
        "symbols",
        "forbidden_spans",
    ],
)
def test_grids_cannot_be_empty(mutate: Any) -> None:
    payload = _payload()
    mutate(payload)

    with pytest.raises(SignalDiagnosisProtocolError):
        protocol_from_payload(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["horizons"].__setitem__("primary_bars", 12),
        lambda p: p["horizons"].__setitem__("secondary_bars", [4, 48]),
        lambda p: p["models"]["ridge"].__setitem__("penalty_grid", [0.1, 1.0, 10.0]),
        lambda p: p["models"]["ridge"].__setitem__(
            "penalty_grid", [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
        ),
        lambda p: p["models"]["lightgbm"].__setitem__("n_estimators", 65),
        lambda p: p["costs"].__setitem__("k_grid", [1, 2, 3]),
        lambda p: p["costs"].__setitem__("cost_multipliers", [1, 3]),
        lambda p: p["bootstrap"].__setitem__("resamples", 1000),
        lambda p: p["bootstrap"].__setitem__("seed", 1),
        lambda p: p["multiple_testing"].__setitem__("bh_fdr_q", 0.05),
        lambda p: p["decision_rules"].__setitem__("min_full_years_same_sign", 3),
        lambda p: p["budget"].__setitem__("max_candidates", 4),
        lambda p: p["stability"].__setitem__("full_years", [2022, 2023, 2024]),
        lambda p: p["stability"].__setitem__("min_covered_months_per_full_year", 9),
        lambda p: p["stability"].__setitem__("covered_month_min_valid_days", 10),
        lambda p: p["families"][1]["feature_names"].pop(),
        lambda p: p["data"].__setitem__("evaluation_end", "2026-09-01T00:00:00Z"),
        lambda p: p.__setitem__("schema_version", "signal_diagnosis_protocol_v3"),
    ],
    ids=[
        "primary_horizon",
        "secondary_horizons",
        "penalty_grid_shrunk",
        "penalty_grid_extended",
        "lightgbm_param",
        "k_grid",
        "cost_multipliers",
        "bootstrap_resamples",
        "bootstrap_seed",
        "fdr_q",
        "year_threshold",
        "candidate_cap",
        "full_years",
        "coverage_months",
        "coverage_days",
        "family_member",
        "window_end",
        "schema_version",
    ],
)
def test_grids_cannot_be_altered_without_a_version_change(mutate: Any) -> None:
    payload = _payload()
    mutate(payload)

    with pytest.raises(SignalDiagnosisProtocolError):
        protocol_from_payload(payload)


def test_direct_construction_cannot_alter_a_frozen_grid() -> None:
    protocol = default_signal_diagnosis_protocol()

    with pytest.raises(SignalDiagnosisProtocolError, match="new protocol version"):
        replace(protocol, baselines=protocol.baselines[:1])
    with pytest.raises(SignalDiagnosisProtocolError):
        replace(protocol.horizons, secondary_bars=())
    with pytest.raises(SignalDiagnosisProtocolError):
        replace(protocol.models.ridge, penalty_grid=(1.0, 1.0))
    with pytest.raises(SignalDiagnosisProtocolError):
        replace(protocol.budget, post_result_additions_allowed=True)
    with pytest.raises(SignalDiagnosisProtocolError):
        replace(protocol.budget, single_run=False)
    with pytest.raises(SignalDiagnosisProtocolError):
        replace(protocol.budget, save_all_cells=False)
    with pytest.raises(SignalDiagnosisProtocolError):
        replace(protocol.data, known_used=False)
    with pytest.raises(SignalDiagnosisProtocolError):
        replace(protocol, execution_status="executed")
    with pytest.raises(SignalDiagnosisProtocolError):
        replace(protocol, claim_ceiling="edge_established")


def test_scalar_types_are_exact() -> None:
    for mutate in (
        lambda p: p["horizons"].__setitem__("primary_bars", 24.0),
        lambda p: p["horizons"].__setitem__("primary_bars", True),
        lambda p: p["bootstrap"].__setitem__("confidence_level", 1),
        lambda p: p["budget"].__setitem__("single_run", 1),
        lambda p: p["models"]["ridge"].__setitem__(
            "penalty_grid", [0.1, 1, 10, 100, 1000]
        ),
        lambda p: p["data"].__setitem__("dataset_id", 5),
        lambda p: p.__setitem__("families", {"a": 1}),
        lambda p: p["multiple_testing"].__setitem__("bh_fdr_q", float("nan")),
    ):
        payload = _payload()
        mutate(payload)
        with pytest.raises(SignalDiagnosisProtocolError):
            protocol_from_payload(payload)


# ---------------------------------------------------------------- strict load


def test_strict_load_round_trips_the_canonical_bytes() -> None:
    protocol = default_signal_diagnosis_protocol()
    raw = protocol_canonical_bytes(protocol)

    assert (
        load_signal_diagnosis_protocol(raw, expected_digest=protocol_digest(protocol))
        == protocol
    )


def test_strict_load_rejects_any_tampered_byte() -> None:
    protocol = default_signal_diagnosis_protocol()
    raw = protocol_canonical_bytes(protocol)
    digest = protocol_digest(protocol)
    position = raw.index(b'"primary_bars":24') + len(b'"primary_bars":2')
    tampered = raw[:position] + b"5" + raw[position + 1 :]

    assert tampered != raw
    with pytest.raises(SignalDiagnosisProtocolError, match="digest"):
        load_signal_diagnosis_protocol(tampered, expected_digest=digest)
    with pytest.raises(SignalDiagnosisProtocolError, match="digest"):
        load_signal_diagnosis_protocol(raw + b" ", expected_digest=digest)
    with pytest.raises(SignalDiagnosisProtocolError, match="digest"):
        load_signal_diagnosis_protocol(raw, expected_digest="0" * 64)


@pytest.mark.parametrize("expected", ["", "abc", "A" * 64, "g" * 64, 5, None])
def test_strict_load_requires_a_lowercase_sha256_expected_digest(
    expected: object,
) -> None:
    raw = protocol_canonical_bytes(default_signal_diagnosis_protocol())

    with pytest.raises(SignalDiagnosisProtocolError, match="expected_digest"):
        load_signal_diagnosis_protocol(raw, expected_digest=expected)  # type: ignore[arg-type]


def test_a_self_consistent_tamper_is_still_rejected_by_the_frozen_reference() -> None:
    payload = _payload()
    payload["models"]["ridge"]["penalty_grid"] = [0.1, 1.0, 10.0]
    raw = _raw(payload)  # attacker also supplies the matching digest

    with pytest.raises(SignalDiagnosisProtocolError, match="new protocol version"):
        load_signal_diagnosis_protocol(
            raw, expected_digest=hashlib.sha256(raw).hexdigest()
        )


def test_strict_load_rejects_non_canonical_duplicate_and_nonfinite_json() -> None:
    protocol = default_signal_diagnosis_protocol()
    raw = protocol_canonical_bytes(protocol)

    def attempt(candidate: bytes) -> None:
        load_signal_diagnosis_protocol(
            candidate, expected_digest=hashlib.sha256(candidate).hexdigest()
        )

    pretty = json.dumps(json.loads(raw), indent=2, sort_keys=True).encode("utf-8")
    with pytest.raises(SignalDiagnosisProtocolError, match="canonical"):
        attempt(pretty)
    decoded = json.loads(raw)
    unsorted = json.dumps(
        {key: decoded[key] for key in reversed(list(decoded))}, separators=(",", ":")
    ).encode("utf-8")
    with pytest.raises(SignalDiagnosisProtocolError, match="canonical"):
        attempt(unsorted)
    duplicate = raw.replace(b'"card":"SD0"', b'"card":"SD0","card":"SD1"', 1)
    with pytest.raises(SignalDiagnosisProtocolError, match="duplicate"):
        attempt(duplicate)
    with pytest.raises(SignalDiagnosisProtocolError, match="JSON"):
        attempt(raw.replace(b'"seed":0', b'"seed":NaN', 1))
    for bad in (b"", b"[]", b"null", b"\xff\xfe", b"{"):
        with pytest.raises(SignalDiagnosisProtocolError):
            attempt(bad)
    with pytest.raises(SignalDiagnosisProtocolError):
        load_signal_diagnosis_protocol("{}", expected_digest=SHA_A)  # type: ignore[arg-type]


# ---------------------------------------------------------------- hard window rejection


def test_windows_inside_the_development_span_are_accepted() -> None:
    reject_forbidden_window(_utc(2021, 1, 1), _utc(2026, 8, 1))
    reject_forbidden_window(_utc(2025, 1, 1), _utc(2026, 1, 1))
    reject_forbidden_window(_utc(2026, 7, 31), _utc(2026, 8, 1))
    # An aware non-UTC bound is normalised, not trusted as written.
    reject_forbidden_window(
        _utc(2025, 1, 1),
        datetime(2026, 8, 1, 9, 0, tzinfo=timezone(timedelta(hours=9))),
    )


@pytest.mark.parametrize(
    ("start", "end", "needle"),
    [
        (_utc(2021, 1, 1), _utc(2026, 8, 1, 0, 0, 1), "warm-up"),
        (_utc(2026, 7, 31), _utc(2026, 8, 2), "warm-up"),
        (_utc(2026, 8, 1), _utc(2026, 9, 1), "warm-up"),
        (_utc(2026, 8, 1), _utc(2026, 11, 1), "warm-up"),
        (_utc(2026, 9, 1), _utc(2026, 10, 1), "warm-up"),
        (_utc(2021, 1, 1), _utc(2026, 11, 1), "warm-up"),
        (_utc(2021, 1, 1), _utc(2026, 11, 1, 0, 0, 1), "sealed"),
        (_utc(2026, 10, 31), _utc(2026, 12, 1), "sealed"),
        (_utc(2026, 11, 1), _utc(2027, 1, 1), "sealed"),
        (_utc(2027, 6, 1), _utc(2027, 7, 1), "sealed"),
        (_utc(2021, 1, 1), _utc(2027, 5, 1), "sealed"),
        (
            datetime(2026, 8, 1, 9, 0, tzinfo=timezone(timedelta(hours=9))),
            _utc(2026, 8, 2),
            None,
        ),
    ],
)
def test_any_window_touching_the_warmup_span_or_sealed_window_is_rejected(
    start: datetime, end: datetime, needle: str | None
) -> None:
    with pytest.raises(ForbiddenEvaluationWindowError) as caught:
        reject_forbidden_window(start, end)
    if needle is not None:
        assert needle in str(caught.value)
    assert isinstance(caught.value, SignalDiagnosisProtocolError)
    assert isinstance(caught.value, ValueError)


def test_malformed_windows_are_rejected_before_the_span_check() -> None:
    with pytest.raises(SignalDiagnosisProtocolError, match="timezone"):
        reject_forbidden_window(datetime(2025, 1, 1), _utc(2025, 2, 1))
    with pytest.raises(SignalDiagnosisProtocolError, match="timezone"):
        reject_forbidden_window(_utc(2025, 1, 1), datetime(2025, 2, 1))
    with pytest.raises(SignalDiagnosisProtocolError, match="before"):
        reject_forbidden_window(_utc(2025, 2, 1), _utc(2025, 2, 1))
    with pytest.raises(SignalDiagnosisProtocolError, match="before"):
        reject_forbidden_window(_utc(2025, 3, 1), _utc(2025, 2, 1))
    with pytest.raises(SignalDiagnosisProtocolError, match="datetime"):
        reject_forbidden_window("2025-01-01", _utc(2025, 2, 1))  # type: ignore[arg-type]


def test_a_protocol_whose_window_reaches_a_forbidden_span_cannot_be_built() -> None:
    payload = _payload()
    payload["data"]["evaluation_end"] = "2026-08-01T00:00:01Z"
    with pytest.raises(ForbiddenEvaluationWindowError):
        protocol_from_payload(payload)
    payload = _payload()
    payload["data"]["evaluation_end"] = "2027-01-01T00:00:00Z"
    with pytest.raises(ForbiddenEvaluationWindowError):
        protocol_from_payload(payload)
    payload = _payload()
    payload["data"]["forbidden_spans"] = payload["data"]["forbidden_spans"][:1]
    with pytest.raises(SignalDiagnosisProtocolError):
        protocol_from_payload(payload)


# ---------------------------------------------------------------- run admission


def test_synthetic_scope_needs_no_authorization_and_no_real_window() -> None:
    protocol = default_signal_diagnosis_protocol()
    request = SignalDiagnosisRunRequest(
        data_scope="synthetic_only",
        protocol_digest=protocol_digest(protocol),
        evaluation_start=None,
        evaluation_end=None,
        real_scope_authorization_digest=None,
    )
    admit_run(request, protocol)

    with pytest.raises(SignalDiagnosisProtocolError, match="synthetic"):
        replace(request, real_scope_authorization_digest=SHA_A)
    with pytest.raises(SignalDiagnosisProtocolError, match="synthetic"):
        replace(
            request, evaluation_start=_utc(2021, 1, 1), evaluation_end=_utc(2022, 1, 1)
        )


def test_real_development_requires_a_lowercase_sha256_authorization_digest() -> None:
    protocol = default_signal_diagnosis_protocol()
    base = {
        "data_scope": "real_development",
        "protocol_digest": protocol_digest(protocol),
        "evaluation_start": _utc(2021, 1, 1),
        "evaluation_end": _utc(2026, 8, 1),
    }
    admit_run(
        SignalDiagnosisRunRequest(**base, real_scope_authorization_digest=SHA_A),
        protocol,
    )
    for bad in (None, "", "a" * 63, "A" * 64, "g" * 64, 7):
        with pytest.raises(SignalDiagnosisProtocolError, match="authorization"):
            SignalDiagnosisRunRequest(
                **base,
                real_scope_authorization_digest=bad,  # type: ignore[arg-type]
            )
    with pytest.raises(SignalDiagnosisProtocolError, match="data_scope"):
        SignalDiagnosisRunRequest(
            **{**base, "data_scope": "unused_final"},
            real_scope_authorization_digest=SHA_A,
        )


def test_real_development_refuses_forbidden_and_out_of_protocol_windows() -> None:
    protocol = default_signal_diagnosis_protocol()

    def request(start: datetime | None, end: datetime | None, **kw: object) -> Any:
        return SignalDiagnosisRunRequest(
            data_scope="real_development",
            protocol_digest=kw.get("protocol_digest", protocol_digest(protocol)),  # type: ignore[arg-type]
            evaluation_start=start,
            evaluation_end=end,
            real_scope_authorization_digest=SHA_A,
        )

    with pytest.raises(ForbiddenEvaluationWindowError):
        request(_utc(2021, 1, 1), _utc(2026, 8, 2))
    with pytest.raises(ForbiddenEvaluationWindowError):
        request(_utc(2026, 11, 1), _utc(2027, 1, 1))
    with pytest.raises(SignalDiagnosisProtocolError, match="window"):
        request(None, None)
    with pytest.raises(SignalDiagnosisProtocolError, match="protocol window"):
        admit_run(request(_utc(2020, 12, 31), _utc(2021, 6, 1)), protocol)
    with pytest.raises(SignalDiagnosisProtocolError, match="protocol digest"):
        admit_run(
            request(_utc(2021, 1, 1), _utc(2022, 1, 1), protocol_digest="b" * 64),
            protocol,
        )
    admit_run(request(_utc(2022, 1, 1), _utc(2023, 1, 1)), protocol)


def test_admission_returns_nothing_and_has_no_runner_or_reader() -> None:
    import trade_rl.evaluation.signal_diagnosis.protocol as module

    for name in dir(module):
        assert not name.lower().startswith(
            ("run_", "execute", "load_dataset", "fetch")
        ), name


def test_first_fit_partial_calendar_year_cannot_count_as_full_stability_year() -> None:
    """The OOS schedule starts in April 2021; 2021 is not a full stability year."""
    protocol = default_signal_diagnosis_protocol()
    first_fit_year = int(protocol.prequential.first_fit_time[:4])

    assert first_fit_year not in protocol.stability.full_years
    assert first_fit_year in protocol.stability.partial_years_reported_only
    assert protocol.stability.full_years == (2022, 2023, 2024, 2025)
    assert protocol.decision_rules.min_full_years_same_sign == len(
        protocol.stability.full_years
    )


def test_a_full_stability_year_must_start_at_or_after_the_first_usable_fit() -> None:
    protocol = default_signal_diagnosis_protocol()
    stability = replace(
        protocol.stability,
        full_years=(2021, 2022, 2023, 2024, 2025),
        partial_years_reported_only=(2026,),
    )

    with pytest.raises(SignalDiagnosisProtocolError, match="first usable fit"):
        replace(protocol, stability=stability)


def test_a_full_stability_year_must_end_inside_the_evaluation_window() -> None:
    protocol = default_signal_diagnosis_protocol()
    stability = replace(
        protocol.stability,
        full_years=(2022, 2023, 2024, 2025, 2026),
        partial_years_reported_only=(2021,),
    )

    with pytest.raises(SignalDiagnosisProtocolError, match="evaluation window"):
        replace(protocol, stability=stability)


def test_full_and_partial_years_must_partition_the_out_of_sample_span() -> None:
    protocol = default_signal_diagnosis_protocol()
    # 2021 (first fit in April) and 2026 (window ends in August) are the partial years.
    dropped = replace(protocol.stability, partial_years_reported_only=(2026,))
    invented = replace(
        protocol.stability, partial_years_reported_only=(2021, 2026, 2019)
    )

    with pytest.raises(SignalDiagnosisProtocolError, match="partition"):
        replace(protocol, stability=dropped)
    with pytest.raises(SignalDiagnosisProtocolError, match="partition"):
        replace(protocol, stability=invented)


def test_year_eligibility_is_a_fixed_coverage_rule_not_a_result_choice() -> None:
    stability = default_signal_diagnosis_protocol().stability

    assert stability.min_covered_months_per_full_year == 12
    assert stability.covered_month_min_valid_days == 15
    for needle in (
        "UTC calendar month",
        "valid daily rank-IC increment",
        "fewer than",
        "does not count",
    ):
        assert needle in stability.coverage_rule, needle
    # The year sign is taken on the decisive (demeaned) persistence target only.
    protocol = default_signal_diagnosis_protocol()
    assert "persistence target" in stability.year_sign_statistic
    assert "strictly greater than zero" in stability.year_sign_statistic
    assert protocol.decision_rules.persistence_target_id in {
        t.target_id for t in protocol.targets
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: replace(s, min_covered_months_per_full_year=0),
        lambda s: replace(s, min_covered_months_per_full_year=13),
        lambda s: replace(s, covered_month_min_valid_days=0),
        lambda s: replace(s, covered_month_min_valid_days=29),
    ],
    ids=["months_zero", "months_over_twelve", "days_zero", "days_over_28"],
)
def test_coverage_rule_bounds_are_validated(mutate: Any) -> None:
    with pytest.raises(SignalDiagnosisProtocolError):
        mutate(default_signal_diagnosis_protocol().stability)


def test_v1_is_not_loadable_and_its_blocked_digest_is_not_reused() -> None:
    payload = _payload()
    payload["schema_version"] = "signal_diagnosis_protocol_v1"

    with pytest.raises(SignalDiagnosisProtocolError, match="unsupported protocol"):
        protocol_from_payload(payload)
    assert PINNED_PROTOCOL_DIGEST != V1_BLOCKED_DIGEST


def test_the_decisive_target_is_named_for_what_it_computes() -> None:
    protocol = default_signal_diagnosis_protocol()
    target = next(
        t
        for t in protocol.targets
        if t.target_id == protocol.decision_rules.persistence_target_id
    )
    definition = target.definition.lower()

    assert "cross-sectional mean" in definition
    assert "no regression coefficient" in definition
    assert "does not remove heterogeneous market beta" in definition
    text = protocol_canonical_bytes(protocol).decode("utf-8").lower()
    for stale in ("beta_removed", "beta-removed", "beta-robust", "survives removal"):
        assert stale not in text, stale
    assert {"beta_robust_signal", "beta_adjusted_alpha"} <= set(
        protocol.forbidden_claims
    )


def test_d4_excess_payoff_is_defined_against_a_zero_cost_drift_policy() -> None:
    protocol = default_signal_diagnosis_protocol()
    targets = {t.target_id: t.definition for t in protocol.targets}
    baselines = {b.baseline_id: b for b in protocol.baselines}
    excess = targets["drift_excess_payoff"]

    assert protocol.decision_rules.tradability_target_id == "drift_excess_payoff"
    assert "same row" in excess and "same cost multiplier" in excess
    assert "drift_sign_policy_zero_cost" in excess
    assert "not a beta-adjusted alpha" in excess
    assert baselines["drift_sign_policy_zero_cost"].role == "decisive_comparator"
    assert "charged no cost" in baselines["drift_sign_policy_zero_cost"].definition
    assert baselines["drift_expanding_mean_monthly"].role == "decisive_comparator"
    assert baselines["zero_forecast"].role == "reported_context"
    assert baselines["beta_regression_on_market_mean"].role == "reported_context"


def test_baselines_are_causal_with_explicit_update_and_availability_rules() -> None:
    protocol = default_signal_diagnosis_protocol()
    drift = next(
        b for b in protocol.baselines if b.baseline_id == "drift_expanding_mean_monthly"
    )
    beta = next(
        b
        for b in protocol.baselines
        if b.baseline_id == "beta_regression_on_market_mean"
    )

    assert "label maturity" in drift.definition
    assert "refit instant" in drift.definition
    assert "not yet matured" in drift.definition
    assert "never enters the verdict" in beta.definition
    assert "720" in beta.definition  # explicit minimum sample, else no estimate


def test_row_inclusion_rule_is_stated_on_the_bar_close_time_basis() -> None:
    measurement = default_signal_diagnosis_protocol().measurement
    rule = measurement.row_inclusion

    assert "close time t" in rule
    assert "evaluation_start < t" in rule
    assert "t + h bars <= evaluation_end" in rule


def test_a_label_closing_exactly_at_the_window_end_is_inside_one_bar_later_is_not() -> (
    None
):
    protocol = default_signal_diagnosis_protocol()
    end = datetime.fromisoformat(protocol.data.evaluation_end[:-1] + "+00:00")
    bar = timedelta(hours=protocol.horizons.bar_hours)

    reject_forbidden_window(end - timedelta(days=1), end)  # t + h == end
    with pytest.raises(ForbiddenEvaluationWindowError):
        reject_forbidden_window(end - timedelta(days=1), end + bar)
