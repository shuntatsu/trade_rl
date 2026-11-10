"""Frozen, versioned protocol of the signal diagnosis (Issue #1146, card SD0).

Issue #1025 (direct decision-value learning) ended INCONCLUSIVE. Before any new
learner is tried, this plan asks whether the current feature families carry a
predictive signal that survives removal of market beta, is stable out of sample in
time order, and exceeds round-trip cost. The question is answered by a single
diagnosis whose every grid, threshold and decision rule is fixed here, before any
market value is computed.

This module owns **only** the protocol: typed, immutable, canonical-JSON with a
SHA-256 digest, a strict loader that requires the expected digest, a hard refusal of
any evaluation window that touches the FS1 warm-up span or the sealed window, and
the admission check a later real-data run must pass (a separate authorization
digest, as in the #1025 plan contract). It reads no dataset, fetches nothing, fits
nothing and owns no runner. The data it describes is the already consumed
development Dataset, so every later result is a known-used, post hoc, exploratory
screening result: the protocol has no field that could carry an edge, G4, G5 or live
claim, and the loader refuses any such field.

A version freezes every value. :func:`protocol_from_payload` rebuilds the object
from a payload, validates each part and finally requires equality with the frozen
reference of the declared version, so an altered grid, threshold or window is
rejected until a new schema version (and a new reference) exists.

``signal_diagnosis_protocol_v3`` replaces the never-merged, review-blocked v1 and v2.
Neither is loadable. v1 counted 2021 (first fit April 2021) as a full stability year,
called a cross-sectional demeaning "beta removal", left the D4 "alpha" undefined and left
the drift and beta baselines without an estimation rule. v2 fixed those but still did not
say how the p-values behind the decisive Benjamini-Hochberg gate are produced. v3 freezes
that procedure (test, null, tail, dependence handling, BH family) before any result
exists; the cross-references below make the earlier defects unrepresentable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import types
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Union, get_args, get_origin, get_type_hints

from trade_rl.artifacts.canonical import canonical_json_bytes, to_json_value
from trade_rl.evaluation.forward_study.registry import Fs1Registry

PROTOCOL_SCHEMA: Final = "signal_diagnosis_protocol_v3"

SCOPE_SYNTHETIC: Final = "synthetic_only"
SCOPE_REAL: Final = "real_development"
DATA_SCOPES: Final = (SCOPE_SYNTHETIC, SCOPE_REAL)

VERDICT_INVALID: Final = "INVALID"
VERDICT_NO_SIGNAL: Final = "NO_SIGNAL"
VERDICT_NOT_TRADABLE: Final = "SIGNAL_NOT_TRADABLE"
VERDICT_CANDIDATE: Final = "SIGNAL_CANDIDATE"
VERDICT_LABELS: Final = (
    VERDICT_INVALID,
    VERDICT_NO_SIGNAL,
    VERDICT_NOT_TRADABLE,
    VERDICT_CANDIDATE,
)

EXECUTION_STATUS: Final = "planned_not_executed"
CLAIM_CEILING: Final = "known_used_exploratory_screening_only"
EVIDENCE_SCOPE: Final = "known_used_post_hoc_exploratory"
# The first usable fit is 2021-04-01, so 2021 is a partial (start-up) year; 2026 is
# partial because the window ends in August. Both are reported, never decisive.
FULL_YEARS: Final = (2022, 2023, 2024, 2025)
PARTIAL_YEARS: Final = (2021, 2026)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CLAIM_WORDS = re.compile(
    r"edge|g4|g5|profit|winner|authoriz|production|live|eligib|sharpe|economic|alpha_claim"
)

# The pins below copy the identity of the consumed development Dataset. A test pins
# their equality with ``dev_realdata_study`` (importing its builder here would pull
# in a network client).
DATASET_ID: Final = "0b95547145a8f35d4deaaea99479ccf93244080bff52ef969203f4fa721de174"
ARTIFACT_DIGEST: Final = (
    "9f00e17c2f3de700f3a9ff08d41dea312ff61c143d85514b73e0cd84e03cd381"
)
ARRAYS_DIGEST: Final = (
    "98805954d0a60eb109c3f0cebaaaaacdac211f527b93094adf4458b75f3966f3"
)
FEATURE_CONFIG_DIGEST: Final = (
    "fef31940dbb16d854032c883bd4d4c53e3582ddb81fd66b8ac22128f8534251c"
)


class SignalDiagnosisProtocolError(ValueError):
    """The protocol, a payload, a window or a run request is not acceptable."""


class ForbiddenEvaluationWindowError(SignalDiagnosisProtocolError):
    """A window touches the FS1 warm-up span or the FS1 sealed window."""


# ------------------------------------------------------------------ small validators


def _fail(message: str) -> SignalDiagnosisProtocolError:
    return SignalDiagnosisProtocolError(message)


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise _fail(f"{field} must be non-empty text")
    return value


def _exact_int(value: object, field: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise _fail(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise _fail(f"{field} must be at least {minimum}")
    return value


def _exact_float(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise _fail(f"{field} must be a finite float")
    return value


def _exact_bool(value: object, field: str, *, expected: bool) -> bool:
    if type(value) is not bool or value is not expected:
        raise _fail(f"{field} must be {expected}")
    return value


def _tuple(value: object, field: str) -> tuple[Any, ...]:
    if type(value) is not tuple or not value:
        raise _fail(f"{field} must be a non-empty tuple")
    return value


def _unique_texts(value: object, field: str) -> tuple[str, ...]:
    items = _tuple(value, field)
    for item in items:
        _text(item, field)
    if len(set(items)) != len(items):
        raise _fail(f"{field} must not contain duplicates")
    return items


def _unique_ints(value: object, field: str, *, minimum: int) -> tuple[int, ...]:
    items = _tuple(value, field)
    for item in items:
        _exact_int(item, field, minimum=minimum)
    if len(set(items)) != len(items):
        raise _fail(f"{field} must not contain duplicates")
    return items


def _sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise _fail(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _same(value: object, expected: object, field: str) -> None:
    if value != expected:
        raise _fail(f"{field} must be {expected!r}")


# ------------------------------------------------------------------ time handling


def _registry_instant(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def _iso_z(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _aware_utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise _fail(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise _fail(f"{field} must include a timezone")
    return value.astimezone(UTC)


def _parse_z(value: object, field: str) -> datetime:
    text = _text(value, field)
    if not text.endswith("Z"):
        raise _fail(f"{field} must be an ISO-8601 UTC instant ending in Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as error:
        raise _fail(f"{field} must be an ISO-8601 UTC instant ending in Z") from error
    if _iso_z(parsed) != text:
        raise _fail(f"{field} must be written in canonical form {_iso_z(parsed)}")
    return parsed


# The FS1 registry owns the two protected instants. The market-data store pins the
# same values for its own embargo (it may not be imported here); a test asserts the
# equality instead of duplicating the dates.
_FS1 = Fs1Registry()
_WARMUP_START: Final = _registry_instant(_FS1.warmup_only_start)
_SEALED_START: Final = _registry_instant(_FS1.scoring_start)


def reject_forbidden_window(start: datetime, end: datetime) -> None:
    """Refuse a half-open bar-period window ``[start, end)`` that reaches FS1 data.

    The development span ends exactly where the FS1 warm-up span begins, so any end
    after that instant touches the warm-up span, and any end after the sealed start
    touches the sealed window. There is no flag that lifts either refusal.
    """

    begin = _aware_utc(start, "window start")
    finish = _aware_utc(end, "window end")
    if not begin < finish:
        raise _fail("window start must be before its end")
    interval = f"[{_iso_z(begin)}, {_iso_z(finish)})"
    if finish > _SEALED_START:
        raise ForbiddenEvaluationWindowError(
            f"window {interval} touches the FS1 sealed window "
            f"(>= {_iso_z(_SEALED_START)}); refusing unconditionally"
        )
    if finish > _WARMUP_START:
        raise ForbiddenEvaluationWindowError(
            f"window {interval} touches the FS1 warm-up span "
            f"[{_iso_z(_WARMUP_START)}, {_iso_z(_SEALED_START)}); it is never "
            "evaluated by this diagnosis"
        )


# ------------------------------------------------------------------ typed parts


@dataclass(frozen=True, slots=True)
class Hypothesis:
    hypothesis_id: str
    statement: str

    def __post_init__(self) -> None:
        _text(self.hypothesis_id, "hypothesis_id")
        _text(self.statement, "hypothesis statement")


@dataclass(frozen=True, slots=True)
class FeatureFamily:
    """One preregistered family: kinds by definition, features listed explicitly."""

    family_id: str
    kinds: tuple[str, ...]
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.family_id, "family_id")
        _unique_texts(self.kinds, f"family {self.family_id} kinds")
        names = _unique_texts(self.feature_names, f"family {self.family_id} features")
        for name in names:
            if name.split("__", 1)[0] not in ("1h", "4h", "1d"):
                raise _fail(f"feature {name} has no 1h/4h/1d timeframe prefix")


@dataclass(frozen=True, slots=True)
class ForbiddenSpan:
    span_id: str
    start: str
    end: str | None
    reason: str

    def __post_init__(self) -> None:
        _text(self.span_id, "span_id")
        _parse_z(self.start, f"span {self.span_id} start")
        if self.end is not None:
            _parse_z(self.end, f"span {self.span_id} end")
        _text(self.reason, f"span {self.span_id} reason")


def _expected_spans() -> tuple[ForbiddenSpan, ForbiddenSpan]:
    return (
        ForbiddenSpan(
            span_id="fs1_warmup_span",
            start=_iso_z(_WARMUP_START),
            end=_iso_z(_SEALED_START),
            reason=(
                "FS1 warm-up span: never scored or read by this diagnosis; fetching "
                "it needs a separate Issue #810 authorization"
            ),
        ),
        ForbiddenSpan(
            span_id="fs1_sealed_window",
            start=_iso_z(_SEALED_START),
            end=None,
            reason=(
                "FS1 sealed window: scored once after 2027-05-02 under a separate "
                "authorization and never visible before"
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class DataScope:
    dataset_id: str
    artifact_digest: str
    arrays_digest: str
    feature_config_digest: str
    symbols: tuple[str, ...]
    base_interval: str
    feature_timeframes: tuple[str, ...]
    n_features: int
    n_global_features: int
    evaluation_start: str
    evaluation_end: str
    window_semantics: str
    labels_must_mature_inside_window: bool
    forbidden_spans: tuple[ForbiddenSpan, ...]
    known_used: bool
    evidence_scope: str
    reference_block: str
    reference_block_role: str

    def __post_init__(self) -> None:
        for name in (
            "dataset_id",
            "artifact_digest",
            "arrays_digest",
            "feature_config_digest",
        ):
            _sha256(getattr(self, name), name)
        _unique_texts(self.symbols, "symbols")
        _text(self.base_interval, "base_interval")
        _unique_texts(self.feature_timeframes, "feature_timeframes")
        _exact_int(self.n_features, "n_features", minimum=1)
        _exact_int(self.n_global_features, "n_global_features", minimum=0)
        begin = _parse_z(self.evaluation_start, "evaluation_start")
        finish = _parse_z(self.evaluation_end, "evaluation_end")
        reject_forbidden_window(begin, finish)
        _same(
            self.window_semantics, "half_open_bar_period_interval", "window_semantics"
        )
        _exact_bool(
            self.labels_must_mature_inside_window,
            "labels_must_mature_inside_window",
            expected=True,
        )
        if self.forbidden_spans != _expected_spans():
            raise _fail(
                "forbidden_spans must be exactly the FS1 warm-up span and sealed window"
            )
        _exact_bool(self.known_used, "known_used", expected=True)
        _same(self.evidence_scope, EVIDENCE_SCOPE, "evidence_scope")
        _text(self.reference_block, "reference_block")
        _same(
            self.reference_block_role,
            "reference_only_not_evidence",
            "reference_block_role",
        )


@dataclass(frozen=True, slots=True)
class HorizonGrid:
    primary_bars: int
    secondary_bars: tuple[int, ...]
    secondary_role: str
    bar_hours: int

    def __post_init__(self) -> None:
        _exact_int(self.primary_bars, "primary_bars", minimum=1)
        _same(self.secondary_role, "reported_not_decisive", "secondary_role")
        secondary = _unique_ints(self.secondary_bars, "secondary_bars", minimum=1)
        if self.primary_bars in secondary:
            raise _fail("primary horizon must not repeat among the secondary horizons")
        _exact_int(self.bar_hours, "bar_hours", minimum=1)


@dataclass(frozen=True, slots=True)
class TargetSpec:
    target_id: str
    definition: str

    def __post_init__(self) -> None:
        _text(self.target_id, "target_id")
        _text(self.definition, "target definition")


@dataclass(frozen=True, slots=True)
class TransformSpec:
    transform_id: str
    standardise: bool
    clip_abs: float
    definition: str

    def __post_init__(self) -> None:
        _text(self.transform_id, "transform_id")
        if type(self.standardise) is not bool:
            raise _fail("standardise must be boolean")
        _exact_float(self.clip_abs, "clip_abs")
        if self.standardise and self.clip_abs <= 0.0:
            raise _fail("a standardising transform needs a positive clip")
        if not self.standardise and self.clip_abs != 0.0:
            raise _fail("the raw transform must not clip")
        _text(self.definition, "transform definition")


@dataclass(frozen=True, slots=True)
class RidgeSpec:
    penalty_grid: tuple[float, ...]
    penalty_scope: str
    inner_cv_scheme: str
    inner_cv_folds: int
    inner_cv_purge: str
    selection_metric: str
    tie_break: str

    def __post_init__(self) -> None:
        grid = _tuple(self.penalty_grid, "penalty_grid")
        for penalty in grid:
            if _exact_float(penalty, "penalty") <= 0.0:
                raise _fail("ridge penalties must be positive")
        if len(set(grid)) != len(grid) or list(grid) != sorted(grid):
            raise _fail("penalty_grid must be strictly increasing")
        _same(self.penalty_scope, "train_window_only", "penalty_scope")
        _same(self.inner_cv_scheme, "forward_chaining_expanding", "inner_cv_scheme")
        _exact_int(self.inner_cv_folds, "inner_cv_folds", minimum=2)
        _same(self.inner_cv_purge, "label_horizon_bars", "inner_cv_purge")
        _same(self.selection_metric, "validation_mse", "selection_metric")
        _same(self.tie_break, "larger_penalty", "tie_break")


@dataclass(frozen=True, slots=True)
class LightGBMSpec:
    objective: str
    n_estimators: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    min_child_samples: int
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    random_state: int
    n_jobs: int
    verbosity: int

    def __post_init__(self) -> None:
        _text(self.objective, "objective")
        for name in ("n_estimators", "num_leaves", "max_depth", "min_child_samples"):
            _exact_int(getattr(self, name), name, minimum=1)
        for name in ("learning_rate", "subsample", "colsample_bytree", "reg_lambda"):
            if _exact_float(getattr(self, name), name) <= 0.0:
                raise _fail(f"{name} must be positive")
        _exact_int(self.random_state, "random_state", minimum=0)
        _exact_int(self.n_jobs, "n_jobs", minimum=1)
        if type(self.verbosity) is not int:
            raise _fail("verbosity must be an integer")


@dataclass(frozen=True, slots=True)
class ModelGrid:
    model_ids: tuple[str, ...]
    ridge: RidgeSpec
    lightgbm: LightGBMSpec

    def __post_init__(self) -> None:
        _unique_texts(self.model_ids, "model_ids")
        if not isinstance(self.ridge, RidgeSpec) or not isinstance(
            self.lightgbm, LightGBMSpec
        ):
            raise _fail("models must carry a RidgeSpec and a LightGBMSpec")


@dataclass(frozen=True, slots=True)
class PrequentialSpec:
    refit_scheme: str
    refit_interval_days: int
    activation_delay_hours: int
    window: str
    first_fit_time: str

    def __post_init__(self) -> None:
        _text(self.refit_scheme, "refit_scheme")
        _exact_int(self.refit_interval_days, "refit_interval_days", minimum=1)
        _exact_int(self.activation_delay_hours, "activation_delay_hours", minimum=0)
        _text(self.window, "prequential window")
        _parse_z(self.first_fit_time, "first_fit_time")


@dataclass(frozen=True, slots=True)
class CostSpec:
    cost_model: str
    fee_bps_per_side: int
    spread_bps_per_side: int
    round_trip_cost_bps: int
    k_grid: tuple[int, ...]
    cost_multipliers: tuple[int, ...]
    entry_rule: str

    def __post_init__(self) -> None:
        _text(self.cost_model, "cost_model")
        _exact_int(self.fee_bps_per_side, "fee_bps_per_side", minimum=0)
        _exact_int(self.spread_bps_per_side, "spread_bps_per_side", minimum=0)
        _exact_int(self.round_trip_cost_bps, "round_trip_cost_bps", minimum=1)
        if self.round_trip_cost_bps != 2 * (
            self.fee_bps_per_side + self.spread_bps_per_side
        ):
            raise _fail("round_trip_cost_bps must be twice fee plus spread per side")
        _unique_ints(self.k_grid, "k_grid", minimum=1)
        _unique_ints(self.cost_multipliers, "cost_multipliers", minimum=1)
        _text(self.entry_rule, "entry_rule")


@dataclass(frozen=True, slots=True)
class BootstrapSpec:
    method: str
    unit: str
    block_days: int
    resamples: int
    seed: int
    confidence_level: float

    def __post_init__(self) -> None:
        _text(self.method, "bootstrap method")
        _text(self.unit, "bootstrap unit")
        _exact_int(self.block_days, "block_days", minimum=1)
        _exact_int(self.resamples, "resamples", minimum=1)
        _exact_int(self.seed, "seed", minimum=0)
        if not 0.0 < _exact_float(self.confidence_level, "confidence_level") < 1.0:
            raise _fail("confidence_level must lie strictly between 0 and 1")


@dataclass(frozen=True, slots=True)
class MultipleTestingSpec:
    reported: tuple[str, ...]
    holm_fwer_alpha: float
    bh_fdr_q: float
    decision_procedure: str
    decision_test_family: str
    reported_test_families: tuple[str, ...]

    def __post_init__(self) -> None:
        reported = _unique_texts(self.reported, "reported corrections")
        for name in ("holm_fwer_alpha", "bh_fdr_q"):
            if not 0.0 < _exact_float(getattr(self, name), name) < 1.0:
                raise _fail(f"{name} must lie strictly between 0 and 1")
        if self.decision_procedure not in reported:
            raise _fail("the decision procedure must be among the reported corrections")
        families = _unique_texts(self.reported_test_families, "reported_test_families")
        if self.decision_test_family not in families:
            raise _fail("the decision test family must be among the reported families")


TAIL_ONE_SIDED_POSITIVE: Final = "one_sided_positive"


@dataclass(frozen=True, slots=True)
class SignificanceSpec:
    """How the decisive rank-IC significance p-values are produced and corrected.

    The gate "BH-adjusted p <= q" is only a protocol if the p-value itself is fixed:
    one test, its null and tail, the handling of overlapping labels, and the exact
    family the correction runs over. A verdict must accept p-values only from this
    procedure.
    """

    procedure_id: str
    statistic: str
    null_hypothesis: str
    tail: str
    p_value_source: str
    dependence_rule: str
    family_rule: str
    bh_family_ids: tuple[str, ...]
    bh_family_cell_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "procedure_id",
            "statistic",
            "null_hypothesis",
            "p_value_source",
            "dependence_rule",
            "family_rule",
        ):
            _text(getattr(self, name), name)
        _same(self.tail, TAIL_ONE_SIDED_POSITIVE, "tail")
        families = _unique_texts(self.bh_family_ids, "bh_family_ids")
        if len(families) != 2:
            raise _fail("exactly two BH families are tested: raw and demeaned target")
        _unique_texts(self.bh_family_cell_ids, "bh_family_cell_ids")


@dataclass(frozen=True, slots=True)
class StabilitySpec:
    full_years: tuple[int, ...]
    partial_years_reported_only: tuple[int, ...]
    year_basis: str
    regime_lookback_days: int
    regime_definition: str
    extreme_day_trim_fraction_per_tail: float
    extreme_day_definition: str
    reported_slices: tuple[str, ...]
    decisive_slices: tuple[str, ...]
    year_sign_statistic: str
    min_covered_months_per_full_year: int
    covered_month_min_valid_days: int
    coverage_rule: str

    def __post_init__(self) -> None:
        years = _unique_ints(self.full_years, "full_years", minimum=1)
        reported = _unique_texts(self.reported_slices, "reported_slices")
        decisive = _unique_texts(self.decisive_slices, "decisive_slices")
        if not set(decisive) <= set(reported):
            raise _fail("decisive slices must be among the reported slices")
        partial = _unique_ints(
            self.partial_years_reported_only, "partial_years_reported_only", minimum=1
        )
        if set(years) & set(partial):
            raise _fail("a year cannot be both full and partial")
        _text(self.year_basis, "year_basis")
        _exact_int(self.regime_lookback_days, "regime_lookback_days", minimum=1)
        _text(self.regime_definition, "regime_definition")
        if (
            not 0.0
            < _exact_float(
                self.extreme_day_trim_fraction_per_tail,
                "extreme_day_trim_fraction_per_tail",
            )
            < 0.5
        ):
            raise _fail(
                "the per-tail trim fraction must lie strictly between 0 and 0.5"
            )
        _text(self.extreme_day_definition, "extreme_day_definition")
        _text(self.year_sign_statistic, "year_sign_statistic")
        months = _exact_int(
            self.min_covered_months_per_full_year,
            "min_covered_months_per_full_year",
            minimum=1,
        )
        if months > 12:
            raise _fail("min_covered_months_per_full_year cannot exceed 12")
        days = _exact_int(
            self.covered_month_min_valid_days, "covered_month_min_valid_days", minimum=1
        )
        if days > 28:
            raise _fail("covered_month_min_valid_days cannot exceed 28")
        _text(self.coverage_rule, "coverage_rule")


@dataclass(frozen=True, slots=True)
class MeasurementSpec:
    """How the daily IC statistics are formed (the plan says only "computed daily")."""

    rank_ic_estimator: str
    linear_ic_estimator: str
    day_boundary: str
    row_inclusion: str
    skipped_days: str
    increment_definition: str

    def __post_init__(self) -> None:
        for name in (
            "rank_ic_estimator",
            "linear_ic_estimator",
            "day_boundary",
            "row_inclusion",
            "skipped_days",
            "increment_definition",
        ):
            _text(getattr(self, name), name)


BASELINE_ROLES: Final = ("decisive_comparator", "reported_context")


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    """One baseline with its causal estimation rule and whether it can decide."""

    baseline_id: str
    role: str
    definition: str

    def __post_init__(self) -> None:
        _text(self.baseline_id, "baseline_id")
        if self.role not in BASELINE_ROLES:
            raise _fail(f"baseline role must be one of {BASELINE_ROLES}")
        _text(self.definition, f"baseline {self.baseline_id} definition")


@dataclass(frozen=True, slots=True)
class LayerSpec:
    layer_id: str
    requires_fit: bool
    cards: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.layer_id, "layer_id")
        if type(self.requires_fit) is not bool:
            raise _fail("requires_fit must be boolean")
        _unique_texts(self.cards, f"layer {self.layer_id} cards")


@dataclass(frozen=True, slots=True)
class DecisionRules:
    """Thresholds of the section-5 table, fixed before any result exists."""

    cell_unit: str
    primary_horizon_bars: int
    primary_transform_id: str
    signal_target_id: str
    persistence_target_id: str
    tradability_target_id: str
    primary_k: int
    min_full_years_same_sign: int
    candidate_ci_lower_floor: float
    candidate_stress_point_floor: float
    tradability_cost_multiplier: int
    stress_cost_multiplier: int
    candidate_ranking: str
    unassigned_corner_resolutions: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.cell_unit, "cell_unit")
        _exact_int(self.primary_horizon_bars, "primary_horizon_bars", minimum=1)
        for name in (
            "primary_transform_id",
            "signal_target_id",
            "persistence_target_id",
            "tradability_target_id",
        ):
            _text(getattr(self, name), name)
        _exact_int(self.primary_k, "primary_k", minimum=1)
        _exact_int(self.min_full_years_same_sign, "min_full_years_same_sign", minimum=1)
        for name in ("candidate_ci_lower_floor", "candidate_stress_point_floor"):
            _exact_float(getattr(self, name), name)
        _exact_int(
            self.tradability_cost_multiplier, "tradability_cost_multiplier", minimum=1
        )
        _exact_int(self.stress_cost_multiplier, "stress_cost_multiplier", minimum=1)
        _text(self.candidate_ranking, "candidate_ranking")
        _unique_texts(
            self.unassigned_corner_resolutions, "unassigned_corner_resolutions"
        )


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    single_run: bool
    save_all_cells: bool
    post_result_additions_allowed: bool
    retune_prior_learner_on_this_data: bool
    max_candidates: int
    native_replay_verification_cap: int

    def __post_init__(self) -> None:
        _exact_bool(self.single_run, "single_run", expected=True)
        _exact_bool(self.save_all_cells, "save_all_cells", expected=True)
        _exact_bool(
            self.post_result_additions_allowed,
            "post_result_additions_allowed",
            expected=False,
        )
        _exact_bool(
            self.retune_prior_learner_on_this_data,
            "retune_prior_learner_on_this_data",
            expected=False,
        )
        _exact_int(self.max_candidates, "max_candidates", minimum=1)
        _exact_int(
            self.native_replay_verification_cap,
            "native_replay_verification_cap",
            minimum=1,
        )


@dataclass(frozen=True, slots=True)
class SignalDiagnosisProtocol:
    """Every field is required; the frozen reference of the version is the contract."""

    schema_version: str
    issue: int
    parent_issue: int
    follows_issue: int
    card: str
    execution_status: str
    claim_ceiling: str
    forbidden_claims: tuple[str, ...]
    hypotheses: tuple[Hypothesis, ...]
    data: DataScope
    layers: tuple[LayerSpec, ...]
    stop_when_upper_layer_stops: bool
    families: tuple[FeatureFamily, ...]
    horizons: HorizonGrid
    measurement: MeasurementSpec
    targets: tuple[TargetSpec, ...]
    transforms: tuple[TransformSpec, ...]
    models: ModelGrid
    prequential: PrequentialSpec
    costs: CostSpec
    bootstrap: BootstrapSpec
    multiple_testing: MultipleTestingSpec
    significance: SignificanceSpec
    baselines: tuple[BaselineSpec, ...]
    stability: StabilitySpec
    decision_rules: DecisionRules
    verdict_labels: tuple[str, ...]
    budget: BudgetSpec

    def __post_init__(self) -> None:
        self._require_types()
        _text(self.schema_version, "schema_version")
        for name in ("issue", "parent_issue", "follows_issue"):
            _exact_int(getattr(self, name), name, minimum=1)
        _text(self.card, "card")
        _same(self.execution_status, EXECUTION_STATUS, "execution_status")
        _same(self.claim_ceiling, CLAIM_CEILING, "claim_ceiling")
        _unique_texts(self.forbidden_claims, "forbidden_claims")
        _tuple(self.hypotheses, "hypotheses")
        _tuple(self.layers, "layers")
        _exact_bool(
            self.stop_when_upper_layer_stops,
            "stop_when_upper_layer_stops",
            expected=True,
        )
        self._validate_families()
        _tuple(self.targets, "targets")
        _tuple(self.transforms, "transforms")
        _unique_texts(self.verdict_labels, "verdict_labels")
        self._validate_cross_references()
        self._require_frozen_reference()

    def _require_types(self) -> None:
        for name, expected in (
            ("data", DataScope),
            ("horizons", HorizonGrid),
            ("measurement", MeasurementSpec),
            ("models", ModelGrid),
            ("prequential", PrequentialSpec),
            ("costs", CostSpec),
            ("bootstrap", BootstrapSpec),
            ("multiple_testing", MultipleTestingSpec),
            ("significance", SignificanceSpec),
            ("stability", StabilitySpec),
            ("decision_rules", DecisionRules),
            ("budget", BudgetSpec),
        ):
            if not isinstance(getattr(self, name), expected):
                raise _fail(f"{name} must be a {expected.__name__}")
        for name, element in (
            ("hypotheses", Hypothesis),
            ("layers", LayerSpec),
            ("families", FeatureFamily),
            ("targets", TargetSpec),
            ("transforms", TransformSpec),
            ("baselines", BaselineSpec),
        ):
            items = _tuple(getattr(self, name), name)
            if any(not isinstance(item, element) for item in items):
                raise _fail(f"{name} must contain only {element.__name__} items")

    def _validate_families(self) -> None:
        families = _tuple(self.families, "families")
        ids = [family.family_id for family in families]
        if len(set(ids)) != len(ids):
            raise _fail("family ids must be unique")
        names = [name for family in families for name in family.feature_names]
        if len(set(names)) != len(names):
            raise _fail("a feature may belong to only one family")
        if len(names) != self.data.n_features:
            raise _fail("the families must cover exactly the dataset's feature count")
        prefixes = {self.data.base_interval, *self.data.feature_timeframes}
        for name in names:
            if name.split("__", 1)[0] not in prefixes:
                raise _fail(f"feature {name} uses a timeframe outside the dataset")

    def _validate_cross_references(self) -> None:
        rules = self.decision_rules
        target_ids = {target.target_id for target in self.targets}
        transform_ids = {transform.transform_id for transform in self.transforms}
        if len(target_ids) != len(self.targets) or len(transform_ids) != len(
            self.transforms
        ):
            raise _fail("target and transform ids must be unique")
        for name in (
            "signal_target_id",
            "persistence_target_id",
            "tradability_target_id",
        ):
            if getattr(rules, name) not in target_ids:
                raise _fail(f"decision_rules.{name} is not a declared target")
        if rules.primary_transform_id not in transform_ids:
            raise _fail(
                "decision_rules.primary_transform_id is not a declared transform"
            )
        horizons = {self.horizons.primary_bars, *self.horizons.secondary_bars}
        if rules.primary_horizon_bars != self.horizons.primary_bars:
            raise _fail("the decision horizon must be the primary horizon")
        if rules.primary_horizon_bars not in horizons:
            raise _fail("the decision horizon is not a declared horizon")
        if rules.primary_k not in self.costs.k_grid:
            raise _fail("decision_rules.primary_k is not in the k grid")
        for name in ("tradability_cost_multiplier", "stress_cost_multiplier"):
            if getattr(rules, name) not in self.costs.cost_multipliers:
                raise _fail(f"decision_rules.{name} is not a declared cost multiplier")
        if rules.min_full_years_same_sign > len(self.stability.full_years):
            raise _fail("the year threshold exceeds the number of full years")
        self._validate_significance()
        baseline_ids = [baseline.baseline_id for baseline in self.baselines]
        if len(set(baseline_ids)) != len(baseline_ids):
            raise _fail("baseline ids must be unique")
        self._validate_stability_years()
        if self.budget.native_replay_verification_cap != self.budget.max_candidates:
            raise _fail("the replay cap must equal the candidate cap")
        if tuple(self.verdict_labels) != VERDICT_LABELS:
            raise _fail("verdict_labels must be the four preregistered labels")

    def _validate_significance(self) -> None:
        """The BH family is the decision cells; the bootstrap block covers label overlap."""

        expected = tuple(
            f"{family.family_id}:{model_id}"
            for family in self.families
            for model_id in self.models.model_ids
        )
        if self.significance.bh_family_cell_ids != expected:
            raise _fail(
                "the BH family cell list must equal the decision cells "
                "(every family by every model, in protocol order)"
            )
        overlap_days = math.ceil(
            self.horizons.primary_bars * self.horizons.bar_hours / 24
        )
        if self.bootstrap.block_days < overlap_days:
            raise _fail(
                f"bootstrap.block_days {self.bootstrap.block_days} must cover the "
                f"label overlap of {overlap_days} day(s)"
            )

    def _validate_stability_years(self) -> None:
        """A full year must be fully out of sample; the years must tile the OOS span.

        Model decisions exist only from ``first_fit_time`` (the shared
        ``PrequentialSchedule`` rejects earlier ones), so a calendar year that starts
        before it cannot be a *full* stability year however its aggregate sign count
        looks. The full and partial years together must be exactly the calendar years
        the out-of-sample span touches.
        """

        stability = self.stability
        first_fit = _parse_z(self.prequential.first_fit_time, "first_fit_time")
        window_start = _parse_z(self.data.evaluation_start, "evaluation_start")
        window_end = _parse_z(self.data.evaluation_end, "evaluation_end")
        oos_start = max(first_fit, window_start)
        if not window_start <= first_fit < window_end:
            raise _fail("first_fit_time must lie inside the evaluation window")
        for year in stability.full_years:
            try:
                begin = datetime(year, 1, 1, tzinfo=UTC)
                finish = datetime(year + 1, 1, 1, tzinfo=UTC)
            except ValueError as error:
                raise _fail(f"full year {year} is not a calendar year") from error
            if begin < oos_start:
                raise _fail(
                    f"full year {year} starts before the first usable fit "
                    f"{_iso_z(oos_start)}; it can only be a partial year"
                )
            if finish > window_end:
                raise _fail(
                    f"full year {year} ends after the evaluation window "
                    f"{_iso_z(window_end)}; it can only be a partial year"
                )
        touched = set(
            range(oos_start.year, (window_end - timedelta(microseconds=1)).year + 1)
        )
        declared = set(stability.full_years) | set(
            stability.partial_years_reported_only
        )
        if declared != touched:
            raise _fail(
                "full and partial years must partition the out-of-sample calendar "
                f"years {sorted(touched)}, got {sorted(declared)}"
            )

    def _require_frozen_reference(self) -> None:
        reference = _FROZEN_REFERENCES.get(self.schema_version)
        if reference is None:
            raise _fail(f"unsupported protocol version {self.schema_version!r}")
        if to_json_value(self) != reference():
            raise _fail(
                f"payload differs from the frozen {self.schema_version} reference; "
                "any change requires a new protocol version"
            )


# ------------------------------------------------------------------ frozen reference


_FAMILY_KINDS: Final = {
    "return_momentum": ("log_return", "body_return", "gap_return"),
    "oscillator": (
        "rsi",
        "stochastic_k",
        "stochastic_d",
        "cci",
        "williams_r",
        "mfi",
        "bollinger_position",
    ),
    "trend": (
        "macd_line",
        "macd_signal",
        "macd_histogram",
        "adx",
        "plus_di",
        "minus_di",
        "di_spread",
        "ema_distance",
        "ema_slope",
        "linear_regression_slope",
        "trend_r2",
        "ichimoku_tenkan_distance",
        "ichimoku_kijun_distance",
        "ichimoku_cloud_position",
        "ichimoku_cloud_thickness",
    ),
    "volatility_range": (
        "realized_volatility",
        "atr_pct",
        "atr_change",
        "bollinger_bandwidth",
        "high_low_range",
        "parkinson_volatility",
        "garman_klass_volatility",
        "downside_volatility",
        "upside_volatility",
        "volatility_of_volatility",
        "range_expansion",
    ),
    "candle_shape": ("upper_wick_ratio", "lower_wick_ratio", "close_location_value"),
    "volume_flow": (
        "volume_zscore",
        "volume_log_change",
        "relative_volume",
        "obv_slope",
        "obv_change",
        "obv_acceleration",
        "cmf",
        "vwap_distance",
        "price_volume_correlation",
    ),
    "funding": ("funding_bps", "funding_change", "funding_zscore"),
    "cross_asset": (
        "relative_return_to_btc",
        "rolling_correlation_to_btc",
        "rolling_beta_to_btc",
        "cross_sectional_momentum_rank",
        "cross_asset_dispersion",
    ),
}

# The 167 maintained feature names (1h base clock plus the 4h and 1d clocks), listed
# explicitly and assigned to exactly one family by feature kind. The list follows the
# maintained feature definitions in code, not any dataset value.
_FAMILY_FEATURES: Final = {
    "return_momentum": (
        "1h__log_return_1bar",
        "1h__log_return_4bar",
        "1h__log_return_24bar",
        "1h__body_return",
        "1h__gap_return",
        "4h__log_return_1bar",
        "4h__log_return_4bar",
        "4h__log_return_24bar",
        "4h__body_return",
        "4h__gap_return",
        "1d__log_return_1bar",
        "1d__log_return_4bar",
        "1d__log_return_24bar",
        "1d__body_return",
    ),
    "oscillator": (
        "1h__rsi_14bar",
        "1h__bollinger_percent_b_centered_20_2",
        "1h__stochastic_k_14bar",
        "1h__stochastic_d_14_3",
        "1h__cci_20bar",
        "1h__williams_r_14bar",
        "1h__mfi_24bar",
        "4h__rsi_14bar",
        "4h__bollinger_percent_b_centered_20_2",
        "4h__stochastic_k_14bar",
        "4h__stochastic_d_14_3",
        "4h__cci_20bar",
        "4h__williams_r_14bar",
        "4h__mfi_18bar",
        "1d__rsi_14bar",
        "1d__bollinger_percent_b_centered_20_2",
        "1d__stochastic_k_14bar",
        "1d__stochastic_d_14_3",
        "1d__cci_20bar",
        "1d__williams_r_14bar",
        "1d__mfi_20bar",
    ),
    "trend": (
        "1h__macd_line_12_26",
        "1h__macd_signal_12_26_9",
        "1h__macd_histogram_12_26_9",
        "1h__adx_14bar",
        "1h__ichimoku_tenkan_distance_9bar",
        "1h__ichimoku_kijun_distance_26bar",
        "1h__ichimoku_cloud_position_9_26_52",
        "1h__ichimoku_cloud_thickness_9_26_52",
        "1h__plus_di_14bar",
        "1h__minus_di_14bar",
        "1h__di_spread_14bar",
        "1h__ema_distance_13_26",
        "1h__ema_slope_26",
        "1h__linear_regression_slope_24bar",
        "1h__trend_r2_24bar",
        "4h__macd_line_12_26",
        "4h__macd_signal_12_26_9",
        "4h__macd_histogram_12_26_9",
        "4h__adx_14bar",
        "4h__ichimoku_tenkan_distance_9bar",
        "4h__ichimoku_kijun_distance_26bar",
        "4h__ichimoku_cloud_position_9_26_52",
        "4h__ichimoku_cloud_thickness_9_26_52",
        "4h__plus_di_14bar",
        "4h__minus_di_14bar",
        "4h__di_spread_14bar",
        "4h__ema_distance_13_26",
        "4h__ema_slope_26",
        "4h__linear_regression_slope_18bar",
        "4h__trend_r2_18bar",
        "1d__macd_line_12_26",
        "1d__macd_signal_12_26_9",
        "1d__macd_histogram_12_26_9",
        "1d__adx_14bar",
        "1d__ichimoku_tenkan_distance_9bar",
        "1d__ichimoku_kijun_distance_26bar",
        "1d__ichimoku_cloud_position_9_26_52",
        "1d__ichimoku_cloud_thickness_9_26_52",
        "1d__plus_di_14bar",
        "1d__minus_di_14bar",
        "1d__di_spread_14bar",
        "1d__ema_distance_13_26",
        "1d__ema_slope_26",
        "1d__linear_regression_slope_20bar",
        "1d__trend_r2_20bar",
    ),
    "volatility_range": (
        "1h__realized_volatility_4bar",
        "1h__realized_volatility_24bar",
        "1h__bollinger_bandwidth_20_2",
        "1h__atr_pct_14bar",
        "1h__high_low_range_pct",
        "1h__parkinson_volatility_24bar",
        "1h__garman_klass_volatility_24bar",
        "1h__downside_volatility_24bar",
        "1h__upside_volatility_24bar",
        "1h__volatility_of_volatility_24bar",
        "1h__range_expansion_24bar",
        "1h__atr_change_14bar",
        "4h__realized_volatility_4bar",
        "4h__realized_volatility_24bar",
        "4h__bollinger_bandwidth_20_2",
        "4h__atr_pct_14bar",
        "4h__high_low_range_pct",
        "4h__parkinson_volatility_18bar",
        "4h__garman_klass_volatility_18bar",
        "4h__downside_volatility_18bar",
        "4h__upside_volatility_18bar",
        "4h__volatility_of_volatility_18bar",
        "4h__range_expansion_18bar",
        "4h__atr_change_14bar",
        "1d__realized_volatility_4bar",
        "1d__realized_volatility_24bar",
        "1d__bollinger_bandwidth_20_2",
        "1d__atr_pct_14bar",
        "1d__high_low_range_pct",
        "1d__parkinson_volatility_20bar",
        "1d__garman_klass_volatility_20bar",
        "1d__downside_volatility_20bar",
        "1d__upside_volatility_20bar",
        "1d__volatility_of_volatility_20bar",
        "1d__range_expansion_20bar",
        "1d__atr_change_14bar",
    ),
    "candle_shape": (
        "1h__upper_wick_ratio",
        "1h__lower_wick_ratio",
        "1h__close_location_value",
        "4h__close_location_value",
        "1d__close_location_value",
    ),
    "volume_flow": (
        "1h__volume_zscore_24bar",
        "1h__obv_slope_24bar",
        "1h__volume_log_change",
        "1h__cmf_24bar",
        "1h__vwap_distance_24bar",
        "1h__price_volume_correlation_24bar",
        "1h__obv_change_24bar",
        "1h__obv_acceleration_24bar",
        "1h__relative_volume_24bar",
        "4h__volume_zscore_24bar",
        "4h__obv_slope_24bar",
        "4h__volume_log_change",
        "4h__cmf_18bar",
        "4h__vwap_distance_18bar",
        "4h__price_volume_correlation_18bar",
        "4h__obv_change_18bar",
        "1d__volume_zscore_24bar",
        "1d__obv_slope_24bar",
        "1d__volume_log_change",
        "1d__cmf_20bar",
        "1d__vwap_distance_20bar",
        "1d__price_volume_correlation_20bar",
    ),
    "funding": (
        "1h__funding_bps",
        "1h__funding_change_bps",
        "1h__funding_zscore_12events",
        "4h__funding_bps",
        "4h__funding_change_bps",
        "4h__funding_zscore_12events",
        "1d__funding_bps",
        "1d__funding_change_bps",
        "1d__funding_zscore_12events",
    ),
    "cross_asset": (
        "1h__relative_return_to_btc_1bar",
        "1h__rolling_correlation_to_btc_24bar",
        "1h__rolling_beta_to_btc_24bar",
        "1h__cross_sectional_momentum_rank_24bar",
        "1h__cross_asset_dispersion_1bar",
        "4h__relative_return_to_btc_1bar",
        "4h__rolling_correlation_to_btc_24bar",
        "4h__rolling_beta_to_btc_24bar",
        "4h__cross_sectional_momentum_rank_24bar",
        "4h__cross_asset_dispersion_1bar",
        "1d__relative_return_to_btc_1bar",
        "1d__rolling_correlation_to_btc_24bar",
        "1d__rolling_beta_to_btc_24bar",
        "1d__cross_sectional_momentum_rank_24bar",
        "1d__cross_asset_dispersion_1bar",
    ),
}

_HYPOTHESES: Final = (
    (
        "H1",
        "At least one feature family has out-of-sample predictive power above the "
        "drift baseline that remains after the same-time cross-sectional mean return "
        "(the common level) is removed; heterogeneous market beta is not removed.",
    ),
    (
        "H0a",
        "Predictive power is indistinguishable from the drift baseline, so the #1025 "
        "outcome is explained on the data side.",
    ),
    (
        "H0b",
        "Apparent predictive power is explained by the common cross-sectional level "
        "or market direction, a few extreme days, concentration in particular years "
        "or symbols, or feature non-stationarity such as a funding_bps drift.",
    ),
    (
        "H0c",
        "A statistical signal exists but does not exceed round-trip cost.",
    ),
)

_CLAIM_FORBIDDEN: Final = (
    "edge",
    "profitability",
    "g4_economic_evidence",
    "g5_eligibility",
    "unused_data_validity",
    "final_evaluation",
    "live_trading",
    "production_eligibility",
    "beta_robust_signal",
    "beta_adjusted_alpha",
)

_LAYERS: Final = (
    ("D0", False, ("SD2",)),
    ("D1", False, ("SD2",)),
    ("D2", False, ("SD2",)),
    ("D3", True, ("SD3",)),
    ("D4", True, ("SD3",)),
    ("D5", False, ("SD2", "SD3")),
)

_TARGETS: Final = (
    (
        "raw_forward_log_return",
        "r(i,t,h) = ln(close(i,t+h) / close(i,t)) from the decision-time bar close at t; "
        "the label is usable only once t+h has matured and rows without a matured "
        "label are dropped, never imputed (the construction of "
        "strategies/forecasts/supervised.py).",
    ),
    (
        "cross_sectionally_demeaned_forward_log_return",
        "r(i,t,h) minus the mean of r(j,t,h) over the symbols j that have a matured "
        "label at t; no regression coefficient is estimated. This removes only the "
        "same-time cross-sectional mean (the common level); it does not remove "
        "heterogeneous market beta, so no result on it is called beta-adjusted.",
    ),
    (
        "cost_adjusted_sign_payoff",
        "Net payoff of a one-unit position s in {-1, 0, +1} held for h bars: "
        "s * r(i,t,h) - |s| * round_trip_cost. D2 uses s = sign of the standardised "
        "feature; D3/D4 use s = sign of the forecast, taken only when |forecast| > "
        "k * round_trip_cost. Cost multipliers 1 and 2 are both computed.",
    ),
    (
        "drift_excess_payoff",
        "For a decision row and a cost multiplier m, the cost_adjusted_sign_payoff of "
        "the signal policy minus the payoff of the drift_sign_policy_zero_cost "
        "comparator on the same row at the same cost multiplier. It is averaged over "
        "the rows of each UTC day and the 20-day moving-block bootstrap interval is "
        "taken on that daily series. It is a payoff relative to a drift-following "
        "policy, not a beta-adjusted alpha.",
    ),
)

_BASELINES: Final = (
    (
        "zero_forecast",
        "reported_context",
        "The forecast is exactly 0 for every row; used for the MSE-skill context "
        "only and never as a decisive comparator.",
    ),
    (
        "drift_expanding_mean_monthly",
        "decisive_comparator",
        "Per symbol, the mean of all h-bar forward log returns whose label maturity "
        "time is at or before the refit instant, over the same expanding window as "
        "the model. It is recomputed at each refit instant (every refit_interval_days "
        "from first_fit_time) and held constant until the next one; a row whose label "
        "has not yet matured at the refit instant is not used and no label of a row "
        "at or after the refit instant enters.",
    ),
    (
        "drift_sign_policy_zero_cost",
        "decisive_comparator",
        "Position s = sign of the symbol's drift_expanding_mean_monthly forecast "
        "(0 if it is exactly 0), taken on every eligible decision row and charged no "
        "cost: a deliberately favourable comparator that approximates holding the "
        "symbol. It is used only to define drift_excess_payoff.",
    ),
    (
        "beta_regression_on_market_mean",
        "reported_context",
        "Context only; it never enters the verdict. Per symbol, the least-squares "
        "slope (with intercept) of matured h-bar log returns on the matured "
        "cross-sectional mean h-bar log return, over the same expanding window and "
        "refit instants as the drift baseline, using only rows whose label matured "
        "at or before the refit instant; a symbol with fewer than 720 matured rows "
        "has no estimate and is reported as missing.",
    ),
)

_TRANSFORMS: Final = (
    (
        "raw",
        False,
        0.0,
        "Feature value as stored in the dataset; no fit and no clip.",
    ),
    (
        "train_window_standardise_clip3",
        True,
        3.0,
        "Per (symbol, feature) column: subtract the train-window mean, divide by the "
        "train-window standard deviation, clip to [-3, 3]; the statistics use only "
        "rows inside the declared train window (the #1025 TD8 rule).",
    ),
)


def _reference_payload() -> dict[str, object]:
    """The frozen v3 payload, built from literals and the FS1 registry instants."""

    warmup_start = _iso_z(_WARMUP_START)
    expected_spans = _expected_spans()
    families = [
        {
            "family_id": family_id,
            "kinds": list(_FAMILY_KINDS[family_id]),
            "feature_names": list(_FAMILY_FEATURES[family_id]),
        }
        for family_id in _FAMILY_KINDS
    ]
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "issue": 1146,
        "parent_issue": 810,
        "follows_issue": 1025,
        "card": "SD0",
        "execution_status": EXECUTION_STATUS,
        "claim_ceiling": CLAIM_CEILING,
        "forbidden_claims": list(_CLAIM_FORBIDDEN),
        "hypotheses": [
            {"hypothesis_id": hid, "statement": text} for hid, text in _HYPOTHESES
        ],
        "data": {
            "dataset_id": DATASET_ID,
            "artifact_digest": ARTIFACT_DIGEST,
            "arrays_digest": ARRAYS_DIGEST,
            "feature_config_digest": FEATURE_CONFIG_DIGEST,
            "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"],
            "base_interval": "1h",
            "feature_timeframes": ["4h", "1d"],
            "n_features": 167,
            "n_global_features": 4,
            "evaluation_start": "2021-01-01T00:00:00Z",
            "evaluation_end": warmup_start,
            "window_semantics": "half_open_bar_period_interval",
            "labels_must_mature_inside_window": True,
            "forbidden_spans": [to_json_value(span) for span in expected_spans],
            "known_used": True,
            "evidence_scope": EVIDENCE_SCOPE,
            "reference_block": "bot-development-2024q1",
            "reference_block_role": "reference_only_not_evidence",
        },
        "layers": [
            {"layer_id": layer_id, "requires_fit": fit, "cards": list(cards)}
            for layer_id, fit, cards in _LAYERS
        ],
        "stop_when_upper_layer_stops": True,
        "families": families,
        "horizons": {
            "primary_bars": 24,
            "secondary_bars": [4, 72],
            "secondary_role": "reported_not_decisive",
            "bar_hours": 1,
        },
        "measurement": {
            "rank_ic_estimator": "spearman_on_the_pooled_rows_of_one_utc_day",
            "linear_ic_estimator": "pearson_on_the_pooled_rows_of_one_utc_day",
            "day_boundary": "00:00 UTC of the decision-time timestamp",
            "row_inclusion": "hourly rows of all five symbols whose features are "
            "available at the decision time and whose label has matured inside the "
            "evaluation window, on the bar-close-time basis: for decision close time "
            "t a row is eligible when evaluation_start < t and t + h bars <= "
            "evaluation_end (a label closing exactly at evaluation_end is inside, a "
            "later one is dropped); nothing is imputed",
            "skipped_days": "a day with fewer than two eligible rows or a constant "
            "score or target has no IC; such days are skipped and counted in the "
            "report, never filled",
            "increment_definition": "daily rank IC of the model forecast minus the "
            "daily rank IC of the drift baseline on the same rows",
        },
        "targets": [
            {"target_id": target_id, "definition": text} for target_id, text in _TARGETS
        ],
        "transforms": [
            {
                "transform_id": transform_id,
                "standardise": standardise,
                "clip_abs": clip,
                "definition": text,
            }
            for transform_id, standardise, clip, text in _TRANSFORMS
        ],
        "models": {
            "model_ids": ["ridge_inner_cv", "lightgbm_fixed"],
            "ridge": {
                "penalty_grid": [0.1, 1.0, 10.0, 100.0, 1000.0],
                "penalty_scope": "train_window_only",
                "inner_cv_scheme": "forward_chaining_expanding",
                "inner_cv_folds": 5,
                "inner_cv_purge": "label_horizon_bars",
                "selection_metric": "validation_mse",
                "tie_break": "larger_penalty",
            },
            "lightgbm": {
                "objective": "regression",
                "n_estimators": 64,
                "learning_rate": 0.05,
                "num_leaves": 7,
                "max_depth": 3,
                "min_child_samples": 20,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 1.0,
                "random_state": 0,
                "n_jobs": 1,
                "verbosity": -1,
            },
        },
        "prequential": {
            "refit_scheme": "monthly_expanding_refit",
            "refit_interval_days": 30,
            "activation_delay_hours": 0,
            "window": "expanding_from_first_dataset_row",
            "first_fit_time": "2021-04-01T00:00:00Z",
        },
        "costs": {
            "cost_model": "fee_plus_spread_both_sides_no_impact_no_funding",
            "fee_bps_per_side": 5,
            "spread_bps_per_side": 2,
            "round_trip_cost_bps": 14,
            "k_grid": [1, 2],
            "cost_multipliers": [1, 2],
            "entry_rule": "take a position only when |forecast| exceeds k times the "
            "round-trip cost fraction (round_trip_cost_bps / 10000 times the cost "
            "multiplier); otherwise stay flat",
        },
        "bootstrap": {
            "method": "moving_block_mean_test",
            "unit": "utc_day_series",
            "block_days": 20,
            "resamples": 2000,
            "seed": 0,
            "confidence_level": 0.95,
        },
        "multiple_testing": {
            "reported": ["holm_fwer", "bh_fdr"],
            "holm_fwer_alpha": 0.05,
            "bh_fdr_q": 0.1,
            "decision_procedure": "bh_fdr",
            "decision_test_family": "decision_cells",
            "reported_test_families": ["decision_cells", "d2_feature_cells"],
        },
        "significance": {
            "procedure_id": "moving_block_bootstrap_one_sided_mean_rank_ic_increment_v1",
            "statistic": "mean over the eligible UTC days of the daily rank-IC "
            "increment (measurement.increment_definition) of one decision cell on one "
            "target (raw or cross-sectionally demeaned) at the primary horizon and "
            "transform",
            "null_hypothesis": "the mean daily rank-IC increment is <= 0 (no "
            "improvement over the drift baseline)",
            "tail": TAIL_ONE_SIDED_POSITIVE,
            "p_value_source": "the p_value of moving_block_mean_test applied to the "
            "daily increment series with block_size = bootstrap.block_days, "
            "n_bootstrap = bootstrap.resamples and seed = bootstrap.seed: one-sided, "
            "p = (number of centred bootstrap means >= the observed mean + 1) / "
            "(resamples + 1) when the observed mean is positive, and exactly 1.0 when "
            "it is not or when fewer than two days are eligible",
            "dependence_rule": "daily increments are dependent through overlapping "
            "labels; label_overlap_days = ceil(primary_horizon_bars * bar_hours / 24), "
            "which is 1 for 24 one-hour bars, and bootstrap.block_days must be at "
            "least label_overlap_days; the block length is fixed here and never "
            "chosen from data",
            "family_rule": "Benjamini-Hochberg at multiple_testing.bh_fdr_q is applied "
            "separately to the raw-target p-values and to the demeaned-target "
            "p-values; each family is exactly the 16 decision cells listed in "
            "bh_family_cell_ids and is adjusted together; a cell is significant only "
            "when both adjusted p-values are at most q; Holm is reported over the "
            "same families and never decides",
            "bh_family_ids": [
                "raw_target_decision_cells",
                "demeaned_target_decision_cells",
            ],
            "bh_family_cell_ids": [
                f"{family_id}:{model_id}"
                for family_id in _FAMILY_KINDS
                for model_id in ("ridge_inner_cv", "lightgbm_fixed")
            ],
        },
        "baselines": [
            {"baseline_id": baseline_id, "role": role, "definition": text}
            for baseline_id, role, text in _BASELINES
        ],
        "stability": {
            "full_years": list(FULL_YEARS),
            "partial_years_reported_only": list(PARTIAL_YEARS),
            "year_basis": "calendar_year_utc_of_decision_time",
            "regime_lookback_days": 30,
            "regime_definition": "sign of the causal trailing 30-day cross-sectional "
            "mean log return of the five symbols from bars closed at or before the "
            "decision time; a zero return belongs to the non-negative regime",
            "extreme_day_trim_fraction_per_tail": 0.01,
            "extreme_day_definition": "UTC days ranked by the cross-sectional mean "
            "realised forward return at the primary horizon; the top 1% and bottom "
            "1% of days are excluded and the rank IC is recomputed",
            "reported_slices": ["year", "symbol", "regime", "extreme_day_exclusion"],
            "decisive_slices": ["year"],
            "year_sign_statistic": "for a decision cell and a full year, the pooled "
            "out-of-sample rank-IC increment over the drift baseline on the "
            "persistence target (decision_rules.persistence_target_id) over the "
            "decision rows whose UTC calendar year is that year; the year has the "
            "required sign only if that increment is strictly greater than zero",
            "min_covered_months_per_full_year": 12,
            "covered_month_min_valid_days": 15,
            "coverage_rule": "a UTC calendar month is covered for a decision cell "
            "when it holds at least covered_month_min_valid_days days with a valid "
            "daily rank-IC increment (a day skipped under measurement.skipped_days "
            "is not valid); a full year is eligible only if at least "
            "min_covered_months_per_full_year of its months are covered, and an "
            "ineligible year does not count toward min_full_years_same_sign whatever "
            "its sign; a year with fewer than that many covered months is reported "
            "with its coverage, never filled, reweighted or replaced by a partial year",
        },
        "decision_rules": {
            "cell_unit": "family_x_model",
            "primary_horizon_bars": 24,
            "primary_transform_id": "train_window_standardise_clip3",
            "signal_target_id": "raw_forward_log_return",
            "persistence_target_id": "cross_sectionally_demeaned_forward_log_return",
            "tradability_target_id": "drift_excess_payoff",
            "primary_k": 1,
            "min_full_years_same_sign": 4,
            "candidate_ci_lower_floor": 0.0,
            "candidate_stress_point_floor": 0.0,
            "tradability_cost_multiplier": 1,
            "stress_cost_multiplier": 2,
            "candidate_ranking": "drift_excess_ci_lower_cost_x1_desc_then_cell_id_asc",
            "unassigned_corner_resolutions": [
                "significant on the raw and the cross-sectionally demeaned target but "
                "positive in fewer than four eligible full years is NO_SIGNAL (H0b: "
                "concentration in a few years)",
                "a full year below the coverage floor is ineligible, so a cell with "
                "fewer than four eligible positive full years is NO_SIGNAL",
                "an interval that excludes zero at cost x1 but a non-positive point "
                "at cost x2 is SIGNAL_NOT_TRADABLE",
                "an interval wholly below zero at cost x1 is SIGNAL_NOT_TRADABLE",
            ],
        },
        "verdict_labels": list(VERDICT_LABELS),
        "budget": {
            "single_run": True,
            "save_all_cells": True,
            "post_result_additions_allowed": False,
            "retune_prior_learner_on_this_data": False,
            "max_candidates": 3,
            "native_replay_verification_cap": 3,
        },
    }


_FROZEN_REFERENCES: Final = {PROTOCOL_SCHEMA: _reference_payload}


# ------------------------------------------------------------------ strict rebuild


def _convert(hint: Any, value: object, path: str) -> object:
    origin = get_origin(hint)
    if origin is tuple:
        item_hint = get_args(hint)[0]
        if not isinstance(value, list):
            raise _fail(f"{path} must be an array")
        return tuple(
            _convert(item_hint, item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if origin in (Union, types.UnionType):
        arguments = [arg for arg in get_args(hint) if arg is not type(None)]
        if value is None:
            return None
        return _convert(arguments[0], value, path)
    if isinstance(hint, type) and is_dataclass(hint):
        return _build(hint, value, path)
    if hint is bool:
        if type(value) is not bool:
            raise _fail(f"{path} must be a boolean")
        return value
    if hint is int:
        if type(value) is not int:
            raise _fail(f"{path} must be an integer")
        return value
    if hint is float:
        if type(value) is not float or not math.isfinite(value):
            raise _fail(f"{path} must be a finite float")
        return value
    if hint is str:
        if type(value) is not str:
            raise _fail(f"{path} must be text")
        return value
    raise _fail(f"{path} has an unsupported declared type")  # pragma: no cover


def _build(cls: type, payload: object, path: str) -> Any:
    if not isinstance(payload, Mapping) or any(type(key) is not str for key in payload):
        raise _fail(f"{path} must be a string-keyed object")
    declared = [field.name for field in fields(cls)]
    missing = [name for name in declared if name not in payload]
    if missing:
        raise _fail(f"{path}: missing field(s) {sorted(missing)}")
    extra = sorted(set(payload) - set(declared))
    if extra:
        claims = [name for name in extra if _CLAIM_WORDS.search(name.lower())]
        if claims:
            raise _fail(
                f"{path}: claim field(s) {claims} are not allowed; this known-used "
                "exploratory protocol carries no edge, G4, G5 or live claim"
            )
        raise _fail(f"{path}: unexpected field(s) {extra}")
    hints = get_type_hints(cls)
    return cls(
        **{
            name: _convert(hints[name], payload[name], f"{path}.{name}")
            for name in declared
        }
    )


def protocol_from_payload(payload: Mapping[str, object]) -> SignalDiagnosisProtocol:
    """Rebuild and validate a protocol; unknown, missing or claim fields fail closed."""

    protocol: SignalDiagnosisProtocol = _build(
        SignalDiagnosisProtocol, payload, "protocol"
    )
    return protocol


def default_signal_diagnosis_protocol() -> SignalDiagnosisProtocol:
    """The frozen ``signal_diagnosis_protocol_v3`` object."""

    return protocol_from_payload(_reference_payload())


def protocol_canonical_bytes(protocol: SignalDiagnosisProtocol) -> bytes:
    """Canonical UTF-8 JSON of every protocol field (what the digest covers)."""

    return canonical_json_bytes(protocol)


def protocol_digest(protocol: SignalDiagnosisProtocol) -> str:
    return hashlib.sha256(protocol_canonical_bytes(protocol)).hexdigest()


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            raise _fail(f"duplicate JSON key {key!r} in the protocol")
        out[key] = value
    return out


def _reject_constant(name: str) -> object:
    raise _fail(f"protocol is not valid strict JSON (non-finite constant {name})")


def load_signal_diagnosis_protocol(
    raw: bytes, *, expected_digest: str
) -> SignalDiagnosisProtocol:
    """Load canonical protocol bytes only if they hash to ``expected_digest``.

    The digest is checked over the raw bytes before parsing; the bytes must then be
    strict (duplicate-free, finite) canonical JSON, rebuild into a valid protocol and
    equal the frozen reference of their declared version.
    """

    if type(expected_digest) is not str or _SHA256.fullmatch(expected_digest) is None:
        raise _fail("expected_digest must be a lowercase SHA-256 hex digest")
    if type(raw) is not bytes:
        raise _fail("raw protocol must be bytes")
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise _fail("protocol digest does not match the expected digest")
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except SignalDiagnosisProtocolError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _fail("protocol is not valid strict JSON") from error
    if not isinstance(decoded, dict):
        raise _fail("protocol must be a JSON object")
    protocol = protocol_from_payload(decoded)
    if protocol_canonical_bytes(protocol) != raw:
        raise _fail("protocol bytes are not the canonical JSON encoding")
    return protocol


# ------------------------------------------------------------------ run admission


@dataclass(frozen=True, slots=True)
class SignalDiagnosisRunRequest:
    """What a later runner must present; SD0 itself runs nothing.

    ``real_development`` needs a lowercase SHA-256 authorization digest, mirroring
    the #1025 plan contract. The digest records that a separate owner authorization
    exists; it is not that authorization and it widens no window.
    """

    data_scope: str
    protocol_digest: str
    evaluation_start: datetime | None
    evaluation_end: datetime | None
    real_scope_authorization_digest: str | None

    def __post_init__(self) -> None:
        if self.data_scope not in DATA_SCOPES:
            raise _fail("data_scope must be synthetic_only or real_development")
        _sha256(self.protocol_digest, "protocol_digest")
        digest = self.real_scope_authorization_digest
        if self.data_scope == SCOPE_SYNTHETIC:
            if self.evaluation_start is not None or self.evaluation_end is not None:
                raise _fail("synthetic runs must not carry a real evaluation window")
            if digest is not None:
                raise _fail("synthetic runs must not carry a real authorization")
            return
        if self.evaluation_start is None or self.evaluation_end is None:
            raise _fail("real_development requires an evaluation window")
        reject_forbidden_window(self.evaluation_start, self.evaluation_end)
        if type(digest) is not str or _SHA256.fullmatch(digest) is None:
            raise _fail(
                "real_development requires a separate authorization digest "
                "(lowercase SHA-256)"
            )


def admit_run(
    request: SignalDiagnosisRunRequest, protocol: SignalDiagnosisProtocol
) -> None:
    """Raise unless the request is bound to this protocol and stays inside its window."""

    if request.protocol_digest != protocol_digest(protocol):
        raise _fail("request protocol digest does not match the protocol")
    if request.data_scope == SCOPE_SYNTHETIC:
        return
    if request.evaluation_start is None or request.evaluation_end is None:
        raise _fail("real_development requires an evaluation window")
    begin = _aware_utc(request.evaluation_start, "window start")
    finish = _aware_utc(request.evaluation_end, "window end")
    allowed_start = _parse_z(protocol.data.evaluation_start, "evaluation_start")
    allowed_end = _parse_z(protocol.data.evaluation_end, "evaluation_end")
    if begin < allowed_start or finish > allowed_end:
        raise _fail("window lies outside the protocol window")


__all__ = [
    "ARTIFACT_DIGEST",
    "ARRAYS_DIGEST",
    "CLAIM_CEILING",
    "DATASET_ID",
    "DATA_SCOPES",
    "EVIDENCE_SCOPE",
    "EXECUTION_STATUS",
    "FEATURE_CONFIG_DIGEST",
    "FULL_YEARS",
    "PARTIAL_YEARS",
    "PROTOCOL_SCHEMA",
    "SCOPE_REAL",
    "SCOPE_SYNTHETIC",
    "TAIL_ONE_SIDED_POSITIVE",
    "VERDICT_CANDIDATE",
    "VERDICT_INVALID",
    "VERDICT_LABELS",
    "VERDICT_NOT_TRADABLE",
    "VERDICT_NO_SIGNAL",
    "BaselineSpec",
    "BootstrapSpec",
    "BudgetSpec",
    "CostSpec",
    "DataScope",
    "DecisionRules",
    "FeatureFamily",
    "ForbiddenEvaluationWindowError",
    "ForbiddenSpan",
    "HorizonGrid",
    "Hypothesis",
    "LayerSpec",
    "LightGBMSpec",
    "MeasurementSpec",
    "ModelGrid",
    "MultipleTestingSpec",
    "PrequentialSpec",
    "RidgeSpec",
    "SignalDiagnosisProtocol",
    "SignalDiagnosisProtocolError",
    "SignalDiagnosisRunRequest",
    "SignificanceSpec",
    "StabilitySpec",
    "TargetSpec",
    "TransformSpec",
    "admit_run",
    "default_signal_diagnosis_protocol",
    "load_signal_diagnosis_protocol",
    "protocol_canonical_bytes",
    "protocol_digest",
    "protocol_from_payload",
    "reject_forbidden_window",
]
