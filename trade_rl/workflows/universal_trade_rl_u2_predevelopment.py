"""Fail-closed research closure before Universal Trade RL U2 Development access."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLSymbolRole
from trade_rl.workflows.universal_trade_rl_u2_contract import U2_TRAINING_SEEDS
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
)

U2_PREDEVELOPMENT_SCHEMA: Final = "universal_trade_rl_u2_predevelopment_contract_v1"
U2_DEVELOPMENT_LOCK_SCHEMA: Final = "universal_trade_rl_u2_development_lock_v1"
U2_EVALUATION_CRN_SCHEMA: Final = "universal_trade_rl_u2_evaluation_crn_v1"
U2_PRODUCTION_STATUS: Final = "NO-GO"
U2_ADMISSION_STATUS: Final = "SEALED"
U2_MIN_TRAIN_SYMBOLS: Final = 9
U2_MIN_DEVELOPMENT_SYMBOLS: Final = 3
U2_MIN_ADMISSION_SYMBOLS: Final = 3

_PREDEVELOPMENT_KEYS: Final = (
    "schema_version",
    "universe_manifest_digest",
    "u2_contract_digest",
    "train_symbol_count",
    "development_symbol_count",
    "admission_symbol_count",
    "role_cardinality_contract_digest",
    "evaluation_crn_contract_digest",
    "selection_metric_contract_digest",
    "bootstrap_panel_contract_digest",
    "resume_contract_digest",
    "training_exposure_contract_digest",
    "production_status",
    "admission_status",
    "artifact_digest",
)
_DEVELOPMENT_LOCK_KEYS: Final = (
    "schema_version",
    "predevelopment_contract_digest",
    "universe_manifest_digest",
    "u1_contract_digest",
    "u1_normalizer_digest",
    "u2_contract_digest",
    "checkpoint_digests",
    "development_scope_closure_digest",
    "evaluation_dataset_digests",
    "source_tree_digest",
    "lockfile_digest",
    "evaluation_runtime_identity_digest",
    "development_numeric_open_count",
    "admission_numeric_open_count",
    "admission_status",
    "production_status",
    "artifact_digest",
)


def _exact_mapping(
    value: object,
    *,
    keys: tuple[str, ...],
    field: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object with exact keys")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field} keys must be strings")
        result[key] = item
    if set(result) != set(keys) or len(result) != len(keys):
        raise ValueError(f"{field} must use exact keys")
    return result


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer and not boolean")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an array")
    return tuple(value)


def _integer_digest_pairs(
    value: object,
    *,
    field: str,
) -> tuple[tuple[int, str], ...]:
    result: list[tuple[int, str]] = []
    for row in _sequence(value, field=field):
        pair = _sequence(row, field=f"{field} row")
        if len(pair) != 2:
            raise ValueError(f"{field} rows must be key/digest pairs")
        result.append(
            (
                _integer(pair[0], field=f"{field} key"),
                _string(pair[1], field=f"{field} digest"),
            )
        )
    return tuple(result)


def _string_digest_pairs(
    value: object,
    *,
    field: str,
) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for row in _sequence(value, field=field):
        pair = _sequence(row, field=f"{field} row")
        if len(pair) != 2:
            raise ValueError(f"{field} rows must be key/digest pairs")
        result.append(
            (
                _string(pair[0], field=f"{field} key"),
                _string(pair[1], field=f"{field} digest"),
            )
        )
    return tuple(result)


def _role_cardinality_contract_payload() -> dict[str, object]:
    return {
        "schema_version": "universal_trade_rl_u2_role_cardinality_contract_v1",
        "minimum_train_symbols": U2_MIN_TRAIN_SYMBOLS,
        "minimum_development_symbols": U2_MIN_DEVELOPMENT_SYMBOLS,
        "minimum_admission_symbols": U2_MIN_ADMISSION_SYMBOLS,
    }


def _evaluation_crn_contract_payload() -> dict[str, object]:
    return {
        "schema_version": "universal_trade_rl_u2_evaluation_crn_contract_v1",
        "seed_schema_version": U2_EVALUATION_CRN_SCHEMA,
        "allowed_evaluation_seeds": U2_TRAINING_SEEDS,
        "seed_source_fields": ("u2_contract_digest", "scope_digest"),
        "training_seed_input_allowed": False,
        "checkpoint_input_allowed": False,
        "same_scope_pairing_required": True,
    }


def _selection_metric_contract_payload() -> dict[str, object]:
    return {
        "schema_version": "universal_trade_rl_u2_selection_metric_contract_v1",
        "leaf_identity": (
            "training_seed",
            "cell",
            "concrete_symbol",
            "tile_identity",
        ),
        "return_domain": "log1p_simple_returns",
        "gross_net_same_replay_required": True,
        "symbol_log_growth_reduction": "sum_leaves",
        "symbol_balanced_log_growth_reduction": "equal_weight_mean_symbols",
        "wealth_transform": "exp_log_growth",
        "median_symbol_net_wealth": "statistics.median",
        "minimum_symbol_net_wealth": "min",
        "positive_scope_rule": "leaf_net_log_growth_strictly_greater_than_zero",
        "cvar10_count_rule": "max_1_ceil_0.10_times_leaf_count",
        "cvar10_value_rule": "mean_worst_leaf_net_log_growth",
        "turnover_per_day_rule": "turnover_total/(decision_count*0.25/24)",
        "turnover_p95_quantile": 0.95,
        "turnover_p95_method": "linear",
        "meaningful_execution_rule": (
            "executed_change_count_gt_0_or_turnover_total_gt_1e-6"
        ),
        "meaningful_execution_turnover_tolerance": 1e-6,
        "positive_gross_retention_rule": (
            "symbol_balanced_net_log_growth/symbol_balanced_gross_log_growth"
        ),
    }


def _bootstrap_panel_contract_payload() -> dict[str, object]:
    return {
        "schema_version": "universal_trade_rl_u2_bootstrap_panel_contract_v1",
        "paired_quantity": "candidate_minus_cash_net_log_excess",
        "symbol_reduction": "equal_weight_mean_per_timestamp",
        "seed_reduction": "median_per_timestamp",
        "time_order": "chronological",
        "aggregate_segments": ("development_future_1", "development_future_2"),
        "blocks_may_cross_segment_boundary": False,
        "bootstrap_method": "moving_block_mean_test",
        "confidence_level": 0.95,
        "resamples": 2_000,
        "bootstrap_seed": 0,
        "block_length_rule": "ceil_sqrt_segment_length_capped",
        "quantile_method": "linear",
        "pass_rule": "lower_95pct_ci_strictly_greater_than_zero",
    }


def _resume_contract_payload() -> dict[str, object]:
    return {
        "schema_version": "universal_trade_rl_u2_resume_contract_v1",
        "exact_mid_episode_resume_supported": False,
        "restart_from_timestep_zero_required": True,
        "intermediate_checkpoint_role": "recovery_debug_evidence_only",
        "selection_checkpoint_rule": "exact_final_fixed_budget_only",
    }


def _training_exposure_contract_payload() -> dict[str, object]:
    return {
        "schema_version": "universal_trade_rl_u2_training_exposure_contract_v1",
        "required_fields": (
            "training_seed",
            "worker_index",
            "concrete_symbol",
            "completed_episode_count",
            "decision_step_count",
            "partial_final_episode_step_count",
            "routing_cycle_count",
        ),
        "posthoc_reweighting_allowed": False,
        "required_before_development_open": True,
    }


def universal_trade_rl_u2_evaluation_seed(
    *,
    u2_contract_digest: str,
    scope_digest: str,
) -> int:
    """Derive the scope-common Development execution RNG seed."""

    require_sha256(u2_contract_digest, field="U2 evaluation CRN contract digest")
    require_sha256(scope_digest, field="U2 evaluation CRN scope digest")
    material = {
        "schema_version": U2_EVALUATION_CRN_SCHEMA,
        "u2_contract_digest": u2_contract_digest,
        "scope_digest": scope_digest,
        "allowed_evaluation_seeds": U2_TRAINING_SEEDS,
    }
    index = int(content_digest(material)[:8], 16) % len(U2_TRAINING_SEEDS)
    return U2_TRAINING_SEEDS[index]


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2PreDevelopmentContract:
    """Metadata-only U2 research contract frozen before real PPO training."""

    universe_manifest_digest: str
    u2_contract_digest: str
    train_symbol_count: int
    development_symbol_count: int
    admission_symbol_count: int
    role_cardinality_contract_digest: str
    evaluation_crn_contract_digest: str
    selection_metric_contract_digest: str
    bootstrap_panel_contract_digest: str
    resume_contract_digest: str
    training_exposure_contract_digest: str
    production_status: str = U2_PRODUCTION_STATUS
    admission_status: str = U2_ADMISSION_STATUS
    schema_version: str = U2_PREDEVELOPMENT_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_PREDEVELOPMENT_SCHEMA:
            raise ValueError("unsupported U2 pre-development contract schema")
        for field_name, value in (
            ("universe_manifest_digest", self.universe_manifest_digest),
            ("u2_contract_digest", self.u2_contract_digest),
            ("role_cardinality_contract_digest", self.role_cardinality_contract_digest),
            ("evaluation_crn_contract_digest", self.evaluation_crn_contract_digest),
            ("selection_metric_contract_digest", self.selection_metric_contract_digest),
            ("bootstrap_panel_contract_digest", self.bootstrap_panel_contract_digest),
            ("resume_contract_digest", self.resume_contract_digest),
            (
                "training_exposure_contract_digest",
                self.training_exposure_contract_digest,
            ),
        ):
            require_sha256(value, field=f"U2 pre-development {field_name}")

        counts = (
            (
                "Train",
                _integer(self.train_symbol_count, field="U2 Train symbol count"),
                U2_MIN_TRAIN_SYMBOLS,
            ),
            (
                "Development",
                _integer(
                    self.development_symbol_count,
                    field="U2 Development symbol count",
                ),
                U2_MIN_DEVELOPMENT_SYMBOLS,
            ),
            (
                "Admission",
                _integer(
                    self.admission_symbol_count,
                    field="U2 Admission symbol count",
                ),
                U2_MIN_ADMISSION_SYMBOLS,
            ),
        )
        for label, count, minimum in counts:
            if count < minimum:
                raise ValueError(
                    f"U2 pre-development {label} role requires at least {minimum} symbols"
                )

        expected_contracts = (
            (
                self.role_cardinality_contract_digest,
                content_digest(_role_cardinality_contract_payload()),
                "role cardinality",
            ),
            (
                self.evaluation_crn_contract_digest,
                content_digest(_evaluation_crn_contract_payload()),
                "evaluation CRN",
            ),
            (
                self.selection_metric_contract_digest,
                content_digest(_selection_metric_contract_payload()),
                "Selection metric",
            ),
            (
                self.bootstrap_panel_contract_digest,
                content_digest(_bootstrap_panel_contract_payload()),
                "bootstrap panel",
            ),
            (
                self.resume_contract_digest,
                content_digest(_resume_contract_payload()),
                "resume",
            ),
            (
                self.training_exposure_contract_digest,
                content_digest(_training_exposure_contract_payload()),
                "training exposure",
            ),
        )
        for actual, expected, label in expected_contracts:
            if actual != expected:
                raise ValueError(f"U2 pre-development {label} contract drifted")

        if self.production_status != U2_PRODUCTION_STATUS:
            raise ValueError("Universal Trade RL U2 remains Production NO-GO")
        if self.admission_status != U2_ADMISSION_STATUS:
            raise ValueError("Universal Trade RL U2 Admission remains SEALED")

        expected_digest = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 pre-development artifact digest")
            if self.digest != expected_digest:
                raise ValueError("U2 pre-development artifact digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "universe_manifest_digest": self.universe_manifest_digest,
            "u2_contract_digest": self.u2_contract_digest,
            "train_symbol_count": self.train_symbol_count,
            "development_symbol_count": self.development_symbol_count,
            "admission_symbol_count": self.admission_symbol_count,
            "role_cardinality_contract_digest": self.role_cardinality_contract_digest,
            "evaluation_crn_contract_digest": self.evaluation_crn_contract_digest,
            "selection_metric_contract_digest": self.selection_metric_contract_digest,
            "bootstrap_panel_contract_digest": self.bootstrap_panel_contract_digest,
            "resume_contract_digest": self.resume_contract_digest,
            "training_exposure_contract_digest": self.training_exposure_contract_digest,
            "production_status": self.production_status,
            "admission_status": self.admission_status,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload

    @classmethod
    def from_payload(cls, value: object) -> UniversalTradeRLU2PreDevelopmentContract:
        payload = _exact_mapping(
            value,
            keys=_PREDEVELOPMENT_KEYS,
            field="U2 pre-development contract",
        )
        return cls(
            universe_manifest_digest=_string(
                payload["universe_manifest_digest"],
                field="U2 pre-development universe manifest digest",
            ),
            u2_contract_digest=_string(
                payload["u2_contract_digest"],
                field="U2 pre-development contract digest",
            ),
            train_symbol_count=_integer(
                payload["train_symbol_count"],
                field="U2 Train symbol count",
            ),
            development_symbol_count=_integer(
                payload["development_symbol_count"],
                field="U2 Development symbol count",
            ),
            admission_symbol_count=_integer(
                payload["admission_symbol_count"],
                field="U2 Admission symbol count",
            ),
            role_cardinality_contract_digest=_string(
                payload["role_cardinality_contract_digest"],
                field="U2 role cardinality contract digest",
            ),
            evaluation_crn_contract_digest=_string(
                payload["evaluation_crn_contract_digest"],
                field="U2 evaluation CRN contract digest",
            ),
            selection_metric_contract_digest=_string(
                payload["selection_metric_contract_digest"],
                field="U2 Selection metric contract digest",
            ),
            bootstrap_panel_contract_digest=_string(
                payload["bootstrap_panel_contract_digest"],
                field="U2 bootstrap panel contract digest",
            ),
            resume_contract_digest=_string(
                payload["resume_contract_digest"],
                field="U2 resume contract digest",
            ),
            training_exposure_contract_digest=_string(
                payload["training_exposure_contract_digest"],
                field="U2 training exposure contract digest",
            ),
            production_status=_string(
                payload["production_status"],
                field="U2 production status",
            ),
            admission_status=_string(
                payload["admission_status"],
                field="U2 Admission status",
            ),
            schema_version=_string(
                payload["schema_version"],
                field="U2 pre-development schema",
            ),
            digest=_string(
                payload["artifact_digest"],
                field="U2 pre-development artifact digest",
            ),
        )


def build_universal_trade_rl_u2_predevelopment_contract(
    *,
    manifest: UniversalTradeRLUniverseManifest,
    u2_contract_digest: str,
    u2_universe_manifest_digest: str,
) -> UniversalTradeRLU2PreDevelopmentContract:
    """Freeze U2 research degrees of freedom using metadata only."""

    if not isinstance(manifest, UniversalTradeRLUniverseManifest):
        raise TypeError("U2 pre-development closure requires a universe manifest")
    require_sha256(u2_contract_digest, field="U2 pre-development U2 contract digest")
    require_sha256(
        u2_universe_manifest_digest,
        field="U2 pre-development U2 universe manifest digest",
    )
    if u2_universe_manifest_digest != manifest.digest:
        raise ValueError("U2 pre-development universe manifest identity mismatch")

    counts = {
        role: sum(entry.role is role for entry in manifest.entries)
        for role in UniversalTradeRLSymbolRole
    }
    return UniversalTradeRLU2PreDevelopmentContract(
        universe_manifest_digest=manifest.digest,
        u2_contract_digest=u2_contract_digest,
        train_symbol_count=counts[UniversalTradeRLSymbolRole.TRAIN],
        development_symbol_count=counts[UniversalTradeRLSymbolRole.DEVELOPMENT],
        admission_symbol_count=counts[UniversalTradeRLSymbolRole.ADMISSION],
        role_cardinality_contract_digest=content_digest(
            _role_cardinality_contract_payload()
        ),
        evaluation_crn_contract_digest=content_digest(
            _evaluation_crn_contract_payload()
        ),
        selection_metric_contract_digest=content_digest(
            _selection_metric_contract_payload()
        ),
        bootstrap_panel_contract_digest=content_digest(
            _bootstrap_panel_contract_payload()
        ),
        resume_contract_digest=content_digest(_resume_contract_payload()),
        training_exposure_contract_digest=content_digest(
            _training_exposure_contract_payload()
        ),
    )


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2DevelopmentLock:
    """Immutable lock published before the first Development numeric source open."""

    predevelopment_contract_digest: str
    universe_manifest_digest: str
    u1_contract_digest: str
    u1_normalizer_digest: str
    u2_contract_digest: str
    checkpoint_digests: tuple[tuple[int, str], ...]
    development_scope_closure_digest: str
    evaluation_dataset_digests: tuple[tuple[str, str], ...]
    source_tree_digest: str
    lockfile_digest: str
    evaluation_runtime_identity_digest: str
    development_numeric_open_count: int
    admission_numeric_open_count: int
    admission_status: str = U2_ADMISSION_STATUS
    production_status: str = U2_PRODUCTION_STATUS
    schema_version: str = U2_DEVELOPMENT_LOCK_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_DEVELOPMENT_LOCK_SCHEMA:
            raise ValueError("unsupported U2 Development lock schema")
        for field_name, value in (
            ("predevelopment_contract_digest", self.predevelopment_contract_digest),
            ("universe_manifest_digest", self.universe_manifest_digest),
            ("u1_contract_digest", self.u1_contract_digest),
            ("u1_normalizer_digest", self.u1_normalizer_digest),
            ("u2_contract_digest", self.u2_contract_digest),
            (
                "development_scope_closure_digest",
                self.development_scope_closure_digest,
            ),
            ("source_tree_digest", self.source_tree_digest),
            ("lockfile_digest", self.lockfile_digest),
            (
                "evaluation_runtime_identity_digest",
                self.evaluation_runtime_identity_digest,
            ),
        ):
            require_sha256(value, field=f"U2 Development lock {field_name}")

        checkpoints = tuple(self.checkpoint_digests)
        if tuple(seed for seed, _digest in checkpoints) != U2_TRAINING_SEEDS:
            raise ValueError(
                "U2 Development lock checkpoint seeds must be the exact canonical closure"
            )
        for seed, digest in checkpoints:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("U2 Development lock checkpoint seed is invalid")
            require_sha256(digest, field="U2 Development lock checkpoint digest")

        datasets = tuple(self.evaluation_dataset_digests)
        if not datasets:
            raise ValueError("U2 Development lock dataset mapping must not be empty")
        symbols = tuple(symbol for symbol, _digest in datasets)
        if symbols != tuple(sorted(symbols)):
            raise ValueError(
                "U2 Development lock dataset mapping must be sorted and canonical"
            )
        if len(set(symbols)) != len(symbols):
            raise ValueError("U2 Development lock dataset mapping must be unique")
        for symbol, digest in datasets:
            if not isinstance(symbol, str) or not symbol:
                raise ValueError("U2 Development lock dataset symbol is invalid")
            require_sha256(digest, field="U2 Development lock dataset digest")

        development_opens = _integer(
            self.development_numeric_open_count,
            field="U2 Development numeric open count",
        )
        admission_opens = _integer(
            self.admission_numeric_open_count,
            field="U2 Admission numeric open count",
        )
        if development_opens != 0 or admission_opens != 0:
            raise ValueError(
                "U2 Development lock requires zero numeric opens while Admission is sealed"
            )
        if self.admission_status != U2_ADMISSION_STATUS:
            raise ValueError("U2 Development lock requires Admission SEALED")
        if self.production_status != U2_PRODUCTION_STATUS:
            raise ValueError("Universal Trade RL U2 remains Production NO-GO")

        object.__setattr__(self, "checkpoint_digests", checkpoints)
        object.__setattr__(self, "evaluation_dataset_digests", datasets)
        expected_digest = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 Development lock artifact digest")
            if self.digest != expected_digest:
                raise ValueError("U2 Development lock artifact digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "predevelopment_contract_digest": self.predevelopment_contract_digest,
            "universe_manifest_digest": self.universe_manifest_digest,
            "u1_contract_digest": self.u1_contract_digest,
            "u1_normalizer_digest": self.u1_normalizer_digest,
            "u2_contract_digest": self.u2_contract_digest,
            "checkpoint_digests": self.checkpoint_digests,
            "development_scope_closure_digest": self.development_scope_closure_digest,
            "evaluation_dataset_digests": self.evaluation_dataset_digests,
            "source_tree_digest": self.source_tree_digest,
            "lockfile_digest": self.lockfile_digest,
            "evaluation_runtime_identity_digest": self.evaluation_runtime_identity_digest,
            "development_numeric_open_count": self.development_numeric_open_count,
            "admission_numeric_open_count": self.admission_numeric_open_count,
            "admission_status": self.admission_status,
            "production_status": self.production_status,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload

    @classmethod
    def from_payload(cls, value: object) -> UniversalTradeRLU2DevelopmentLock:
        payload = _exact_mapping(
            value,
            keys=_DEVELOPMENT_LOCK_KEYS,
            field="U2 Development lock",
        )
        return cls(
            predevelopment_contract_digest=_string(
                payload["predevelopment_contract_digest"],
                field="U2 Development lock pre-development digest",
            ),
            universe_manifest_digest=_string(
                payload["universe_manifest_digest"],
                field="U2 Development lock universe digest",
            ),
            u1_contract_digest=_string(
                payload["u1_contract_digest"],
                field="U2 Development lock U1 digest",
            ),
            u1_normalizer_digest=_string(
                payload["u1_normalizer_digest"],
                field="U2 Development lock normalizer digest",
            ),
            u2_contract_digest=_string(
                payload["u2_contract_digest"],
                field="U2 Development lock U2 digest",
            ),
            checkpoint_digests=_integer_digest_pairs(
                payload["checkpoint_digests"],
                field="U2 Development lock checkpoint mapping",
            ),
            development_scope_closure_digest=_string(
                payload["development_scope_closure_digest"],
                field="U2 Development lock scope closure digest",
            ),
            evaluation_dataset_digests=_string_digest_pairs(
                payload["evaluation_dataset_digests"],
                field="U2 Development lock dataset mapping",
            ),
            source_tree_digest=_string(
                payload["source_tree_digest"],
                field="U2 Development lock source tree digest",
            ),
            lockfile_digest=_string(
                payload["lockfile_digest"],
                field="U2 Development lock lockfile digest",
            ),
            evaluation_runtime_identity_digest=_string(
                payload["evaluation_runtime_identity_digest"],
                field="U2 Development lock runtime identity digest",
            ),
            development_numeric_open_count=_integer(
                payload["development_numeric_open_count"],
                field="U2 Development numeric open count",
            ),
            admission_numeric_open_count=_integer(
                payload["admission_numeric_open_count"],
                field="U2 Admission numeric open count",
            ),
            admission_status=_string(
                payload["admission_status"],
                field="U2 Admission status",
            ),
            production_status=_string(
                payload["production_status"],
                field="U2 production status",
            ),
            schema_version=_string(
                payload["schema_version"],
                field="U2 Development lock schema",
            ),
            digest=_string(
                payload["artifact_digest"],
                field="U2 Development lock artifact digest",
            ),
        )


def build_universal_trade_rl_u2_development_lock(
    *,
    predevelopment_contract: UniversalTradeRLU2PreDevelopmentContract,
    u1_contract_digest: str,
    u1_normalizer_digest: str,
    checkpoint_digests: tuple[tuple[int, str], ...],
    development_scope_closure_digest: str,
    evaluation_dataset_digests: tuple[tuple[str, str], ...],
    source_tree_digest: str,
    lockfile_digest: str,
    evaluation_runtime_identity_digest: str,
    development_numeric_open_count: int,
    admission_numeric_open_count: int,
) -> UniversalTradeRLU2DevelopmentLock:
    """Freeze exact Development inputs before any Development numeric open."""

    if not isinstance(
        predevelopment_contract,
        UniversalTradeRLU2PreDevelopmentContract,
    ):
        raise TypeError("U2 Development lock requires a pre-development contract")
    return UniversalTradeRLU2DevelopmentLock(
        predevelopment_contract_digest=predevelopment_contract.digest,
        universe_manifest_digest=predevelopment_contract.universe_manifest_digest,
        u1_contract_digest=u1_contract_digest,
        u1_normalizer_digest=u1_normalizer_digest,
        u2_contract_digest=predevelopment_contract.u2_contract_digest,
        checkpoint_digests=checkpoint_digests,
        development_scope_closure_digest=development_scope_closure_digest,
        evaluation_dataset_digests=evaluation_dataset_digests,
        source_tree_digest=source_tree_digest,
        lockfile_digest=lockfile_digest,
        evaluation_runtime_identity_digest=evaluation_runtime_identity_digest,
        development_numeric_open_count=development_numeric_open_count,
        admission_numeric_open_count=admission_numeric_open_count,
    )


__all__ = [
    "U2_ADMISSION_STATUS",
    "U2_DEVELOPMENT_LOCK_SCHEMA",
    "U2_EVALUATION_CRN_SCHEMA",
    "U2_MIN_ADMISSION_SYMBOLS",
    "U2_MIN_DEVELOPMENT_SYMBOLS",
    "U2_MIN_TRAIN_SYMBOLS",
    "U2_PREDEVELOPMENT_SCHEMA",
    "U2_PRODUCTION_STATUS",
    "UniversalTradeRLU2DevelopmentLock",
    "UniversalTradeRLU2PreDevelopmentContract",
    "build_universal_trade_rl_u2_development_lock",
    "build_universal_trade_rl_u2_predevelopment_contract",
    "universal_trade_rl_u2_evaluation_seed",
]
