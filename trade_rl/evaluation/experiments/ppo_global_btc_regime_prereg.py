"""Result-blind preregistration for the Issue 605 PPO BTC-regime factor."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest

_SCHEMA_VERSION = "ppo_global_btc_regime_prereg_v1"
_ISSUE_NUMBER = 605
_SOURCE_MAIN_SHA = "c5a1ce8395feaedd8833f7e596fddc9d8f3115fc"
_SOURCE_DATASET_ID = "d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f"
_SOURCE_DATASET_ARTIFACT_DIGEST = (
    "77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8"
)
_SOURCE_STUDY_DIGEST = (
    "3d8404061a4082a8e9b3c786d9f5fc9a4347631dff39c201e3cba70470dfeb79"
)
_SOURCE_EVIDENCE_FINGERPRINT = (
    "526b485d60b394739b7a8d03535e1cfa08b26fa920119c70ee53f05faa53dd29"
)
_SOURCE_STATUS_DOCUMENT = "docs/research/current-status.md"

_CONTROLLED_FACTOR = "ppo_global_btc_regime_context"
_REFERENCE_SYMBOL = "BTCUSDT"
_REFERENCE_FEATURE = "1h__log_return_24bar"
_PPO_SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True, slots=True)
class PPOGlobalBTCRegimeProtocol:
    """Immutable single-factor authority for Issue 605."""

    schema_version: str = _SCHEMA_VERSION
    issue_number: int = _ISSUE_NUMBER
    source_main_sha: str = _SOURCE_MAIN_SHA
    source_dataset_id: str = _SOURCE_DATASET_ID
    source_dataset_artifact_digest: str = _SOURCE_DATASET_ARTIFACT_DIGEST
    source_study_digest: str = _SOURCE_STUDY_DIGEST
    source_evidence_fingerprint: str = _SOURCE_EVIDENCE_FINGERPRINT
    source_status_document: str = _SOURCE_STATUS_DOCUMENT

    controlled_factor: str = _CONTROLLED_FACTOR
    reference_symbol: str = _REFERENCE_SYMBOL
    reference_feature: str = _REFERENCE_FEATURE
    reference_requires_available: bool = True
    reference_requires_finite: bool = True
    reference_includes_normalized_staleness: bool = True
    reference_missing_fails_closed: bool = True
    reference_uses_existing_dataset_only: bool = True
    reference_uses_future_or_global_statistics: bool = False
    reference_availability_semantics: str = "canonical_dataset_feature_available"
    reference_staleness_semantics: str = "observation_v2_normalized_staleness"

    ppo_seeds: tuple[int, ...] = _PPO_SEEDS
    factor_slots_authorized: int = 1
    only_ppo_may_change: bool = True
    unaffected_strategies_raw_return_invariance_required: bool = True
    reuse_existing_controlled_experiment_semantics: bool = True
    development_window_changed: bool = False
    ppo_hyperparameters_changed: bool = False
    ppo_seed_budget_changed: bool = False

    symbol_id_allowed: bool = False
    symbol_embedding_allowed: bool = False
    symbol_specific_coefficients_allowed: bool = False
    cross_sectional_aggregation_allowed: bool = False
    instrument_profiles_allowed: bool = False
    residual_beta_fitting_allowed: bool = False
    new_source_data_allowed: bool = False
    alternate_lookback_or_horizon_allowed: bool = False
    extra_seeds_allowed: bool = False
    extra_symbols_allowed: bool = False
    zero_snap_allowed: bool = False
    neutral_decay_allowed: bool = False
    gate_relaxation_allowed: bool = False

    action_contract_changed: bool = False
    execution_contract_changed: bool = False
    reward_contract_changed: bool = False
    risk_contract_changed: bool = False
    economics_contract_changed: bool = False

    final_test_access_authorized: bool = False
    economic_execution_authorized: bool = False
    economic_result_inspected: bool = False
    production_eligible: bool = False
    live_trading_authorized: bool = False

    def __post_init__(self) -> None:
        canonical = _canonical_field_values()
        for field in fields(self):
            actual = getattr(self, field.name)
            expected = canonical[field.name]
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    f"{field.name} differs from preregistered Issue 605 authority"
                )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "issue_number": self.issue_number,
            "source_main_sha": self.source_main_sha,
            "source_dataset_id": self.source_dataset_id,
            "source_dataset_artifact_digest": self.source_dataset_artifact_digest,
            "source_study_digest": self.source_study_digest,
            "source_evidence_fingerprint": self.source_evidence_fingerprint,
            "source_status_document": self.source_status_document,
            "controlled_factor": self.controlled_factor,
            "reference_symbol": self.reference_symbol,
            "reference_feature": self.reference_feature,
            "reference_requires_available": self.reference_requires_available,
            "reference_requires_finite": self.reference_requires_finite,
            "reference_includes_normalized_staleness": (
                self.reference_includes_normalized_staleness
            ),
            "reference_missing_fails_closed": self.reference_missing_fails_closed,
            "reference_uses_existing_dataset_only": (
                self.reference_uses_existing_dataset_only
            ),
            "reference_uses_future_or_global_statistics": (
                self.reference_uses_future_or_global_statistics
            ),
            "reference_availability_semantics": self.reference_availability_semantics,
            "reference_staleness_semantics": self.reference_staleness_semantics,
            "ppo_seeds": list(self.ppo_seeds),
            "factor_slots_authorized": self.factor_slots_authorized,
            "only_ppo_may_change": self.only_ppo_may_change,
            "unaffected_strategies_raw_return_invariance_required": (
                self.unaffected_strategies_raw_return_invariance_required
            ),
            "reuse_existing_controlled_experiment_semantics": (
                self.reuse_existing_controlled_experiment_semantics
            ),
            "development_window_changed": self.development_window_changed,
            "ppo_hyperparameters_changed": self.ppo_hyperparameters_changed,
            "ppo_seed_budget_changed": self.ppo_seed_budget_changed,
            "symbol_id_allowed": self.symbol_id_allowed,
            "symbol_embedding_allowed": self.symbol_embedding_allowed,
            "symbol_specific_coefficients_allowed": (
                self.symbol_specific_coefficients_allowed
            ),
            "cross_sectional_aggregation_allowed": (
                self.cross_sectional_aggregation_allowed
            ),
            "instrument_profiles_allowed": self.instrument_profiles_allowed,
            "residual_beta_fitting_allowed": self.residual_beta_fitting_allowed,
            "new_source_data_allowed": self.new_source_data_allowed,
            "alternate_lookback_or_horizon_allowed": (
                self.alternate_lookback_or_horizon_allowed
            ),
            "extra_seeds_allowed": self.extra_seeds_allowed,
            "extra_symbols_allowed": self.extra_symbols_allowed,
            "zero_snap_allowed": self.zero_snap_allowed,
            "neutral_decay_allowed": self.neutral_decay_allowed,
            "gate_relaxation_allowed": self.gate_relaxation_allowed,
            "action_contract_changed": self.action_contract_changed,
            "execution_contract_changed": self.execution_contract_changed,
            "reward_contract_changed": self.reward_contract_changed,
            "risk_contract_changed": self.risk_contract_changed,
            "economics_contract_changed": self.economics_contract_changed,
            "final_test_access_authorized": self.final_test_access_authorized,
            "economic_execution_authorized": self.economic_execution_authorized,
            "economic_result_inspected": self.economic_result_inspected,
            "production_eligible": self.production_eligible,
            "live_trading_authorized": self.live_trading_authorized,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


def _canonical_field_values() -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "issue_number": _ISSUE_NUMBER,
        "source_main_sha": _SOURCE_MAIN_SHA,
        "source_dataset_id": _SOURCE_DATASET_ID,
        "source_dataset_artifact_digest": _SOURCE_DATASET_ARTIFACT_DIGEST,
        "source_study_digest": _SOURCE_STUDY_DIGEST,
        "source_evidence_fingerprint": _SOURCE_EVIDENCE_FINGERPRINT,
        "source_status_document": _SOURCE_STATUS_DOCUMENT,
        "controlled_factor": _CONTROLLED_FACTOR,
        "reference_symbol": _REFERENCE_SYMBOL,
        "reference_feature": _REFERENCE_FEATURE,
        "reference_requires_available": True,
        "reference_requires_finite": True,
        "reference_includes_normalized_staleness": True,
        "reference_missing_fails_closed": True,
        "reference_uses_existing_dataset_only": True,
        "reference_uses_future_or_global_statistics": False,
        "reference_availability_semantics": "canonical_dataset_feature_available",
        "reference_staleness_semantics": "observation_v2_normalized_staleness",
        "ppo_seeds": _PPO_SEEDS,
        "factor_slots_authorized": 1,
        "only_ppo_may_change": True,
        "unaffected_strategies_raw_return_invariance_required": True,
        "reuse_existing_controlled_experiment_semantics": True,
        "development_window_changed": False,
        "ppo_hyperparameters_changed": False,
        "ppo_seed_budget_changed": False,
        "symbol_id_allowed": False,
        "symbol_embedding_allowed": False,
        "symbol_specific_coefficients_allowed": False,
        "cross_sectional_aggregation_allowed": False,
        "instrument_profiles_allowed": False,
        "residual_beta_fitting_allowed": False,
        "new_source_data_allowed": False,
        "alternate_lookback_or_horizon_allowed": False,
        "extra_seeds_allowed": False,
        "extra_symbols_allowed": False,
        "zero_snap_allowed": False,
        "neutral_decay_allowed": False,
        "gate_relaxation_allowed": False,
        "action_contract_changed": False,
        "execution_contract_changed": False,
        "reward_contract_changed": False,
        "risk_contract_changed": False,
        "economics_contract_changed": False,
        "final_test_access_authorized": False,
        "economic_execution_authorized": False,
        "economic_result_inspected": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }


def canonical_ppo_global_btc_regime_protocol() -> PPOGlobalBTCRegimeProtocol:
    return PPOGlobalBTCRegimeProtocol()


def load_ppo_global_btc_regime_protocol(path: Path) -> PPOGlobalBTCRegimeProtocol:
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    raw = path.read_bytes()
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("preregistered protocol JSON is malformed") from error
    if not isinstance(decoded, dict) or any(
        not isinstance(key, str) for key in decoded
    ):
        raise ValueError("preregistered protocol must be a JSON object")

    expected = canonical_ppo_global_btc_regime_protocol().to_payload()
    if set(decoded) != set(expected):
        raise ValueError("preregistered protocol keys are unknown or missing")
    if raw != canonical_json_bytes(decoded):
        raise ValueError("preregistered protocol must use canonical JSON bytes")
    if raw != canonical_json_bytes(expected):
        raise ValueError("protocol values differ from preregistered authority")
    return canonical_ppo_global_btc_regime_protocol()


__all__ = [
    "PPOGlobalBTCRegimeProtocol",
    "canonical_ppo_global_btc_regime_protocol",
    "load_ppo_global_btc_regime_protocol",
]
