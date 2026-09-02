"""Frozen artifact identity for Universal Trade RL U1 semantics and economics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.rl.universal_normalization import UniversalTradeSequenceNormalizer
from trade_rl.rl.universal_trade_environment import UniversalTradeEnvironment
from trade_rl.rl.universal_trade_observation import UniversalTradeObservationBuilder
from trade_rl.workflows.universal_trade_rl_data_provenance import (
    UniversalTradeRLFitProvenance,
    UniversalTradeRLFitPurpose,
    require_universal_trade_rl_train_only_provenance,
)
from trade_rl.workflows.universal_trade_rl_run_identity import (
    UniversalTradeRLRunIdentity,
    UniversalTradeRLRunStage,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
)

UNIVERSAL_TRADE_RL_U1_CONTRACT_SCHEMA: Final = "universal_trade_rl_u1_contract_v1"
U1_PRODUCTION_STATUS: Final = "NO-GO"
U1_NORMALIZER_CLIP_VALUE: Final = 10.0

_U1_CONTRACT_KEYS: Final = (
    "schema_version",
    "production_status",
    "universe_manifest_digest",
    "u0_identity_digest",
    "policy_contract_digest",
    "normalizer_digest",
    "normalizer_provenance_digest",
    "normalizer_knowledge_cutoff_ns",
    "normalizer_clip_value",
    "observation_schema_digest",
    "state_layout_digest",
    "policy_state_fields",
    "runtime_config_digest",
    "execution_policy_digest",
    "pretrade_risk_digest",
    "portfolio_risk_digest",
    "artifact_digest",
)


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("U1 contract must be an object with exact keys")
    result = {str(key): item for key, item in value.items()}
    if set(result) != set(_U1_CONTRACT_KEYS) or len(result) != len(_U1_CONTRACT_KEYS):
        raise ValueError("U1 contract must use exact keys")
    return result


def _require_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a SHA-256 string")
    return require_sha256(value, field=field)


def _policy_state_fields(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("U1 policy-state fields must be an array")
    result = tuple(value)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise ValueError("U1 policy-state fields must be non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError("U1 policy-state fields must be unique")
    return result


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _normalizer_clip(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("U1 normalizer clip value must be numeric")
    resolved = float(value)
    if resolved != U1_NORMALIZER_CLIP_VALUE:
        raise ValueError(
            f"U1 normalizer clip value must be exactly {U1_NORMALIZER_CLIP_VALUE}"
        )
    return resolved


def _runtime_config_digest(environment: UniversalTradeEnvironment) -> str:
    base = environment.base_env
    config_payload = asdict(base.config)
    config_payload["episode_boundary_mode"] = base.config.episode_boundary_mode.value
    config_payload["action_validation_mode"] = base.config.action_validation_mode.value
    config_payload["resolved_sequence_windows"] = base.config.resolved_sequence_windows
    config_payload["resolved_reward_config"] = asdict(
        base.config.resolved_reward_config()
    )
    return content_digest(
        {
            "action_spec_digest": environment.action_spec_digest,
            "config": config_payload,
            "policy_contract_digest": environment.contract.digest,
            "schema_version": "universal_trade_rl_u1_runtime_config_v1",
        }
    )


def _pretrade_risk_digest(environment: UniversalTradeEnvironment) -> str:
    return content_digest(
        {
            "config": asdict(environment.base_env.pre_trade_risk.config),
            "schema_version": "universal_trade_rl_u1_pretrade_risk_v1",
        }
    )


def _portfolio_risk_digest(environment: UniversalTradeEnvironment) -> str:
    return content_digest(
        {
            "config": asdict(environment.base_env.portfolio_risk.config),
            "implementation_digest": environment.base_env.portfolio_risk.implementation_digest,
            "schema_version": "universal_trade_rl_u1_portfolio_risk_v1",
        }
    )


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU1Contract:
    """Content-addressed U1 identity consumed by later Base-RL stages."""

    universe_manifest_digest: str
    u0_identity_digest: str
    policy_contract_digest: str
    normalizer_digest: str
    normalizer_provenance_digest: str
    normalizer_knowledge_cutoff_ns: int
    normalizer_clip_value: float
    observation_schema_digest: str
    state_layout_digest: str
    policy_state_fields: tuple[str, ...]
    runtime_config_digest: str
    execution_policy_digest: str
    pretrade_risk_digest: str
    portfolio_risk_digest: str
    production_status: str = U1_PRODUCTION_STATUS
    schema_version: str = UNIVERSAL_TRADE_RL_U1_CONTRACT_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != UNIVERSAL_TRADE_RL_U1_CONTRACT_SCHEMA:
            raise ValueError("unsupported Universal Trade RL U1 contract schema")
        if self.production_status != U1_PRODUCTION_STATUS:
            raise ValueError(
                "Universal Trade RL U1 production status must remain NO-GO"
            )
        for field_name in (
            "universe_manifest_digest",
            "u0_identity_digest",
            "policy_contract_digest",
            "normalizer_digest",
            "normalizer_provenance_digest",
            "observation_schema_digest",
            "state_layout_digest",
            "runtime_config_digest",
            "execution_policy_digest",
            "pretrade_risk_digest",
            "portfolio_risk_digest",
        ):
            require_sha256(getattr(self, field_name), field=f"U1 {field_name}")
        object.__setattr__(
            self,
            "normalizer_knowledge_cutoff_ns",
            _positive_integer(
                self.normalizer_knowledge_cutoff_ns,
                field="U1 normalizer knowledge cutoff",
            ),
        )
        object.__setattr__(
            self,
            "normalizer_clip_value",
            _normalizer_clip(self.normalizer_clip_value),
        )
        object.__setattr__(
            self,
            "policy_state_fields",
            _policy_state_fields(self.policy_state_fields),
        )

        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U1 artifact digest")
            if self.digest != expected:
                raise ValueError("Universal Trade RL U1 contract digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "production_status": self.production_status,
            "universe_manifest_digest": self.universe_manifest_digest,
            "u0_identity_digest": self.u0_identity_digest,
            "policy_contract_digest": self.policy_contract_digest,
            "normalizer_digest": self.normalizer_digest,
            "normalizer_provenance_digest": self.normalizer_provenance_digest,
            "normalizer_knowledge_cutoff_ns": self.normalizer_knowledge_cutoff_ns,
            "normalizer_clip_value": self.normalizer_clip_value,
            "observation_schema_digest": self.observation_schema_digest,
            "state_layout_digest": self.state_layout_digest,
            "policy_state_fields": self.policy_state_fields,
            "runtime_config_digest": self.runtime_config_digest,
            "execution_policy_digest": self.execution_policy_digest,
            "pretrade_risk_digest": self.pretrade_risk_digest,
            "portfolio_risk_digest": self.portfolio_risk_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> UniversalTradeRLU1Contract:
        values = _mapping(payload)
        if values["schema_version"] != UNIVERSAL_TRADE_RL_U1_CONTRACT_SCHEMA:
            raise ValueError("unsupported Universal Trade RL U1 contract schema")
        if values["production_status"] != U1_PRODUCTION_STATUS:
            raise ValueError(
                "Universal Trade RL U1 production status must remain NO-GO"
            )

        fields = _policy_state_fields(values["policy_state_fields"])
        return cls(
            universe_manifest_digest=_require_digest(
                values["universe_manifest_digest"], field="U1 universe manifest digest"
            ),
            u0_identity_digest=_require_digest(
                values["u0_identity_digest"], field="U1 U0 identity digest"
            ),
            policy_contract_digest=_require_digest(
                values["policy_contract_digest"], field="U1 policy contract digest"
            ),
            normalizer_digest=_require_digest(
                values["normalizer_digest"], field="U1 normalizer digest"
            ),
            normalizer_provenance_digest=_require_digest(
                values["normalizer_provenance_digest"],
                field="U1 normalizer provenance digest",
            ),
            normalizer_knowledge_cutoff_ns=_positive_integer(
                values["normalizer_knowledge_cutoff_ns"],
                field="U1 normalizer knowledge cutoff",
            ),
            normalizer_clip_value=_normalizer_clip(values["normalizer_clip_value"]),
            observation_schema_digest=_require_digest(
                values["observation_schema_digest"],
                field="U1 observation schema digest",
            ),
            state_layout_digest=_require_digest(
                values["state_layout_digest"], field="U1 state layout digest"
            ),
            policy_state_fields=fields,
            runtime_config_digest=_require_digest(
                values["runtime_config_digest"], field="U1 runtime config digest"
            ),
            execution_policy_digest=_require_digest(
                values["execution_policy_digest"], field="U1 execution policy digest"
            ),
            pretrade_risk_digest=_require_digest(
                values["pretrade_risk_digest"], field="U1 pretrade risk digest"
            ),
            portfolio_risk_digest=_require_digest(
                values["portfolio_risk_digest"], field="U1 portfolio risk digest"
            ),
            production_status=U1_PRODUCTION_STATUS,
            schema_version=UNIVERSAL_TRADE_RL_U1_CONTRACT_SCHEMA,
            digest=_require_digest(
                values["artifact_digest"], field="U1 artifact digest"
            ),
        )


def build_universal_trade_rl_u1_contract(
    *,
    manifest: UniversalTradeRLUniverseManifest,
    u0_identity: UniversalTradeRLRunIdentity,
    normalizer_provenance: UniversalTradeRLFitProvenance,
    environment: UniversalTradeEnvironment,
) -> UniversalTradeRLU1Contract:
    """Validate and bind one frozen U0/U1 generation without opening Admission."""

    if not isinstance(manifest, UniversalTradeRLUniverseManifest):
        raise TypeError("U1 manifest is invalid")
    if not isinstance(u0_identity, UniversalTradeRLRunIdentity):
        raise TypeError("U1 U0 identity is invalid")
    if not isinstance(normalizer_provenance, UniversalTradeRLFitProvenance):
        raise TypeError("U1 normalizer provenance is invalid")
    if not isinstance(environment, UniversalTradeEnvironment):
        raise TypeError("U1 environment is invalid")

    if u0_identity.stage is not UniversalTradeRLRunStage.UNIVERSE_MATERIALIZATION:
        raise ValueError("U1 requires the U0 universe materialization stage identity")
    if u0_identity.universe_manifest_digest != manifest.digest:
        raise ValueError("U1 U0 identity universe manifest mismatch")

    provenance = require_universal_trade_rl_train_only_provenance(
        normalizer_provenance,
        manifest=manifest,
    )
    if provenance.purpose is not UniversalTradeRLFitPurpose.FEATURE_NORMALIZATION:
        raise ValueError(
            "U1 normalizer provenance purpose must be feature normalization"
        )

    normalizer = environment.sequence_normalizer
    if not isinstance(normalizer, UniversalTradeSequenceNormalizer):
        raise ValueError("U1 contract requires a frozen sequence normalizer")
    if normalizer.universe_manifest_digest != manifest.digest:
        raise ValueError("U1 normalizer universe manifest mismatch")
    if normalizer.provenance_digest != provenance.digest:
        raise ValueError("U1 normalizer provenance mismatch")
    if normalizer.source_dataset_digests != provenance.source_dataset_digests:
        raise ValueError("U1 normalizer source identity mismatch")
    if normalizer.knowledge_cutoff_ns != provenance.knowledge_cutoff:
        raise ValueError("U1 normalizer knowledge cutoff mismatch")
    if normalizer.contract_digest != environment.contract.digest:
        raise ValueError("U1 normalizer policy contract mismatch")
    if normalizer.clip_value != U1_NORMALIZER_CLIP_VALUE:
        raise ValueError(
            f"U1 normalizer clip value must be frozen at {U1_NORMALIZER_CLIP_VALUE}"
        )

    observation = UniversalTradeObservationBuilder(
        contract=environment.contract,
        normalizer=normalizer,
    )
    return UniversalTradeRLU1Contract(
        universe_manifest_digest=manifest.digest,
        u0_identity_digest=u0_identity.digest,
        policy_contract_digest=environment.contract.digest,
        normalizer_digest=normalizer.digest,
        normalizer_provenance_digest=provenance.digest,
        normalizer_knowledge_cutoff_ns=normalizer.knowledge_cutoff_ns,
        normalizer_clip_value=normalizer.clip_value,
        observation_schema_digest=observation.schema_digest,
        state_layout_digest=observation.state_layout_digest,
        policy_state_fields=observation.policy_state_fields,
        runtime_config_digest=_runtime_config_digest(environment),
        execution_policy_digest=environment.base_env.execution_policy_digest,
        pretrade_risk_digest=_pretrade_risk_digest(environment),
        portfolio_risk_digest=_portfolio_risk_digest(environment),
    )


__all__ = [
    "UNIVERSAL_TRADE_RL_U1_CONTRACT_SCHEMA",
    "U1_NORMALIZER_CLIP_VALUE",
    "U1_PRODUCTION_STATUS",
    "UniversalTradeRLU1Contract",
    "build_universal_trade_rl_u1_contract",
]
