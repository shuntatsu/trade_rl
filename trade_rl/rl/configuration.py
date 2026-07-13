"""Content-addressed environment experiment manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketCalendarKind
from trade_rl.domain.common import require_sha256
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.rl.actions import ACTION_SCHEMA, ActionSpec, AlphaContract
from trade_rl.rl.environment import ResidualMarketEnvConfig
from trade_rl.rl.observations import OBSERVATION_SCHEMA
from trade_rl.rl.rewards import REWARD_SCHEMA, RewardConfig
from trade_rl.strategies.trend import TrendConfig


@dataclass(frozen=True, slots=True)
class EnvironmentExperimentManifest:
    """One complete, reproducible environment and training-market contract."""

    digest: str
    calendar_kind: MarketCalendarKind | str
    action_spec: ActionSpec
    alpha_contract: AlphaContract
    environment: ResidualMarketEnvConfig
    risk: PreTradeRiskConfig
    reward: RewardConfig
    trend: TrendConfig
    alpha_artifact_digest: str | None = None
    factor_artifact_digest: str | None = None
    normalizer_digest: str | None = None
    schema_version: str = "environment_experiment_manifest_v1"

    def __post_init__(self) -> None:
        calendar = MarketCalendarKind(self.calendar_kind)
        object.__setattr__(self, "calendar_kind", calendar)
        for field_name, value in (
            ("alpha_artifact_digest", self.alpha_artifact_digest),
            ("factor_artifact_digest", self.factor_artifact_digest),
            ("normalizer_digest", self.normalizer_digest),
        ):
            if value is not None:
                require_sha256(value, field=field_name)
        if self.action_spec.alpha_enabled != (self.alpha_artifact_digest is not None):
            raise ValueError("alpha action and alpha artifact identity must agree")
        if self.action_spec.n_factors > 0 and self.factor_artifact_digest is None:
            raise ValueError("factor actions require factor_artifact_digest")
        if self.action_spec.n_factors == 0 and self.factor_artifact_digest is not None:
            raise ValueError("factor_artifact_digest requires factor actions")
        if self.environment.resolved_reward_config() != self.reward:
            raise ValueError("environment and manifest reward configurations differ")
        expected = content_digest(self.digest_payload())
        if self.digest != expected:
            raise ValueError("environment experiment digest does not match content")

    def digest_payload(self) -> dict[str, object]:
        return {
            "action_schema": ACTION_SCHEMA,
            "action_spec": asdict(self.action_spec),
            "alpha_artifact_digest": self.alpha_artifact_digest,
            "alpha_contract": asdict(self.alpha_contract),
            "calendar_kind": MarketCalendarKind(self.calendar_kind).value,
            "environment": asdict(self.environment),
            "factor_artifact_digest": self.factor_artifact_digest,
            "normalizer_digest": self.normalizer_digest,
            "observation_schema": OBSERVATION_SCHEMA,
            "reward": asdict(self.reward),
            "reward_schema": REWARD_SCHEMA,
            "risk": asdict(self.risk),
            "schema_version": self.schema_version,
            "trend": asdict(self.trend),
        }

    @classmethod
    def build(
        cls,
        *,
        calendar_kind: MarketCalendarKind | str,
        action_spec: ActionSpec,
        alpha_contract: AlphaContract,
        environment: ResidualMarketEnvConfig,
        risk: PreTradeRiskConfig,
        reward: RewardConfig,
        trend: TrendConfig,
        alpha_artifact_digest: str | None = None,
        factor_artifact_digest: str | None = None,
        normalizer_digest: str | None = None,
    ) -> EnvironmentExperimentManifest:
        temporary = {
            "action_schema": ACTION_SCHEMA,
            "action_spec": asdict(action_spec),
            "alpha_artifact_digest": alpha_artifact_digest,
            "alpha_contract": asdict(alpha_contract),
            "calendar_kind": MarketCalendarKind(calendar_kind).value,
            "environment": asdict(environment),
            "factor_artifact_digest": factor_artifact_digest,
            "normalizer_digest": normalizer_digest,
            "observation_schema": OBSERVATION_SCHEMA,
            "reward": asdict(reward),
            "reward_schema": REWARD_SCHEMA,
            "risk": asdict(risk),
            "schema_version": "environment_experiment_manifest_v1",
            "trend": asdict(trend),
        }
        return cls(
            digest=content_digest(temporary),
            calendar_kind=calendar_kind,
            action_spec=action_spec,
            alpha_contract=alpha_contract,
            environment=environment,
            risk=risk,
            reward=reward,
            trend=trend,
            alpha_artifact_digest=alpha_artifact_digest,
            factor_artifact_digest=factor_artifact_digest,
            normalizer_digest=normalizer_digest,
        )
