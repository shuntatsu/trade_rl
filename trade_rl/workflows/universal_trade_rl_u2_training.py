"""Fixed per-seed Base PPO training identity for Universal Trade RL U2."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.integrations.sb3_training import StableBaselines3PPOBackend
from trade_rl.rl.checkpointing import CheckpointManifest
from trade_rl.rl.training import PolicyTrainingResult, ResidualTrainingConfig
from trade_rl.workflows.universal_trade_rl_u2_contract import (
    U2_FINAL_TIMESTEPS,
    UniversalTradeRLU2Contract,
    build_universal_trade_rl_u2_training_config,
)
from trade_rl.workflows.universal_trade_rl_u2_environment import (
    UniversalTradeRLU2EnvironmentFactory,
)
from trade_rl.workflows.universal_trade_rl_u2_preflight import U2TrainingSourceClosure

U2_SEED_TRAINING_PLAN_SCHEMA: Final = "universal_trade_rl_u2_seed_training_plan_v1"


class U2TrainingBackend(Protocol):
    """Minimal backend surface required by one fixed U2 PPO member."""

    def train(
        self,
        *,
        seed: int,
        config: ResidualTrainingConfig,
        output_path: Path,
    ) -> PolicyTrainingResult: ...


U2TrainingBackendFactory = Callable[
    [UniversalTradeRLU2EnvironmentFactory],
    U2TrainingBackend,
]


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2SeedTrainingPlan:
    """Content-addressed identity for one preregistered U2 seed run."""

    u2_contract_digest: str
    source_closure_digest: str
    u1_contract_digest: str
    normalizer_digest: str
    time_partition_digest: str
    training_config_digest: str
    seed: int
    final_timesteps: int
    primary_candidate: bool
    production_status: str
    schema_version: str = U2_SEED_TRAINING_PLAN_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_SEED_TRAINING_PLAN_SCHEMA:
            raise ValueError(
                "unsupported Universal Trade RL U2 seed training plan schema"
            )
        for field_name, value in (
            ("u2_contract_digest", self.u2_contract_digest),
            ("source_closure_digest", self.source_closure_digest),
            ("u1_contract_digest", self.u1_contract_digest),
            ("normalizer_digest", self.normalizer_digest),
            ("time_partition_digest", self.time_partition_digest),
            ("training_config_digest", self.training_config_digest),
        ):
            require_sha256(value, field=f"U2 seed training plan {field_name}")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError(
                "U2 seed training plan seed must be a non-negative integer"
            )
        if (
            isinstance(self.final_timesteps, bool)
            or not isinstance(self.final_timesteps, int)
            or self.final_timesteps != U2_FINAL_TIMESTEPS
        ):
            raise ValueError("U2 seed training plan must use the fixed final timestep")
        if not isinstance(self.primary_candidate, bool):
            raise ValueError("U2 seed training plan primary_candidate must be boolean")
        if self.production_status != "NO-GO":
            raise ValueError("Universal Trade RL U2 remains Production NO-GO")

        expected = content_digest(self.digest_payload())
        if self.digest:
            require_sha256(self.digest, field="U2 seed training plan artifact digest")
            if self.digest != expected:
                raise ValueError("U2 seed training plan digest mismatch")
        object.__setattr__(self, "digest", expected)

    def digest_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "u2_contract_digest": self.u2_contract_digest,
            "source_closure_digest": self.source_closure_digest,
            "u1_contract_digest": self.u1_contract_digest,
            "normalizer_digest": self.normalizer_digest,
            "time_partition_digest": self.time_partition_digest,
            "training_config_digest": self.training_config_digest,
            "seed": self.seed,
            "final_timesteps": self.final_timesteps,
            "primary_candidate": self.primary_candidate,
            "production_status": self.production_status,
        }


def build_universal_trade_rl_u2_seed_training_plan(
    *,
    contract: UniversalTradeRLU2Contract,
    source_closure: U2TrainingSourceClosure,
    seed: int,
) -> UniversalTradeRLU2SeedTrainingPlan:
    """Bind one allowed seed to the exact frozen U2 contract and FIT source closure."""

    if not isinstance(contract, UniversalTradeRLU2Contract):
        raise TypeError("U2 seed training plan requires a UniversalTradeRLU2Contract")
    if not isinstance(source_closure, U2TrainingSourceClosure):
        raise TypeError("U2 seed training plan requires a U2TrainingSourceClosure")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("U2 training seed must be an integer and not boolean")
    if seed not in contract.training_seeds:
        raise ValueError("U2 training seed is outside the preregistered seed closure")

    if source_closure.u2_contract_digest != contract.digest:
        raise ValueError("U2 source closure contract identity mismatch")
    if source_closure.universe_manifest_digest != contract.universe_manifest_digest:
        raise ValueError("U2 source closure universe identity mismatch")
    if source_closure.u1_contract_digest != contract.u1_contract_digest:
        raise ValueError("U2 source closure U1 identity mismatch")
    if source_closure.normalizer_digest != contract.u1_normalizer_digest:
        raise ValueError("U2 source closure normalizer identity mismatch")
    if source_closure.time_partition_digest != contract.time_partition_digest:
        raise ValueError("U2 source closure time-partition identity mismatch")
    if source_closure.fit_last_timestamp_ns != contract.fit_end_ns:
        raise ValueError("U2 source closure FIT end mismatch")

    config = build_universal_trade_rl_u2_training_config()
    training_config_digest = content_digest(config.digest_payload())
    if training_config_digest != contract.training_config_digest:
        raise ValueError("U2 training config identity mismatch")
    if config.timesteps != U2_FINAL_TIMESTEPS:
        raise ValueError("U2 training config final timestep drifted")

    return UniversalTradeRLU2SeedTrainingPlan(
        u2_contract_digest=contract.digest,
        source_closure_digest=source_closure.digest,
        u1_contract_digest=contract.u1_contract_digest,
        normalizer_digest=contract.u1_normalizer_digest,
        time_partition_digest=contract.time_partition_digest,
        training_config_digest=contract.training_config_digest,
        seed=seed,
        final_timesteps=config.timesteps,
        primary_candidate=seed == contract.primary_candidate_seed,
        production_status=contract.production_status,
    )


def train_universal_trade_rl_u2_seed(
    *,
    plan: UniversalTradeRLU2SeedTrainingPlan,
    environment_factory: UniversalTradeRLU2EnvironmentFactory,
    output_path: Path,
    backend_factory: U2TrainingBackendFactory = StableBaselines3PPOBackend,
) -> PolicyTrainingResult:
    """Train exactly one preregistered U2 PPO member under its frozen lineage."""

    if not isinstance(plan, UniversalTradeRLU2SeedTrainingPlan):
        raise TypeError("U2 training requires a seed training plan")
    if environment_factory.run_seed != plan.seed:
        raise ValueError("U2 training environment member seed mismatch")
    if environment_factory.source_closure_digest != plan.source_closure_digest:
        raise ValueError("U2 training environment source closure mismatch")

    config = build_universal_trade_rl_u2_training_config()
    if content_digest(config.digest_payload()) != plan.training_config_digest:
        raise ValueError("U2 training plan configuration mismatch")

    probe = environment_factory()
    try:
        if (
            probe.environment_digest
            != environment_factory.environment_generation_digest
        ):
            raise ValueError("U2 training environment generation mismatch")
    finally:
        probe.close()

    backend = backend_factory(environment_factory)
    result = backend.train(
        seed=plan.seed,
        config=config,
        output_path=output_path,
    )
    if result.environment_digest != environment_factory.environment_generation_digest:
        raise ValueError("U2 training result environment mismatch")
    if result.actual_timesteps != plan.final_timesteps:
        raise ValueError("U2 training result timestep mismatch")
    return result


def require_universal_trade_rl_u2_selection_checkpoint(
    *,
    plan: UniversalTradeRLU2SeedTrainingPlan,
    checkpoint: CheckpointManifest,
    expected_environment_digest: str,
) -> CheckpointManifest:
    """Accept only the exact fixed-budget checkpoint for one frozen U2 seed plan."""

    if not isinstance(plan, UniversalTradeRLU2SeedTrainingPlan):
        raise TypeError("U2 selection checkpoint requires a seed training plan")
    if not isinstance(checkpoint, CheckpointManifest):
        raise TypeError("U2 selection checkpoint requires a CheckpointManifest")
    require_sha256(
        expected_environment_digest,
        field="U2 selection expected environment digest",
    )
    if checkpoint.algorithm != "ppo":
        raise ValueError("U2 selection checkpoint algorithm mismatch")
    if checkpoint.seed != plan.seed:
        raise ValueError("U2 selection checkpoint seed mismatch")
    if checkpoint.training_config_digest != plan.training_config_digest:
        raise ValueError("U2 selection checkpoint training config mismatch")
    if checkpoint.environment_digest != expected_environment_digest:
        raise ValueError("U2 selection checkpoint environment mismatch")
    if (
        checkpoint.requested_timestep != plan.final_timesteps
        or checkpoint.observed_timestep != plan.final_timesteps
    ):
        raise ValueError("U2 selection checkpoint must be the exact final timestep")
    return checkpoint


__all__ = [
    "U2_SEED_TRAINING_PLAN_SCHEMA",
    "U2TrainingBackend",
    "U2TrainingBackendFactory",
    "UniversalTradeRLU2SeedTrainingPlan",
    "build_universal_trade_rl_u2_seed_training_plan",
    "require_universal_trade_rl_u2_selection_checkpoint",
    "train_universal_trade_rl_u2_seed",
]
