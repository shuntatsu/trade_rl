from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest

from tests.workflows.test_universal_trade_rl_u2_contract import _fixture
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLSymbolRole
from trade_rl.rl.checkpointing import (
    CHECKPOINT_POLICY_NAME,
    CheckpointManifest,
)
from trade_rl.rl.training import PolicyTrainingResult, ResidualTrainingConfig
from trade_rl.workflows.universal_trade_rl_u2_contract import (
    U2_FINAL_TIMESTEPS,
    UniversalTradeRLU2Contract,
    build_universal_trade_rl_u2_contract,
    build_universal_trade_rl_u2_training_config,
)
from trade_rl.workflows.universal_trade_rl_u2_environment import (
    UniversalTradeRLU2EnvironmentFactory,
)
from trade_rl.workflows.universal_trade_rl_u2_preflight import (
    U2TrainingSource,
    U2TrainingSourceClosure,
)
from trade_rl.workflows.universal_trade_rl_u2_training import (
    UniversalTradeRLU2SeedTrainingPlan,
)

_ENVIRONMENT_DIGEST = content_digest({"fixture": "u2-training-environment"})


@dataclass(frozen=True, slots=True)
class U2TrainingRuntimeFixture:
    contract: UniversalTradeRLU2Contract
    closure: U2TrainingSourceClosure


def _module():
    try:
        return importlib.import_module(
            "trade_rl.workflows.universal_trade_rl_u2_training"
        )
    except ModuleNotFoundError:
        pytest.fail("Universal Trade RL U2 training orchestration is not implemented")


def _training_runner():
    runner = getattr(_module(), "train_universal_trade_rl_u2_seed", None)
    if not callable(runner):
        pytest.fail("U2 fixed member training orchestration is not implemented")
    return runner


def _runtime_fixture() -> U2TrainingRuntimeFixture:
    base = _fixture()
    contract = build_universal_trade_rl_u2_contract(
        manifest=base.manifest,
        u1_contract=base.u1_contract,
        time_partition=base.time_partition,
        rl_training_provenance=base.rl_training_provenance,
        training_config=build_universal_trade_rl_u2_training_config(),
    )
    fit = base.time_partition.window("fit")
    entries = tuple(
        sorted(
            (
                entry
                for entry in base.manifest.entries
                if entry.role is UniversalTradeRLSymbolRole.TRAIN
            ),
            key=lambda entry: entry.symbol,
        )
    )
    closure = U2TrainingSourceClosure(
        u2_contract_digest=contract.digest,
        universe_manifest_digest=contract.universe_manifest_digest,
        u1_contract_digest=contract.u1_contract_digest,
        normalizer_digest=contract.u1_normalizer_digest,
        normalizer_provenance_digest=base.u1_contract.normalizer_provenance_digest,
        time_partition_digest=contract.time_partition_digest,
        fit_first_timestamp_ns=fit.first_timestamp_ns,
        fit_last_timestamp_ns=fit.last_timestamp_ns,
        fit_stop_timestamp_ns_exclusive=(
            fit.last_timestamp_ns + base.time_partition.decision_step_ns
        ),
        fit_bar_count=fit.bar_count,
        sources=tuple(
            U2TrainingSource(
                symbol=entry.symbol,
                dataset_digest=entry.dataset_digest,
                source_first_timestamp_ns=entry.first_timestamp_ns,
                source_last_timestamp_ns=entry.last_timestamp_ns,
                source_row_count=entry.row_count,
                fit_first_timestamp_ns=fit.first_timestamp_ns,
                fit_last_timestamp_ns=fit.last_timestamp_ns,
                fit_stop_timestamp_ns_exclusive=(
                    fit.last_timestamp_ns + base.time_partition.decision_step_ns
                ),
                fit_bar_count=fit.bar_count,
            )
            for entry in entries
        ),
    )
    return U2TrainingRuntimeFixture(contract=contract, closure=closure)


def _plan(fixture: U2TrainingRuntimeFixture):
    return _module().build_universal_trade_rl_u2_seed_training_plan(
        contract=fixture.contract,
        source_closure=fixture.closure,
        seed=0,
    )


def _checkpoint(*, plan, timestep: int) -> CheckpointManifest:
    policy_digest = content_digest({"fixture": "u2-policy", "timestep": timestep})
    payload = {
        "algorithm": "ppo",
        "environment_digest": _ENVIRONMENT_DIGEST,
        "observed_timestep": timestep,
        "policy_digest": policy_digest,
        "policy_file": CHECKPOINT_POLICY_NAME,
        "requested_timestep": timestep,
        "schema_version": "policy_checkpoint_v1",
        "seed": plan.seed,
        "training_config_digest": plan.training_config_digest,
    }
    return CheckpointManifest(
        digest=content_digest(payload),
        algorithm="ppo",
        seed=plan.seed,
        requested_timestep=timestep,
        observed_timestep=timestep,
        environment_digest=_ENVIRONMENT_DIGEST,
        training_config_digest=plan.training_config_digest,
        policy_digest=policy_digest,
        policy_path=Path("policy.zip"),
    )


class _Probe:
    def __init__(self, *, environment_digest: str, owner: _Factory) -> None:
        self.environment_digest = environment_digest
        self._owner = owner

    def close(self) -> None:
        self._owner.probe_close_calls += 1


class _Factory:
    def __init__(
        self,
        *,
        source_closure_digest: str,
        run_seed: int = 0,
        environment_generation_digest: str = _ENVIRONMENT_DIGEST,
        probe_environment_digest: str | None = None,
    ) -> None:
        self.run_seed = run_seed
        self.source_closure_digest = source_closure_digest
        self.environment_generation_digest = environment_generation_digest
        self.probe_environment_digest = (
            environment_generation_digest
            if probe_environment_digest is None
            else probe_environment_digest
        )
        self.probe_calls = 0
        self.probe_close_calls = 0

    def __call__(self) -> _Probe:
        self.probe_calls += 1
        return _Probe(
            environment_digest=self.probe_environment_digest,
            owner=self,
        )


def _as_u2_factory(factory: _Factory) -> UniversalTradeRLU2EnvironmentFactory:
    return cast(UniversalTradeRLU2EnvironmentFactory, factory)


def _result(
    *,
    output_path: Path,
    environment_digest: str,
    actual_timesteps: int = U2_FINAL_TIMESTEPS,
) -> PolicyTrainingResult:
    return PolicyTrainingResult(
        checkpoint_path=output_path,
        actual_timesteps=actual_timesteps,
        resolved_device="cpu",
        environment_digest=environment_digest,
        initial_capital=100_000.0,
        action_size=1,
        action_names=("target_weight:INSTRUMENT",),
        action_spec_digest=content_digest({"fixture": "u2-action-spec"}),
        observation_size=1,
    )


class _Backend:
    def __init__(self, result: PolicyTrainingResult) -> None:
        self.result = result
        self.calls: list[tuple[int, ResidualTrainingConfig, Path]] = []

    def train(
        self,
        *,
        seed: int,
        config: ResidualTrainingConfig,
        output_path: Path,
    ) -> PolicyTrainingResult:
        self.calls.append((seed, config, output_path))
        return self.result


class _BackendFactory:
    def __init__(self, backend: _Backend) -> None:
        self.backend = backend
        self.calls: list[Any] = []

    def __call__(self, environment_factory: Any) -> _Backend:
        self.calls.append(environment_factory)
        return self.backend


def test_u2_seed_training_plan_binds_contract_source_closure_and_fixed_budget() -> None:
    fixture = _runtime_fixture()
    plan = _plan(fixture)

    assert plan.u2_contract_digest == fixture.contract.digest
    assert plan.source_closure_digest == fixture.closure.digest
    assert plan.u1_contract_digest == fixture.contract.u1_contract_digest
    assert plan.normalizer_digest == fixture.contract.u1_normalizer_digest
    assert plan.time_partition_digest == fixture.contract.time_partition_digest
    assert plan.training_config_digest == fixture.contract.training_config_digest
    assert plan.seed == 0
    assert plan.final_timesteps == 524_288
    assert plan.primary_candidate is True
    assert plan.production_status == "NO-GO"
    assert len(plan.digest) == 64


def test_u2_selection_checkpoint_requires_exact_final_fixed_budget() -> None:
    fixture = _runtime_fixture()
    module = _module()
    plan = _plan(fixture)
    validator = getattr(
        module,
        "require_universal_trade_rl_u2_selection_checkpoint",
        None,
    )
    assert callable(validator), "U2 final-only checkpoint validator is not implemented"

    final = _checkpoint(plan=plan, timestep=524_288)
    assert (
        validator(
            plan=plan,
            checkpoint=final,
            expected_environment_digest=_ENVIRONMENT_DIGEST,
        )
        is final
    )

    for timestep in (32_768, 262_144):
        with pytest.raises(ValueError, match="final|timestep|checkpoint"):
            validator(
                plan=plan,
                checkpoint=_checkpoint(plan=plan, timestep=timestep),
                expected_environment_digest=_ENVIRONMENT_DIGEST,
            )


def test_u2_training_orchestrates_one_exact_member(tmp_path: Path) -> None:
    fixture = _runtime_fixture()
    plan = _plan(fixture)
    factory = _Factory(source_closure_digest=fixture.closure.digest)
    output_path = tmp_path / "seed-0.zip"
    expected_result = _result(
        output_path=output_path,
        environment_digest=factory.environment_generation_digest,
    )
    backend = _Backend(expected_result)
    backend_factory = _BackendFactory(backend)

    result = _training_runner()(
        plan=plan,
        environment_factory=_as_u2_factory(factory),
        output_path=output_path,
        backend_factory=backend_factory,
    )

    assert result is expected_result
    assert factory.probe_calls == 1
    assert factory.probe_close_calls == 1
    assert backend_factory.calls == [factory]
    assert len(backend.calls) == 1
    seed, config, observed_output = backend.calls[0]
    assert seed == plan.seed
    assert content_digest(config.digest_payload()) == plan.training_config_digest
    assert observed_output == output_path


def test_u2_training_rejects_lineage_drift_before_backend_creation(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture()
    plan = _plan(fixture)
    cases = (
        (
            _Factory(source_closure_digest=fixture.closure.digest, run_seed=1),
            plan,
            "seed",
        ),
        (
            _Factory(source_closure_digest=content_digest({"fixture": "wrong-closure"})),
            plan,
            "source|closure",
        ),
        (
            _Factory(source_closure_digest=fixture.closure.digest),
            replace(
                plan,
                training_config_digest=content_digest(
                    {"fixture": "wrong-training-config"}
                ),
                digest="",
            ),
            "config|configuration",
        ),
    )

    for index, (factory, candidate_plan, message) in enumerate(cases):
        output_path = tmp_path / f"invalid-{index}.zip"
        backend = _Backend(
            _result(
                output_path=output_path,
                environment_digest=factory.environment_generation_digest,
            )
        )
        backend_factory = _BackendFactory(backend)

        with pytest.raises(ValueError, match=message):
            _training_runner()(
                plan=candidate_plan,
                environment_factory=_as_u2_factory(factory),
                output_path=output_path,
                backend_factory=backend_factory,
            )

        assert backend_factory.calls == []
        assert backend.calls == []
        assert factory.probe_calls == 0


def test_u2_training_rejects_probe_generation_mismatch_and_closes_probe(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture()
    factory = _Factory(
        source_closure_digest=fixture.closure.digest,
        probe_environment_digest=content_digest({"fixture": "wrong-probe-generation"}),
    )
    output_path = tmp_path / "policy.zip"
    backend = _Backend(
        _result(
            output_path=output_path,
            environment_digest=factory.environment_generation_digest,
        )
    )
    backend_factory = _BackendFactory(backend)

    with pytest.raises(ValueError, match="environment|generation"):
        _training_runner()(
            plan=_plan(fixture),
            environment_factory=_as_u2_factory(factory),
            output_path=output_path,
            backend_factory=backend_factory,
        )

    assert factory.probe_calls == 1
    assert factory.probe_close_calls == 1
    assert backend_factory.calls == []
    assert backend.calls == []


@pytest.mark.parametrize(
    ("result_environment_digest", "result_timesteps", "message"),
    (
        (
            content_digest({"fixture": "wrong-result-generation"}),
            U2_FINAL_TIMESTEPS,
            "environment",
        ),
        (_ENVIRONMENT_DIGEST, U2_FINAL_TIMESTEPS - 1, "timestep|timesteps"),
    ),
)
def test_u2_training_rejects_backend_result_identity_drift(
    tmp_path: Path,
    result_environment_digest: str,
    result_timesteps: int,
    message: str,
) -> None:
    fixture = _runtime_fixture()
    plan = _plan(fixture)
    factory = _Factory(source_closure_digest=fixture.closure.digest)
    output_path = tmp_path / "policy.zip"
    backend = _Backend(
        _result(
            output_path=output_path,
            environment_digest=result_environment_digest,
            actual_timesteps=result_timesteps,
        )
    )
    backend_factory = _BackendFactory(backend)

    with pytest.raises(ValueError, match=message):
        _training_runner()(
            plan=plan,
            environment_factory=_as_u2_factory(factory),
            output_path=output_path,
            backend_factory=backend_factory,
        )

    assert factory.probe_calls == 1
    assert factory.probe_close_calls == 1
    assert backend_factory.calls == [factory]
    assert len(backend.calls) == 1
