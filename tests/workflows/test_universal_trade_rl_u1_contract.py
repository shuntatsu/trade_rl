from __future__ import annotations

import importlib
from dataclasses import replace

import pytest

from tests.rl.universal_trade_test_support import make_u1_market, make_u1_wrapper
from tests.workflows.universal_trade_rl_u1_test_support import (
    U1WorkflowFixture,
    build_u1_workflow_fixture,
)
from trade_rl.rl.universal_normalization import (
    build_universal_trade_sequence_normalizer,
)
from trade_rl.rl.universal_trade_observation import UniversalTradeObservationBuilder
from trade_rl.workflows.universal_trade_rl_run_identity import (
    UniversalTradeRLRunIdentity,
    UniversalTradeRLRunStage,
)


@pytest.fixture(scope="module")
def u1_fixture() -> U1WorkflowFixture:
    return build_u1_workflow_fixture()


def _module():
    return importlib.import_module("trade_rl.workflows.universal_trade_rl_u1_contract")


def test_u1_contract_binds_u0_normalizer_policy_and_no_go(
    u1_fixture: U1WorkflowFixture,
) -> None:
    contract = u1_fixture.build_contract()
    environment = u1_fixture.environment
    normalizer = environment.sequence_normalizer
    assert normalizer is not None
    observation = UniversalTradeObservationBuilder(
        contract=environment.contract,
        normalizer=normalizer,
    )

    assert contract.universe_manifest_digest == u1_fixture.manifest.digest
    assert contract.u0_identity_digest == u1_fixture.u0_identity.digest
    assert contract.policy_contract_digest == environment.contract.digest
    assert contract.normalizer_digest == normalizer.digest
    assert (
        contract.normalizer_provenance_digest == u1_fixture.normalizer_provenance.digest
    )
    assert contract.normalizer_knowledge_cutoff_ns == normalizer.knowledge_cutoff_ns
    assert contract.normalizer_clip_value == pytest.approx(10.0)
    assert contract.observation_schema_digest == observation.schema_digest
    assert contract.state_layout_digest == observation.state_layout_digest
    assert contract.policy_state_fields == observation.policy_state_fields
    assert (
        contract.execution_policy_digest == environment.base_env.execution_policy_digest
    )
    assert contract.production_status == "NO-GO"


def test_u1_contract_rejects_normalizer_from_another_universe_generation(
    u1_fixture: U1WorkflowFixture,
) -> None:
    module = _module()
    environment = u1_fixture.environment
    normalizer = environment.sequence_normalizer
    assert normalizer is not None
    eth = make_u1_market(symbol="ETHUSDT", n_bars=6200, feature_level=0.25)
    wrong_normalizer = build_universal_trade_sequence_normalizer(
        symbol_datasets={"BTCUSDT": environment.dataset, "ETHUSDT": eth},
        contract=environment.contract,
        source_dataset_digests=normalizer.source_dataset_digests,
        knowledge_cutoff_ns=normalizer.knowledge_cutoff_ns,
        universe_manifest_digest="e" * 64,
        provenance_digest=normalizer.provenance_digest,
    )
    wrong_environment = make_u1_wrapper(
        dataset=environment.dataset,
        contract=environment.contract,
        normalizer=wrong_normalizer,
    )

    with pytest.raises(ValueError, match="normalizer.*universe|universe.*normalizer"):
        module.build_universal_trade_rl_u1_contract(
            manifest=u1_fixture.manifest,
            u0_identity=u1_fixture.u0_identity,
            normalizer_provenance=u1_fixture.normalizer_provenance,
            environment=wrong_environment,
        )


def test_u1_contract_rejects_non_materialization_u0_identity(
    u1_fixture: U1WorkflowFixture,
) -> None:
    module = _module()
    wrong_identity = UniversalTradeRLRunIdentity(
        stage=UniversalTradeRLRunStage.BASE_TRAINING,
        universe_manifest_digest=u1_fixture.manifest.digest,
        model_config_digest="a" * 64,
        fit_provenance_digests=(u1_fixture.normalizer_provenance.digest,),
    )

    with pytest.raises(ValueError, match="materialization|stage"):
        module.build_universal_trade_rl_u1_contract(
            manifest=u1_fixture.manifest,
            u0_identity=wrong_identity,
            normalizer_provenance=u1_fixture.normalizer_provenance,
            environment=u1_fixture.environment,
        )


def test_u1_contract_requires_frozen_sequence_normalizer(
    u1_fixture: U1WorkflowFixture,
) -> None:
    module = _module()
    environment = make_u1_wrapper(
        dataset=u1_fixture.environment.dataset,
        contract=u1_fixture.environment.contract,
        normalizer=None,
    )

    with pytest.raises(ValueError, match="normalizer"):
        module.build_universal_trade_rl_u1_contract(
            manifest=u1_fixture.manifest,
            u0_identity=u1_fixture.u0_identity,
            normalizer_provenance=u1_fixture.normalizer_provenance,
            environment=environment,
        )


def test_u1_contract_from_payload_rejects_tampered_runtime_digest(
    u1_fixture: U1WorkflowFixture,
) -> None:
    module = _module()
    contract = u1_fixture.build_contract()
    payload = contract.to_payload()
    payload["runtime_config_digest"] = "f" * 64

    with pytest.raises(ValueError, match="digest mismatch"):
        module.UniversalTradeRLU1Contract.from_payload(payload)


def test_u1_contract_rejects_non_v1_normalizer_clip(
    u1_fixture: U1WorkflowFixture,
) -> None:
    contract = u1_fixture.build_contract()

    with pytest.raises(ValueError, match="clip"):
        replace(contract, normalizer_clip_value=9.0, digest="")


def test_u1_policy_state_order_changes_u1_identity(
    u1_fixture: U1WorkflowFixture,
) -> None:
    contract = u1_fixture.build_contract()
    changed = replace(
        contract,
        policy_state_fields=tuple(reversed(contract.policy_state_fields)),
        digest="",
    )

    assert changed.digest != contract.digest


@pytest.mark.parametrize(
    "field",
    (
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
    ),
)
def test_every_semantic_digest_changes_u1_identity(
    u1_fixture: U1WorkflowFixture,
    field: str,
) -> None:
    contract = u1_fixture.build_contract()
    replacement = "0" * 64 if getattr(contract, field) != "0" * 64 else "1" * 64
    changed = replace(contract, **{field: replacement, "digest": ""})

    assert changed.digest != contract.digest
