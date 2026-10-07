"""Declared envelopes retain native metadata without becoming access-use logs."""

from copy import copy
from hashlib import sha256
from importlib import import_module, util

import pytest

from tests.evaluation.test_allocation_nonrl_walk_forward import global_control_fixture
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data.contracts import VolumeUnit


def capability():
    name = "trade_rl.evaluation.rl_allocation.global_execution_context"
    assert util.find_spec(name) is not None, "shared global context owner is missing"
    return import_module(name)


def test_extracted_fee_context_preserves_original_native_bytes_and_errors():
    from trade_rl.evaluation.rl_allocation import fee_stress_admission as fee

    api = capability()
    env = global_control_fixture()
    assert api.allocation_source_envelope_digest(env) == fee._source_context(env)
    assert api.allocation_execution_runtime(env) == fee._runtime(env)
    # Literal bytes captured from the immutable parent, before extraction.
    assert api.allocation_source_envelope_digest(env) == (
        "57d9ab65d6e12520b644b9a269542464658c3c389f9eab705bf0df41057ecb3e"
    )
    runtime = canonical_json_bytes(api.allocation_execution_runtime(env))
    assert len(runtime) == 9075
    assert sha256(runtime).hexdigest() == (
        "41f5dfcd9957c781e09c7f217f867b1301e2c6003efc3ad5e8cdbe7939356686"
    )
    object.__setattr__(env.execution_cost, "fee_rate", 0.004)
    for function in (fee._runtime, api.allocation_execution_runtime):
        with pytest.raises(ValueError) as raised:
            function(env)
        assert (
            str(raised.value)
            == "fee native cost contents changed behind cached economics"
        )
        with pytest.raises(ValueError) as raised:
            function(object())
        assert str(raised.value) == "fee view requires native allocation execution"


def test_shared_raw_envelope_does_not_depend_on_candidate_feature_selection():
    api = capability()
    env = global_control_fixture()
    other = copy(env)
    other.feature_indices = ()  # Only this declared-source helper is called.
    assert api.global_declared_source_digest(env) == api.global_declared_source_digest(
        other
    )
    assert api.allocation_source_envelope_digest(
        env
    ) != api.allocation_source_envelope_digest(other)
    original = api.global_declared_source_digest(env)
    object.__setattr__(env.dataset, "volume_units", (VolumeUnit.QUOTE_NOTIONAL,))
    assert api.global_declared_source_digest(env) != original
