"""Declared envelopes retain native metadata without becoming access-use logs."""

from copy import copy
from dataclasses import replace
from hashlib import sha256
from importlib import import_module, util

import numpy as np
import pytest

from tests.evaluation.allocation_historical_snapshots import (
    use_historical_allocation_array_roster,
)
from tests.evaluation.test_allocation_global_comparison_evidence import clone_carrier
from tests.evaluation.test_allocation_global_execution import declare
from tests.evaluation.test_allocation_nonrl_walk_forward import (
    global_control_fixture,
    global_control_folds,
)
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data.contracts import VolumeUnit


def capability():
    name = "trade_rl.evaluation.rl_allocation.global_execution_context"
    assert util.find_spec(name) is not None, "shared global context owner is missing"
    return import_module(name)


@pytest.mark.parametrize("historical_snapshot", [False, True])
def test_extracted_fee_context_preserves_original_native_bytes_and_errors(
    monkeypatch, historical_snapshot
):
    from trade_rl.evaluation.rl_allocation import fee_stress_admission as fee

    api = capability()
    env = global_control_fixture()
    if historical_snapshot:
        use_historical_allocation_array_roster(monkeypatch)
    assert api.allocation_source_envelope_digest(env) == fee._source_context(env)
    assert api.allocation_execution_runtime(env) == fee._runtime(env)
    if historical_snapshot:
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


@pytest.mark.parametrize("unsupported", ["canonical", "settlement_products"])
def test_historical_snapshots_refuse_bound_or_nonfallback_datasets(
    monkeypatch, unsupported
):
    dataset = global_control_fixture().dataset
    if unsupported == "canonical":
        dataset = dataset.with_content_identity()
    else:
        dataset = replace(dataset, funding_price_rate=np.full(dataset.close.shape, 0.1))
    use_historical_allocation_array_roster(monkeypatch)
    with pytest.raises(ValueError, match="historical snapshots require"):
        dataset.identity_arrays()


def test_current_settlement_products_bind_identity_envelopes_and_execution(monkeypatch):
    from trade_rl.evaluation import allocation_global_execution as execution

    api = capability()
    original = global_control_fixture()
    # Two events may cancel their rates while retaining a price-weighted amount.
    unbound = replace(
        original.dataset,
        funding_event_count=np.full(original.dataset.close.shape, 2, dtype=np.int32),
    )
    products = unbound.resolved_array("funding_price_rate").copy()
    products[7, 0] = 0.1
    changed = replace(unbound, funding_price_rate=products)
    for name, values in unbound.identity_arrays().items():
        if name != "funding_price_rate":
            np.testing.assert_array_equal(values, changed.identity_arrays()[name])
    first, second = unbound.with_content_identity(), changed.with_content_identity()
    assert first.dataset_id != second.dataset_id

    def carrier(dataset):
        source = copy(original)
        source.stream = replace(original.stream, dataset_id=dataset.dataset_id)
        return clone_carrier(source, dataset=dataset)

    env, other = carrier(first), carrier(second)
    source_pin = api.allocation_source_envelope_digest(env)
    shared_pin = api.global_declared_source_digest(env)
    assert source_pin != api.allocation_source_envelope_digest(other)
    assert shared_pin != api.global_declared_source_digest(other)
    assert api.allocation_execution_runtime(other)[
        "source_context_digest"
    ] == api.allocation_source_envelope_digest(other)
    assert api.allocation_execution_runtime(env)["source_context_digest"] == source_pin

    folds = global_control_folds(7)
    plan = declare(execution, env, folds)
    # A cached Dataset ID alone cannot hide changes to the bound array bytes.
    object.__setattr__(env.dataset, "funding_price_rate", products)
    assert env.dataset.dataset_id == first.dataset_id
    assert api.allocation_source_envelope_digest(env) != source_pin
    assert api.global_declared_source_digest(env) != shared_pin
    assert api.allocation_execution_runtime(env)["source_context_digest"] != source_pin
    monkeypatch.setattr(env, "reset", lambda **kwargs: pytest.fail("native reset"))
    monkeypatch.setattr(env, "step", lambda action: pytest.fail("native execution"))
    with pytest.raises(
        ValueError, match="global current declaration differs from frozen plan"
    ):
        execution.run_declared_global_allocation_execution(folds, env, plan)
    assert not hasattr(env, "book") and env._transition_recorder is None
