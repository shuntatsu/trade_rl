"""Independent literal/counterfactual oracles before any learned outcomes."""

import json
from fractions import Fraction
from importlib import import_module, util
from itertools import product

import numpy as np
import pytest


def capability():
    name = "tests.integrations.settled_learnability_fixture"
    assert util.find_spec(name), "causal settled learnability fixture is missing"
    return import_module(name)


@pytest.mark.parametrize("side,expected", [(1, 1009.495), (-1, 1009.505)])
def test_literal_delayed_native_fill_fee_and_full_reserve(side, expected):
    module = capability()
    env = module.environment((side, 0, 0, 0))
    actions = iter(([3 if side == 1 else 1, 0, 0, 2] + [0] * 12))
    result = module.run(env, lambda _: next(actions))
    native = [row for transition in result["transitions"] for row in transition["rows"]]
    assert len(native) == 18
    assert [row["processing_index"] for row in native] == list(range(7, 25))
    assert native[0]["native"]["execution"]["fill_count"] == 1
    assert native[1]["native"]["execution"]["fill_count"] == 0
    assert native[2]["native"]["execution"]["fill_count"] == 0
    assert result["actor_rewards"][:3] == pytest.approx([-0.00025, 0, 0.01])
    assert env.book.portfolio_value == pytest.approx(expected)
    assert result["gain"] == pytest.approx((expected - 1000) / 1000)
    assert sum(result["actor_rewards"]) == pytest.approx(result["gain"])
    assert result["witness"]["eligible"]
    assert result["fill_count"] == 2


@pytest.mark.parametrize("side", [-1, 1])
def test_late_entry_cannot_capture_delayed_payoff_or_survive_flat_cancel(side):
    module = capability()
    actions = iter([0, 0, 3 if side == 1 else 1, 2] + [0] * 12)
    result = module.run(module.environment((side, 0, 0, 0)), lambda _: next(actions))
    assert result["gain"] == 0
    assert result["fill_count"] == 0
    assert result["witness"]["eligible"]


def test_native_fee_only_cash_and_one_quantity_hold_roundtrip():
    module = capability()
    cash = module.run(
        module.environment((1, -1, -1, 1), price_response=0, fee=0.005), lambda _: 0
    )
    actions = iter([3] + [0] * 15)
    hold = module.run(
        module.environment((1, -1, -1, 1), price_response=0, fee=0.005),
        lambda _: next(actions),
    )
    assert cash["gain"] == 0 and cash["fill_count"] == 0
    assert hold["gain"] == pytest.approx(-0.005)
    assert hold["fill_count"] == 2
    assert cash["witness"]["physical_steps_processed"] == 18
    assert hold["witness"]["eligible"]


def test_post_payoff_entry_can_only_pay_a_native_roundtrip_fee():
    module = capability()
    actions = iter([0, 0, 0, 3, 2] + [0] * 11)
    result = module.run(module.environment((1, 0, 0, 0)), lambda _: next(actions))
    assert result["gain"] == pytest.approx(-0.0005)
    assert result["fill_count"] == 2
    assert result["witness"]["eligible"]


def test_balanced_factorial_cue_rule_matches_independent_four_block_ledger():
    module = capability()
    patterns = tuple(product((-1, 1), repeat=4))
    assert module.PATTERNS == patterns
    assert np.asarray(patterns).sum(axis=0).tolist() == [0] * 4
    observed = []
    for signs in patterns:
        wealth = Fraction(1000)
        for side in signs:
            wealth *= Fraction("1.01") - Fraction("0.0005") * (
                1 + Fraction("0.01") * side
            )
        result = module.run(module.environment(signs), module.cue_rule)
        assert result["gain"] == pytest.approx(float(wealth / 1000 - 1))
        assert result["alignment"] == 1
        assert result["witness"]["eligible"]
        observed.append(result["gain"])
    assert np.mean(observed) == pytest.approx(float(Fraction("1.0095") ** 4 - 1))


def test_future_price_capacity_and_cue_changes_do_not_leak_into_initial_observation():
    module = capability()
    original = module.environment((1, -1, -1, 1))
    altered = module.environment((1, 1, 1, -1), price_response=-0.07, capacity=200000)
    first, _ = original.reset()
    other, _ = altered.reset()
    assert first.shape == (47,)
    np.testing.assert_array_equal(first, other)
    assert original.recipe_digest == altered.recipe_digest
    assert all(packet.expected_simple_return == 0 for packet in original.stream.packets)
    assert original.decision.baseline.target_weight == 0


@pytest.mark.parametrize("index", range(6, 22))
def test_future_finance_and_capacity_do_not_leak_at_any_actor_decision(index):
    module = capability()
    original = module.environment()
    changed = module.environment(future_perturbation=index)
    first, _ = original.reset(seed=41)
    second, _ = changed.reset(seed=41)
    while original.index < index:
        np.testing.assert_array_equal(first, second)
        action = module.cue_rule(first)
        first, *_ = original.step(action)
        second, *_ = changed.step(action)
    assert original.dataset.close[index, 0] == changed.dataset.close[index, 0]
    assert original.dataset.close[index + 1, 0] != changed.dataset.close[index + 1, 0]
    assert original.dataset.volume[index + 1, 0] != changed.dataset.volume[index + 1, 0]
    np.testing.assert_array_equal(first, second)


def test_sole_directional_cue_erasure_and_inversion_preserve_finance():
    module = capability()
    zero = module.environment(cue_transform=0)
    inverse = module.environment(cue_transform=-1)
    ordinary = module.environment()
    for candidate in (zero, inverse):
        np.testing.assert_array_equal(candidate.dataset.close, ordinary.dataset.close)
        np.testing.assert_array_equal(candidate.dataset.volume, ordinary.dataset.volume)
        assert candidate.recipe_digest == ordinary.recipe_digest
        assert all(
            packet.expected_simple_return == packet.fit_prefix_marginal_variance == 0
            for packet in candidate.stream.packets
        )
        observation, _ = candidate.reset()
        assert candidate.decision.baseline.target_weight == 0
        assert observation[1] == observation[9] == 0
    np.testing.assert_array_equal(zero.dataset.features[:, 0, 1], 0)
    np.testing.assert_array_equal(
        inverse.dataset.features[:, 0, 1], -ordinary.dataset.features[:, 0, 1]
    )


def test_failed_actor_preserves_actual_partial_native_frontier():
    module = capability()
    env = module.environment()
    diagnostic = {}
    primary = RuntimeError("actor failed after first native return")
    calls = []

    def action(_):
        calls.append(1)
        if len(calls) == 2:
            raise primary
        return 3

    with pytest.raises(RuntimeError) as caught:
        module.run(env, action, diagnostic=diagnostic)
    assert caught.value is primary
    assert diagnostic["status"] == "INVALID"
    assert len(diagnostic["transitions"]) == 1
    assert diagnostic["witness"]["frontier_index"] == 7
    assert not diagnostic["witness"]["eligible"]
    assert diagnostic["actor_rewards"] == [-0.00025]


def test_factorial_failure_retains_prior_cells_and_explicit_not_run_tail():
    from tests.integrations import test_allocation_settled_learnability as runner

    primary = RuntimeError("third pattern failed")
    report, starts = {}, []

    def actor(_observation, env):
        if env.index == 6:
            starts.append(1)
        if len(starts) == 3:
            raise primary
        return 0

    with pytest.raises(RuntimeError) as caught:
        runner.evaluate("signal", actor, storage=report, key="trained")
    assert caught.value is primary
    rows = report["trained"]["patterns"]
    assert [row["status"] for row in rows] == ["VALID", "VALID", "INVALID"] + [
        "NOT_RUN"
    ] * 13
    assert len(rows[0]["transitions"]) == 16
    assert rows[2]["witness"]["frontier_index"] == 6
    assert report["trained"]["mean_gain"] is None


def test_persistence_failure_keeps_primary_exception_and_attempts_json(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from tests.integrations import test_allocation_settled_learnability as runner

    primary = RuntimeError("fit failed")

    def fail_trace(*_args, **_kwargs):
        raise OSError("trace storage failed")

    monkeypatch.setattr(runner.gzip, "open", fail_trace)
    report = {"software_conformance": "INVALID", "learning_status": "NOT_ESTABLISHED"}
    with pytest.raises(RuntimeError) as caught:
        try:
            raise primary
        finally:
            runner.persist(
                tmp_path,
                report,
                SimpleNamespace(events=(b"{}",)),
                primary_error=primary,
            )
    assert caught.value is primary
    assert (tmp_path / "learnability-evidence.json").is_file()
    assert report["persistence_errors"][0]["type"] == "OSError"


def test_evidence_checker_rejects_return_tampering_with_native_cash_control():
    from tests.integrations import test_allocation_settled_learnability as runner

    group = runner.evaluate("signal", lambda _obs, _env: 0)
    group["patterns"][0]["gain"] = 0.1
    group["mean_gain"] = 0.1 / 16
    with pytest.raises(AssertionError):
        runner.verify_group(group)


def test_aggregate_never_qualifies_missing_or_duplicate_slots(tmp_path):
    from tests.integrations import test_allocation_settled_learnability as runner

    missing = runner.aggregate_slots([])
    assert missing["software_conformance"] == "INCOMPLETE"
    assert missing["learning_status"] == "NOT_ESTABLISHED"
    assert len(missing["slots"]) == 6
    assert all(row["status"] == "NOT_RUN" for row in missing["slots"])
    report = {"seed": 0, "case": "signal", "software_conformance": "VALID"}
    path = tmp_path / "learnability-evidence.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    duplicate = runner.aggregate_slots([path, path])
    assert duplicate["software_conformance"] == "INVALID"
    assert duplicate["learning_status"] == "NOT_ESTABLISHED"
    assert duplicate["errors"]


def test_evidence_checker_rejects_exact_residue_below_float_precision():
    from tests.integrations import test_allocation_settled_learnability as runner

    group = runner.evaluate("signal", lambda _obs, _env: 0)
    runner.verify_group(group)
    cell = group["patterns"][0]
    residue = "0." + "0" * 400 + "1"
    assert float(residue) == 0 and Fraction(residue) > 0
    cell["witness"]["native_frontier"]["book"]["exact_quantities"] = [residue]
    cell["transitions"][-1]["settlement_witness"]["native_frontier"]["book"][
        "exact_quantities"
    ] = [residue]
    cell["transitions"][-1]["rows"][-1]["native"]["book"]["exact_quantities"] = [
        residue
    ]
    with pytest.raises(AssertionError):
        runner.verify_group(group)


def test_unrenderable_actor_error_survives_frontier_and_storage_failures(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from tests.integrations import test_allocation_settled_learnability as runner

    class Unrenderable(RuntimeError):
        add_note = None

        def __str__(self):
            raise ValueError("error rendering failed")

    primary, diagnostic = Unrenderable(), {}

    def actor(_):
        raise primary

    def fail_trace(*_args, **_kwargs):
        raise OSError("trace storage failed")

    monkeypatch.setattr(runner.gzip, "open", fail_trace)
    with pytest.raises(Unrenderable) as caught:
        try:
            capability().run(capability().environment(), actor, diagnostic=diagnostic)
        finally:
            runner.persist(
                tmp_path, diagnostic, SimpleNamespace(events=()), primary_error=primary
            )
    assert caught.value is primary
    assert diagnostic["status"] == "INVALID"
    assert diagnostic["witness"]["frontier_index"] == 6
    assert (tmp_path / "learnability-evidence.json").is_file()
